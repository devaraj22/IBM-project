# backend/app/services/drug_service.py
import sqlite3
import requests
import json
import asyncio
import logging
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import hashlib
import aiohttp
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import os

logger = logging.getLogger(__name__)

class DrugInteractionService:
    """
    Service for handling drug-drug interactions using multiple data sources:
    - DrugBank API
    - RxNorm API
    - Local database for caching
    - OpenFDA FAERS data for adverse events
    """
    
    def __init__(self):
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
        os.makedirs(data_dir, exist_ok=True)
        self.db_path = os.path.join(data_dir, "drug_interactions.db")
        self.cache_duration = timedelta(hours=24)  # Cache for 24 hours
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # API endpoints
        self.rxnorm_base = "https://rxnav.nlm.nih.gov/REST"
        self.openfda_base = "https://api.fda.gov"
        
        # Initialize database
        self._init_database()
        
    def _init_database(self):
        """Initialize SQLite database for caching interactions"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS drug_interactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        drug1 TEXT NOT NULL,
                        drug2 TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        description TEXT,
                        mechanism TEXT,
                        management TEXT,
                        evidence_level TEXT,
                        source TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS drug_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        drug_hash TEXT UNIQUE NOT NULL,
                        drug_names TEXT NOT NULL,
                        interaction_data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS drug_info (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        drug_name TEXT UNIQUE NOT NULL,
                        rxcui TEXT,
                        generic_name TEXT,
                        brand_names TEXT,
                        drug_class TEXT,
                        mechanism TEXT,
                        indications TEXT,
                        contraindications TEXT,
                        side_effects TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Create indexes for better performance
                conn.execute("CREATE INDEX IF NOT EXISTS idx_drug_interactions_drugs ON drug_interactions(drug1, drug2)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_drug_cache_hash ON drug_cache(drug_hash)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_drug_info_name ON drug_info(drug_name)")
                
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
            raise

    async def initialize(self):
        """Initialize the service and load initial data"""
        logger.info("Initializing Drug Interaction Service")
        # Load sample drug interaction data
        await self._load_sample_data()
        
    async def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)
        
    def _generate_cache_key(self, drugs: List[str]) -> str:
        """Generate cache key for drug combination"""
        sorted_drugs = sorted([drug.lower().strip() for drug in drugs])
        return hashlib.md5("|".join(sorted_drugs).encode()).hexdigest()
    
    async def _load_sample_data(self):
        """Load sample drug interaction data for demonstration"""
        sample_interactions = [
            {
                "drug1": "warfarin", "drug2": "aspirin", "severity": "major",
                "description": "Increased risk of bleeding when used together",
                "mechanism": "Both drugs affect blood coagulation",
                "management": "Monitor INR closely, consider alternative antiplatelet therapy",
                "evidence_level": "level_1", "source": "drugbank"
            },
            {
                "drug1": "warfarin", "drug2": "acetaminophen", "severity": "moderate",
                "description": "Acetaminophen may increase anticoagulant effect of warfarin",
                "mechanism": "Acetaminophen may inhibit vitamin K synthesis",
                "management": "Monitor INR, avoid doses >2g/day acetaminophen",
                "evidence_level": "level_2", "source": "drugbank"
            },
            {
                "drug1": "lisinopril", "drug2": "ibuprofen", "severity": "moderate",
                "description": "NSAIDs may reduce antihypertensive effect of ACE inhibitors",
                "mechanism": "NSAIDs reduce renal prostaglandin synthesis",
                "management": "Monitor blood pressure, consider alternative pain relief",
                "evidence_level": "level_1", "source": "drugbank"
            },
            {
                "drug1": "metformin", "drug2": "furosemide", "severity": "minor",
                "description": "Diuretics may affect blood glucose control",
                "mechanism": "Diuretics can cause hyperglycemia",
                "management": "Monitor blood glucose levels",
                "evidence_level": "level_2", "source": "drugbank"
            },
            {
                "drug1": "digoxin", "drug2": "furosemide", "severity": "moderate",
                "description": "Increased risk of digoxin toxicity due to electrolyte imbalance",
                "mechanism": "Diuretic-induced hypokalemia increases digoxin sensitivity",
                "management": "Monitor digoxin levels and electrolytes",
                "evidence_level": "level_1", "source": "drugbank"
            }
        ]
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                for interaction in sample_interactions:
                    conn.execute("""
                        INSERT OR REPLACE INTO drug_interactions 
                        (drug1, drug2, severity, description, mechanism, management, evidence_level, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        interaction["drug1"], interaction["drug2"], interaction["severity"],
                        interaction["description"], interaction["mechanism"], 
                        interaction["management"], interaction["evidence_level"], interaction["source"]
                    ))
                    
                    # Also insert reverse combination
                    conn.execute("""
                        INSERT OR REPLACE INTO drug_interactions 
                        (drug1, drug2, severity, description, mechanism, management, evidence_level, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        interaction["drug2"], interaction["drug1"], interaction["severity"],
                        interaction["description"], interaction["mechanism"], 
                        interaction["management"], interaction["evidence_level"], interaction["source"]
                    ))
                
                logger.info(f"Loaded {len(sample_interactions)} sample drug interactions")
                
        except Exception as e:
            logger.error(f"Error loading sample data: {str(e)}")

    async def check_interactions(self, drugs: List[str], patient_age: Optional[int] = None, 
                               severity_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Check for drug-drug interactions between multiple medications
        """
        if len(drugs) < 2:
            return []
            
        # Generate cache key
        cache_key = self._generate_cache_key(drugs)
        
        # Check cache first
        cached_result = await self._get_cached_interactions(cache_key)
        if cached_result:
            interactions = json.loads(cached_result)
        else:
            # Get interactions from multiple sources
            interactions = await self._get_interactions_from_sources(drugs)
            
            # Cache the result
            await self._cache_interactions(cache_key, drugs, interactions)
        
        # Filter by severity if specified
        if severity_filter:
            interactions = [i for i in interactions if i.get('severity', '').lower() == severity_filter.lower()]
        
        # Add age-specific warnings if patient age provided
        if patient_age is not None:
            interactions = await self._add_age_specific_warnings(interactions, patient_age)
        
        return interactions
    
    async def _get_cached_interactions(self, cache_key: str) -> Optional[str]:
        """Get cached interactions if available and not expired"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT interaction_data, created_at 
                    FROM drug_cache 
                    WHERE drug_hash = ?
                """, (cache_key,))
                
                result = cursor.fetchone()
                if result:
                    data, created_at = result
                    created_datetime = datetime.fromisoformat(created_at)
                    
                    if datetime.now() - created_datetime < self.cache_duration:
                        return data
                        
        except Exception as e:
            logger.error(f"Error getting cached interactions: {str(e)}")
            
        return None
    
    async def _cache_interactions(self, cache_key: str, drugs: List[str], interactions: List[Dict]):
        """Cache interaction results"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO drug_cache (drug_hash, drug_names, interaction_data)
                    VALUES (?, ?, ?)
                """, (cache_key, "|".join(drugs), json.dumps(interactions)))
                
        except Exception as e:
            logger.error(f"Error caching interactions: {str(e)}")
    
    async def _get_interactions_from_sources(self, drugs: List[str]) -> List[Dict[str, Any]]:
        """Get interactions from multiple data sources"""
        interactions = []
        
        # Check local database first
        local_interactions = await self._get_local_interactions(drugs)
        interactions.extend(local_interactions)
        
        # Check RxNorm API for additional interactions
        try:
            rxnorm_interactions = await self._get_rxnorm_interactions(drugs)
            interactions.extend(rxnorm_interactions)
        except Exception as e:
            logger.warning(f"RxNorm API error: {str(e)}")
        
        # Remove duplicates based on drug pair
        seen_pairs = set()
        unique_interactions = []
        
        for interaction in interactions:
            pair = tuple(sorted([interaction.get('drug1', '').lower(), interaction.get('drug2', '').lower()]))
            if pair not in seen_pairs:
                seen_pairs.add(pair)
                unique_interactions.append(interaction)
        
        return unique_interactions
    
    async def _get_local_interactions(self, drugs: List[str]) -> List[Dict[str, Any]]:
        """Get interactions from local database"""
        interactions = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check all pairs of drugs
                for i in range(len(drugs)):
                    for j in range(i + 1, len(drugs)):
                        drug1, drug2 = drugs[i].lower().strip(), drugs[j].lower().strip()
                        
                        cursor = conn.execute("""
                            SELECT drug1, drug2, severity, description, mechanism, 
                                   management, evidence_level, source
                            FROM drug_interactions 
                            WHERE (LOWER(drug1) = ? AND LOWER(drug2) = ?) 
                               OR (LOWER(drug1) = ? AND LOWER(drug2) = ?)
                        """, (drug1, drug2, drug2, drug1))
                        
                        result = cursor.fetchone()
                        if result:
                            interactions.append({
                                'drug1': drugs[i],
                                'drug2': drugs[j],
                                'severity': result[2],
                                'description': result[3],
                                'mechanism': result[4],
                                'management': result[5],
                                'evidence_level': result[6],
                                'source': result[7]
                            })
                            
        except Exception as e:
            logger.error(f"Error getting local interactions: {str(e)}")
        
        return interactions
    
    async def _get_rxnorm_interactions(self, drugs: List[str]) -> List[Dict[str, Any]]:
        """Get interactions from RxNorm API"""
        interactions = []
        
        try:
            # First, get RxCUIs for each drug
            drug_rxcuis = {}
            
            async with aiohttp.ClientSession() as session:
                for drug in drugs:
                    try:
                        url = f"{self.rxnorm_base}/rxcui.json?name={drug}"
                        async with session.get(url) as response:
                            if response.status == 200:
                                data = await response.json()
                                if 'idGroup' in data and 'rxnormId' in data['idGroup']:
                                    rxcuis = data['idGroup']['rxnormId']
                                    if rxcuis:
                                        drug_rxcuis[drug] = rxcuis[0]
                    except Exception as e:
                        logger.warning(f"Error getting RxCUI for {drug}: {str(e)}")
                        continue
                
                # Check interactions for each drug with RxCUI
                for drug, rxcui in drug_rxcuis.items():
                    try:
                        url = f"{self.rxnorm_base}/interaction/interaction.json?rxcui={rxcui}&sources=DrugBank"
                        async with session.get(url) as response:
                            if response.status == 200:
                                data = await response.json()
                                
                                if 'interactionTypeGroup' in data:
                                    for group in data['interactionTypeGroup']:
                                        if 'interactionType' in group:
                                            for interaction_type in group['interactionType']:
                                                if 'interactionPair' in interaction_type:
                                                    for pair in interaction_type['interactionPair']:
                                                        interacting_drug = pair.get('interactionConcept', [{}])[1].get('minConceptItem', {}).get('name', '')
                                                        
                                                        # Check if interacting drug is in our list
                                                        if any(interacting_drug.lower() in d.lower() or d.lower() in interacting_drug.lower() for d in drugs if d != drug):
                                                            interactions.append({
                                                                'drug1': drug,
                                                                'drug2': interacting_drug,
                                                                'severity': 'moderate',  # RxNorm doesn't provide severity
                                                                'description': pair.get('description', 'Drug interaction detected'),
                                                                'mechanism': 'See interaction details',
                                                                'management': 'Consult healthcare provider',
                                                                'evidence_level': 'level_2',
                                                                'source': 'rxnorm'
                                                            })
                    except Exception as e:
                        logger.warning(f"Error getting RxNorm interactions for {drug}: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error in RxNorm API call: {str(e)}")
        
        return interactions
    
    async def _add_age_specific_warnings(self, interactions: List[Dict], patient_age: int) -> List[Dict]:
        """Add age-specific warnings to interactions"""
        for interaction in interactions:
            warnings = []
            
            # Pediatric warnings (age < 18)
            if patient_age < 18:
                warnings.append("⚠️ Pediatric patient - dosing adjustments may be required")
                if interaction.get('severity') == 'major':
                    warnings.append("⚠️ High-risk interaction in pediatric population")
            
            # Geriatric warnings (age >= 65)
            elif patient_age >= 65:
                warnings.append("⚠️ Elderly patient - increased risk of adverse effects")
                if interaction.get('severity') in ['major', 'moderate']:
                    warnings.append("⚠️ Enhanced monitoring recommended in elderly")
            
            interaction['age_warnings'] = warnings
        
        return interactions
    
    def generate_recommendations(self, interactions: List[Dict]) -> List[str]:
        """Generate clinical recommendations based on interactions"""
        recommendations = []
        
        if not interactions:
            recommendations.append("✅ No significant drug interactions detected")
            return recommendations
        
        major_count = sum(1 for i in interactions if i.get('severity') == 'major')
        moderate_count = sum(1 for i in interactions if i.get('severity') == 'moderate')
        
        if major_count > 0:
            recommendations.append(f"🚨 {major_count} major interaction(s) detected - immediate review required")
            recommendations.append("📞 Contact prescribing physician before administration")
        
        if moderate_count > 0:
            recommendations.append(f"⚠️ {moderate_count} moderate interaction(s) detected - monitoring required")
            recommendations.append("📊 Implement enhanced patient monitoring")
        
        # Add specific management recommendations
        for interaction in interactions:
            if interaction.get('management'):
                recommendations.append(f"💊 {interaction['drug1']} + {interaction['drug2']}: {interaction['management']}")
        
        return recommendations
    
    async def get_interaction_details(self, drug1: str, drug2: str) -> Optional[Dict]:
        """Get detailed information about interaction between two drugs"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM drug_interactions 
                    WHERE (LOWER(drug1) = ? AND LOWER(drug2) = ?) 
                       OR (LOWER(drug1) = ? AND LOWER(drug2) = ?)
                """, (drug1.lower(), drug2.lower(), drug2.lower(), drug1.lower()))
                
                result = cursor.fetchone()
                if result:
                    columns = [description[0] for description in cursor.description]
                    return dict(zip(columns, result))
                    
        except Exception as e:
            logger.error(f"Error getting interaction details: {str(e)}")
        
        return None
    
    async def search_drugs(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for drugs by name"""
        results = []
        
        try:
            # Search in local database first
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT DISTINCT drug_name, generic_name, brand_names, drug_class
                    FROM drug_info 
                    WHERE LOWER(drug_name) LIKE ? 
                       OR LOWER(generic_name) LIKE ?
                       OR LOWER(brand_names) LIKE ?
                    LIMIT ?
                """, (f"%{query.lower()}%", f"%{query.lower()}%", f"%{query.lower()}%", limit))
                
                for row in cursor.fetchall():
                    results.append({
                        'name': row[0],
                        'generic_name': row[1],
                        'brand_names': row[2].split('|') if row[2] else [],
                        'drug_class': row[3]
                    })
            
            # If no local results, try RxNorm API
            if not results:
                try:
                    async with aiohttp.ClientSession() as session:
                        url = f"{self.rxnorm_base}/drugs.json?name={query}"
                        async with session.get(url) as response:
                            if response.status == 200:
                                data = await response.json()
                                # Process RxNorm results
                                # This would need more detailed implementation
                                pass
                                
                except Exception as e:
                    logger.warning(f"RxNorm search error: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error searching drugs: {str(e)}")
        
        return results[:limit]
    
    async def get_drug_info(self, drug_name: str) -> Optional[Dict]:
        """Get comprehensive drug information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM drug_info WHERE LOWER(drug_name) = ?
                """, (drug_name.lower(),))
                
                result = cursor.fetchone()
                if result:
                    columns = [description[0] for description in cursor.description]
                    drug_info = dict(zip(columns, result))
                    
                    # Convert pipe-separated strings to lists
                    if drug_info.get('brand_names'):
                        drug_info['brand_names'] = drug_info['brand_names'].split('|')
                    if drug_info.get('indications'):
                        drug_info['indications'] = drug_info['indications'].split('|')
                    if drug_info.get('contraindications'):
                        drug_info['contraindications'] = drug_info['contraindications'].split('|')
                    if drug_info.get('side_effects'):
                        drug_info['side_effects'] = drug_info['side_effects'].split('|')
                    
                    return drug_info
                    
        except Exception as e:
            logger.error(f"Error getting drug info: {str(e)}")
        
        return None