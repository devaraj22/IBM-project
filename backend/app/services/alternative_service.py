# backend/app/services/alternative_service.py
import sqlite3
import logging
import asyncio
import requests
from typing import List, Dict, Optional, Any
import json
import aiohttp

logger = logging.getLogger(__name__)

class AlternativeDrugService:
    """
    Service for finding alternative medications based on contraindications and patient factors
    """
    
    def __init__(self):
        self.db_path = "data/alternatives_database.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize alternatives database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS drug_alternatives (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        original_drug TEXT NOT NULL,
                        alternative_drug TEXT NOT NULL,
                        drug_class TEXT,
                        mechanism TEXT,
                        similarity_score REAL,
                        safety_profile TEXT,
                        advantages TEXT,
                        considerations TEXT,
                        contraindications TEXT,
                        age_restrictions TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS drug_classes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        drug_name TEXT NOT NULL,
                        therapeutic_class TEXT,
                        mechanism_class TEXT,
                        chemical_class TEXT
                    )
                """)
                
                # Load sample data
                self._load_sample_alternatives_data(conn)
                
        except Exception as e:
            logger.error(f"Error initializing alternatives database: {str(e)}")
    
    def _load_sample_alternatives_data(self, conn):
        """Load sample alternative drug data"""
        
        # Sample alternatives
        alternatives = [
            {
                "original_drug": "aspirin", "alternative_drug": "ibuprofen",
                "drug_class": "NSAID", "mechanism": "COX inhibition",
                "similarity_score": 0.85, "safety_profile": "Medium",
                "advantages": "Less GI bleeding risk in some patients",
                "considerations": "Avoid in kidney disease",
                "contraindications": "Severe renal impairment"
            },
            {
                "original_drug": "aspirin", "alternative_drug": "acetaminophen", 
                "drug_class": "Analgesic", "mechanism": "Central COX inhibition",
                "similarity_score": 0.70, "safety_profile": "High",
                "advantages": "Safer in bleeding disorders, pregnancy safe",
                "considerations": "No anti-inflammatory effect",
                "contraindications": "Severe liver disease"
            },
            {
                "original_drug": "lisinopril", "alternative_drug": "losartan",
                "drug_class": "Antihypertensive", "mechanism": "RAAS blockade", 
                "similarity_score": 0.90, "safety_profile": "High",
                "advantages": "Less cough than ACE inhibitors",
                "considerations": "Similar contraindications",
                "contraindications": "Pregnancy, bilateral renal artery stenosis"
            },
            {
                "original_drug": "warfarin", "alternative_drug": "rivaroxaban",
                "drug_class": "Anticoagulant", "mechanism": "Factor Xa inhibition",
                "similarity_score": 0.80, "safety_profile": "Medium",
                "advantages": "No monitoring required, fewer interactions",
                "considerations": "Higher cost, limited reversal options",
                "contraindications": "Severe renal impairment"
            },
            # Augmentin alternatives
            {
                "original_drug": "augmentin", "alternative_drug": "azithromycin",
                "drug_class": "Macrolide antibiotic", "mechanism": "Protein synthesis inhibition",
                "similarity_score": 0.70, "safety_profile": "High",
                "advantages": "Once daily dosing, fewer interactions",
                "considerations": "Different spectrum of coverage",
                "contraindications": "QT prolongation, myasthenia gravis"
            },
            {
                "original_drug": "augmentin", "alternative_drug": "ciprofloxacin",
                "drug_class": "Fluoroquinolone antibiotic", "mechanism": "DNA gyrase inhibition",
                "similarity_score": 0.75, "safety_profile": "Medium",
                "advantages": "Broad spectrum coverage",
                "considerations": "Tendon issues, QT prolongation",
                "contraindications": "Pregnancy, children under 18"
            },
            # Amoxicillin alternatives
            {
                "original_drug": "amoxicillin", "alternative_drug": "cefuroxime",
                "drug_class": "Cephalosporin antibiotic", "mechanism": "Cell wall synthesis inhibition",
                "similarity_score": 0.85, "safety_profile": "High",
                "advantages": "Good skin and soft tissue coverage",
                "considerations": "Different allergy cross-reactivity",
                "contraindications": "Severe penicillin allergy"
            },
            {
                "original_drug": "amoxicillin", "alternative_drug": "doxycycline",
                "drug_class": "Tetracycline antibiotic", "mechanism": "Protein synthesis inhibition",
                "similarity_score": 0.70, "safety_profile": "Medium",
                "advantages": "Good intracellular activity",
                "considerations": "Photosensitivity, dental staining in children",
                "contraindications": "Pregnancy, children under 8"
            },
            # Metformin alternatives
            {
                "original_drug": "metformin", "alternative_drug": "glipizide",
                "drug_class": "Sulfonylurea", "mechanism": "Pancreatic insulin secretion",
                "similarity_score": 0.65, "safety_profile": "Medium",
                "advantages": "Rapid glycemic control",
                "considerations": "Weight gain, hypoglycemia risk",
                "contraindications": "Severe renal impairment, pancreatic disease"
            },
            {
                "original_drug": "metformin", "alternative_drug": "sitagliptin",
                "drug_class": "DPP-4 inhibitor", "mechanism": "Incretin enhancement",
                "similarity_score": 0.70, "safety_profile": "High",
                "advantages": "Weight neutral, low hypoglycemia risk",
                "considerations": "Higher cost",
                "contraindications": "Severe renal impairment"
            },
            # Ibuprofen alternatives
            {
                "original_drug": "ibuprofen", "alternative_drug": "diclofenac",
                "drug_class": "NSAID", "mechanism": "COX inhibition",
                "similarity_score": 0.90, "safety_profile": "Medium",
                "advantages": "High potency and anti-inflammatory effect",
                "considerations": "More GI and CV adverse effects",
                "contraindications": "Severe heart disease, uncontrolled HTN"
            },
            {
                "original_drug": "ibuprofen", "alternative_drug": "naproxen",
                "drug_class": "NSAID", "mechanism": "COX inhibition",
                "similarity_score": 0.88, "safety_profile": "Medium",
                "advantages": "Longer duration of action",
                "considerations": "Similar GI and CV adverse effects",
                "contraindications": "Severe heart disease, uncontrolled HTN"
            },
            # Omeprazole alternatives
            {
                "original_drug": "omeprazole", "alternative_drug": "pantoprazole",
                "drug_class": "Proton pump inhibitor", "mechanism": "H+,K+-ATPase inhibition",
                "similarity_score": 0.95, "safety_profile": "High",
                "advantages": "Fewer drug interactions, IV formulation available",
                "considerations": "No major differences in efficacy",
                "contraindications": "Similar to other PPIs"
            },
            {
                "original_drug": "omeprazole", "alternative_drug": "ranitidine",
                "drug_class": "H2 blocker", "mechanism": "H2 receptor blockade",
                "similarity_score": 0.70, "safety_profile": "High",
                "advantages": "No cytochrome P450 interactions, faster onset",
                "considerations": "Less potent than PPIs, multiple daily doses",
                "contraindications": "Rare allergic reactions"
            }
        ]
        
        for alt in alternatives:
            conn.execute("""
                INSERT OR REPLACE INTO drug_alternatives
                (original_drug, alternative_drug, drug_class, mechanism, similarity_score,
                 safety_profile, advantages, considerations, contraindications)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alt["original_drug"], alt["alternative_drug"], alt["drug_class"],
                alt["mechanism"], alt["similarity_score"], alt["safety_profile"],
                alt["advantages"], alt["considerations"], alt["contraindications"]
            ))
        
        # Sample drug classes
        drug_classes = [
            {"drug_name": "aspirin", "therapeutic_class": "NSAID", "mechanism_class": "COX inhibitor"},
            {"drug_name": "ibuprofen", "therapeutic_class": "NSAID", "mechanism_class": "COX inhibitor"},
            {"drug_name": "acetaminophen", "therapeutic_class": "Analgesic", "mechanism_class": "Central acting"},
            {"drug_name": "lisinopril", "therapeutic_class": "ACE inhibitor", "mechanism_class": "RAAS blocker"},
            {"drug_name": "losartan", "therapeutic_class": "ARB", "mechanism_class": "RAAS blocker"},
            {"drug_name": "warfarin", "therapeutic_class": "Anticoagulant", "mechanism_class": "Vitamin K antagonist"},
            {"drug_name": "rivaroxaban", "therapeutic_class": "Anticoagulant", "mechanism_class": "Factor Xa inhibitor"},
            {"drug_name": "amoxicillin", "therapeutic_class": "Penicillin antibiotic", "mechanism_class": "Cell wall synthesis inhibitor"},
            {"drug_name": "augmentin", "therapeutic_class": "Penicillin antibiotic", "mechanism_class": "Cell wall synthesis inhibitor"},
            {"drug_name": "azithromycin", "therapeutic_class": "Macrolide antibiotic", "mechanism_class": "Protein synthesis inhibitor"},
            {"drug_name": "ciprofloxacin", "therapeutic_class": "Fluoroquinolone antibiotic", "mechanism_class": "DNA gyrase inhibitor"},
            {"drug_name": "cefuroxime", "therapeutic_class": "Cephalosporin antibiotic", "mechanism_class": "Cell wall synthesis inhibitor"},
            {"drug_name": "doxycycline", "therapeutic_class": "Tetracycline antibiotic", "mechanism_class": "Protein synthesis inhibitor"},
            {"drug_name": "metformin", "therapeutic_class": "Biguanide", "mechanism_class": "AMPK activator"},
            {"drug_name": "glipizide", "therapeutic_class": "Sulfonylurea", "mechanism_class": "Pancreatic insulin secretagogue"},
            {"drug_name": "sitagliptin", "therapeutic_class": "DPP-4 inhibitor", "mechanism_class": "Incretin enhancer"},
            {"drug_name": "diclofenac", "therapeutic_class": "NSAID", "mechanism_class": "COX inhibitor"},
            {"drug_name": "naproxen", "therapeutic_class": "NSAID", "mechanism_class": "COX inhibitor"},
            {"drug_name": "omeprazole", "therapeutic_class": "PPI", "mechanism_class": "H+,K+-ATPase inhibitor"},
            {"drug_name": "pantoprazole", "therapeutic_class": "PPI", "mechanism_class": "H+,K+-ATPase inhibitor"},
            {"drug_name": "ranitidine", "therapeutic_class": "H2 blocker", "mechanism_class": "H2 receptor antagonist"},
            {"drug_name": "prednisone", "therapeutic_class": "Corticosteroid", "mechanism_class": "Glucocorticoid receptor agonist"},
            {"drug_name": "furosemide", "therapeutic_class": "Loop diuretic", "mechanism_class": "Na+/K+/2Cl- cotransporter inhibitor"},
            {"drug_name": "enzoflam", "therapeutic_class": "NSAID", "mechanism_class": "COX inhibitor"},
            {"drug_name": "pan-d", "therapeutic_class": "Multivitamin", "mechanism_class": "Nutritional supplement"}
        ]
        
        for drug_class in drug_classes:
            conn.execute("""
                INSERT OR REPLACE INTO drug_classes
                (drug_name, therapeutic_class, mechanism_class)
                VALUES (?, ?, ?)
            """, (drug_class["drug_name"], drug_class["therapeutic_class"], drug_class["mechanism_class"]))
    
    async def initialize(self):
        """Initialize the alternative drug service"""
        logger.info("Initializing Alternative Drug Service")
    
    async def cleanup(self):
        """Cleanup resources"""
        pass
    
    async def find_alternatives(self, original_drug: str, contraindications: Optional[List[str]] = None,
                               allergies: Optional[List[str]] = None, patient_age: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find alternative medications based on patient constraints"""

        try:
            alternatives = []

            # Get alternatives from database
            db_alternatives = self._get_database_alternatives(original_drug)
            alternatives.extend(db_alternatives)

            # Filter alternatives based on contraindications and allergies
            filtered_alternatives = self._filter_alternatives(
                alternatives, contraindications, allergies, patient_age
            )

            # Remove duplicates by medication name
            filtered_alternatives = self._deduplicate_alternatives(filtered_alternatives)

            # Rank alternatives by suitability
            ranked_alternatives = self._rank_alternatives(filtered_alternatives, patient_age)

            return ranked_alternatives[:5]  # Return top 5 alternatives

        except Exception as e:
            logger.error(f"Error finding alternatives: {str(e)}")
            return []
    
    def _get_database_alternatives(self, original_drug: str) -> List[Dict]:
        """Get alternatives from local database"""
        alternatives = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM drug_alternatives 
                    WHERE LOWER(original_drug) = ?
                    ORDER BY similarity_score DESC
                """, (original_drug.lower(),))
                
                results = cursor.fetchall()
                columns = [description[0] for description in cursor.description]
                
                for result in results:
                    alt_dict = dict(zip(columns, result))
                    
                    # Format the alternative data
                    try:
                        alternative = {
                            "name": alt_dict["alternative_drug"].title(),
                            "drug_class": alt_dict["drug_class"],
                            "mechanism": alt_dict["mechanism"],
                            "similarity_score": alt_dict["similarity_score"],
                            "safety_profile": alt_dict["safety_profile"],
                            "advantages": alt_dict["advantages"].split("|") if alt_dict["advantages"] else [],
                            "considerations": alt_dict["considerations"].split("|") if alt_dict["considerations"] else [],
                            "contraindications": alt_dict["contraindications"].split("|") if alt_dict["contraindications"] else [],
                            "source": "database"
                        }

                        alternatives.append(alternative)
                    except Exception as e:
                        logger.error(f"Error processing alternative {alt_dict}: {str(e)}")
                        continue
                    
        except Exception as e:
            logger.error(f"Error getting database alternatives: {str(e)}")
        
        return alternatives
    
    def _filter_alternatives(self, alternatives: List[Dict], contraindications: Optional[List[str]] = None,
                           allergies: Optional[List[str]] = None, patient_age: Optional[int] = None) -> List[Dict]:
        """Filter alternatives based on patient constraints"""
        
        if not contraindications:
            contraindications = []
        if not allergies:
            allergies = []
        
        filtered = []
        
        for alt in alternatives:
            exclude_alt = False
            
            # Check contraindications
            alt_contraindications = [c.lower().strip() for c in alt.get("contraindications", [])]
            patient_contraindications = [c.lower().strip() for c in contraindications]
            
            for contraind in patient_contraindications:
                if any(contraind in alt_contraind for alt_contraind in alt_contraindications):
                    exclude_alt = True
                    break
            
            # Check allergies (simplified - would need more sophisticated mapping)
            alt_name = alt["name"].lower()
            for allergy in allergies:
                allergy_lower = allergy.lower()
                if ("penicillin" in allergy_lower and "cillin" in alt_name) or \
                   ("nsaid" in allergy_lower and alt.get("drug_class", "").lower() == "nsaid") or \
                   ("sulfa" in allergy_lower and "sulfa" in alt_name):
                    exclude_alt = True
                    break
            
            # Age restrictions
            if patient_age is not None:
                if patient_age < 18 and "pediatric" in str(alt.get("age_restrictions", "")).lower():
                    # Check if contraindicated in pediatrics
                    if "contraindicated" in str(alt.get("age_restrictions", "")).lower():
                        exclude_alt = True
            
            if not exclude_alt:
                filtered.append(alt)
        
        return filtered
    
    def _rank_alternatives(self, alternatives: List[Dict], patient_age: Optional[int] = None) -> List[Dict]:
        """Rank alternatives by suitability score"""
        
        for alt in alternatives:
            score = alt.get("similarity_score", 0.5)
            
            # Adjust score based on safety profile
            safety = alt.get("safety_profile", "Medium").lower()
            if safety == "high":
                score += 0.1
            elif safety == "low":
                score -= 0.1
            
            # Age-specific adjustments
            if patient_age is not None:
                if patient_age >= 65:
                    # Prefer safer alternatives for elderly
                    if safety == "high":
                        score += 0.05
                elif patient_age < 18:
                    # Pediatric considerations
                    if "pediatric" in str(alt.get("considerations", "")).lower():
                        if "safe" in str(alt.get("considerations", "")).lower():
                            score += 0.05
                        else:
                            score -= 0.05
            
            alt["suitability_score"] = min(score, 1.0)  # Cap at 1.0
        
        # Sort by suitability score
        return sorted(alternatives, key=lambda x: x.get("suitability_score", 0), reverse=True)

    def _deduplicate_alternatives(self, alternatives: List[Dict]) -> List[Dict]:
        """Remove duplicate alternatives based on medication name"""
        seen_names = set()
        deduplicated = []

        for alt in alternatives:
            name = alt.get("name", "").lower().strip()
            if name and name not in seen_names:
                seen_names.add(name)
                deduplicated.append(alt)

        return deduplicated

    async def get_drug_classes(self, drug_name: str) -> List[Dict]:
        """Get therapeutic classes for a drug"""
        classes = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM drug_classes 
                    WHERE LOWER(drug_name) = ?
                """, (drug_name.lower(),))
                
                result = cursor.fetchone()
                if result:
                    columns = [description[0] for description in cursor.description]
                    class_info = dict(zip(columns, result))
                    
                    classes = [
                        {
                            "type": "Therapeutic Class",
                            "name": class_info.get("therapeutic_class", "Unknown")
                        },
                        {
                            "type": "Mechanism Class", 
                            "name": class_info.get("mechanism_class", "Unknown")
                        }
                    ]
                    
                    if class_info.get("chemical_class"):
                        classes.append({
                            "type": "Chemical Class",
                            "name": class_info["chemical_class"]
                        })
                        
        except Exception as e:
            logger.error(f"Error getting drug classes: {str(e)}")
        
        return classes