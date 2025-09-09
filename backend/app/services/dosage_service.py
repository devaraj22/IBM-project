# backend/app/services/dosage_service.py
import sqlite3
import logging
import json
import asyncio
from typing import Dict, Optional, List, Any
from datetime import datetime
import requests
import aiohttp

logger = logging.getLogger(__name__)

class DosageService:
    """
    Service for calculating age-specific medication dosages
    Now with API integration for unknown drugs
    """

    def __init__(self):
        self.db_path = "data/dosage_database.db"
        self.api_service = None  # Will be set by main.py
        self.logger = logging.getLogger(__name__)
        self._init_database()
        
    def _init_database(self):
        """Initialize dosage database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS drug_dosages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        drug_name TEXT NOT NULL,
                        age_group TEXT NOT NULL,
                        min_age INTEGER,
                        max_age INTEGER,
                        base_dose REAL,
                        max_dose REAL,
                        unit TEXT,
                        frequency TEXT,
                        route TEXT,
                        indication TEXT,
                        weight_based BOOLEAN DEFAULT 0,
                        renal_adjustment BOOLEAN DEFAULT 0,
                        hepatic_adjustment BOOLEAN DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS dosage_adjustments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        drug_name TEXT NOT NULL,
                        condition_type TEXT NOT NULL,
                        condition_severity TEXT,
                        adjustment_factor REAL,
                        adjustment_description TEXT,
                        contraindicated BOOLEAN DEFAULT 0
                    )
                """)
                
                # Load sample dosage data
                self._load_sample_dosage_data(conn)
                
        except Exception as e:
            logger.error(f"Error initializing dosage database: {str(e)}")
    
    def _load_sample_dosage_data(self, conn):
        """Load sample dosage data"""
        sample_dosages = [
            # Acetaminophen
            {
                "drug_name": "acetaminophen", "age_group": "infant", "min_age": 0, "max_age": 2,
                "base_dose": 10.0, "max_dose": 15.0, "unit": "mg/kg", "frequency": "every 4-6 hours",
                "route": "oral", "weight_based": True
            },
            {
                "drug_name": "acetaminophen", "age_group": "child", "min_age": 2, "max_age": 12,
                "base_dose": 10.0, "max_dose": 15.0, "unit": "mg/kg", "frequency": "every 4-6 hours", 
                "route": "oral", "weight_based": True
            },
            {
                "drug_name": "acetaminophen", "age_group": "adult", "min_age": 18, "max_age": 65,
                "base_dose": 325.0, "max_dose": 1000.0, "unit": "mg", "frequency": "every 4-6 hours",
                "route": "oral", "weight_based": False
            },
            {
                "drug_name": "acetaminophen", "age_group": "elderly", "min_age": 65, "max_age": 120,
                "base_dose": 325.0, "max_dose": 650.0, "unit": "mg", "frequency": "every 6-8 hours",
                "route": "oral", "weight_based": False
            },
            
            # Ibuprofen
            {
                "drug_name": "ibuprofen", "age_group": "child", "min_age": 6, "max_age": 12,
                "base_dose": 5.0, "max_dose": 10.0, "unit": "mg/kg", "frequency": "every 6-8 hours",
                "route": "oral", "weight_based": True
            },
            {
                "drug_name": "ibuprofen", "age_group": "adult", "min_age": 18, "max_age": 65,
                "base_dose": 200.0, "max_dose": 800.0, "unit": "mg", "frequency": "every 6-8 hours",
                "route": "oral", "weight_based": False
            },
            
            # Amoxicillin
            {
                "drug_name": "amoxicillin", "age_group": "infant", "min_age": 0, "max_age": 2,
                "base_dose": 20.0, "max_dose": 40.0, "unit": "mg/kg/day", "frequency": "divided twice daily",
                "route": "oral", "weight_based": True
            },
            {
                "drug_name": "amoxicillin", "age_group": "child", "min_age": 2, "max_age": 12,
                "base_dose": 25.0, "max_dose": 45.0, "unit": "mg/kg/day", "frequency": "divided twice daily",
                "route": "oral", "weight_based": True
            },
            {
                "drug_name": "amoxicillin", "age_group": "adult", "min_age": 18, "max_age": 65,
                "base_dose": 250.0, "max_dose": 500.0, "unit": "mg", "frequency": "three times daily",
                "route": "oral", "weight_based": False
            },
            # Augmentin (Amoxicillin + Clavulanate)
            {
                "drug_name": "augmentin", "age_group": "infant", "min_age": 0, "max_age": 2,
                "base_dose": 25.0, "max_dose": 45.0, "unit": "mg/kg/day", "frequency": "divided twice daily",
                "route": "oral", "weight_based": True
            },
            {
                "drug_name": "augmentin", "age_group": "child", "min_age": 2, "max_age": 12,
                "base_dose": 25.0, "max_dose": 40.0, "unit": "mg/kg/day", "frequency": "divided twice daily",
                "route": "oral", "weight_based": True
            },
            {
                "drug_name": "augmentin", "age_group": "adult", "min_age": 18, "max_age": 65,
                "base_dose": 250.0, "max_dose": 1000.0, "unit": "mg", "frequency": "three times daily",
                "route": "oral", "weight_based": False
            },
            # Metformin
            {
                "drug_name": "metformin", "age_group": "child", "min_age": 10, "max_age": 16,
                "base_dose": 250.0, "max_dose": 500.0, "unit": "mg", "frequency": "twice daily",
                "route": "oral", "weight_based": False
            },
            {
                "drug_name": "metformin", "age_group": "adult", "min_age": 18, "max_age": 65,
                "base_dose": 500.0, "max_dose": 2000.0, "unit": "mg", "frequency": "twice daily",
                "route": "oral", "weight_based": False
            },
            # Omeprazole
            {
                "drug_name": "omeprazole", "age_group": "adult", "min_age": 18, "max_age": 65,
                "base_dose": 20.0, "max_dose": 40.0, "unit": "mg", "frequency": "once daily",
                "route": "oral", "weight_based": False
            },
            {
                "drug_name": "omeprazole", "age_group": "elderly", "min_age": 65, "max_age": 120,
                "base_dose": 20.0, "max_dose": 20.0, "unit": "mg", "frequency": "once daily",
                "route": "oral", "weight_based": False
            },
            # Prednisone
            {
                "drug_name": "prednisone", "age_group": "child", "min_age": 1, "max_age": 12,
                "base_dose": 0.5, "max_dose": 2.0, "unit": "mg/kg/day", "frequency": "once daily",
                "route": "oral", "weight_based": True
            },
            {
                "drug_name": "prednisone", "age_group": "adult", "min_age": 18, "max_age": 65,
                "base_dose": 5.0, "max_dose": 60.0, "unit": "mg", "frequency": "once daily",
                "route": "oral", "weight_based": False
            },
            # Warfarin
            {
                "drug_name": "warfarin", "age_group": "adult", "min_age": 18, "max_age": 120,
                "base_dose": 2.0, "max_dose": 10.0, "unit": "mg", "frequency": "variable - titrate to INR",
                "route": "oral", "weight_based": False
            },
            # Furosemide
            {
                "drug_name": "furosemide", "age_group": "adult", "min_age": 18, "max_age": 65,
                "base_dose": 20.0, "max_dose": 80.0, "unit": "mg", "frequency": "once daily",
                "route": "oral", "weight_based": False
            }
        ]
        
        # Insert sample data
        for dosage in sample_dosages:
            conn.execute("""
                INSERT OR REPLACE INTO drug_dosages 
                (drug_name, age_group, min_age, max_age, base_dose, max_dose, unit, frequency, route, weight_based)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                dosage["drug_name"], dosage["age_group"], dosage["min_age"], dosage["max_age"],
                dosage["base_dose"], dosage["max_dose"], dosage["unit"], dosage["frequency"],
                dosage["route"], dosage["weight_based"]
            ))
        
        # Sample adjustments
        adjustments = [
            {
                "drug_name": "acetaminophen", "condition_type": "hepatic_impairment",
                "condition_severity": "mild", "adjustment_factor": 0.75,
                "adjustment_description": "Reduce dose by 25% in mild hepatic impairment"
            },
            {
                "drug_name": "acetaminophen", "condition_type": "hepatic_impairment", 
                "condition_severity": "severe", "adjustment_factor": 0.0,
                "adjustment_description": "Contraindicated in severe hepatic impairment",
                "contraindicated": True
            },
            {
                "drug_name": "ibuprofen", "condition_type": "renal_impairment",
                "condition_severity": "moderate", "adjustment_factor": 0.5,
                "adjustment_description": "Reduce dose by 50% in moderate renal impairment"
            }
        ]
        
        for adj in adjustments:
            conn.execute("""
                INSERT OR REPLACE INTO dosage_adjustments
                (drug_name, condition_type, condition_severity, adjustment_factor, adjustment_description, contraindicated)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                adj["drug_name"], adj["condition_type"], adj["condition_severity"],
                adj["adjustment_factor"], adj["adjustment_description"], adj.get("contraindicated", False)
            ))
    
    def set_api_service(self, api_service):
        """Set the API service for fallback drug data"""
        self.api_service = api_service
        logger.info("API service linked to Dosage Service")

    async def initialize(self):
        """Initialize the dosage service"""
        logger.info("Initializing Dosage Service")

    async def cleanup(self):
        """Cleanup resources"""
        pass
    
    async def calculate_dosage(self, drug_name: str, age: int, weight: float,
                              indication: Optional[str] = None,
                              kidney_function: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Calculate age and weight-specific dosage with API fallback"""

        try:
            # Determine age group
            age_group = self._determine_age_group(age)

            # Get base dosage information from local database
            dosage_info = self._get_dosage_info(drug_name.lower(), age_group, age)

            if dosage_info:
                # Use local dosage data
                calculated_dose = self._calculate_dose(dosage_info, weight, age)
                adjusted_dose, adjustments, warnings = await self._apply_adjustments(
                    drug_name, calculated_dose, age, kidney_function
                )

                result = {
                    "drug_name": drug_name.title(),
                    "recommended_dose": adjusted_dose,
                    "unit": dosage_info["unit"],
                    "frequency": dosage_info["frequency"],
                    "route": dosage_info["route"],
                    "age_group": age_group,
                    "weight_based": dosage_info["weight_based"],
                    "base_calculation": calculated_dose,
                    "adjustments": adjustments,
                    "warnings": warnings,
                    "max_daily_dose": self._calculate_max_daily(dosage_info, weight),
                    "indication": indication,
                    "data_source": "local_database",
                    "confidence_level": "high"
                }
            else:
                # Local data not found - try API fallback
                logger.info(f"No local dosage data for {drug_name}, attempting API fallback")
                result = await self._generate_api_based_dosage(drug_name, age, weight, age_group, indication, kidney_function)

            return result

        except Exception as e:
            logger.error(f"Error calculating dosage: {str(e)}")
            return None
    
    def _determine_age_group(self, age: int) -> str:
        """Determine age group category"""
        if age < 1:
            return "neonate"
        elif age < 2:
            return "infant"
        elif age < 12:
            return "child"
        elif age < 18:
            return "adolescent"
        elif age < 65:
            return "adult"
        else:
            return "elderly"
    
    def _get_dosage_info(self, drug_name: str, age_group: str, age: int) -> Optional[Dict]:
        """Get dosage information from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM drug_dosages 
                    WHERE LOWER(drug_name) = ? 
                    AND (age_group = ? OR (min_age <= ? AND max_age >= ?))
                    ORDER BY 
                        CASE WHEN age_group = ? THEN 1 ELSE 2 END,
                        ABS(min_age - ?) + ABS(max_age - ?)
                    LIMIT 1
                """, (drug_name, age_group, age, age, age_group, age, age))
                
                result = cursor.fetchone()
                if result:
                    columns = [description[0] for description in cursor.description]
                    return dict(zip(columns, result))
                    
        except Exception as e:
            logger.error(f"Error getting dosage info: {str(e)}")
        
        return None
    
    def _calculate_dose(self, dosage_info: Dict, weight: float, age: int) -> float:
        """Calculate the actual dose based on dosage information"""
        base_dose = dosage_info["base_dose"]
        max_dose = dosage_info["max_dose"]
        weight_based = dosage_info["weight_based"]
        
        if weight_based:
            # Weight-based dosing
            calculated_dose = base_dose * weight
            max_calculated = max_dose * weight if max_dose else float('inf')
            
            # Apply pediatric weight limits
            if age < 18:
                # Ensure reasonable weight for age
                expected_weight = self._get_expected_weight(age)
                if weight > expected_weight * 1.5:  # 50% above expected
                    # Cap the weight used in calculation
                    calculated_dose = base_dose * (expected_weight * 1.2)
            
            return min(calculated_dose, max_calculated)
        else:
            # Fixed dosing - start with base dose for most patients
            return base_dose
    
    def _get_expected_weight(self, age: int) -> float:
        """Get expected weight for age (simplified formula)"""
        if age < 1:
            return 3.5 + (age * 12 * 0.5)  # Birth weight + monthly gain
        elif age < 2:
            return 10 + (age - 1) * 2.5  # Toddler weight gain
        elif age < 12:
            return 10 + (age * 2.3)  # Pediatric formula
        elif age < 18:
            return 30 + ((age - 12) * 5)  # Adolescent
        else:
            return 70  # Average adult weight
    
    def _calculate_max_daily(self, dosage_info: Dict, weight: float) -> float:
        """Calculate maximum daily dose"""
        max_dose = dosage_info["max_dose"]
        weight_based = dosage_info["weight_based"]
        frequency = dosage_info["frequency"].lower()
        
        # Estimate daily frequency from frequency string
        if "three times" in frequency or "tid" in frequency:
            daily_frequency = 3
        elif "twice" in frequency or "bid" in frequency:
            daily_frequency = 2
        elif "four times" in frequency or "qid" in frequency:
            daily_frequency = 4
        elif "every 4" in frequency:
            daily_frequency = 6
        elif "every 6" in frequency:
            daily_frequency = 4
        elif "every 8" in frequency:
            daily_frequency = 3
        elif "every 12" in frequency:
            daily_frequency = 2
        else:
            daily_frequency = 1
        
        if weight_based:
            return (max_dose * weight) * daily_frequency
        else:
            return max_dose * daily_frequency
    
    async def _apply_adjustments(self, drug_name: str, base_dose: float, age: int, 
                                kidney_function: Optional[str] = None) -> tuple:
        """Apply dose adjustments based on patient factors"""
        
        adjusted_dose = base_dose
        adjustments = {}
        warnings = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check for renal adjustments
                if kidney_function and kidney_function.lower() != "normal":
                    cursor = conn.execute("""
                        SELECT * FROM dosage_adjustments 
                        WHERE LOWER(drug_name) = ? AND condition_type = 'renal_impairment'
                        AND condition_severity = ?
                    """, (drug_name.lower(), kidney_function.lower().replace(" impairment", "")))
                    
                    adjustment = cursor.fetchone()
                    if adjustment:
                        if adjustment[6]:  # contraindicated
                            warnings.append(f"⚠️ {drug_name} is contraindicated in {kidney_function}")
                            adjusted_dose = 0
                        else:
                            adjustment_factor = adjustment[3]
                            adjusted_dose *= adjustment_factor
                            adjustments["renal"] = {
                                "factor": adjustment_factor,
                                "description": adjustment[4]
                            }
                
                # Age-specific adjustments
                if age >= 65:
                    # Elderly patients often need dose reduction
                    elderly_factor = 0.75
                    adjusted_dose *= elderly_factor
                    adjustments["elderly"] = {
                        "factor": elderly_factor,
                        "description": "Dose reduced for elderly patient"
                    }
                    warnings.append("👴 Elderly patient - monitor for increased sensitivity")
                
                elif age < 18:
                    warnings.append("👶 Pediatric patient - verify dosing calculation")
                
        except Exception as e:
            logger.error(f"Error applying adjustments: {str(e)}")
        
        return adjusted_dose, adjustments, warnings

    async def _generate_api_based_dosage(self, drug_name: str, age: int, weight: float,
                                        age_group: str, indication: Optional[str],
                                        kidney_function: Optional[str]) -> Optional[Dict[str, Any]]:
        """Generate estimated dosage based on API drug data"""

        if not self.api_service:
            logger.warning("API service not available for dosage fallback")
            return None

        try:
            # Get comprehensive drug data from APIs
            api_data = await self.api_service.get_drug_with_fallback(drug_name)

            estimated_dosage = None

            # Try API data first if confidence is reasonable
            if api_data and api_data.get('confidence_score', 0) >= 0.3:
                estimated_dosage = self._extract_dosage_from_api(api_data, age, weight, age_group)
                logger.info(f"Generated API-based dosage for {drug_name}")

            # If API data extraction failed, try name-based estimation
            if not estimated_dosage:
                logger.info(f"No API data reliable for {drug_name}, using name-based estimation")
                estimated_dosage = self._generate_name_based_estimation(drug_name, age, weight, age_group)

            if not estimated_dosage:
                logger.warning(f"Failed to generate any dosage estimate for {drug_name}")
                return None

            # Apply adjustments based on patient factors
            adjusted_dose, adjustments, warnings = await self._apply_adjustments(
                drug_name, estimated_dosage['base_dose'], age, kidney_function
            )

            # Add API-specific warnings
            warnings.append("⚠️ Dosage estimate based on available API data - consult healthcare provider")
            warnings.append("📊 Confidence level: moderate (API-derived estimation)")

            return {
                "drug_name": drug_name.title(),
                "recommended_dose": adjusted_dose,
                "unit": estimated_dosage["unit"],
                "frequency": estimated_dosage["frequency"],
                "route": estimated_dosage["route"],
                "age_group": age_group,
                "weight_based": estimated_dosage["weight_based"],
                "base_calculation": estimated_dosage['base_dose'],
                "adjustments": adjustments,
                "warnings": warnings,
                "max_daily_dose": estimated_dosage['max_daily_dose'],
                "indication": indication,
                "data_source": f"api_derived ({api_data.get('sources_used', [])})",
                "confidence_level": "moderate - api_estimated",
                "api_confidence": api_data.get('confidence_score', 0)
            }

        except Exception as e:
            logger.error(f"Error generating API-based dosage: {str(e)}")
            return None

    def _extract_dosage_from_api(self, api_data: Dict, age: int, weight: float, age_group: str) -> Optional[Dict]:
        """Extract and estimate dosage from API drug data"""

        # Get drug usage information from API
        usage = api_data.get('usage', '').lower()
        drug_class = api_data.get('drug_class', '').lower() if 'drug_class' in api_data else ''

        # Analgesic/Pain relief (acetaminophen-like)
        if 'pain' in usage or 'analgesic' in usage or 'headache' in usage:
            if age < 12:
                base_dose = 10.0  # mg/kg
                unit = "mg/kg"
                weight_based = True
                frequency = "every 4-6 hours"
            elif age < 65:
                base_dose = 325.0 if weight < 80 else 500.0  # mg
                unit = "mg"
                weight_based = False
                frequency = "every 4-6 hours"
            else:
                base_dose = 325.0  # mg - lower dose for elderly
                unit = "mg"
                weight_based = False
                frequency = "every 6-8 hours"

        # Antibiotic/Infection treatment (amoxicillin-like)
        elif 'antibiotic' in usage or 'infection' in usage or 'bacterial' in usage:
            if age < 12:
                base_dose = 25.0  # mg/kg/day
                unit = "mg/kg/day"
                weight_based = True
                frequency = "divided twice daily"
            else:
                base_dose = 250.0 if weight < 70 else 500.0  # mg
                unit = "mg"
                weight_based = False
                frequency = "three times daily"

        # Gastrointestinal/Acid related (omeprazole-like)
        elif 'heartburn' in usage or 'acid' in usage or 'stomach' in usage or 'ulcer' in usage:
            base_dose = 20.0 if age >= 18 else 10.0  # mg
            unit = "mg"
            weight_based = False
            frequency = "once daily"

        # Default adult oral medication
        elif age >= 18:
            base_dose = 100.0  # Conservative default
            unit = "mg"
            weight_based = False
            frequency = "twice daily"

        # Pediatric default
        elif age >= 2:
            base_dose = 5.0  # mg/kg conservative
            unit = "mg/kg"
            weight_based = True
            frequency = "twice daily"

        else:
            # Infant - very conservative
            base_dose = 2.5  # mg/kg very conservative
            unit = "mg/kg"
            weight_based = True
            frequency = "twice daily"

        # Calculate max daily dose
        if unit.endswith("/kg") or unit.endswith("/kg/day"):
            max_daily_dose = base_dose * 2 * weight if weight_based else base_dose * 2
        else:
            max_daily_dose = base_dose * 4  # Conservative multiple for safety

        return {
            'base_dose': base_dose,
            'unit': unit,
            'frequency': frequency,
            'route': 'oral',  # Most common, could be enhanced
            'weight_based': weight_based,
            'max_daily_dose': max_daily_dose
        }

    def _generate_name_based_estimation(self, drug_name: str, age: int, weight: float, age_group: str) -> Optional[Dict]:
        """Generate estimated dosage based on drug name analysis when APIs fail"""

        drug_lower = drug_name.lower()

        # Analgesic/Pain medications
        if 'enzoflam' in drug_lower or 'ibuprofen' in drug_lower or 'pain' in drug_lower or 'inflammation' in drug_lower:
            # NSAIDs - similar to ibuprofen
            if age < 12:
                base_dose = 5.0  # mg/kg
                unit = "mg/kg"
                weight_based = True
                frequency = "every 6-8 hours"  # Less frequent than acetaminophen
                max_daily_dose = 40.0  # mg/kg/day max
            else:
                base_dose = 200.0 if weight < 70 else 400.0  # mg
                unit = "mg"
                weight_based = False
                frequency = "three times daily"
                max_daily_dose = 1200.0  # mg/day typical max

        # Pancreatic enzymes
        elif 'pan-d' in drug_lower or 'pancreas' in drug_lower or 'enzyme' in drug_lower:
            # Similar to Creon/Pancrelipase
            base_dose = 20000.0 if age < 4 else 40000.0  # units
            unit = "units"
            weight_based = False
            frequency = "with each meal"
            # Note: pancreatic enzymes are not weight-based typically
            max_daily_dose = 120000.0  # units/day

        # Local anesthetics/Topical
        elif 'hexigel' in drug_lower or 'gum paint' in drug_lower or 'gel' in drug_lower or 'local' in drug_lower:
            # Topical anesthetic - similar to lidocaine gel
            base_dose = 1.0  # Application based, not weight based
            unit = "application"
            weight_based = False
            frequency = "apply 2-3 times daily"
            max_daily_dose = 3.0  # applications per day

        # Default conservative estimation for unknown drugs
        else:
            logger.warning(f"No classification found for {drug_name}, using ultra-conservative defaults")
            if age < 12:
                base_dose = 1.0  # mg/kg very conservative
                unit = "mg/kg"
                weight_based = True
                frequency = "once daily"
                max_daily_dose = 5.0  # mg/kg/day very conservative
            else:
                base_dose = 10.0  # mg very conservative
                unit = "mg"
                weight_based = False
                frequency = "once daily"
                max_daily_dose = 30.0  # mg/day very conservative

        # Calculate max daily dose properly
        if weight_based and 'kg' in unit:
            max_calculated = max_daily_dose / weight
        else:
            max_calculated = max_daily_dose

        return {
            'base_dose': base_dose,
            'unit': unit,
            'frequency': frequency,
            'route': 'oral',  # Most common, could be topical for some
            'weight_based': weight_based,
            'max_daily_dose': max_calculated
        }

    async def get_guidelines(self, drug_name: str) -> Optional[Dict]:
        """Get dosing guidelines for a drug"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM drug_dosages 
                    WHERE LOWER(drug_name) = ?
                    ORDER BY min_age
                """, (drug_name.lower(),))
                
                results = cursor.fetchall()
                if results:
                    columns = [description[0] for description in cursor.description]
                    guidelines = []
                    
                    for result in results:
                        guideline = dict(zip(columns, result))
                        guidelines.append(guideline)
                    
                    return {
                        "drug_name": drug_name.title(),
                        "age_groups": guidelines,
                        "last_updated": datetime.now().isoformat()
                    }
                    
        except Exception as e:
            logger.error(f"Error getting guidelines: {str(e)}")
        
        return None


