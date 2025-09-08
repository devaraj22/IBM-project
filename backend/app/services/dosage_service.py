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
    """
    
    def __init__(self):
        self.db_path = "data/dosage_database.db"
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
    
    async def initialize(self):
        """Initialize the dosage service"""
        logger.info("Initializing Dosage Service")
    
    async def cleanup(self):
        """Cleanup resources"""
        pass
    
    async def calculate_dosage(self, drug_name: str, age: int, weight: float, 
                             indication: Optional[str] = None, 
                             kidney_function: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Calculate age and weight-specific dosage"""
        
        try:
            # Determine age group
            age_group = self._determine_age_group(age)
            
            # Get base dosage information
            dosage_info = self._get_dosage_info(drug_name.lower(), age_group, age)
            
            if not dosage_info:
                return None
            
            # Calculate dose based on weight if needed
            calculated_dose = self._calculate_dose(dosage_info, weight, age)
            
            # Apply adjustments
            adjusted_dose, adjustments, warnings = await self._apply_adjustments(
                drug_name, calculated_dose, age, kidney_function
            )
            
            return {
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
                "indication": indication
            }
            
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


