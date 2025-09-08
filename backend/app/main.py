# backend/app/main.py
import sys
import os
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import sqlite3
import json
import logging
from datetime import datetime
from contextlib import contextmanager

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import services
from app.services.drug_service import DrugInteractionService
from app.services.nlp_service import PrescriptionNLPService
from app.services.dosage_service import DosageService
from app.services.alternative_service import AlternativeDrugService
from app.utils.config import get_settings

# Initialize FastAPI app
app = FastAPI(
    title="AI Medical Prescription Verification API",
    description="AI-powered medical prescription verification system with drug interaction detection, age-specific dosage recommendations, and NLP-based prescription parsing",
    version="1.0.0",
    contact={
        "name": "Medical AI Team",
        "email": "support@medical-ai.com",
    },
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
settings = get_settings()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class DrugInteractionRequest(BaseModel):
    drugs: List[str] = Field(..., description="List of drug names to check for interactions")
    patient_age: Optional[int] = Field(None, ge=0, le=120, description="Patient age for context")
    severity_filter: Optional[str] = Field(None, description="Filter by severity: minor, moderate, major")

class DosageRequest(BaseModel):
    drug_name: str = Field(..., description="Name of the medication")
    patient_age: int = Field(..., ge=0, le=120, description="Patient age in years")
    weight: float = Field(..., ge=0.5, le=300, description="Patient weight in kg")
    indication: Optional[str] = Field(None, description="Medical indication for the drug")
    kidney_function: Optional[str] = Field(None, description="Kidney function level")

class PrescriptionText(BaseModel):
    text: str = Field(..., description="Prescription text to parse")
    language: Optional[str] = Field("en", description="Language of the prescription")

class AlternativeRequest(BaseModel):
    drug_name: str = Field(..., description="Original drug name")
    contraindications: List[str] = Field(default=[], description="List of contraindications")
    allergies: List[str] = Field(default=[], description="Patient allergies")
    patient_age: Optional[int] = Field(None, description="Patient age")

class DrugInteractionResponse(BaseModel):
    interactions: List[Dict[str, Any]]
    safe: bool
    risk_level: str
    recommendations: List[str]

class DosageResponse(BaseModel):
    drug_name: str
    recommended_dose: float
    unit: str
    frequency: str
    route: str
    age_group: str
    warnings: List[str]
    adjustments: Dict[str, str]

class PrescriptionParseResponse(BaseModel):
    medications: List[Dict[str, Any]]
    dosages: List[Dict[str, Any]]
    frequencies: List[str]
    routes: List[str]
    confidence_score: float
    warnings: List[str]

# Initialize services
drug_service = DrugInteractionService()
nlp_service = PrescriptionNLPService()
dosage_service = DosageService()
alternative_service = AlternativeDrugService()

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint to verify API status"""

    # Test database connections
    services_status = {}
    try:
        # Test drug service
        await drug_service.initialize()
        services_status["drug_interaction"] = "operational"
    except Exception as e:
        services_status["drug_interaction"] = "error"
        logger.error(f"Drug service error: {str(e)}")

    try:
        # Test NLP service
        await nlp_service.initialize()
        services_status["nlp_processing"] = "operational"
    except Exception as e:
        services_status["nlp_processing"] = "error"
        logger.error(f"NLP service error: {str(e)}")

    try:
        # Test dosage service
        await dosage_service.initialize()
        services_status["dosage_calculation"] = "operational"
    except Exception as e:
        services_status["dosage_calculation"] = "error"
        logger.error(f"Dosage service error: {str(e)}")

    try:
        # Test alternative service
        await alternative_service.initialize()
        services_status["alternative_drugs"] = "operational"
    except Exception as e:
        services_status["alternative_drugs"] = "error"
        logger.error(f"Alternative service error: {str(e)}")

    # Overall status
    overall_status = "healthy" if all(status == "operational" for status in services_status.values()) else "degraded"

    return {
        "status": overall_status,
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0",
        "services": services_status
    }

# Drug interaction endpoints
@app.post("/check-interactions", response_model=DrugInteractionResponse)
async def check_drug_interactions(request: DrugInteractionRequest):
    """
    Check for drug-drug interactions between multiple medications
    """
    try:
        logger.info(f"Checking interactions for drugs: {request.drugs}")
        
        # Get drug interactions
        interactions = await drug_service.check_interactions(
            drugs=request.drugs,
            patient_age=request.patient_age,
            severity_filter=request.severity_filter
        )
        
        # Determine risk level
        if not interactions:
            risk_level = "low"
            safe = True
        elif any(i.get('severity') == 'major' for i in interactions):
            risk_level = "high"
            safe = False
        elif any(i.get('severity') == 'moderate' for i in interactions):
            risk_level = "medium"
            safe = False
        else:
            risk_level = "low"
            safe = True
        
        # Generate recommendations
        recommendations = drug_service.generate_recommendations(interactions)
        
        return DrugInteractionResponse(
            interactions=interactions,
            safe=safe,
            risk_level=risk_level,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Error checking drug interactions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error checking drug interactions: {str(e)}")

@app.get("/interaction-details/{drug1}/{drug2}")
async def get_interaction_details(drug1: str, drug2: str):
    """
    Get detailed information about interaction between two specific drugs
    """
    try:
        details = await drug_service.get_interaction_details(drug1, drug2)
        if not details:
            raise HTTPException(status_code=404, detail="Interaction not found")
        return details
    except Exception as e:
        logger.error(f"Error getting interaction details: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Age-specific dosage endpoints
@app.post("/age-dosage", response_model=DosageResponse)
async def get_age_specific_dosage(request: DosageRequest):
    """
    Get age-specific dosage recommendations for a medication
    """
    try:
        logger.info(f"Getting dosage for {request.drug_name}, age {request.patient_age}, weight {request.weight}")
        
        dosage_info = await dosage_service.calculate_dosage(
            drug_name=request.drug_name,
            age=request.patient_age,
            weight=request.weight,
            indication=request.indication,
            kidney_function=request.kidney_function
        )
        
        if not dosage_info:
            raise HTTPException(
                status_code=404, 
                detail=f"Dosage information not available for {request.drug_name}"
            )
        
        return DosageResponse(**dosage_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error calculating dosage: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error calculating dosage: {str(e)}")

@app.get("/dosage-guidelines/{drug_name}")
async def get_dosage_guidelines(drug_name: str):
    """
    Get general dosage guidelines for a specific drug
    """
    try:
        guidelines = await dosage_service.get_guidelines(drug_name)
        if not guidelines:
            raise HTTPException(status_code=404, detail="Guidelines not found")
        return guidelines
    except Exception as e:
        logger.error(f"Error getting dosage guidelines: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# NLP prescription parsing endpoints
@app.post("/parse-prescription", response_model=PrescriptionParseResponse)
async def parse_prescription_text(request: PrescriptionText):
    """
    Parse prescription text using NLP to extract structured information
    """
    try:
        logger.info(f"Parsing prescription text of length: {len(request.text)}")
        
        # Process with NLP service
        language = request.language if request.language else "en"
        parsed_data = await nlp_service.parse_prescription(
            text=request.text,
            language=language
        )
        
        return PrescriptionParseResponse(**parsed_data)
        
    except Exception as e:
        logger.error(f"Error parsing prescription: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error parsing prescription: {str(e)}")

@app.post("/extract-entities")
async def extract_medical_entities(request: PrescriptionText):
    """
    Extract medical entities from prescription text
    """
    try:
        entities = await nlp_service.extract_entities(request.text)
        return {"entities": entities}
    except Exception as e:
        logger.error(f"Error extracting entities: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Alternative medication endpoints
@app.post("/alternative-drugs")
async def get_alternative_medications(request: AlternativeRequest):
    """
    Get alternative medication suggestions based on contraindications and patient profile
    """
    try:
        logger.info(f"Finding alternatives for {request.drug_name}")
        
        alternatives = await alternative_service.find_alternatives(
            original_drug=request.drug_name,
            contraindications=request.contraindications,
            allergies=request.allergies,
            patient_age=request.patient_age
        )
        
        return {"alternatives": alternatives}
        
    except Exception as e:
        logger.error(f"Error finding alternatives: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error finding alternatives: {str(e)}")

@app.get("/drug-classes/{drug_name}")
async def get_drug_classes(drug_name: str):
    """
    Get therapeutic classes for a specific drug
    """
    try:
        classes = await alternative_service.get_drug_classes(drug_name)
        return {"drug_classes": classes}
    except Exception as e:
        logger.error(f"Error getting drug classes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Search and utility endpoints
@app.get("/search-drugs")
async def search_drugs(query: str, limit: int = 10):
    """
    Search for drugs by name or partial match
    """
    try:
        results = await drug_service.search_drugs(query, limit)
        return {"results": results}
    except Exception as e:
        logger.error(f"Error searching drugs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drug-info/{drug_name}")
async def get_drug_information(drug_name: str):
    """
    Get comprehensive drug information
    """
    try:
        info = await drug_service.get_drug_info(drug_name)
        if not info:
            raise HTTPException(status_code=404, detail="Drug not found")
        return info
    except Exception as e:
        logger.error(f"Error getting drug info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Analytics and monitoring endpoints
@app.get("/analytics/usage")
async def get_usage_analytics():
    """
    Get API usage analytics (for monitoring)
    """
    try:
        # This would typically pull from a monitoring database
        return {
            "total_requests": 12500,
            "interactions_checked": 8750,
            "prescriptions_parsed": 2100,
            "alternatives_suggested": 1650,
            "uptime": "99.8%"
        }
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# File upload endpoints
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
import os

@app.post("/parse-prescription-file")
async def parse_prescription_file(file: UploadFile = File(...), language: Optional[str] = "en"):
    """
    Parse prescription from uploaded file using NLP
    Supports PDF, DOCX, TXT, JPG, PNG files
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")

        # Save uploaded file temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)

        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        try:
            # Extract text from file based on type
            if file.filename.lower().endswith(('.txt', '.md')):
                text = content.decode('utf-8', errors='ignore')
            elif file.filename.lower().endswith(('.pdf', 'docx')):
                # For PDFs and DOCX, we'd need additional libraries like PyPDF2, python-docx
                # For now, return a helpful message
                raise HTTPException(
                    status_code=400,
                    detail="PDF and DOCX files are not supported in this demo version. Please convert to text or use the text input feature."
                )
            elif file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # For images, we'd need OCR libraries like pytesseract, easyocr
                raise HTTPException(
                    status_code=400,
                    detail="Image file processing requires additional OCR setup. Please use text input for now."
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file format. Supported: TXT, PDF (in future), DOCX (in future), JPG/PNG (in future)"
                )

            # Parse the extracted text
            parsed_data = await nlp_service.parse_prescription(text, language or "en")

            # Clean up temp file
            os.remove(file_path)

            return parsed_data

        except Exception as e:
            # Clean up temp file in case of error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise HTTPException(status_code=500, detail=f"File processing error: {str(e)}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing file upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File upload error: {str(e)}")

# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))

@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request, exc):
    return HTTPException(status_code=404, detail="Resource not found")

# Startup event
@app.on_event("startup")
async def startup_event():
    """
    Initialize services and connections on startup
    """
    logger.info("Starting AI Medical Prescription Verification API")
    
    # Initialize database connections
    await drug_service.initialize()
    await nlp_service.initialize()
    await dosage_service.initialize()
    await alternative_service.initialize()
    
    logger.info("All services initialized successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup on shutdown
    """
    logger.info("Shutting down AI Medical Prescription Verification API")
    
    # Close database connections and cleanup
    await drug_service.cleanup()
    await nlp_service.cleanup()
    await dosage_service.cleanup()
    await alternative_service.cleanup()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )