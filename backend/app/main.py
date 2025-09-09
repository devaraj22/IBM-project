# backend/app/main.py
import sys
import os
import asyncio
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

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
from app.services.api_drugs_service import MedicalAPIDrugService
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

class DrugInfoRequest(BaseModel):
    drug_name: str = Field(..., description="Name of the drug to search for")
    use_api: Optional[bool] = Field(True, description="Whether to use external APIs for data")

class ComprehensiveDrugResponse(BaseModel):
    drug_name: str
    sources_used: List[str]
    data_quality: str
    confidence_score: float
    brand_name: Optional[List[str]]
    generic_name: Optional[List[str]]
    usage: Optional[str]
    mechanism_of_action: Optional[List[str]]
    warnings: Optional[List[str]]
    adverse_reactions: Optional[List[str]]
    contraindications: Optional[List[str]]
    last_updated: str
    molecular_formula: Optional[str]
    molecular_weight: Optional[float]
    standard_names: Optional[List[Dict]]

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
api_drug_service = MedicalAPIDrugService()

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

    try:
        # Test API drug service
        await api_drug_service.initialize()
        services_status["api_drugs"] = "operational"
    except Exception as e:
        services_status["api_drugs"] = "error"
        logger.error(f"API Drug service error: {str(e)}")

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
@app.post("/parse-prescription")
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

        # Validate and sanitize the response before returning
        try:
            # Ensure all required fields exist with fallback defaults
            parsed_data_configs = {
                'medications': parsed_data.get('medications', []),
                'dosages': parsed_data.get('dosages', []),
                'frequencies': parsed_data.get('frequencies', []),
                'routes': parsed_data.get('routes', []),
                'confidence_score': parsed_data.get('confidence_score', 0.0),
                'warnings': parsed_data.get('warnings', [])
            }

            # Add processing method info if available
            if 'processing_method' in parsed_data:
                parsed_data_configs['processing_method'] = parsed_data['processing_method']

            # Validate medication list structure
            validated_medications = []
            for med in parsed_data_configs['medications']:
                if isinstance(med, dict) and 'name' in med:
                    validated_medications.append({
                        'name': med['name'],
                        'confidence': med.get('confidence', 0.0)
                    })
            parsed_data_configs['medications'] = validated_medications

            return PrescriptionParseResponse(**parsed_data_configs)

        except Exception as validation_error:
            logger.error(f"Pydantic validation error: {str(validation_error)}")

            # Return fallback response with safe defaults
            fallback_result = {
                'medications': [],
                'dosages': [],
                'frequencies': [],
                'routes': [],
                'confidence_score': 0.0,
                'warnings': [f"Response validation failed: {str(validation_error)}"],
                'processing_method': 'validation_fallback'
            }

            return PrescriptionParseResponse(**fallback_result)

    except Exception as e:
        logger.error(f"Error parsing prescription: {str(e)}")

        # Enhanced error response with debugging info
        error_result = {
            'medications': [],
            'dosages': [],
            'frequencies': [],
            'routes': [],
            'confidence_score': 0.0,
            'warnings': [
                f"🚨 Parsing failed due to internal error",
                f"Error type: {type(e).__name__}",
                f"Please retry or contact support if the issue persists"
            ],
            'processing_method': 'error_fallback'
        }

        try:
            return PrescriptionParseResponse(**error_result)
        except Exception as fallback_error:
            logger.error(f"Fallback response failed: {str(fallback_error)}")
            # Last resort - return basic dict
            return {
                'medications': [],
                'dosages': [],
                'frequencies': [],
                'routes': [],
                'confidence_score': 0.0,
                'warnings': ['Critical system error occurred'],
                'processing_method': 'critical_error'
            }

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
async def get_drug_information(drug_name: str, use_api: Optional[bool] = None):
    """
    Get comprehensive drug information
    Now supports API-first approach with optional parameter
    """
    try:
        # If API usage is explicitly requested, use the new API service
        if use_api is True:
            api_result = await api_drug_service.get_drug_with_fallback(drug_name)
            if api_result:
                return api_result

        # Default behavior: try local database first, then fall back to APIs
        info = await drug_service.get_drug_info(drug_name)
        if info:
            # Mark that this came from local database if API was enabled
            if use_api is not False:  # use_api is None or True
                info['data_source'] = 'local_database_enhanced'
            return info

        # If no local data and APIs weren't explicitly disabled, try APIs as fallback
        if use_api is not False:
            logger.info(f"No local data for {drug_name}, trying APIs")
            api_result = await api_drug_service.get_drug_with_fallback(drug_name)
            if api_result:
                return api_result

        # No data found anywhere
        raise HTTPException(status_code=404, detail=f"No information found for {drug_name}")

    except Exception as e:
        logger.error(f"Error getting drug info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== API-DRIVEN DRUG ENDPOINTS ====================

@app.post("/api-drug-info")
async def get_api_drug_information(request: DrugInfoRequest):
    """
    Get comprehensive drug information using multiple free APIs (OpenFDA, PubChem, RxNorm)
    API-First approach with local database fallback
    """
    try:
        logger.info(f"🔍 Getting comprehensive API drug data for: {request.drug_name}")

        if request.use_api:
            # Use API-first approach with fallback
            api_result = await api_drug_service.get_drug_with_fallback(request.drug_name)
        else:
            # Use only local fallback
            local_result = await drug_service.get_drug_info(request.drug_name)
            if local_result:
                api_result = {
                    'drug_name': request.drug_name,
                    'source': 'local_database_only',
                    'data_quality': 'local',
                    'confidence_score': 0.7,
                    **local_result
                }
            else:
                raise HTTPException(status_code=404, detail=f"No information found for {request.drug_name}")

        return api_result

    except Exception as e:
        logger.error(f"Error getting API drug info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving drug information: {str(e)}")

@app.get("/api-drug-search/{query}")
async def search_api_drugs(query: str, sources: Optional[str] = None):
    """
    Search for drugs using free APIs with enhanced data
    Parameters:
    - query: Partial drug name or chemical identifier
    - sources: Comma-separated list of sources (openfda, pubchem, rxnorm). Default: all sources
    """
    try:
        logger.info(f"🔍 API drug search for: {query}")

        # Parse sources parameter
        requested_sources = sources.split(',') if sources else ['openfda', 'pubchem', 'rxnorm']

        search_results = {
            'query': query,
            'sources_searched': requested_sources,
            'comprehensive_results': []
        }

        # Try comprehensive API search first
        try:
            comprehensive_data = await api_drug_service.get_drug_with_fallback(query)

            if comprehensive_data and comprehensive_data.get('confidence_score', 0) > 0.6:
                search_results['comprehensive_results'].append({
                    'drug': query,
                    'data': comprehensive_data,
                    'search_method': 'comprehensive_api_search'
                })
        except Exception as api_error:
            logger.warning(f"API comprehensive search failed: {str(api_error)}")

        # Also search in local database as additional source
        local_info = await drug_service.get_drug_info(query)
        if local_info:
            search_results['comprehensive_results'].append({
                'drug': query,
                'data': {**local_info, 'source': 'local_database'},
                'search_method': 'local_database'
            })

        return search_results

    except Exception as e:
        logger.error(f"Error in API drug search: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in drug search: {str(e)}")

@app.get("/drug-comparison/{drug1}/{drug2}")
async def compare_drugs_api(drug1: str, drug2: str):
    """
    Compare two drugs using API data from multiple sources
    """
    try:
        logger.info(f"🔍 Comparing drugs: {drug1} vs {drug2}")

        # Get data for both drugs
        drug1_data = await api_drug_service.get_drug_with_fallback(drug1)
        drug2_data = await api_drug_service.get_drug_with_fallback(drug2)

        comparison = {
            'drug1': drug1,
            'drug2': drug2,
            'comparison_data': {
                'drug1_info': drug1_data,
                'drug2_info': drug2_data,
                'comparison_timestamp': datetime.utcnow().isoformat(),
                'comparison_sources': 'multiple_free_apis'
            },
            'similarities': [],
            'differences': []
        }

        # Simple comparison logic
        if drug1_data and drug2_data:
            # Compare manufacturers/brands
            drug1_brand = drug1_data.get('brand_name', [])
            drug2_brand = drug2_data.get('brand_name', [])

            if set(drug1_brand) & set(drug2_brand):
                comparison['similarities'].append("Shared brand names or manufacturers")

            # Compare molecular weights if available
            drug1_mw = drug1_data.get('molecular_weight')
            drug2_mw = drug2_data.get('molecular_weight')

            if drug1_mw and drug2_mw:
                mw_diff = abs(drug1_mw - drug2_mw)
                if mw_diff < 10:
                    comparison['similarities'].append(f"Similar molecular weight ({drug1_mw:.1f} vs {drug2_mw:.1f})")
                else:
                    comparison['differences'].append(f"Different molecular weight ({drug1_mw:.1f} vs {drug2_mw:.1f})")

        return comparison

    except Exception as e:
        logger.error(f"Error comparing drugs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error comparing drugs: {str(e)}")

@app.get("/drug-stats")
async def get_api_drug_stats():
    """
    Get statistics about API drug data usage and availability
    """
    try:
        stats = {
            'api_services': ['OpenFDA_API', 'PubChem_API', 'RxNorm_API'],
            'service_status': 'all_free_public_apis',
            'data_sources': [
                {
                    'name': 'US FDA Drug Label Database',
                    'purpose': 'Official drug labeling, usage, warnings, contraindications',
                    'accessibility': 'Public API - No authentication required'
                },
                {
                    'name': 'PubChem Chemical Database',
                    'purpose': 'Chemical structure, molecular formula, pharmacological data',
                    'accessibility': 'Public API - No authentication required'
                },
                {
                    'name': 'RxNorm Drug Standardization',
                    'purpose': 'Standardized drug names, coding, therapeutic classifications',
                    'accessibility': 'Public API - No authentication required'
                }
            ],
            'last_updated': datetime.utcnow().isoformat(),
            'api_features': [
                'Real-time data from authoritative sources',
                'Comprehensive drug information',
                'Multiple source validation',
                'Automatic fallback to local database',
                '400+ million chemical compounds in PubChem',
                'Quarterly updated FDA data',
                'International drug name standardization'
            ]
        }

        return stats

    except Exception as e:
        logger.error(f"Error getting API stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting statistics: {str(e)}")

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

@app.post("/extract-text-from-image")
async def extract_text_from_image(file: UploadFile = File(...), language: Optional[str] = "en"):
    """
    Extract text from prescription image using OCR (step 1)
    Returns extracted text immediately for user review
    """
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file uploaded")

        if not file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            raise HTTPException(status_code=400, detail="Only image files (JPG, PNG) are supported for OCR extraction")

        # Save uploaded file temporarily
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)

        file_path = os.path.join(temp_dir, file.filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        try:
            logger.info(f"🔍 Starting OCR text extraction from image: {file.filename}")

            # Load image
            import PIL.Image
            image = PIL.Image.open(file_path)

            # Use Gemini OCR to extract text only
            prompt = """
            Extract all visible text from this prescription image as accurately as possible.
            Return ONLY the raw text content that you can read. Do not add explanations,
            summaries, or any additional content. Just return the extracted text exactly as it appears.
            """

            if nlp_service.ocr_model:
                loop = asyncio.get_event_loop()

                def _extract_text():
                    try:
                        response = nlp_service.ocr_model.generate_content([prompt, image])
                        return response.text.strip()
                    except Exception as e:
                        logger.error(f"OCR extraction failed: {str(e)}")
                        return "OCR extraction failed. The image may be unclear or unsupported."

                extracted_text = await loop.run_in_executor(None, _extract_text)

                if extracted_text and len(extracted_text) > 10:
                    logger.info(f"✅ OCR extraction successful, extracted {len(extracted_text)} characters")
                    ocr_success = True
                else:
                    logger.warning("⚠️ OCR extraction produced very little text")
                    extracted_text = extracted_text or "Minimal text extracted - image may be unclear"
                    ocr_success = False
            else:
                extracted_text = "⚠️ OCR service unavailable. Please check Gemini API configuration."
                ocr_success = False

            # Clean up temp file
            os.remove(file_path)

            return {
                "extracted_text": extracted_text,
                "ocr_success": ocr_success,
                "file_processed": file.filename,
                "language": language or "en",
                "timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            # Clean up temp file
            if os.path.exists(file_path):
                os.remove(file_path)

            logger.error(f"OCR extraction error: {str(e)}")
            return {
                "extracted_text": f"❌ OCR extraction failed: {str(e)}",
                "ocr_success": False,
                "file_processed": file.filename,
                "language": language or "en",
                "timestamp": datetime.utcnow().isoformat()
            }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR processing error: {str(e)}")

@app.post("/analyze-extracted-text")
async def analyze_extracted_text(request: PrescriptionText):
    """
    Analyze previously extracted text with Granite AI (step 2)
    """
    try:
        logger.info(f"🤖 Analyzing extracted text with Granite AI: {len(request.text)} characters")

        # Use the existing prescription parsing with enhanced Granite processing
        parsed_data = await nlp_service.parse_prescription(
            text=request.text,
            language=request.language or "en"
        )

        # Add metadata to show this came from OCR analysis
        parsed_data['analyzed_from_ocr'] = True
        parsed_data['processing_method'] = 'ocr_to_granite_analysis'

        logger.info(f"✅ Granite AI analysis complete - confidence: {parsed_data.get('confidence_score', 0):.2f}")

        return parsed_data

    except Exception as e:
        logger.error(f"Granite analysis error: {str(e)}")
        return {
            'medications': [],
            'dosages': [],
            'frequencies': [],
            'routes': [],
            'confidence_score': 0.0,
            'warnings': [f"Granite AI analysis failed: {str(e)}"],
            'processing_method': 'ocr_fallback_analysis',
            'analyzed_from_ocr': False
        }

@app.post("/parse-prescription-file")
async def parse_prescription_file(file: UploadFile = File(...), language: Optional[str] = "en"):
    """
    Legacy endpoint - now redirects to two-step process for images
    For backward compatibility with existing code
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
            parsed_data = None

            # Extract text from file based on type
            if file.filename.lower().endswith(('.txt', '.md')):
                # Use Ollama Granite for text files
                text = content.decode('utf-8', errors='ignore')
                logger.info(f"Processing text file with Granite AI model")
                parsed_data = await nlp_service.parse_prescription(text, language or "en")

            elif file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Use the new two-step OCR process for images
                logger.info(f"🔍 Processing image file with OCR + Granite workflow")
                parsed_data = await nlp_service.parse_prescription_from_image(file_path, language or "en")

            elif file.filename.lower().endswith(('.pdf', 'docx')):
                raise HTTPException(
                    status_code=400,
                    detail="PDF and DOCX files require additional libraries. For now, please convert to text or use image files for OCR."
                )
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Unsupported file format. Supported: TXT (Granite AI), JPG/PNG (Gemini OCR)"
                )

            # Clean up temp file
            os.remove(file_path)

            # Add processing metadata
            parsed_data['processed_by'] = 'Ollama Granite' if file.filename.lower().endswith(('.txt', '.md')) else 'Gemini OCR + Granite Analysis'

            return parsed_data

        except Exception as e:
            # Clean up temp file in case of error
            if os.path.exists(file_path):
                os.remove(file_path)

            # Ensure we have a valid response even on error
            error_result = {
                'medications': [],
                'dosages': [],
                'frequencies': [],
                'routes': [],
                'confidence_score': 0.0,
                'warnings': [f"File processing failed: {str(e)}"],
                'processing_method': 'error_fallback',
                'processed_by': 'Processing Failed'
            }

            return error_result

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
    await api_drug_service.initialize()

    # Set up API service fallback to local drug service
    api_drug_service.set_fallback_service(drug_service)

    # Link API service to dosage service for unknown drugs
    dosage_service.set_api_service(api_drug_service)

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
    await api_drug_service.cleanup()

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )