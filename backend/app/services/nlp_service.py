# backend/app/services/nlp_service.py
import logging
import json
import asyncio
import re
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import os
from dotenv import load_dotenv


load_dotenv()




# Google Gemini imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Gemini SDK not available")

# Ollama imports
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama SDK not available")

import aiohttp
import sqlite3

logger = logging.getLogger(__name__)

class PrescriptionNLPService:
    """
    NLP service for parsing medical prescriptions using:
    1. Ollama Granite AI (for text processing)
    2. Google Gemini 1.5 Flash (for OCR/image processing)
    3. Custom rule-based extraction (fallback)
    """
    
    def __init__(self):
        self.db_path = "data/prescription_cache.db"
        # Try granite model, fallback to dependable models if needed
        self.ollama_model = self._determine_best_ollama_model()

        self.gemini_model_name = "gemini-1.5-flash"

        logger.info(f"🤖 Initializing with Ollama model: {self.ollama_model}")

        # Initialize Ollama for text processing
        self.ollama_client = None
        self._init_ollama()

        # Initialize Gemini for OCR processing
        self.gemini_model = None  # For OCR with vision capabilities
        self.ocr_model = None     # Specific OCR model
        self._init_gemini()

        # Initialize database
        self._init_database()

        # Medical entity patterns
        self._init_patterns()

    def _determine_best_ollama_model(self) -> str:
        """Determine the best available Ollama model for text processing"""
        if not OLLAMA_AVAILABLE:
            return "none"

        # Preferred model order - use granite if available
        preferred_models = [
            "granite3.3:2b",    # IBM Granite 3.3B
            "granite3.1:2b",    # IBM Granite 3.1B
            "llama3.2:3b",      # Llama 3.2 3B (lightweight)
            "mistral:7b",       # Mistral 7B
            "llama3.1:8b",      # Llama 3.1 8B
            "codellama:7b",     # Code Llama 7B
            "llama2:7b",        # Llama 2 7B as fallback
        ]

        # Try to find an available model
        try:
            # List available models from Ollama
            host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            client = ollama.Client(host=host)

            # Check if any preferred models are available
            available_models = [model['name'] for model in client.list()['models']]

            logger.info(f"Available Ollama models: {available_models}")

            for model in preferred_models:
                if model in available_models:
                    logger.info(f"✅ Selected Ollama model: {model}")
                    return model

            # If no preferred model is available, use first available model (if any)
            if available_models:
                first_available = available_models[0]
                logger.warning(f"⚠️  No preferred model found. Using: {first_available}")
                return first_available
            else:
                logger.error("❌ No Ollama models available")
                return "none"

        except Exception as e:
            logger.error(f"❌ Failed to check Ollama models: {str(e)}")
            return "none"

    def _init_ollama(self):
        """Initialize Ollama client"""
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama SDK not available")
            self.ollama_client = None
            return

        try:
            host = os.getenv('OLLAMA_HOST', 'http://localhost:11434')
            self.ollama_client = ollama.Client(host=host)
            logger.info("Ollama client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {str(e)}")
            self.ollama_client = None
    
    def _init_gemini(self):
        """Initialize Google Gemini 1.5 Flash for OCR processing"""
        if not GEMINI_AVAILABLE:
            logger.warning("Google Gemini SDK not available")
            self.gemini_model = None
            self.ocr_model = None
            return

        try:
            api_key = os.getenv('GEMINI_API_KEY')
            logger.info(f"Gemini API key detected: {'Yes' if api_key else 'No'}")
            if api_key:
                logger.info("Configuring Gemini with API key...")
                genai.configure(api_key=api_key)
                logger.info("Creating Gemini 1.5 Flash model...")
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')
                self.ocr_model = self.gemini_model  # Same model for both
                logger.info("✅ Google Gemini 1.5 Flash (OCR-ready) initialized successfully")
                logger.info("🧠 OCR processing capabilities are now available")
            else:
                logger.warning("❌ Gemini API key not found in environment variables")
                logger.warning("GEMINI_API_KEY environment variable must be set")
                logger.warning("Please check your .env file or environment settings")
                self.gemini_model = None
                self.ocr_model = None

        except Exception as e:
            logger.error(f"❌ Failed to initialize Gemini: {str(e)}")
            logger.error("Make sure GOOGLE_GENERATIVEAI_API_KEY is properly configured")
            self.gemini_model = None
            self.ocr_model = None
    
    def _init_database(self):
        """Initialize SQLite database for caching"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS prescription_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        text_hash TEXT UNIQUE NOT NULL,
                        original_text TEXT NOT NULL,
                        parsed_data TEXT NOT NULL,
                        processing_method TEXT,
                        confidence_score REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS entity_cache (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entity_text TEXT NOT NULL,
                        entity_type TEXT NOT NULL,
                        confidence REAL,
                        source TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                logger.info("NLP database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing NLP database: {str(e)}")
    
    def _init_patterns(self):
        """Initialize regex patterns for medical entity extraction"""
        self.patterns = {
            'medications': [
                r'\b(?:tablet|capsule|mg|mcg|ml|units?)\s+(?:of\s+)?([A-Za-z]+(?:\s+[A-Za-z]+)*)\b',
                r'\b([A-Za-z]+(?:\s+[A-Za-z]+)*)\s+(?:\d+\s*(?:mg|mcg|ml|units?))',
                r'\b(Aspirin|Ibuprofen|Acetaminophen|Warfarin|Lisinopril|Metformin|Atorvastatin|Levothyroxine|Amlodipine|Metoprolol|Omeprazole|Simvastatin|Furosemide|Digoxin|Insulin)\b'
            ],
            'dosages': [
                r'\b(\d+(?:\.\d+)?)\s*(mg|mcg|g|ml|units?|tablets?|capsules?)\b',
                r'\b(\d+)\s*x\s*(\d+)\s*(mg|mcg|ml|units?)\b',
                r'\b(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)\s*(mg|ml)\b'
            ],
            'frequencies': [
                r'\b(?:take\s+)?(?:(\d+)\s*(?:times?|x)\s*(?:per\s+|a\s+)?(?:day|daily|week|month))\b',
                r'\b(?:once|twice|thrice)\s*(?:per\s+|a\s+)?(?:day|daily|week|month)\b',
                r'\b(?:every\s+\d+\s*hours?|q\d+h|bid|tid|qid|qd)\b',
                r'\b(?:morning|evening|bedtime|as\s*needed|prn)\b'
            ],
            'routes': [
                r'\b(oral|orally|by\s*mouth|sublingual|topical|intramuscular|intravenous|subcutaneous|rectal|vaginal|ophthalmic|otic|nasal|transdermal|inhaled)\b',
                r'\b(PO|IM|IV|SC|SQ|PR|PV|OU|OS|OD|NAS|TD|INH)\b'
            ],
            'indications': [
                r'\bfor\s+(?:the\s+)?(?:treatment\s+of\s+)?([A-Za-z\s]+)\b',
                r'\b(?:to\s+treat|treating)\s+([A-Za-z\s]+)\b',
                r'\b(?:indicated\s+for|used\s+for)\s+([A-Za-z\s]+)\b'
            ]
        }
    
    async def initialize(self):
        """Initialize the NLP service"""
        logger.info("Initializing Prescription NLP Service")
        
    async def cleanup(self):
        """Cleanup resources"""
        pass
    
    async def parse_prescription_from_image(self, image_path: str, language: str = "en") -> Dict[str, Any]:
        """
        Parse prescription from image using two-phase approach:
        1. Gemini OCR extracts text from image
        2. Granite analyzes the extracted text
        """
        try:
            if not self.ocr_model:
                raise Exception("Gemini OCR model not available")

            logger.info(f"Starting OCR phase 1: Text extraction from image {image_path}")

            # Load image
            import PIL.Image
            image = PIL.Image.open(image_path)

            # Phase 1: Extract text from image using Gemini OCR
            ocr_prompt = """
            Extract all the text content from this prescription image.
            Return ONLY the raw text that you can read from the image.

            Do not include your own analysis or explanations. Just return the exact text content as it appears.
            If you cannot read certain parts, indicate [UNCLEAR TEXT] for those sections.

            Focus on:
            - Patient information
            - Medication names and dosages
            - Administration instructions
            - Physician/prescriber details
            """

            ocr_response = await self._call_gemini_ocr_only(image, ocr_prompt)
            extracted_text = ocr_response.strip()

            logger.info(f"OCR Phase 1 completed. Extracted text length: {len(extracted_text)}")
            logger.info(f"Extracted text preview: {extracted_text[:100]}...")

            # Phase 2: Process the extracted text with Granite (primary) and fallbacks
            logger.info("Starting Phase 2: Text analysis with Granite model")

            # Use the main parse_prescription method with the extracted text
            structured_result = await self.parse_prescription(extracted_text, language)

            # Add OCR-specific metadata
            structured_result['ocr_extracted_text'] = extracted_text

            # Keep original processing method info and add OCR phases
            original_method = structured_result.get('processing_method', 'unknown')
            if extracted_text and len(extracted_text) > 10:
                structured_result['processing_method'] = f'gemini_ocr_extractor + {original_method}'
                structured_result['phase1_ocr_success'] = True

                logger.info(f"Phase 2 completed successfully. Processing method: {structured_result['processing_method']}")
                logger.info(f"Confidence score: {structured_result.get('confidence_score', 0):.2f}")
            else:
                structured_result['processing_method'] = f'gemini_ocr_failed + {original_method}'
                structured_result['phase1_ocr_success'] = False
                structured_result['warnings'] = structured_result.get('warnings', []) + ["OCR extraction may have failed or returned empty text"]

            logger.info(f"Complete OCR+Analysis pipeline completed. Method: {structured_result.get('processing_method')}")
            logger.info(f"Confidence: {structured_result.get('confidence_score', 0):.2f}")

            return structured_result

        except Exception as e:
            logger.error(f"OCR pipeline error: {str(e)}")
            # Return a basic error result with extracted text if available
            return await self._parse_with_rules(f"ocr_pipeline_failed: {str(e)}")

    async def parse_prescription(self, text: str, language: str = "en") -> Dict[str, Any]:
        """
        Main method to parse prescription text using multiple NLP approaches with improved fallbacks
        """
        try:
            # Input validation
            if not text or not text.strip():
                logger.warning("Empty prescription text provided")
                result = self._create_empty_result()
                result['processing_method'] = 'empty_input'
                result['warnings'] = ['No prescription text provided']
                return result

            text_hash = self._generate_text_hash(text.strip())

            # Check cache first (skip for very short texts)
            if len(text.strip()) > 10:
                cached_result = await self._get_cached_result(text_hash)
                if cached_result:
                    logger.info("✅ Using cached prescription analysis")
                    return cached_result

            # Try different NLP approaches in order of preference
            result = None
            processing_method = "unknown"

            # 1. Try Ollama Granite (primary for text)
            if self.ollama_client and hasattr(self.ollama_client, 'chat') and self.ollama_model != "none":
                try:
                    logger.info(f"🤖 Attempting to parse with Ollama Granite ({self.ollama_model})")
                    result = await self._parse_with_ollama(text)
                    processing_method = f"ollama_{self.ollama_model.replace(':', '_')}"
                    logger.info(f"✅ Successfully parsed with Ollama {self.ollama_model}")

                    # Check if we got meaningful results
                    if result.get('medications') and len(result['medications']) > 0:
                        logger.info(f"✅ Found {len(result['medications'])} medications")
                    else:
                        logger.warning("⚠️ Ollama returned empty medication list - considering fallback")

                except Exception as e:
                    logger.warning(f"❌ Ollama parsing failed: {str(e)}")
                    logger.info("🔄 Falling back to rule-based parsing")
                    result = None

            # 2. Try rule-based extraction (fallback) if AI failed or results are inadequate
            if not result or result.get('confidence_score', 0) < 0.1 or not result.get('medications'):
                logger.info("🔄 Using rule-based parsing as fallback method")
                rule_result = await self._parse_with_rules(text)

                if result and result.get('medications'):
                    # Keep AI results if they exist, but enhance with rule-based if needed
                    result = rule_result
                else:
                    # Use rule-based results if no AI results
                    result = rule_result

                if processing_method.startswith("ollama_") and result.get('medications'):
                    processing_method = "ollama_enhanced_rules"
                else:
                    processing_method = "rules_fallback"

            # Ensure we have a valid result structure
            if not self._is_valid_result(result):
                logger.warning("Invalid result structure - creating fallback result")
                result = await self._parse_with_rules(text)
                processing_method = "fallback_rules_only"

            # Add processing metadata
            result['processing_method'] = processing_method
            result['parsed_at'] = datetime.utcnow().isoformat()
            result['input_length'] = len(text.strip())

            # Add source information for debugging
            result['ollama_available'] = bool(self.ollama_client and self.ollama_model != "none")
            result['gemini_available'] = bool(self.gemini_model)

            # Cache the result if meaningful (avoid caching empty results)
            if len(text.strip()) > 10 and result.get('confidence_score', 0) > 0:
                try:
                    await self._cache_result(text_hash, text.strip(), result)
                except Exception as cache_error:
                    logger.warning(f"Failed to cache result: {cache_error}")

            logger.info(f"📊 Final parsing result: {processing_method}, confidence: {result.get('confidence_score', 0):.2f}")

            return result

        except Exception as e:
            logger.error(f"Critical error in parse_prescription: {str(e)}")

            # Return fallback result with error information
            fallback_result = await self._parse_with_rules(text)
            fallback_result['processing_method'] = 'error_fallback_rules'
            fallback_result['warnings'] = fallback_result.get('warnings', []) + [f'Critical parsing error: {str(e)}']
            fallback_result['confidence_score'] = 0.0
            fallback_result['ollama_available'] = bool(self.ollama_client and self.ollama_model != "none")

            return fallback_result
    
    async def _parse_with_ollama(self, text: str) -> Dict[str, Any]:
        """Parse prescription using Ollama Granite model with improved prompts"""
        if not self.ollama_client:
            raise Exception("Ollama client not initialized")

        try:
            enhanced_prompt = f"""
            # MEDICAL PRESCRIPTION ANALYSIS TASK

            You are a medical prescription parser. Analyze the following prescription text and extract structured information.

            ## INPUT TEXT:
            {text}

            ## OUTPUT FORMAT:
            You must respond with JSON only. Extract medical information and return this exact JSON structure:

            {{
                "medications": [
                    {{
                        "name": "medication_name",
                        "confidence": 0.85
                    }}
                ],
                "dosages": [
                    {{
                        "amount": "dose_amount",
                        "unit": "dose_unit",
                        "medication": "medication_name"
                    }}
                ],
                "frequencies": [
                    "frequency_text (e.g., 'twice daily', 'qid')"
                ],
                "routes": [
                    "administration_route (e.g., 'oral', 'IV', 'topical')"
                ],
                "confidence_score": 0.85,
                "warnings": ["any warnings or notes"]
            }}

            ## EXTRACTION RULES:
            1. Only extract information explicitly present in the text
            2. Do not make assumptions or add external knowledge
            3. Leave arrays empty if information is not found
            4. Use exact medication names from the text
            5. Be case-insensitive but return medication names with proper capitalization
            6. If no information, return empty arrays but valid JSON

            ## IMPORTANT:
            JSON response only - no explanations, no markdown, no additional text outside the JSON structure.
            """

            response = await self._call_ollama_async(enhanced_prompt)

            # Parse Ollama response with better error handling
            try:
                # Look for JSON in the response
                response_clean = response.strip()

                # Remove any text before first {{ and after last }}
                json_start = response_clean.find('{')
                json_end = response_clean.rfind('}') + 1

                if json_start >= 0 and json_end > json_start:
                    json_str = response_clean[json_start:json_end]
                    ollama_result = json.loads(json_str)

                    logger.info(f"Successfully parsed Ollama response with {len(ollama_result.get('medications', []))} medications")

                    # Validate and enhance results
                    if 'medications' not in ollama_result:
                        ollama_result['medications'] = []

                    if 'medications' in ollama_result:
                        for med in ollama_result['medications']:
                            med['source'] = 'ollama_granite'
                            if 'confidence' not in med or med['confidence'] is None:
                                med['confidence'] = 0.8
                            # Ensure proper name capitalization
                            if 'name' in med:
                                med['name'] = med['name'].title()

                    # Ensure all required fields
                    ollama_result.setdefault('dosages', [])
                    ollama_result.setdefault('frequencies', [])
                    ollama_result.setdefault('routes', [])
                    ollama_result.setdefault('indications', [])
                    ollama_result.setdefault('warnings', [])

                    # Calculate confidence score
                    med_confidences = [m.get('confidence', 0) for m in ollama_result.get('medications', [])]
                    if med_confidences:
                        ollama_result['confidence_score'] = sum(med_confidences) / len(med_confidences)
                    else:
                        ollama_result['confidence_score'] = 0.1  # Low confidence if no medications found

                    return ollama_result
                else:
                    # If no JSON found, try to find any JSON content
                    json_pattern = r'\{.*\}'
                    match = re.search(json_pattern, response_clean, re.DOTALL)
                    if match:
                        json_str = "{ " + match.group(0)[1:-1] + " }"
                        json_str = re.sub(r'[^{},\[\]:{}\n\t"]*(?=[{,\[\]:{} ])', '""', json_str)  # Fill missing keys with null
                        ollama_result = json.loads(json_str)
                        logger.warning("Recovered malformed JSON from Ollama response")

                        # Apply same validation as above
                        ollama_result.setdefault('medications', [])
                        for med in ollama_result['medications']:
                            med['source'] = 'ollama_recovered'
                            med['confidence'] = med.get('confidence', 0.5)
                        ollama_result.setdefault('dosages', [])
                        ollama_result.setdefault('frequencies', [])
                        ollama_result.setdefault('routes', [])
                        ollama_result.setdefault('indications', [])
                        ollama_result.setdefault('warnings', [])
                        ollama_result['confidence_score'] = 0.1
                        return ollama_result

                    raise ValueError("No JSON structure found in Ollama response")

            except json.JSONDecodeError as e:
                logger.warning(f"Ollama JSON parsing failed: {str(e)}. Response: {response[:200]}...")
                logger.info("Falling back to rule-based parsing due to Ollama JSON parsing issues")

                # Fall back to rule-based extraction with error indication
                result = await self._parse_with_rules(text)
                result['processing_method'] = 'ollama_json_failed_fallback'
                result['warnings'].append('🤖 AI model response was not in proper JSON format - using fallback parsing')
                return result

        except Exception as e:
            logger.error(f"Ollama parsing error: {str(e)}")
            logger.info("Using rule-based fallback due to Ollama error")

            # Return rule-based parsing as final fallback
            result = await self._parse_with_rules(text)
            result['processing_method'] = 'ollama_error_fallback'
            result['warnings'].append(f'Ollama processing failed: {str(e)}')
            return result
    
    
    async def _parse_with_gemini(self, text: str) -> Dict[str, Any]:
        """Parse prescription using Google Gemini AI"""
        if not self.gemini_model:
            raise Exception("Gemini model not initialized")
        
        try:
            # Optimize prompt for Granite AI model
            prompt = f"""Analyze this medical prescription carefully and extract the key information:

PRESCRIPTION TEXT: "{text}"

INSTRUCTIONS:
1. Identify all medications mentioned (including brand and generic names)
2. Extract exact dosages (amounts and units)
3. Note administration frequencies (daily, BID, TID, etc.)
4. Identify route of administration (oral, IV, topical, etc.)
5. Look for medical indications or conditions being treated

FORMAT YOUR RESPONSE AS JSON:
{{
    "medications": [
        {{"name": "medication_name", "confidence": 0.85}}
    ],
    "dosages": [
        {{"amount": "dose_amount", "unit": "dose_unit", "medication": "related_medication"}}
    ],
    "frequencies": ["how_many_times_per_day"],
    "routes": ["administration_method"],
    "indications": ["medical_conditions"],
    "confidence_score": 0.85,
    "warnings": []
}}

IMPORTANT:
- Only include information actually present in the text
- Use exact statements from the prescription
- If something is missing, leave the array empty
- Be precise about medication names and dosages
- Do not make assumptions or add information
- Focus on medical facts from the prescription

Now analyze the prescription text above."""
            
            response = await self._call_gemini_async(prompt)
            
            # Parse Gemini response
            try:
                # Extract JSON from response
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    gemini_result = json.loads(json_str)
                    
                    # Add confidence and source information
                    if 'medications' in gemini_result:
                        for med in gemini_result['medications']:
                            med['source'] = 'gemini_ai'
                            if 'confidence' not in med:
                                med['confidence'] = 0.8  # Default confidence for Gemini

                    # Ensure all required fields are present
                    gemini_result.setdefault('dosages', [])
                    gemini_result.setdefault('frequencies', [])
                    gemini_result.setdefault('routes', [])
                    gemini_result.setdefault('warnings', [])
                    gemini_result.setdefault('confidence_score',
                        self._calculate_confidence(gemini_result.get('medications', [])))

                    return gemini_result
                else:
                    raise ValueError("No valid JSON in Gemini response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Failed to parse Gemini JSON response: {str(e)}")
                # Fall back to rule-based extraction
                return await self._parse_with_rules(text)
            
        except Exception as e:
            logger.error(f"Gemini AI error: {str(e)}")
            raise
    
    async def _call_gemini_async(self, prompt: str) -> str:
        """Call Gemini API asynchronously"""
        if not self.gemini_model:
            raise Exception("Gemini model not available")

        loop = asyncio.get_event_loop()

        def _call_gemini():
            try:
                response = self.gemini_model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"Error calling Gemini: {str(e)}"

        return await loop.run_in_executor(None, _call_gemini)

    async def _call_gemini_for_image(self, image, prompt: str) -> str:
        """Call Gemini for image OCR processing"""
        if not self.ocr_model:
            raise Exception("Gemini OCR model not available")

        loop = asyncio.get_event_loop()

        def _call_gemini_ocr():
            try:
                # Import PIL here to handle image processing
                import PIL.Image

                response = self.ocr_model.generate_content([prompt, image])
                return response.text
            except Exception as e:
                return f"Error calling Gemini OCR: {str(e)}"

        return await loop.run_in_executor(None, _call_gemini_ocr)

    async def _call_gemini_ocr_only(self, image, prompt: str) -> str:
        """Call Gemini for pure text extraction from image (OCR only)"""
        if not self.ocr_model:
            raise Exception("Gemini OCR model not available for text extraction")

        loop = asyncio.get_event_loop()

        def _call_gemini_ocr():
            try:
                response = self.ocr_model.generate_content([prompt, image])
                return response.text
            except Exception as e:
                logger.error(f"Gemini OCR text extraction error: {str(e)}")
                return "OCR text extraction failed"

        return await loop.run_in_executor(None, _call_gemini_ocr)

    async def _call_ollama_async(self, prompt: str) -> str:
        """Call Ollama API asynchronously"""
        if not self.ollama_client:
            raise Exception("Ollama client not available")

        loop = asyncio.get_event_loop()

        def _call_ollama():
            try:
                response = self.ollama_client.chat(model=self.ollama_model, messages=[{'role': 'user', 'content': prompt}])
                return response['message']['content']
            except Exception as e:
                return f"Error calling Ollama: {str(e)}"

        return await loop.run_in_executor(None, _call_ollama)
    
    async def _parse_with_rules(self, text: str) -> Dict[str, Any]:
        """Parse prescription using rule-based regex patterns"""
        result: Dict[str, Any] = {
            'medications': [],
            'dosages': [],
            'frequencies': [],
            'routes': [],
            'indications': [],
            'confidence_score': 0.7,  # Default confidence for rule-based
            'warnings': []  # Initialize warnings
        }
        
        try:
            text_lower = text.lower()
            
            # Extract medications with better filtering
            for pattern in self.patterns['medications']:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    med_name = match.group(1).strip()

                    # Filter out very short matches and non-medication-like words
                    if self._is_valid_medication_name(med_name):
                        result['medications'].append({
                            'name': med_name.title(),
                            'confidence': 0.7,
                            'source': 'regex_pattern',
                            'pattern_matched': pattern
                        })
            
            # Extract dosages
            for pattern in self.patterns['dosages']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    dosage_info = {
                        'amount': match.group(1),
                        'unit': match.group(2) if len(match.groups()) > 1 else 'units',
                        'source': 'regex_pattern'
                    }
                    result['dosages'].append(dosage_info)
            
            # Extract frequencies
            for pattern in self.patterns['frequencies']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    result['frequencies'].append(match.group(0))
            
            # Extract routes
            for pattern in self.patterns['routes']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    result['routes'].append(match.group(1))
            
            # Extract indications
            for pattern in self.patterns['indications']:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    result['indications'].append(match.group(1))
            
            # Calculate confidence based on number of entities found
            total_entities = sum(len(result[key]) for key in result.keys())
            result['confidence_score'] = min(total_entities * 0.1, 1.0)
            result['warnings'] = self._generate_warnings(result['medications'])
            
            return result
            
        except Exception as e:
            logger.error(f"Rule-based parsing error: {str(e)}")
            return result
    
    def _is_valid_medication_name(self, text: str) -> bool:
        """Check if text is likely a valid medication name"""
        if not text or len(text) < 3 or len(text) > 50:
            return False

        text_lower = text.lower()

        # Filter out common English words, process terms, and single letters
        invalid_words = {
            'and', 'the', 'for', 'with', 'take', 'process', 'backend',
            'one', 'two', 'three', 'four', 'five', 'six', 'can', 'may',
            'will', 'now', 'time', 'day', 'week', 'month', 'year',
            'milk', 'water', 'food', 'diet', 'exercise', 'add', 'send',
            'this', 'that', 'have', 'been', 'are', 'was', 'were',
            'very', 'more', 'some', 'then', 'well', 'much', 'must',
            'from', 'into', 'upon', 'over', 'under', 'last', 'next',
            'place', 'ward', 'card', 'note', 'order', 'phone', 'call'
        }

        if text_lower in invalid_words:
            return False

        # Common medication suffixes and patterns
        valid_patterns = [
            r'.*(?:cillin|mycin|pril|artan|statin|zole|pine|ide|ine|ol)$',
            r'.*(?:tab|cap|capsule|vial|inhaler|drops|syringe)\b',
            r'^(?:aspirin|ibuprofen|acetaminophen|warfarin|lisinopril|metformin|atorvastatin|levothyroxine|amlodipine|metoprolol|omeprazole|simvastatin|furosemide|digoxin|insulin|advair|albuterol|salbutamol|prednisone|fluoxetine|sertraline|gabapentin|tramadol)$'
        ]

        return any(re.search(pattern, text_lower) for pattern in valid_patterns)

    def _is_medication(self, text: str) -> bool:
        """Check if text is likely a medication name"""
        return self._is_valid_medication_name(text)
    
    def _deduplicate_entities(self, entities: List[Dict]) -> List[Dict]:
        """Remove duplicate entities based on name similarity"""
        unique_entities = []
        seen_names = set()
        
        for entity in entities:
            name = entity.get('name', '').lower().strip()
            if name and name not in seen_names:
                seen_names.add(name)
                unique_entities.append(entity)
        
        # Sort by confidence
        return sorted(unique_entities, key=lambda x: x.get('confidence', 0), reverse=True)
    
    def _calculate_confidence(self, medications: List[Dict]) -> float:
        """Calculate overall confidence score"""
        if not medications:
            return 0.0
        
        confidences = [med.get('confidence', 0) for med in medications]
        return sum(confidences) / len(confidences)
    
    def _generate_warnings(self, medications: List[Dict]) -> List[str]:
        """Generate warnings based on extracted medications"""
        warnings = []
        
        if not medications:
            warnings.append("⚠️ No medications clearly identified - manual review recommended")
        
        low_confidence_meds = [med for med in medications if med.get('confidence', 0) < 0.5]
        if low_confidence_meds:
            warnings.append(f"⚠️ {len(low_confidence_meds)} medication(s) identified with low confidence")
        
        # Check for high-risk medications
        high_risk_drugs = ['warfarin', 'insulin', 'digoxin', 'lithium']
        found_high_risk = [med for med in medications 
                          if any(risk_drug in med.get('name', '').lower() for risk_drug in high_risk_drugs)]
        
        if found_high_risk:
            warnings.append("🚨 High-risk medication detected - verify dosing carefully")
        
        return warnings
    
    def _create_empty_result(self) -> Dict[str, Any]:
        """Create a standard empty result structure"""
        return {
            'medications': [],
            'dosages': [],
            'frequencies': [],
            'routes': [],
            'indications': [],
            'confidence_score': 0.0,
            'warnings': [],
            'processing_method': 'empty_result',
            'parsed_at': datetime.utcnow().isoformat()
        }

    def _is_valid_result(self, result: Dict[str, Any]) -> bool:
        """Check if a result has the required structure"""
        if not isinstance(result, dict):
            return False

        required_fields = ['medications', 'dosages', 'frequencies', 'routes', 'confidence_score']
        for field in required_fields:
            if field not in result:
                logger.warning(f"Result missing required field: {field}")
                return False

        # Check that fields have correct types
        if not isinstance(result['medications'], list):
            return False
        if not isinstance(result['dosages'], list):
            return False
        if not isinstance(result['frequencies'], list):
            return False
        if not isinstance(result['routes'], list):
            return False
        if not isinstance(result['confidence_score'], (int, float)):
            return False

        return True

    def _generate_text_hash(self, text: str) -> str:
        """Generate hash for text caching"""
        import hashlib
        return hashlib.md5(text.encode()).hexdigest()
    
    async def _get_cached_result(self, text_hash: str) -> Optional[Dict]:
        """Get cached parsing result"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT parsed_data FROM prescription_cache 
                    WHERE text_hash = ? AND created_at > datetime('now', '-1 hour')
                """, (text_hash,))
                
                result = cursor.fetchone()
                if result:
                    return json.loads(result[0])
                    
        except Exception as e:
            logger.error(f"Error getting cached result: {str(e)}")
        
        return None
    
    async def _cache_result(self, text_hash: str, original_text: str, result: Dict):
        """Cache parsing result"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO prescription_cache 
                    (text_hash, original_text, parsed_data, processing_method, confidence_score)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    text_hash, 
                    original_text, 
                    json.dumps(result),
                    result.get('processing_method', 'unknown'),
                    result.get('confidence_score', 0)
                ))
                
        except Exception as e:
            logger.error(f"Error caching result: {str(e)}")
    
    async def extract_entities(self, text: str) -> List[Dict]:
        """Extract medical entities from text"""
        entities = []
        
        # Try all available NLP methods
        if self.ollama_client:
            try:
                ollama_entities = await self._extract_entities_ollama(text)
                entities.extend(ollama_entities)
            except Exception as e:
                logger.warning(f"Ollama entity extraction failed: {str(e)}")

        # Rule-based extraction
        rule_entities = await self._extract_entities_rules(text)
        entities.extend(rule_entities)
        
        return self._deduplicate_entities(entities)

    async def _extract_entities_ollama(self, text: str) -> List[Dict]:
        """Extract entities using Ollama"""
        entities = []
        
        try:
            parsed_data = await self._parse_with_ollama(text)
            if 'medications' in parsed_data:
                for med in parsed_data['medications']:
                    entities.append({
                        'name': med['name'],
                        'type': 'MEDICATION',
                        'confidence': med['confidence'],
                        'source': 'ollama'
                    })
        except Exception as e:
            logger.error(f"Ollama entity extraction error: {str(e)}")
        
        return entities
    
    
    
    
    async def _extract_entities_rules(self, text: str) -> List[Dict]:
        """Extract entities using rule-based patterns"""
        entities = []
        
        try:
            # Use existing rule-based parsing
            parsed = await self._parse_with_rules(text)
            
            for med in parsed['medications']:
                entities.append({
                    'name': med['name'],
                    'type': 'MEDICATION',
                    'confidence': med['confidence'],
                    'source': 'rules'
                })
                
        except Exception as e:
            logger.error(f"Rule-based entity extraction error: {str(e)}")
        
        return entities
