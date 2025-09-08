# backend/app/services/nlp_service.py
import logging
import json
import asyncio
import re
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
import os

# IBM Watson imports
try:
    from ibm_watson import NaturalLanguageUnderstandingV1
    from ibm_watson.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions
    from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
    WATSON_AVAILABLE = True
except ImportError:
    WATSON_AVAILABLE = False
    logging.warning("IBM Watson SDK not available")

# HuggingFace imports
try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    import torch
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logging.warning("HuggingFace transformers not available")

# Google Gemini imports
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logging.warning("Google Gemini SDK not available")

import aiohttp
import sqlite3

logger = logging.getLogger(__name__)

class PrescriptionNLPService:
    """
    NLP service for parsing medical prescriptions using:
    1. IBM Watson Natural Language Understanding (primary)
    2. HuggingFace medical NER models 
    3. Google Gemini AI (fallback)
    4. Custom rule-based extraction
    """
    
    def __init__(self):
        self.db_path = "data/prescription_cache.db"
        
        # Initialize IBM Watson
        self.watson_nlu = None
        self._init_watson()
        
        # Initialize HuggingFace models
        self.medical_ner = None
        self.tokenizer = None
        self._init_huggingface()
        
        # Initialize Gemini
        self.gemini_model = None
        self._init_gemini()
        
        # Initialize database
        self._init_database()
        
        # Medical entity patterns
        self._init_patterns()
    
    def _init_watson(self):
        """Initialize IBM Watson NLU"""
        if not WATSON_AVAILABLE:
            logger.warning("IBM Watson SDK not available")
            return
            
        try:
            api_key = os.getenv('IBM_WATSON_API_KEY')
            service_url = os.getenv('IBM_WATSON_URL', 'https://api.us-south.natural-language-understanding.watson.cloud.ibm.com/instances/your-instance-id')
            
            if api_key:
                authenticator = IAMAuthenticator(api_key)
                self.watson_nlu = NaturalLanguageUnderstandingV1(
                    version='2022-04-07',
                    authenticator=authenticator
                )
                self.watson_nlu.set_service_url(service_url)
                logger.info("IBM Watson NLU initialized successfully")
            else:
                logger.warning("IBM Watson API key not found")
                
        except Exception as e:
            logger.error(f"Failed to initialize IBM Watson: {str(e)}")
    
    def _init_huggingface(self):
        """Initialize HuggingFace medical NER model"""
        if not HF_AVAILABLE:
            logger.warning("HuggingFace transformers not available")
            return
            
        try:
            # Use a medical NER model - BioMistral or clinical BERT
            model_name = "Ishan0612/biobert-ner-disease-ncbi"  # Alternative: "BioMistral/BioMistral-7B"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            self.medical_ner = pipeline(
                "ner",
                model=model,
                tokenizer=self.tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            logger.info(f"HuggingFace medical NER model loaded: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize HuggingFace model: {str(e)}")
            
            # Fallback to a simpler model
            try:
                self.medical_ner = pipeline(
                    "ner",
                    model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple"
                )
                logger.info("Loaded fallback NER model")
            except Exception as e2:
                logger.error(f"Failed to load fallback NER model: {str(e2)}")
    
    def _init_gemini(self):
        """Initialize Google Gemini AI"""
        if not GEMINI_AVAILABLE:
            logger.warning("Google Gemini SDK not available")
            return
            
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if api_key:
                genai.configure(api_key=api_key)
                self.gemini_model = genai.GenerativeModel('gemini-pro')
                logger.info("Google Gemini initialized successfully")
            else:
                logger.warning("Gemini API key not found")
                
        except Exception as e:
            logger.error(f"Failed to initialize Gemini: {str(e)}")
    
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
                r'\b(?:tablet|capsule|mg|mcg|ml|units?)\s+(?:of\s+)?(\w+(?:\s+\w+)?)\b',
                r'\b(\w+(?:\s+\w+)?)\s+(?:\d+\s*(?:mg|mcg|ml|units?))',
                r'\b(aspirin|ibuprofen|acetaminophen|warfarin|lisinopril|metformin|atorvastatin|levothyroxine|amlodipine|metoprolol)\b'
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
                r'\b(?:morning|evening|bedtime|as needed|prn)\b'
            ],
            'routes': [
                r'\b(oral|orally|by mouth|sublingual|topical|intramuscular|intravenous|subcutaneous|rectal|vaginal|ophthalmic|otic|nasal)\b',
                r'\b(PO|IM|IV|SC|SQ|PR|PV|OU|OS|OD|NAS)\b'
            ],
            'indications': [
                r'\bfor\s+(?:the\s+)?(?:treatment\s+of\s+)?(\w+(?:\s+\w+)*)\b',
                r'\b(?:to\s+treat|treating)\s+(\w+(?:\s+\w+)*)\b',
                r'\b(?:indicated\s+for|used\s+for)\s+(\w+(?:\s+\w+)*)\b'
            ]
        }
    
    async def initialize(self):
        """Initialize the NLP service"""
        logger.info("Initializing Prescription NLP Service")
        
    async def cleanup(self):
        """Cleanup resources"""
        pass
    
    async def parse_prescription(self, text: str, language: str = "en") -> Dict[str, Any]:
        """
        Main method to parse prescription text using multiple NLP approaches
        """
        try:
            # Check cache first
            text_hash = self._generate_text_hash(text)
            cached_result = await self._get_cached_result(text_hash)
            
            if cached_result:
                return cached_result
            
            # Try different NLP approaches in order of preference
            result = None
            processing_method = "unknown"
            
            # 1. Try IBM Watson (primary)
            if self.watson_nlu:
                try:
                    result = await self._parse_with_watson(text)
                    processing_method = "watson"
                    logger.info("Successfully parsed with IBM Watson")
                except Exception as e:
                    logger.warning(f"Watson parsing failed: {str(e)}")
            
            # 2. Try HuggingFace medical NER (secondary)
            if not result and self.medical_ner:
                try:
                    result = await self._parse_with_huggingface(text)
                    processing_method = "huggingface"
                    logger.info("Successfully parsed with HuggingFace")
                except Exception as e:
                    logger.warning(f"HuggingFace parsing failed: {str(e)}")
            
            # 3. Try Gemini AI (fallback)
            if not result and self.gemini_model:
                try:
                    result = await self._parse_with_gemini(text)
                    processing_method = "gemini"
                    logger.info("Successfully parsed with Gemini")
                except Exception as e:
                    logger.warning(f"Gemini parsing failed: {str(e)}")
            
            # 4. Use rule-based extraction (last resort)
            if not result:
                result = await self._parse_with_rules(text)
                processing_method = "rules"
                logger.info("Using rule-based parsing")
            
            # Add processing metadata
            result['processing_method'] = processing_method
            result['parsed_at'] = datetime.utcnow().isoformat()
            
            # Cache the result
            await self._cache_result(text_hash, text, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error parsing prescription: {str(e)}")
            # Return basic rule-based parsing as fallback
            return await self._parse_with_rules(text)
    
    async def _parse_with_watson(self, text: str) -> Dict[str, Any]:
        """Parse prescription using IBM Watson NLU"""
        if not self.watson_nlu:
            raise Exception("Watson NLU not initialized")
        
        try:
            response = self.watson_nlu.analyze(
                text=text,
                features=Features(
                    entities=EntitiesOptions(
                        model='en-entities',  # Use custom medical model if available
                        limit=50
                    ),
                    keywords=KeywordsOptions(limit=50)
                ),
                language='en'
            ).get_result()
            
            # Process Watson response
            medications = []
            dosages = []
            frequencies = []
            routes = []
            
            # Extract entities
            entities = response.get('entities', [])
            keywords = response.get('keywords', [])
            
            # Process entities to identify medical concepts
            for entity in entities:
                entity_type = entity.get('type', '').lower()
                text_val = entity.get('text', '')
                confidence = entity.get('confidence', 0)
                
                # Map Watson entity types to medical concepts
                if entity_type in ['organization', 'other'] and self._is_medication(text_val):
                    medications.append({
                        'name': text_val,
                        'confidence': confidence,
                        'source': 'watson_entities'
                    })
            
            # Process keywords for additional medical terms
            for keyword in keywords:
                text_val = keyword.get('text', '')
                relevance = keyword.get('relevance', 0)
                
                if self._is_medication(text_val):
                    medications.append({
                        'name': text_val,
                        'confidence': relevance,
                        'source': 'watson_keywords'
                    })
            
            # Use rule-based extraction to supplement Watson results
            rule_based = await self._parse_with_rules(text)
            
            # Combine and deduplicate results
            all_medications = medications + rule_based['medications']
            unique_medications = self._deduplicate_entities(all_medications)
            
            return {
                'medications': unique_medications,
                'dosages': rule_based['dosages'],
                'frequencies': rule_based['frequencies'],
                'routes': rule_based['routes'],
                'confidence_score': self._calculate_confidence(unique_medications),
                'warnings': self._generate_warnings(unique_medications),
                'watson_entities': entities,
                'watson_keywords': keywords
            }
            
        except Exception as e:
            logger.error(f"Watson NLU error: {str(e)}")
            raise
    
    async def _parse_with_huggingface(self, text: str) -> Dict[str, Any]:
        """Parse prescription using HuggingFace medical NER"""
        if not self.medical_ner:
            raise Exception("HuggingFace model not initialized")
        
        try:
            # Run NER on the text
            entities = self.medical_ner(text)
            
            medications = []
            symptoms = []
            conditions = []
            
            # Process HuggingFace NER results
            for entity in entities:
                entity_type = entity.get('entity_group', '').upper()
                text_val = entity.get('word', '').replace('##', '')  # Clean BERT tokens
                confidence = entity.get('score', 0)
                
                # Map entity types to medical concepts
                if entity_type in ['DRUG', 'MEDICATION', 'CHEMICAL']:
                    medications.append({
                        'name': text_val,
                        'confidence': confidence,
                        'source': 'huggingface_ner'
                    })
                elif entity_type in ['DISEASE', 'SYMPTOM']:
                    conditions.append({
                        'name': text_val,
                        'confidence': confidence,
                        'type': entity_type.lower()
                    })
            
            # Use rule-based extraction to supplement HF results
            rule_based = await self._parse_with_rules(text)
            
            # Combine results
            all_medications = medications + rule_based['medications']
            unique_medications = self._deduplicate_entities(all_medications)
            
            return {
                'medications': unique_medications,
                'dosages': rule_based['dosages'],
                'frequencies': rule_based['frequencies'],
                'routes': rule_based['routes'],
                'conditions': conditions,
                'confidence_score': self._calculate_confidence(unique_medications),
                'warnings': self._generate_warnings(unique_medications),
                'hf_entities': entities
            }
            
        except Exception as e:
            logger.error(f"HuggingFace NER error: {str(e)}")
            raise
    
    async def _parse_with_gemini(self, text: str) -> Dict[str, Any]:
        """Parse prescription using Google Gemini AI"""
        if not self.gemini_model:
            raise Exception("Gemini model not initialized")
        
        try:
            prompt = f"""
            Analyze the following medical prescription text and extract structured information. 
            Return the results in JSON format with the following structure:
            {{
                "medications": [
                    {{"name": "medication_name", "confidence": 0.0-1.0}}
                ],
                "dosages": [
                    {{"amount": "amount", "unit": "unit", "medication": "medication_name"}}
                ],
                "frequencies": ["frequency_descriptions"],
                "routes": ["administration_routes"],
                "indications": ["medical_conditions_being_treated"]
            }}

            Prescription text: "{text}"
            
            Extract only the medical information present in the text. Do not make assumptions.
            """
            
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
                    
                    gemini_result['confidence_score'] = self._calculate_confidence(
                        gemini_result.get('medications', [])
                    )
                    gemini_result['warnings'] = self._generate_warnings(
                        gemini_result.get('medications', [])
                    )
                    
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
        loop = asyncio.get_event_loop()
        
        def _call_gemini():
            response = self.gemini_model.generate_content(prompt)
            return response.text
        
        return await loop.run_in_executor(None, _call_gemini)
    
    async def _parse_with_rules(self, text: str) -> Dict[str, Any]:
        """Parse prescription using rule-based regex patterns"""
        result = {
            'medications': [],
            'dosages': [],
            'frequencies': [],
            'routes': [],
            'indications': []
        }
        
        try:
            text_lower = text.lower()
            
            # Extract medications
            for pattern in self.patterns['medications']:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    med_name = match.group(1).strip()
                    if len(med_name) > 2:  # Filter out very short matches
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
    
    def _is_medication(self, text: str) -> bool:
        """Check if text is likely a medication name"""
        # Common medication suffixes and patterns
        med_patterns = [
            r'.*(?:cillin|mycin|pril|sartan|statin|zole|pine|ide|ine|ol)$',
            r'.*(?:mg|mcg|ml|units?).*',
            r'^(?:aspirin|ibuprofen|acetaminophen|warfarin|lisinopril|metformin)$'
        ]
        
        text_lower = text.lower()
        return any(re.match(pattern, text_lower) for pattern in med_patterns)
    
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
        if self.watson_nlu:
            try:
                watson_entities = await self._extract_entities_watson(text)
                entities.extend(watson_entities)
            except Exception as e:
                logger.warning(f"Watson entity extraction failed: {str(e)}")
        
        if self.medical_ner:
            try:
                hf_entities = await self._extract_entities_huggingface(text)
                entities.extend(hf_entities)
            except Exception as e:
                logger.warning(f"HuggingFace entity extraction failed: {str(e)}")
        
        # Rule-based extraction
        rule_entities = await self._extract_entities_rules(text)
        entities.extend(rule_entities)
        
        return self._deduplicate_entities(entities)
    
    async def _extract_entities_watson(self, text: str) -> List[Dict]:
        """Extract entities using Watson NLU"""
        entities = []
        
        try:
            response = self.watson_nlu.analyze(
                text=text,
                features=Features(
                    entities=EntitiesOptions(limit=50)
                )
            ).get_result()
            
            for entity in response.get('entities', []):
                entities.append({
                    'name': entity.get('text', ''),
                    'type': entity.get('type', ''),
                    'confidence': entity.get('confidence', 0),
                    'source': 'watson'
                })
                
        except Exception as e:
            logger.error(f"Watson entity extraction error: {str(e)}")
        
        return entities
    
    async def _extract_entities_huggingface(self, text: str) -> List[Dict]:
        """Extract entities using HuggingFace NER"""
        entities = []
        
        try:
            ner_results = self.medical_ner(text)
            
            for entity in ner_results:
                entities.append({
                    'name': entity.get('word', '').replace('##', ''),
                    'type': entity.get('entity_group', ''),
                    'confidence': entity.get('score', 0),
                    'source': 'huggingface'
                })
                
        except Exception as e:
            logger.error(f"HuggingFace entity extraction error: {str(e)}")
        
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