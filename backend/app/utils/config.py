# backend/app/utils/config.py
import os
from typing import Optional
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings and configuration"""
    
    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True
    
    # Database Configuration
    database_url: str = "sqlite:///./app/data/medical_ai.db"
    
    # IBM Watson Configuration
    ibm_watson_api_key: Optional[str] = None
    ibm_watson_url: Optional[str] = None
    
    # Google Gemini Configuration
    gemini_api_key: Optional[str] = None
    
    # Cache Configuration
    cache_duration_hours: int = 24
    enable_caching: bool = True
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Security Configuration
    secret_key: str = "your-super-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # External API Configuration
    rxnorm_api_base: str = "https://rxnav.nlm.nih.gov/REST"
    openfda_api_base: str = "https://api.fda.gov"
    
    # Rate Limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 3600  # 1 hour
    
    # File Upload Configuration
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_file_types: list = [".txt", ".pdf", ".docx", ".jpg", ".png"]
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# backend/app/utils/database_utils.py
import sqlite3
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import os
from pathlib import Path

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Utility class for database operations"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.ensure_directory_exists()
    
    def ensure_directory_exists(self):
        """Ensure the database directory exists"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def get_connection(self):
        """Get database connection with proper configuration"""
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        conn.execute("PRAGMA foreign_keys = ON")  # Enable foreign keys
        conn.execute("PRAGMA journal_mode = WAL")  # Enable WAL mode
        return conn
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict]:
        """Execute a SELECT query and return results as list of dicts"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Database query error: {str(e)}")
            raise
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """Execute an INSERT/UPDATE/DELETE query and return affected rows"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute(query, params)
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Database update error: {str(e)}")
            raise
    
    def bulk_insert(self, table: str, data: List[Dict]) -> int:
        """Bulk insert data into a table"""
        if not data:
            return 0
        
        try:
            columns = list(data[0].keys())
            placeholders = ", ".join(["?" for _ in columns])
            query = f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
            
            with self.get_connection() as conn:
                cursor = conn.executemany(query, [tuple(row[col] for col in columns) for row in data])
                conn.commit()
                return cursor.rowcount
        except Exception as e:
            logger.error(f"Bulk insert error: {str(e)}")
            raise

def setup_medical_database():
    """Set up the main medical database with all required tables"""
    
    db_path = "app/data/medical_ai.db"
    db_manager = DatabaseManager(db_path)
    
    # Medical database schema
    schema_queries = [
        """
        CREATE TABLE IF NOT EXISTS drug_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            drug1 TEXT NOT NULL,
            drug2 TEXT NOT NULL,
            severity TEXT NOT NULL CHECK(severity IN ('minor', 'moderate', 'major')),
            description TEXT,
            mechanism TEXT,
            management TEXT,
            evidence_level TEXT,
            source TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(drug1, drug2, source)
        )
        """,
        
        """
        CREATE TABLE IF NOT EXISTS drug_information (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            drug_name TEXT UNIQUE NOT NULL,
            generic_name TEXT,
            brand_names TEXT,
            drug_class TEXT,
            therapeutic_class TEXT,
            mechanism TEXT,
            indications TEXT,
            contraindications TEXT,
            side_effects TEXT,
            interactions_count INTEGER DEFAULT 0,
            rxcui TEXT,
            ndc_codes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        
        """
        CREATE TABLE IF NOT EXISTS dosage_guidelines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            drug_name TEXT NOT NULL,
            age_group TEXT NOT NULL,
            min_age INTEGER,
            max_age INTEGER,
            weight_min REAL,
            weight_max REAL,
            base_dose REAL NOT NULL,
            max_dose REAL,
            unit TEXT NOT NULL,
            frequency TEXT,
            route TEXT,
            indication TEXT,
            weight_based BOOLEAN DEFAULT 0,
            renal_adjustment BOOLEAN DEFAULT 0,
            hepatic_adjustment BOOLEAN DEFAULT 0,
            special_populations TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        
        """
        CREATE TABLE IF NOT EXISTS prescription_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text_hash TEXT UNIQUE NOT NULL,
            original_text TEXT NOT NULL,
            parsed_data TEXT NOT NULL,
            processing_method TEXT,
            confidence_score REAL,
            medications_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        )
        """,
        
        """
        CREATE TABLE IF NOT EXISTS api_usage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            endpoint TEXT NOT NULL,
            method TEXT NOT NULL,
            request_data TEXT,
            response_status INTEGER,
            response_time_ms INTEGER,
            user_id TEXT,
            ip_address TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """,
        
        """
        CREATE TABLE IF NOT EXISTS alternative_medications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            original_drug TEXT NOT NULL,
            alternative_drug TEXT NOT NULL,
            therapeutic_class TEXT,
            similarity_score REAL,
            safety_profile TEXT,
            cost_comparison TEXT,
            availability TEXT,
            advantages TEXT,
            disadvantages TEXT,
            contraindications TEXT,
            age_restrictions TEXT,
            pregnancy_category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    ]
    
    # Create indexes
    index_queries = [
        "CREATE INDEX IF NOT EXISTS idx_drug_interactions_drugs ON drug_interactions(drug1, drug2)",
        "CREATE INDEX IF NOT EXISTS idx_drug_interactions_severity ON drug_interactions(severity)",
        "CREATE INDEX IF NOT EXISTS idx_drug_info_name ON drug_information(drug_name)",
        "CREATE INDEX IF NOT EXISTS idx_drug_info_class ON drug_information(drug_class)",
        "CREATE INDEX IF NOT EXISTS idx_dosage_drug_age ON dosage_guidelines(drug_name, age_group)",
        "CREATE INDEX IF NOT EXISTS idx_prescription_hash ON prescription_cache(text_hash)",
        "CREATE INDEX IF NOT EXISTS idx_api_logs_endpoint ON api_usage_logs(endpoint, created_at)",
        "CREATE INDEX IF NOT EXISTS idx_alternatives_original ON alternative_medications(original_drug)"
    ]
    
    try:
        # Execute schema creation
        for query in schema_queries:
            db_manager.execute_update(query)
        
        # Create indexes
        for query in index_queries:
            db_manager.execute_update(query)
        
        logger.info("Medical database setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database setup failed: {str(e)}")
        return False

def load_initial_data():
    """Load initial sample data into the database"""
    
    db_path = "app/data/medical_ai.db"
    db_manager = DatabaseManager(db_path)
    
    # Sample drug interactions
    interactions_data = [
        {
            "drug1": "warfarin", "drug2": "aspirin", "severity": "major",
            "description": "Increased risk of bleeding when warfarin and aspirin are used together",
            "mechanism": "Both drugs affect hemostasis through different mechanisms",
            "management": "Monitor INR closely, consider gastroprotection, evaluate bleeding risk",
            "evidence_level": "level_1", "source": "clinical_guidelines"
        },
        {
            "drug1": "lisinopril", "drug2": "ibuprofen", "severity": "moderate", 
            "description": "NSAIDs may reduce the antihypertensive effect of ACE inhibitors",
            "mechanism": "NSAIDs reduce renal prostaglandin synthesis affecting blood pressure control",
            "management": "Monitor blood pressure, consider alternative analgesic",
            "evidence_level": "level_1", "source": "clinical_guidelines"
        },
        {
            "drug1": "metformin", "drug2": "furosemide", "severity": "minor",
            "description": "Loop diuretics may affect glucose control in diabetic patients", 
            "mechanism": "Diuretics can cause hyperglycemia and hypokalemia",
            "management": "Monitor blood glucose and electrolytes",
            "evidence_level": "level_2", "source": "clinical_guidelines"
        }
    ]
    
    # Sample drug information
    drugs_data = [
        {
            "drug_name": "aspirin", "generic_name": "acetylsalicylic acid",
            "brand_names": "Bayer|Bufferin|Ecotrin", "drug_class": "NSAID",
            "therapeutic_class": "Antiplatelet|Analgesic|Anti-inflammatory",
            "mechanism": "Irreversible COX-1 and COX-2 inhibition",
            "indications": "Pain|Fever|Inflammation|Cardiovascular protection",
            "contraindications": "Active bleeding|Peptic ulcer|Severe liver disease",
            "side_effects": "GI bleeding|Tinnitus|Reye syndrome in children"
        },
        {
            "drug_name": "warfarin", "generic_name": "warfarin sodium",
            "brand_names": "Coumadin|Jantoven", "drug_class": "Anticoagulant",
            "therapeutic_class": "Vitamin K antagonist",
            "mechanism": "Inhibits vitamin K-dependent clotting factors",
            "indications": "Atrial fibrillation|DVT|PE|Mechanical heart valves",
            "contraindications": "Active bleeding|Pregnancy|Severe liver disease",
            "side_effects": "Bleeding|Skin necrosis|Purple toe syndrome"
        },
        {
            "drug_name": "lisinopril", "generic_name": "lisinopril",
            "brand_names": "Prinivil|Zestril", "drug_class": "ACE inhibitor",
            "therapeutic_class": "Antihypertensive",
            "mechanism": "Inhibits angiotensin-converting enzyme",
            "indications": "Hypertension|Heart failure|Post-MI",
            "contraindications": "Pregnancy|Bilateral renal artery stenosis|Angioedema",
            "side_effects": "Dry cough|Hyperkalemia|Angioedema"
        }
    ]
    
    # Sample dosage guidelines
    dosage_data = [
        {
            "drug_name": "acetaminophen", "age_group": "pediatric", "min_age": 0, "max_age": 17,
            "base_dose": 10.0, "max_dose": 15.0, "unit": "mg/kg/dose",
            "frequency": "every 4-6 hours", "route": "oral", "weight_based": 1
        },
        {
            "drug_name": "acetaminophen", "age_group": "adult", "min_age": 18, "max_age": 64,
            "base_dose": 325.0, "max_dose": 1000.0, "unit": "mg",
            "frequency": "every 4-6 hours", "route": "oral", "weight_based": 0
        },
        {
            "drug_name": "ibuprofen", "age_group": "pediatric", "min_age": 6, "max_age": 17,
            "base_dose": 5.0, "max_dose": 10.0, "unit": "mg/kg/dose",
            "frequency": "every 6-8 hours", "route": "oral", "weight_based": 1
        }
    ]
    
    try:
        # Load data into tables
        db_manager.bulk_insert("drug_interactions", interactions_data)
        db_manager.bulk_insert("drug_information", drugs_data)
        db_manager.bulk_insert("dosage_guidelines", dosage_data)
        
        logger.info("Initial data loaded successfully")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load initial data: {str(e)}")
        return False

# Deployment and Testing Scripts
def create_deployment_script():
    """Create deployment automation script"""
    
    deployment_script = '''#!/bin/bash
# deployment/deploy.sh
# AI Medical Prescription Verification System Deployment Script

set -e

echo "🏥 Starting AI Medical Prescription Verification System Deployment"

# Check prerequisites
echo "📋 Checking prerequisites..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker is required but not installed"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is required but not installed"
    exit 1
fi

# Check environment variables
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please copy .env.example to .env and configure"
    exit 1
fi

# Load environment variables
source .env

if [ -z "$IBM_WATSON_API_KEY" ]; then
    echo "⚠️  Warning: IBM_WATSON_API_KEY not set - Watson NLP will not work"
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  Warning: GEMINI_API_KEY not set - Gemini fallback will not work"
fi

# Build and start services
echo "🔨 Building Docker images..."
docker-compose build --no-cache

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Health checks
echo "🔍 Performing health checks..."

# Check API health
API_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health || echo "000")
if [ "$API_HEALTH" = "200" ]; then
    echo "✅ Backend API is healthy"
else
    echo "❌ Backend API health check failed (HTTP $API_HEALTH)"
    docker-compose logs api
    exit 1
fi

# Check frontend
FRONTEND_HEALTH=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501 || echo "000")
if [ "$FRONTEND_HEALTH" = "200" ]; then
    echo "✅ Frontend is healthy"
else
    echo "❌ Frontend health check failed (HTTP $FRONTEND_HEALTH)"
    docker-compose logs frontend
    exit 1
fi

echo "🎉 Deployment completed successfully!"
echo ""
echo "📊 Service URLs:"
echo "Frontend: http://localhost:8501"
echo "Backend API: http://localhost:8000"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "🔧 Management commands:"
echo "View logs: docker-compose logs -f"
echo "Stop services: docker-compose down"
echo "Restart: docker-compose restart"

# Show running containers
echo ""
echo "📦 Running containers:"
docker-compose ps
'''
    
    # Write deployment script
    os.makedirs("deployment", exist_ok=True)
    with open("deployment/deploy.sh", "w") as f:
        f.write(deployment_script)
    
    # Make executable
    os.chmod("deployment/deploy.sh", 0o755)
    
    print("✅ Deployment script created: deployment/deploy.sh")

def create_test_suite():
    """Create comprehensive test suite"""
    
    test_script = '''# tests/test_drug_interactions.py
import pytest
import asyncio
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.services.drug_service import DrugInteractionService
from app.services.nlp_service import PrescriptionNLPService
from app.services.dosage_service import DosageService

@pytest.fixture
def drug_service():
    """Create drug service instance for testing"""
    service = DrugInteractionService()
    return service

@pytest.fixture
def nlp_service():
    """Create NLP service instance for testing"""
    service = PrescriptionNLPService()
    return service

@pytest.fixture
def dosage_service():
    """Create dosage service instance for testing"""
    service = DosageService()
    return service

class TestDrugInteractions:
    """Test drug interaction detection"""
    
    @pytest.mark.asyncio
    async def test_warfarin_aspirin_interaction(self, drug_service):
        """Test major drug interaction detection"""
        drugs = ["warfarin", "aspirin"]
        interactions = await drug_service.check_interactions(drugs)
        
        assert len(interactions) > 0
        assert any(i.get('severity') == 'major' for i in interactions)
    
    @pytest.mark.asyncio
    async def test_no_interactions(self, drug_service):
        """Test case with no interactions"""
        drugs = ["acetaminophen"]  # Single drug should have no interactions
        interactions = await drug_service.check_interactions(drugs)
        
        assert len(interactions) == 0
    
    @pytest.mark.asyncio
    async def test_age_specific_warnings(self, drug_service):
        """Test age-specific interaction warnings"""
        drugs = ["warfarin", "aspirin"]
        interactions = await drug_service.check_interactions(drugs, patient_age=75)
        
        assert len(interactions) > 0
        # Should have age-specific warnings for elderly patient
        for interaction in interactions:
            age_warnings = interaction.get('age_warnings', [])
            assert any('elderly' in warning.lower() for warning in age_warnings)

class TestDosageCalculation:
    """Test dosage calculation functionality"""
    
    @pytest.mark.asyncio
    async def test_pediatric_dosage(self, dosage_service):
        """Test pediatric dosage calculation"""
        result = await dosage_service.calculate_dosage("acetaminophen", age=8, weight=25.0)
        
        assert result is not None
        assert result['age_group'] == 'child'
        assert result['weight_based'] == True
        assert result['recommended_dose'] > 0
    
    @pytest.mark.asyncio
    async def test_adult_dosage(self, dosage_service):
        """Test adult dosage calculation"""
        result = await dosage_service.calculate_dosage("acetaminophen", age=35, weight=70.0)
        
        assert result is not None
        assert result['age_group'] == 'adult'
        assert result['recommended_dose'] > 0
    
    @pytest.mark.asyncio
    async def test_renal_adjustment(self, dosage_service):
        """Test dosage adjustment for renal impairment"""
        result = await dosage_service.calculate_dosage(
            "acetaminophen", age=65, weight=70.0, kidney_function="moderate impairment"
        )
        
        assert result is not None
        assert 'adjustments' in result
        # Should have warnings for renal impairment

class TestNLPParsing:
    """Test prescription NLP parsing"""
    
    @pytest.mark.asyncio
    async def test_simple_prescription_parsing(self, nlp_service):
        """Test parsing of simple prescription"""
        prescription = "Take acetaminophen 500mg by mouth every 6 hours for pain"
        
        result = await nlp_service.parse_prescription(prescription)
        
        assert result is not None
        assert len(result.get('medications', [])) > 0
        assert result['medications'][0]['name'].lower() == 'acetaminophen'
    
    @pytest.mark.asyncio
    async def test_complex_prescription_parsing(self, nlp_service):
        """Test parsing of complex prescription"""
        prescription = """
        Rx: Amoxicillin 500mg capsules
        Sig: Take 1 capsule by mouth three times daily for 10 days
        Disp: 30 capsules
        For: Bacterial infection
        """
        
        result = await nlp_service.parse_prescription(prescription)
        
        assert result is not None
        assert len(result.get('medications', [])) > 0
        assert len(result.get('dosages', [])) > 0
        assert len(result.get('frequencies', [])) > 0

class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, drug_service, dosage_service, nlp_service):
        """Test complete workflow from prescription to recommendations"""
        
        # 1. Parse prescription
        prescription = "Take warfarin 5mg daily and aspirin 81mg daily"
        parsed = await nlp_service.parse_prescription(prescription)
        
        assert len(parsed.get('medications', [])) >= 2
        
        # 2. Extract drug names
        drug_names = [med['name'] for med in parsed['medications']]
        
        # 3. Check interactions
        interactions = await drug_service.check_interactions(drug_names)
        
        assert len(interactions) > 0
        
        # 4. Calculate dosages for elderly patient
        for drug in drug_names:
            dosage = await dosage_service.calculate_dosage(drug, age=75, weight=70.0)
            # Some drugs may not have dosage info in test database
            if dosage:
                assert dosage['age_group'] == 'elderly'

# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
'''
    
    # Create tests directory and file
    os.makedirs("tests", exist_ok=True)
    with open("tests/test_medical_ai.py", "w") as f:
        f.write(test_script)
    
    # Create pytest configuration
    pytest_config = '''[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
    unit: marks tests as unit tests
'''
    
    with open("pytest.ini", "w") as f:
        f.write(pytest_config)
    
    print("✅ Test suite created: tests/test_medical_ai.py")

if __name__ == "__main__":
    print("🔧 Setting up AI Medical Prescription Verification System")
    
    # Setup database
    if setup_medical_database():
        print("✅ Database setup completed")
        
        if load_initial_data():
            print("✅ Initial data loaded")
        else:
            print("⚠️  Warning: Failed to load initial data")
    else:
        print("❌ Database setup failed")
    
    # Create deployment script
    create_deployment_script()
    
    # Create test suite  
    create_test_suite()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📝 Next steps:")
    print("1. Configure .env file with your API keys")
    print("2. Run: ./deployment/deploy.sh")
    print("3. Test: python -m pytest tests/")
    print("4. Access: http://localhost:8501")