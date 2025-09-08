# Configuration and Requirements Files

## backend/requirements.txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.4.2
sqlalchemy==2.0.23
sqlite3
python-multipart==0.0.6
requests==2.31.0
aiohttp==3.9.1
pandas==1.5.3
numpy==1.24.3

# Google Gemini
google-generativeai==0.3.1

# Ollama
ollama==0.1.7

# Async support
asyncio
aiofiles==23.2.1

# Database
alembic==1.12.1

# Logging and monitoring
python-json-logger==2.0.7
structlog==23.2.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
httpx==0.25.2

## frontend/requirements.txt
streamlit==1.28.1
pandas==1.5.3
numpy==1.24.3
requests==2.31.0
plotly==5.17.0
altair==5.1.2

# File handling
python-docx==0.8.11
PyPDF2==3.0.1
Pillow==10.1.0

# Additional Streamlit components
streamlit-aggrid==0.3.4
streamlit-option-menu==0.3.6
streamlit-authenticator==0.2.3

## .env.example


# Google Gemini Configuration  
GEMINI_API_KEY=your_gemini_api_key_here

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434

# Database Configuration
DATABASE_URL=sqlite:///./app/data/medical_ai.db

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=True

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Cache Configuration
CACHE_DURATION_HOURS=24
ENABLE_CACHING=True

# Security
SECRET_KEY=your-super-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

## docker-compose.yml
version: '3.8'

services:
  api:
    build: 
      context: ./backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./backend/app/data:/app/data
      - ./backend/app/logs:/app/logs
    environment:
      - DATABASE_URL=sqlite:///./data/medical_ai.db
      
      - GEMINI_API_KEY=${GEMINI_API_KEY}
      - OLLAMA_HOST=${OLLAMA_HOST}
      - LOG_LEVEL=INFO
    networks:
      - medical-ai-network
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "8501:8501"
    depends_on:
      - api
    environment:
      - API_BASE_URL=http://api:8000/api
    networks:
      - medical-ai-network
    restart: unless-stopped
    volumes:
      - ./frontend/data:/app/data

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_DB=medical_ai
      - POSTGRES_USER=medical_user
      - POSTGRES_PASSWORD=medical_password
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./backend/init.sql:/docker-entrypoint-initdb.d/init.sql
    ports:
      - "5432:5432"
    networks:
      - medical-ai-network
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    networks:
      - medical-ai-network
    restart: unless-stopped
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data

networks:
  medical-ai-network:
    driver: bridge

volumes:
  postgres_data:
  redis_data:

## backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create necessary directories
RUN mkdir -p /app/data /app/logs

# Set environment variables
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

## frontend/Dockerfile  
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory
RUN mkdir -p /app/data

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Run the application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]

## Setup and Installation Script
# setup.py
#!/usr/bin/env python3

import os
import subprocess
import sys
import json
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return None

def setup_environment():
    """Set up the development environment"""
    
    print("🏥 AI Medical Prescription Verification System Setup")
    print("=" * 60)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Create directory structure
    directories = [
        "backend/app/data",
        "backend/app/logs", 
        "frontend/data",
        "data/datasets",
        "docs",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {directory}")
    
    # Install backend dependencies
    print("\n🔧 Setting up backend...")
    os.chdir("backend")
    
    if run_command("python -m venv venv", "Creating virtual environment"):
        # Activate virtual environment and install dependencies
        if os.name == 'nt':  # Windows
            activate_cmd = "venv\\Scripts\\activate && pip install -r requirements.txt"
        else:  # Unix/Linux/MacOS
            activate_cmd = "source venv/bin/activate && pip install -r requirements.txt"
        
        run_command(activate_cmd, "Installing backend dependencies")
    
    os.chdir("..")
    
    # Install frontend dependencies
    print("\n🎨 Setting up frontend...")
    os.chdir("frontend")
    
    if run_command("python -m venv venv", "Creating virtual environment"):
        if os.name == 'nt':
            activate_cmd = "venv\\Scripts\\activate && pip install -r requirements.txt"
        else:
            activate_cmd = "source venv/bin/activate && pip install -r requirements.txt"
        
        run_command(activate_cmd, "Installing frontend dependencies")
    
    os.chdir("..")
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        print("\n📝 Creating .env file...")
        with open(".env.example", "r") as example_file:
            env_content = example_file.read()
        
        with open(".env", "w") as env_file:
            env_file.write(env_content)
        
        print("✅ .env file created. Please update with your API keys.")
    
    # Initialize database
    print("\n🗄️ Initializing database...")
    run_command("python data/scripts/setup_database.py", "Setting up database")
    
    # Download sample datasets
    print("\n📊 Downloading sample datasets...")
    run_command("python data/scripts/download_datasets.py", "Downloading datasets")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📝 Next steps:")
    print("1. Update .env file with your API keys (IBM Watson, Gemini)")
    print("2. Install Ollama and pull the granite3.2-vision model: ollama pull granite3.2-vision")
    print("3. Start the backend: cd backend && uvicorn app.main:app --reload")
    print("4. Start the frontend: cd frontend && streamlit run streamlit_app.py")
    print("5. Open http://localhost:8501 in your browser")
    
    return True

if __name__ == "__main__":
    setup_environment()

## Data Download Script
# data/scripts/download_datasets.py
#!/usr/bin/env python3

import requests
import zipfile
import pandas as pd
import json
import os
from pathlib import Path
import sqlite3
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_file(url, filename):
    """Download a file from URL"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        logger.info(f"Downloaded: {filename}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {str(e)}")
        return False

def setup_sample_data():
    """Set up sample medical data"""
    
    # Create data directories
    data_dir = Path("data/datasets")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample drug interaction data
    sample_interactions = [
        {
            "drug1": "warfarin", "drug2": "aspirin", "severity": "major",
            "description": "Increased risk of bleeding", 
            "mechanism": "Additive anticoagulant effects",
            "management": "Monitor INR closely, consider gastroprotection"
        },
        {
            "drug1": "lisinopril", "drug2": "ibuprofen", "severity": "moderate",
            "description": "Reduced antihypertensive effect",
            "mechanism": "NSAID-induced sodium retention", 
            "management": "Monitor blood pressure"
        },
        # Add more sample interactions...
    ]
    
    # Save to JSON
    with open(data_dir / "sample_interactions.json", "w") as f:
        json.dump(sample_interactions, f, indent=2)
    
    # Sample drug information
    sample_drugs = [
        {
            "name": "Aspirin",
            "generic_name": "acetylsalicylic acid",
            "drug_class": "NSAID",
            "indications": ["pain", "fever", "inflammation", "cardiovascular protection"],
            "contraindications": ["bleeding disorders", "peptic ulcer"],
            "typical_dose": "81-325mg daily"
        },
        {
            "name": "Warfarin", 
            "generic_name": "warfarin sodium",
            "drug_class": "Anticoagulant",
            "indications": ["atrial fibrillation", "DVT", "PE"],
            "contraindications": ["active bleeding", "pregnancy"],
            "typical_dose": "2-10mg daily (individualized)"
        }
        # Add more drugs...
    ]
    
    with open(data_dir / "sample_drugs.json", "w") as f:
        json.dump(sample_drugs, f, indent=2)
    
    logger.info("Sample data created successfully")

def download_public_datasets():
    """Download publicly available medical datasets"""
    
    datasets = {
        "RxNorm": {
            "url": "https://download.nlm.nih.gov/umls/kss/rxnorm/RxNorm_full_current.zip",
            "description": "RxNorm drug database"
        }
        # Add more public datasets as needed
    }
    
    # Note: In practice, you would implement proper dataset downloads
    # For this demo, we'll use sample data
    logger.info("Using sample data for demonstration")

if __name__ == "__main__":
    setup_sample_data()
    download_public_datasets()
    logger.info("Dataset setup completed")

## README.md
# 🏥 AI Medical Prescription Verification System

A comprehensive AI-powered system for medical prescription verification, drug interaction detection, age-specific dosage recommendations, and NLP-based prescription parsing.

## 🚀 Features

- **Drug Interaction Detection**: Check for dangerous drug-drug interactions using multiple medical databases
- **Age-Specific Dosage**: Calculate appropriate dosages based on patient age, weight, and medical conditions  
- **NLP Prescription Parser**: Extract structured information from prescription text using Ollama (granite3.2-vision model) and Google Gemini AI
- **Alternative Medication Finder**: Suggest safer alternatives when contraindications exist
- **Interactive Dashboard**: User-friendly Streamlit interface for healthcare professionals

## 🛠️ Technology Stack

### Backend (FastAPI)
- **FastAPI**: High-performance API framework
- **SQLite/PostgreSQL**: Database for caching and storage
- **Ollama**: Primary NLP processing (using granite3.2-vision model)
- **Google Gemini AI**: Fallback AI processing
- **RxNorm API**: Drug terminology and interactions
- **OpenFDA**: Adverse event data

### Frontend (Streamlit)
- **Streamlit**: Interactive web interface
- **Plotly**: Data visualizations
- **Pandas**: Data manipulation
- **Requests**: API communication

## 📋 Prerequisites

- Python 3.8+
- Google Gemini API key (optional)
- Docker (for containerized deployment)
- Ollama with granite3.2-vision model

## 🔧 Installation

### Quick Setup
```bash
# Clone the repository
git clone https://github.com/your-org/ai-medical-prescription-verification.git
cd ai-medical-prescription-verification

# Run setup script
python setup.py
```

### Manual Setup

1. **Backend Setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Frontend Setup**  
```bash
cd frontend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Environment Configuration**
```bash
cp .env.example .env
# Edit .env with your API keys
```

4. **Ollama Setup**
```bash
# Install Ollama from https://ollama.com/
# Pull the granite3.2-vision model
ollama pull granite3.2-vision
```

## 🚀 Running the Application

### Development Mode

1. **Start Backend**
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Start Frontend**
```bash
cd frontend  
streamlit run streamlit_app.py --server.port 8501
```

3. **Access Application**
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

### Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Or run individual services
docker-compose up api
docker-compose up frontend
```

## 📖 API Documentation

The system provides RESTful APIs for all functionality:

### Drug Interactions
- `POST /api/check-interactions` - Check drug interactions
- `GET /api/interaction-details/{drug1}/{drug2}` - Get interaction details

### Dosage Calculations  
- `POST /api/age-dosage` - Calculate age-specific dosage
- `GET /api/dosage-guidelines/{drug_name}` - Get dosage guidelines

### Prescription Parsing
- `POST /api/parse-prescription` - Parse prescription text
- `POST /api/extract-entities` - Extract medical entities

### Alternative Medications
- `POST /api/alternative-drugs` - Find alternative medications
- `GET /api/drug-classes/{drug_name}` - Get drug therapeutic classes

## 🔐 Security & Compliance

- **HIPAA Considerations**: The system is designed with healthcare compliance in mind
- **Data Encryption**: All sensitive data is encrypted at rest and in transit
- **Access Controls**: API authentication and authorization
- **Audit Logging**: Comprehensive logging for compliance tracking

⚠️ **Important**: This system is for educational/research purposes. Always consult healthcare professionals for medical decisions.

## 🧪 Testing

```bash
# Run backend tests
cd backend
pytest tests/

# Run frontend tests  
cd frontend
pytest tests/
```

## 📊 Monitoring

The system includes built-in monitoring and analytics:
- API usage metrics
- Error tracking and logging
- Performance monitoring
- User interaction analytics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Ollama for local AI processing with granite3.2-vision model
- Google for Gemini AI
- OpenFDA for adverse event data
- RxNorm for drug terminology

## 📞 Support

For support and questions:
- 📧 Email: support@medical-ai.com  
- 📖 Documentation: [docs/](docs/)
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/ai-medical-prescription-verification/issues)

---

**Disclaimer**: This system is intended for educational and research purposes only. It should not be used as the sole basis for medical decisions. Always consult qualified healthcare professionals for medical advice.