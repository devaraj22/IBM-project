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
- **Ollama**: Primary NLP processing (using granite3.3:2b model)
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
- Docker (for containerized deployment)
- Ollama (for local AI processing) with granite3.3:2b model

## 🔧 Installation

### Quick Setup with Docker (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd ibm-project

# Build and run with Docker Compose
docker-compose up --build
```

### Manual Setup

1. **Backend Setup**
```bash
cd backend
python -m venv venv
# On Windows: venv\Scripts\activate
# On Unix/MacOS: source venv/bin/activate
pip install -r requirements.txt
```

2. **Frontend Setup**  
```bash
cd frontend
python -m venv venv
# On Windows: venv\Scripts\activate
# On Unix/MacOS: source venv/bin/activate
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
# Pull the granite3.3:2b model
ollama pull granite3.3:2b
```

## 🚀 Running the Application

### With Docker (Recommended)

```bash
# Build and run with Docker Compose
docker-compose up --build

# Access the application:
# Frontend: http://localhost:8501
# Backend API: http://localhost:8000
# API Documentation: http://localhost:8000/docs
```

### Development Mode

1. **Start Backend**
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Start Frontend**
```bash
cd frontend  
streamlit run streamlit-frontend.py --server.port 8501
```

3. **Access Application**
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

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

### Unit Tests
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


## 🙏 Acknowledgments

- Ollama for local AI processing with granite3.3: model
- Google for Gemini AI
- OpenFDA for adverse event data
- RxNorm for drug terminology


For support and questions:
- 🐛 Issues: [GitHub Issues](https://github.com/your-org/ai-medical-prescription-verification/issues)

---

**Disclaimer**: This system is intended for educational and research purposes only. It should not be used as the sole basis for medical decisions. Always consult qualified healthcare professionals for medical advice.
