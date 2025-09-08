# Let's create the complete project structure and core files
import os
import json

# Define the project structure
project_structure = {
    "AI-Medical-Prescription-Verification/": {
        "backend/": {
            "app/": {
                "__init__.py": "",
                "main.py": "FastAPI main application",
                "models/": {
                    "__init__.py": "",
                    "database.py": "Database models",
                    "schemas.py": "Pydantic schemas"
                },
                "services/": {
                    "__init__.py": "",
                    "drug_service.py": "Drug interaction service",
                    "nlp_service.py": "NLP processing service",
                    "dosage_service.py": "Age-specific dosage service",
                    "alternative_service.py": "Alternative drug service"
                },
                "utils/": {
                    "__init__.py": "",
                    "config.py": "Configuration settings",
                    "database_utils.py": "Database utilities"
                },
                "data/": {
                    "drug_interactions.db": "SQLite database",
                    "datasets/": "Raw datasets"
                }
            },
            "requirements.txt": "Backend dependencies",
            "Dockerfile": "Backend container"
        },
        "frontend/": {
            "streamlit_app.py": "Main Streamlit application",
            "pages/": {
                "1_Drug_Interactions.py": "Drug interaction checker",
                "2_Age_Dosage.py": "Age-specific dosage",
                "3_Prescription_Parser.py": "NLP prescription parser",
                "4_Alternative_Drugs.py": "Alternative medications"
            },
            "utils/": {
                "api_client.py": "API client utilities",
                "ui_components.py": "Reusable UI components"
            },
            "requirements.txt": "Frontend dependencies"
        },
        "data/": {
            "scripts/": {
                "download_datasets.py": "Dataset download script",
                "process_faers.py": "FAERS data processor",
                "load_drug_data.py": "Drug data loader"
            }
        },
        "docker-compose.yml": "Docker compose configuration",
        "README.md": "Project documentation",
        ".env.example": "Environment variables template",
        "setup.py": "Project setup script"
    }
}

# Create a visual representation of the project structure
def print_tree(structure, prefix="", is_last=True):
    items = list(structure.items())
    for i, (name, content) in enumerate(items):
        is_last_item = i == len(items) - 1
        current_prefix = "└── " if is_last_item else "├── "
        print(f"{prefix}{current_prefix}{name}")
        
        if isinstance(content, dict):
            extension = "    " if is_last_item else "│   "
            print_tree(content, prefix + extension, is_last_item)
        elif content:
            # Show description for files
            description_prefix = "    " if is_last_item else "│   "
            print(f"{prefix}{description_prefix}    # {content}")

print("AI Medical Prescription Verification System - Project Structure")
print("=" * 70)
print_tree(project_structure)