#!/usr/bin/env python3
"""
Local setup script for NBA Game Predictor
Sets up the application without Docker for testing
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def setup_backend():
    """Setup the FastAPI backend"""
    print("\n🚀 Setting up FastAPI Backend...")
    
    # Change to backend directory
    os.chdir('backend')
    
    # Install requirements
    if not run_command("pip install -r requirements.txt", "Installing backend dependencies"):
        print("⚠️  Some dependencies failed to install. Trying with --no-deps...")
        run_command("pip install fastapi uvicorn pandas scikit-learn xgboost", "Installing core dependencies")
    
    # Go back to root
    os.chdir('..')
    return True

def setup_frontend():
    """Setup the React frontend"""
    print("\n🎨 Setting up React Frontend...")
    
    # Check if Node.js is installed
    if not run_command("node --version", "Checking Node.js installation"):
        print("❌ Node.js is not installed. Please install Node.js 16+ from https://nodejs.org/")
        return False
    
    # Check if npm is installed
    if not run_command("npm --version", "Checking npm installation"):
        print("❌ npm is not installed. Please install npm")
        return False
    
    # Change to frontend directory
    os.chdir('frontend')
    
    # Install dependencies
    if not run_command("npm install", "Installing frontend dependencies"):
        return False
    
    # Go back to root
    os.chdir('..')
    return True

def create_start_scripts():
    """Create startup scripts for local development"""
    print("\n📝 Creating startup scripts...")
    
    # Create backend start script
    backend_script = """@echo off
echo 🚀 Starting NBA Game Predictor Backend...
cd backend
python main.py
pause
"""
    
    with open('start_backend.bat', 'w') as f:
        f.write(backend_script)
    
    # Create frontend start script
    frontend_script = """@echo off
echo 🎨 Starting NBA Game Predictor Frontend...
cd frontend
npm start
pause
"""
    
    with open('start_frontend.bat', 'w') as f:
        f.write(frontend_script)
    
    print("✅ Startup scripts created")
    return True

def main():
    """Main setup function"""
    print("🏀 NBA Game Predictor - Local Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Setup backend
    if not setup_backend():
        print("❌ Backend setup failed")
        return False
    
    # Setup frontend
    if not setup_frontend():
        print("❌ Frontend setup failed")
        return False
    
    # Create startup scripts
    create_start_scripts()
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Start the backend: start_backend.bat")
    print("2. Start the frontend: start_frontend.bat (in a new terminal)")
    print("3. Open http://localhost:3000 in your browser")
    print("\n💡 The backend will load all models on startup (may take a few minutes)")
    print("💡 Once loaded, predictions will be instant!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the errors above.")
        sys.exit(1)
