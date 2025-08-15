@echo off
REM Real-time News RAG with Fact-Checking - Quick Start Script for Windows
REM This script helps you get the system running quickly on Windows

echo 🚀 Real-time News RAG with Fact-Checking - Quick Start
echo ==================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH. Please install Python 3.11+ first.
    pause
    exit /b 1
)

REM Check Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo ✅ Python %PYTHON_VERSION% detected

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📚 Installing dependencies...
pip install -r requirements.txt

REM Check if .env file exists
if not exist ".env" (
    echo ⚠️  No .env file found. Creating from template...
    if exist "env.example" (
        copy env.example .env
        echo ✅ .env file created from template
        echo 📝 Please edit .env file with your API keys before running the app
    ) else (
        echo ❌ env.example not found. Please create .env file manually
    )
) else (
    echo ✅ .env file found
)

REM Test the system
echo 🧪 Testing system components...
python test_system.py

echo.
echo 🎉 Setup complete!
echo.
echo Next steps:
echo 1. Edit .env file with your API keys
echo 2. Run: streamlit run app.py
echo 3. Open browser to: http://localhost:8501
echo.
echo For help, see README.md and DEPLOYMENT.md
pause
