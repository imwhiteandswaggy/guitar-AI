@echo off
REM Guitar Teacher AI - Setup Script for Windows
REM This script sets up the environment and installs dependencies

echo ==========================================
echo Guitar Teacher AI - Setup Script
echo ==========================================
echo.

REM Check Python version
echo Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

python --version
echo [OK] Python found

REM Create virtual environment
echo.
echo Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)

REM Activate virtual environment
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip --quiet

REM Install dependencies
echo.
echo Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo [ERROR] Failed to install dependencies
    echo Please check error messages above
    pause
    exit /b 1
)

echo.
echo [OK] Dependencies installed

REM Download model if not present
echo.
echo Checking for model file...
if not exist "trained_models\real_guitar_test3\weights\best.pt" (
    echo Model file not found. Downloading...
    python download_model.py
) else (
    echo [OK] Model file found
)

echo.
echo ==========================================
echo [OK] Setup complete!
echo ==========================================
echo.
echo To run the app:
echo   1. Activate virtual environment: venv\Scripts\activate
echo   2. Run the app: python app.py
echo   3. Open browser: http://localhost:5000
echo.
echo To deactivate virtual environment later:
echo   deactivate
echo.
pause
