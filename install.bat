@echo off
echo ========================================
echo    Ara - AI Stock Analysis Platform
echo         Quick Install Script
echo ========================================
echo.

echo [1/4] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python from https://python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)
echo SUCCESS: Python is installed

echo.
echo [2/4] Creating virtual environment...
python -m venv ara_env
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)
echo SUCCESS: Virtual environment created

echo.
echo [3/4] Activating virtual environment...
call ara_env\Scripts\activate.bat

echo.
echo [4/4] Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ========================================
echo     Installation Complete!
echo ========================================
echo.
echo To use Ara:
echo 1. Run: ara_env\Scripts\activate.bat
echo 2. Then: python ara.py AAPL
echo.
echo Example commands:
echo   python ara.py AAPL
echo   python ara.py TSLA --days 30
echo   python ara.py --gpu-info
echo.
pause