@echo off
title Ara AI Stock Analysis - Installation
color 0A

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                 ARA AI STOCK ANALYSIS                        ║
echo ║              Perfect Prediction System                       ║
echo ║                   Installation                               ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.

REM Check if Python is installed
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo.
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

python --version
echo ✅ Python found!
echo.

REM Check Python version
echo [2/5] Verifying Python version...
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"
if errorlevel 1 (
    echo ❌ Python 3.8+ required
    echo Please upgrade your Python installation
    pause
    exit /b 1
)
echo ✅ Python version compatible!
echo.

REM Upgrade pip
echo [3/5] Upgrading pip...
python -m pip install --upgrade pip --quiet
echo ✅ Pip upgraded!
echo.

REM Install required packages
echo [4/5] Installing dependencies...
echo This may take a few minutes...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ❌ Installation failed
    echo Please check your internet connection and try again
    pause
    exit /b 1
)
echo ✅ Dependencies installed!
echo.

REM Verify installation
echo [5/5] Verifying installation...
python -c "import torch, pandas, numpy, yfinance, rich; print('✅ All packages verified!')"
if errorlevel 1 (
    echo ❌ Verification failed
    echo Some packages may not have installed correctly
    pause
    exit /b 1
)

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                   INSTALLATION COMPLETE!                    ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo 🚀 Ready to use Ara AI Stock Analysis!
echo.
echo 📊 USAGE EXAMPLES:
echo    python ara.py AAPL --verbose    (Detailed Apple analysis)
echo    python ara.py NVDA              (Quick NVIDIA analysis)
echo    python ara.py TSLA --verbose    (Detailed Tesla analysis)
echo.
echo 📈 UTILITY COMMANDS:
echo    python check_accuracy.py        (View prediction accuracy)
echo    python view_predictions.py      (View prediction history)
echo    python comprehensive_report.py  (Full system report)
echo.
echo 🎯 FEATURES:
echo    • Perfect prediction system (sub-1%% error target)
echo    • 62 advanced features
echo    • Ultra-advanced neural networks
echo    • Real-time accuracy validation
echo    • Automated learning system
echo.
pause