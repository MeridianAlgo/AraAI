@echo off
title Ara AI Stock Analysis - Perfect Prediction System
color 0B

cls
echo.
echo                    ╔══════════════════════════════════════════════════════════════╗
echo                    ║                 ARA AI STOCK ANALYSIS                        ║
echo                    ║              Perfect Prediction System                       ║
echo                    ║                 Windows Installation                         ║
echo                    ╚══════════════════════════════════════════════════════════════╝
echo.
echo                           🚀 Ultra-Advanced Neural Network Predictions
echo                              📊 Sub-1%% Error Rate Target
echo                                 🎯 62 Advanced Features
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

REM Setup environment variables
echo [5/7] Setting up environment variables...
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env" >nul
        echo ✅ Created .env file from template
    ) else (
        echo ⚠️  .env.example not found, creating basic .env file
        echo # Ara AI Stock Analysis - Environment Variables > .env
        echo GEMINI_API_KEY=your_gemini_api_key_here >> .env
        echo PAPER_TRADING=true >> .env
        echo LOG_LEVEL=INFO >> .env
    )
) else (
    echo ℹ️  .env file already exists
)
echo.

REM API Key Setup Guide
echo [6/7] API Key Setup Required...
echo 🔑 REQUIRED API KEYS:
echo.
echo 1. Google Gemini API (Required for AI analysis):
echo    • Visit: https://makersuite.google.com/app/apikey
echo    • Sign in with Google account
echo    • Click 'Create API Key'
echo    • Copy the key and paste it in your .env file
echo.
echo 2. Optional APIs for enhanced features:
echo    • Alpaca Trading: https://alpaca.markets/ (for live trading)
echo    • News API: https://newsapi.org/ (for sentiment analysis)
echo.
echo ⚠️  IMPORTANT: You must set up your API keys before running the program!
echo    Edit the .env file and replace 'your_gemini_api_key_here' with your actual key
echo.

REM Verify installation
echo [7/7] Verifying installation...
python -c "import torch, pandas, numpy, yfinance, rich; print('✅ All packages verified!')"
if errorlevel 1 (
    echo ❌ Verification failed
    echo Some packages may not have installed correctly
    pause
    exit /b 1
)

REM Open IDE for API key setup
echo Opening your code editor for API key setup...
echo 📝 NEXT STEPS:
echo 1. Set up your API keys in the .env file
echo 2. Save the file
echo 3. Run the program!
echo.

REM Try to open VS Code, then other editors
where code >nul 2>&1
if %errorlevel% == 0 (
    echo ℹ️  Opening VS Code...
    start code .
) else (
    where cursor >nul 2>&1
    if %errorlevel% == 0 (
        echo ℹ️  Opening Cursor...
        start cursor .
    ) else (
        where notepad++ >nul 2>&1
        if %errorlevel% == 0 (
            echo ℹ️  Opening Notepad++...
            start notepad++ .env
        ) else (
            echo ⚠️  No supported code editor found
            echo ℹ️  Opening .env file with Notepad...
            start notepad .env
        )
    )
)

echo.
echo ╔══════════════════════════════════════════════════════════════╗
echo ║                   INSTALLATION COMPLETE!                    ║
echo ╚══════════════════════════════════════════════════════════════╝
echo.
echo 🚀 Ready to use Ara AI Stock Analysis!
echo.
echo ⚠️  BEFORE RUNNING: Set up your API keys in the .env file!
echo.
echo 📊 USAGE OPTIONS:
echo    python run_ara.py               (Interactive launcher - recommended)
echo    python ara.py AAPL --verbose    (Detailed Apple analysis)
echo    python ara.py NVDA              (Quick NVIDIA analysis)
echo    python ara.py TSLA --verbose    (Detailed Tesla analysis)
echo.
echo 📈 UTILITY COMMANDS:
echo    python test_api.py              (Test your API key setup)
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
echo 🔑 API SETUP REMINDER:
echo    1. Get Gemini API key: https://makersuite.google.com/app/apikey
echo    2. Edit .env file and replace 'your_gemini_api_key_here'
echo    3. Save the file and run the program!
echo.
echo 💡 TIP: Your code editor should be opening now to edit the .env file!
echo.
echo 🎯 QUICK START: After setting up your API key, double-click "Ara_AI_Launcher.bat"
echo.
pause