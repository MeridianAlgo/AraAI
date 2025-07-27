@echo off
title Ara AI Stock Analysis - Launcher
color 0B

cd /d "%~dp0"

cls
echo.
echo                    ╔══════════════════════════════════════════════════════════════╗
echo                    ║                 ARA AI STOCK ANALYSIS                        ║
echo                    ║              Perfect Prediction System                       ║
echo                    ║                    Quick Launcher                            ║
echo                    ╚══════════════════════════════════════════════════════════════╝
echo.

REM Check if .env file exists and has API key
if not exist ".env" (
    echo ❌ API Key Setup Required!
    echo.
    echo Please set up your API keys first:
    echo 1. Get Gemini API key: https://makersuite.google.com/app/apikey
    echo 2. Edit the .env file in this folder
    echo 3. Replace 'your_gemini_api_key_here' with your actual key
    echo 4. Save the file and run this again
    echo.
    echo Opening .env file for editing...
    if exist "code.exe" (
        start code .env
    ) else (
        start notepad .env
    )
    pause
    exit /b 1
)

findstr /C:"your_gemini_api_key_here" .env >nul
if %errorlevel% == 0 (
    echo ❌ API Key Setup Required!
    echo.
    echo Please replace 'your_gemini_api_key_here' in your .env file with your actual Gemini API key
    echo Get your key at: https://makersuite.google.com/app/apikey
    echo.
    echo Opening .env file for editing...
    start notepad .env
    pause
    exit /b 1
)

echo ✅ API keys configured!
echo.
echo 🚀 Ready to analyze stocks!
echo.
echo Popular symbols: AAPL, NVDA, TSLA, MSFT, GOOGL, AMZN, META
echo.
set /p SYMBOL="Enter stock symbol: "

if "%SYMBOL%"=="" (
    echo No symbol entered. Exiting.
    pause
    exit /b 1
)

echo.
echo 🔍 Analyzing %SYMBOL%...
echo This may take a moment for the first run...
echo.

python run_ara.py

pause