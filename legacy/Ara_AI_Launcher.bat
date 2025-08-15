@echo off
title Ara AI Stock Analysis - Launcher
color 0B

cd /d "%~dp0"

cls
echo.
echo                    ╔══════════════════════════════════════════════════════════════╗
echo                    ║                 🚀 ARA AI STOCK ANALYSIS 🚀                  ║
echo                    ║              Advanced Prediction System                      ║
echo                    ║                    Quick Launcher                            ║
echo                    ╚══════════════════════════════════════════════════════════════╝
echo.
echo                           📊 Yahoo Finance • 🆓 No Setup Required • ⚡ Instant Analysis
echo.

echo ✅ System ready! No API keys required - uses Yahoo Finance.
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