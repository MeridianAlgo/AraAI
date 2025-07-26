#!/bin/bash

echo "========================================"
echo "   Ara - AI Stock Analysis Platform"
echo "        Quick Install Script"
echo "========================================"
echo

echo "[1/4] Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python is not installed"
        echo "Please install Python:"
        echo "  Ubuntu/Debian: sudo apt install python3 python3-pip python3-venv"
        echo "  CentOS/RHEL: sudo yum install python3 python3-pip"
        echo "  macOS: brew install python"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi
echo "SUCCESS: Python is installed ($PYTHON_CMD)"

echo
echo "[2/4] Creating virtual environment..."
$PYTHON_CMD -m venv ara_env
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi
echo "SUCCESS: Virtual environment created"

echo
echo "[3/4] Activating virtual environment..."
source ara_env/bin/activate

echo
echo "[4/4] Installing dependencies..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo
echo "========================================"
echo "     Installation Complete!"
echo "========================================"
echo
echo "To use Ara:"
echo "1. Run: source ara_env/bin/activate"
echo "2. Then: $PYTHON_CMD ara.py AAPL"
echo
echo "Example commands:"
echo "  $PYTHON_CMD ara.py AAPL"
echo "  $PYTHON_CMD ara.py TSLA --days 30"
echo "  $PYTHON_CMD ara.py --gpu-info"
echo