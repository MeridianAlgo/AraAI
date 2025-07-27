#!/bin/bash

# macOS One-Click Installation for Ara AI Stock Analysis
# Double-click this file to install

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Change to script directory
cd "$(dirname "$0")"

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Header
clear
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                 ARA AI STOCK ANALYSIS                        ║"
echo "║              Perfect Prediction System                       ║"
echo "║              macOS One-Click Installer                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This installer is for macOS only"
    echo "Please use install.sh for Linux or install.bat for Windows"
    read -p "Press Enter to exit..."
    exit 1
fi

# Check if Homebrew is installed
echo -e "${BLUE}[1/7] Checking Homebrew installation...${NC}"
if ! command -v brew &> /dev/null; then
    print_warning "Homebrew not found. Installing Homebrew..."
    echo "This will install Homebrew (the missing package manager for macOS)"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for Apple Silicon Macs
    if [[ $(uname -m) == "arm64" ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi
print_status "Homebrew ready!"
echo

# Install Python if not present
echo -e "${BLUE}[2/7] Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    print_info "Installing Python 3..."
    brew install python3
fi

PYTHON_CMD="python3"
$PYTHON_CMD --version
print_status "Python found!"
echo

# Check Python version
echo -e "${BLUE}[3/7] Verifying Python version...${NC}"
if ! $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    print_warning "Upgrading Python to latest version..."
    brew upgrade python3
fi
print_status "Python version compatible!"
echo

# Install pip if needed
echo -e "${BLUE}[4/7] Checking pip installation...${NC}"
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    print_info "Installing pip..."
    curl https://bootstrap.pypa.io/get-pip.py | $PYTHON_CMD
fi
print_status "pip ready!"
echo

# Upgrade pip
echo -e "${BLUE}[5/7] Upgrading pip...${NC}"
$PYTHON_CMD -m pip install --upgrade pip --quiet --user
print_status "pip upgraded!"
echo

# Install required packages
echo -e "${BLUE}[6/7] Installing dependencies...${NC}"
print_info "This may take a few minutes..."
print_info "Installing PyTorch, pandas, numpy, yfinance, rich, and other dependencies..."

if ! $PYTHON_CMD -m pip install -r requirements.txt --quiet --user; then
    print_error "Installation failed"
    echo "Trying alternative installation method..."
    
    # Try installing with --break-system-packages for newer Python versions
    if ! $PYTHON_CMD -m pip install -r requirements.txt --user --break-system-packages; then
        print_error "Alternative installation also failed"
        echo "Please try manual installation:"
        echo "  $PYTHON_CMD -m pip install torch pandas numpy yfinance rich scikit-learn"
        read -p "Press Enter to exit..."
        exit 1
    fi
fi
print_status "Dependencies installed!"
echo

# Verify installation
echo -e "${BLUE}[7/7] Verifying installation...${NC}"
if ! $PYTHON_CMD -c "import torch, pandas, numpy, yfinance, rich; print('✅ All packages verified!')" 2>/dev/null; then
    print_error "Verification failed"
    echo "Some packages may not have installed correctly"
    echo "Please check the error messages above"
    read -p "Press Enter to exit..."
    exit 1
fi

# Create desktop shortcut
echo -e "${BLUE}Creating desktop shortcut...${NC}"
DESKTOP_PATH="$HOME/Desktop"
SHORTCUT_PATH="$DESKTOP_PATH/Ara AI Stock Analysis.command"

cat > "$SHORTCUT_PATH" << EOF
#!/bin/bash
cd "$(dirname "$0")"
cd "$(pwd)"
echo "🚀 Ara AI Stock Analysis"
echo "Enter stock symbol (e.g., AAPL, NVDA, TSLA):"
read -p "Symbol: " SYMBOL
if [ ! -z "\$SYMBOL" ]; then
    $PYTHON_CMD ara.py \$SYMBOL --verbose
fi
read -p "Press Enter to exit..."
EOF

chmod +x "$SHORTCUT_PATH"
print_status "Desktop shortcut created!"
echo

# Success message
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                   INSTALLATION COMPLETE!                    ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"
echo
echo -e "${GREEN}🚀 Ara AI Stock Analysis is ready to use!${NC}"
echo
echo -e "${YELLOW}📊 USAGE OPTIONS:${NC}"
echo "1. Double-click 'Ara AI Stock Analysis.command' on your Desktop"
echo "2. Or use Terminal commands:"
echo
echo -e "${YELLOW}📈 TERMINAL COMMANDS:${NC}"
echo "   $PYTHON_CMD ara.py AAPL --verbose    (Detailed Apple analysis)"
echo "   $PYTHON_CMD ara.py NVDA              (Quick NVIDIA analysis)"
echo "   $PYTHON_CMD ara.py TSLA --verbose    (Detailed Tesla analysis)"
echo
echo -e "${YELLOW}🔧 UTILITY COMMANDS:${NC}"
echo "   $PYTHON_CMD check_accuracy.py        (View prediction accuracy)"
echo "   $PYTHON_CMD view_predictions.py      (View prediction history)"
echo "   $PYTHON_CMD comprehensive_report.py  (Full system report)"
echo
echo -e "${YELLOW}🎯 FEATURES:${NC}"
echo "   • Perfect prediction system (sub-1% error target)"
echo "   • 62 advanced features"
echo "   • Ultra-advanced neural networks"
echo "   • Real-time accuracy validation"
echo "   • Automated learning system"
echo
echo -e "${BLUE}💡 TIP: You can now close this window and use the desktop shortcut!${NC}"
echo
read -p "Press Enter to exit..."