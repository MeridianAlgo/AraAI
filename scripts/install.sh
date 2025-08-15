#!/bin/bash
# Ara AI Stock Analysis Platform - Unix/Linux/macOS Installer
# Enhanced security and error handling

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo
echo "========================================"
echo "  Ara AI Stock Analysis Platform"
echo "  Unix/Linux/macOS Installation Script"
echo "========================================"
echo

# Check if running as root (not recommended)
if [[ $EUID -eq 0 ]]; then
    print_warning "Running as root is not recommended"
    print_warning "Consider running as a regular user"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

print_status "[1/6] Checking Python installation..."

# Check Python installation
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        print_error "Python is not installed"
        echo
        echo "Please install Python 3.8+ using your system package manager:"
        echo "  Ubuntu/Debian: sudo apt update && sudo apt install python3 python3-pip"
        echo "  CentOS/RHEL:   sudo yum install python3 python3-pip"
        echo "  macOS:         brew install python3"
        echo "  Or visit:      https://python.org/downloads/"
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | cut -d' ' -f2)
print_success "Python $PYTHON_VERSION found"

echo
print_status "[2/6] Checking pip installation..."

# Check pip installation
if ! $PYTHON_CMD -m pip --version &> /dev/null; then
    print_error "pip is not available"
    print_status "Installing pip..."
    
    if command -v curl &> /dev/null; then
        curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        $PYTHON_CMD get-pip.py --user
        rm get-pip.py
    elif command -v wget &> /dev/null; then
        wget https://bootstrap.pypa.io/get-pip.py
        $PYTHON_CMD get-pip.py --user
        rm get-pip.py
    else
        print_error "Neither curl nor wget found. Please install pip manually."
        exit 1
    fi
fi

print_success "pip is available"

echo
print_status "[3/6] Creating Ara AI directory structure..."

# Create directories in user's home folder (no sudo needed)
ARA_DIR="$HOME/AraAI"
DATA_DIR="$ARA_DIR/data"
CACHE_DIR="$ARA_DIR/cache"
LOGS_DIR="$ARA_DIR/logs"
CONFIG_DIR="$ARA_DIR/config"

mkdir -p "$ARA_DIR" "$DATA_DIR" "$CACHE_DIR" "$LOGS_DIR" "$CONFIG_DIR"

print_success "Directory structure created at: $ARA_DIR"

echo
print_status "[4/6] Installing MeridianAlgo package..."

# Install with user flag to avoid permission issues
if ! $PYTHON_CMD -m pip install --user --upgrade meridianalgo; then
    print_warning "User installation failed, trying system installation..."
    if ! $PYTHON_CMD -m pip install --upgrade meridianalgo; then
        print_error "Installation failed completely"
        exit 1
    fi
fi

print_success "MeridianAlgo package installed"

echo
print_status "[5/6] Creating configuration files..."

# Create config file
cat > "$CONFIG_DIR/ara_config.ini" << EOF
# Ara AI Configuration
[DEFAULT]
data_dir = $DATA_DIR
cache_dir = $CACHE_DIR
logs_dir = $LOGS_DIR
verbose = false
cache_enabled = true
max_cache_age_hours = 24
EOF

# Create shell alias (optional)
SHELL_RC=""
if [[ "$SHELL" == *"bash"* ]]; then
    SHELL_RC="$HOME/.bashrc"
elif [[ "$SHELL" == *"zsh"* ]]; then
    SHELL_RC="$HOME/.zshrc"
fi

if [[ -n "$SHELL_RC" ]] && [[ -w "$SHELL_RC" ]]; then
    if ! grep -q "alias ara=" "$SHELL_RC" 2>/dev/null; then
        echo "" >> "$SHELL_RC"
        echo "# Ara AI alias" >> "$SHELL_RC"
        echo "alias ara='cd $ARA_DIR && ara'" >> "$SHELL_RC"
        print_status "Added ara alias to $SHELL_RC"
    fi
fi

# Make sure user's local bin is in PATH
USER_BIN="$HOME/.local/bin"
if [[ -d "$USER_BIN" ]] && [[ ":$PATH:" != *":$USER_BIN:"* ]]; then
    print_warning "User bin directory not in PATH: $USER_BIN"
    print_status "You may need to add it to your PATH or restart your shell"
fi

print_success "Configuration files created"

echo
print_status "[6/6] Testing installation..."

# Test the installation
if command -v ara &> /dev/null; then
    print_success "ara command is working"
elif $PYTHON_CMD -c "import meridianalgo; print('MeridianAlgo imported successfully')" &> /dev/null; then
    print_success "MeridianAlgo package is working"
    print_warning "'ara' command not found in PATH"
    print_status "You can use: python3 -c \"from meridianalgo import quick_predict; print(quick_predict('AAPL', 5))\""
else
    print_error "Installation test failed"
    exit 1
fi

echo
echo "========================================"
echo "  Installation Complete!"
echo "========================================"
echo
echo "Ara AI has been installed successfully!"
echo
echo "Installation Directory: $ARA_DIR"
echo
echo "Usage:"
echo "  ara AAPL --days 5        # Predict AAPL for 5 days"
echo "  ara --system-info        # Show system information"
echo "  ara --help               # Show all commands"
echo
echo "Alternative usage (if ara command not found):"
echo "  python3 -c \"from meridianalgo import quick_predict; print(quick_predict('AAPL', 5))\""
echo
echo "For support: https://github.com/MeridianAlgo/Ara"
echo

# Source shell config if it was modified
if [[ -n "$SHELL_RC" ]] && [[ -w "$SHELL_RC" ]]; then
    echo "Note: Restart your shell or run 'source $SHELL_RC' to use the ara alias"
fi