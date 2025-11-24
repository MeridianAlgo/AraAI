#  Troubleshooting Guide - ARA AI

**Solutions to common issues and problems**

##  **Quick Fixes**

### **Most Common Issues**
```bash
# 1. Models not trained
python ara_fast.py AAPL --retrain

# 2. Installation problems
pip install --upgrade pip
python install_ultimate_requirements.py

# 3. Permission errors (Windows)
pip install --user transformers torch
python install_ultimate_requirements.py

# 4. Clear cache and restart
rm -rf models/
python ara_fast.py AAPL --retrain
```

##  **Installation Issues**

### **Problem: Installation Fails**
```
Error: Failed to install requirements
```

#### **Solution 1: Upgrade Pip**
```bash
# Upgrade pip first
python -m pip install --upgrade pip

# Try installation again
python install_ultimate_requirements.py
```

#### **Solution 2: Manual Installation**
```bash
# Install core packages manually
pip install scikit-learn xgboost lightgbm
pip install numpy pandas yfinance rich
pip install transformers torch

# Test installation
python test_ultimate_system.py
```

#### **Solution 3: Use Virtual Environment**
```bash
# Create virtual environment
python -m venv araai_env

# Activate environment
source araai_env/bin/activate  # Linux/Mac
araai_env\Scripts\activate     # Windows

# Install in virtual environment
pip install -r requirements.txt
```

### **Problem: Permission Denied (Windows)**
```
Error: Permission denied when installing packages
```

#### **Solution 1: User Installation**
```bash
# Install for current user only
pip install --user transformers torch
python install_ultimate_requirements.py
```

#### **Solution 2: Run as Administrator**
```bash
# Right-click Command Prompt -> "Run as Administrator"
python install_ultimate_requirements.py
```

### **Problem: Python Not Found**
```
Error: 'python' is not recognized as an internal or external command
```

#### **Solution: Add Python to PATH**
```bash
# Windows: Add Python to PATH in System Properties
# Or use full path:
C:\Python39\python.exe install_ultimate_requirements.py

# Linux/Mac: Install Python
sudo apt install python3 python3-pip  # Ubuntu
brew install python                    # macOS
```

##  **Model Issues**

### **Problem: Models Not Trained**
```
Error: Models not trained. Loading or training...
```

#### **Solution: Train Models**
```bash
# Train models with default settings
python ara_fast.py AAPL --retrain

# Train with verbose output to see progress
python ara_fast.py AAPL --retrain --verbose

# Train with specific period
python ara_fast.py AAPL --retrain --period 1y
```

### **Problem: Model Loading Fails**
```
Error: Failed to load models
```

#### **Solution 1: Clear Model Cache**
```bash
# Remove corrupted models
rm -rf models/           # Linux/Mac
rmdir /s models          # Windows

# Retrain from scratch
python ara_fast.py AAPL --retrain
```

#### **Solution 2: Check Disk Space**
```bash
# Check available disk space
df -h .                  # Linux/Mac
dir                      # Windows

# Models need ~50MB, Hugging Face models need ~1GB
```

### **Problem: Hugging Face Models Fail to Download**
```
Error: Failed to download Hugging Face models
```

#### **Solution 1: Check Internet Connection**
```bash
# Test connection to Hugging Face
ping huggingface.co

# Test HTTPS access
curl -I https://huggingface.co
```

#### **Solution 2: Manual Model Download**
```bash
# Test model download
python check_hf_models.py

# Force model download
python -c "
from transformers import pipeline
pipeline('sentiment-analysis', model='ProsusAI/finbert')
"
```

#### **Solution 3: Proxy/Firewall Issues**
```bash
# Configure proxy (if needed)
export https_proxy=http://proxy.company.com:8080
pip install --proxy http://proxy.company.com:8080 transformers

# Or download offline
pip download transformers torch
pip install --no-index --find-links . transformers torch
```

##  **Prediction Issues**

### **Problem: Symbol Not Found**
```
Error: No data available for SYMBOL
```

#### **Solution 1: Check Symbol**
```bash
# Use correct ticker symbol
python ara_fast.py AAPL    # Correct
python ara_fast.py APPLE   # Incorrect

# Check symbol on Yahoo Finance first
```

#### **Solution 2: Try Different Symbol**
```bash
# Some symbols may be delisted or unavailable
python ara_fast.py MSFT    # Try a different stock
python ara_fast.py SPY     # Try an ETF
```

### **Problem: Low Accuracy**
```
Warning: Model accuracy below expected
```

#### **Solution 1: Retrain with More Data**
```bash
# Use longer training period
python ara_fast.py AAPL --retrain --period 2y

# Use maximum data
python ara_fast.py AAPL --retrain --period 5y
```

#### **Solution 2: Check Market Conditions**
```bash
# Volatile markets may have lower accuracy
# Check if stock has recent major events (earnings, splits, etc.)
python ara_fast.py AAPL --verbose  # See detailed analysis
```

### **Problem: Slow Predictions**
```
Issue: Predictions take too long
```

#### **Solution 1: Use Cached Models**
```bash
# Avoid unnecessary retraining
python ara_fast.py AAPL    # Uses cached models (fast)

# Only retrain when needed
python ara_fast.py AAPL --retrain  # Retrains (slower)
```

#### **Solution 2: Reduce Training Data**
```bash
# Use shorter training period for speed
python ara_fast.py AAPL --retrain --period 6mo
```

##  **Network Issues**

### **Problem: Network Connection Fails**
```
Error: Failed to fetch market data
```

#### **Solution 1: Check Internet Connection**
```bash
# Test basic connectivity
ping google.com
ping query1.finance.yahoo.com

# Test HTTPS access
curl -I https://query1.finance.yahoo.com
```

#### **Solution 2: Firewall/Proxy Issues**
```bash
# Check firewall settings
# Allow outbound HTTPS (port 443) to:
# - query1.finance.yahoo.com
# - huggingface.co (initial setup only)
# - pypi.org (installation only)

# Configure proxy if needed
export https_proxy=http://proxy:8080
```

#### **Solution 3: Use Offline Mode**
```bash
# After initial setup, ARA AI can work offline
# Use cached models and data
python ara_fast.py AAPL  # Works with cached data
```

### **Problem: SSL Certificate Errors**
```
Error: SSL certificate verification failed
```

#### **Solution 1: Update Certificates**
```bash
# Update system certificates
# Windows: Windows Update
# macOS: Software Update
# Linux: sudo apt update && sudo apt upgrade ca-certificates
```

#### **Solution 2: Python Certificate Fix**
```bash
# macOS: Run certificate installer
/Applications/Python\ 3.x/Install\ Certificates.command

# Or update certifi package
pip install --upgrade certifi
```

##  **Performance Issues**

### **Problem: High Memory Usage**
```
Issue: System runs out of memory
```

#### **Solution 1: Close Other Applications**
```bash
# Close unnecessary applications
# ARA AI needs 2-4GB RAM during training

# Check memory usage
free -h                  # Linux
vm_stat                  # macOS
tasklist /fi "imagename eq python.exe"  # Windows
```

#### **Solution 2: Reduce Training Data**
```bash
# Use smaller training dataset
python ara_fast.py AAPL --retrain --period 6mo

# Train on fewer stocks (for development)
# Edit ultimate_ml.py to reduce max_symbols
```

### **Problem: High CPU Usage**
```
Issue: System becomes unresponsive
```

#### **Solution 1: Reduce Parallel Processing**
```bash
# Disable parallel processing in training
# Edit ultimate_ml.py: use_parallel=False

# Or limit CPU cores
export OMP_NUM_THREADS=2  # Limit to 2 cores
```

#### **Solution 2: Run During Off-Hours**
```bash
# Train models when system is not busy
# Training is CPU-intensive but predictions are fast
```

### **Problem: Slow Disk I/O**
```
Issue: Model loading/saving is slow
```

#### **Solution 1: Check Disk Space**
```bash
# Ensure sufficient free space
df -h .                  # Linux/Mac
dir                      # Windows

# Clean up old files if needed
```

#### **Solution 2: Use SSD**
```bash
# Move to SSD for better performance
# Or use faster storage for models/ directory
```

##  **Python Issues**

### **Problem: Wrong Python Version**
```
Error: Python version 3.7 or lower detected
```

#### **Solution: Upgrade Python**
```bash
# Check current version
python --version

# Install Python 3.8+
# Windows: Download from python.org
# macOS: brew install python@3.9
# Linux: sudo apt install python3.9
```

### **Problem: Module Not Found**
```
Error: ModuleNotFoundError: No module named 'transformers'
```

#### **Solution 1: Install Missing Module**
```bash
# Install specific module
pip install transformers

# Or install all requirements
pip install -r requirements.txt
```

#### **Solution 2: Check Python Environment**
```bash
# Check which Python is being used
which python
python -c "import sys; print(sys.path)"

# Make sure you're using the right environment
```

### **Problem: Package Conflicts**
```
Error: Package version conflicts
```

#### **Solution 1: Use Virtual Environment**
```bash
# Create clean environment
python -m venv araai_clean
source araai_clean/bin/activate  # Linux/Mac
araai_clean\Scripts\activate     # Windows

# Install fresh
pip install -r requirements.txt
```

#### **Solution 2: Force Reinstall**
```bash
# Force reinstall conflicting packages
pip install --force-reinstall transformers torch
```

##  **Platform-Specific Issues**

### **Windows Issues**

#### **Problem: Long Path Names**
```
Error: Path too long
```
**Solution:**
```bash
# Enable long paths in Windows
# Run as Administrator:
# New-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name "LongPathsEnabled" -Value 1 -PropertyType DWORD -Force

# Or use shorter path
cd C:\araai
```

#### **Problem: Antivirus Interference**
```
Issue: Antivirus blocks model files
```
**Solution:**
```bash
# Add exclusions to antivirus:
# - ARA AI directory
# - Python installation directory
# - %USERPROFILE%\.cache\huggingface\
```

### **macOS Issues**

#### **Problem: Gatekeeper Blocks Execution**
```
Error: "python" cannot be opened because the developer cannot be verified
```
**Solution:**
```bash
# Allow in System Preferences > Security & Privacy
# Or use command line:
xattr -d com.apple.quarantine /path/to/python
```

#### **Problem: Homebrew Issues**
```
Error: Homebrew installation problems
```
**Solution:**
```bash
# Update Homebrew
brew update && brew upgrade

# Reinstall Python
brew reinstall python@3.9
```

### **Linux Issues**

#### **Problem: Missing System Dependencies**
```
Error: Failed to build package
```
**Solution:**
```bash
# Install build dependencies
sudo apt install build-essential python3-dev  # Ubuntu/Debian
sudo yum groupinstall "Development Tools"     # CentOS/RHEL
```

#### **Problem: Permission Issues**
```
Error: Permission denied
```
**Solution:**
```bash
# Use user installation
pip install --user transformers

# Or fix permissions
sudo chown -R $USER:$USER ~/.local/
```

##  **Debugging Tools**

### **Diagnostic Commands**
```bash
# Check system status
python test_ultimate_system.py

# Check model status
python check_hf_models.py

# Verbose prediction (see all details)
python ara_fast.py AAPL --verbose

# Check Python environment
python -c "
import sys
print('Python:', sys.version)
print('Path:', sys.path)
"

# Check installed packages
pip list | grep -E "(transformers|torch|scikit|pandas)"
```

### **Log Analysis**
```bash
# Run with verbose output
python ara_fast.py AAPL --verbose > debug.log 2>&1

# Check for errors
grep -i error debug.log
grep -i warning debug.log
grep -i failed debug.log
```

### **Network Debugging**
```bash
# Monitor network connections
netstat -an | grep python     # Linux/Mac
netstat -an | findstr python  # Windows

# Test specific connections
curl -I https://query1.finance.yahoo.com/v8/finance/chart/AAPL
curl -I https://huggingface.co
```

##  **Getting Help**

### **Before Asking for Help**
1. **Check this guide**: Look for your specific issue above
2. **Run diagnostics**: Use the diagnostic commands
3. **Check logs**: Look for error messages in verbose output
4. **Search issues**: Check GitHub Issues for similar problems

### **How to Report Issues**
When reporting problems, include:

#### **System Information**
```bash
# Collect system info
python -c "
import sys, platform
print('OS:', platform.system(), platform.release())
print('Python:', sys.version)
print('Architecture:', platform.architecture())
"

# Package versions
pip list | grep -E "(transformers|torch|scikit|pandas|numpy)"
```

#### **Error Details**
- **Full error message**: Copy the complete error
- **Steps to reproduce**: What commands you ran
- **Expected behavior**: What should have happened
- **Actual behavior**: What actually happened

#### **Environment Details**
- Operating system and version
- Python version
- ARA AI version/commit
- Whether you're using virtual environment
- Any proxy/firewall configuration

### **Where to Get Help**
- **GitHub Issues**: Report bugs and problems
- **GitHub Discussions**: Ask questions and get help
- **Documentation**: Check all docs in [docs/](.) folder
- **Code Review**: Examine source code for understanding

##  **Prevention Tips**

### **Avoid Common Issues**
1. **Keep updated**: Regular updates prevent many issues
2. **Use virtual environments**: Avoid package conflicts
3. **Check requirements**: Ensure system meets minimum requirements
4. **Monitor resources**: Watch CPU, memory, and disk usage
5. **Regular maintenance**: Clean cache and update dependencies

### **Best Practices**
1. **Test after installation**: Run `python test_ultimate_system.py`
2. **Start simple**: Begin with basic predictions before advanced features
3. **Monitor performance**: Watch for degradation over time
4. **Backup models**: Save trained models before major changes
5. **Document changes**: Keep track of configuration changes

---

** Still having issues? Don't hesitate to ask for help on GitHub!**

**Most problems have simple solutions - we're here to help you succeed.**