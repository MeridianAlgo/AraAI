# Ara AI Installation Guide

## 🚀 Quick Installation

### Option 1: PyPI Package (Recommended)
```bash
pip install meridianalgo
```

### Option 2: Secure Installation Scripts

#### Windows
```cmd
# Download and run the secure installer
curl -O https://raw.githubusercontent.com/MeridianAlgo/Ara/main/scripts/install.bat
install.bat
```

#### Linux/macOS
```bash
# Download and run the secure installer
curl -O https://raw.githubusercontent.com/MeridianAlgo/Ara/main/scripts/install.sh
chmod +x install.sh
./install.sh
```

## 🔒 Security Features

### Enhanced Security Measures
- **No Administrator Rights Required**: Installs in user directory
- **Secure Directory Permissions**: Proper file permissions on Unix systems
- **Input Validation**: Comprehensive input sanitization
- **Error Handling**: Graceful failure handling
- **Logging**: Comprehensive audit trail

### Directory Structure
```
~/AraAI/                    # Base directory (Windows: ~/Documents/AraAI/)
├── data/                   # Market data storage
├── cache/                  # Prediction cache
├── logs/                   # System logs
├── config/                 # Configuration files
└── models/                 # ML model storage
```

## 🛠️ System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 1GB free space
- **Internet**: Required for market data

### Recommended Requirements
- **Python**: 3.9+ for best performance
- **RAM**: 16GB for large datasets
- **GPU**: NVIDIA/AMD/Intel for acceleration
- **Storage**: 5GB for extended cache

## 🔧 Configuration

### Automatic Configuration
The installer creates a secure configuration file at:
- **Windows**: `~/Documents/AraAI/config/ara_config.ini`
- **Unix/Linux/macOS**: `~/AraAI/config/ara_config.ini`

### Manual Configuration
```ini
[DEFAULT]
data_dir = /path/to/data
cache_dir = /path/to/cache
logs_dir = /path/to/logs
verbose = false
cache_enabled = true
max_cache_age_hours = 24
max_predictions_per_symbol = 100
enable_gpu = true
security_mode = strict
```

## 🧪 Testing Installation

### Basic Test
```bash
ara --version
ara --system-info
```

### Prediction Test
```bash
ara AAPL --days 5
```

### Python API Test
```python
from meridianalgo import quick_predict
result = quick_predict('AAPL', days=5)
print(result)
```

## 🚨 Troubleshooting

### Common Issues

#### 1. Permission Denied Errors
**Solution**: Use user installation
```bash
pip install --user meridianalgo
```

#### 2. Command Not Found: 'ara'
**Solutions**:
- Restart terminal/command prompt
- Add Python Scripts to PATH
- Use: `python -m meridianalgo.cli`

#### 3. Import Errors
**Solution**: Reinstall with dependencies
```bash
pip uninstall meridianalgo
pip install meridianalgo --upgrade
```

#### 4. GPU Not Detected
**Solutions**:
- Install GPU drivers
- Install GPU-specific packages:
  - NVIDIA: `pip install torch[cuda]`
  - AMD: `pip install torch-directml`
  - Intel: `pip install intel-extension-for-pytorch`

### Error Logs
Check logs at:
- **Windows**: `~/Documents/AraAI/logs/`
- **Unix/Linux/macOS**: `~/AraAI/logs/`

## 🔄 Updating

### Update Package
```bash
pip install --upgrade meridianalgo
```

### Update Configuration
Delete config file to regenerate with new defaults:
```bash
# Windows
del "%USERPROFILE%\Documents\AraAI\config\ara_config.ini"

# Unix/Linux/macOS
rm ~/AraAI/config/ara_config.ini
```

## 🆘 Support

### Getting Help
1. **Documentation**: Check this guide first
2. **Logs**: Review error logs in `~/AraAI/logs/`
3. **GitHub Issues**: https://github.com/MeridianAlgo/Ara/issues
4. **System Info**: Run `ara --system-info` for diagnostics

### Reporting Issues
Include the following information:
- Operating system and version
- Python version (`python --version`)
- MeridianAlgo version (`ara --version`)
- Error logs from `~/AraAI/logs/`
- Steps to reproduce the issue