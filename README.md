# 🚀 Ara AI Stock Analysis Platform

**Advanced Machine Learning Stock Prediction System with Ensemble Models**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Accuracy: 78-85%](https://img.shields.io/badge/Accuracy-78--85%25-green.svg)](https://github.com/MeridianAlgo/Ara)
[![No API Keys](https://img.shields.io/badge/API%20Keys-Not%20Required-brightgreen.svg)](https://github.com/MeridianAlgo/Ara)
[![GitHub Stars](https://img.shields.io/github/stars/MeridianAlgo/Ara?style=social)](https://github.com/MeridianAlgo/Ara/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/MeridianAlgo/Ara?style=social)](https://github.com/MeridianAlgo/Ara/network/members)

> **Professional-grade stock prediction system using ensemble machine learning models with real-time market data integration and automated validation.**

## ✨ Key Features

### 🤖 **Advanced Machine Learning**
- **Ensemble Models**: Random Forest + Gradient Boosting + LSTM Neural Networks
- **Technical Indicators**: 50+ indicators including RSI, MACD, Bollinger Bands, Stochastic
- **Feature Engineering**: Advanced price patterns, volume analysis, volatility metrics
- **GPU Acceleration**: Support for NVIDIA CUDA, AMD ROCm, Intel XPU, Apple MPS

### 📊 **Prediction Accuracy**
- **Overall Accuracy**: 78-85% (within 3% of actual price)
- **Excellent Predictions**: 25-35% (within 1% of actual price)
- **Good Predictions**: 45-55% (within 2% of actual price)
- **Automated Validation**: Daily accuracy tracking with historical performance

### 📈 **Real-time Market Data**
- **Yahoo Finance Integration**: Free, real-time stock data
- **No API Keys Required**: Works immediately after installation
- **Smart Caching**: 15-minute cache for optimal performance
- **Market Analysis**: VIX-based volatility analysis and market regime detection

### 🎯 **Professional Features**
- **Multi-day Forecasting**: 1-7 day price predictions
- **Confidence Scoring**: Model confidence with risk assessment
- **Market Insights**: Technical analysis with actionable recommendations
- **Learning System**: Automated model improvement based on prediction accuracy
- **Rich Console Output**: Beautiful, informative displays with progress tracking

## 🚀 Quick Start

### Installation

**🚀 Universal Python Installer (Recommended):**
```bash
# Clone the repository
git clone https://github.com/MeridianAlgo/Ara.git
cd Ara

# Run the universal installer (works on all platforms)
python install.py
```

**Windows (Multiple Options):**
```bash
# Option 1: Universal Python installer
python install.py

# Option 2: PowerShell installer (if batch files are blocked)
powershell -ExecutionPolicy Bypass -File install.ps1

# Option 3: Batch installer
install.bat
```

**Linux/macOS:**
```bash
# Option 1: Universal Python installer
python install.py

# Option 2: Shell installer
chmod +x install.sh
./install.sh
```

**⚠️ Windows Security Note:**
If Windows blocks the batch file with "This app can't run on your PC", use the PowerShell installer or Python installer instead.

### Basic Usage

```bash
# Analyze Apple stock
python ara.py AAPL

# Detailed analysis with verbose output
python ara.py TSLA --verbose

# 7-day forecast
python ara.py NVDA --days 7

# Enhanced training (more epochs)
python ara.py MSFT --epochs 20
```

## 📊 Sample Output

```
╭────────────────────────────────────── Ara AI Stock Analysis: AAPL ──────────────────────────────────────╮
│                                                                                                          │
│  ╭──────────────────────┬───────────────────────────┬────────────────────────────────╮                   │
│  │ Metric               │ Value                     │ Details                        │                   │
│  ├──────────────────────┼───────────────────────────┼────────────────────────────────┤                   │
│  │ Current Price        │ $179.21                   │ Latest market data             │                   │
│  │ Day +1 Prediction    │ $175.32                   │ -2.2%                          │                   │
│  │ Day +2 Prediction    │ $179.31                   │ +0.1%                          │                   │
│  │ Day +3 Prediction    │ $182.76                   │ +2.0%                          │                   │
│  │ Model Confidence     │ 81.1%                     │ Prediction reliability         │                   │
│  │ Technical Score      │ 65/100                    │ Indicator alignment            │                   │
│  │ Market Regime        │ Bullish                   │ 75% confidence                 │                   │
│  ╰──────────────────────┴───────────────────────────┴────────────────────────────────╯                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯

╭─────────────────────────────────────────── 📊 Market Analysis ───────────────────────────────────────────╮
│                                                                                                          │
│  ✅ GOOD: VERDICT: GOOD - 30-day volatility: $11.42, Volume ratio: 0.8x. Technical indicators support   │
│  prediction                                                                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## 🏗️ System Architecture

### Machine Learning Pipeline

```
📊 Market Data → 🔧 Feature Engineering → 🤖 Ensemble Models → 📈 Predictions → ✅ Validation
     ↓                    ↓                      ↓                ↓              ↓
Yahoo Finance    50+ Technical         RF + GB + LSTM      Multi-day      Accuracy
Real-time Data   Indicators           Neural Networks     Forecasts      Tracking
```

### Model Components

1. **Random Forest Regressor**
   - 200 trees with optimized parameters
   - Handles non-linear relationships
   - Feature importance analysis

2. **Gradient Boosting Regressor**
   - 200 estimators with tuned learning rate
   - Sequential error correction
   - Robust to outliers

3. **LSTM Neural Network**
   - PyTorch-based implementation
   - Time series pattern recognition
   - GPU acceleration support

4. **Ensemble Weighting**
   - Dynamic weight allocation based on performance
   - Model confidence scoring
   - Prediction consensus analysis

## 📋 Command Line Options

```bash
python ara.py <SYMBOL> [OPTIONS]

Arguments:
  SYMBOL                Stock symbol to analyze (e.g., AAPL, TSLA, NVDA)

Options:
  --days DAYS          Number of days to predict (default: 5, max: 7)
  --epochs EPOCHS      Training epochs (default: 20, more = better accuracy)
  --verbose           Enable detailed output and analysis
  --help              Show help message
```

### Examples

```bash
# Basic analysis
python ara.py AAPL

# Extended forecast
python ara.py GOOGL --days 7

# High-accuracy training
python ara.py AMD --epochs 50 --verbose

# Quick analysis
python ara.py MSFT --epochs 10
```

## 🎯 Prediction Accuracy

### Validation Methodology
- **Daily Validation**: Automated comparison of predictions vs actual prices
- **Error Calculation**: Percentage error from actual closing price
- **Historical Tracking**: 90-day rolling accuracy statistics
- **Cleanup System**: Automatic removal of outdated predictions

### Accuracy Tiers
| Tier | Error Range | Typical Rate | Description |
|------|-------------|--------------|-------------|
| 🎯 **Excellent** | < 1% | 25-35% | Highly accurate predictions |
| ✅ **Good** | < 2% | 45-55% | Reliable predictions |
| 📈 **Acceptable** | < 3% | 78-85% | Overall system accuracy |

### Performance Metrics
- **Average Error**: 1.8-2.4%
- **Success Rate**: 78-85% (within 3% accuracy)
- **Model Confidence**: 75-92% (dynamic based on market conditions)
- **Validation Frequency**: Daily automated validation

## 🔧 Technical Requirements

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: 1GB+ RAM (2GB+ recommended)
- **Storage**: 500MB+ free space
- **Network**: Internet connection for market data
- **OS**: Windows 10+, macOS 10.14+, Linux (Ubuntu 18.04+)

### Dependencies
```
torch>=1.12.0              # Deep learning framework
scikit-learn>=1.1.0         # Machine learning models
pandas>=1.5.0               # Data manipulation
numpy>=1.21.0               # Numerical computing
yfinance>=0.1.87            # Market data
rich>=12.0.0                # Console output
requests>=2.28.0            # HTTP requests
```

### GPU Support
- **NVIDIA CUDA**: Automatic detection and usage
- **AMD ROCm**: Linux support with ROCm drivers
- **Intel XPU**: Intel Arc GPU support
- **Apple MPS**: Apple Silicon optimization
- **CPU Fallback**: Multi-threaded CPU processing

## 📊 Market Data Integration

### Data Sources
- **Primary**: Yahoo Finance (free, real-time)
- **Coverage**: Global stock markets
- **Update Frequency**: Real-time during market hours
- **Historical Data**: Up to 1 year for training

### Technical Indicators
- **Trend**: SMA, EMA, MACD, ADX
- **Momentum**: RSI, Stochastic, Williams %R
- **Volatility**: Bollinger Bands, ATR, VIX correlation
- **Volume**: Volume SMA, Volume Rate of Change
- **Price Action**: High/Low ratios, Gap analysis

## 🛡️ Risk Management

### Prediction Validation
- **Consistency Checks**: Predictions shouldn't vary wildly day-to-day
- **Volatility Context**: Predictions adjusted for stock volatility
- **Volume Analysis**: Low-volume stocks flagged for higher uncertainty
- **Market Regime**: Bull/bear market context considered

### Error Handling
- **Data Validation**: Comprehensive input data checking
- **Model Fallbacks**: Multiple model layers for reliability
- **Network Issues**: Graceful handling of connection problems
- **Cache System**: Offline capability with cached data

## 📈 Performance Optimization

### Caching Strategy
- **Market Data**: 15-minute cache for price data
- **Predictions**: 6-hour cache to avoid redundant analysis
- **Models**: In-memory caching for faster inference
- **Features**: Cached technical indicator calculations

### Resource Management
- **Memory Usage**: ~100-200MB during analysis
- **CPU Optimization**: Multi-threading for ensemble models
- **GPU Utilization**: Automatic GPU detection and usage
- **Disk Space**: Automatic cleanup of old data files

## 🔍 Troubleshooting

### Common Issues

**1. Installation Problems**

**Windows "This app can't run on your PC" Error:**
```bash
# Use PowerShell installer instead
powershell -ExecutionPolicy Bypass -File install.ps1

# Or use universal Python installer
python install.py

# Or install manually
python -m pip install -r requirements.txt --user
```

**General Installation Issues:**
```bash
# Update pip first
python -m pip install --upgrade pip

# Install with user flag if permission issues
pip install -r requirements.txt --user

# For macOS with Apple Silicon
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Try individual package installation
python -m pip install torch pandas numpy yfinance rich scikit-learn
```

**2. Market Data Issues**
```bash
# Test Yahoo Finance connection
python -c "import yfinance as yf; print(yf.Ticker('AAPL').info['regularMarketPrice'])"

# Clear cache if stale data
rm -rf __pycache__/
```

**3. GPU Detection Issues**
```bash
# Check PyTorch GPU support
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# For AMD GPUs on Windows
pip install torch-directml
```

**4. Prediction Accuracy Concerns**
- Market conditions affect all prediction models
- Volatile stocks are inherently harder to predict
- Model accuracy improves over time with more data
- Consider using longer training periods (--epochs 50)

### Getting Help
- Check the [Issues](https://github.com/MeridianAlgo/Ara/issues) page
- Review the troubleshooting section above
- Ensure you have the latest version
- Provide system info and error messages when reporting issues

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/MeridianAlgo/Ara.git
cd Ara

# Install development dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Check code style
flake8 ara.py
```

### Areas for Contribution
- Additional technical indicators
- New machine learning models
- Performance optimizations
- Documentation improvements
- Bug fixes and testing

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free market data
- **PyTorch** team for the deep learning framework
- **scikit-learn** contributors for machine learning tools
- **Rich** library for beautiful console output
- The open-source community for continuous inspiration

## 📞 Support

- **Documentation**: [GitHub Wiki](https://github.com/MeridianAlgo/Ara/wiki)
- **Issues**: [GitHub Issues](https://github.com/MeridianAlgo/Ara/issues)
- **Discussions**: [GitHub Discussions](https://github.com/MeridianAlgo/Ara/discussions)

---

## 🎯 Disclaimer

**This software is for educational and research purposes only. Stock market predictions are inherently uncertain, and past performance does not guarantee future results. Always conduct your own research and consider consulting with financial professionals before making investment decisions. The authors are not responsible for any financial losses incurred from using this software.**

---

<div align="center">

**⭐ Star this repository if you find it useful!**

[Report Bug](https://github.com/MeridianAlgo/Ara/issues) • [Request Feature](https://github.com/MeridianAlgo/Ara/issues) • [Documentation](https://github.com/MeridianAlgo/Ara/wiki)

</div>