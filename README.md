# Ara - AI Stock Analysis Platform

Ara is an advanced AI-powered stock analysis platform that uses ensemble machine learning models to provide accurate stock price predictions and comprehensive technical analysis.

## 🚀 Features

- **Advanced ML Ensemble**: Combines LSTM, Transformer, and XGBoost models for superior accuracy
- **Multi-GPU Support**: Optimized for NVIDIA CUDA, AMD ROCm/DirectML, Intel XPU, and Apple MPS
- **Technical Indicators**: 17+ technical indicators including RSI, MACD, Bollinger Bands, and more
- **Real-time Analysis**: Live market data integration with yfinance
- **Confidence Scoring**: Advanced confidence metrics and prediction reliability assessment
- **CSV Export**: Automatic export of predictions with timestamps
- **Clean Interface**: Minimalist black/white design with color-coded status messages

## 📋 Prerequisites

Before installing Ara, ensure you have the following software installed:

### 1. Python 3.8 or Higher
**Windows:**
- Download from [python.org](https://www.python.org/downloads/)
- During installation, check "Add Python to PATH"
- Verify installation: `python --version`

**macOS:**
```bash
# Using Homebrew (recommended)
brew install python

# Or download from python.org
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv
```

**Linux (CentOS/RHEL):**
```bash
sudo yum install python3 python3-pip
```

### 2. Git (Optional but recommended)
**Windows:** Download from [git-scm.com](https://git-scm.com/download/win)
**macOS:** `brew install git` or download from git-scm.com
**Linux:** `sudo apt install git` (Ubuntu) or `sudo yum install git` (CentOS)

## 🔧 Installation

### Method 1: Git Clone (Recommended)
```bash
# Clone the repository
git clone https://github.com/MeridianAlgo/Ara.git
cd Ara

# Create virtual environment (recommended)
python -m venv ara_env

# Activate virtual environment
# Windows:
ara_env\Scripts\activate
# macOS/Linux:
source ara_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Method 2: Direct Download
1. Download ZIP from [GitHub](https://github.com/MeridianAlgo/Ara/archive/main.zip)
2. Extract to desired folder
3. Open terminal/command prompt in the extracted folder
4. Run: `pip install -r requirements.txt`

### Method 3: One-Line Install (Windows PowerShell)
```powershell
git clone https://github.com/MeridianAlgo/Ara.git; cd Ara; python -m venv ara_env; ara_env\Scripts\activate; pip install -r requirements.txt
```

### Method 4: One-Line Install (macOS/Linux)
```bash
git clone https://github.com/MeridianAlgo/Ara.git && cd Ara && python3 -m venv ara_env && source ara_env/bin/activate && pip install -r requirements.txt
```

## 🚀 Quick Start

### Basic Stock Analysis
```bash
# Analyze Apple stock
python ara.py AAPL

# Analyze Tesla with 30 days of data
python ara.py TSLA --days 30

# Analyze Microsoft with verbose output
python ara.py MSFT --verbose
```

### Advanced Usage
```bash
# Extended analysis with more training data and epochs
python ara.py NVDA --days 90 --epochs 20

# Check GPU acceleration status
python ara.py --gpu-info

# Get help and see all options
python ara.py --help
```

## 📊 Understanding the Output

Ara provides comprehensive analysis including:

### Price Predictions
- **Current Price**: Latest market price
- **Day +1/+2/+3 Predictions**: Short-term price forecasts
- **Change Percentage**: Expected price movement

### Confidence Metrics
- **Model Confidence**: Overall prediction reliability (65-92%)
- **Technical Score**: Technical indicator alignment (0-100)
- **Volatility Adjusted**: Risk-adjusted confidence
- **Consistency**: Prediction stability score

### Market Analysis
- **Market Regime**: Bullish/Bearish/Sideways trend detection
- **Training Data**: Amount of historical data used
- **Device**: Hardware acceleration status

## ⚙️ Configuration Options

### Command Line Parameters
- `symbol`: Stock symbol (required) - e.g., AAPL, TSLA, MSFT, GOOGL
- `--days`: Historical data days (default: 60, range: 20-200)
- `--epochs`: Training epochs (default: 10, range: 5-50)
- `--verbose`: Show detailed logs and error messages
- `--gpu-info`: Display hardware acceleration information

### Environment Variables
Create a `.env` file in the project directory:
```bash
# Optional: Enable AI fact-checking (requires Gemini API key)
GEMINI_API_KEY=your_api_key_here
```

## 🖥️ Hardware Acceleration

Ara automatically detects and optimizes for available hardware:

### Supported GPUs
- **NVIDIA**: CUDA-compatible GPUs (GTX 10 series and newer)
- **AMD**: ROCm (Linux) or DirectML (Windows) compatible GPUs
- **Intel**: Arc GPUs with XPU support
- **Apple**: M1/M2/M3 Silicon with MPS

### Performance Comparison
- **CPU**: ~2-3 seconds per 10 epochs
- **GPU**: ~0.5-1 seconds per 10 epochs (2-6x faster)
- **Batch Size**: CPU=32, GPU=64+ (better accuracy)

### GPU Setup
Check GPU status: `python ara.py --gpu-info`

For detailed GPU setup instructions, see our [GPU Setup Guide](https://github.com/MeridianAlgo/Ara/wiki/GPU-Setup).

## 📁 Output Files

Ara automatically generates:
- **predictions.csv**: Timestamped predictions with metadata
- **Logs**: Detailed analysis logs (when using `--verbose`)

## 🔍 Example Analysis Session

```bash
$ python ara.py AAPL --days 45 --verbose

Using CPU with 8 threads

Ara - AI Stock Analysis for AAPL
Training Days: 45 | Epochs: 10 | Device: CPU (8 threads)

SUCCESS: Features prepared: 8 features, 76 samples
Training Advanced Ensemble Models...
SUCCESS: Advanced ensemble training completed
SUCCESS: Ultra-accurate predictions generated
Model Confidence: 74.2%
Technical Score: 78/100
Volatility Adjusted: 89%

┌─────────────────── Ara AI Stock Analysis: AAPL ───────────────────┐
│ Current Price    │ $213.88    │ Latest market data              │
│ Day +1 Prediction│ $216.45    │ +1.2%                          │
│ Day +2 Prediction│ $218.12    │ +2.0%                          │
│ Day +3 Prediction│ $219.87    │ +2.8%                          │
│ Model Confidence │ 74.2%      │ Prediction reliability         │
│ Market Regime    │ Bullish    │ 78% confidence                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🛠️ Troubleshooting

### Common Issues

**"ModuleNotFoundError"**
```bash
# Ensure virtual environment is activated and dependencies installed
pip install -r requirements.txt
```

**"No module named 'torch'"**
```bash
# Install PyTorch manually
pip install torch torchvision torchaudio
```

**"Permission denied" (Linux/macOS)**
```bash
# Use python3 instead of python
python3 ara.py AAPL
```

**"Stock symbol not found"**
- Verify the stock symbol exists (e.g., AAPL, not Apple)
- Check internet connection for market data

### Getting Help
- Run `python ara.py --help` for command options
- Use `--verbose` flag for detailed error messages
- Check [Issues](https://github.com/MeridianAlgo/Ara/issues) on GitHub

## 📈 Supported Stock Symbols

Ara works with any valid stock symbol from major exchanges:
- **US Markets**: AAPL, TSLA, MSFT, GOOGL, AMZN, NVDA, META, etc.
- **International**: Use Yahoo Finance format (e.g., ASML.AS, TSM, etc.)
- **Crypto**: BTC-USD, ETH-USD, etc.

## 🔒 Privacy & Security

- **No data collection**: All analysis runs locally
- **No account required**: No sign-up or registration
- **Optional API**: Gemini integration is completely optional
- **Open source**: Full transparency of code and algorithms

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📞 Support

- **GitHub Issues**: [Report bugs or request features](https://github.com/MeridianAlgo/Ara/issues)
- **Documentation**: [Wiki](https://github.com/MeridianAlgo/Ara/wiki)
- **Discussions**: [Community discussions](https://github.com/MeridianAlgo/Ara/discussions)

---

**Made with ❤️ by MeridianAlgo**