# Ara - AI Stock Analysis Platform

Ara is an advanced AI-powered stock analysis platform that uses ensemble machine learning models to provide accurate stock price predictions and comprehensive technical analysis.

## Features

- **Advanced ML Ensemble**: Combines LSTM, Transformer, and XGBoost models for superior accuracy
- **Multi-GPU Support**: Optimized for NVIDIA CUDA, AMD ROCm/DirectML, Intel XPU, and Apple MPS
- **Technical Indicators**: 17+ technical indicators including RSI, MACD, Bollinger Bands, and more
- **Real-time Analysis**: Live market data integration with yfinance
- **Confidence Scoring**: Advanced confidence metrics and prediction reliability assessment
- **CSV Export**: Automatic export of predictions with timestamps
- **Clean Interface**: Minimalist black/white design with color-coded status messages

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MeridianAlgo/Ara.git
cd Ara
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up GPU acceleration - see GPU_SETUP_GUIDE.md

## Usage

### Basic Analysis
```bash
python ara.py AAPL
```

### Advanced Options
```bash
python ara.py TSLA --days 90 --epochs 20 --verbose
```

### GPU Information
```bash
python ara.py --gpu-info
```

## Parameters

- `symbol`: Stock symbol (required, e.g., AAPL, TSLA, MSFT)
- `--days`: Historical data days for training (default: 60)
- `--epochs`: Training epochs (default: 10)
- `--verbose`: Show detailed logs and errors
- `--gpu-info`: Display GPU setup information

## Output

Ara provides:
- Current stock price
- 3-day price predictions with confidence intervals
- Technical analysis scores
- Market regime detection (Bullish/Bearish/Sideways)
- Volatility-adjusted confidence metrics
- Prediction consistency scores
- CSV export of all predictions

## Hardware Acceleration

Ara automatically detects and optimizes for:
- NVIDIA GPUs (CUDA)
- AMD GPUs (ROCm/DirectML)
- Intel Arc GPUs (XPU)
- Apple Silicon (MPS)
- Multi-threaded CPU fallback

## API Integration

Set the `GEMINI_API_KEY` environment variable to enable AI fact-checking of predictions.

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please read CONTRIBUTING.md for guidelines.