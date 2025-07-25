# 🚀 MeridianAlgo Smart Trader

**Ultra-Accurate AI Stock Analysis with Universal GPU Support**

[![PyPI version](https://badge.fury.io/py/meridianalgo-smarttrader.svg)](https://badge.fury.io/py/meridianalgo-smarttrader)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GPU Support](https://img.shields.io/badge/GPU-AMD%20%E2%80%A2%20Intel%20%E2%80%A2%20NVIDIA%20%E2%80%A2%20Apple-green.svg)](https://github.com/MeridianAlgo/In-Python)

Professional-grade stock analysis powered by ensemble machine learning with universal GPU acceleration. Features advanced volatility spike detection and real-time technical analysis.

## ✨ Key Features

- 🎯 **Ultra-Accurate Predictions**: Ensemble ML models (LSTM + Transformer + XGBoost)
- 🔥 **Universal GPU Support**: AMD • Intel • NVIDIA • Apple Silicon
- ⚡ **Volatility Spike Detection**: Predict market turbulence before it happens
- 📊 **Real-time Analysis**: Live market data with technical indicators
- 🎨 **Clean Output**: Simplified, essential information only
- 🚀 **Easy Integration**: Simple Python API and CLI

## 🚀 Quick Start

### Installation

```bash
pip install meridianalgo-smarttrader
```

### Command Line Usage

```bash
# Analyze Apple stock
smart-trader AAPL

# Custom parameters
smart-trader TSLA --days 90 --epochs 15

# Show GPU information
smart-trader --gpu-info
```

### Python API Usage

```python
from meridianalgo import SmartTrader, analyze_stock

# Simple analysis
result = analyze_stock('AAPL')
print(f"Current: ${result['current_price']:.2f}")
print(f"Tomorrow: ${result['predictions'][0]:.2f}")

# Advanced usage
trader = SmartTrader(verbose=True)
analysis = trader.analyze('TSLA', days=60, epochs=10)

# Check volatility spike risk
vol_risk = analysis['volatility_spike']['spike_probability']
if vol_risk > 60:
    print("⚠️ High volatility spike risk detected!")
```

## 📊 Sample Output

```
🚀 AAPL Analysis
Device: CPU (8 threads)

┌────────────┬──────────┬──────────────────┐
│ Metric     │ Value    │ Info             │
├────────────┼──────────┼──────────────────┤
│ Current    │ $213.96  │ Real-time        │
│ Day +1     │ $216.45  │ +1.2%            │
│ Day +2     │ $218.30  │ +2.0%            │
│ Day +3     │ $215.80  │ +0.9%            │
│ Confidence │ 84%      │ Model reliability│
│ Vol Risk   │ 23%      │ ✅ Low risk      │
└────────────┴──────────┴──────────────────┘
```

## 🔥 Universal GPU Support

Smart Trader automatically detects and optimizes for your GPU:

| Vendor | Technology | Status |
|--------|------------|--------|
| 🟢 NVIDIA | CUDA | ✅ Supported |
| 🔴 AMD | ROCm/DirectML | ✅ Supported |
| 🔵 Intel | XPU | ✅ Supported |
| 🍎 Apple | MPS | ✅ Supported |

### GPU Setup

```bash
# NVIDIA GPU
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# AMD GPU (Windows)
pip install torch-directml

# Intel GPU
pip install intel-extension-for-pytorch

# Apple Silicon (automatic)
pip install torch torchvision torchaudio
```

## ⚡ Volatility Spike Detection

Smart Trader's advanced algorithm analyzes historical volatility patterns to predict future market turbulence:

```python
from meridianalgo import detect_volatility_spikes
import yfinance as yf

# Get stock data
data = yf.Ticker('AAPL').history(period='1y')

# Detect volatility spikes
spike_info = detect_volatility_spikes(data)

print(f"Spike Probability: {spike_info['spike_probability']:.1f}%")
print(f"Expected in: {spike_info['expected_spike_days']} days")
print(f"Risk Level: {spike_info['risk_level']}")
```

## 🎯 Advanced Features

### Ensemble Models
- **LSTM**: Captures long-term dependencies
- **Transformer**: Attention-based pattern recognition  
- **XGBoost**: Gradient boosting for robustness

### Technical Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume analysis

### Risk Management
- Volatility spike prediction
- Market regime detection
- Confidence scoring
- Position sizing recommendations

## 📈 Performance

| Metric | CPU | GPU |
|--------|-----|-----|
| Training Time (10 epochs) | ~2-3 seconds | ~0.5-1 seconds |
| Batch Size | 32 | 64+ |
| Memory Usage | 2-4 GB RAM | GPU VRAM |
| Accuracy | High | Higher |

## 🛠️ Development

### Local Installation

```bash
git clone https://github.com/MeridianAlgo/In-Python.git
cd In-Python
pip install -e .
```

### Running Tests

```bash
pytest tests/
```

### Building Package

```bash
python setup.py sdist bdist_wheel
twine upload dist/*
```

## 📚 Documentation

- [GPU Setup Guide](GPU_SETUP_GUIDE.md)
- [API Reference](docs/api.md)
- [Examples](examples/)
- [Contributing](CONTRIBUTING.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- **PyPI**: https://pypi.org/project/meridianalgo/
- **GitHub**: https://github.com/MeridianAlgo/In-Python
- **Documentation**: https://meridianalgo.github.io/In-Python/
- **Issues**: https://github.com/MeridianAlgo/In-Python/issues

## 🏆 About MeridianAlgo

MeridianAlgo specializes in advanced financial AI solutions. Our mission is to democratize professional-grade trading tools through cutting-edge machine learning and universal GPU acceleration.

---

**Made with ❤️ by MeridianAlgo**

*Empowering traders with AI-driven insights*