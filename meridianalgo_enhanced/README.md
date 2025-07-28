# 🚀 MeridianAlgo - Advanced Stock Prediction System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/meridianalgo.svg)](https://badge.fury.io/py/meridianalgo)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Yahoo Finance](https://img.shields.io/badge/Data-Yahoo%20Finance-purple.svg)](https://finance.yahoo.com/)

**Advanced AI-powered stock prediction system using Yahoo Finance - Zero setup, no API keys required!**

## ⚡ Quick Start

```bash
pip install meridianalgo
```

```python
from meridianalgo import MLPredictor

# Initialize predictor (no API keys needed!)
predictor = MLPredictor()

# Get predictions for any stock
result = predictor.predict_ml('AAPL', days=60, epochs=10)

print(f"Current Price: ${result['current_price']:.2f}")
print(f"Day +1 Prediction: ${result['predictions'][0]:.2f}")
print(f"Confidence: {result['confidence']:.1f}%")
```

## 🎯 Key Features

- **🆓 Zero Setup**: No API keys, no registration - uses free Yahoo Finance data
- **🧠 Advanced AI**: 62 sophisticated features with deep neural networks
- **⚡ Real-Time Learning**: Automated accuracy validation and model adaptation
- **📊 Comprehensive Analysis**: Technical indicators, market sentiment, and volatility analysis
- **🔄 Smart Caching**: Intelligent prediction caching to avoid redundant analysis
- **📈 Multi-Symbol Support**: Analyze any stock symbol with persistent data storage
- **🛡️ Prediction Validation**: Multi-tier accuracy system with intelligent failsafes

## 📊 System Performance

```
🎯 Data Source: Yahoo Finance (Free, Real-time)
🔧 Setup Time: 0 seconds (no API keys required)
📈 Features: 62 advanced technical indicators
🧠 Model: Deep neural networks with attention mechanisms
⚡ Speed: Instant analysis with smart caching
🔄 Learning: Continuous model improvement
```

## 💻 Usage Examples

### Basic Stock Prediction

```python
from meridianalgo import MLPredictor

predictor = MLPredictor()

# Simple prediction
result = predictor.predict_simple('NVDA', days=30)
print(f"NVIDIA Prediction: ${result['predictions'][0]:.2f}")

# Advanced ML prediction
result = predictor.predict_ml('TSLA', days=60, epochs=15)
print(f"Tesla Confidence: {result['confidence']:.1f}%")
```

### Technical Analysis

```python
from meridianalgo import Indicators

indicators = Indicators()

# Calculate technical indicators
data = indicators.get_stock_data('AAPL', period='1y')
rsi = indicators.calculate_rsi(data['Close'])
macd = indicators.calculate_macd(data['Close'])

print(f"Current RSI: {rsi[-1]:.2f}")
print(f"MACD Signal: {macd['signal'][-1]:.4f}")
```

### Ensemble Models

```python
from meridianalgo import EnsembleModels

ensemble = EnsembleModels()

# Train ensemble models
data = ensemble.get_training_data('GOOGL', days=90)
training_results = ensemble.train_ensemble(data['X'], data['y'], epochs=20)

# Make predictions
predictions = ensemble.predict_ensemble(data['X'][-1:], forecast_days=5)
print(f"5-day predictions: {predictions['ensemble_predictions']}")
```

## 🛡️ Prediction Validation & Failsafes

### Multi-Tier Accuracy System
- **🎯 Excellent (<1% error)**: Highest quality predictions
- **✅ Good (<2% error)**: Strong prediction reliability
- **⚠️ Acceptable (<3% error)**: Minimum acceptable threshold
- **❌ Poor (>3% error)**: Triggers conservative fallback system

### Intelligent Failsafes
1. **Extreme Change Detection**: Flags predictions >50% change as unreliable
2. **Consistency Validation**: Ensures smooth day-to-day prediction transitions
3. **Confidence Thresholds**: Requires minimum 60% model confidence
4. **Volatility Context**: Adjusts expectations based on stock stability
5. **Volume Validation**: Considers trading volume for prediction reliability
6. **Conservative Fallbacks**: Applies ultra-safe predictions when validation fails

## 📦 Installation

### Standard Installation
```bash
pip install meridianalgo
```

### With ML Dependencies
```bash
pip install meridianalgo[ml]
```

### With Visualization
```bash
pip install meridianalgo[visualization]
```

### Full Installation
```bash
pip install meridianalgo[ml,visualization]
```

## 🔧 System Requirements

- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum, 8GB recommended
- **Storage**: 500MB free space
- **Internet**: Required for real-time stock data
- **OS**: Windows, macOS, Linux

## 📚 API Reference

### MLPredictor Class

```python
class MLPredictor:
    def predict_simple(self, symbol: str, days: int = 60, forecast_days: int = 5) -> Dict
    def predict_ml(self, symbol: str, days: int = 60, epochs: int = 10, forecast_days: int = 5) -> Dict
```

### Indicators Class

```python
class Indicators:
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series
    def calculate_macd(self, prices: pd.Series) -> Dict
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Dict
```

### EnsembleModels Class

```python
class EnsembleModels:
    def train_ensemble(self, X: np.ndarray, y: np.ndarray, epochs: int = 10) -> Dict
    def predict_ensemble(self, X: np.ndarray, forecast_days: int = 5) -> Dict
```

## 🎯 Advanced Features

### 62 Technical Features
1. **Market Microstructure**: VWAP, price ranges, volume-price trends
2. **Multi-Timeframe Momentum**: 7 different time horizons (1, 2, 3, 5, 8, 13, 21 days)
3. **Advanced Volatility**: GARCH-like modeling with clustering
4. **Market Regime Detection**: Trend strength and mean reversion
5. **Fractal Analysis**: Hurst exponent and fractal dimensions
6. **Technical Patterns**: Support/resistance and breakout probability

### Neural Network Architecture
- **Multi-scale feature extraction** with 1024, 512, 256-dim extractors
- **16-head attention mechanism** for pattern recognition
- **6 deep transformer blocks** for sequential processing
- **7 prediction heads** with uncertainty quantification
- **Advanced weight initialization** for optimal convergence

## 📊 Data Sources & Privacy

### Data Sources
- **Stock Data**: Yahoo Finance (yfinance)
- **Technical Indicators**: Custom implementations
- **Market Data**: Real-time price feeds
- **Validation**: Historical price verification

### Privacy & Security
- **No personal data** collection
- **Local processing** only
- **No data transmission** except for stock price fetching
- **Open source** and transparent

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](https://github.com/MeridianAlgo/Packages/blob/main/CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **PyTorch** team for the ML framework
- **Yahoo Finance** for stock data API
- **Rich** library for beautiful terminal output
- **Open source community** for inspiration and tools

---

**⚡ Ready to start predicting stocks with zero setup? Install now!**

```bash
pip install meridianalgo
```

**🎯 Advanced predictions with rigorous validation and intelligent failsafes!**