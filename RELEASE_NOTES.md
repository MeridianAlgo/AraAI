# Smart Trader v1.0.0

## 🚀 Features

### Ultra-Accurate AI Stock Analysis
- **Ensemble ML Models**: LSTM + Transformer + XGBoost
- **Advanced Technical Analysis**: 17+ indicators
- **Real-time Market Data**: Live price feeds
- **Confidence Scoring**: 70-88% realistic confidence ranges

### Universal GPU Support
- **🔴 AMD GPUs**: ROCm + DirectML support
- **🔵 Intel Arc GPUs**: XPU acceleration  
- **🟢 NVIDIA GPUs**: CUDA acceleration
- **🍎 Apple Silicon**: MPS optimization
- **💻 CPU Fallback**: Multi-threaded optimization

### Professional Features
- **Clean Interface**: Minimalistic, essential information
- **CSV Export**: Detailed prediction data
- **Performance Metrics**: Technical scores and accuracy tracking
- **Market Regime Detection**: Bullish/Bearish/Sideways identification

## 📦 Installation

```bash
pip install meridianalgo-smarttrader
```

## 🎯 Quick Start

```bash
# Analyze any stock
smart-trader AAPL --epochs 10

# Check GPU support
smart-trader --gpu-info
```

## 🔧 GPU Setup

### AMD GPUs
```bash
pip install torch-directml  # Windows
pip install torch --index-url https://download.pytorch.org/whl/rocm5.6  # Linux
```

### NVIDIA GPUs
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Intel Arc GPUs
```bash
pip install intel-extension-for-pytorch
```

## 📈 Performance

- **Training Time**: 1-3 seconds (CPU), 0.3-1 seconds (GPU)
- **Accuracy**: Professional-grade predictions with confidence scoring
- **Universal**: Works on any hardware (CPU/GPU)

## 🎯 What's New in v1.0.0

- ✅ Fixed model confidence (no more 0.0%)
- ✅ Minimalistic colors (only for important info)
- ✅ Universal GPU support (AMD, Intel, NVIDIA, Apple)
- ✅ Enhanced prediction accuracy
- ✅ Professional-grade output
- ✅ Comprehensive documentation

---

**Made with ❤️ by MeridianAlgo**
