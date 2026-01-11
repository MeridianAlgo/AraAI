"""
MeridianAlgo - Advanced AI Stock Analysis Package
Enhanced with Ara AI's ensemble ML system, intelligent caching, and multi-GPU support

This package provides:
- Ensemble ML models (Random Forest + Gradient Boosting + LSTM)
- Multi-vendor GPU acceleration (NVIDIA, AMD, Intel, Apple)
- Intelligent prediction caching with accuracy tracking
- Real-time market data integration
- Advanced technical indicators
- Automated model validation and learning

Version: 3.0.0
"""

__version__ = "3.1.1"
__author__ = "MeridianAlgo Team"
__email__ = "support@meridianalgo.com"
__license__ = "MIT"

# Delayed imports are recommended to avoid pulling optional/legacy modules at import time

# Version info
VERSION_INFO = {
    "version": __version__,
    "features": [
        "ULTIMATE ML System (8 Models: XGBoost, LightGBM, Random Forest, etc.)",
        "Realistic Stock Predictions (±5% daily bounds)",
        "Advanced Financial Health Analysis (A+ to F grades)",
        "AI-Powered Sentiment Analysis (Hugging Face RoBERTa)",
        "Comprehensive Sector Detection (Technology, Finance, etc.)",
        "Multi-GPU Support (NVIDIA/AMD/Intel/Apple)",
        "Intelligent Prediction Caching",
        "Real-time Market Data Integration",
        "44 Advanced Technical Indicators",
        "Performance Monitoring & Optimization",
        "Automated Model Training & Validation",
    ],
    "gpu_support": ["NVIDIA CUDA", "AMD ROCm/DirectML", "Intel XPU", "Apple MPS"],
    "python_versions": ["3.9+", "3.10+", "3.11+", "3.12+"],
}


def get_version_info():
    return VERSION_INFO


def check_gpu_support():
    try:
        from .utils import GPUManager

        return GPUManager.detect_gpu_vendor()
    except Exception:
        return "unknown"


def quick_predict(*args, **kwargs):
    try:
        from .core import AraAI

        ara = AraAI()
        return ara.predict(*args, **kwargs)
    except Exception:
        return {"error": "core predictor unavailable"}


def analyze_accuracy(symbol=None):
    try:
        from .utils import AccuracyTracker

        tracker = AccuracyTracker()
        return tracker.analyze_accuracy(symbol)
    except Exception:
        return {"error": "accuracy tracker unavailable"}


# Package metadata
__all__ = [
    "__version__",
    "VERSION_INFO",
    "get_version_info",
    "check_gpu_support",
    "quick_predict",
    "analyze_accuracy",
]
