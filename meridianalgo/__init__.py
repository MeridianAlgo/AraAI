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

Version: 2.0.0
"""

__version__ = "2.1.0"
__author__ = "MeridianAlgo Team"
__email__ = "support@meridianalgo.com"
__license__ = "MIT"

# Core imports for easy access
from .core import AraAI, StockPredictor
from .models import EnsembleMLSystem, LSTMModel
from .data import MarketDataManager, TechnicalIndicators
from .utils import GPUManager, CacheManager, AccuracyTracker

# Main prediction function for backward compatibility
from .core import predict_stock, analyze_stock

# Version info
VERSION_INFO = {
    'version': __version__,
    'features': [
        'Ensemble ML Models (RF + GB + LSTM)',
        'Multi-GPU Support (NVIDIA/AMD/Intel/Apple)',
        'Intelligent Prediction Caching',
        'Automated Accuracy Tracking',
        'Real-time Market Data',
        'Advanced Technical Indicators',
        'Online Learning System'
    ],
    'gpu_support': [
        'NVIDIA CUDA',
        'AMD ROCm/DirectML', 
        'Intel XPU',
        'Apple MPS'
    ],
    'python_versions': ['3.8+', '3.9+', '3.10+', '3.11+', '3.12+']
}

def get_version_info():
    """Get detailed version and feature information"""
    return VERSION_INFO

def check_gpu_support():
    """Check available GPU acceleration options"""
    from .utils import GPUManager
    return GPUManager.detect_gpu_vendor()

# Convenience functions
def quick_predict(symbol, days=5, use_cache=True):
    """
    Quick stock prediction with intelligent caching
    
    Args:
        symbol (str): Stock symbol (e.g., 'AAPL', 'TSLA')
        days (int): Number of days to predict (default: 5)
        use_cache (bool): Use cached predictions if available (default: True)
    
    Returns:
        dict: Prediction results with prices, accuracy info, and metadata
    """
    from .core import AraAI
    ara = AraAI()
    return ara.predict(symbol, days=days, use_cache=use_cache)

def analyze_accuracy(symbol=None):
    """
    Analyze prediction accuracy for a symbol or all symbols
    
    Args:
        symbol (str, optional): Specific symbol to analyze, or None for all
    
    Returns:
        dict: Accuracy statistics and performance metrics
    """
    from .utils import AccuracyTracker
    tracker = AccuracyTracker()
    return tracker.analyze_accuracy(symbol)

# Package metadata
__all__ = [
    # Core classes
    'AraAI',
    'StockPredictor', 
    'EnsembleMLSystem',
    'LSTMModel',
    'MarketDataManager',
    'TechnicalIndicators',
    'GPUManager',
    'CacheManager',
    'AccuracyTracker',
    
    # Main functions
    'predict_stock',
    'analyze_stock',
    'quick_predict',
    'analyze_accuracy',
    
    # Utility functions
    'get_version_info',
    'check_gpu_support',
    
    # Version info
    '__version__',
    'VERSION_INFO'
]