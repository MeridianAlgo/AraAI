"""
MeridianAlgo - Advanced stock prediction system using Yahoo Finance - Zero setup, no API keys required
"""

__version__ = "0.3.0"
__author__ = "MeridianAlgo"
__email__ = "meridianalgo@gmail.com"

# Import main modules
from .trading_engine import TradingEngine
from .backtest_engine import BacktestEngine
from .indicators import Indicators
from .utils import TradeUtils

# Import new ML modules
from .ml_predictor import MLPredictor
from .ai_analyzer import AIAnalyzer
from .ensemble_models import EnsembleModels

__all__ = [
    "TradingEngine",
    "BacktestEngine", 
    "Indicators",
    "TradeUtils",
    "MLPredictor",
    "AIAnalyzer",
    "EnsembleModels"
]