"""
MeridianAlgo - A Python library for algorithmic trading and financial analysis
"""

__version__ = "0.1.1"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main modules
from .trading_engine import TradingEngine
from .backtest_engine import BacktestEngine
from .indicators import Indicators
from .utils import TradeUtils

__all__ = [
    "TradingEngine",
    "BacktestEngine", 
    "Indicators",
    "TradeUtils"
] 