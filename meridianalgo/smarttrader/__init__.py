"""
Smart Trader - Ultra-Accurate AI Stock Analysis
Universal GPU Support: AMD • Intel • NVIDIA • Apple Silicon
"""

import sys
import os

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class SmartTrader:
    """
    Smart Trader - Ultra-Accurate AI Stock Analysis
    
    Features:
    - Universal GPU Support (AMD, Intel, NVIDIA, Apple Silicon)
    - Ensemble ML Models (LSTM + Transformer + XGBoost)
    - Advanced Technical Analysis
    - Real-time Market Data
    - Professional-grade Predictions
    """
    
    def __init__(self):
        self.version = "1.0.0"
        self.description = "Ultra-Accurate AI Stock Analysis with Universal GPU Support"
    
    def analyze(self, symbol, days=60, epochs=10, verbose=False):
        """
        Analyze a stock symbol with AI predictions
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'TSLA')
            days (int): Historical data days (default: 60)
            epochs (int): Training epochs (default: 10)
            verbose (bool): Show detailed logs (default: False)
        
        Returns:
            bool: True if analysis successful, False otherwise
        """
        try:
            from smart_trader_fixed import smart_trade_analysis, VERBOSE
            
            # Set global verbose flag
            import smart_trader_fixed
            smart_trader_fixed.VERBOSE = verbose
            
            return smart_trade_analysis(symbol, days, epochs)
        except ImportError:
            try:
                from smart_trader import smart_trade_analysis
                return smart_trade_analysis(symbol, days, epochs)
            except ImportError:
                print("Error: Smart Trader modules not found")
                return False
    
    def gpu_info(self):
        """Show GPU information and setup status"""
        try:
            from smart_trader_fixed import show_gpu_info
            show_gpu_info()
        except ImportError:
            try:
                from smart_trader import show_gpu_setup_info
                show_gpu_setup_info()
            except ImportError:
                print("Error: GPU info module not found")

__all__ = ["SmartTrader"]       self.version = "1.0.0"
        self.description = "Ultra-Accurate AI Stock Analysis with Universal GPU Support"
    
    def analyze(self, symbol, days=60, epochs=10, verbose=False):
        """
        Analyze a stock symbol with AI predictions
        
        Args:
            symbol (str): Stock symbol (e.g., 'AAPL', 'TSLA')
            days (int): Historical data days (default: 60)
            epochs (int): Training epochs (default: 10)
            verbose (bool): Show detailed logs (default: False)
        
        Returns:
            bool: True if analysis successful, False otherwise
        """
        try:
            from smart_trader_fixed import smart_trade_analysis, VERBOSE
            
            # Set global verbose flag
            import smart_trader_fixed
            smart_trader_fixed.VERBOSE = verbose
            
            return smart_trade_analysis(symbol, days, epochs)
        except ImportError:
            try:
                from smart_trader import smart_trade_analysis
                return smart_trade_analysis(symbol, days, epochs)
            except ImportError:
                print("Error: Smart Trader modules not found")
                return False
    
    def gpu_info(self):
        """Show GPU information and setup status"""
        try:
            from smart_trader_fixed import show_gpu_info
            show_gpu_info()
        except ImportError:
            try:
                from smart_trader import show_gpu_setup_info
                show_gpu_setup_info()
            except ImportError:
                print("Error: GPU info module not found")

__all__ = ["SmartTrader"]