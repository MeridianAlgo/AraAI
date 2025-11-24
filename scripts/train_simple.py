"""
Simple Training Script for ARA AI
Trains models on 2 stocks and 2 forex pairs
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meridianalgo.unified_ml import UnifiedStockML
from meridianalgo.forex_ml import ForexML
from meridianalgo.console import ConsoleManager

def train_stocks():
    """Train on AAPL and MSFT"""
    console = ConsoleManager()
    console.print_header("Training Stock Models")
    
    stocks = ['AAPL', 'MSFT']
    
    for symbol in stocks:
        console.print_info(f"Training {symbol}...")
        try:
            ml = UnifiedStockML()
            success = ml.train_ultimate_models(
                period='2y',
                use_parallel=False,
                target_symbol=symbol
            )
            if success:
                console.print_success(f"{symbol} training completed!")
            else:
                console.print_warning(f"{symbol} training completed with warnings")
        except Exception as e:
            console.print_error(f"{symbol} training failed: {e}")

def train_forex():
    """Train on EURUSD and GBPUSD"""
    console = ConsoleManager()
    console.print_header("Training Forex Models")
    
    pairs = ['EURUSD', 'GBPUSD']
    
    for pair in pairs:
        console.print_info(f"Training {pair}...")
        try:
            forex = ForexML()
            target_symbol = forex.get_forex_symbol(pair)
            success = forex.train_ultimate_models(
                target_symbol=target_symbol,
                period='2y',
                use_parallel=False
            )
            if success:
                console.print_success(f"{pair} training completed!")
            else:
                console.print_warning(f"{pair} training completed with warnings")
        except Exception as e:
            console.print_error(f"{pair} training failed: {e}")

def test_predictions():
    """Test predictions on all trained models"""
    console = ConsoleManager()
    console.print_header("Testing Predictions")
    
    # Test stocks
    stocks = ['AAPL', 'MSFT']
    for symbol in stocks:
        try:
            ml = UnifiedStockML()
            result = ml.predict_ultimate(symbol, days=5)
            if result and 'predictions' in result:
                console.print_success(f"{symbol}: ${result['current_price']:.2f} -> ${result['predictions'][-1]['predicted_price']:.2f}")
            else:
                console.print_warning(f"{symbol}: No predictions available")
        except Exception as e:
            console.print_error(f"{symbol} prediction failed: {e}")
    
    # Test forex
    pairs = ['EURUSD', 'GBPUSD']
    for pair in pairs:
        try:
            forex = ForexML()
            result = forex.predict_forex(pair, days=5)
            if result and 'predictions' in result:
                console.print_success(f"{pair}: {result['current_price']:.5f} -> {result['predictions'][-1]['predicted_price']:.5f}")
            else:
                console.print_warning(f"{pair}: No predictions available")
        except Exception as e:
            console.print_error(f"{pair} prediction failed: {e}")

if __name__ == "__main__":
    console = ConsoleManager()
    console.print_header("ARA AI - Simple Training System")
    
    print("\n1. Training Stock Models (AAPL, MSFT)...")
    train_stocks()
    
    print("\n2. Training Forex Models (EURUSD, GBPUSD)...")
    train_forex()
    
    print("\n3. Testing All Predictions...")
    test_predictions()
    
    console.print_success("\nAll training and testing completed!")
