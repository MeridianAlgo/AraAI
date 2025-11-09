"""
Test both Quick Mode and Advanced Mode for stocks and forex
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meridianalgo.unified_ml import UnifiedStockML as UltimateStockML
from meridianalgo.forex_ml import ForexML
from meridianalgo.console import ConsoleManager

def test_quick_mode_stock(console):
    """Test Quick Mode for stocks"""
    console.print_header("Testing Quick Mode - Stocks")
    
    try:
        ml = UltimateStockML()
        console.print_info("Initialized UltimateStockML")
        
        # Check if models exist
        status = ml.get_model_status()
        console.print_info(f"Models loaded: {status['is_trained']}")
        console.print_info(f"Model path: {status['model_path']}")
        
        if not status['is_trained']:
            console.print_info("Training on AAPL (Quick Mode)...")
            success = ml.train_ultimate_models(target_symbol='AAPL', period='6mo')
            if success:
                console.print_success("Training successful!")
            else:
                console.print_error("Training failed!")
                return False
        
        # Make prediction
        console.print_info("Making prediction for AAPL...")
        result = ml.predict_ultimate('AAPL', days=3)
        
        if result and 'predictions' in result:
            console.print_success("Prediction successful!")
            print(f"Current Price: ${result['current_price']:.2f}")
            for pred in result['predictions']:
                print(f"  Day {pred['day']}: ${pred['predicted_price']:.2f} (Confidence: {pred['confidence']:.1%})")
            return True
        else:
            console.print_error("Prediction failed!")
            return False
            
    except Exception as e:
        console.print_error(f"Quick Mode Stock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_quick_mode_forex(console):
    """Test Quick Mode for forex"""
    console.print_header("Testing Quick Mode - Forex")
    
    try:
        forex = ForexML()
        console.print_info("Initialized ForexML")
        
        # Check if models exist
        status = forex.get_model_status()
        console.print_info(f"Models loaded: {status['is_trained']}")
        console.print_info(f"Model path: {status['model_path']}")
        
        if not status['is_trained']:
            console.print_info("Training on EURUSD (Quick Mode)...")
            success = forex.train_ultimate_models(target_symbol='EURUSD=X', period='1y')
            if success:
                console.print_success("Training successful!")
            else:
                console.print_error("Training failed!")
                return False
        
        # Make prediction
        console.print_info("Making prediction for EURUSD...")
        result = forex.predict_forex('EURUSD', days=3)
        
        if result and 'predictions' in result:
            console.print_success("Prediction successful!")
            print(f"Current Rate: {result['current_price']:.5f}")
            for pred in result['predictions']:
                print(f"  Day {pred['day']}: {pred['predicted_price']:.5f} (Pips: {pred['pips']:+.1f}, Confidence: {pred['confidence']:.1%})")
            return True
        else:
            console.print_error("Prediction failed!")
            return False
            
    except Exception as e:
        console.print_error(f"Quick Mode Forex test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_mode_stock(console):
    """Test Advanced Mode for stocks"""
    console.print_header("Testing Advanced Mode - Stocks")
    
    try:
        import yfinance as yf
        
        # Download dataset
        console.print_info("Downloading MSFT dataset...")
        ticker = yf.Ticker('MSFT')
        data = ticker.history(period='2y')
        
        datasets_dir = Path('datasets')
        datasets_dir.mkdir(exist_ok=True)
        csv_file = datasets_dir / 'MSFT_test.csv'
        data.to_csv(csv_file)
        console.print_success(f"Downloaded {len(data)} days of data")
        
        # Train from dataset
        ml = UltimateStockML(model_path='models/test_stock.pt')
        console.print_info("Training from dataset...")
        success = ml.train_from_dataset(str(csv_file), 'MSFT')
        
        if not success:
            console.print_error("Training failed!")
            return False
        
        console.print_success("Training successful!")
        
        # Make prediction
        console.print_info("Making prediction for MSFT...")
        result = ml.predict_ultimate('MSFT', days=3)
        
        if result and 'predictions' in result:
            console.print_success("Prediction successful!")
            print(f"Current Price: ${result['current_price']:.2f}")
            print(f"Trained on: {result.get('trained_on', 'Unknown')}")
            for pred in result['predictions']:
                print(f"  Day {pred['day']}: ${pred['predicted_price']:.2f} (Confidence: {pred['confidence']:.1%})")
            return True
        else:
            console.print_error("Prediction failed!")
            return False
            
    except Exception as e:
        console.print_error(f"Advanced Mode Stock test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_advanced_mode_forex(console):
    """Test Advanced Mode for forex"""
    console.print_header("Testing Advanced Mode - Forex")
    
    try:
        import yfinance as yf
        
        # Download dataset
        console.print_info("Downloading GBPUSD dataset...")
        ticker = yf.Ticker('GBPUSD=X')
        data = ticker.history(period='2y')
        
        datasets_dir = Path('datasets')
        datasets_dir.mkdir(exist_ok=True)
        csv_file = datasets_dir / 'GBPUSD_test.csv'
        data.to_csv(csv_file)
        console.print_success(f"Downloaded {len(data)} days of data")
        
        # Train from dataset
        forex = ForexML(model_path='models/test_forex.pt')
        console.print_info("Training from dataset...")
        success = forex.train_from_dataset(str(csv_file), 'GBPUSD')
        
        if not success:
            console.print_error("Training failed!")
            return False
        
        console.print_success("Training successful!")
        
        # Make prediction
        console.print_info("Making prediction for GBPUSD...")
        result = forex.predict_forex('GBPUSD', days=3)
        
        if result and 'predictions' in result:
            console.print_success("Prediction successful!")
            print(f"Current Rate: {result['current_price']:.5f}")
            for pred in result['predictions']:
                print(f"  Day {pred['day']}: {pred['predicted_price']:.5f} (Pips: {pred['pips']:+.1f}, Confidence: {pred['confidence']:.1%})")
            return True
        else:
            console.print_error("Prediction failed!")
            return False
            
    except Exception as e:
        console.print_error(f"Advanced Mode Forex test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    console = ConsoleManager()
    console.print_header("ARA AI - Testing Both Modes")
    
    results = {
        'Quick Mode - Stock': False,
        'Quick Mode - Forex': False,
        'Advanced Mode - Stock': False,
        'Advanced Mode - Forex': False
    }
    
    # Test all modes
    print("\n" + "="*70)
    results['Quick Mode - Stock'] = test_quick_mode_stock(console)
    
    print("\n" + "="*70)
    results['Quick Mode - Forex'] = test_quick_mode_forex(console)
    
    print("\n" + "="*70)
    results['Advanced Mode - Stock'] = test_advanced_mode_stock(console)
    
    print("\n" + "="*70)
    results['Advanced Mode - Forex'] = test_advanced_mode_forex(console)
    
    # Summary
    print("\n" + "="*70)
    console.print_header("Test Results Summary")
    
    for test_name, passed in results.items():
        if passed:
            console.print_success(f"{test_name}: PASSED")
        else:
            console.print_error(f"{test_name}: FAILED")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        console.print_success("ALL TESTS PASSED!")
        console.print_info("\nBoth Quick Mode and Advanced Mode are working correctly.")
        console.print_info("The system is ready for use!")
    else:
        console.print_warning("SOME TESTS FAILED")
        console.print_info("Please review the errors above.")
    
    return all_passed

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
