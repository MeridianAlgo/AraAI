#!/usr/bin/env python3
"""
Test script for MeridianAlgo Smart Trader package
"""

def test_imports():
    """Test that all imports work correctly"""
    try:
        from meridianalgo.smarttrader import SmartTrader, analyze_stock, detect_volatility_spikes
        from meridianalgo.smarttrader.gpu import detect_all_gpus, get_best_device
        from meridianalgo.smarttrader.analysis import calculate_technical_indicators
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_gpu_detection():
    """Test GPU detection"""
    try:
        from meridianalgo.smarttrader.gpu import detect_all_gpus, get_best_device
        
        gpu_info = detect_all_gpus()
        device, device_name = get_best_device()
        
        print(f"✅ GPU detection successful")
        print(f"   Device: {device_name}")
        print(f"   GPU Support: {gpu_info}")
        return True
    except Exception as e:
        print(f"❌ GPU detection failed: {e}")
        return False

def test_analysis():
    """Test stock analysis"""
    try:
        from meridianalgo.smarttrader import analyze_stock
        
        print("🚀 Testing stock analysis...")
        result = analyze_stock('AAPL', days=30, epochs=3, verbose=False)
        
        if 'error' in result:
            print(f"❌ Analysis failed: {result['error']}")
            return False
        
        print("✅ Analysis successful")
        print(f"   Symbol: {result['symbol']}")
        print(f"   Current Price: ${result['current_price']:.2f}")
        print(f"   Confidence: {result['confidence']:.1f}%")
        print(f"   Vol Spike Risk: {result['volatility_spike']['spike_probability']:.1f}%")
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return False

def test_volatility_detection():
    """Test volatility spike detection"""
    try:
        import yfinance as yf
        from meridianalgo.smarttrader.analysis import detect_volatility_spikes
        
        print("⚡ Testing volatility spike detection...")
        
        # Get sample data
        ticker = yf.Ticker('AAPL')
        data = ticker.history(period='3mo')
        
        vol_info = detect_volatility_spikes(data)
        
        print("✅ Volatility detection successful")
        print(f"   Spike Probability: {vol_info['spike_probability']:.1f}%")
        print(f"   Risk Level: {vol_info['risk_level']}")
        return True
        
    except Exception as e:
        print(f"❌ Volatility detection failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing MeridianAlgo Smart Trader Package\n")
    
    tests = [
        ("Import Test", test_imports),
        ("GPU Detection", test_gpu_detection),
        ("Stock Analysis", test_analysis),
        ("Volatility Detection", test_volatility_detection)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}:")
        if test_func():
            passed += 1
        print("-" * 50)
    
    print(f"\n🎯 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Package is ready for deployment.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)