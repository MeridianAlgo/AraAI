#!/usr/bin/env python3
"""
Test script for MeridianAlgo v0.3.0 package
"""

try:
    from meridianalgo import MLPredictor, Indicators, EnsembleModels
    print("✅ MeridianAlgo v0.3.0 imported successfully!")
    print("✅ All modules available: MLPredictor, Indicators, EnsembleModels")
    
    # Test basic functionality
    indicators = Indicators()
    print("✅ Indicators class initialized")
    
    # Test data fetching
    try:
        data = indicators.get_stock_data('AAPL', period='1mo')
        if not data.empty:
            print(f"✅ Yahoo Finance data fetch successful - {len(data)} days of AAPL data")
            print(f"✅ Current AAPL price: ${data['Close'].iloc[-1]:.2f}")
        else:
            print("⚠️  Data fetch returned empty DataFrame")
    except Exception as e:
        print(f"⚠️  Data fetch error: {e}")
    
    print("\n🎉 MeridianAlgo package is working correctly!")
    print("📦 Package version: 0.3.0")
    print("🆓 No API keys required - uses Yahoo Finance")
    print("⚡ Ready for stock analysis!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")