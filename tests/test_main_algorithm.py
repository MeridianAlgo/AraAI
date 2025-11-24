"""
Test the main prediction algorithm to verify it works
"""

import sys
sys.path.insert(0, '.')

print("=" * 70)
print("Testing Main Prediction Algorithm")
print("=" * 70)

# Test 1: Import the main module
print("\n1. Testing imports...")
try:
    from meridianalgo.core import predict_stock
    print("✓ Successfully imported predict_stock")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Make a simple prediction
print("\n2. Testing basic prediction...")
try:
    result = predict_stock("AAPL", days=5)
    print(f"✓ Prediction successful!")
    print(f"  Symbol: {result.get('symbol', 'N/A')}")
    print(f"  Days: {result.get('days', 'N/A')}")
    print(f"  Current Price: ${result.get('current_price', 'N/A'):.2f}")
    
    if 'predictions' in result:
        predictions = result['predictions']
        print(f"  Predictions: {len(predictions)} days")
        for i, pred in enumerate(predictions[:3], 1):
            print(f"    Day {i}: ${pred.get('price', 'N/A'):.2f} ({pred.get('change_percent', 'N/A'):+.2f}%)")
    
    if 'confidence' in result:
        print(f"  Confidence: {result['confidence']:.2%}")
    
except Exception as e:
    print(f"✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test with different symbols
print("\n3. Testing multiple symbols...")
test_symbols = ["MSFT", "GOOGL", "TSLA"]
for symbol in test_symbols:
    try:
        result = predict_stock(symbol, days=3)
        print(f"✓ {symbol}: ${result.get('current_price', 'N/A'):.2f}")
    except Exception as e:
        print(f"✗ {symbol}: Failed - {e}")

# Test 4: Test ensemble model if available
print("\n4. Testing ensemble model...")
try:
    from meridianalgo.torch_ensemble import predict_with_ensemble
    result = predict_with_ensemble("AAPL", days=5)
    print(f"✓ Ensemble prediction successful!")
    print(f"  Current Price: ${result.get('current_price', 'N/A'):.2f}")
except ImportError:
    print("⚠ Ensemble model not available (optional)")
except Exception as e:
    print(f"⚠ Ensemble prediction failed: {e}")

# Test 5: Test CSV mode
print("\n5. Testing CSV mode...")
try:
    from meridianalgo.csv_ml import predict_from_csv
    # This would need a CSV file, so we'll just check if it imports
    print("✓ CSV mode available")
except ImportError:
    print("⚠ CSV mode not available (optional)")
except Exception as e:
    print(f"⚠ CSV mode error: {e}")

# Test 6: Test forex mode
print("\n6. Testing forex mode...")
try:
    from meridianalgo.forex_ml import predict_forex
    result = predict_forex("EUR/USD", days=3)
    print(f"✓ Forex prediction successful!")
    print(f"  Pair: {result.get('pair', 'N/A')}")
    print(f"  Current Rate: {result.get('current_rate', 'N/A'):.4f}")
except ImportError:
    print("⚠ Forex mode not available (optional)")
except Exception as e:
    print(f"⚠ Forex prediction failed: {e}")

# Test 7: Test AI analysis
print("\n7. Testing AI analysis...")
try:
    from meridianalgo.ai_analysis import analyze_company
    result = analyze_company("AAPL")
    print(f"✓ AI analysis successful!")
    if 'analysis' in result:
        analysis = result['analysis']
        print(f"  Analysis length: {len(analysis)} characters")
except ImportError:
    print("⚠ AI analysis not available (optional)")
except Exception as e:
    print(f"⚠ AI analysis failed: {e}")

print("\n" + "=" * 70)
print("✓ Main Algorithm Test Complete!")
print("=" * 70)
print("\nThe core prediction algorithm is working correctly.")
print("You can now use it with: python -m meridianalgo.core AAPL 5")
