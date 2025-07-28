#!/usr/bin/env python3
"""
Test script for the newly organized MeridianAlgo v0.3.1 package
"""

print("🧪 Testing MeridianAlgo v0.3.1 - Organized Package Structure")
print("=" * 60)

try:
    # Test main imports (backward compatibility)
    print("\n1️⃣ Testing main imports (backward compatibility):")
    from meridianalgo import MLPredictor, Indicators, EnsembleModels, AIAnalyzer, TradingEngine, BacktestEngine
    print("✅ All main classes imported successfully")
    
    # Test organized submodule imports
    print("\n2️⃣ Testing organized submodule imports:")
    
    # Prediction module
    from meridianalgo.prediction import MLPredictor as PredMLPredictor, EnsembleModels as PredEnsemble
    print("✅ Prediction module: MLPredictor, EnsembleModels")
    
    # Analysis module  
    from meridianalgo.analysis import Indicators as AnalysisIndicators, AIAnalyzer as AnalysisAI
    print("✅ Analysis module: Indicators, AIAnalyzer")
    
    # Trading module
    from meridianalgo.trading import TradingEngine as TradingEng, BacktestEngine as BacktestEng
    print("✅ Trading module: TradingEngine, BacktestEngine")
    
    # Test initialization
    print("\n3️⃣ Testing class initialization:")
    
    # Test MLPredictor
    predictor = MLPredictor()
    print("✅ MLPredictor initialized")
    
    # Test Indicators
    indicators = Indicators()
    print("✅ Indicators initialized")
    
    # Test AIAnalyzer (no API keys required)
    ai_analyzer = AIAnalyzer()
    print("✅ AIAnalyzer initialized (no API keys required)")
    
    # Test version
    print("\n4️⃣ Testing package metadata:")
    import meridianalgo
    print(f"✅ Package version: {meridianalgo.__version__}")
    print(f"✅ Author: {meridianalgo.__author__}")
    
    # Test package structure
    print("\n5️⃣ Testing package structure:")
    print(f"✅ Available modules: {[m for m in dir(meridianalgo) if not m.startswith('_')]}")
    
    print("\n🎉 ALL TESTS PASSED!")
    print("📦 MeridianAlgo v0.3.1 is properly organized and functional")
    print("🆓 No API keys required - uses Yahoo Finance")
    print("⚡ Ready for advanced stock prediction!")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")

print("\n" + "=" * 60)
print("🚀 Package Organization Summary:")
print("📁 meridianalgo/")
print("  ├── prediction/     # ML models and ensemble methods")
print("  ├── analysis/       # Technical indicators and AI analysis")
print("  ├── trading/        # Trading engines and backtesting")
print("  └── utils.py        # Utility functions")
print("\n💡 Usage Examples:")
print("from meridianalgo import MLPredictor")
print("from meridianalgo.prediction import EnsembleModels")
print("from meridianalgo.analysis import Indicators")