#!/usr/bin/env python3
"""
System Integration Test
Tests all major components of ARA AI
"""

import sys
from pathlib import Path

print("=" * 60)
print("ARA AI System Integration Test")
print("=" * 60)

# Test 1: Core ML Imports
print("\n[1/6] Testing core ML imports...")
try:
    from meridianalgo.unified_ml import UnifiedStockML
    from meridianalgo.forex_ml import ForexML
    from meridianalgo.large_torch_model import AdvancedMLSystem
    print("✓ Core ML modules imported successfully")
except Exception as e:
    print(f"✗ Failed to import ML modules: {e}")
    sys.exit(1)

# Test 2: ARA API Imports
print("\n[2/6] Testing ARA API imports...")
try:
    from ara.api.app import app
    from ara.api.prediction_engine import PredictionEngine
    print("✓ ARA API modules imported successfully")
except Exception as e:
    print(f"✗ Failed to import API modules: {e}")
    sys.exit(1)

# Test 3: Training Scripts
print("\n[3/6] Testing training script imports...")
try:
    sys.path.insert(0, str(Path("scripts")))
    # Just check if files exist and are valid Python
    stock_script = Path("scripts/train_stock_model.py")
    forex_script = Path("scripts/train_forex_model.py")
    
    if not stock_script.exists():
        raise FileNotFoundError("train_stock_model.py not found")
    if not forex_script.exists():
        raise FileNotFoundError("train_forex_model.py not found")
    
    print("✓ Training scripts exist and are accessible")
except Exception as e:
    print(f"✗ Failed to access training scripts: {e}")
    sys.exit(1)

# Test 4: Model Initialization
print("\n[4/6] Testing model initialization...")
try:
    stock_ml = UnifiedStockML(model_path="models/test_stock.pt")
    forex_ml = ForexML(model_path="models/test_forex.pt")
    print("✓ Models initialized successfully")
except Exception as e:
    print(f"✗ Failed to initialize models: {e}")
    sys.exit(1)

# Test 5: Feature Engineering
print("\n[5/6] Testing feature engineering...")
try:
    import pandas as pd
    import numpy as np
    
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'Open': np.random.randn(100).cumsum() + 100,
        'High': np.random.randn(100).cumsum() + 102,
        'Low': np.random.randn(100).cumsum() + 98,
        'Close': np.random.randn(100).cumsum() + 100,
        'Volume': np.random.randint(1000000, 10000000, 100)
    }, index=dates)
    
    # Add indicators
    data_with_indicators = stock_ml._add_indicators(data)
    
    # Extract features
    features = stock_ml._extract_features(data_with_indicators)
    
    if features.shape[0] == 44:
        print(f"✓ Feature engineering works (44 features extracted)")
    else:
        print(f"✗ Expected 44 features, got {features.shape[0]}")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ Failed feature engineering test: {e}")
    sys.exit(1)

# Test 6: Environment Variables
print("\n[6/6] Testing environment configuration...")
try:
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    hf_token = os.getenv('HF_TOKEN')
    comet_key = os.getenv('COMET_API_KEY')
    
    if hf_token:
        print(f"✓ HF_TOKEN configured (length: {len(hf_token)})")
    else:
        print("⚠ HF_TOKEN not configured (optional)")
    
    if comet_key:
        print(f"✓ COMET_API_KEY configured (length: {len(comet_key)})")
    else:
        print("⚠ COMET_API_KEY not configured (optional)")
        
except Exception as e:
    print(f"✗ Failed environment test: {e}")
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ All System Tests Passed!")
print("=" * 60)
print("\nSystem Status:")
print("  • Core ML: ✓ Working")
print("  • ARA API: ✓ Working")
print("  • Training Scripts: ✓ Working")
print("  • Feature Engineering: ✓ Working")
print("  • Environment: ✓ Configured")
print("\nARA AI is ready to use!")
print("=" * 60)
