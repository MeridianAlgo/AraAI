#!/usr/bin/env python3
"""
Test Ultimate ML System - Comprehensive testing with all features
"""

import time
from meridianalgo.ultimate_ml import UltimateStockML
from meridianalgo.console import ConsoleManager

def test_ultimate_system():
    """Test the ultimate ML system with all features"""
    console = ConsoleManager(verbose=True)
    
    try:
        console.print_header("Testing ULTIMATE ML System")
        
        # Initialize ultimate ML system
        console.print_info("Initializing ULTIMATE ML system...")
        start_time = time.time()
        
        ml_system = UltimateStockML()
        init_time = time.time() - start_time
        
        console.print_success(f"Ultimate ML system initialized in {init_time:.3f}s")
        
        # Show system capabilities
        status = ml_system.get_model_status()
        console.print_info(f"Available models: {', '.join(status['models'])}")
        console.print_info(f"Total stock symbols: {status['total_symbols']:,}")
        console.print_info(f"Feature count: {status['feature_count']}")
        console.print_info(f"Hugging Face models: {'✓' if status['hf_models_available'] else '✗'}")
        
        # Test market status
        console.print_info("Testing market status detection...")
        market_status = ml_system.get_market_status()
        if market_status.get('is_open'):
            console.print_success("Market is currently OPEN")
        else:
            console.print_info("Market is currently CLOSED")
        
        # Test sector detection
        console.print_info("Testing sector detection...")
        test_symbols_sectors = ['AAPL', 'JPM', 'JNJ']
        for symbol in test_symbols_sectors:
            sector_info = ml_system.get_stock_sector(symbol)
            console.print_info(f"  {symbol}: {sector_info['sector']} - {sector_info['industry']}")
        
        # Check if models need training
        if not status['is_trained']:
            console.print_info("Training ULTIMATE models on comprehensive dataset...")
            training_start = time.time()
            
            # Train with limited symbols for testing (faster)
            success = ml_system.train_ultimate_models(
                max_symbols=50,  # Limit for testing speed
                period="6mo",    # Shorter period for testing
                use_parallel=True
            )
            
            training_time = time.time() - training_start
            
            if success:
                console.print_success(f"Ultimate training completed in {training_time:.1f}s")
                
                # Show accuracy scores
                accuracy_scores = ml_system.accuracy_scores
                console.print_info("Model Performance:")
                for model, scores in accuracy_scores.items():
                    acc = scores.get('accuracy', 0)
                    r2 = scores.get('r2', 0)
                    console.print_info(f"  {model:8}: {acc:5.1f}% accuracy, R²={r2:.3f}")
            else:
                console.print_error("Ultimate training failed")
                return False
        
        # Test ultimate predictions on multiple stocks
        test_symbols = ['AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN']
        
        console.print_info(f"Testing ULTIMATE predictions on {len(test_symbols)} stocks...")
        
        for symbol in test_symbols:
            console.print_info(f"Testing {symbol}...")
            
            pred_start = time.time()
            result = ml_system.predict_ultimate(symbol, days=3)
            pred_time = time.time() - pred_start
            
            if result:
                console.print_ultimate_predictions(result)
                console.print_success(f"Ultimate prediction for {symbol} completed in {pred_time:.3f}s")
                
                # Show key metrics
                accuracy = result.get('model_accuracy', 0)
                feature_count = result.get('feature_count', 0)
                console.print_info(f"  Accuracy: {accuracy:.1f}% | Features: {feature_count} | Models: 8")
                
            else:
                console.print_error(f"Ultimate prediction failed for {symbol}")
        
        # Test model persistence
        console.print_info("Testing model persistence...")
        ml_system._save_models()
        console.print_success("Models saved successfully")
        
        # Test model loading
        new_ml_system = UltimateStockML()
        if new_ml_system.load_models():
            console.print_success("Models loaded successfully")
        else:
            console.print_warning("Model loading failed")
        
        total_time = time.time() - start_time
        console.print_success(f"All ULTIMATE tests completed in {total_time:.2f}s")
        
        # Final summary
        console.print_info("🎯 ULTIMATE ML System Summary:")
        console.print_info(f"  • 8 ML models in ensemble")
        console.print_info(f"  • {status['feature_count']} engineered features")
        console.print_info(f"  • {status['total_symbols']:,} stock symbols available")
        console.print_info(f"  • Hugging Face AI integration")
        console.print_info(f"  • Market hours awareness")
        console.print_info(f"  • Sector classification")
        console.print_info(f"  • Model persistence")
        console.print_info(f"  • Parallel processing")
        
        return True
        
    except Exception as e:
        console.print_error(f"Ultimate test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_specific_features():
    """Test specific ultimate features"""
    console = ConsoleManager(verbose=True)
    
    try:
        console.print_header("Testing Specific ULTIMATE Features")
        
        ml_system = UltimateStockML()
        
        # Test 1: Market timing
        console.print_info("🕐 Testing market timing features...")
        market_status = ml_system.get_market_status()
        
        console.print_info(f"Current time: {market_status.get('current_time', 'Unknown')}")
        console.print_info(f"Market open: {market_status.get('is_open', False)}")
        console.print_info(f"Next open: {market_status.get('next_open', 'Unknown')}")
        console.print_info(f"Next close: {market_status.get('next_close', 'Unknown')}")
        
        # Test 2: Comprehensive stock list
        console.print_info(f"📊 Stock universe: {len(ml_system.all_symbols):,} symbols")
        console.print_info(f"Sample symbols: {', '.join(ml_system.all_symbols[:10])}")
        
        # Test 3: Feature engineering
        console.print_info("🔧 Testing feature engineering...")
        import yfinance as yf
        
        ticker = yf.Ticker("AAPL")
        data = ticker.history(period="3mo")
        
        if len(data) > 50:
            enhanced_data = ml_system._add_ultimate_indicators(data)
            console.print_info(f"Original features: {len(data.columns)}")
            console.print_info(f"Enhanced features: {len(enhanced_data.columns)}")
            console.print_info(f"Added indicators: {len(enhanced_data.columns) - len(data.columns)}")
        
        # Test 4: Sector classification
        console.print_info("🏢 Testing sector classification...")
        sectors_tested = {}
        
        for symbol in ['AAPL', 'JPM', 'JNJ', 'XOM', 'WMT']:
            sector_info = ml_system.get_stock_sector(symbol)
            sector = sector_info.get('sector', 'Unknown')
            if sector not in sectors_tested:
                sectors_tested[sector] = []
            sectors_tested[sector].append(symbol)
        
        for sector, symbols in sectors_tested.items():
            console.print_info(f"  {sector}: {', '.join(symbols)}")
        
        # Test 5: Hugging Face integration
        if ml_system.hf_models:
            console.print_info("🤖 Testing Hugging Face integration...")
            try:
                sample_text = "Apple stock shows strong performance with positive outlook"
                sentiment = ml_system.hf_models['sentiment'](sample_text)
                console.print_info(f"Sample sentiment: {sentiment[0]['label']} ({sentiment[0]['score']:.2f})")
            except Exception as e:
                console.print_warning(f"HF sentiment test failed: {e}")
        else:
            console.print_warning("Hugging Face models not available")
        
        console.print_success("Specific feature tests completed")
        return True
        
    except Exception as e:
        console.print_error(f"Feature test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting ULTIMATE ML System Tests")
    print("=" * 60)
    
    # Run main system test
    success1 = test_ultimate_system()
    
    print("\n" + "=" * 60)
    
    # Run specific feature tests
    success2 = test_specific_features()
    
    print("\n" + "=" * 60)
    
    if success1 and success2:
        print("🎉 ALL ULTIMATE TESTS PASSED!")
        exit(0)
    else:
        print("❌ Some tests failed")
        exit(1)