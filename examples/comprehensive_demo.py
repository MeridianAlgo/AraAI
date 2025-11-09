#!/usr/bin/env python3
"""
Comprehensive Ara AI Demo
Demonstrates all advanced features including self-learning ML, pattern recognition, and AI analysis
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from meridianalgo.core import AraAI
from meridianalgo.ai_analysis import LightweightAIAnalyzer
from meridianalgo.advanced_ml import ChartPatternRecognizer
from meridianalgo.console import ConsoleManager
import time

def demo_basic_prediction():
    """Demo basic stock prediction"""
    print("\n" + "="*60)
    print(" BASIC STOCK PREDICTION DEMO")
    print("="*60)
    
    console = ConsoleManager(verbose=True)
    ara = AraAI(verbose=True)
    
    # Test symbols
    symbols = ['AAPL', 'TSLA', 'NVDA']
    
    for symbol in symbols:
        try:
            console.print_info(f"Analyzing {symbol}...")
            result = ara.predict(symbol, days=5, include_analysis=False)
            
            if result:
                console.print_prediction_results(result)
            else:
                console.print_error(f"Failed to analyze {symbol}")
                
            time.sleep(1)  # Brief pause between analyses
            
        except Exception as e:
            console.print_error(f"Error analyzing {symbol}: {e}")

def demo_advanced_analysis():
    """Demo advanced analysis with AI insights"""
    print("\n" + "="*60)
    print(" ADVANCED AI ANALYSIS DEMO")
    print("="*60)
    
    console = ConsoleManager(verbose=True)
    ara = AraAI(verbose=True)
    
    # Test with comprehensive analysis
    symbol = 'MSFT'
    
    try:
        console.print_info(f"Performing comprehensive AI analysis for {symbol}...")
        result = ara.predict(symbol, days=7, include_analysis=True)
        
        if result:
            console.print_prediction_results(result)
            
            # Show additional insights if available
            if 'company_analysis' in result:
                company_analysis = result['company_analysis']
                if 'ai_insights' in company_analysis:
                    console.print_info("AI Insights:")
                    for insight in company_analysis['ai_insights']:
                        print(f"   {insight}")
        else:
            console.print_error(f"Failed to analyze {symbol}")
            
    except Exception as e:
        console.print_error(f"Advanced analysis error: {e}")

def demo_pattern_recognition():
    """Demo chart pattern recognition"""
    print("\n" + "="*60)
    print(" CHART PATTERN RECOGNITION DEMO")
    print("="*60)
    
    console = ConsoleManager(verbose=True)
    ara = AraAI(verbose=True)
    
    symbol = 'GOOGL'
    
    try:
        console.print_info(f"Analyzing chart patterns for {symbol}...")
        
        # Get market data
        data = ara.data_manager.get_stock_data(symbol)
        prices = data['Close'].values
        
        # Analyze patterns
        pattern_recognizer = ChartPatternRecognizer()
        
        triangles = pattern_recognizer.detect_triangles(prices)
        wedges = pattern_recognizer.detect_wedges(prices)
        hs_patterns = pattern_recognizer.detect_head_and_shoulders(prices)
        double_patterns = pattern_recognizer.detect_double_patterns(prices)
        
        all_patterns = triangles + wedges + hs_patterns + double_patterns
        
        if all_patterns:
            console.print_info(f"Found {len(all_patterns)} chart patterns:")
            
            for i, pattern in enumerate(all_patterns[:5]):  # Show top 5
                pattern_type = pattern['type'].replace('_', ' ').title()
                direction = pattern['breakout_direction'].upper()
                confidence = pattern['confidence'] * 100
                
                direction_color = "green" if direction == "BULLISH" else "red" if direction == "BEARISH" else "yellow"
                print(f"  {i+1}. {pattern_type} - [{direction_color}]{direction}[/] ({confidence:.0f}% confidence)")
        else:
            console.print_warning("No significant patterns detected")
            
    except Exception as e:
        console.print_error(f"Pattern recognition error: {e}")

def demo_ai_company_analysis():
    """Demo AI-powered company analysis"""
    print("\n" + "="*60)
    print(" AI COMPANY ANALYSIS DEMO")
    print("="*60)
    
    console = ConsoleManager(verbose=True)
    
    try:
        # Initialize AI analyzer
        ai_analyzer = LightweightAIAnalyzer(use_gpu=False)
        
        symbol = 'AMZN'
        console.print_info(f"Performing AI analysis for {symbol}...")
        
        # Perform AI analysis
        analysis = ai_analyzer.analyze_company_with_ai(symbol)
        
        if analysis and 'error' not in analysis:
            console.print_ai_analysis(analysis)
        else:
            error_msg = analysis.get('error', 'Unknown error') if analysis else 'Analysis failed'
            console.print_error(f"AI analysis failed: {error_msg}")
            
    except Exception as e:
        console.print_error(f"AI analysis error: {e}")

def demo_accuracy_tracking():
    """Demo accuracy tracking system"""
    print("\n" + "="*60)
    print(" ACCURACY TRACKING DEMO")
    print("="*60)
    
    console = ConsoleManager(verbose=True)
    ara = AraAI(verbose=True)
    
    try:
        console.print_info("Validating previous predictions...")
        
        # Validate predictions
        validation_result = ara.validate_predictions()
        
        if validation_result:
            console.print_validation_summary(validation_result)
        else:
            console.print_warning("No predictions available for validation")
        
        # Show accuracy for specific symbols
        test_symbols = ['AAPL', 'TSLA', 'MSFT']
        
        for symbol in test_symbols:
            accuracy_stats = ara.analyze_accuracy(symbol)
            if accuracy_stats.get('total_predictions', 0) > 0:
                console.print_accuracy_summary(accuracy_stats)
            
    except Exception as e:
        console.print_error(f"Accuracy tracking error: {e}")

def demo_gpu_acceleration():
    """Demo GPU acceleration capabilities"""
    print("\n" + "="*60)
    print(" GPU ACCELERATION DEMO")
    print("="*60)
    
    console = ConsoleManager(verbose=True)
    ara = AraAI(verbose=True)
    
    try:
        # Show GPU information
        gpu_info = ara.gpu_manager.get_device_info()
        console.print_gpu_info(gpu_info)
        
        # Show system information
        system_info = ara.get_system_info()
        
        console.print_info("System Information:")
        print(f"    Device: {system_info.get('device', 'Unknown')}")
        print(f"   GPU Vendor: {gpu_info.get('vendor', 'Unknown')}")
        print(f"   GPU Available: {'Yes' if gpu_info.get('available') else 'No'}")
        
        if gpu_info.get('details'):
            print("   GPU Details:")
            for detail in gpu_info['details']:
                print(f"    • {detail}")
        
    except Exception as e:
        console.print_error(f"GPU demo error: {e}")

def demo_caching_system():
    """Demo intelligent caching system"""
    print("\n" + "="*60)
    print(" INTELLIGENT CACHING DEMO")
    print("="*60)
    
    console = ConsoleManager(verbose=True)
    ara = AraAI(verbose=True)
    
    try:
        # Show cache statistics
        cache_stats = ara.cache_manager.get_cache_stats()
        
        console.print_info("Cache Statistics:")
        print(f"   Total Predictions Cached: {cache_stats.get('total_predictions', 0)}")
        print(f"   Symbols Cached: {cache_stats.get('symbols', 0)}")
        print(f"   Cache File Size: {cache_stats.get('file_size', 0)} bytes")
        print(f"   Market Data Files: {cache_stats.get('market_data_files', 0)}")
        
        # Test caching with a prediction
        symbol = 'AAPL'
        console.print_info(f"Testing cache with {symbol}...")
        
        # First prediction (will be cached)
        start_time = time.time()
        result1 = ara.predict(symbol, days=3, use_cache=True)
        time1 = time.time() - start_time
        
        # Second prediction (should use cache)
        start_time = time.time()
        result2 = ara.predict(symbol, days=3, use_cache=True)
        time2 = time.time() - start_time
        
        console.print_info(f"First prediction: {time1:.2f}s")
        console.print_info(f"Second prediction: {time2:.2f}s")
        
        if time2 < time1 * 0.5:
            console.print_success("Cache is working efficiently!")
        else:
            console.print_warning("Cache may not be working optimally")
        
    except Exception as e:
        console.print_error(f"Caching demo error: {e}")

def demo_self_learning():
    """Demo self-learning capabilities"""
    print("\n" + "="*60)
    print(" SELF-LEARNING SYSTEM DEMO")
    print("="*60)
    
    console = ConsoleManager(verbose=True)
    ara = AraAI(verbose=True)
    
    try:
        # Get learning insights
        if hasattr(ara.ml_system, 'get_learning_insights'):
            insights = ara.ml_system.get_learning_insights()
            
            if insights:
                console.print_info("Self-Learning Insights:")
                
                if 'ensemble' in insights:
                    ensemble_info = insights['ensemble']
                    print(f"   Total Predictions: {ensemble_info.get('total_predictions', 0)}")
                    print(f"   Average Error: {ensemble_info.get('avg_error', 0):.3f}")
                    print(f"   Error Trend: {ensemble_info.get('error_trend', 0):.6f}")
                    
                    weights = ensemble_info.get('current_weights', {})
                    print("    Current Model Weights:")
                    for model, weight in weights.items():
                        print(f"    • {model.upper()}: {weight:.3f}")
                
                if 'lstm' in insights:
                    lstm_info = insights['lstm']
                    print(f"   LSTM Average Error: {lstm_info.get('avg_error', 0):.3f}")
                    print(f"   LSTM Improvement Rate: {lstm_info.get('improvement_rate', 0):.6f}")
            else:
                console.print_warning("No learning insights available yet")
        else:
            console.print_warning("Self-learning system not fully initialized")
        
    except Exception as e:
        console.print_error(f"Self-learning demo error: {e}")

def main():
    """Run comprehensive demo"""
    print(" ARA AI COMPREHENSIVE DEMO")
    print("Advanced Stock Analysis with Self-Learning AI")
    print("=" * 60)
    
    demos = [
        ("Basic Prediction", demo_basic_prediction),
        ("Advanced AI Analysis", demo_advanced_analysis),
        ("Pattern Recognition", demo_pattern_recognition),
        ("AI Company Analysis", demo_ai_company_analysis),
        ("GPU Acceleration", demo_gpu_acceleration),
        ("Caching System", demo_caching_system),
        ("Accuracy Tracking", demo_accuracy_tracking),
        ("Self-Learning", demo_self_learning)
    ]
    
    print("\nAvailable Demos:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all demos...")
    
    for name, demo_func in demos:
        try:
            print(f"\n Starting: {name}")
            demo_func()
            print(f" Completed: {name}")
        except KeyboardInterrupt:
            print(f"\n⏹  Demo interrupted by user")
            break
        except Exception as e:
            print(f" Error in {name}: {e}")
            continue
    
    print("\n" + "="*60)
    print(" DEMO COMPLETED!")
    print("="*60)
    print("\nFor more information:")
    print("• Check README.md for detailed documentation")
    print("• Visit examples/ directory for more examples")
    print("• Run 'python ara.py --help' for CLI options")

if __name__ == "__main__":
    main()