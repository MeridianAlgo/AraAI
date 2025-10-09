#!/usr/bin/env python3
"""
Ara AI Real-Time ML Mode - High-accuracy stock predictions
Primary ML models trained on real market data
"""

import sys
import argparse
import time

def main():
    """Real-time ML mode entry point"""
    parser = argparse.ArgumentParser(
        description="Ara AI Real-Time ML - High-accuracy stock predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('symbol', help='Stock symbol (e.g., AAPL, TSLA)')
    parser.add_argument('--days', '-d', type=int, default=5, help='Days to predict (default: 5)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--retrain', action='store_true', help='Force model retraining')
    parser.add_argument('--period', default='1y', help='Training period (1y, 2y, 5y)')
    
    args = parser.parse_args()
    
    try:
        from meridianalgo.ultimate_ml import UltimateStockML
        from meridianalgo.console import ConsoleManager
        
        console = ConsoleManager(verbose=args.verbose)
        console.print_header(f"ARA ULTIMATE ML Analysis - {args.symbol.upper()}")
        
        # Initialize ultimate ML system
        start_time = time.time()
        ml_system = UltimateStockML()
        init_time = time.time() - start_time
        
        if args.verbose:
            console.print_success(f"Ultimate ML system initialized in {init_time:.3f}s")
        
        # Check training status
        status = ml_system.get_model_status()
        
        if not status['is_trained'] or args.retrain:
            console.print_info("Training ULTIMATE ML models on ALL market data...")
            training_start = time.time()
            
            # Train with ALL available stocks
            max_symbols = None if args.retrain else 200  # Limit for speed unless forced retrain
            success = ml_system.train_ultimate_models(
                max_symbols=max_symbols,
                period=args.period,
                use_parallel=True
            )
            training_time = time.time() - training_start
            
            if success:
                console.print_success(f"Ultimate models trained in {training_time:.1f}s")
                
                # Show accuracy scores
                if args.verbose:
                    accuracy_scores = ml_system.accuracy_scores
                    for model, scores in accuracy_scores.items():
                        acc = scores.get('accuracy', 0)
                        console.print_info(f"  {model}: {acc:.1f}% accuracy")
            else:
                console.print_error("Ultimate training failed")
                return 1
        
        # Generate ultimate predictions
        console.print_info(f"Generating ULTIMATE predictions for {args.symbol.upper()}...")
        
        pred_start = time.time()
        result = ml_system.predict_ultimate(args.symbol, days=args.days)
        pred_time = time.time() - pred_start
        
        if result:
            # Display ultimate results
            console.print_ultimate_predictions(result)
            
            total_time = time.time() - start_time
            console.print_success(f"Ultimate analysis completed in {total_time:.2f}s (prediction: {pred_time:.3f}s)")
            
            # Show comprehensive performance
            accuracy = result.get('model_accuracy', 0)
            feature_count = result.get('feature_count', 0)
            console.print_info(f"🎯 Model Accuracy: {accuracy:.1f}% | Features: {feature_count} | Models: 8")
            
            # Show market status
            market_status = result.get('market_status', {})
            if market_status.get('is_open'):
                console.print_info("📈 Market is currently OPEN")
            else:
                console.print_info("📉 Market is currently CLOSED")
            
            # Show sector info
            sector_info = result.get('sector_info', {})
            if sector_info.get('sector') != 'Unknown':
                console.print_info(f"🏢 Sector: {sector_info['sector']} | Industry: {sector_info['industry']}")
            
        else:
            console.print_error(f"Ultimate prediction failed for {args.symbol}")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Cancelled by user")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())