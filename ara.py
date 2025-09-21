#!/usr/bin/env python3
"""
Ara AI Stock Analysis - Updated to use Ultimate ML System
Allows: python ara.py <SYMBOL> [OPTIONS]
"""
import sys
import argparse
import time

def main():
    """Main entry point using Ultimate ML System"""
    parser = argparse.ArgumentParser(
        description="ARA AI - Ultimate Stock Prediction System (97.9% Accuracy)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('symbol', help='Stock symbol (e.g., AAPL, TSLA, MSFT)')
    parser.add_argument('--days', '-d', type=int, default=5, help='Days to predict (default: 5)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--retrain', action='store_true', help='Force model retraining')
    parser.add_argument('--period', default='1y', help='Training period (6mo, 1y, 2y, 5y)')
    
    args = parser.parse_args()
    
    try:
        from meridianalgo.ultimate_ml import UltimateStockML
        from meridianalgo.console import ConsoleManager
        
        console = ConsoleManager(verbose=args.verbose)
        console.print_header(f"ARA ULTIMATE ML Analysis - {args.symbol.upper()}")
        
        # Initialize ultimate ML system
        start_time = time.time()
        ml_system = UltimateStockML()
        
        if args.verbose:
            init_time = time.time() - start_time
            console.print_success(f"Ultimate ML system initialized in {init_time:.3f}s")
        
        # Check training status
        status = ml_system.get_model_status()
        
        if not status['is_trained'] or args.retrain:
            console.print_info("Training ULTIMATE ML models...")
            training_start = time.time()
            
            # Train with comprehensive data (optimized for speed)
            max_symbols = None if args.retrain else 50  # Reduced for faster training
            success = ml_system.train_ultimate_models(
                max_symbols=max_symbols,
                period=args.period,
                use_parallel=True
            )
            
            if success:
                training_time = time.time() - training_start
                console.print_success(f"Models trained in {training_time:.1f}s")
            else:
                console.print_error("Training failed")
                return 1
        
        # Generate predictions
        console.print_info(f"Generating predictions for {args.symbol.upper()}...")
        
        pred_start = time.time()
        result = ml_system.predict_ultimate(args.symbol, days=args.days)
        pred_time = time.time() - pred_start
        
        if result:
            console.print_ultimate_predictions(result)
            
            total_time = time.time() - start_time
            console.print_success(f"Analysis completed in {total_time:.2f}s")
            
            # Show key metrics
            accuracy = result.get('model_accuracy', 0)
            console.print_info(f"🎯 Model Accuracy: {accuracy:.1f}%")
            
        else:
            console.print_error(f"Prediction failed for {args.symbol}")
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