"""
ARA AI v3.0.0 - Simple Runner
Run stock predictions with the Ultimate ML system
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meridianalgo.unified_ml import UnifiedStockML
from meridianalgo.console import ConsoleManager

def main():
    parser = argparse.ArgumentParser(description='ARA AI - Stock Prediction System')
    parser.add_argument('symbol', nargs='?', default='AAPL', help='Stock symbol (default: AAPL)')
    parser.add_argument('--days', type=int, default=5, help='Number of days to predict (default: 5)')
    parser.add_argument('--train', action='store_true', help='Force model training')
    parser.add_argument('--stocks', type=int, default=1, help='Ignored - always trains on target stock only')
    parser.add_argument('--period', default='2y', help='Training period (default: 2y)')
    
    args = parser.parse_args()
    
    console = ConsoleManager()
    
    # Print header
    console.print_header(f"ARA AI v3.0.0 - {args.symbol} Analysis")
    
    # Initialize system
    console.print_info("Initializing Ultimate ML System...")
    ml = UnifiedStockML()
    
    # Check if models are trained
    status = ml.get_model_status()
    
    if not status['is_trained'] or args.train:
        console.print_warning(f"Training models on {args.symbol} with maximum historical data ({args.period} period)...")
        console.print_info("This may take 1-2 minutes...")
        
        try:
            success = ml.train_ultimate_models(
                period=args.period,
                use_parallel=False,
                target_symbol=args.symbol  # Train ONLY on target symbol with all historical data
            )
            
            if success:
                console.print_success("Training completed!")
            else:
                console.print_warning("Training completed with warnings")
        except Exception as e:
            console.print_error(f"Training failed: {e}")
            console.print_info("Attempting to use pre-trained models...")
    else:
        console.print_success("Using pre-trained models")
    
    # Make prediction
    console.print_info(f"Predicting {args.symbol} for next {args.days} days...")
    
    try:
        result = ml.predict_ultimate(args.symbol, days=args.days)
        
        if result and 'predictions' in result:
            # Display results
            print("\n" + "=" * 70)
            print(f"{args.symbol} - Stock Prediction Results")
            print("=" * 70)
            
            print(f"\nCurrent Price: ${result.get('current_price', 0):.2f}")
            if 'ensemble_accuracy' in result:
                print(f"Ensemble Accuracy: {result['ensemble_accuracy']:.1f}%")
            
            # Financial health
            if 'financial_health' in result:
                health = result['financial_health']
                grade = health['health_grade']
                score = health['health_score']
                risk = health['risk_grade']
                
                print(f"\nFinancial Health: {grade} (Score: {score:.1f}/100)")
                print(f"Risk Grade: {risk}")
            
            # Sector info
            if 'sector' in result:
                print(f"Sector: {result['sector']}")
            
            # Predictions
            print(f"\n{args.days}-Day Forecast:")
            print("-" * 70)
            print(f"{'Date':<12} {'Price':<12} {'Change':<12} {'Confidence':<12}")
            print("-" * 70)
            
            for pred in result['predictions']:
                date = pred['date']
                price = pred['predicted_price']
                pred_return = pred.get('predicted_return', 0) * 100  # Convert to percentage
                conf = pred['confidence']
                
                # Format change
                if pred_return > 0:
                    change_str = f"+{pred_return:.2f}%"
                elif pred_return < 0:
                    change_str = f"{pred_return:.2f}%"
                else:
                    change_str = f"{pred_return:.2f}%"
                
                print(f"{date:<12} ${price:<10.2f} {change_str:<12} {conf:.1%}")
            
            print("-" * 70)
            
            # Summary
            avg_change = sum(p.get('predicted_return', 0) * 100 for p in result['predictions']) / len(result['predictions'])
            final_price = result['predictions'][-1]['predicted_price']
            total_change = ((final_price - result.get('current_price', final_price)) / result.get('current_price', final_price)) * 100
            
            print(f"\nSummary:")
            print(f"   Average Daily Change: {avg_change:+.2f}%")
            print(f"   Final Predicted Price: ${final_price:.2f}")
            print(f"   Total Change: {total_change:+.2f}%")
            
            # Recommendation
            if total_change > 5:
                print(f"\nOutlook: Strong Bullish")
            elif total_change > 2:
                print(f"\nOutlook: Bullish")
            elif total_change > -2:
                print(f"\nOutlook: Neutral")
            elif total_change > -5:
                print(f"\nOutlook: Bearish")
            else:
                print(f"\nOutlook: Strong Bearish")
            
            print("\n" + "=" * 70)
            console.print_success("Prediction completed successfully!")
            
        else:
            console.print_error("No prediction results available")
            console.print_info("Try running with --train flag to retrain models")
            
    except Exception as e:
        console.print_error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
