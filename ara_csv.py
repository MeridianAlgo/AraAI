"""
ARA AI CSV - Custom Data Prediction
Train and predict on your own CSV data files
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from meridianalgo.csv_ml import CSVML
from meridianalgo.console import ConsoleManager

def main():
    parser = argparse.ArgumentParser(description='ARA AI CSV - Custom Data Prediction')
    parser.add_argument('csv_file', help='Path to CSV file')
    parser.add_argument('--days', type=int, default=5, help='Number of days to predict (default: 5)')
    parser.add_argument('--type', choices=['stock', 'forex', 'auto'], default='auto', 
                       help='Data type: stock, forex, or auto-detect (default: auto)')
    parser.add_argument('--name', help='Custom name for the data (default: auto-generated)')
    parser.add_argument('--train', action='store_true', help='Force model training')
    
    args = parser.parse_args()
    
    console = ConsoleManager()
    
    # Check if file exists
    if not Path(args.csv_file).exists():
        console.print_error(f"CSV file not found: {args.csv_file}")
        print("\nExpected CSV format:")
        print("Date,Open,High,Low,Close,Volume")
        print("2023-01-01,100.0,105.0,99.0,104.0,1000000")
        print("2023-01-02,104.0,106.0,103.0,105.0,1200000")
        print("...")
        print("\nNote: Volume column is optional")
        return
    
    # Print header
    console.print_header(f"ARA AI CSV v3.0.2 - Custom Data Analysis")
    
    # Initialize system
    console.print_info("Initializing CSV ML System...")
    csv_ml = CSVML()
    
    # Load CSV data
    console.print_info(f"Loading CSV data from: {args.csv_file}")
    
    if not csv_ml.load_csv_data(args.csv_file, args.type, args.name):
        console.print_error("Failed to load CSV data")
        return
    
    console.print_success(f"Loaded {csv_ml.symbol_name} ({csv_ml.data_type} data)")
    
    # Check if models are trained
    status = csv_ml.get_model_status()
    
    if not status['is_trained'] or args.train:
        console.print_warning("Training models on CSV data...")
        console.print_info("This may take 1-2 minutes...")
        
        try:
            success = csv_ml.train_csv_models()
            
            if success:
                console.print_success("Training completed!")
            else:
                console.print_error("Training failed")
                return
        except Exception as e:
            console.print_error(f"Training failed: {e}")
            return
    else:
        console.print_success("Using pre-trained models")
    
    # Make prediction
    console.print_info(f"Predicting {csv_ml.symbol_name} for next {args.days} days...")
    
    try:
        result = csv_ml.predict_csv(days=args.days)
        
        if 'error' in result:
            console.print_error(f"Prediction failed: {result['error']}")
            return
        
        if result and 'predictions' in result:
            # Display results
            print("\n" + "=" * 70)
            print(f"{result['symbol']} - CSV Prediction Results")
            print("=" * 70)
            
            print(f"\nData Information:")
            print(f"   Type: {result['data_type'].title()}")
            print(f"   Data Points: {result['data_points']:,}")
            print(f"   Date Range: {result['date_range']}")
            
            if result['data_type'] == 'forex':
                print(f"\nCurrent Rate: {result['current_price']:.5f}")
            else:
                print(f"\nCurrent Price: ${result['current_price']:.2f}")
            
            print(f"Trend: {result['trend']}")
            print(f"Volatility: {result['volatility']:.2f}%")
            
            # Predictions
            print(f"\n{args.days}-Day Forecast:")
            print("-" * 70)
            
            if result['data_type'] == 'forex':
                print(f"{'Date':<12} {'Rate':<12} {'Pips':<12} {'Change':<12} {'Confidence':<12}")
            else:
                print(f"{'Date':<12} {'Price':<12} {'Change':<12} {'Confidence':<12}")
            
            print("-" * 70)
            
            for pred in result['predictions']:
                date = pred['date']
                price = pred['predicted_price']
                pred_return = pred['predicted_return'] * 100
                conf = pred['confidence']
                
                # Format change
                if pred_return > 0:
                    change_str = f"+{pred_return:.2f}%"
                elif pred_return < 0:
                    change_str = f"{pred_return:.2f}%"
                else:
                    change_str = f"{pred_return:.2f}%"
                
                if result['data_type'] == 'forex' and 'pips' in pred:
                    pips = pred['pips']
                    pips_str = f"{pips:+.1f}" if pips != 0 else "0.0"
                    if result['current_price'] < 10:
                        print(f"{date:<12} {price:<12.5f} {pips_str:<12} {change_str:<12} {conf:.1%}")
                    else:
                        print(f"{date:<12} {price:<12.3f} {pips_str:<12} {change_str:<12} {conf:.1%}")
                else:
                    print(f"{date:<12} ${price:<11.2f} {change_str:<12} {conf:.1%}")
            
            print("-" * 70)
            
            # Summary
            avg_change = sum(p['predicted_return'] * 100 for p in result['predictions']) / len(result['predictions'])
            final_price = result['predictions'][-1]['predicted_price']
            total_change = ((final_price - result['current_price']) / result['current_price']) * 100
            
            print(f"\nSummary:")
            print(f"   Average Daily Change: {avg_change:+.2f}%")
            
            if result['data_type'] == 'forex':
                print(f"   Final Predicted Rate: {final_price:.5f}")
                if 'pips' in result['predictions'][-1]:
                    total_pips = result['predictions'][-1]['pips']
                    print(f"   Total Pips: {total_pips:+.1f}")
            else:
                print(f"   Final Predicted Price: ${final_price:.2f}")
            
            print(f"   Total Change: {total_change:+.2f}%")
            
            # Market outlook
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
            
    except Exception as e:
        console.print_error(f"Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()