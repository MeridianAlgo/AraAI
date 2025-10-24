"""
ARA AI Forex - Currency Pair Prediction
Simple command-line interface for forex predictions
"""

import sys
import argparse
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from meridianalgo.forex_ml import ForexML
from meridianalgo.console import ConsoleManager

def main():
    parser = argparse.ArgumentParser(description='ARA AI Forex - Currency Pair Prediction')
    parser.add_argument('pair', nargs='?', default='EURUSD', help='Currency pair (e.g., EURUSD, EUR/USD, EUR-USD)')
    parser.add_argument('--days', type=int, default=5, help='Number of days to predict (default: 5)')
    parser.add_argument('--train', action='store_true', help='Force model training')
    parser.add_argument('--period', default='2y', help='Training period (default: 2y)')
    
    args = parser.parse_args()
    
    console = ConsoleManager()
    
    # Print header
    console.print_header(f"ARA AI Forex v3.0.2 - {args.pair} Analysis")
    
    # Initialize system
    console.print_info("Initializing Forex ML System...")
    forex = ForexML()
    
    # Check if models are trained
    status = forex.get_model_status()
    
    if not status['is_trained'] or args.train:
        console.print_warning(f"Training models on {args.pair} with maximum historical data ({args.period} period)...")
        console.print_info("This may take 1-2 minutes...")
    
    # Make prediction
    console.print_info(f"Predicting {args.pair} for next {args.days} days...")
    
    try:
        result = forex.predict_forex(args.pair, days=args.days, period=args.period)
        
        if 'error' in result:
            console.print_error(f"Prediction failed: {result['error']}")
            return
        
        if result and 'predictions' in result:
            # Display results
            print("\n" + "=" * 70)
            print(f"💱 {result['pair']} - Forex Prediction Results")
            print("=" * 70)
            
            pair_info = result['pair_info']
            print(f"\n📊 Pair Information:")
            print(f"   Base: {pair_info['base_name']} ({pair_info['base_currency']})")
            print(f"   Quote: {pair_info['quote_name']} ({pair_info['quote_currency']})")
            print(f"   Type: {pair_info['type']} Pair")
            print(f"   Regions: {pair_info['base_region']} / {pair_info['quote_region']}")
            
            print(f"\n💰 Current Rate: {result['current_price']:.5f}")
            print(f"📈 Trend: {result['trend']}")
            print(f"📊 Volatility: {result['volatility']:.2f}%")
            
            # Predictions
            print(f"\n📈 {args.days}-Day Forecast:")
            print("-" * 70)
            print(f"{'Date':<12} {'Rate':<12} {'Pips':<12} {'Change':<12} {'Confidence':<12}")
            print("-" * 70)
            
            for pred in result['predictions']:
                date = pred['date']
                price = pred['predicted_price']
                pips = pred['pips']
                pred_return = pred['predicted_return'] * 100
                conf = pred['confidence']
                
                # Format change with indicator
                if pred_return > 0:
                    change_str = f"+{pred_return:.2f}% 📈"
                    pips_str = f"+{pips:.1f}"
                elif pred_return < 0:
                    change_str = f"{pred_return:.2f}% 📉"
                    pips_str = f"{pips:.1f}"
                else:
                    change_str = f"{pred_return:.2f}% ➡️"
                    pips_str = f"{pips:.1f}"
                
                print(f"{date:<12} {price:<12.5f} {pips_str:<12} {change_str:<12} {conf:.1%}")
            
            print("-" * 70)
            
            # Summary
            avg_change = sum(p['predicted_return'] * 100 for p in result['predictions']) / len(result['predictions'])
            final_price = result['predictions'][-1]['predicted_price']
            total_change = ((final_price - result['current_price']) / result['current_price']) * 100
            total_pips = result['predictions'][-1]['pips']
            
            print(f"\n📊 Summary:")
            print(f"   Average Daily Change: {avg_change:+.2f}%")
            print(f"   Final Predicted Rate: {final_price:.5f}")
            print(f"   Total Change: {total_change:+.2f}%")
            print(f"   Total Pips: {total_pips:+.1f}")
            
            # Market outlook
            if total_change > 2:
                print(f"\n💡 Outlook: Strong Bullish 🚀")
                print(f"   {pair_info['base_currency']} expected to strengthen vs {pair_info['quote_currency']}")
            elif total_change > 0.5:
                print(f"\n💡 Outlook: Bullish 📈")
                print(f"   {pair_info['base_currency']} likely to gain vs {pair_info['quote_currency']}")
            elif total_change > -0.5:
                print(f"\n💡 Outlook: Neutral ➡️")
                print(f"   {pair_info['base_currency']} stable vs {pair_info['quote_currency']}")
            elif total_change > -2:
                print(f"\n💡 Outlook: Bearish 📉")
                print(f"   {pair_info['base_currency']} likely to weaken vs {pair_info['quote_currency']}")
            else:
                print(f"\n💡 Outlook: Strong Bearish ⚠️")
                print(f"   {pair_info['base_currency']} expected to weaken vs {pair_info['quote_currency']}")
            
            # Market status
            market_status = forex.get_forex_market_status()
            print(f"\n🌍 Market Status: {market_status['status']}")
            
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
