#!/usr/bin/env python3
"""
Teach AI Tool
Provide actual stock prices to help the AI learn from its mistakes
"""

import sys
sys.path.append('src/python')

import argparse
from datetime import datetime, timedelta

def teach_ai_feedback(symbol, actual_price, days_ago=1):
    """Provide feedback to help AI learn"""
    
    print(f"🧠 TEACHING AI ABOUT {symbol.upper()}")
    print("=" * 50)
    
    try:
        from ml_engine import ml_engine
        
        # Calculate the target date (when the prediction was for)
        target_date = datetime.now() - timedelta(days=days_ago-1)
        
        print(f"📅 Providing feedback for prediction made {days_ago} day(s) ago")
        print(f"🎯 Target date: {target_date.strftime('%Y-%m-%d')}")
        print(f"💰 Actual price: ${actual_price:.2f}")
        
        # Provide feedback to the learning system
        feedback_result = ml_engine.provide_prediction_feedback(
            symbol, target_date, actual_price
        )
        
        if 'error' in feedback_result:
            print(f"⚠️ {feedback_result['error']}")
            print("💡 Make sure you've made a tracked prediction first using:")
            print(f"   python analyze_stock.py {symbol}")
            return False
        
        # Show learning results
        predicted_price = feedback_result['predicted_price']
        prediction_error = feedback_result['prediction_error']
        percentage_error = feedback_result['percentage_error']
        direction_correct = feedback_result['direction_correct']
        learning_triggered = feedback_result['learning_triggered']
        
        print(f"\n📊 LEARNING RESULTS:")
        print(f"   🎯 AI Predicted: ${predicted_price:.2f}")
        print(f"   📈 Actual Price: ${actual_price:.2f}")
        print(f"   📊 Prediction Error: ${prediction_error:+.2f}")
        print(f"   📈 Percentage Error: {percentage_error:+.2f}%")
        print(f"   🎯 Direction: {'✅ Correct' if direction_correct else '❌ Wrong'}")
        
        if learning_triggered:
            print(f"   🧠 Learning: ✅ AI Updated Its Knowledge!")
            print(f"   🎉 AI is now smarter about {symbol}")
        else:
            print(f"   🧠 Learning: ⚠️ No learning triggered")
        
        # Show updated learning metrics
        if 'learning_metrics' in feedback_result:
            metrics = feedback_result['learning_metrics']
            if metrics:
                print(f"\n📈 UPDATED AI PERFORMANCE:")
                print(f"   Total Predictions: {metrics.get('total_predictions', 0)}")
                print(f"   Directional Accuracy: {metrics.get('directional_accuracy', 0):.1%}")
                print(f"   Recent Performance: {metrics.get('recent_accuracy', 0):.1%}")
                
                improvement = metrics.get('improvement_trend', 0)
                if improvement > 0:
                    print(f"   📈 Trend: Improving (+{improvement:.2f}%)")
                    print(f"   🎉 AI is getting better at predicting {symbol}!")
                else:
                    print(f"   📊 Trend: Still learning")
        
        # Provide encouragement based on error
        if abs(percentage_error) < 2.0:
            print(f"\n🎉 Excellent prediction! Error under 2%")
        elif abs(percentage_error) < 5.0:
            print(f"\n👍 Good prediction! Error under 5%")
        elif abs(percentage_error) < 10.0:
            print(f"\n📚 Learning opportunity - AI will improve")
        else:
            print(f"\n🔄 Big learning moment - AI will adapt significantly")
        
        return True
        
    except Exception as e:
        print(f"❌ Teaching failed: {e}")
        return False

def show_learning_progress(symbol):
    """Show AI learning progress for a symbol"""
    
    print(f"📊 AI LEARNING PROGRESS FOR {symbol.upper()}")
    print("=" * 50)
    
    try:
        from ml_engine import ml_engine
        
        summary = ml_engine.get_learning_summary(symbol)
        
        if summary and 'error' not in summary:
            metrics = summary.get('learning_metrics', {})
            total_predictions = summary.get('total_predictions', 0)
            
            if total_predictions > 0:
                print(f"📈 Learning Statistics:")
                print(f"   Total Predictions: {total_predictions}")
                print(f"   Overall Accuracy: {metrics.get('directional_accuracy', 0):.1%}")
                print(f"   Recent Performance: {metrics.get('recent_accuracy', 0):.1%}")
                print(f"   Mean Absolute Error: ${metrics.get('mean_absolute_error', 0):.2f}")
                
                improvement = metrics.get('improvement_trend', 0)
                if improvement > 0:
                    print(f"   📈 Improvement Trend: +{improvement:.2f}%")
                    print(f"   🎉 AI is getting smarter!")
                elif improvement < 0:
                    print(f"   📉 Learning Curve: {improvement:.2f}%")
                    print(f"   🔄 AI is still adapting")
                else:
                    print(f"   📊 Performance: Stable")
                
                # Show recent performance
                if 'recent_performance' in summary:
                    recent = summary['recent_performance']
                    print(f"\n🔥 Recent Performance:")
                    print(f"   Mean Error: {recent.get('mean_percentage_error', 0):.2f}%")
                    print(f"   Directional Accuracy: {recent.get('directional_accuracy', 0):.1%}")
                    
                    if 'best_prediction' in recent:
                        best = recent['best_prediction']
                        print(f"   🏆 Best Prediction: {best.percentage_error:.2f}% error")
                    
                    if 'worst_prediction' in recent:
                        worst = recent['worst_prediction']
                        print(f"   📚 Learning From: {worst.percentage_error:.2f}% error")
            else:
                print(f"🆕 No predictions yet for {symbol}")
                print(f"💡 Make a prediction first: python analyze_stock.py {symbol}")
        else:
            print(f"🆕 No learning data found for {symbol}")
            print(f"💡 Start by making a prediction: python analyze_stock.py {symbol}")
            
    except Exception as e:
        print(f"❌ Could not retrieve learning progress: {e}")

def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Teach AI from actual stock prices')
    parser.add_argument('symbol', help='Stock symbol (e.g., AAPL, TSLA)')
    parser.add_argument('actual_price', nargs='?', type=float, help='Actual stock price')
    parser.add_argument('--days-ago', type=int, default=1, help='How many days ago the prediction was made (default: 1)')
    parser.add_argument('--progress', action='store_true', help='Show learning progress only')
    
    args = parser.parse_args()
    
    if args.progress:
        # Just show learning progress
        show_learning_progress(args.symbol)
    elif args.actual_price is not None:
        # Provide feedback
        success = teach_ai_feedback(args.symbol, args.actual_price, args.days_ago)
        if success:
            print(f"\n💡 Next time you analyze {args.symbol}, the AI will be smarter!")
            print(f"   Run: python analyze_stock.py {args.symbol}")
    else:
        # Interactive mode
        symbol = args.symbol.upper()
        
        print(f"🧠 TEACH AI ABOUT {symbol}")
        print("=" * 30)
        print("Help the AI learn by providing actual stock prices!")
        print()
        
        try:
            # First show current learning progress
            show_learning_progress(symbol)
            print()
            
            # Get actual price from user
            actual_price = float(input(f"💰 Enter actual closing price for {symbol}: $"))
            days_ago = int(input("📅 How many days ago was the prediction made? (default 1): ") or "1")
            
            print()
            success = teach_ai_feedback(symbol, actual_price, days_ago)
            
            if success:
                print(f"\n🎉 Thank you for teaching the AI!")
                print(f"💡 The AI is now smarter about {symbol}")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except ValueError:
            print("❌ Please enter valid numbers")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()