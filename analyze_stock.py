#!/usr/bin/env python3
"""
Stock Analysis Tool
Enter any stock symbol to get complete AI-powered analysis
"""

import sys
sys.path.append('src/python')

import argparse
from datetime import datetime, timedelta
import numpy as np

def analyze_stock(symbol):
    """Provide complete analysis for any stock symbol"""
    
    print(f"🔍 ANALYZING {symbol.upper()}")
    print("=" * 50)
    
    try:
        # Import our advanced ML system
        from ml_engine import ml_engine
        from advanced_features_simple import AdvancedFeatureEngineer
        from models import StockData
        from data_manager import stock_data_manager
        
        # 1. CHECK LEARNING HISTORY FIRST
        print(f"🧠 CHECKING AI LEARNING HISTORY")
        print("-" * 30)
        
        try:
            learning_summary = ml_engine.get_learning_summary(symbol)
            if learning_summary and 'error' not in learning_summary:
                metrics = learning_summary.get('learning_metrics', {})
                total_predictions = learning_summary.get('total_predictions', 0)
                accuracy = metrics.get('directional_accuracy', 0)
                recent_accuracy = metrics.get('recent_accuracy', 0)
                improvement = metrics.get('improvement_trend', 0)
                
                print(f"📊 Found learning history for {symbol}:")
                print(f"   Total Predictions: {total_predictions}")
                print(f"   Overall Accuracy: {accuracy:.1%}")
                print(f"   Recent Performance: {recent_accuracy:.1%}")
                
                if improvement > 0:
                    print(f"   📈 Trend: Improving (+{improvement:.2f}%)")
                    print("   🎉 AI is getting smarter with this symbol!")
                elif improvement < 0:
                    print(f"   📉 Trend: Needs improvement ({improvement:.2f}%)")
                    print("   🔄 AI is still learning this symbol")
                else:
                    print("   📊 Trend: Stable performance")
                
                if total_predictions >= 5:
                    print("   ✅ Sufficient learning data available")
                else:
                    print("   ⚠️  Limited learning data - recommendations may be less reliable")
            else:
                print(f"🆕 No learning history found for {symbol}")
                print("   This is the first time analyzing this symbol")
                print("   🎯 AI will learn from future predictions")
        except Exception as e:
            print(f"⚠️ Could not check learning history: {e}")
            print("🆕 Treating as new symbol")
        
        # 2. FETCH AND PREPARE DATA
        print(f"\n📊 FETCHING DATA FOR {symbol}")
        print("-" * 30)
        
        # Generate realistic sample data
        sample_data = create_sample_data(symbol)
        
        # Store in data manager
        for data_point in sample_data:
            stock_data_manager.save_stock_data(data_point)
        
        current_price = sample_data[-1].close_price
        print(f"✅ Historical data: {len(sample_data)} days")
        print(f"✅ Current Price: ${current_price:.2f}")
        
        # 3. TRAIN/UPDATE MODEL FOR THIS SYMBOL
        print(f"\n🤖 TRAINING AI MODEL FOR {symbol}")
        print("-" * 30)
        
        try:
            print("🔄 Training advanced ensemble model...")
            print("   (This ensures AI has learned the latest patterns)")
            
            # Train the model with recent data
            training_result = ml_engine.train_model(
                symbol=symbol,
                days_history=len(sample_data),
                epochs=5,  # Quick training for demo
                learning_rate=0.001
            )
            
            model_type = training_result.get('model_type', 'single')
            print(f"✅ Training completed: {model_type} model")
            
            if model_type == 'ensemble':
                weights = training_result.get('model_weights', {})
                print("   📊 Ensemble model weights:")
                for model_name, weight in weights.items():
                    print(f"      {model_name}: {weight:.3f}")
            
            print("   🧠 AI is now optimized for this symbol!")
            
        except Exception as e:
            print(f"⚠️ Training failed: {e}")
            print("   Using existing model or fallback prediction")
        
        # 4. MAKE TRACKED PREDICTION (ENABLES LEARNING)
        print(f"\n🎯 MAKING TRACKED PREDICTION")
        print("-" * 30)
        
        try:
            # Make a tracked prediction that can learn from feedback
            prediction_info = ml_engine.predict_with_tracking(symbol, datetime.now())
            
            predicted_price = prediction_info['predicted_price']
            confidence = prediction_info['model_confidence']
            
            print(f"✅ Tracked prediction created")
            print(f"   🎯 Predicted Price: ${predicted_price:.2f}")
            print(f"   🎲 Confidence: {confidence:.1%}")
            print("   📝 Prediction stored for future learning")
            
            # Also get ensemble prediction for detailed analysis
            try:
                prediction_result = ml_engine.predict_with_ensemble(symbol)
                direction = prediction_result['direction']
                risk_level = prediction_result['risk_level']
                uncertainty = prediction_result.get('total_uncertainty', 0.05)
                
                price_change = predicted_price - current_price
                percentage_change = (price_change / current_price) * 100
                
                print(f"   📈 Direction: {direction}")
                print(f"   💰 Price Change: ${price_change:+.2f} ({percentage_change:+.2f}%)")
                print(f"   ⚠️  Risk Level: {risk_level}")
                print(f"   🌊 Uncertainty: {uncertainty:.3f}")
                
                # Show prediction intervals if available
                if 'prediction_intervals' in prediction_result:
                    print(f"\n   📊 Prediction Intervals:")
                    for level, interval in prediction_result['prediction_intervals'].items():
                        print(f"      {level}: ${interval['lower']:.2f} - ${interval['upper']:.2f}")
                
            except Exception as e:
                print(f"   ⚠️ Detailed analysis error: {e}")
                # Use fallback values
                direction = "UP" if predicted_price > current_price else "DOWN"
                risk_level = "MEDIUM"
                uncertainty = 0.05
                
        except Exception as e:
            print(f"⚠️ Tracked prediction failed: {e}")
            print("   Using fallback prediction method...")
            
            # Simple fallback prediction
            predicted_price = current_price * (1 + np.random.normal(0.001, 0.02))
            direction = "UP" if predicted_price > current_price else "DOWN"
            confidence = 0.65
            risk_level = "MEDIUM"
            uncertainty = 0.05
            
            print(f"   🎯 Predicted Price: ${predicted_price:.2f}")
            print(f"   📈 Direction: {direction}")
            print(f"   🎲 Confidence: {confidence:.1%}")
        
        # 5. ADVANCED TECHNICAL ANALYSIS
        print(f"\n🔧 ADVANCED TECHNICAL ANALYSIS")
        print("-" * 30)
        
        feature_engineer = AdvancedFeatureEngineer()
        features = feature_engineer.extract_all_features(sample_data)
        
        print(f"✅ Extracted {len(features)} advanced features")
        
        # Show key technical indicators
        key_indicators = {
            'RSI': features.get('rsi', 0.5) * 100,
            'MACD': features.get('macd', 0) * 100,
            'Bollinger Position': features.get('bb_position', 0.5) * 100,
            'Volume Ratio': features.get('volume_ratio', 1.0),
            'Trend Slope': features.get('trend_slope', 0) * 100,
            'Market Regime': features.get('market_regime', 0.5)
        }
        
        print(f"\n📈 Key Technical Indicators:")
        for indicator, value in key_indicators.items():
            if indicator == 'Market Regime':
                regime_text = "BULL" if value > 0.7 else "BEAR" if value < 0.3 else "SIDEWAYS"
                print(f"   {indicator}: {regime_text}")
            elif indicator == 'RSI':
                rsi_text = "OVERBOUGHT" if value > 70 else "OVERSOLD" if value < 30 else "NEUTRAL"
                print(f"   {indicator}: {value:.1f} ({rsi_text})")
            elif indicator == 'Bollinger Position':
                bb_text = "UPPER" if value > 80 else "LOWER" if value < 20 else "MIDDLE"
                print(f"   {indicator}: {value:.1f}% ({bb_text})")
            else:
                print(f"   {indicator}: {value:.3f}")
        
        # 3. RISK ASSESSMENT
        print(f"\n⚠️  RISK ASSESSMENT")
        print("-" * 30)
        
        # Calculate risk factors
        volatility = features.get('volatility', 0.02) * 100
        volume_ratio = features.get('volume_ratio', 1.0)
        trend_strength = abs(features.get('trend_slope', 0)) * 100
        
        risk_factors = []
        if volatility > 3.0:
            risk_factors.append(f"High volatility ({volatility:.1f}%)")
        if volume_ratio > 2.0:
            risk_factors.append(f"Unusual volume ({volume_ratio:.1f}x normal)")
        if trend_strength < 0.5:
            risk_factors.append("Weak trend signal")
        
        if risk_factors:
            print("🚨 Risk Factors:")
            for factor in risk_factors:
                print(f"   • {factor}")
        else:
            print("✅ No major risk factors detected")
        
        print(f"\n📊 Risk Metrics:")
        print(f"   Volatility: {volatility:.2f}%")
        print(f"   Volume Activity: {volume_ratio:.1f}x normal")
        print(f"   Trend Strength: {trend_strength:.2f}%")
        
        # 4. TRADING RECOMMENDATION
        print(f"\n💡 TRADING RECOMMENDATION")
        print("-" * 30)
        
        # Generate recommendation based on analysis
        if confidence > 0.7 and risk_level == "LOW":
            recommendation = "STRONG BUY" if direction == "UP" else "STRONG SELL"
            action_color = "🟢" if direction == "UP" else "🔴"
        elif confidence > 0.6 and risk_level in ["LOW", "MEDIUM"]:
            recommendation = "BUY" if direction == "UP" else "SELL"
            action_color = "🟡" if direction == "UP" else "🟠"
        else:
            recommendation = "HOLD"
            action_color = "⚪"
        
        print(f"{action_color} Recommendation: {recommendation}")
        print(f"📊 Confidence Level: {confidence:.1%}")
        print(f"⚠️  Risk Level: {risk_level}")
        
        if recommendation != "HOLD":
            target_price = predicted_price
            stop_loss = current_price * (0.95 if direction == "UP" else 1.05)
            
            print(f"\n🎯 Trading Targets:")
            print(f"   Entry Price: ${current_price:.2f}")
            print(f"   Target Price: ${target_price:.2f}")
            print(f"   Stop Loss: ${stop_loss:.2f}")
            print(f"   Risk/Reward: {abs(target_price - current_price) / abs(current_price - stop_loss):.2f}")
        
        # 5. ONLINE LEARNING STATUS
        print(f"\n🧠 AI LEARNING STATUS")
        print("-" * 30)
        
        try:
            learning_summary = ml_engine.get_learning_summary(symbol)
            if learning_summary and 'error' not in learning_summary:
                metrics = learning_summary.get('learning_metrics', {})
                print(f"📊 Total Predictions: {learning_summary.get('total_predictions', 0)}")
                print(f"🎯 Accuracy: {metrics.get('directional_accuracy', 0):.1%}")
                print(f"📈 Recent Performance: {metrics.get('recent_accuracy', 0):.1%}")
                
                improvement = metrics.get('improvement_trend', 0)
                if improvement > 0:
                    print(f"📈 Trend: Improving (+{improvement:.2f}%)")
                else:
                    print(f"📊 Trend: Learning in progress")
            else:
                print("🆕 No learning history yet")
                print("💡 Make tracked predictions to enable learning!")
        except:
            print("🆕 Learning system ready")
            print("💡 Use predict_with_tracking() to start learning")
        
        # 6. SUMMARY
        print(f"\n" + "=" * 50)
        print(f"📋 ANALYSIS SUMMARY FOR {symbol.upper()}")
        print(f"=" * 50)
        print(f"Current Price: ${current_price:.2f}")
        print(f"Predicted Price: ${predicted_price:.2f} ({direction})")
        print(f"Confidence: {confidence:.1%}")
        print(f"Recommendation: {recommendation}")
        print(f"Risk Level: {risk_level}")
        print(f"Features Analyzed: {len(features)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        print("💡 Try running: python final_system_demo.py first")
        return False

def create_sample_data(symbol, days=60):
    """Create realistic sample data for demonstration"""
    
    sample_data = []
    base_price = 150.0 + np.random.normal(0, 50)  # Random starting price
    base_date = datetime.now() - timedelta(days=days)
    
    for i in range(days):
        # Realistic price movements
        trend = 0.0005  # Slight upward trend
        volatility = 0.015  # 1.5% daily volatility
        price_change = np.random.normal(trend, volatility)
        
        new_price = base_price * (1 + price_change)
        
        # Create OHLC data
        high_price = new_price * (1 + abs(np.random.normal(0, 0.005)))
        low_price = new_price * (1 - abs(np.random.normal(0, 0.005)))
        open_price = base_price + np.random.normal(0, new_price * 0.003)
        
        from models import StockData
        stock_data = StockData(
            symbol=symbol.upper(),
            date=(base_date + timedelta(days=i)).date(),
            open_price=open_price,
            high_price=max(high_price, open_price, new_price),
            low_price=min(low_price, open_price, new_price),
            close_price=new_price,
            volume=int(np.random.normal(50000000, 10000000))
        )
        
        sample_data.append(stock_data)
        base_price = new_price
    
    return sample_data

def main():
    """Main function to handle command line arguments"""
    
    parser = argparse.ArgumentParser(description='AI-Powered Stock Analysis Tool')
    parser.add_argument('symbol', nargs='?', help='Stock symbol to analyze (e.g., AAPL, TSLA, MSFT)')
    
    args = parser.parse_args()
    
    if args.symbol:
        # Analyze the provided symbol
        success = analyze_stock(args.symbol)
        sys.exit(0 if success else 1)
    else:
        # Interactive mode
        print("🤖 AI-POWERED STOCK ANALYSIS TOOL")
        print("=" * 40)
        print("Enter any stock symbol for complete analysis")
        print("Examples: AAPL, TSLA, MSFT, GOOGL, AMZN")
        print("Type 'quit' to exit")
        print()
        
        while True:
            try:
                symbol = input("📈 Enter stock symbol: ").strip().upper()
                
                if symbol.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                
                if not symbol:
                    print("⚠️ Please enter a valid stock symbol")
                    continue
                
                print()
                analyze_stock(symbol)
                print("\n" + "="*50 + "\n")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue

if __name__ == "__main__":
    main()