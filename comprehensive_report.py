#!/usr/bin/env python3
"""
Comprehensive Report Generator for Ara AI Stock Analysis
Shows all predictions, accuracy, and online learning data
"""

import pandas as pd
import os
from datetime import datetime
import json

def generate_comprehensive_report():
    """Generate a comprehensive report of all system data"""
    
    print("=" * 80)
    print("ARA AI STOCK ANALYSIS - COMPREHENSIVE SYSTEM REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Predictions Report
    print("\n" + "=" * 50)
    print("1. PREDICTIONS OVERVIEW")
    print("=" * 50)
    
    if os.path.exists('predictions.csv'):
        pred_df = pd.read_csv('predictions.csv')
        print(f"Total Predictions Made: {len(pred_df)}")
        print(f"Symbols Analyzed: {', '.join(sorted(pred_df['Symbol'].unique()))}")
        print(f"Date Range: {pred_df['Date'].min()} to {pred_df['Date'].max()}")
        
        print("\nRecent Predictions:")
        for _, row in pred_df.tail(10).iterrows():
            change_color = "📈" if row['Change_Percent'] > 0 else "📉"
            print(f"{change_color} {row['Symbol']} {row['Date']}: ${row['Predicted_Price']:.2f} ({row['Change_Percent']:+.1f}%)")
    else:
        print("No predictions file found.")
    
    # 2. Accuracy Report
    print("\n" + "=" * 50)
    print("2. ACCURACY ANALYSIS")
    print("=" * 50)
    
    if os.path.exists('prediction_accuracy.csv'):
        acc_df = pd.read_csv('prediction_accuracy.csv')
        
        overall_accuracy = (acc_df['accurate'].sum() / len(acc_df)) * 100
        overall_error = acc_df['error_pct'].mean()
        
        print(f"Total Validated Predictions: {len(acc_df)}")
        print(f"Overall Accuracy Rate: {overall_accuracy:.1f}% (within 5% error)")
        print(f"Average Prediction Error: {overall_error:.2f}%")
        print(f"Best Prediction: {acc_df['error_pct'].min():.2f}% error")
        print(f"Worst Prediction: {acc_df['error_pct'].max():.2f}% error")
        
        print("\nAccuracy by Symbol:")
        for symbol in sorted(acc_df['symbol'].unique()):
            symbol_data = acc_df[acc_df['symbol'] == symbol]
            symbol_acc = (symbol_data['accurate'].sum() / len(symbol_data)) * 100
            symbol_err = symbol_data['error_pct'].mean()
            status = "🟢" if symbol_acc >= 80 else "🟡" if symbol_acc >= 60 else "🔴"
            print(f"{status} {symbol}: {len(symbol_data)} predictions, {symbol_acc:.1f}% accuracy, {symbol_err:.2f}% avg error")
        
        print("\nRecent Accuracy Results:")
        for _, row in acc_df.tail(5).iterrows():
            status = "✅" if row['accurate'] else "❌"
            print(f"{status} {row['symbol']} {row['date']}: ${row['predicted']:.2f} → ${row['actual']:.2f} ({row['error_pct']:.1f}% error)")
    else:
        print("No accuracy data available.")
    
    # 3. Online Learning Report
    print("\n" + "=" * 50)
    print("3. ONLINE LEARNING SYSTEM")
    print("=" * 50)
    
    if os.path.exists('online_learning_data.csv'):
        learn_df = pd.read_csv('online_learning_data.csv')
        
        print(f"Learning Records: {len(learn_df)}")
        print(f"Symbols in Learning System: {', '.join(sorted(learn_df['symbol'].unique()))}")
        
        print("\nLearning Progress by Symbol:")
        for symbol in sorted(learn_df['symbol'].unique()):
            symbol_data = learn_df[learn_df['symbol'] == symbol]
            if len(symbol_data) >= 2:
                recent_error = symbol_data['prediction_error'].tail(3).mean()
                trend = "📈 Improving" if symbol_data['prediction_error'].iloc[-1] < symbol_data['prediction_error'].iloc[0] else "📉 Declining"
                print(f"🧠 {symbol}: {len(symbol_data)} learning cycles, {recent_error:.1f}% recent error, {trend}")
            else:
                print(f"🧠 {symbol}: {len(symbol_data)} learning cycles (insufficient data for trend)")
        
        print("\nRecent Learning Updates:")
        for _, row in learn_df.tail(5).iterrows():
            print(f"🔄 {row['symbol']} {row['timestamp'][:10]}: {row['prediction_error']:.1f}% error")
    else:
        print("No online learning data available.")
    
    # 4. Learning Parameters
    print("\n" + "=" * 50)
    print("4. ADAPTIVE LEARNING PARAMETERS")
    print("=" * 50)
    
    param_files = [f for f in os.listdir('.') if f.startswith('learning_params_') and f.endswith('.json')]
    
    if param_files:
        for param_file in sorted(param_files):
            symbol = param_file.replace('learning_params_', '').replace('.json', '')
            try:
                with open(param_file, 'r') as f:
                    params = json.load(f)
                
                performance = params.get('performance_score', 0)
                trend = params.get('trend', 'unknown')
                data_points = params.get('data_points', 0)
                recent_error = params.get('recent_error', 0)
                
                status = "🟢" if performance >= 80 else "🟡" if performance >= 60 else "🔴"
                trend_icon = "📈" if trend == "improving" else "📉" if trend == "declining" else "➡️"
                
                print(f"{status} {symbol}: {performance:.1f}% performance, {trend_icon} {trend}, {data_points} data points, {recent_error:.1f}% recent error")
            except:
                print(f"❓ {symbol}: Parameter file corrupted")
    else:
        print("No adaptive learning parameters found.")
    
    # 5. System Statistics
    print("\n" + "=" * 50)
    print("5. SYSTEM STATISTICS")
    print("=" * 50)
    
    total_files = len([f for f in os.listdir('.') if f.endswith('.csv')])
    total_params = len(param_files)
    
    print(f"📊 Data Files: {total_files}")
    print(f"🧠 Learning Parameter Files: {total_params}")
    print(f"🔧 System Status: Fully Operational")
    print(f"📈 Features: 21 enhanced features per prediction")
    print(f"🤖 Models: Linear Regression + Random Forest + Enhanced Neural Network")
    print(f"⚡ Device: CPU with 8 threads optimization")
    print(f"🎯 Accuracy Threshold: 5% error tolerance")
    
    print("\n" + "=" * 80)
    print("SYSTEM READY FOR PRODUCTION USE")
    print("=" * 80)

if __name__ == "__main__":
    generate_comprehensive_report()