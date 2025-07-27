#!/usr/bin/env python3
"""
View Predictions History - Shows all predictions organized by symbol and date
"""

import pandas as pd
import os
from datetime import datetime

def view_predictions():
    """Display all predictions in an organized format"""
    
    predictions_file = 'predictions.csv'
    
    if not os.path.exists(predictions_file):
        print("No predictions file found. Run some predictions first!")
        return
    
    df = pd.read_csv(predictions_file)
    
    if df.empty:
        print("No predictions available.")
        return
    
    print("=" * 80)
    print("PREDICTION HISTORY - ALL SYMBOLS")
    print("=" * 80)
    print(f"Total Predictions: {len(df)}")
    print(f"Symbols: {', '.join(sorted(df['Symbol'].unique()))}")
    print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Group by symbol
    for symbol in sorted(df['Symbol'].unique()):
        symbol_df = df[df['Symbol'] == symbol]
        
        print(f"\n{'='*20} {symbol} ({'='*20}")
        print(f"Total Predictions: {len(symbol_df)}")
        
        # Group by timestamp (prediction session)
        symbol_df['timestamp_date'] = pd.to_datetime(symbol_df['Timestamp']).dt.date
        
        for session_date in sorted(symbol_df['timestamp_date'].unique()):
            session_df = symbol_df[symbol_df['timestamp_date'] == session_date]
            
            if len(session_df) > 0:
                first_row = session_df.iloc[0]
                current_price = first_row['Current_Price']
                avg_prediction = session_df['Predicted_Price'].mean()
                avg_change = session_df['Change_Percent'].mean()
                
                print(f"\n📅 Session: {session_date}")
                print(f"   Current Price: ${current_price:.2f}")
                print(f"   Avg Prediction: ${avg_prediction:.2f} ({avg_change:+.1f}%)")
                print(f"   Predictions: {len(session_df)} days")
                
                # Show individual predictions
                for _, row in session_df.iterrows():
                    change_icon = "📈" if row['Change_Percent'] > 0 else "📉" if row['Change_Percent'] < 0 else "➡️"
                    print(f"   {change_icon} {row['Date']}: ${row['Predicted_Price']:.2f} ({row['Change_Percent']:+.1f}%)")
    
    # Summary statistics
    print(f"\n{'='*20} SUMMARY STATISTICS {'='*20}")
    
    summary_stats = df.groupby('Symbol').agg({
        'Predicted_Price': ['count', 'mean'],
        'Change_Percent': ['mean', 'std'],
        'Timestamp': ['min', 'max']
    }).round(2)
    
    for symbol in sorted(df['Symbol'].unique()):
        symbol_data = df[df['Symbol'] == symbol]
        prediction_count = len(symbol_data)
        avg_change = symbol_data['Change_Percent'].mean()
        change_std = symbol_data['Change_Percent'].std()
        first_prediction = symbol_data['Timestamp'].min()
        last_prediction = symbol_data['Timestamp'].max()
        
        print(f"\n📊 {symbol}:")
        print(f"   Predictions: {prediction_count}")
        print(f"   Avg Change: {avg_change:+.1f}% (±{change_std:.1f}%)")
        print(f"   First: {first_prediction[:10]}")
        print(f"   Latest: {last_prediction[:10]}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    view_predictions()