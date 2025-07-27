#!/usr/bin/env python3
"""
Simple script to check prediction accuracy results
"""

import pandas as pd
import os
from datetime import datetime

def check_accuracy():
    """Check and display prediction accuracy statistics"""
    
    accuracy_file = 'prediction_accuracy.csv'
    
    if not os.path.exists(accuracy_file):
        print("No accuracy data found. Run some predictions first!")
        return
    
    # Load accuracy data
    df = pd.read_csv(accuracy_file)
    
    if df.empty:
        print("No accuracy data available.")
        return
    
    print("=" * 60)
    print("PREDICTION ACCURACY REPORT")
    print("=" * 60)
    
    # Overall statistics
    total_predictions = len(df)
    accurate_predictions = df['accurate'].sum()
    accuracy_rate = (accurate_predictions / total_predictions) * 100
    avg_error = df['error_pct'].mean()
    
    print(f"\nOVERALL STATISTICS:")
    print(f"Total Predictions Validated: {total_predictions}")
    print(f"Accurate Predictions (within 5%): {accurate_predictions}")
    print(f"Overall Accuracy Rate: {accuracy_rate:.1f}%")
    print(f"Average Error: {avg_error:.2f}%")
    print(f"Best Prediction Error: {df['error_pct'].min():.2f}%")
    print(f"Worst Prediction Error: {df['error_pct'].max():.2f}%")
    
    # By symbol
    print(f"\nBY SYMBOL:")
    for symbol in df['symbol'].unique():
        symbol_df = df[df['symbol'] == symbol]
        symbol_accuracy = (symbol_df['accurate'].sum() / len(symbol_df)) * 100
        symbol_avg_error = symbol_df['error_pct'].mean()
        print(f"{symbol}: {len(symbol_df)} predictions, {symbol_accuracy:.1f}% accuracy, {symbol_avg_error:.2f}% avg error")
    
    # Recent predictions
    print(f"\nRECENT PREDICTIONS:")
    recent = df.tail(10)
    for _, row in recent.iterrows():
        status = "✓" if row['accurate'] else "✗"
        print(f"{status} {row['symbol']} {row['date']}: ${row['predicted']:.2f} → ${row['actual']:.2f} ({row['error_pct']:.1f}% error)")
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    check_accuracy()