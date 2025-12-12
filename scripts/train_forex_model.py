#!/usr/bin/env python3
"""
Train forex prediction models from database
Supports both full and incremental training
"""

import argparse
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meridianalgo.forex_ml import ForexML

def load_forex_data_from_db(db_file, pair, use_all_data=True):
    """Load forex training data from database"""
    conn = sqlite3.connect(db_file)
    
    # Convert pair to yfinance format
    symbol = f"{pair}=X" if not pair.endswith('=X') else pair
    
    if use_all_data:
        # Load all historical data
        query = '''
            SELECT date, open, high, low, close, volume
            FROM market_data
            WHERE symbol = ? AND asset_type = 'forex'
            ORDER BY date ASC
        '''
    else:
        # Load only recent data (last 90 days)
        query = '''
            SELECT date, open, high, low, close, volume
            FROM market_data
            WHERE symbol = ? AND asset_type = 'forex'
            AND date >= date('now', '-90 days')
            ORDER BY date ASC
        '''
    
    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()
    
    if df.empty:
        raise ValueError(f"No data found for {pair}")
    
    # Rename columns to match expected format
    df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    return df

def train_forex_model(pair, db_file, output_path, epochs=100, use_all_data=True, incremental=False):
    """Train forex model for a currency pair"""
    print(f"\n{'='*60}")
    print(f"Training forex model for {pair}")
    print(f"{'='*60}")
    
    # Load data
    print("Loading data from database...")
    data = load_forex_data_from_db(db_file, pair, use_all_data)
    print(f"  ✓ Loaded {len(data)} rows")
    print(f"  Date range: {data.index.min()} to {data.index.max()}")
    
    # Initialize Forex ML system
    forex_ml = ForexML(model_path=output_path)
    
    # Train model
    print(f"\nTraining model ({epochs} epochs)...")
    
    if incremental and Path(output_path).exists():
        print("  Using incremental training mode")
        # Load existing model and continue training
        result = forex_ml.train_ultimate_models(
            target_symbol=pair,
            period='custom',
            custom_data=data,
            epochs=epochs,
            quick_mode=False
        )
    else:
        print("  Using full training mode")
        # Train from scratch
        result = forex_ml.train_ultimate_models(
            target_symbol=pair,
            period='custom',
            custom_data=data,
            epochs=epochs,
            quick_mode=False
        )
    
    if result.get('success'):
        print(f"\n✓ Training completed successfully")
        print(f"  Final loss: {result.get('final_loss', 'N/A')}")
        print(f"  Model saved to: {output_path}")
        
        # Store metadata in database
        store_model_metadata(db_file, pair, output_path, result)
        
        return True
    else:
        print(f"\n✗ Training failed: {result.get('error', 'Unknown error')}")
        return False

def store_model_metadata(db_file, pair, model_path, training_result):
    """Store model metadata in database"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO model_metadata 
        (symbol, model_type, training_date, accuracy, loss, epochs, model_path)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        pair,
        'forex_ml',
        pd.Timestamp.now(),
        training_result.get('accuracy', 0.0),
        training_result.get('final_loss', 0.0),
        training_result.get('epochs', 0),
        str(model_path)
    ))
    
    conn.commit()
    conn.close()

def main():
    parser = argparse.ArgumentParser(description='Train forex prediction model')
    parser.add_argument('--pair', required=True, help='Forex pair (e.g., EURUSD)')
    parser.add_argument('--db-file', required=True, help='SQLite database file')
    parser.add_argument('--output', required=True, help='Output model file')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--use-all-data', action='store_true', help='Use all historical data')
    parser.add_argument('--incremental', action='store_true', help='Incremental training')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Train model
    success = train_forex_model(
        pair=args.pair,
        db_file=args.db_file,
        output_path=args.output,
        epochs=args.epochs,
        use_all_data=args.use_all_data,
        incremental=args.incremental
    )
    
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
