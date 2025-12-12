#!/usr/bin/env python3
"""
Store training data in SQLite database
Handles both full and incremental updates
"""

import argparse
import pandas as pd
import sqlite3
from pathlib import Path
from datetime import datetime
import sys

def create_database(db_file):
    """Create database schema"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    
    # Market data table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            asset_type TEXT NOT NULL,
            date TIMESTAMP NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            fetch_date TIMESTAMP,
            timeframe TEXT,
            interval TEXT,
            hour INTEGER,
            UNIQUE(symbol, date, interval)
        )
    ''')
    
    # Training runs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date TIMESTAMP NOT NULL,
            mode TEXT NOT NULL,
            timeframe TEXT,
            hour INTEGER,
            symbols_count INTEGER,
            rows_processed INTEGER,
            status TEXT,
            notes TEXT
        )
    ''')
    
    # Model metadata table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS model_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            model_type TEXT NOT NULL,
            training_date TIMESTAMP NOT NULL,
            accuracy REAL,
            loss REAL,
            epochs INTEGER,
            model_path TEXT,
            UNIQUE(symbol, model_type, training_date)
        )
    ''')
    
    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_symbol_date ON market_data(symbol, date)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_asset_type ON market_data(asset_type)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_timeframe ON market_data(timeframe)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_hour ON market_data(hour)')
    
    conn.commit()
    conn.close()
    print("✓ Database schema created")

def store_data(data_dir, db_file, mode='full', timeframe='1d', hour=None):
    """Store CSV data in database"""
    conn = sqlite3.connect(db_file)
    
    data_dir = Path(data_dir)
    csv_files = list(data_dir.glob('*.csv'))
    
    if not csv_files:
        print("Error: No CSV files found")
        sys.exit(1)
    
    total_rows = 0
    symbols_processed = set()
    
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Prepare data for insertion
            df['date'] = pd.to_datetime(df['Date'])
            
            # Extract hour from datetime
            df['hour'] = df['date'].dt.hour
            
            # Get timeframe and interval from CSV if available
            csv_timeframe = df['Timeframe'].iloc[0] if 'Timeframe' in df.columns else timeframe
            csv_interval = df['Interval'].iloc[0] if 'Interval' in df.columns else '1d'
            
            # Select relevant columns
            columns_to_select = [
                'Symbol', 'AssetType', 'date', 
                'Open', 'High', 'Low', 'Close', 'Volume', 'FetchDate', 'hour'
            ]
            
            insert_df = df[columns_to_select].copy()
            
            insert_df.columns = [
                'symbol', 'asset_type', 'date',
                'open', 'high', 'low', 'close', 'volume', 'fetch_date', 'hour'
            ]
            
            # Add timeframe and interval
            insert_df['timeframe'] = csv_timeframe
            insert_df['interval'] = csv_interval
            
            # Insert or replace data
            if mode == 'full':
                # Full mode: append all data
                insert_df.to_sql('market_data', conn, if_exists='append', index=False)
            else:
                # Hourly/Refresh mode: only insert new data
                for _, row in insert_df.iterrows():
                    try:
                        conn.execute('''
                            INSERT OR IGNORE INTO market_data 
                            (symbol, asset_type, date, open, high, low, close, volume, 
                             fetch_date, timeframe, interval, hour)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', tuple(row))
                    except sqlite3.IntegrityError:
                        pass  # Skip duplicates
            
            symbols_processed.update(df['Symbol'].unique())
            total_rows += len(insert_df)
            print(f"  ✓ Stored {len(insert_df)} rows")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    # Record training run
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO training_runs (run_date, mode, timeframe, hour, symbols_count, rows_processed, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (datetime.now(), mode, timeframe, hour, len(symbols_processed), total_rows, 'completed'))
    
    conn.commit()
    conn.close()
    
    print(f"\n✓ Stored {total_rows} total rows")
    print(f"  Symbols: {len(symbols_processed)}")
    print(f"  Mode: {mode}")

def main():
    parser = argparse.ArgumentParser(description='Store training data in database')
    parser.add_argument('--data-dir', required=True, help='Directory containing CSV files')
    parser.add_argument('--db-file', required=True, help='SQLite database file')
    parser.add_argument('--mode', choices=['full', 'hourly', 'refresh'], default='full')
    parser.add_argument('--timeframe', default='1d', help='Timeframe identifier')
    parser.add_argument('--hour', type=int, help='Current hour (for hourly mode)')
    
    args = parser.parse_args()
    
    # Create database if it doesn't exist
    create_database(args.db_file)
    
    # Store data
    store_data(args.data_dir, args.db_file, args.mode, args.timeframe, args.hour)

if __name__ == '__main__':
    main()
