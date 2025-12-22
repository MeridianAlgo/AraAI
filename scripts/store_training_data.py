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
    
    # Filter out combined files to avoid duplicates
    csv_files = [f for f in csv_files if not f.name.startswith('all_')]
    
    if not csv_files:
        print("Error: No CSV files found")
        sys.exit(1)
    
    total_rows = 0
    symbols_processed = set()
    
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        
        try:
            df = pd.read_csv(csv_file)
            
            # Debug: print columns
            print(f"  Columns: {list(df.columns)}")
            
            # Handle Date column - convert to datetime
            if 'Date' in df.columns:
                df['date'] = pd.to_datetime(df['Date'], utc=True, errors='coerce')
            elif 'Datetime' in df.columns:
                df['date'] = pd.to_datetime(df['Datetime'], utc=True, errors='coerce')
            else:
                print(f"  Error: No Date or Datetime column found")
                continue
            
            # Drop rows with invalid dates
            df = df.dropna(subset=['date'])
            
            if df.empty:
                print(f"  Warning: No valid data after date parsing")
                continue
            
            # Extract hour from datetime
            df['hour'] = df['date'].dt.hour
            
            # Get timeframe and interval from CSV if available
            csv_timeframe = df['Timeframe'].iloc[0] if 'Timeframe' in df.columns else timeframe
            csv_interval = df['Interval'].iloc[0] if 'Interval' in df.columns else '1d'
            
            # Prepare insert dataframe with correct column mapping
            insert_df = pd.DataFrame()
            insert_df['symbol'] = df['Symbol']
            insert_df['asset_type'] = df['AssetType']
            insert_df['date'] = df['date'].dt.strftime('%Y-%m-%d %H:%M:%S')  # Convert to string
            insert_df['open'] = df['Open']
            insert_df['high'] = df['High']
            insert_df['low'] = df['Low']
            insert_df['close'] = df['Close']
            insert_df['volume'] = df['Volume'].fillna(0).astype(int)
            insert_df['fetch_date'] = df['FetchDate'] if 'FetchDate' in df.columns else datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            insert_df['hour'] = df['hour']
            insert_df['timeframe'] = csv_timeframe
            insert_df['interval'] = csv_interval
            
            # Insert data with INSERT OR REPLACE to handle duplicates
            for _, row in insert_df.iterrows():
                try:
                    conn.execute('''
                        INSERT OR REPLACE INTO market_data 
                        (symbol, asset_type, date, open, high, low, close, volume, 
                         fetch_date, timeframe, interval, hour)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row['symbol'], row['asset_type'], row['date'], 
                        row['open'], row['high'], row['low'], row['close'], 
                        row['volume'], row['fetch_date'], row['timeframe'], 
                        row['interval'], row['hour']
                    ))
                except Exception as e:
                    print(f"  Warning: Failed to insert row: {e}")
            
            conn.commit()
            symbols_processed.update(df['Symbol'].unique())
            total_rows += len(insert_df)
            print(f"  ✓ Stored {len(insert_df)} rows")
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()
            continue
            
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()
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
