#!/usr/bin/env python3
"""
Fetch market data for training
Supports both full historical and incremental refresh modes
"""

import argparse
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import sys
from pathlib import Path

def fetch_data(symbols, asset_type='stock', mode='full', period='2y', interval='1d', timeframe='1d'):
    """
    Fetch market data for given symbols
    
    Args:
        symbols: List of symbols to fetch
        asset_type: 'stock' or 'forex'
        mode: 'full' (all historical), 'hourly' (1 hour), or 'refresh' (recent only)
        period: Data period for full mode
        interval: Data interval ('1d', '1h', '5m', etc.)
        timeframe: Timeframe identifier for database
    """
    all_data = []
    
    for symbol in symbols:
        try:
            # Convert forex symbols to yfinance format
            if asset_type == 'forex':
                if not symbol.endswith('=X'):
                    symbol = f"{symbol}=X"
            
            print(f"Fetching {symbol} ({interval} interval)...")
            
            # Determine period and interval based on mode
            if mode == 'hourly':
                # Fetch last 2 hours of data (to ensure we get the latest hour)
                fetch_period = '2h'
                fetch_interval = '1h'
            elif mode == 'refresh':
                # Only fetch last 7 days for refresh
                fetch_period = '7d'
                fetch_interval = interval
            else:
                # Full historical data
                fetch_period = period
                fetch_interval = interval
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=fetch_period, interval=fetch_interval)
            
            if data.empty:
                print(f"Warning: No data for {symbol}")
                continue
            
            # Add metadata columns
            data['Symbol'] = symbol
            data['AssetType'] = asset_type
            data['FetchDate'] = datetime.now().isoformat()
            data['Timeframe'] = timeframe
            data['Interval'] = fetch_interval
            
            # Reset index to make Date/Datetime a column
            data = data.reset_index()
            
            # Rename Datetime column to Date if it exists
            if 'Datetime' in data.columns:
                data = data.rename(columns={'Datetime': 'Date'})
            
            all_data.append(data)
            print(f"  ✓ Fetched {len(data)} rows for {symbol}")
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            continue
    
    if not all_data:
        print("Error: No data fetched")
        sys.exit(1)
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

def main():
    parser = argparse.ArgumentParser(description='Fetch market data for training')
    parser.add_argument('--symbols', nargs='+', required=True, help='Symbols to fetch')
    parser.add_argument('--asset-type', choices=['stock', 'forex'], default='stock')
    parser.add_argument('--mode', choices=['full', 'hourly', 'refresh'], default='full')
    parser.add_argument('--period', default='2y', help='Period for full mode (e.g., 1y, 2y, 5y)')
    parser.add_argument('--interval', default='1d', help='Data interval (1d, 1h, 5m, etc.)')
    parser.add_argument('--timeframe', default='1d', help='Timeframe identifier')
    parser.add_argument('--output', required=True, help='Output CSV file')
    
    args = parser.parse_args()
    
    print(f"Fetching {args.asset_type} data in {args.mode} mode...")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Interval: {args.interval}, Timeframe: {args.timeframe}")
    
    # Fetch data
    data = fetch_data(
        symbols=args.symbols,
        asset_type=args.asset_type,
        mode=args.mode,
        period=args.period,
        interval=args.interval,
        timeframe=args.timeframe
    )
    
    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved {len(data)} rows to {output_path}")
    print(f"  Date range: {data['Date'].min()} to {data['Date'].max()}")
    print(f"  Symbols: {data['Symbol'].nunique()}")

if __name__ == '__main__':
    main()
