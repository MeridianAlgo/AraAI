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

def fetch_data(symbols, asset_type='stock', mode='full', period='2y'):
    """
    Fetch market data for given symbols
    
    Args:
        symbols: List of symbols to fetch
        asset_type: 'stock' or 'forex'
        mode: 'full' (all historical) or 'refresh' (recent only)
        period: Data period for full mode
    """
    all_data = []
    
    for symbol in symbols:
        try:
            # Convert forex symbols to yfinance format
            if asset_type == 'forex':
                if not symbol.endswith('=X'):
                    symbol = f"{symbol}=X"
            
            print(f"Fetching {symbol}...")
            
            # Determine period based on mode
            if mode == 'refresh':
                # Only fetch last 7 days for refresh
                fetch_period = '7d'
            else:
                # Full historical data
                fetch_period = period
            
            # Fetch data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=fetch_period)
            
            if data.empty:
                print(f"Warning: No data for {symbol}")
                continue
            
            # Add symbol column
            data['Symbol'] = symbol
            data['AssetType'] = asset_type
            data['FetchDate'] = datetime.now().isoformat()
            
            # Reset index to make Date a column
            data = data.reset_index()
            
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
    parser.add_argument('--mode', choices=['full', 'refresh'], default='full')
    parser.add_argument('--period', default='2y', help='Period for full mode (e.g., 1y, 2y, 5y)')
    parser.add_argument('--output', required=True, help='Output CSV file')
    
    args = parser.parse_args()
    
    print(f"Fetching {args.asset_type} data in {args.mode} mode...")
    print(f"Symbols: {', '.join(args.symbols)}")
    
    # Fetch data
    data = fetch_data(
        symbols=args.symbols,
        asset_type=args.asset_type,
        mode=args.mode,
        period=args.period
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
