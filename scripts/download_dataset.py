"""
Download historical data and save as dataset CSV
"""

import sys
import argparse
from pathlib import Path
import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent))

from meridianalgo.console import ConsoleManager

def main():
    parser = argparse.ArgumentParser(description='Download historical data as dataset CSV')
    parser.add_argument('symbol', help='Stock symbol or forex pair (e.g., AAPL, EURUSD)')
    parser.add_argument('--period', default='5y', help='Period to download (default: 5y)')
    parser.add_argument('--output', help='Output CSV file (default: datasets/SYMBOL.csv)')
    parser.add_argument('--type', choices=['stock', 'forex'], default='stock', help='Data type')
    
    args = parser.parse_args()
    
    console = ConsoleManager()
    console.print_header("ARA AI - Dataset Downloader")
    
    # Determine output file
    if args.output:
        output_file = Path(args.output)
    else:
        datasets_dir = Path('datasets')
        datasets_dir.mkdir(exist_ok=True)
        output_file = datasets_dir / f"{args.symbol.upper()}.csv"
    
    # Convert forex symbol if needed
    symbol = args.symbol
    if args.type == 'forex':
        if '=' not in symbol:
            symbol = f"{symbol}=X"
    
    console.print_info(f"Downloading {args.symbol} ({args.period})...")
    console.print_info(f"Type: {args.type}")
    
    try:
        # Download data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=args.period)
        
        if len(data) == 0:
            console.print_error("No data downloaded!")
            return
        
        console.print_success(f"Downloaded {len(data)} days of data")
        console.print_info(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        # Save to CSV
        data.to_csv(output_file)
        console.print_success(f"Saved to: {output_file}")
        
        # Show sample
        print("\nFirst 5 rows:")
        print(data.head())
        
        print("\nLast 5 rows:")
        print(data.tail())
        
        console.print_info(f"\nTo train models on this dataset:")
        console.print_info(f"python train_from_dataset.py {output_file} --type {args.type} --name {args.symbol}")
        
    except Exception as e:
        console.print_error(f"Download failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
