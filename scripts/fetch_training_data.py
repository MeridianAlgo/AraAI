#!/usr/bin/env python3
"""
Fetch market data for training
Supports stocks and forex pairs
"""

import argparse
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime
import sys


def fetch_stock_data(symbol, period="2y", interval="1d"):
    """Fetch stock data from Yahoo Finance"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            print(f"  Warning: No data for {symbol}")
            return None

        # Reset index to get Date as a column
        df = df.reset_index()

        # Standardize column names
        df = df.rename(columns={"index": "Date"})

        # Ensure Date column exists and is properly named
        if "Date" not in df.columns and "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})

        # Convert Date to string format for CSV storage
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        # Add metadata columns
        df["Symbol"] = symbol
        df["AssetType"] = "stock"
        df["FetchDate"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df["Timeframe"] = period
        df["Interval"] = interval

        # Ensure all required columns exist
        required_cols = [
            "Date",
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "Symbol",
            "AssetType",
            "FetchDate",
            "Timeframe",
            "Interval",
        ]
        for col in required_cols:
            if col not in df.columns:
                print(f"  Warning: Missing column {col}")

        return df
    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return None


def fetch_forex_data(pair, period="2y", interval="1d"):
    """Fetch forex data from Yahoo Finance"""
    try:
        # Convert pair to yfinance format
        symbol = f"{pair}=X" if not pair.endswith("=X") else pair

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            print(f"  Warning: No data for {pair} (symbol: {symbol})")
            return None

        # Reset index to get Date as a column
        df = df.reset_index()

        # Standardize column names
        df = df.rename(columns={"index": "Date"})

        # Ensure Date column exists and is properly named
        if "Date" not in df.columns and "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})

        # Convert Date to string format for CSV storage
        df["Date"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d %H:%M:%S")

        # Add metadata columns
        df["Symbol"] = pair  # Store original pair name, not symbol with =X
        df["AssetType"] = "forex"
        df["FetchDate"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df["Timeframe"] = period
        df["Interval"] = interval

        # Forex often has 0 volume, set to 0 if missing
        if "Volume" not in df.columns:
            df["Volume"] = 0

        return df
    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        print(
            f"  Error fetching {pair} (symbol: {symbol if 'symbol' in locals() else 'unknown'}): {e}"
        )
        print(f"  Details: {error_details}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Fetch market data for training")
    parser.add_argument(
        "--symbols",
        required=True,
        nargs="+",
        help="List of symbols (space-separated) or a single comma-separated string",
    )
    parser.add_argument(
        "--output-dir", required=True, help="Output directory for CSV files"
    )
    parser.add_argument("--period", default="2y", help="Data period (e.g., 1y, 2y, 5y)")
    parser.add_argument("--interval", default="1d", help="Data interval (e.g., 1d, 1h)")
    parser.add_argument("--asset-type", choices=["stock", "forex"], default="stock")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(args.symbols) == 1 and "," in args.symbols[0]:
        symbols = [s.strip() for s in args.symbols[0].split(",") if s.strip()]
    else:
        symbols = [s.strip() for s in args.symbols if s.strip()]

    print(f"Fetching {args.asset_type} data for {len(symbols)} symbols...")
    print(f"Period: {args.period}, Interval: {args.interval}")

    all_data = []

    successful = 0
    failed = 0

    for symbol in symbols:
        print(f"  Fetching {symbol}...", flush=True)

        if args.asset_type == "stock":
            df = fetch_stock_data(symbol, args.period, args.interval)
        else:
            df = fetch_forex_data(symbol, args.period, args.interval)

        if df is not None:
            all_data.append(df)
            # Save individual CSV
            csv_path = output_dir / f"{symbol.replace('=X', '')}.csv"
            df.to_csv(csv_path, index=False)
            print(f"    ✓ Saved {len(df)} rows to {csv_path}", flush=True)
            successful += 1
        else:
            failed += 1
            print(f"    ✗ Failed to fetch {symbol}", flush=True)

    print(
        f"\nSummary: {successful} successful, {failed} failed out of {len(symbols)} symbols"
    )

    if all_data:
        # Save combined CSV
        combined = pd.concat(all_data, ignore_index=True)
        combined_path = output_dir / f"all_{args.asset_type}_data.csv"
        combined.to_csv(combined_path, index=False)
        print(f"\n✓ Saved combined data ({len(combined)} rows) to {combined_path}")
    else:
        print("\n✗ No data fetched")
        sys.exit(1)


if __name__ == "__main__":
    main()
