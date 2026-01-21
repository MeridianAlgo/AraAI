#!/usr/bin/env python3
"""
Fetch market data and store in database
Supports stocks and forex pairs
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))


def init_database(db_file):
    """Initialize database with required tables"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create market_data table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS market_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            asset_type TEXT,
            timeframe TEXT,
            interval TEXT,
            fetch_date TEXT,
            UNIQUE(symbol, date, interval)
        )
    """
    )

    # Create model_metadata table
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS model_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            model_type TEXT,
            training_date TEXT,
            accuracy REAL,
            loss REAL,
            epochs INTEGER,
            model_path TEXT,
            timeframe TEXT,
            training_mode TEXT,
            hour INTEGER
        )
    """
    )

    conn.commit()
    conn.close()


def fetch_and_store_stock(symbol, db_file, period="2y", interval="1d"):
    """Fetch stock data and store in database"""
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            print(f"  Warning: No data for {symbol}")
            return 0

        df = df.reset_index()

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        rows_added = 0
        for _, row in df.iterrows():
            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO market_data 
                    (symbol, date, open, high, low, close, volume, asset_type, timeframe, interval, fetch_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        symbol,
                        (
                            row["Date"].strftime("%Y-%m-%d %H:%M:%S")
                            if hasattr(row["Date"], "strftime")
                            else str(row["Date"])
                        ),
                        float(row["Open"]),
                        float(row["High"]),
                        float(row["Low"]),
                        float(row["Close"]),
                        int(row["Volume"]),
                        "stock",
                        period,
                        interval,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                )
                rows_added += 1
            except sqlite3.IntegrityError:
                pass  # Row already exists

        conn.commit()
        conn.close()

        return rows_added

    except Exception as e:
        print(f"  Error fetching {symbol}: {e}")
        return 0


def fetch_and_store_forex(pair, db_file, period="2y", interval="1d"):
    """Fetch forex data and store in database"""
    try:
        symbol = f"{pair}=X" if not pair.endswith("=X") else pair

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            print(f"  Warning: No data for {pair}")
            return 0

        df = df.reset_index()

        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        rows_added = 0
        for _, row in df.iterrows():
            try:
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO market_data 
                    (symbol, date, open, high, low, close, volume, asset_type, timeframe, interval, fetch_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        symbol,
                        (
                            row["Date"].strftime("%Y-%m-%d %H:%M:%S")
                            if hasattr(row["Date"], "strftime")
                            else str(row["Date"])
                        ),
                        float(row["Open"]),
                        float(row["High"]),
                        float(row["Low"]),
                        float(row["Close"]),
                        int(row.get("Volume", 0)),
                        "forex",
                        period,
                        interval,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                )
                rows_added += 1
            except sqlite3.IntegrityError:
                pass  # Row already exists

        conn.commit()
        conn.close()

        return rows_added

    except Exception as e:
        print(f"  Error fetching {pair}: {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Fetch and store market data")
    parser.add_argument("--db-file", required=True, help="Database file path")
    parser.add_argument("--asset-type", choices=["stock", "forex"], required=True)
    parser.add_argument(
        "--limit", type=int, default=10, help="Number of symbols to fetch"
    )
    parser.add_argument("--period", default="2y", help="Data period")
    parser.add_argument("--interval", default="1d", help="Data interval")

    args = parser.parse_args()

    # Initialize database
    init_database(args.db_file)

    # Default symbols
    if args.asset_type == "stock":
        symbols = [
            "AAPL",
            "GOOGL",
            "MSFT",
            "AMZN",
            "TSLA",
            "META",
            "NVDA",
            "JPM",
            "V",
            "WMT",
        ][: args.limit]
    else:
        symbols = [
            "EURUSD",
            "GBPUSD",
            "USDJPY",
            "AUDUSD",
            "USDCAD",
            "NZDUSD",
            "EURGBP",
            "EURJPY",
        ][: args.limit]

    print(f"Fetching {args.asset_type} data for {len(symbols)} symbols...")
    print(f"Period: {args.period}, Interval: {args.interval}")

    total_rows = 0
    successful = 0

    for symbol in symbols:
        print(f"  Fetching {symbol}...")

        if args.asset_type == "stock":
            rows = fetch_and_store_stock(
                symbol, args.db_file, args.period, args.interval
            )
        else:
            rows = fetch_and_store_forex(
                symbol, args.db_file, args.period, args.interval
            )

        if rows > 0:
            print(f"    [OK] Stored {rows} rows")
            total_rows += rows
            successful += 1
        else:
            print("    [SKIP] No new data")

    print(
        f"\nSummary: {successful}/{len(symbols)} symbols, {total_rows} total rows stored"
    )

    if total_rows == 0:
        print("[WARNING] No data was stored")
        sys.exit(1)


if __name__ == "__main__":
    main()
