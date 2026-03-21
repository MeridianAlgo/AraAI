#!/usr/bin/env python3
"""
Fetch market data and store in database
Supports stocks and forex pairs
Fetches MULTIPLE timeframes to maximize training data
"""

import argparse
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))

# Multiple timeframe configs to maximize data per symbol
STOCK_TIMEFRAMES = [
    ("max", "1d"),  # Max daily history (~20+ years for major stocks)
    ("2y", "1h"),  # 2 years of hourly data (yfinance limit)
    ("5y", "1wk"),  # 5 years of weekly data
]

FOREX_TIMEFRAMES = [
    ("max", "1d"),  # Max daily history
    ("2y", "1h"),  # 2 years hourly
    ("5y", "1wk"),  # 5 years weekly
]


def init_database(db_file):
    """Initialize database with required tables"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create market_data table
    cursor.execute("""
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
    """)

    # Create model_metadata table
    cursor.execute("""
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
    """)

    conn.commit()
    conn.close()


def fetch_and_store(symbol, db_file, asset_type, period="2y", interval="1d"):
    """Fetch data for a symbol and store in database"""
    try:
        ticker_symbol = symbol
        if asset_type == "forex" and not symbol.endswith("=X"):
            ticker_symbol = f"{symbol}=X"

        ticker = yf.Ticker(ticker_symbol)
        df = ticker.history(period=period, interval=interval)

        if df.empty:
            return 0

        df = df.reset_index()
        date_col = "Date" if "Date" in df.columns else "Datetime"

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
                        ticker_symbol if asset_type == "forex" else symbol,
                        (
                            row[date_col].strftime("%Y-%m-%d %H:%M:%S")
                            if hasattr(row[date_col], "strftime")
                            else str(row[date_col])
                        ),
                        float(row["Open"]),
                        float(row["High"]),
                        float(row["Low"]),
                        float(row["Close"]),
                        int(row.get("Volume", 0)),
                        asset_type,
                        period,
                        interval,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ),
                )
                rows_added += 1
            except sqlite3.IntegrityError:
                pass

        conn.commit()
        conn.close()
        return rows_added

    except Exception as e:
        print(f"  Error fetching {symbol} ({period}/{interval}): {e}")
        return 0


def main():
    parser = argparse.ArgumentParser(description="Fetch and store market data")
    parser.add_argument("--db-file", required=True, help="Database file path")
    parser.add_argument("--asset-type", choices=["stock", "forex"], required=True)
    parser.add_argument("--limit", type=int, default=10, help="Number of symbols to fetch")
    parser.add_argument("--period", default=None, help="Data period (overrides multi-timeframe)")
    parser.add_argument(
        "--interval", default=None, help="Data interval (overrides multi-timeframe)"
    )

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
            "JNJ",
            "PG",
            "MA",
            "UNH",
            "HD",
            "DIS",
            "BAC",
            "VZ",
            "ADBE",
            "CMCSA",
            "NFLX",
            "PFE",
            "INTC",
            "KO",
            "PEP",
            "CSCO",
            "ABT",
            "CRM",
            "T",
            "ABBV",
            "CVX",
            "NKE",
            "MRK",
            "MCD",
            "MDT",
            "TXN",
            "HON",
            "BA",
            "UNP",
            "AMGN",
            "IBM",
            "QCOM",
            "ORCL",
            "SBUX",
            "GS",
            "MMM",
            "CAT",
            "GE",
            "F",
            "GM",
            "C",
            "TGT",
            "LMT",
            "DE",
            "LOW",
            "UPS",
            "USB",
            "AXP",
            "MS",
            "WFC",
            "COP",
            "SLB",
            "EOG",
            "OXY",
            "PXD",
            "VLO",
            "MPC",
            "PSX",
            "KMI",
            "WMB",
            "NEE",
            "DUK",
            "SO",
            "D",
            "AEP",
            "EXC",
            "SRE",
            "XEL",
            "PEG",
            "WEC",
            "AMT",
            "PLD",
            "CCI",
            "EQIX",
            "PSA",
            "DLR",
            "O",
            "WELL",
            "SPG",
            "AVB",
            "VRTX",
            "REGN",
            "ISRG",
            "SYK",
            "ZTS",
            "BSX",
            "EW",
            "GILD",
            "BIIB",
            "ILMN",
        ]
        timeframes = STOCK_TIMEFRAMES
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
            "GBPJPY",
            "CHFJPY",
            "EURCHF",
            "AUDJPY",
            "NZDJPY",
            "CADJPY",
            "EURAUD",
            "EURCAD",
            "GBPAUD",
            "GBPCAD",
            "AUDCAD",
            "AUDNZD",
            "EURNZD",
            "GBPNZD",
        ]
        timeframes = FOREX_TIMEFRAMES

    import random

    random.shuffle(symbols)
    symbols = symbols[: args.limit]

    # Determine timeframes to fetch
    if args.period and args.interval:
        # User specified exact period/interval — use only that
        fetch_configs = [(args.period, args.interval)]
    else:
        # Use multi-timeframe strategy for maximum data
        fetch_configs = timeframes

    print(f"Fetching {args.asset_type} data for {len(symbols)} symbols...")
    print(f"Timeframes: {fetch_configs}")

    total_rows = 0
    successful = 0

    for symbol in symbols:
        symbol_rows = 0
        for period, interval in fetch_configs:
            rows = fetch_and_store(symbol, args.db_file, args.asset_type, period, interval)
            symbol_rows += rows

        if symbol_rows > 0:
            print(f"  [OK] {symbol}: {symbol_rows} rows across {len(fetch_configs)} timeframes")
            total_rows += symbol_rows
            successful += 1
        else:
            print(f"  [SKIP] {symbol}: No data")

    print(f"\nSummary: {successful}/{len(symbols)} symbols, {total_rows} total rows stored")

    if total_rows == 0:
        print("[WARNING] No data was stored")
        sys.exit(0)


if __name__ == "__main__":
    main()
