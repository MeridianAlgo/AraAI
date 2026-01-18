#!/usr/bin/env python3
"""
Train unified models - ONE model for all stocks, ONE for all forex
Much more efficient than training separate models per ticker
"""

import argparse
import sqlite3
import pandas as pd
from pathlib import Path
import sys
import warnings
import time
import random

warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meridianalgo.unified_ml import UnifiedStockML  # noqa: E402
from meridianalgo.forex_ml import ForexML  # noqa: E402


def load_stock_symbols(db_file):
    """Load available stock symbols from database."""
    conn = sqlite3.connect(db_file)
    query = """
        SELECT DISTINCT symbol
        FROM market_data
        WHERE asset_type = 'stock'
        ORDER BY symbol ASC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    symbols = df["symbol"].tolist() if not df.empty else []
    if not symbols:
        raise ValueError("No stock symbols found in database")
    return symbols


def load_stock_data_for_symbols(db_file, symbols, limit=None):
    """Load OHLCV rows for a subset of stock symbols from database."""
    if not symbols:
        raise ValueError("No symbols provided")

    conn = sqlite3.connect(db_file)
    placeholders = ",".join(["?"] * len(symbols))
    query = f"""
        SELECT symbol, date, open, high, low, close, volume
        FROM market_data
        WHERE asset_type = 'stock'
          AND symbol IN ({placeholders})
        ORDER BY symbol, date ASC
    """

    if limit:
        query += f" LIMIT {limit}"

    df = pd.read_sql_query(query, conn, params=list(symbols))
    conn.close()

    if df.empty:
        raise ValueError("No stock data found in database for selected symbols")

    print(f"  ✓ Loaded {len(df)} rows for {df['symbol'].nunique()} stocks")
    return df


def load_all_forex_data(db_file, limit=None):
    """Load data for all forex pairs from database"""
    conn = sqlite3.connect(db_file)

    query = """
        SELECT symbol, date, open, high, low, close, volume
        FROM market_data
        WHERE asset_type = 'forex'
        ORDER BY symbol, date ASC
    """

    if limit:
        query += f" LIMIT {limit}"

    df = pd.read_sql_query(query, conn)
    conn.close()

    if df.empty:
        raise ValueError("No forex data found in database")

    print(f"  ✓ Loaded {len(df)} rows for {df['symbol'].nunique()} forex pairs")
    return df


def train_unified_stock_model(
    db_file, output_path, epochs=500, sample_size=1, seed=None
):
    """Train ONE model for ALL stocks"""
    print(f"\n{'=' * 60}")
    print("Training Unified Stock Model")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    all_symbols = load_stock_symbols(db_file)
    rng = random.Random(seed if seed is not None else time.time())
    effective_size = min(sample_size, len(all_symbols))
    selected_symbols = (
        all_symbols
        if effective_size == len(all_symbols)
        else rng.sample(all_symbols, effective_size)
    )
    # Don't sort - keep random order for training
    print(f"Randomly selected {len(selected_symbols)} symbol(s) from {len(all_symbols)} available")

    print(
        "Loading stock data from database "
        f"(sample_size={effective_size}, total_symbols={len(all_symbols)})..."
    )
    data = load_stock_data_for_symbols(db_file, selected_symbols)

    print(
        f"Training on {len(selected_symbols)} stocks: "
        f"{', '.join(selected_symbols[:5])}{'...' if len(selected_symbols) > 5 else ''}"
    )

    # Initialize ML system
    ml = UnifiedStockML(model_path=output_path)

    # Train on combined data
    print(f"\nTraining unified model ({epochs} epochs)...")

    # Convert to format expected by training
    data.columns = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]
    data["Date"] = pd.to_datetime(data["Date"])

    # Train on all data at once
    result = ml.train_ultimate_models(
        target_symbol="UNIFIED_STOCKS",
        period="custom",
        custom_data=data,
        epochs=epochs,
        quick_mode=False,
    )

    training_time = time.time() - start_time

    if result.get("success"):
        print("\n✓ Training completed successfully")
        print(f"  Final loss: {result.get('final_loss', 'N/A')}")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Stocks trained: {len(selected_symbols)}")
        print(f"  Model saved to: {output_path}")
        return True
    else:
        print(f"\n✗ Training failed: {result.get('error', 'Unknown error')}")
        return False


def train_unified_forex_model(db_file, output_path, epochs=500):
    """Train ONE model for ALL forex pairs"""
    print(f"\n{'=' * 60}")
    print("Training Unified Forex Model")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    # Load all forex data
    print("Loading all forex data from database...")
    data = load_all_forex_data(db_file)

    # Group by symbol and prepare for training
    pairs = data["symbol"].unique()
    print(f"Training on {len(pairs)} forex pairs: {', '.join(pairs)}")

    # Initialize ML system
    forex_ml = ForexML(model_path=output_path)

    # Train on combined data
    print(f"\nTraining unified model ({epochs} epochs)...")

    # Convert to format expected by training
    data.columns = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]
    data["Date"] = pd.to_datetime(data["Date"])

    # Train on all data at once
    result = forex_ml.train_ultimate_models(
        target_symbol="UNIFIED_FOREX",
        period="custom",
        custom_data=data,
        epochs=epochs,
        quick_mode=False,
    )

    training_time = time.time() - start_time

    if result.get("success"):
        print("\n✓ Training completed successfully")
        print(f"  Final loss: {result.get('final_loss', 'N/A')}")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Forex pairs trained: {len(pairs)}")
        print(f"  Model saved to: {output_path}")
        return True
    else:
        print(f"\n✗ Training failed: {result.get('error', 'Unknown error')}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Train unified models - ONE for all stocks, ONE for all forex"
    )
    parser.add_argument("--db-file", required=True, help="SQLite database file")
    parser.add_argument(
        "--stock-output",
        default="models/unified_stock_model.pt",
        help="Output path for unified stock model",
    )
    parser.add_argument(
        "--forex-output",
        default="models/unified_forex_model.pt",
        help="Output path for unified forex model",
    )
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs")
    parser.add_argument(
        "--stock-sample-size",
        type=int,
        default=1,
        help="Number of stock symbols to sample per run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for stock sampling (optional)",
    )
    parser.add_argument(
        "--stocks-only", action="store_true", help="Train only stock model"
    )
    parser.add_argument(
        "--forex-only", action="store_true", help="Train only forex model"
    )

    args = parser.parse_args()

    # Create output directories
    Path(args.stock_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.forex_output).parent.mkdir(parents=True, exist_ok=True)

    success = True

    # Train stock model
    if not args.forex_only:
        stock_success = train_unified_stock_model(
            args.db_file,
            args.stock_output,
            args.epochs,
            sample_size=args.stock_sample_size,
            seed=args.seed,
        )
        success = success and stock_success

    # Train forex model
    if not args.stocks_only:
        forex_success = train_unified_forex_model(
            args.db_file, args.forex_output, args.epochs
        )
        success = success and forex_success

    print(f"\n{'=' * 60}")
    print("Training Summary")
    print(f"{'=' * 60}")
    print(f"Stock Model: {args.stock_output}")
    print(f"Forex Model: {args.forex_output}")
    print(f"Status: {'✓ Success' if success else '✗ Failed'}")
    print(f"{'=' * 60}\n")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
