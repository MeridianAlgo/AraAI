#!/usr/bin/env python3
"""
Revolutionary Stock Model Training System
- Optimized for single unified stock model
- Comet ML experiment tracking
- Advanced feature engineering
- Efficient batch processing
"""

# Comet ML integration (must be before torch)
try:
    import comet_ml

    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("Warning: comet_ml not installed. Install with: pip install comet-ml")


import argparse
import os
import random
import sqlite3
import sys
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meridianalgo.unified_ml import UnifiedStockML


def init_comet(project_name, experiment_name, config, api_key=None):
    """Initialize Comet ML experiment tracking"""
    if not COMET_AVAILABLE or not api_key:
        return None

    try:
        experiment = comet_ml.Experiment(
            api_key=api_key, project_name=project_name, workspace="meridianalgo"
        )
        experiment.set_name(experiment_name)
        experiment.log_parameters(config)
        print(f"  Comet ML initialized: {experiment.url}")
        return experiment
    except Exception as e:
        print(f"  Warning: Failed to initialize Comet ML: {e}")
        return None


def load_stock_symbols(db_file, limit=None):
    """Load available stock symbols from database"""
    conn = sqlite3.connect(db_file)
    query = """
        SELECT DISTINCT symbol
        FROM market_data
        WHERE asset_type = 'stock'
        ORDER BY symbol ASC
    """
    if limit:
        query += f" LIMIT {limit}"

    df = pd.read_sql_query(query, conn)
    conn.close()

    symbols = df["symbol"].tolist() if not df.empty else []
    if not symbols:
        raise ValueError("No stock symbols found in database")
    return symbols


def load_stock_data(db_file, symbols, use_all_data=True, timeframe=None):
    """Load stock data for training with optional timeframe filtering"""
    conn = sqlite3.connect(db_file)
    placeholders = ",".join(["?"] * len(symbols))

    # Timeframe to days mapping - increased for sufficient data
    timeframe_days = {
        "1h": 30,  # 30 days for hourly
        "4h": 60,  # 60 days for 4-hour
        "1d": 365,  # 1 year for daily
        "1w": 730,  # 2 years for weekly
    }

    if use_all_data and not timeframe:
        query = f"""
            SELECT symbol, date, open, high, low, close, volume
            FROM market_data
            WHERE asset_type = 'stock' AND symbol IN ({placeholders})
            ORDER BY symbol, date ASC
        """
    elif timeframe and timeframe in timeframe_days:
        days = timeframe_days[timeframe]
        query = f"""
            SELECT symbol, date, open, high, low, close, volume
            FROM market_data
            WHERE asset_type = 'stock' AND symbol IN ({placeholders})
            AND date >= datetime('now', '-{days} days')
            ORDER BY symbol, date ASC
        """
        print(f"  Using timeframe: {timeframe} ({days} days of data)")
    else:
        query = f"""
            SELECT symbol, date, open, high, low, close, volume
            FROM market_data
            WHERE asset_type = 'stock' AND symbol IN ({placeholders})
            AND date >= datetime('now', '-90 days')
            ORDER BY symbol, date ASC
        """

    df = pd.read_sql_query(query, conn, params=list(symbols))

    if df.empty:
        raise ValueError("No stock data found in database")

    # Check if we have sufficient data
    min_rows_per_symbol = 30  # Minimum rows needed per symbol
    rows_per_symbol = len(df) / len(symbols)

    if rows_per_symbol < min_rows_per_symbol:
        print(
            f"  ⚠️  Warning: Only {rows_per_symbol:.0f} rows per symbol (minimum {min_rows_per_symbol} recommended)"
        )
        print("  Fetching all available data instead...")

        # Fallback to all data
        query = f"""
            SELECT symbol, date, open, high, low, close, volume
            FROM market_data
            WHERE asset_type = 'stock' AND symbol IN ({placeholders})
            ORDER BY symbol, date ASC
        """
        df = pd.read_sql_query(query, conn, params=list(symbols))

    print(
        f"  ✓ Loaded {len(df)} rows for {df['symbol'].nunique()} stocks ({len(df) / df['symbol'].nunique():.0f} rows/stock)"
    )
    conn.close()
    return df


def train_stock_model(
    db_file,
    output_path,
    epochs=500,
    batch_size=64,
    lr=0.0005,
    sample_size=None,
    use_all_data=True,
    comet_api_key=None,
    seed=None,
    timeframe=None,
):
    """Train unified stock model with Comet ML tracking"""
    print(f"\n{'=' * 60}")
    print("Revolutionary Stock Model Training")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    # Load symbols
    all_symbols = load_stock_symbols(db_file)

    # Sample symbols if requested
    if sample_size and sample_size < len(all_symbols):
        rng = random.Random(seed if seed is not None else time.time())
        selected_symbols = rng.sample(all_symbols, sample_size)
        print(
            f"Randomly selected {len(selected_symbols)} symbols from {len(all_symbols)} available"
        )
    else:
        selected_symbols = all_symbols
        print(f"Training on all {len(selected_symbols)} symbols")

    # Load data
    print("Loading stock data from database...")
    data = load_stock_data(db_file, selected_symbols, use_all_data, timeframe)

    # Initialize Comet ML
    config = {
        "model_type": "Stock_Pred",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "symbols_count": len(selected_symbols),
        "data_rows": len(data),
        "use_all_data": use_all_data,
        "seed": seed,
        "timeframe": timeframe or "all",
        "direction_loss": True,
    }

    experiment = init_comet(
        project_name="meridian-algo-stocks",
        experiment_name=f"MeridianAlgo_Stocks_{config['timeframe']}_{int(time.time())}",
        config=config,
        api_key=comet_api_key,
    )

    # Prepare data format
    data.columns = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]
    data["Date"] = pd.to_datetime(data["Date"])

    # Initialize ML system with Revolutionary 2026 Architecture
    ml = UnifiedStockML(model_path=output_path)

    # Train model with Revolutionary 2026 Architecture
    print(
        f"\nTraining unified stock model ({epochs} epochs) with Revolutionary 2026 Architecture..."
    )
    result = ml.train_ultimate_models(
        target_symbol="UNIFIED_STOCKS",
        period="custom",
        custom_data=data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        quick_mode=False,
        comet_experiment=experiment,  # Pass Comet experiment
    )

    training_time = time.time() - start_time

    # Log results to Comet ML
    if experiment:
        experiment.log_metrics(
            {
                "final_loss": result.get("final_loss", 0),
                "accuracy": result.get("accuracy", 0),
                "training_time": training_time,
                "success": 1 if result.get("success") else 0,
            }
        )
        experiment.end()

    if result.get("success"):
        print("\nTraining completed successfully")
        print(f"  Final loss: {result.get('final_loss', 'N/A')}")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Stocks trained: {len(selected_symbols)}")
        print(f"  Model saved to: {output_path}")
        return True
    else:
        print(f"\nTraining failed: {result.get('error', 'Unknown error')}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train unified stock prediction model")
    parser.add_argument("--db-file", required=True, help="SQLite database file")
    parser.add_argument(
        "--output", default="models/MeridianAlgo_Stocks.pt", help="Output model path"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument(
        "--sample-size", type=int, help="Number of symbols to sample (default: all)"
    )
    parser.add_argument(
        "--use-all-data",
        action="store_true",
        default=True,
        help="Use all historical data",
    )
    parser.add_argument("--comet-api-key", help="Comet ML API key")
    parser.add_argument("--seed", type=int, help="Random seed for sampling")
    parser.add_argument(
        "--timeframe",
        choices=["1h", "4h", "1d", "1w"],
        help="Timeframe for data filtering (will be randomized if not specified)",
    )

    args = parser.parse_args()

    # Randomize timeframe if not specified
    if not args.timeframe:
        args.timeframe = random.choice(["1h", "4h", "1d", "1w"])
        print(f"Randomly selected timeframe: {args.timeframe}")

    # Get Comet API key from args or environment
    comet_api_key = args.comet_api_key or os.environ.get("COMET_API_KEY")
    if comet_api_key:
        comet_api_key = comet_api_key.strip()

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Train model
    success = train_stock_model(
        db_file=args.db_file,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        sample_size=args.sample_size,
        use_all_data=args.use_all_data,
        comet_api_key=comet_api_key,
        seed=args.seed,
        timeframe=args.timeframe,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
