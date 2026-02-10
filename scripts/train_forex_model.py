#!/usr/bin/env python3
"""
Revolutionary Forex Model Training System
- Optimized for unified forex model
- Comet ML experiment tracking
- Multi-currency pair support
- Advanced technical indicators
"""

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

from meridianalgo.forex_ml import ForexML

# Comet ML integration
try:
    import comet_ml

    COMET_AVAILABLE = True
except ImportError:
    COMET_AVAILABLE = False
    print("Warning: comet_ml not installed. Install with: pip install comet-ml")


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


def load_forex_pairs(db_file, limit=None):
    """Load available forex pairs from database"""
    conn = sqlite3.connect(db_file)
    query = """
        SELECT DISTINCT symbol
        FROM market_data
        WHERE asset_type = 'forex'
        ORDER BY symbol ASC
    """
    if limit:
        query += f" LIMIT {limit}"

    df = pd.read_sql_query(query, conn)
    conn.close()

    pairs = df["symbol"].tolist() if not df.empty else []
    if not pairs:
        raise ValueError("No forex pairs found in database")
    return pairs


def load_forex_data(db_file, pairs, use_all_data=True, timeframe=None):
    """Load forex data for training with optional timeframe filtering"""
    conn = sqlite3.connect(db_file)
    placeholders = ",".join(["?"] * len(pairs))

    # Timeframe to days mapping - increased for sufficient data
    timeframe_days = {
        "15m": 30,  # 30 days for 15-minute (more data needed)
        "1h": 60,  # 60 days for hourly
        "4h": 90,  # 90 days for 4-hour
        "1d": 365,  # 1 year for daily
    }

    if use_all_data and not timeframe:
        query = f"""
            SELECT symbol, date, open, high, low, close, volume
            FROM market_data
            WHERE asset_type = 'forex' AND symbol IN ({placeholders})
            ORDER BY symbol, date ASC
        """
    elif timeframe and timeframe in timeframe_days:
        days = timeframe_days[timeframe]
        query = f"""
            SELECT symbol, date, open, high, low, close, volume
            FROM market_data
            WHERE asset_type = 'forex' AND symbol IN ({placeholders})
            AND date >= datetime('now', '-{days} days')
            ORDER BY symbol, date ASC
        """
        print(f"  Using timeframe: {timeframe} ({days} days of data)")
    else:
        query = f"""
            SELECT symbol, date, open, high, low, close, volume
            FROM market_data
            WHERE asset_type = 'forex' AND symbol IN ({placeholders})
            AND date >= datetime('now', '-90 days')
            ORDER BY symbol, date ASC
        """

    df = pd.read_sql_query(query, conn, params=list(pairs))
    conn.close()

    if df.empty:
        raise ValueError("No forex data found in database")

    # Check if we have sufficient data
    min_rows_per_pair = 30  # Minimum rows needed per pair
    rows_per_pair = len(df) / len(pairs)

    if rows_per_pair < min_rows_per_pair:
        print(
            f"  ⚠️  Warning: Only {rows_per_pair:.0f} rows per pair (minimum {min_rows_per_pair} recommended)"
        )
        print(f"  Fetching all available data instead...")

        # Fallback to all data
        query = f"""
            SELECT symbol, date, open, high, low, close, volume
            FROM market_data
            WHERE asset_type = 'forex' AND symbol IN ({placeholders})
            ORDER BY symbol, date ASC
        """
        df = pd.read_sql_query(query, conn, params=list(pairs))

    print(
        f"  ✓ Loaded {len(df)} rows for {df['symbol'].nunique()} pairs ({len(df) / df['symbol'].nunique():.0f} rows/pair)"
    )
    return df


def train_forex_model(
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
    """Train unified forex model with Comet ML tracking"""
    print(f"\n{'=' * 60}")
    print("Revolutionary Forex Model Training")
    print(f"{'=' * 60}\n")

    start_time = time.time()

    # Load pairs
    all_pairs = load_forex_pairs(db_file)

    # Sample pairs if requested
    if sample_size and sample_size < len(all_pairs):
        rng = random.Random(seed if seed is not None else time.time())
        selected_pairs = rng.sample(all_pairs, sample_size)
        print(f"Randomly selected {len(selected_pairs)} pairs from {len(all_pairs)} available")
    else:
        selected_pairs = all_pairs
        print(f"Training on all {len(selected_pairs)} pairs")

    # Load data
    print("Loading forex data from database...")
    data = load_forex_data(db_file, selected_pairs, use_all_data, timeframe)

    # Initialize Comet ML
    config = {
        "model_type": "unified_forex",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "pairs_count": len(selected_pairs),
        "data_rows": len(data),
        "use_all_data": use_all_data,
        "seed": seed,
        "timeframe": timeframe or "all",
    }

    experiment = init_comet(
        project_name="ara-ai-forex",
        experiment_name=f"forex-unified-{int(time.time())}",
        config=config,
        api_key=comet_api_key,
    )

    # Prepare data format
    data.columns = ["Symbol", "Date", "Open", "High", "Low", "Close", "Volume"]
    data["Date"] = pd.to_datetime(data["Date"])

    # Initialize ML system with Revolutionary 2026 Architecture
    ml = ForexML(model_path=output_path)

    # Train model with Revolutionary 2026 Architecture
    print(
        f"\nTraining unified forex model ({epochs} epochs) with Revolutionary 2026 Architecture..."
    )
    result = ml.train_ultimate_models(
        target_symbol="UNIFIED_FOREX",
        period="custom",
        custom_data=data,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        quick_mode=False,
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
        print("\n✓ Training completed successfully")
        print(f"  Final loss: {result.get('final_loss', 'N/A')}")
        print(f"  Training time: {training_time:.2f}s")
        print(f"  Pairs trained: {len(selected_pairs)}")
        print(f"  Model saved to: {output_path}")
        return True
    else:
        print(f"\n✗ Training failed: {result.get('error', 'Unknown error')}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train unified forex prediction model")
    parser.add_argument("--db-file", required=True, help="SQLite database file")
    parser.add_argument("--output", default="models/Forex_Pred.pt", help="Output model path")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--sample-size", type=int, help="Number of pairs to sample (default: all)")
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
        choices=["15m", "1h", "4h", "1d"],
        help="Timeframe for data filtering",
    )

    args = parser.parse_args()

    # Get Comet API key from args or environment
    comet_api_key = args.comet_api_key or os.environ.get("COMET_API_KEY")
    if comet_api_key:
        comet_api_key = comet_api_key.strip()

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Train model
    success = train_forex_model(
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
