#!/usr/bin/env python3
"""
Train forex prediction models from database
Supports both full and incremental training
Includes Weights & Biases (wandb) integration for experiment tracking
"""

import argparse
import sqlite3
import pandas as pd
from pathlib import Path
import sys
import warnings
import os

warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from meridianalgo.forex_ml import ForexML  # noqa: E402

# Optional wandb import
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")


def load_forex_data_from_db(
    db_file, pair, use_all_data=True, timeframe="1d", training_mode="full"
):
    """Load forex training data from database with timeframe awareness"""
    conn = sqlite3.connect(db_file)

    # Convert pair to yfinance format
    symbol = f"{pair}=X" if not pair.endswith("=X") else pair

    if use_all_data:
        # Load all historical data for this timeframe
        query = """
            SELECT date, open, high, low, close, volume, timeframe, interval
            FROM market_data
            WHERE symbol = ? AND asset_type = 'forex'
            ORDER BY date ASC
        """
    elif training_mode == "hourly":
        # Load hourly data (last 30 days of hourly data + today's hours)
        query = """
            SELECT date, open, high, low, close, volume, timeframe, interval
            FROM market_data
            WHERE symbol = ? AND asset_type = 'forex'
            AND interval = '1h'
            AND date >= datetime('now', '-30 days')
            ORDER BY date ASC
        """
    else:
        # Load only recent data (last 90 days)
        query = """
            SELECT date, open, high, low, close, volume, timeframe, interval
            FROM market_data
            WHERE symbol = ? AND asset_type = 'forex'
            AND date >= datetime('now', '-90 days')
            ORDER BY date ASC
        """

    df = pd.read_sql_query(query, conn, params=(symbol,))
    conn.close()

    if df.empty:
        raise ValueError(f"No data found for {pair}")

    # Store timeframe info
    detected_timeframe = (
        df["timeframe"].iloc[0] if "timeframe" in df.columns else timeframe
    )
    detected_interval = df["interval"].iloc[0] if "interval" in df.columns else "1d"

    # Rename columns to match expected format
    df.columns = [
        "Date",
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        "Timeframe",
        "Interval",
    ]
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date")

    print(f"  Detected timeframe: {detected_timeframe}, interval: {detected_interval}")

    return df, detected_timeframe, detected_interval


def init_wandb(project, run_name, config, enabled=True):
    """Initialize Weights & Biases tracking"""
    if not enabled or not WANDB_AVAILABLE:
        return None

    try:
        run = wandb.init(project=project, name=run_name, config=config, reinit=True)
        print(f"  ✓ W&B initialized: {run.url}")
        return run
    except Exception as e:
        print(f"  Warning: Failed to initialize wandb: {e}")
        return None


def log_to_wandb(run, metrics):
    """Log metrics to wandb if available"""
    if run is not None and WANDB_AVAILABLE:
        try:
            wandb.log(metrics)
        except Exception as e:
            print(f"  Warning: Failed to log to wandb: {e}")


def train_forex_model(
    pair,
    db_file,
    output_path,
    epochs=100,
    use_all_data=True,
    incremental=False,
    timeframe="1d",
    training_mode="full",
    hour=None,
    wandb_project=None,
    wandb_run_name=None,
):
    """Train forex model for a currency pair with timeframe awareness and wandb tracking"""
    print(f"\n{'='*60}")
    print(f"Training forex model for {pair}")
    print(f"Mode: {training_mode}, Timeframe: {timeframe}")
    if hour is not None:
        print(f"Hour: {hour}:00 UTC")
    print(f"{'='*60}")

    # Load data
    print("Loading data from database...")
    data, detected_timeframe, detected_interval = load_forex_data_from_db(
        db_file, pair, use_all_data, timeframe, training_mode
    )
    print(f"  ✓ Loaded {len(data)} rows")
    print(f"  Date range: {data.index.min()} to {data.index.max()}")

    # Initialize wandb
    wandb_config = {
        "pair": pair,
        "epochs": epochs,
        "timeframe": timeframe,
        "training_mode": training_mode,
        "incremental": incremental,
        "data_rows": len(data),
        "model_type": "forex_ml",
        "asset_type": "forex",
    }

    wandb_enabled = wandb_project is not None and os.environ.get("WANDB_API_KEY")
    run = init_wandb(
        project=wandb_project or "ara-ai",
        run_name=wandb_run_name or f"forex-{pair}",
        config=wandb_config,
        enabled=wandb_enabled,
    )

    # Drop metadata columns before training
    if "Timeframe" in data.columns:
        data = data.drop(columns=["Timeframe", "Interval"])

    # Initialize Forex ML system
    forex_ml = ForexML(model_path=output_path)

    # Train model
    print(f"\nTraining model ({epochs} epochs)...")

    if incremental and Path(output_path).exists():
        print("  Using incremental training mode")
        result = forex_ml.train_ultimate_models(
            target_symbol=pair,
            period="custom",
            custom_data=data,
            epochs=epochs,
            quick_mode=False,
        )
    else:
        print("  Using full training mode")
        result = forex_ml.train_ultimate_models(
            target_symbol=pair,
            period="custom",
            custom_data=data,
            epochs=epochs,
            quick_mode=False,
        )

    if result.get("success"):
        print("\n✓ Training completed successfully")
        print(f"  Final loss: {result.get('final_loss', 'N/A')}")
        print(f"  Timeframe: {timeframe} ({training_mode} mode)")
        print(f"  Model saved to: {output_path}")

        # Log final metrics to wandb
        log_to_wandb(
            run,
            {
                "final_loss": result.get("final_loss", 0),
                "accuracy": result.get("accuracy", 0),
                "training_success": 1,
            },
        )

        # Store metadata in database
        store_model_metadata(
            db_file, pair, output_path, result, timeframe, training_mode, hour
        )

        # Finish wandb run
        if run is not None:
            wandb.finish()

        return True
    else:
        print(f"\n✗ Training failed: {result.get('error', 'Unknown error')}")

        # Log failure to wandb
        log_to_wandb(
            run, {"training_success": 0, "error": result.get("error", "Unknown")}
        )

        if run is not None:
            wandb.finish(exit_code=1)

        return False


def store_model_metadata(
    db_file,
    pair,
    model_path,
    training_result,
    timeframe="1d",
    training_mode="full",
    hour=None,
):
    """Store model metadata in database with timeframe info"""
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Ensure the table has the new columns
    try:
        cursor.execute("ALTER TABLE model_metadata ADD COLUMN timeframe TEXT")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE model_metadata ADD COLUMN training_mode TEXT")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute("ALTER TABLE model_metadata ADD COLUMN hour INTEGER")
    except sqlite3.OperationalError:
        pass

    cursor.execute(
        """
        INSERT INTO model_metadata 
        (symbol, model_type, training_date, accuracy, loss, epochs, model_path,
         timeframe, training_mode, hour)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
        (
            pair,
            "forex_ml",
            pd.Timestamp.now().to_pydatetime(),
            training_result.get("accuracy", 0.0),
            training_result.get("final_loss", 0.0),
            training_result.get("epochs", 0),
            str(model_path),
            timeframe,
            training_mode,
            hour,
        ),
    )

    conn.commit()
    conn.close()


def main():
    parser = argparse.ArgumentParser(description="Train forex prediction model")
    parser.add_argument("--pair", required=True, help="Forex pair (e.g., EURUSD)")
    parser.add_argument("--db-file", required=True, help="SQLite database file")
    parser.add_argument("--output", default="models/forex_model.pt", help="Output model file (default: models/forex_model.pt)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument(
        "--use-all-data", action="store_true", default=True, help="Use all historical data"
    )
    parser.add_argument(
        "--incremental", action="store_true", default=True, help="Incremental training (default: True)"
    )
    parser.add_argument("--timeframe", default="1d", help="Timeframe identifier")
    parser.add_argument(
        "--training-mode", default="full", help="Training mode (full, hourly)"
    )
    parser.add_argument("--hour", type=int, help="Current hour (for hourly mode)")
    parser.add_argument("--wandb-project", help="Weights & Biases project name")
    parser.add_argument("--wandb-run-name", help="Weights & Biases run name")

    args = parser.parse_args()

    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Train model
    success = train_forex_model(
        pair=args.pair,
        db_file=args.db_file,
        output_path=args.output,
        epochs=args.epochs,
        use_all_data=args.use_all_data,
        incremental=args.incremental,
        timeframe=args.timeframe,
        training_mode=args.training_mode,
        hour=args.hour,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
