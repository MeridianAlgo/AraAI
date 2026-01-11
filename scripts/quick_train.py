#!/usr/bin/env python3
"""
Quick training script for testing and timing
Trains a single model to measure performance
"""

import sys
import time
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_quick_training(symbol="AAPL", epochs=50):
    """Run a quick training session and measure time"""

    print(f"\n{'='*60}")
    print(f"Quick Training Test - {symbol}")
    print(f"Started at: {datetime.now()}")
    print(f"{'='*60}\n")

    start_time = time.time()

    # Import here to measure actual training time
    from scripts.fetch_training_data import main as fetch_data
    from scripts.store_training_data import main as store_data
    from scripts.train_model import train_model

    # Step 1: Fetch data
    print("Step 1: Fetching data...")
    step1_start = time.time()
    sys.argv = [
        "fetch_training_data.py",
        "--symbols",
        symbol,
        "--output-dir",
        "datasets/training_data",
        "--period",
        "2y",
        "--interval",
        "1d",
        "--asset-type",
        "stock",
    ]
    try:
        fetch_data()
    except SystemExit:
        pass
    step1_time = time.time() - step1_start
    print(f"✓ Data fetched in {step1_time:.2f} seconds\n")

    # Step 2: Store in database
    print("Step 2: Storing in database...")
    step2_start = time.time()
    sys.argv = [
        "store_training_data.py",
        "--data-dir",
        "datasets/training_data",
        "--db-file",
        "training.db",
    ]
    try:
        store_data()
    except SystemExit:
        pass
    step2_time = time.time() - step2_start
    print(f"✓ Data stored in {step2_time:.2f} seconds\n")

    # Step 3: Train model
    print(f"Step 3: Training model ({epochs} epochs)...")
    step3_start = time.time()

    success = train_model(
        symbol=symbol,
        db_file="training.db",
        output_path=f"models/stock_{symbol}.pt",
        epochs=epochs,
        use_all_data=True,
        incremental=False,
        timeframe="1d",
        training_mode="full",
    )

    step3_time = time.time() - step3_start
    print(f"✓ Model trained in {step3_time:.2f} seconds\n")

    # Total time
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"Symbol:           {symbol}")
    print(f"Epochs:           {epochs}")
    print(f"Data Fetch:       {step1_time:.2f}s")
    print(f"Data Storage:     {step2_time:.2f}s")
    print(f"Model Training:   {step3_time:.2f}s")
    print(f"{'='*60}")
    print(f"TOTAL TIME:       {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"{'='*60}")
    print(f"Completed at: {datetime.now()}")
    print(f"{'='*60}\n")

    if success:
        print("✓ Training completed successfully!")
        print(f"Model saved to: models/stock_{symbol}.pt")
    else:
        print("✗ Training failed!")
        return False

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick training test")
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol to train")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")

    args = parser.parse_args()

    success = run_quick_training(args.symbol, args.epochs)
    sys.exit(0 if success else 1)
