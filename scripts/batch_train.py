#!/usr/bin/env python3
"""
Batch Training Script - Train multiple models in sequence
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_model import train_model
from scripts.fetch_training_data import main as fetch_data
from scripts.store_training_data import main as store_data


def batch_train(symbols, epochs=50, period="2y", interval="1d"):
    """Train multiple models in batch"""

    print(f"\n{'='*70}")
    print("BATCH TRAINING SESSION")
    print(f"{'='*70}")
    print(f"Symbols:  {', '.join(symbols)}")
    print(f"Epochs:   {epochs}")
    print(f"Period:   {period}")
    print(f"Interval: {interval}")
    print(f"Started:  {datetime.now()}")
    print(f"{'='*70}\n")

    total_start = time.time()
    results = []

    for i, symbol in enumerate(symbols, 1):
        print(f"\n{'='*70}")
        print(f"Training {i}/{len(symbols)}: {symbol}")
        print(f"{'='*70}\n")

        symbol_start = time.time()

        try:
            # Fetch data
            print(f"[1/3] Fetching data for {symbol}...")
            sys.argv = [
                "fetch_training_data.py",
                "--symbols",
                symbol,
                "--output-dir",
                "datasets/training_data",
                "--period",
                period,
                "--interval",
                interval,
                "--asset-type",
                "stock",
            ]
            try:
                fetch_data()
            except SystemExit:
                pass

            # Store data
            print("[2/3] Storing data in database...")
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

            # Train model
            print("[3/3] Training model...")
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

            symbol_time = time.time() - symbol_start

            if success:
                print(f"\n✓ {symbol} trained successfully in {symbol_time:.2f}s")
                results.append({"symbol": symbol, "success": True, "time": symbol_time})
            else:
                print(f"\n✗ {symbol} training failed after {symbol_time:.2f}s")
                results.append(
                    {"symbol": symbol, "success": False, "time": symbol_time}
                )

        except Exception as e:
            symbol_time = time.time() - symbol_start
            print(f"\n✗ {symbol} error: {e}")
            results.append(
                {
                    "symbol": symbol,
                    "success": False,
                    "time": symbol_time,
                    "error": str(e),
                }
            )

    # Summary
    total_time = time.time() - total_start
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print(f"\n{'='*70}")
    print("BATCH TRAINING SUMMARY")
    print(f"{'='*70}")
    print(f"Total Models:     {len(symbols)}")
    print(f"Successful:       {successful}")
    print(f"Failed:           {failed}")
    print(f"Total Time:       {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Avg Time/Model:   {total_time/len(symbols):.2f}s")
    print(f"Completed:        {datetime.now()}")
    print(f"{'='*70}\n")

    # Detailed results
    print("DETAILED RESULTS:")
    print(f"{'Symbol':<10} {'Status':<10} {'Time':<10}")
    print("-" * 30)
    for r in results:
        status = "✓ Success" if r["success"] else "✗ Failed"
        print(f"{r['symbol']:<10} {status:<10} {r['time']:.2f}s")

    print(f"\n{'='*70}\n")

    return successful == len(symbols)


def main():
    parser = argparse.ArgumentParser(description="Batch train multiple models")
    parser.add_argument(
        "--symbols", nargs="+", help="Stock symbols to train (space-separated)"
    )
    parser.add_argument("--file", help="File containing symbols (one per line)")
    parser.add_argument(
        "--random", type=int, help="Train N random symbols from all_tickers.txt"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument(
        "--period", default="2y", help="Data period (e.g., 2y, 1y, 6mo)"
    )
    parser.add_argument("--interval", default="1d", help="Data interval (e.g., 1d, 1h)")

    args = parser.parse_args()

    # Determine symbols to train
    symbols = []

    if args.symbols:
        symbols = args.symbols
    elif args.file:
        with open(args.file, "r") as f:
            symbols = [line.strip() for line in f if line.strip()]
    elif args.random:
        import random

        with open("all_tickers.txt", "r") as f:
            all_symbols = [line.strip() for line in f if line.strip()]
        symbols = random.sample(all_symbols, min(args.random, len(all_symbols)))
    else:
        # Default: train popular stocks
        symbols = ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
        print("No symbols specified, using default: AAPL, GOOGL, MSFT, TSLA, AMZN")

    if not symbols:
        print("Error: No symbols to train!")
        sys.exit(1)

    success = batch_train(
        symbols=symbols, epochs=args.epochs, period=args.period, interval=args.interval
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
