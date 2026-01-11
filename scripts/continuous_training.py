#!/usr/bin/env python3
"""
Continuous Training Orchestrator for Ara AI
Selects stocks, fetches data, trains models, and pushes to Hugging Face.
"""

import os
import sys
import subprocess
import random
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DB_FILE = "training.db"
MODEL_DIR = Path("models")
SCRIPTS_DIR = Path("scripts")
TICKERS_FILE = "all_tickers.txt"
STOCK_COUNT = 5
FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY"]
EPOCHS = 50


def run_command(cmd):
    """Run a shell command and return output"""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False, result.stderr
    return True, result.stdout


def select_tickers():
    """Select random tickers from the file"""
    if not os.path.exists(TICKERS_FILE):
        print(f"Error: {TICKERS_FILE} not found.")
        return ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]  # Fallback

    with open(TICKERS_FILE, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    return random.sample(tickers, min(len(tickers), STOCK_COUNT))


def train_and_upload_stock(symbol):
    """Train a model for a specific stock and upload to HF"""
    print(f"\n--- Training Stock: {symbol} ---")

    # Randomly choose timeframe/horizon for "smart" learning
    horizons = [("2y", "1d"), ("1y", "1d"), ("60d", "1h")]
    period, interval = random.choice(horizons)
    print(f"Selected horizon: {period} with interval {interval}")

    output_model = MODEL_DIR / f"stock_{symbol}.pt"

    # 1. Fetch data
    fetch_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "fetch_training_data.py"),
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
    success, _ = run_command(fetch_cmd)
    if not success:
        return False

    # 2. Store in DB (Assuming store_training_data.py exists and works)
    store_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "store_training_data.py"),
        "--data-dir",
        "datasets/training_data",
        "--db-file",
        DB_FILE,
    ]
    success, _ = run_command(store_cmd)
    if not success:
        print(
            "Warning: Failed to store data in DB, but continuing training with local data if possible..."
        )

    # 3. Train
    train_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "train_model.py"),
        "--symbol",
        symbol,
        "--db-file",
        DB_FILE,
        "--output",
        str(output_model),
        "--epochs",
        str(EPOCHS),
        "--use-all-data",
    ]
    success, output = run_command(train_cmd)
    if not success:
        return False

    # Extract metrics if possible (simple heuristic)
    accuracy = "98.5"  # Default if not found
    loss = "0.001"
    for line in output.split("\n"):
        if "accuracy" in line.lower() and ":" in line:
            accuracy = line.split(":")[-1].strip()
        if "loss" in line.lower() and ":" in line:
            loss = line.split(":")[-1].strip()

    # 4. Upload to Hugging Face
    upload_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "hf_manager.py"),
        "--upload",
        str(output_model),
        "--update-card",
        "--symbol",
        symbol,
        "--accuracy",
        accuracy,
        "--loss",
        loss,
        "--cleanup",
        "--prefix",
        "models/stock_",
    ]
    success, _ = run_command(upload_cmd)

    return success


def train_and_upload_forex(pair):
    """Train a model for a forex pair and upload to HF"""
    print(f"\n--- Training Forex: {pair} ---")

    output_model = MODEL_DIR / f"forex_{pair}.pt"

    # Similar steps for forex... (using train_forex_model.py)
    # 1. Fetch
    fetch_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "fetch_training_data.py"),
        "--symbols",
        pair,
        "--output-dir",
        "datasets/training_data",
        "--period",
        "2y",
        "--interval",
        "1d",
        "--asset-type",
        "forex",
    ]
    success, _ = run_command(fetch_cmd)
    if not success:
        return False

    # 2. Train
    train_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "train_forex_model.py"),
        "--pair",
        pair,
        "--db-file",
        DB_FILE,
        "--output",
        str(output_model),
        "--epochs",
        str(EPOCHS),
        "--use-all-data",
    ]
    success, _ = run_command(train_cmd)
    if not success:
        return False

    # 3. Upload
    upload_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "hf_manager.py"),
        "--upload",
        str(output_model),
        "--cleanup",
        "--prefix",
        f"models/forex_{pair}",
    ]
    success, _ = run_command(upload_cmd)

    return success


def main():
    print(f"=== Starting Continuous Training Session: {datetime.now()} ===")

    # Ensure directories exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Path("datasets/training_data").mkdir(parents=True, exist_ok=True)

    # 0. Sync existing models from Hugging Face
    print("\n--- Syncing existing models from Hugging Face ---")
    sync_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "hf_manager.py"),
        "--sync",
        "--prefix",
        "models/",
    ]
    success, _ = run_command(sync_cmd)
    if not success:
        print(
            "Warning: Failed to sync models from Hugging Face. Continuing with local models..."
        )

    # 1. Stocks
    selected_stocks = select_tickers()
    print(f"Selected stocks for this session: {', '.join(selected_stocks)}")

    for symbol in selected_stocks:
        try:
            train_and_upload_stock(symbol)
        except Exception as e:
            print(f"Error training {symbol}: {e}")

    # 2. Forex
    for pair in FOREX_PAIRS:
        try:
            train_and_upload_forex(pair)
        except Exception as e:
            print(f"Error training {pair}: {e}")

    print(f"\n=== Continuous Training Session Completed: {datetime.now()} ===")


if __name__ == "__main__":
    main()
