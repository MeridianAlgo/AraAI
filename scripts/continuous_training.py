#!/usr/bin/env python3
"""
Continuous Training Orchestrator for Ara AI
Trains ONE unified model for all stocks and ONE for all forex pairs.
Much more efficient than training separate models per ticker.
"""

import os
import sys
import subprocess
import random
import argparse
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
STOCK_COUNT = 10  # Fetch more stocks for unified training
FOREX_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
EPOCHS = 50
UNIFIED_STOCK_MODEL = MODEL_DIR / "unified_stock_model.pt"
UNIFIED_FOREX_MODEL = MODEL_DIR / "unified_forex_model.pt"


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
        print(f"Warning: {TICKERS_FILE} not found. Using fallback tickers.")
        return ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]  # Fallback

    with open(TICKERS_FILE, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    return random.sample(tickers, min(len(tickers), STOCK_COUNT))


def fetch_and_store_stocks(symbols):
    """Fetch data for multiple stocks and store in DB"""
    print(f"\n--- Fetching Stock Data for {len(symbols)} symbols ---")

    # Fetch data for all stocks
    fetch_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "fetch_training_data.py"),
        "--symbols",
        *symbols,
        "--output-dir",
        "datasets/training_data",
        "--period",
        "2y",
        "--interval",
        "1d",
        "--asset-type",
        "stock",
    ]
    success, _ = run_command(fetch_cmd)
    if not success:
        return False

    # Store in DB
    store_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "store_training_data.py"),
        "--data-dir",
        "datasets/training_data",
        "--db-file",
        DB_FILE,
    ]
    success, _ = run_command(store_cmd)
    return success


def fetch_and_store_forex(pairs):
    """Fetch data for multiple forex pairs and store in DB"""
    print(f"\n--- Fetching Forex Data for {len(pairs)} pairs ---")

    # Fetch data for all forex pairs
    fetch_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "fetch_training_data.py"),
        "--symbols",
        *pairs,
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

    # Store in DB
    store_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "store_training_data.py"),
        "--data-dir",
        "datasets/training_data",
        "--db-file",
        DB_FILE,
    ]
    success, _ = run_command(store_cmd)
    return success


def train_unified_models():
    """Train unified models - ONE for all stocks, ONE for all forex"""
    print("\n--- Training Unified Models ---")

    train_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "train_unified_model.py"),
        "--db-file",
        DB_FILE,
        "--stock-output",
        str(UNIFIED_STOCK_MODEL),
        "--forex-output",
        str(UNIFIED_FOREX_MODEL),
        "--epochs",
        str(EPOCHS),
    ]
    success, _ = run_command(train_cmd)
    return success


def upload_unified_models():
    """Upload unified models to Hugging Face"""
    print("\n--- Uploading Unified Models to Hugging Face ---")

    # Upload stock model
    stock_upload_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "hf_manager.py"),
        "--upload",
        str(UNIFIED_STOCK_MODEL),
        "--cleanup",
        "--prefix",
        "models/unified_stock_model",
    ]
    stock_success, _ = run_command(stock_upload_cmd)

    # Upload forex model
    forex_upload_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "hf_manager.py"),
        "--upload",
        str(UNIFIED_FOREX_MODEL),
        "--cleanup",
        "--prefix",
        "models/unified_forex_model",
    ]
    forex_success, _ = run_command(forex_upload_cmd)

    return stock_success and forex_success


def upload_stock_model():
    """Upload unified stock model to Hugging Face"""
    print("\n--- Uploading Unified Stock Model to Hugging Face ---")
    stock_upload_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "hf_manager.py"),
        "--upload",
        str(UNIFIED_STOCK_MODEL),
        "--cleanup",
        "--prefix",
        "models/unified_stock_model",
    ]
    stock_success, _ = run_command(stock_upload_cmd)
    return stock_success


def upload_forex_model():
    """Upload unified forex model to Hugging Face"""
    print("\n--- Uploading Unified Forex Model to Hugging Face ---")
    forex_upload_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "hf_manager.py"),
        "--upload",
        str(UNIFIED_FOREX_MODEL),
        "--cleanup",
        "--prefix",
        "models/unified_forex_model",
    ]
    forex_success, _ = run_command(forex_upload_cmd)
    return forex_success


def main():
    parser = argparse.ArgumentParser(
        description="Continuous Training Orchestrator for Ara AI"
    )
    parser.add_argument(
        "--workflow",
        choices=["stock", "forex", "both"],
        default="both",
        help="Which workflow to run: stock-only, forex-only, or both",
    )
    args = parser.parse_args()

    print(f"=== Starting Unified Training Session: {datetime.now()} ===")
    if args.workflow == "stock":
        print("Running STOCK workflow only")
    elif args.workflow == "forex":
        print("Running FOREX workflow only")
    else:
        print("Training ONE model for all stocks and ONE for all forex pairs")

    # Ensure directories exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Path("datasets/training_data").mkdir(parents=True, exist_ok=True)

    # 1. Stock workflow
    if args.workflow in {"stock", "both"}:
        selected_stocks = select_tickers()
        print(f"\nSelected {len(selected_stocks)} stocks: {', '.join(selected_stocks)}")

        try:
            if not fetch_and_store_stocks(selected_stocks):
                print("Error: Failed to fetch stock data")
                return
        except Exception as e:
            print(f"Error fetching stocks: {e}")
            return

    # 2. Forex workflow
    if args.workflow in {"forex", "both"}:
        try:
            if not fetch_and_store_forex(FOREX_PAIRS):
                print("Error: Failed to fetch forex data")
                return
        except Exception as e:
            print(f"Error fetching forex: {e}")
            return

    # 3. Train unified models (scoped)
    try:
        train_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "train_unified_model.py"),
            "--db-file",
            DB_FILE,
            "--stock-output",
            str(UNIFIED_STOCK_MODEL),
            "--forex-output",
            str(UNIFIED_FOREX_MODEL),
            "--epochs",
            str(EPOCHS),
        ]
        if args.workflow == "stock":
            train_cmd.append("--stocks-only")
        elif args.workflow == "forex":
            train_cmd.append("--forex-only")

        success, _ = run_command(train_cmd)
        if not success:
            print("Error: Failed to train unified models")
            return
    except Exception as e:
        print(f"Error training models: {e}")
        return

    # 4. Upload to Hugging Face (optional)
    if os.environ.get("HF_TOKEN"):
        try:
            if args.workflow == "stock":
                upload_stock_model()
            elif args.workflow == "forex":
                upload_forex_model()
            else:
                upload_unified_models()
        except Exception as e:
            print(f"Warning: Failed to upload models: {e}")
    else:
        print("\nSkipping Hugging Face upload (no HF_TOKEN found)")

    print(f"\n=== Unified Training Session Completed: {datetime.now()} ===")
    if args.workflow in {"stock", "both"}:
        print(f"✓ Stock Model: {UNIFIED_STOCK_MODEL}")
    if args.workflow in {"forex", "both"}:
        print(f"✓ Forex Model: {UNIFIED_FOREX_MODEL}")


if __name__ == "__main__":
    main()
