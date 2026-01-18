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
STOCK_COUNT = 1
FOREX_PAIRS = [
    "EURUSD",
]  # Single forex pair for training
EPOCHS = 500
UNIFIED_STOCK_MODEL = MODEL_DIR / "unified_stock_model.pt"
UNIFIED_FOREX_MODEL = MODEL_DIR / "unified_forex_model.pt"


def run_command(cmd, log_file=None):
    """Run a shell command and return output with detailed error handling"""

    cmd_str = " ".join(cmd)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_msg = f"[{timestamp}] Running: {cmd_str}"
    print(log_msg)
    if log_file:
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8", errors="replace"
    )

    # Always log stdout
    if result.stdout:
        print("STDOUT:", result.stdout)
        if log_file:
            with open(log_file, "a") as f:
                f.write("STDOUT: " + result.stdout + "\n")

    if result.returncode != 0:
        error_msg = f"[{timestamp}] Command failed with exit code {result.returncode}"
        if result.stderr:
            error_msg += f"\nSTDERR: {result.stderr}"
        if result.stdout:
            error_msg += f"\nSTDOUT: {result.stdout}"

        print(f"ERROR: {error_msg}")
        if log_file:
            with open(log_file, "a") as f:
                f.write("ERROR: " + error_msg + "\n")

        return False, result.stderr or result.stdout or error_msg

    return True, result.stdout


def select_tickers(use_all=False, stock_count=STOCK_COUNT, seed=None):
    """Select tickers from the file"""
    if not os.path.exists(TICKERS_FILE):
        print(f"Warning: {TICKERS_FILE} not found. Using fallback tickers.")
        return ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]  # Fallback

    with open(TICKERS_FILE, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]

    if use_all or len(tickers) <= stock_count:
        print(f"Using all {len(tickers)} tickers from {TICKERS_FILE}")
        return tickers
    else:
        rng = random.Random(seed)
        selected = rng.sample(tickers, stock_count)
        print(
            f"Selected {len(selected)} random tickers from {len(tickers)} available "
            f"(seed={seed})"
        )
        return selected


def fetch_and_store_stocks(symbols, log_file=None):
    """Fetch data for multiple stocks and store in DB"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"\n[{timestamp}] --- Fetching Stock Data for {len(symbols)} symbols ---"
    print(log_msg)
    if log_file:
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

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
    success, output = run_command(fetch_cmd, log_file=log_file)
    if not success:
        error_msg = f"[{timestamp}] Failed to fetch stock data. Output: {output}"
        print(error_msg)
        if log_file:
            with open(log_file, "a") as f:
                f.write(error_msg + "\n")
        return False

    # Store in DB
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] Storing stock data in database..."
    print(log_msg)
    if log_file:
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

    store_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "store_training_data.py"),
        "--data-dir",
        "datasets/training_data",
        "--db-file",
        DB_FILE,
    ]
    success, output = run_command(store_cmd, log_file=log_file)
    if not success:
        error_msg = f"[{timestamp}] Failed to store stock data. Output: {output}"
        print(error_msg)
        if log_file:
            with open(log_file, "a") as f:
                f.write(error_msg + "\n")
        return False

    return True


def fetch_and_store_forex(pairs, log_file=None):
    """Fetch data for multiple forex pairs and store in DB"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"\n[{timestamp}] --- Fetching Forex Data for {len(pairs)} pairs ---"
    print(log_msg)
    if log_file:
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

    print(f"[{timestamp}] Forex pairs: {', '.join(pairs)}")
    if log_file:
        with open(log_file, "a") as f:
            f.write(f"[{timestamp}] Forex pairs: {', '.join(pairs)}\n")

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
    success, output = run_command(fetch_cmd, log_file=log_file)
    if not success:
        error_msg = f"[{timestamp}] Failed to fetch forex data. Output: {output}"
        print(error_msg)
        if log_file:
            with open(log_file, "a") as f:
                f.write(error_msg + "\n")
        # Try fetching pairs individually to identify which ones fail
        print(
            f"[{timestamp}] Attempting to fetch forex pairs individually to identify failures..."
        )
        if log_file:
            with open(log_file, "a") as f:
                f.write(
                    f"[{timestamp}] Attempting to fetch forex pairs individually to identify failures...\n"
                )

        successful_pairs = []
        for pair in pairs:
            single_cmd = [
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
            pair_success, pair_output = run_command(single_cmd, log_file=log_file)
            if pair_success:
                successful_pairs.append(pair)
                print(f"[{timestamp}] ✓ Successfully fetched {pair}")
            else:
                print(f"[{timestamp}] ✗ Failed to fetch {pair}: {pair_output}")

        if not successful_pairs:
            return False

        print(
            f"[{timestamp}] Successfully fetched {len(successful_pairs)}/{len(pairs)} forex pairs"
        )
        if log_file:
            with open(log_file, "a") as f:
                f.write(
                    f"[{timestamp}] Successfully fetched {len(successful_pairs)}/{len(pairs)} forex pairs\n"
                )

    # Store in DB
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"[{timestamp}] Storing forex data in database..."
    print(log_msg)
    if log_file:
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

    store_cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "store_training_data.py"),
        "--data-dir",
        "datasets/training_data",
        "--db-file",
        DB_FILE,
    ]
    success, output = run_command(store_cmd, log_file=log_file)
    if not success:
        error_msg = f"[{timestamp}] Failed to store forex data. Output: {output}"
        print(error_msg)
        if log_file:
            with open(log_file, "a") as f:
                f.write(error_msg + "\n")
        return False

    return True


def train_unified_models(log_file=None):
    """Train unified models - ONE for all stocks, ONE for all forex"""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_msg = f"\n[{timestamp}] --- Training Unified Models ---"
    print(log_msg)
    if log_file:
        with open(log_file, "a") as f:
            f.write(log_msg + "\n")

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
    success, output = run_command(train_cmd, log_file=log_file)
    if not success:
        error_msg = f"[{timestamp}] Failed to train unified models. Output: {output}"
        print(error_msg)
        if log_file:
            with open(log_file, "a") as f:
                f.write(error_msg + "\n")
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
    import traceback

    parser = argparse.ArgumentParser(
        description="Continuous Training Orchestrator for Ara AI"
    )
    parser.add_argument(
        "--workflow",
        choices=["stock", "forex", "both"],
        default="both",
        help="Which workflow to run: stock-only, forex-only, or both",
    )
    parser.add_argument(
        "--use-all-tickers",
        action="store_true",
        help="Use all tickers from all_tickers.txt instead of random sample",
    )
    parser.add_argument(
        "--stock-count",
        type=int,
        default=STOCK_COUNT,
        help="Number of stocks to fetch/store per run",
    )
    parser.add_argument(
        "--stock-sample-size",
        type=int,
        default=STOCK_COUNT,
        help="Number of stocks to train on per run",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for stock sampling (optional)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Number of training epochs per run",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Path to log file for detailed logging",
    )
    args = parser.parse_args()

    # Determine log file path
    log_file = args.log_file
    if not log_file:
        # Use environment variable or default
        log_file = os.environ.get("TRAINING_LOG_FILE")
        if not log_file:
            # Default log file name based on workflow
            log_file = f"training_{args.workflow}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    # Initialize log file
    start_time = datetime.now()
    start_msg = f"=== Starting Unified Training Session: {start_time} ==="
    print(start_msg)
    with open(log_file, "w") as f:
        f.write(start_msg + "\n")
        f.write(f"Workflow: {args.workflow}\n")
        f.write(f"Use all tickers: {args.use_all_tickers}\n")
        f.write(f"Stock count (fetch/store): {args.stock_count}\n")
        f.write(f"Stock sample size (train): {args.stock_sample_size}\n")
        f.write(f"Seed: {args.seed}\n")
        f.write(f"Log file: {log_file}\n\n")

    try:
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
            stock_count = args.stock_count
            if args.use_all_tickers:
                stock_count = 10**9
            selected_stocks = select_tickers(
                use_all=args.use_all_tickers,
                stock_count=stock_count,
                seed=args.seed,
            )
            if len(selected_stocks) <= 20:
                print(
                    f"\nSelected {len(selected_stocks)} stocks: {', '.join(selected_stocks)}"
                )
            else:
                print(
                    f"\nSelected {len(selected_stocks)} stocks (showing first 20): {', '.join(selected_stocks[:20])}..."
                )
            with open(log_file, "a") as f:
                f.write(f"Selected {len(selected_stocks)} stocks for training\n")

            try:
                if not fetch_and_store_stocks(selected_stocks, log_file=log_file):
                    error_msg = "Error: Failed to fetch stock data"
                    print(error_msg)
                    with open(log_file, "a") as f:
                        f.write(error_msg + "\n")
                    return
            except Exception as e:
                error_msg = f"Error fetching stocks: {e}\n{traceback.format_exc()}"
                print(error_msg)
                with open(log_file, "a") as f:
                    f.write(error_msg + "\n")
                return

        # 2. Forex workflow
        if args.workflow in {"forex", "both"}:
            try:
                if not fetch_and_store_forex(FOREX_PAIRS, log_file=log_file):
                    error_msg = "Error: Failed to fetch forex data"
                    print(error_msg)
                    with open(log_file, "a") as f:
                        f.write(error_msg + "\n")
                    # Continue anyway - some pairs might have succeeded
                    print("Warning: Continuing with available forex data...")
            except Exception as e:
                error_msg = f"Error fetching forex: {e}\n{traceback.format_exc()}"
                print(error_msg)
                with open(log_file, "a") as f:
                    f.write(error_msg + "\n")
                print("Warning: Continuing with available forex data...")

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
                str(args.epochs),
                "--stock-sample-size",
                str(args.stock_sample_size),
            ]
            if args.seed is not None:
                train_cmd.extend(["--seed", str(args.seed)])
            if args.workflow == "stock":
                train_cmd.append("--stocks-only")
            elif args.workflow == "forex":
                train_cmd.append("--forex-only")

            success, output = run_command(train_cmd, log_file=log_file)
            if not success:
                error_msg = f"Error: Failed to train unified models. Output: {output}"
                print(error_msg)
                with open(log_file, "a") as f:
                    f.write(error_msg + "\n")
                return
        except Exception as e:
            error_msg = f"Error training models: {e}\n{traceback.format_exc()}"
            print(error_msg)
            with open(log_file, "a") as f:
                f.write(error_msg + "\n")
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
                warning_msg = f"Warning: Failed to upload models: {e}"
                print(warning_msg)
                with open(log_file, "a") as f:
                    f.write(warning_msg + "\n")
        else:
            msg = "\nSkipping Hugging Face upload (no HF_TOKEN found)"
            print(msg)
            with open(log_file, "a") as f:
                f.write(msg + "\n")

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        completion_msg = f"\n=== Unified Training Session Completed: {end_time} ==="
        print(completion_msg)
        print(f"Duration: {duration:.2f} seconds")
        with open(log_file, "a") as f:
            f.write(completion_msg + "\n")
            f.write(f"Duration: {duration:.2f} seconds\n")

        if args.workflow in {"stock", "both"}:
            print(f"✓ Stock Model: {UNIFIED_STOCK_MODEL}")
        if args.workflow in {"forex", "both"}:
            print(f"✓ Forex Model: {UNIFIED_FOREX_MODEL}")

        print(f"\n✓ Detailed log saved to: {log_file}")

    except Exception as e:
        error_msg = f"Fatal error in main: {e}\n{traceback.format_exc()}"
        print(error_msg)
        with open(log_file, "a") as f:
            f.write(error_msg + "\n")
        raise


if __name__ == "__main__":
    main()
