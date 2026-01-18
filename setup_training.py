#!/usr/bin/env python3
"""
Quick setup script for ARA AI training system
"""

import sys
import subprocess
from pathlib import Path


def print_header(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")


def run_command(cmd, description):
    """Run a command and show progress"""
    print(f"→ {description}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print("  ✓ Success")
        return True
    else:
        print(f"  ✗ Failed: {result.stderr}")
        return False


def main():
    print_header("ARA AI Training System Setup")

    # Check Python version
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 9:
        print(f"  ✓ Python {version.major}.{version.minor}.{version.micro}")
    else:
        print(f"  ✗ Python 3.9+ required (found {version.major}.{version.minor})")
        return False

    # Create directories
    print("\nCreating directories...")
    dirs = ["models", "datasets/training_data", "datasets/test_data"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {d}")

    # Check requirements
    print("\nChecking dependencies...")
    try:
        import pandas  # noqa: F401
        import numpy  # noqa: F401
        import sklearn  # noqa: F401
        import torch  # noqa: F401
        import yfinance  # noqa: F401
        import rich  # noqa: F401

        print("  ✓ All core dependencies installed")
    except ImportError as e:
        print(f"  ✗ Missing dependency: {e}")
        print("\n  Installing dependencies...")
        run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "Installing requirements",
        )

    # Test data fetch
    print_header("Testing Data Fetch")
    success = run_command(
        f"{sys.executable} scripts/fetch_training_data.py --symbols AAPL --output-dir datasets/training_data --period 60d --interval 1d --asset-type stock",
        "Fetching AAPL data",
    )

    if not success:
        print("\n✗ Setup failed at data fetch stage")
        return False

    # Test database storage
    print_header("Testing Database Storage")
    success = run_command(
        f"{sys.executable} scripts/store_training_data.py --data-dir datasets/training_data --db-file training.db",
        "Storing data in database",
    )

    if not success:
        print("\n✗ Setup failed at database stage")
        return False

    # Success message
    print_header("Setup Complete!")

    print("✓ All systems ready for training!\n")
    print("Next steps:")
    print("  1. Test training:  python scripts/quick_train.py --symbol AAPL")
    print("  2. View dashboard: python scripts/training_dashboard.py")
    print("  3. Batch training: python scripts/batch_train.py --random 5")
    print("  4. Read guide:     TRAINING_GUIDE.md")
    print("\nGitHub Actions:")
    print("  • Elite Hourly:    .github/workflows/hourly-training.yml")
    print("\nTo enable automated training:")
    print("  1. Push to GitHub")
    print("  2. Enable Actions in repository settings")
    print("  3. (Optional) Add HF_TOKEN and WANDB_API_KEY secrets")

    print(f"\n{'='*70}\n")

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Setup error: {e}")
        sys.exit(1)
