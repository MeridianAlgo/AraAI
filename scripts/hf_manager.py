#!/usr/bin/env python3
"""
Hugging Face Repository Manager for Ara AI
Handles model uploads, deletions, and model card updates.
"""

import os
import sys
import argparse
from pathlib import Path
from datetime import datetime
from huggingface_hub import HfApi
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Increase Hugging Face Hub timeout to 3 hours (10800 seconds)
os.environ["HUGGINGFACE_HUB_READ_TIMEOUT"] = "10800"
os.environ["HUGGINGFACE_HUB_WRITE_TIMEOUT"] = "10800"


class HFManager:
    def __init__(self, repo_id="MeridianAlgo/ARA.AI", token=None):
        self.repo_id = repo_id
        self.token = token or os.getenv("HF_TOKEN")
        if self.token:
            self.token = self.token.strip()
        if not self.token:
            print("Error: HF_TOKEN not found in environment or .env file.")
            sys.exit(1)
        self.api = HfApi(token=self.token)

    def download_model(self, remote_path, local_path=None):
        """Download a model file from Hugging Face"""
        if local_path is None:
            local_path = Path("models") / Path(remote_path).name
        else:
            local_path = Path(local_path)

        local_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading {self.repo_id}/{remote_path} to {local_path}...")
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                repo_id=self.repo_id,
                filename=remote_path,
                local_dir=str(local_path.parent),
                token=self.token,
            )
            print(f"[OK] Download successful: {path}")
            return True
        except Exception as e:
            print(f"[FAIL] Download failed: {e}")
            return False

    def sync_models(self, prefix="models/"):
        """Download all latest models from HF to local storage"""
        print(f"Syncing all models with prefix '{prefix}' from {self.repo_id}...")
        try:
            files = self.api.list_repo_files(repo_id=self.repo_id)
            target_files = [f for f in files if f.startswith(prefix)]

            success_count = 0
            for file_path in target_files:
                if self.download_model(file_path):
                    success_count += 1

            print(f"[OK] Synced {success_count}/{len(target_files)} models.")
            return True
        except Exception as e:
            print(f"[FAIL] Sync failed: {e}")
            return False

    def upload_model(self, local_path, remote_path=None):
        """Upload a model file to Hugging Face"""
        local_path = Path(local_path)
        if not local_path.exists():
            print(f"Error: Local file {local_path} does not exist.")
            return False

        if remote_path is None:
            remote_path = f"models/{local_path.name}"

        print(f"Uploading {local_path} to {self.repo_id}/{remote_path}...")
        try:
            self.api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_path,
                repo_id=self.repo_id,
                repo_type="model",
            )
            print(f"[OK] Upload successful: {remote_path}")
            return True
        except Exception as e:
            print(f"[FAIL] Upload failed: {e}")
            return False

    def delete_old_models(self, keep_count=5, prefix="models/stock_"):
        """Delete old models to save space on Hugging Face"""
        print(f"Cleaning up old models with prefix '{prefix}' (keeping last {keep_count})...")
        try:
            files = self.api.list_repo_files(repo_id=self.repo_id)
            target_files = sorted([f for f in files if f.startswith(prefix)])

            if len(target_files) > keep_count:
                to_delete = target_files[:-keep_count]
                for file_path in to_delete:
                    print(f"Deleting old file: {file_path}")
                    self.api.delete_file(
                        path_in_repo=file_path, repo_id=self.repo_id, repo_type="model"
                    )
                print(f"[OK] Deleted {len(to_delete)} old files.")
            else:
                print("No old files to delete.")
            return True
        except Exception as e:
            print(f"[FAIL] Cleanup failed: {e}")
            return False

    def update_model_card(self, content):
        """Update the README.md (Model Card) on Hugging Face"""
        print(f"Updating model card for {self.repo_id}...")
        try:
            # We can use upload_file for README.md
            temp_readme = Path("temp_README.md")
            temp_readme.write_text(content)

            self.api.upload_file(
                path_or_fileobj=str(temp_readme),
                path_in_repo="README.md",
                repo_id=self.repo_id,
                repo_type="model",
            )
            temp_readme.unlink()
            print("[OK] Model card updated.")
            return True
        except Exception as e:
            print(f"[FAIL] Model card update failed: {e}")
            return False


def generate_model_card(symbol, accuracy, loss, trained_on_count):
    """Generate a comprehensive model card content"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"""---
language: en
license: mit
tags:
- financial-forecasting
- stock-prediction
- time-series
- pytorch
- ara-ai
- ensemble-learning
datasets:
- yfinance
metrics:
- accuracy
- mse
---

# Ara AI (ARA.AI) - Financial Prediction Engine

## Overview
Ara AI is an advanced financial prediction system designed for multi-asset forecasting. This repository contains the latest weights for the ensemble models trained on market data.

## Model Architecture
The system employs a sophisticated ensemble architecture:
- **Feature Extraction**: 44+ technical indicators (RSI, MACD, Bollinger Bands, ATR, etc.)
- **Neural Core**: A large PyTorch model with 4M+ parameters
- **Attention Mechanism**: Multi-head attention for identifying key temporal features
- **Ensemble Heads**: Specialized prediction heads inspired by XGBoost, LightGBM, Random Forest, and Gradient Boosting
- **Dynamic Weighting**: Softmax-based attention weights for weighted prediction averaging

## Latest Training Stats ({timestamp})
- **Last Trained Symbol**: {symbol}
- **Validation Accuracy**: {accuracy}%
- **Validation Loss (MSE)**: {loss}
- **Total Unique Symbols in Training History**: {trained_on_count}

## Continuous Training
This model is part of a self-evolving system. It is retrained daily on a rotation of 6,800+ tickers and 20+ forex pairs to maintain high accuracy across different market conditions and time horizons (1D, 1H).

## Usage
### Loading the model
```python
import torch
from meridianalgo.unified_ml import UnifiedStockML

# Download the model file from this repo first
ml = UnifiedStockML(model_path="stock_AAPL.pt")
prediction = ml.predict_ultimate("AAPL", days=5)
print(prediction)
```

## Disclaimer
**Not Financial Advice.** This software is for educational purposes only. Trading involves significant risk. The authors are not responsible for any financial losses incurred.
"""


def main():
    parser = argparse.ArgumentParser(description="Manage Ara AI Hugging Face Repository")
    parser.add_argument("--upload", help="Local path of the model to upload")
    parser.add_argument("--download", help="Remote path of the model to download")
    parser.add_argument("--sync", action="store_true", help="Sync all models from HF repo")
    parser.add_argument("--remote-path", help="Remote path in the repository")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old models")
    parser.add_argument("--prefix", default="models/", help="Prefix for files to cleanup or sync")
    parser.add_argument("--keep", type=int, default=5, help="Number of models to keep")
    parser.add_argument("--update-card", action="store_true", help="Update the model card")
    parser.add_argument("--symbol", help="Symbol for the model card")
    parser.add_argument("--accuracy", help="Accuracy for the model card")
    parser.add_argument("--loss", help="Loss for the model card")
    parser.add_argument("--trained-count", help="Trained symbols count for the model card")

    args = parser.parse_args()
    manager = HFManager()

    if args.upload:
        manager.upload_model(args.upload, args.remote_path)

    if args.download:
        manager.download_model(args.download, args.remote_path)

    if args.sync:
        manager.sync_models(prefix=args.prefix)

    if args.cleanup:
        manager.delete_old_models(keep_count=args.keep, prefix=args.prefix)

    if args.update_card:
        content = generate_model_card(
            args.symbol or "Multi-Symbol",
            args.accuracy or "N/A",
            args.loss or "N/A",
            args.trained_count or "N/A",
        )
        manager.update_model_card(content)


if __name__ == "__main__":
    main()
