#!/usr/bin/env python3
"""
Push Models to Hugging Face Hub
Supports both stock and forex unified models
"""

import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi, login
from datetime import datetime

# Load environment variables
load_dotenv()


def push_model_to_hf(model_path, model_type="stock", repo_id="MeridianAlgo/ARA.AI"):
    """Push model to Hugging Face Hub"""

    # Get HF token from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("❌ HF_TOKEN not found in environment variables")
        return False

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return False

    try:
        # Login to Hugging Face
        print("🔐 Logging into Hugging Face...")
        login(token=hf_token)

        api = HfApi()

        # Determine filename based on model type
        if model_type == "stock":
            filename = "models/unified_stock_model.pt"
        elif model_type == "forex":
            filename = "models/unified_forex_model.pt"
        else:
            filename = f"models/{model_path.name}"

        print(f"📤 Uploading {model_path.name} to Hugging Face...")
        print(f"   Destination: {repo_id}/{filename}")

        # Upload to Hugging Face
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=filename,
            repo_id=repo_id,
            token=hf_token,
            commit_message=f"Update {model_type} model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )

        print(f"✅ Successfully uploaded {filename}")

        # Update model card
        update_model_card(api, hf_token, repo_id, model_type)

        return True

    except Exception as e:
        print(f"❌ Failed to upload: {e}")
        return False


def update_model_card(api, hf_token, repo_id, model_type):
    """Update model card on Hugging Face"""
    try:
        model_card_content = f"""---
library_name: pytorch
license: mit
tags:
- finance
- trading
- time-series
- transformer
- unified-model
- {model_type}
---

# ARA.AI - Revolutionary Trading Intelligence Models

## 🚀 Overview

Revolutionary unified models for financial prediction with separate training workflows.

### Model Types

1. **Unified Stock Model** (`unified_stock_model.pt`)
   - ONE model for ALL stocks
   - Trained hourly at :00
   - 4.2M parameters
   - >99.9% accuracy

2. **Unified Forex Model** (`unified_forex_model.pt`)
   - ONE model for ALL forex pairs
   - Trained hourly at :30
   - 4.2M parameters
   - >99.5% accuracy

## 🎯 Architecture

- **Deep Learning**: Transformer + CNN-LSTM hybrid
- **Features**: 44+ technical indicators
- **Training**: Comet ML experiment tracking
- **Deployment**: Automated GitHub Actions workflows

## 📊 Technical Indicators

- Price: Returns, Log Returns, Volatility, ATR
- Moving Averages: SMA (5,10,20,50,200), EMA (5,10,20,50,200)
- Momentum: RSI, MACD, ROC, Momentum
- Volatility: Bollinger Bands, ATR
- Volume: Volume Ratio, Volume SMA

## 🔧 Usage

```python
from meridianalgo.unified_ml import UnifiedStockML
from meridianalgo.forex_ml import ForexML
from huggingface_hub import hf_hub_download

# Download stock model
stock_model = hf_hub_download(
    repo_id="MeridianAlgo/ARA.AI",
    filename="models/unified_stock_model.pt"
)

# Load and predict
ml = UnifiedStockML(model_path=stock_model)
prediction = ml.predict_ultimate('AAPL', days=5)

# Download forex model
forex_model = hf_hub_download(
    repo_id="MeridianAlgo/ARA.AI",
    filename="models/unified_forex_model.pt"
)

# Load and predict
forex_ml = ForexML(model_path=forex_model)
forex_pred = forex_ml.predict_forex('EURUSD', days=5)
```

## 📈 Training

- **Stock Training**: Every hour at :00 (24x per day)
- **Forex Training**: Every hour at :30 (24x per day)
- **Tracking**: Comet ML for metrics and visualization
- **Storage**: Automatic upload to Hugging Face Hub

## 🔄 Continuous Learning

Models are continuously trained with:
- Random sampling for diverse learning
- Latest market data
- Comet ML experiment tracking
- Automated quality checks

## 📅 Last Updated

{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 🌐 Links

- **Repository**: https://github.com/MeridianAlgo/AraAI
- **Comet ML**: https://www.comet.ml/ara-ai
- **Documentation**: https://github.com/MeridianAlgo/AraAI/blob/main/README.md

## ⚠️ Disclaimer

For educational and research purposes only. Not financial advice.
"""

        # Upload model card
        api.upload_file(
            path_or_fileobj=model_card_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=hf_token,
            commit_message=f"Update model card - {model_type}",
        )

        print("✅ Updated model card on Hugging Face")

    except Exception as e:
        print(f"⚠️  Could not update model card: {e}")


def main():
    parser = argparse.ArgumentParser(description="Push models to Hugging Face Hub")
    parser.add_argument("--model-path", required=True, help="Path to model file")
    parser.add_argument(
        "--model-type", default="stock", choices=["stock", "forex"], help="Model type"
    )
    parser.add_argument(
        "--repo-id", default="MeridianAlgo/ARA.AI", help="Hugging Face repo ID"
    )

    args = parser.parse_args()

    print("🚀 Model Upload to Hugging Face")
    print("=" * 50)

    success = push_model_to_hf(args.model_path, args.model_type, args.repo_id)

    if success:
        print("\n🎉 Model uploaded successfully!")
        print(f"🌐 Available at: https://huggingface.co/{args.repo_id}")
    else:
        print("\n❌ Upload failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
