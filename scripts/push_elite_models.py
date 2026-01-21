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
        # Read the professional model card
        model_card_path = Path(__file__).parent.parent / "docs" / "MODEL_CARD.md"
        
        if model_card_path.exists():
            with open(model_card_path, 'r', encoding='utf-8') as f:
                model_card_content = f.read()
        else:
            # Fallback to basic model card
            model_card_content = f"""---
library_name: pytorch
license: mit
tags:
- finance
- trading
- time-series
- {model_type}
---

# ARA.AI - Advanced Financial Prediction Models

Professional financial AI models for stock and forex prediction.

See full documentation at: https://github.com/MeridianAlgo/AraAI

**Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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
