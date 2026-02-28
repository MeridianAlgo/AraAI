#!/usr/bin/env python3
"""
Push Models to Hugging Face Hub
Supports both stock and forex unified models
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, login

# Load environment variables
load_dotenv()

# Increase Hugging Face Hub timeout to 3 hours (10800 seconds)
os.environ["HUGGINGFACE_HUB_READ_TIMEOUT"] = "10800"
os.environ["HUGGINGFACE_HUB_WRITE_TIMEOUT"] = "10800"


def push_model_to_hf(model_path, model_type="stock", repo_id="MeridianAlgo/Meridian.AI"):
    """Push model to Hugging Face Hub"""

    # Get HF token from environment
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        hf_token = hf_token.strip()
    if not hf_token:
        print("ERROR: HF_TOKEN not found in environment variables")
        return False

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return False

    try:
        # Login to Hugging Face
        print("Logging into Hugging Face...")
        login(token=hf_token)

        api = HfApi()

        # Determine filename based on model type
        if model_type == "stock":
            filename = "models/Meridian.AI_Stocks.pt"
        elif model_type == "forex":
            filename = "models/Meridian.AI_Forex.pt"
        else:
            filename = f"models/{model_path.name}"

        print(f"Uploading {model_path.name} to Hugging Face...")
        print(f"   Destination: {repo_id}/{filename}")

        # Upload to Hugging Face
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=filename,
            repo_id=repo_id,
            token=hf_token,
            commit_message=f"Update {model_type} model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        )

        print(f"Successfully uploaded {filename}")

        # Update model card
        update_model_card(api, hf_token, repo_id, model_type)

        return True

    except Exception as e:
        print(f"ERROR: Failed to upload: {e}")
        return False


def update_model_card(api, hf_token, repo_id, model_type):
    """Update model card on Hugging Face"""
    try:
        # Read the professional model card
        model_card_path = Path(__file__).parent.parent / "docs" / "MODEL_CARD.md"

        if model_card_path.exists():
            with open(model_card_path, encoding="utf-8") as f:
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

# Meridian.AI - Advanced Financial Prediction Models

Professional financial AI models for stock and forex prediction.

See full documentation at: https://github.com/MeridianAlgo/Meridian.AI

**Last Updated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

        # Upload model card
        api.upload_file(
            path_or_fileobj=model_card_content.encode(),
            path_in_repo="README.md",
            repo_id=repo_id,
            token=hf_token,
            commit_message=f"Update model card - {model_type}",
        )

        print("Successfully updated model card on Hugging Face")

    except Exception as e:
        print(f"WARNING: Could not update model card: {e}")


def main():
    parser = argparse.ArgumentParser(description="Push models to Hugging Face Hub")
    parser.add_argument("--model-path", required=True, help="Path to model file")
    parser.add_argument(
        "--model-type", default="stock", choices=["stock", "forex"], help="Model type"
    )
    parser.add_argument(
        "--repo-id", default="MeridianAlgo/Meridian.AI", help="Hugging Face repo ID"
    )

    args = parser.parse_args()

    print("=" * 50)
    print("Model Upload to Hugging Face")
    print("=" * 50)

    success = push_model_to_hf(args.model_path, args.model_type, args.repo_id)

    if success:
        print("\nModel uploaded successfully!")
        print(f"Available at: https://huggingface.co/{args.repo_id}")
    else:
        print("\nUpload failed. Check the logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
