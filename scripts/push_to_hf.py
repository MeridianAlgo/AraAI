#!/usr/bin/env python3
"""
Push Models to Hugging Face Hub
Supports both stock and forex unified models
"""

import argparse
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

# Load environment variables
load_dotenv()

# Increase Hugging Face Hub timeout to 3 hours (10800 seconds)
os.environ["HUGGINGFACE_HUB_READ_TIMEOUT"] = "10800"
os.environ["HUGGINGFACE_HUB_WRITE_TIMEOUT"] = "10800"


def _upload_with_retry(api, *, attempts=5, base_delay=10, **upload_kwargs):
    """Call api.upload_file with backoff on HTTP 429 (rate limit).

    The stock and forex workflows push hourly, each uploading a model file
    plus a README. That can trip Hugging Face's per-endpoint rate limits
    (notably /whoami-v2 and the commit endpoint), which surface as 429s.
    Retrying with exponential backoff lets the second pusher wait out the
    window instead of failing the whole job.
    """
    for attempt in range(1, attempts + 1):
        try:
            return api.upload_file(**upload_kwargs)
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 429 and attempt < attempts:
                delay = base_delay * (2 ** (attempt - 1))
                print(
                    f"   Rate limited (429) on attempt {attempt}/{attempts}; "
                    f"retrying in {delay}s..."
                )
                time.sleep(delay)
                continue
            raise


def push_model_to_hf(model_path, model_type="stock", repo_id="meridianal/ARA.AI"):
    """Push model to Hugging Face Hub."""

    # Accept CI uppercase secret OR .env lowercase key
    hf_token = (
        os.getenv("HF_TOKEN") or os.getenv("huggingface_token") or os.getenv("HUGGINGFACE_TOKEN")
    )
    if hf_token:
        hf_token = hf_token.strip()
    if not hf_token:
        print("ERROR: HF token not found (HF_TOKEN / huggingface_token)")
        return False

    model_path = Path(model_path)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return False

    try:
        # Authenticate by passing the token to every API call instead of
        # calling login(). login() validates via /whoami-v2, which is
        # rate-limited hard; with stocks + forex pushing hourly we were
        # hitting that limit and failing the upload before it even started.
        api = HfApi(token=hf_token)

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
        _upload_with_retry(
            api,
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
        _upload_with_retry(
            api,
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
    parser.add_argument("--repo-id", default="meridianal/ARA.AI", help="Hugging Face repo ID")

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
