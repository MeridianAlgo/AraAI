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
from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.utils import HfHubHTTPError

# Load environment variables
load_dotenv()

# Increase Hugging Face Hub timeout to 3 hours (10800 seconds)
os.environ["HUGGINGFACE_HUB_READ_TIMEOUT"] = "10800"
os.environ["HUGGINGFACE_HUB_WRITE_TIMEOUT"] = "10800"


def _retry_after_seconds(e):
    """Extract the server-suggested wait from a 429 response, if present."""
    headers = getattr(getattr(e, "response", None), "headers", None) or {}
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw:
        try:
            return max(1, int(float(raw)))
        except (TypeError, ValueError):
            return None
    return None


def _commit_with_retry(api, *, attempts=8, base_delay=15, max_delay=300, **commit_kwargs):
    """Call api.create_commit with backoff on HTTP 429 (rate limit).

    The stock and forex workflows push hourly to the same repo. That can trip
    Hugging Face's per-endpoint rate limits (notably /preupload and the commit
    endpoint), which surface as 429s and can persist for several minutes.
    Retrying with exponential backoff — and honoring the server's Retry-After
    header when it sends one — lets the pusher wait out the window instead of
    failing the whole job.
    """
    for attempt in range(1, attempts + 1):
        try:
            return api.create_commit(**commit_kwargs)
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 429 and attempt < attempts:
                delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                delay = _retry_after_seconds(e) or delay
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

        # Commit the model file and the model card in a SINGLE commit. Doing
        # them as two separate upload_file() calls meant two commits (= two
        # /preupload round-trips) per run; with stocks + forex pushing hourly
        # to the same repo that doubled the request rate and tripped HF's 429
        # rate limiter. One atomic commit halves the per-run request count.
        operations = [
            CommitOperationAdd(path_in_repo=filename, path_or_fileobj=str(model_path)),
        ]

        model_card_content = _build_model_card(model_type)
        if model_card_content is not None:
            operations.append(
                CommitOperationAdd(
                    path_in_repo="README.md",
                    path_or_fileobj=model_card_content.encode(),
                )
            )

        _commit_with_retry(
            api,
            repo_id=repo_id,
            operations=operations,
            token=hf_token,
            commit_message=(
                f"Update {model_type} model - "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ),
        )

        print(f"Successfully uploaded {filename} (+ model card)")

        return True

    except Exception as e:
        print(f"ERROR: Failed to upload: {e}")
        return False


def _build_model_card(model_type):
    """Return the model-card markdown to commit alongside the model.

    Returns None if the card cannot be built, so the model still gets pushed.
    """
    try:
        # Read the professional model card
        model_card_path = Path(__file__).parent.parent / "docs" / "MODEL_CARD.md"

        if model_card_path.exists():
            with open(model_card_path, encoding="utf-8") as f:
                return f.read()

        # Fallback to basic model card
        return f"""---
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
    except Exception as e:
        print(f"WARNING: Could not build model card: {e}")
        return None


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
