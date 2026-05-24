#!/usr/bin/env python3
"""
Migrate current HF models to models/legacy/ with version + date suffix.

Run once locally to snapshot the existing models/Meridian.AI_Forex.pt and
models/Meridian.AI_Stocks.pt into models/legacy/ before the next training
generation overwrites them.

Usage:
    python scripts/migrate_legacy.py [--dry-run]

Requires HF_TOKEN (or huggingface_token) in env / .env.
"""

import argparse
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

load_dotenv()

REPO_ID = "meridianal/ARA.AI"
SOURCES = [
    "models/Meridian.AI_Forex.pt",
    "models/Meridian.AI_Stocks.pt",
]


def get_token():
    tok = (
        os.getenv("HF_TOKEN")
        or os.getenv("huggingface_token")
        or os.getenv("HUGGINGFACE_TOKEN")
    )
    return tok.strip() if tok else None


def inspect_checkpoint(local_path):
    """Pull version + training_date out of the .pt so the legacy filename is informative."""
    try:
        import torch

        m = torch.load(local_path, map_location="cpu", weights_only=False)
        version = str(m.get("version", "unknown"))
        md = m.get("metadata", {}) or {}
        training_date = md.get("training_date") or m.get("training_date") or ""
        date_tag = ""
        if training_date:
            try:
                date_tag = datetime.fromisoformat(training_date).strftime("%Y%m%d")
            except Exception:
                date_tag = training_date[:10].replace("-", "")
        return version, date_tag
    except Exception as e:
        print(f"  warning: could not inspect {local_path}: {e}")
        return "unknown", ""


def legacy_name(source_path, version, date_tag):
    """models/Meridian.AI_Forex.pt -> models/legacy/Meridian.AI_Forex_v5.1.0_20260515.pt"""
    base = Path(source_path).name
    stem = Path(base).stem  # Meridian.AI_Forex
    suffix = Path(base).suffix  # .pt
    parts = [stem]
    if version and version != "unknown":
        parts.append(f"v{version}")
    if date_tag:
        parts.append(date_tag)
    return f"models/legacy/{'_'.join(parts)}{suffix}"


def main():
    parser = argparse.ArgumentParser(description="Move current HF models to models/legacy/")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be uploaded without actually doing it",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        default=True,
        help="Keep the original at models/ in addition to the legacy copy (default: true)",
    )
    args = parser.parse_args()

    token = get_token()
    if not token:
        print("ERROR: no HF_TOKEN / huggingface_token found in env or .env")
        sys.exit(1)

    api = HfApi(token=token)

    # Verify the repo exists + token works
    try:
        info = api.repo_info(repo_id=REPO_ID, repo_type="model")
        print(f"Connected to HF repo: {REPO_ID} ({info.id})")
    except Exception as e:
        print(f"ERROR: cannot access {REPO_ID}: {e}")
        sys.exit(1)

    repo_files = set(api.list_repo_files(repo_id=REPO_ID))
    print(f"  Repo contains {len(repo_files)} files")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        for src in SOURCES:
            print(f"\n=== {src} ===")
            if src not in repo_files:
                print("  not present on HF, skipping")
                continue

            try:
                local = hf_hub_download(
                    repo_id=REPO_ID,
                    filename=src,
                    token=token,
                    local_dir=str(tmp_dir),
                )
            except Exception as e:
                print(f"  download failed: {e}")
                continue

            size_mb = os.path.getsize(local) / (1024 * 1024)
            print(f"  downloaded {size_mb:.1f} MB -> {local}")

            version, date_tag = inspect_checkpoint(local)
            dest = legacy_name(src, version, date_tag)
            print(f"  legacy destination: {dest}")

            if dest in repo_files:
                print("  already exists in legacy/, skipping upload")
                continue

            if args.dry_run:
                print("  [dry-run] would upload")
                continue

            try:
                api.upload_file(
                    path_or_fileobj=local,
                    path_in_repo=dest,
                    repo_id=REPO_ID,
                    token=token,
                    commit_message=f"Archive {Path(src).name} to legacy (v{version}, {date_tag})",
                )
                print(f"  uploaded -> {dest}")
            except Exception as e:
                print(f"  upload failed: {e}")
                continue

            if not args.keep_original:
                try:
                    api.delete_file(
                        path_in_repo=src,
                        repo_id=REPO_ID,
                        token=token,
                        commit_message=f"Remove {Path(src).name} (archived to legacy/)",
                    )
                    print(f"  deleted original {src}")
                except Exception as e:
                    print(f"  delete failed (non-fatal): {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
