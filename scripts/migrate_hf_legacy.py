#!/usr/bin/env python3
"""Move pre-v5 model artifacts on Hugging Face Hub into a ``legacy/`` folder.

The script lists every file under ``models/`` in the target repo, downloads
each ``.pt`` checkpoint, inspects its ``version`` / ``architecture`` fields,
and moves anything older than v5 to ``legacy/<original_filename>``. Files that
are already at v5+ stay in ``models/``.

Run once after rolling out v5.1.0. Safe to re-run; it is idempotent.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import torch
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download

load_dotenv()

# Hugging Face hub timeouts — large checkpoints can take a while.
os.environ.setdefault("HUGGINGFACE_HUB_READ_TIMEOUT", "10800")
os.environ.setdefault("HUGGINGFACE_HUB_WRITE_TIMEOUT", "10800")


def get_hf_token():
    """Read token from CI secret or .env file."""
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("huggingface_token")
        or os.getenv("HUGGINGFACE_TOKEN")
    )
    return token.strip() if token else None


def _parse_version(v):
    try:
        return tuple(int(p) for p in str(v).split("."))
    except Exception:
        return (0,)


def classify_checkpoint(local_path):
    """Return (version_str, architecture_str) — never raises."""
    try:
        ckpt = torch.load(local_path, map_location="cpu")
        return str(ckpt.get("version", "0")), str(ckpt.get("architecture", "unknown"))
    except Exception as e:
        return f"unreadable:{e.__class__.__name__}", "unknown"


def migrate(repo_id, token, dry_run=False, min_version=(5, 0)):
    api = HfApi(token=token)
    files = api.list_repo_files(repo_id=repo_id)
    model_files = [
        f for f in files if f.startswith("models/") and f.endswith(".pt")
    ]
    print(f"Found {len(model_files)} .pt files under models/ in {repo_id}")

    moves = []
    tmp_root = Path(tempfile.mkdtemp(prefix="hf_legacy_"))
    for hf_path in model_files:
        print(f"\nInspecting {hf_path} ...")
        try:
            local = hf_hub_download(
                repo_id=repo_id, filename=hf_path, token=token, local_dir=tmp_root
            )
        except Exception as e:
            print(f"  WARN download failed ({e}); skipping")
            continue

        version, arch = classify_checkpoint(local)
        parsed = _parse_version(version)
        is_legacy = parsed < min_version

        marker = "LEGACY" if is_legacy else "current"
        print(f"  version={version} architecture={arch} -> {marker}")

        if is_legacy:
            legacy_path = "legacy/" + Path(hf_path).name
            moves.append((hf_path, legacy_path, local))

    if not moves:
        print("\nNo legacy files found. Nothing to migrate.")
        return

    print(f"\nPlanned moves ({len(moves)}):")
    for src, dst, _ in moves:
        print(f"  {src}  ->  {dst}")

    if dry_run:
        print("\n[dry-run] Not uploading or deleting. Re-run without --dry-run to apply.")
        return

    for src, dst, local in moves:
        print(f"\nUploading {src} -> {dst}")
        api.upload_file(
            path_or_fileobj=local,
            path_in_repo=dst,
            repo_id=repo_id,
            token=token,
            commit_message=f"Archive {Path(src).name} to legacy/ (v5 migration)",
        )
        print(f"Deleting old path {src}")
        try:
            api.delete_file(
                path_in_repo=src,
                repo_id=repo_id,
                token=token,
                commit_message=f"Remove {src} (archived to {dst})",
            )
        except Exception as e:
            print(f"  WARN delete failed: {e}")

    print("\nMigration complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Move pre-v5 HF checkpoints to legacy/ subfolder"
    )
    parser.add_argument("--repo-id", default="meridianal/ARA.AI")
    parser.add_argument(
        "--dry-run", action="store_true", help="List planned moves; do not modify the repo"
    )
    parser.add_argument(
        "--min-major",
        type=int,
        default=5,
        help="Treat checkpoints with major version below this as legacy (default: 5)",
    )
    args = parser.parse_args()

    token = get_hf_token()
    if not token:
        print("ERROR: HF token not found (HF_TOKEN / huggingface_token)")
        sys.exit(1)

    migrate(args.repo_id, token, dry_run=args.dry_run, min_version=(args.min_major, 0))


if __name__ == "__main__":
    main()
