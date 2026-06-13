#!/usr/bin/env python3
"""Download the existing model from Hugging Face with 429-aware retries.

The hourly stock + forex workflows both HEAD/GET the same HF repo, which
periodically trips Hugging Face's rate limiter. hf_hub_download() has its own
internal HEAD retries, but they back off for at most ~30s total — far shorter
than a real rate-limit window — and then the workflow silently falls back to
training from scratch, throwing away the warm start (run 27101472126 is a
worked example). This wrapper retries the whole download with long exponential
backoff, honoring the server's Retry-After header when present.

The warm start is best-effort: this script ALWAYS exits 0. A missing model
must never fail the pipeline — training fresh is the designed fallback.
"""

import argparse
import os
import random
import sys
import time

from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError


def _retry_after_seconds(exc):
    """Extract the server-suggested wait from a 429 response, if present."""
    headers = getattr(getattr(exc, "response", None), "headers", None) or {}
    raw = headers.get("Retry-After") or headers.get("retry-after")
    if raw:
        try:
            return max(1, int(float(raw)))
        except (TypeError, ValueError):
            return None
    return None


def main():
    ap = argparse.ArgumentParser(description="Best-effort HF model download with retries")
    ap.add_argument("--repo-id", default="meridianal/ARA.AI")
    ap.add_argument("--filename", required=True, help="Path of the file inside the repo")
    ap.add_argument("--local-dir", default=".")
    ap.add_argument("--attempts", type=int, default=6)
    ap.add_argument("--base-delay", type=float, default=20.0)
    ap.add_argument("--max-delay", type=float, default=180.0)
    args = ap.parse_args()

    token = (
        os.getenv("HF_TOKEN") or os.getenv("huggingface_token") or os.getenv("HUGGINGFACE_TOKEN")
    )
    if token:
        token = token.strip()

    for attempt in range(1, args.attempts + 1):
        try:
            path = hf_hub_download(
                repo_id=args.repo_id,
                filename=args.filename,
                local_dir=args.local_dir,
                token=token,
                etag_timeout=30,
            )
            print(f"Downloaded existing model: {path}")
            return 0
        except Exception as e:  # noqa: BLE001 — warm start is best-effort by design
            status = getattr(getattr(e, "response", None), "status_code", None)
            # A definitive 404 means the file genuinely is not there — no point
            # retrying. Everything else (429, 5xx, timeouts, the LocalEntry-
            # NotFoundError hf_hub raises after ITS retries are exhausted) is
            # treated as transient.
            if isinstance(e, HfHubHTTPError) and status == 404:
                print(f"No existing model on the Hub (404) — will train fresh: {e}")
                return 0
            if attempt >= args.attempts:
                print(f"Giving up after {attempt} attempts — will train fresh: {e}")
                return 0
            delay = min(args.max_delay, args.base_delay * (2 ** (attempt - 1)))
            delay = _retry_after_seconds(e) or delay
            delay += random.uniform(0, 5)  # jitter so stock/forex don't sync up
            print(
                f"Download attempt {attempt}/{args.attempts} failed "
                f"({type(e).__name__}{f', HTTP {status}' if status else ''}); "
                f"retrying in {delay:.0f}s..."
            )
            time.sleep(delay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
