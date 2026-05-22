"""
Cancel and delete GitHub Actions workflow runs for the AraAI repository,
keeping only Lint workflow runs.

Used to wipe the v4-era training-run history before the v5 release so the
Actions tab shows a clean slate.

Usage
-----
  set GITHUB_TOKEN=<your_personal_access_token>
  python scripts/clean_workflow_runs.py            # actually do it
  python scripts/clean_workflow_runs.py --dry-run  # print plan, change nothing

Token needs scopes: `repo` (private repo access) and `workflow`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any

import requests

OWNER = "MeridianAlgo"
REPO = "AraAI"
API = "https://api.github.com"

# Workflows whose runs we want to KEEP. Match is case-insensitive substring
# against the workflow's `name` field (so "Lint" matches "Lint" or "Linting").
KEEP_WORKFLOW_NAME_SUBSTRINGS = ("lint",)

# Run statuses that mean "still going" — these must be cancelled before delete.
ACTIVE_STATUSES = {"queued", "in_progress", "waiting", "pending", "requested"}


def session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "araai-clean-workflow-runs",
        }
    )
    return s


def paginate(s: requests.Session, url: str, key: str) -> list[dict[str, Any]]:
    """Walk paginated list endpoints, returning the accumulated `key` array."""
    out: list[dict[str, Any]] = []
    page = 1
    while True:
        r = s.get(url, params={"per_page": 100, "page": page}, timeout=30)
        r.raise_for_status()
        data = r.json()
        items = data.get(key, [])
        if not items:
            break
        out.extend(items)
        if len(items) < 100:
            break
        page += 1
    return out


def list_workflows(s: requests.Session) -> list[dict[str, Any]]:
    return paginate(
        s, f"{API}/repos/{OWNER}/{REPO}/actions/workflows", "workflows"
    )


def list_runs_for_workflow(
    s: requests.Session, workflow_id: int
) -> list[dict[str, Any]]:
    return paginate(
        s,
        f"{API}/repos/{OWNER}/{REPO}/actions/workflows/{workflow_id}/runs",
        "workflow_runs",
    )


def cancel_run(s: requests.Session, run_id: int) -> bool:
    r = s.post(
        f"{API}/repos/{OWNER}/{REPO}/actions/runs/{run_id}/cancel", timeout=30
    )
    # 202 = accepted, 409 = already finished/cancelled — both fine.
    return r.status_code in (202, 409)


def delete_run(s: requests.Session, run_id: int) -> bool:
    r = s.delete(f"{API}/repos/{OWNER}/{REPO}/actions/runs/{run_id}", timeout=30)
    # 204 = deleted, 404 = already gone.
    return r.status_code in (204, 404)


def should_keep(workflow_name: str) -> bool:
    name = workflow_name.lower()
    return any(sub in name for sub in KEEP_WORKFLOW_NAME_SUBSTRINGS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List runs that would be cancelled/deleted, change nothing.",
    )
    ap.add_argument(
        "--token",
        default=os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"),
        help="GitHub PAT. Defaults to $GITHUB_TOKEN / $GH_TOKEN.",
    )
    args = ap.parse_args()

    if not args.token:
        print(
            "ERROR: no token. Set GITHUB_TOKEN (scopes: repo, workflow) or pass --token.",
            file=sys.stderr,
        )
        return 2

    s = session(args.token)

    print(f"Repo: {OWNER}/{REPO}")
    print("Listing workflows...")
    workflows = list_workflows(s)
    if not workflows:
        print("  (no workflows found)")
        return 0

    keep_ids: set[int] = set()
    purge: list[dict[str, Any]] = []
    for wf in workflows:
        if should_keep(wf["name"]):
            keep_ids.add(wf["id"])
            print(f"  KEEP   #{wf['id']:<12} {wf['name']!r}  ({wf['path']})")
        else:
            purge.append(wf)
            print(f"  PURGE  #{wf['id']:<12} {wf['name']!r}  ({wf['path']})")

    if not purge:
        print("Nothing to purge — every workflow matched the keep list.")
        return 0

    total_runs = 0
    total_cancelled = 0
    total_deleted = 0
    total_failed = 0

    for wf in purge:
        wf_id = wf["id"]
        wf_name = wf["name"]
        print(f"\n--- Workflow: {wf_name!r} (#{wf_id}) ---")
        runs = list_runs_for_workflow(s, wf_id)
        print(f"  {len(runs)} run(s)")
        total_runs += len(runs)

        # Cancel active runs first so they're deletable.
        active = [r for r in runs if r.get("status") in ACTIVE_STATUSES]
        for r in active:
            run_id = r["id"]
            label = f"run {run_id} ({r.get('status')}, {r.get('display_title') or r.get('name')!r})"
            if args.dry_run:
                print(f"    [dry-run] would cancel {label}")
                continue
            ok = cancel_run(s, run_id)
            if ok:
                total_cancelled += 1
                print(f"    cancelled {label}")
            else:
                total_failed += 1
                print(f"    FAILED to cancel {label}")

        # Give active runs a moment to transition out of the cancelling state
        # before we attempt to delete them.
        if active and not args.dry_run:
            time.sleep(5)

        for r in runs:
            run_id = r["id"]
            label = (
                f"run {run_id} ({r.get('status')}/{r.get('conclusion')}, "
                f"{r.get('display_title') or r.get('name')!r})"
            )
            if args.dry_run:
                print(f"    [dry-run] would delete {label}")
                continue
            ok = delete_run(s, run_id)
            if ok:
                total_deleted += 1
                # Gentle pacing — GH API allows ~5k req/hr but we don't need to
                # sprint. 50ms keeps a 1000-run repo well under any limit.
                time.sleep(0.05)
            else:
                total_failed += 1
                print(f"    FAILED to delete {label}")

    print("\n=== Summary ===")
    print(f"  Workflows kept       : {len(keep_ids)}")
    print(f"  Workflows purged     : {len(purge)}")
    print(f"  Runs found in purged : {total_runs}")
    if args.dry_run:
        print("  (dry run — nothing changed)")
    else:
        print(f"  Runs cancelled       : {total_cancelled}")
        print(f"  Runs deleted         : {total_deleted}")
        print(f"  Failures             : {total_failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
