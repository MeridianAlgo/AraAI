#!/usr/bin/env python3
"""
Select random stock tickers from all_tickers.txt
"""

import argparse
import random


def load_tickers(file_path):
    """Load tickers from file"""
    with open(file_path, "r") as f:
        tickers = [line.strip() for line in f if line.strip()]
    return tickers


def select_random_tickers(tickers, count=10, exclude=None):
    """Select random tickers, excluding certain ones"""
    exclude = exclude or []

    # Filter out excluded tickers
    available = [t for t in tickers if t not in exclude]

    # Select random tickers
    if len(available) < count:
        print(f"Warning: Only {len(available)} tickers available, requested {count}")
        count = len(available)

    selected = random.sample(available, count)
    return selected


def main():
    parser = argparse.ArgumentParser(description="Select random stock tickers")
    parser.add_argument("--file", default="all_tickers.txt", help="Ticker file path")
    parser.add_argument(
        "--count", type=int, default=10, help="Number of tickers to select"
    )
    parser.add_argument("--exclude", help="Comma-separated list of tickers to exclude")
    parser.add_argument(
        "--output-format", choices=["comma", "json", "space"], default="comma"
    )

    args = parser.parse_args()

    # Load tickers
    tickers = load_tickers(args.file)
    print(f"Loaded {len(tickers)} tickers from {args.file}", flush=True)

    # Parse exclusions
    exclude = []
    if args.exclude:
        exclude = [t.strip() for t in args.exclude.split(",")]

    # Select random tickers
    selected = select_random_tickers(tickers, args.count, exclude)

    # Output in requested format
    if args.output_format == "comma":
        output = ",".join(selected)
    elif args.output_format == "json":
        import json

        output = json.dumps(selected)
    else:  # space
        output = " ".join(selected)

    print(f"Selected tickers: {output}", flush=True)
    print(output)  # For GitHub Actions output


if __name__ == "__main__":
    main()
