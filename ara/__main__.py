"""Ara.AI v8 CLI:  python -m ara train|eval|predict"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from . import model as M


def _dataset(args):
    df = M.load_panel(args.db_file, args.asset)
    print(f"  loaded {len(df):,} bars / {df['symbol'].nunique()} symbols")
    data = M.make_dataset(df)
    print(
        f"  usable samples {len(data):,}  ({data['date'].min().date()} -> {data['date'].max().date()})"
    )
    return data


def cmd_train(args):
    data = _dataset(args)
    if args.train_end:
        data = data[data["date"] < pd.Timestamp(args.train_end)]
        print(f"  train cutoff {args.train_end}: {len(data):,} samples")
    res = M.train(data, seed=args.seed, max_iter=args.max_iter)
    out = M.save(res, args.output)
    print(f"  trained {res.n_train:,} samples in {res.train_seconds:.1f}s -> {out}")
    return 0


def cmd_eval(args):
    data = _dataset(args)
    report = M.walk_forward(
        data,
        start=args.holdout_start,
        folds=args.folds,
        seed=args.seed,
        max_iter=args.max_iter,
    )
    m = report["pooled"]
    print("-" * 66)
    print(
        f"  universe: {report['n_symbols']} symbols, "
        f"median {report['median_universe_per_day']:.0f} names/day"
    )
    for f in report["folds"]:
        print(
            f"  fold {f['fold']} {f['test_start']}->{f['test_end']}  "
            f"IC {f['mean_ic']:+.4f} (t {f['ic_t_stat']:+.2f})  "
            f"L/S {f['ls_mean_daily'] * 1e4:+.1f}bp/d  Sharpe {f['ls_sharpe_annual']:+.2f}"
        )
    print("-" * 66)
    print(f"  pooled over {m['n_days']} test days")
    print(f"  mean daily rank IC   : {m['mean_ic']:+.4f}   (t-stat {m['ic_t_stat']:+.2f})")
    print(f"  IC hit rate          : {m['ic_hit_rate'] * 100:.1f}% of days positive")
    print(
        f"  long/short spread    : {m['ls_mean_daily'] * 1e4:+.2f} bp/day  "
        f"(Sharpe {m['ls_sharpe_annual']:+.2f}, win {m['ls_win_rate'] * 100:.1f}%)"
    )
    print(
        f"  vs 1-day reversal    : IC {m['baseline_reversal_ic']:+.4f} "
        f"(t {m['baseline_reversal_ic_t']:+.2f}), "
        f"L/S {m['baseline_reversal_ls_daily'] * 1e4:+.2f} bp/d "
        f"(Sharpe {m['baseline_reversal_ls_sharpe']:+.2f})"
    )
    print(f"  direction (residual) : {m['direction_accuracy_residual']:.2f}%   [50.00% = no skill]")
    print(
        f"  direction (raw)      : {m['direction_accuracy_raw']:.2f}%   "
        f"[always-up {m['baseline_always_up_raw']:.2f}%]"
    )
    print(
        f"  total train time     : {report['total_train_seconds']}s for {len(report['folds'])} folds"
    )
    print("-" * 66)
    beats_rev = m["mean_ic"] > m["baseline_reversal_ic"]
    verdict = "EDGE" if m["ic_t_stat"] > 2 and beats_rev else "NO significant edge"
    print(f"  VERDICT: {verdict} (needs pooled IC t > 2 AND beating 1-day reversal)")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2))
        print(f"  wrote {args.json_out}")

    # Universe gate first: a shrunken universe invalidates the IC gate below
    # rather than failing it, so it has to be its own check.
    if args.min_symbols is not None and report["n_symbols"] < args.min_symbols:
        print(f"  GATE FAILED: {report['n_symbols']} symbols < {args.min_symbols} expected")
        return 1
    if args.min_ic is not None and m["mean_ic"] < args.min_ic:
        print(f"  GATE FAILED: mean IC {m['mean_ic']:.4f} < {args.min_ic}")
        return 1
    return 0


def cmd_predict(args):
    res = M.load(args.model_path)
    data = _dataset(args)
    day = data[data["date"] == data["date"].max()].copy()
    day["pred"] = M.predict(res, day)
    day = day.sort_values("pred", ascending=False)
    print(f"\n  ranking for {day['date'].iloc[0].date()} ({len(day)} names)\n")
    for _, r in day.iterrows():
        print(f"    {r['symbol']:<8} {r['pred'] * 100:+7.3f}%  (vs universe)")
    return 0


def main(argv=None):
    # Shared flags live on a parent parser so they work before OR after the
    # subcommand — `-m ara eval --db-file x` is what everyone types first.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db-file", default="training.db")
    common.add_argument("--asset", default="stock", choices=["stock", "forex"])
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--max-iter", type=int, default=400)

    ap = argparse.ArgumentParser(
        prog="ara", parents=[common], description="Ara.AI v8 cross-sectional stock model"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser(
        "train", parents=[common], help="fit on all data (optionally before --train-end) and save"
    )
    t.add_argument("--output", default="models/ara_v8_stocks.joblib")
    t.add_argument("--train-end", help="train only on data strictly before this date")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", parents=[common], help="expanding-window walk-forward backtest")
    e.add_argument("--holdout-start", default="2025-06-01")
    e.add_argument("--folds", type=int, default=4)
    e.add_argument("--json-out")
    e.add_argument("--min-ic", type=float, help="exit 1 if mean IC is below this (CI gate)")
    e.add_argument(
        "--min-symbols", type=int, help="exit 1 if the universe is smaller than this (CI gate)"
    )
    e.set_defaults(fn=cmd_eval)

    p = sub.add_parser("predict", parents=[common], help="rank the most recent day in the DB")
    p.add_argument("--model-path", default="models/ara_v8_stocks.joblib")
    p.set_defaults(fn=cmd_predict)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
