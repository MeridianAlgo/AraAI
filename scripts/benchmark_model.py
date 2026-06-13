#!/usr/bin/env python3
"""Benchmark a trained checkpoint on a chronological holdout.

Builds per-symbol 30-day feature windows from the training database exactly
the way training does, holds out the most recent fraction of samples by date
(never seen during this evaluation's comparison baselines either), and
reports:

  - direction accuracy (sign of predicted next-day return)
  - F1 on the "up" class
  - MAE / RMSE of the predicted return vs realized return
  - prediction distribution stats (mean/std/%positive) to expose degeneracy
  - baselines: always-up, always-down, yesterday's-sign (momentum), and a
    52-week SMA trend rule — a model only matters if it beats these

Usage:
  python scripts/benchmark_model.py --model-path models/Meridian.AI_Forex.pt \
      --model-type forex --db-file training.db [--holdout 0.15]
"""

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from meridianalgo.large_torch_model import AdvancedMLSystem  # noqa: E402
from meridianalgo.unified_ml import UnifiedStockML  # noqa: E402

LOOKBACK = 30


def build_dataset(db_file, model_type, embargo=0):
    """Per-symbol windows + next-day returns + target-base dates + prev-day return.

    Mirrors UnifiedStockML.train_from_dataset: indicators per symbol, windows
    never cross symbol boundaries, samples sorted by target-base date.
    With embargo=e the input window ends e days BEFORE the target's base day
    (forex trains with embargo_days=1: the source bars leak the next close
    through day-t high/low — see scripts/diag_feat_corr.py).
    """
    asset = "stock" if model_type == "stock" else "forex"
    conn = sqlite3.connect(db_file)
    symbols = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT symbol FROM market_data WHERE asset_type = ? AND interval = '1d'",
            (asset,),
        ).fetchall()
    ]
    feat = UnifiedStockML.__new__(UnifiedStockML)  # feature engineering only

    X_parts, y_parts, d_parts, prev_parts = [], [], [], []
    for sym in symbols:
        rows = conn.execute(
            """SELECT date, open, high, low, close, volume
               FROM market_data
               WHERE asset_type = ? AND interval = '1d' AND symbol = ?
               ORDER BY date""",
            (asset, sym),
        ).fetchall()
        if len(rows) < LOOKBACK + embargo + 10:
            continue
        df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = UnifiedStockML._add_indicators(feat, df)
        fm = df[UnifiedStockML.FEATURE_COLS].values.astype(np.float32)
        cv = df["Close"].values.astype(np.float64)
        dv = df["Date"].values
        n = len(df)
        lo = LOOKBACK + embargo  # first target-base index
        Xg = np.array(
            [fm[i - LOOKBACK + 1 - embargo : i + 1 - embargo] for i in range(lo, n - 1)],
            dtype=np.float32,
        )
        yg = ((cv[lo + 1 : n] - cv[lo : n - 1]) / (cv[lo : n - 1] + 1e-10)).astype(np.float32)
        # realized return of the day BEFORE each target (for momentum baseline)
        pg = ((cv[lo : n - 1] - cv[lo - 1 : n - 2]) / (cv[lo - 1 : n - 2] + 1e-10)).astype(
            np.float32
        )
        dg = dv[lo : n - 1]
        m = np.isfinite(Xg).all(axis=(1, 2)) & np.isfinite(yg) & np.isfinite(pg)
        if not m.any():
            continue
        X_parts.append(Xg[m])
        y_parts.append(yg[m])
        d_parts.append(dg[m])
        prev_parts.append(pg[m])
    conn.close()

    X = np.concatenate(X_parts)
    y = np.concatenate(y_parts)
    dates = np.concatenate(d_parts)
    prev = np.concatenate(prev_parts)
    order = np.argsort(dates, kind="stable")
    return X[order], y[order], dates[order], prev[order]


def direction_stats(pred_sign, y):
    up = y > 0
    hit = pred_sign == up
    tp = int(np.sum(pred_sign & up))
    fp = int(np.sum(pred_sign & ~up))
    fn = int(np.sum(~pred_sign & up))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return float(hit.mean()) * 100, f1 * 100


def main():
    ap = argparse.ArgumentParser(description="Benchmark a checkpoint on a chronological holdout")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-type", choices=["stock", "forex"], required=True)
    ap.add_argument("--db-file", required=True)
    ap.add_argument("--holdout", type=float, default=0.15, help="Most-recent fraction held out")
    ap.add_argument(
        "--holdout-start",
        help="Evaluate only samples with window end-date >= this date (YYYY-MM-DD). "
        "Use with a model trained on data strictly before this date for an honest "
        "out-of-sample test. Overrides --holdout.",
    )
    ap.add_argument("--json-out", help="Optional path to write metrics JSON")
    ap.add_argument(
        "--embargo",
        type=int,
        default=None,
        help="Days between window end and the target's base day. Default: 1 for "
        "forex (matches training; the source bars leak the next close through "
        "day-t high/low), 0 for stocks.",
    )
    args = ap.parse_args()
    embargo = args.embargo if args.embargo is not None else (1 if args.model_type == "forex" else 0)

    print("=" * 64)
    print(f"  BENCHMARK — {args.model_type.upper()} | {args.model_path}")
    print("=" * 64)

    print(f"  embargo       : {embargo} day(s)")
    X, y, dates, prev = build_dataset(args.db_file, args.model_type, embargo=embargo)
    if args.holdout_start:
        cut = np.datetime64(pd.Timestamp(args.holdout_start))
        mask = dates >= cut
        if not mask.any():
            print(f"  FATAL: no samples on/after {args.holdout_start}")
            return 1
        Xh, yh, dh, ph = X[mask], y[mask], dates[mask], prev[mask]
    else:
        n_hold = max(1, int(len(X) * args.holdout))
        Xh, yh, dh, ph = X[-n_hold:], y[-n_hold:], dates[-n_hold:], prev[-n_hold:]
    print(f"  samples total : {len(X)}")
    print(f"  holdout       : {len(Xh)}  ({pd.Timestamp(dh[0]).date()} -> {pd.Timestamp(dh[-1]).date()})")

    sysm = AdvancedMLSystem(Path(args.model_path), model_type=args.model_type)
    if not sysm.is_trained():
        print("  FATAL: model failed to load")
        return 1

    t0 = time.time()
    preds = []
    bs = 512
    for i in range(0, len(Xh), bs):
        p, _ = sysm.predict(Xh[i : i + bs])
        preds.append(np.asarray(p).reshape(-1))
    preds = np.concatenate(preds)
    infer_s = time.time() - t0

    acc, f1 = direction_stats(preds > 0, yh)
    mae = float(np.mean(np.abs(preds - yh)))
    rmse = float(np.sqrt(np.mean((preds - yh) ** 2)))

    base_up, _ = direction_stats(np.ones_like(yh, dtype=bool), yh)
    base_dn, _ = direction_stats(np.zeros_like(yh, dtype=bool), yh)
    base_mom, _ = direction_stats(ph > 0, yh)
    # trend rule: predicted sign = sign of close-vs-SMA50 distance at window end
    trend_feat = Xh[:, -1, UnifiedStockML.FEATURE_COLS.index("close_vs_sma_50")]
    base_trend, _ = direction_stats(trend_feat > 0, yh)

    # MAE of always predicting 0 (the degenerate optimum for plain regression)
    mae_zero = float(np.mean(np.abs(yh)))

    print("-" * 64)
    print(f"  direction accuracy : {acc:6.2f}%   (F1 up-class {f1:.2f})")
    print(f"  return MAE / RMSE  : {mae:.5f} / {rmse:.5f}   (zero-pred MAE {mae_zero:.5f})")
    print(f"  pred mean/std      : {preds.mean():+.5f} / {preds.std():.5f}")
    print(f"  pred %positive     : {100 * float(np.mean(preds > 0)):.1f}%   (true %up {100 * float(np.mean(yh > 0)):.1f}%)")
    print(f"  inference          : {1000 * infer_s / len(Xh):.2f} ms/sample ({len(Xh)} samples, {infer_s:.1f}s)")
    print("-" * 64)
    print("  baselines (direction accuracy):")
    print(f"    always-up        : {base_up:6.2f}%")
    print(f"    always-down      : {base_dn:6.2f}%")
    print(f"    momentum (t-1)   : {base_mom:6.2f}%")
    print(f"    sma50 trend      : {base_trend:6.2f}%")
    best_base = max(base_up, base_dn, base_mom, base_trend)
    edge = acc - best_base
    print("-" * 64)
    print(f"  EDGE vs best baseline: {edge:+.2f} pts {'(beats baselines)' if edge > 0 else '(NO edge)'}")

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "model_path": str(args.model_path),
                    "model_type": args.model_type,
                    "embargo_days": embargo,
                    "holdout_samples": int(len(Xh)),
                    "holdout_start": str(pd.Timestamp(dh[0]).date()),
                    "holdout_end": str(pd.Timestamp(dh[-1]).date()),
                    "direction_accuracy": acc,
                    "f1_up": f1,
                    "mae": mae,
                    "rmse": rmse,
                    "mae_zero_pred": mae_zero,
                    "pred_mean": float(preds.mean()),
                    "pred_std": float(preds.std()),
                    "pred_pct_positive": 100 * float(np.mean(preds > 0)),
                    "true_pct_up": 100 * float(np.mean(yh > 0)),
                    "baseline_always_up": base_up,
                    "baseline_always_down": base_dn,
                    "baseline_momentum": base_mom,
                    "baseline_sma50_trend": base_trend,
                    "edge_vs_best_baseline": edge,
                    "ms_per_sample": 1000 * infer_s / len(Xh),
                },
                indent=2,
            )
        )
        print(f"  metrics written to {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
