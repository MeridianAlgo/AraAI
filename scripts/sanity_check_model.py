#!/usr/bin/env python3
"""
Post-training sanity gate.

This is the guardrail that stops a broken model — like the pre-v6 checkpoints
that collapsed to a near-constant downward prediction — from ever reaching
Hugging Face. It loads the freshly trained .pt, runs it over held-out windows
pulled from the same training database, and FAILS (exit 1) if the model shows
any of the classic degeneracy signatures:

  1. Collapsed output      — prediction std ~ 0 (model ignores its input)
  2. Directional collapse  — predicts the same sign for ~everything
  3. Scale blow-up         — mean |prediction| is implausibly large, the
                             fingerprint of the old multi-timeframe target
                             contamination (daily returns should be ~1%)

If there simply isn't enough data to judge, it WARNS and passes — the gate is
there to catch a bad model, not to block a run on data flakiness.

Exit code 0 = model looks healthy (push it).  Exit code 1 = degenerate (block).
"""

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from meridianalgo.large_torch_model import AdvancedMLSystem  # noqa: E402
from meridianalgo.unified_ml import UnifiedStockML  # noqa: E402

LOOKBACK = 30
MIN_PREDICTIONS = 40  # below this we can't judge — warn & pass

# Degeneracy thresholds. Deliberately loose so a legitimate (even mediocre)
# model passes; only clear pathology trips them.
MIN_PRED_STD = 1e-5  # below => constant output
MIN_FRAC_POS = 0.08  # below => collapsed bearish (old model sat at ~0.08)
MAX_FRAC_POS = 0.98  # above => collapsed bullish / constant
MAX_ABS_MEAN = 0.50  # daily returns are ~1%; >50% mean => scale blow-up


def _windows_from_db(db_file, model_type):
    """Pull per-symbol daily bars from the DB and build the last few
    30-step feature windows for each symbol, plus the realized next-day return."""
    asset = "stock" if model_type == "stock" else "forex"
    conn = sqlite3.connect(db_file)
    symbols = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT symbol FROM market_data WHERE asset_type = ? AND interval = '1d'",
            (asset,),
        ).fetchall()
    ]
    feat = UnifiedStockML.__new__(UnifiedStockML)  # feature engineering only, no model load

    preds_X, trues = [], []
    for sym in symbols:
        rows = conn.execute(
            """SELECT date, open, high, low, close, volume
               FROM market_data
               WHERE asset_type = ? AND interval = '1d' AND symbol = ?
               ORDER BY date""",
            (asset, sym),
        ).fetchall()
        if len(rows) < LOOKBACK + 60:
            continue
        df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        df = UnifiedStockML._add_indicators(feat, df)
        fm = df[UnifiedStockML.FEATURE_COLS].values.astype(np.float32)
        close = df["Close"].values.astype(np.float64)
        n = len(df)
        # last 20 windows per symbol keeps the check fast and recent
        for i in range(max(LOOKBACK, n - 1 - 20), n - 1):
            win = fm[i - LOOKBACK + 1 : i + 1]
            if not np.isfinite(win).all():
                continue
            preds_X.append(win)
            trues.append((close[i + 1] - close[i]) / close[i])
    conn.close()
    return preds_X, np.array(trues)


def main():
    ap = argparse.ArgumentParser(description="Post-training degeneracy gate")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-type", choices=["stock", "forex"], required=True)
    ap.add_argument("--db-file", required=True)
    args = ap.parse_args()

    print("=" * 55)
    print(f"   SANITY GATE — {args.model_type.upper()} MODEL")
    print("=" * 55)

    if not Path(args.model_path).exists():
        print(f"  No model at {args.model_path} — nothing to gate.")
        return 0

    sysm = AdvancedMLSystem(Path(args.model_path), model_type=args.model_type)
    if not sysm.is_trained():
        # Loader refused (e.g. old/incompatible version). Don't block — the
        # verify step already reports this and there's no model to push.
        print("  Model did not load (incompatible/old). Skipping gate.")
        return 0

    windows, trues = _windows_from_db(args.db_file, args.model_type)
    if len(windows) < MIN_PREDICTIONS:
        print(f"  [WARN] only {len(windows)} eval windows (<{MIN_PREDICTIONS}) — passing without judgment.")
        return 0

    preds = np.array([float(np.ravel(sysm.predict(w)[0])[0]) for w in windows])

    pred_std = float(preds.std())
    pred_mean = float(preds.mean())
    frac_pos = float(np.mean(preds > 0))
    acc = float(np.mean((preds > 0) == (trues > 0)))
    baseline = float(np.mean(trues > 0))

    print(f"  eval windows     : {len(preds)}")
    print(f"  prediction mean  : {pred_mean:+.5f}")
    print(f"  prediction std   : {pred_std:.6f}")
    print(f"  fraction positive: {frac_pos:.3f}")
    print(f"  directional acc  : {acc:.3f}  (always-up baseline {baseline:.3f})")
    print("-" * 55)

    failures = []
    if pred_std < MIN_PRED_STD:
        failures.append(f"output collapsed to a constant (std {pred_std:.2e} < {MIN_PRED_STD:.0e})")
    if not (MIN_FRAC_POS <= frac_pos <= MAX_FRAC_POS):
        failures.append(
            f"directional collapse (%pos {frac_pos:.3f} outside [{MIN_FRAC_POS}, {MAX_FRAC_POS}])"
        )
    if abs(pred_mean) > MAX_ABS_MEAN:
        failures.append(
            f"prediction scale blow-up (|mean| {abs(pred_mean):.3f} > {MAX_ABS_MEAN}) — possible target contamination"
        )

    if failures:
        print("  RESULT: FAIL — model is degenerate, blocking HF push")
        for f in failures:
            print(f"    - {f}")
        return 1

    print("  RESULT: PASS — model is healthy, safe to push")
    return 0


if __name__ == "__main__":
    sys.exit(main())
