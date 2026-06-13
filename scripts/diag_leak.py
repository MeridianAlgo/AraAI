#!/usr/bin/env python3
"""Leakage diagnostics for a suspiciously high benchmark score.

Checks, on the honest holdout:
  1. corr(pred, y) and accuracy by |y| quantile — a leak shows up as high
     accuracy even on tiny moves, where real signal is weakest.
  2. skip-day target: accuracy predicting the t+2 return instead of t+1.
     Real signal should mostly vanish; a window/target misalignment won't.
  3. lag-1 target: accuracy "predicting" the LAST return inside the window
     (already-known info). High accuracy here means the model is just
     reading a feature back, and high (1) would mean y overlaps with it.
  4. per-symbol accuracy spread.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


from meridianalgo.large_torch_model import AdvancedMLSystem  # noqa: E402

LOOKBACK = 30


def build_with_symbols(db_file, model_type):
    """build_dataset, but also returns symbol id and a t+2 target."""
    import sqlite3

    from meridianalgo.unified_ml import UnifiedStockML

    asset = "stock" if model_type == "stock" else "forex"
    conn = sqlite3.connect(db_file)
    symbols = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT symbol FROM market_data WHERE asset_type = ? AND interval = '1d'",
            (asset,),
        ).fetchall()
    ]
    feat = UnifiedStockML.__new__(UnifiedStockML)
    out = []
    for sym in symbols:
        rows = conn.execute(
            """SELECT date, open, high, low, close, volume FROM market_data
               WHERE asset_type = ? AND interval = '1d' AND symbol = ? ORDER BY date""",
            (asset, sym),
        ).fetchall()
        if len(rows) < LOOKBACK + 12:
            continue
        df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
        df["Date"] = pd.to_datetime(df["Date"])
        df = UnifiedStockML._add_indicators(feat, df)
        fm = df[UnifiedStockML.FEATURE_COLS].values.astype(np.float32)
        cv = df["Close"].values.astype(np.float64)
        dv = df["Date"].values
        n = len(df)
        # window ends at i, i in [LOOKBACK, n-3] so t+2 exists
        idx = np.arange(LOOKBACK, n - 2)
        Xg = np.array([fm[i - LOOKBACK + 1 : i + 1] for i in idx], dtype=np.float32)
        y1 = ((cv[idx + 1] - cv[idx]) / (cv[idx] + 1e-10)).astype(np.float32)
        y2 = ((cv[idx + 2] - cv[idx + 1]) / (cv[idx + 1] + 1e-10)).astype(np.float32)
        ylag = ((cv[idx] - cv[idx - 1]) / (cv[idx - 1] + 1e-10)).astype(np.float32)
        dg = dv[idx]
        m = np.isfinite(Xg).all(axis=(1, 2)) & np.isfinite(y1) & np.isfinite(y2) & np.isfinite(ylag)
        if m.any():
            out.append((sym, Xg[m], y1[m], y2[m], ylag[m], dg[m]))
    conn.close()
    return out


def main():
    model_path = sys.argv[1]
    model_type = sys.argv[2]
    db_file = sys.argv[3]
    cut = np.datetime64(pd.Timestamp(sys.argv[4]))

    data = build_with_symbols(db_file, model_type)
    sysm = AdvancedMLSystem(Path(model_path), model_type=model_type)

    all_pred, all_y1, all_y2, all_ylag, all_sym = [], [], [], [], []
    for sym, X, y1, y2, ylag, d in data:
        mask = d >= cut
        if not mask.any():
            continue
        Xh = X[mask]
        preds = []
        for i in range(0, len(Xh), 512):
            p, _ = sysm.predict(Xh[i : i + 512])
            preds.append(np.asarray(p).reshape(-1))
        preds = np.concatenate(preds)
        all_pred.append(preds)
        all_y1.append(y1[mask])
        all_y2.append(y2[mask])
        all_ylag.append(ylag[mask])
        all_sym.append(np.full(mask.sum(), sym))

    pred = np.concatenate(all_pred)
    y1 = np.concatenate(all_y1)
    y2 = np.concatenate(all_y2)
    ylag = np.concatenate(all_ylag)
    syms = np.concatenate(all_sym)

    def acc(p, y):
        return 100 * float(np.mean((p > 0) == (y > 0)))

    print(f"n={len(pred)}")
    print(f"acc vs t+1 target        : {acc(pred, y1):6.2f}%")
    print(f"acc vs t+2 target (skip) : {acc(pred, y2):6.2f}%")
    print(f"acc vs lag (known) ret   : {acc(pred, ylag):6.2f}%")
    print(f"corr(pred, y1)           : {np.corrcoef(pred, y1)[0, 1]:+.4f}")
    print(f"corr(pred, ylag)         : {np.corrcoef(pred, ylag)[0, 1]:+.4f}")
    print(f"corr(y1, ylag)           : {np.corrcoef(y1, ylag)[0, 1]:+.4f}")

    q = np.quantile(np.abs(y1), [0.25, 0.5, 0.75])
    bins = [
        ("|y| <= q25 (tiny moves) ", np.abs(y1) <= q[0]),
        ("q25 < |y| <= q50        ", (np.abs(y1) > q[0]) & (np.abs(y1) <= q[1])),
        ("q50 < |y| <= q75        ", (np.abs(y1) > q[1]) & (np.abs(y1) <= q[2])),
        ("|y| > q75 (big moves)   ", np.abs(y1) > q[2]),
    ]
    print("\naccuracy by |move| size:")
    for name, m in bins:
        print(f"  {name}: {acc(pred[m], y1[m]):6.2f}%  (n={m.sum()}, %up {100 * float(np.mean(y1[m] > 0)):.1f})")

    print("\nper-symbol accuracy (top/bottom 5):")
    rows = []
    for s in np.unique(syms):
        m = syms == s
        rows.append((s, acc(pred[m], y1[m]), int(m.sum()), 100 * float(np.mean(y1[m] > 0))))
    rows.sort(key=lambda r: -r[1])
    for s, a, n, pu in rows[:5]:
        print(f"  {s:12s} {a:6.2f}%  n={n}  %up={pu:.1f}")
    print("  ...")
    for s, a, n, pu in rows[-5:]:
        print(f"  {s:12s} {a:6.2f}%  n={n}  %up={pu:.1f}")


if __name__ == "__main__":
    main()
