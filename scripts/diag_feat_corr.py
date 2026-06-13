#!/usr/bin/env python3
"""Correlate each window-end feature with the next-day return on the holdout.

If a 'feature' carries future information (data artifact or pipeline bug),
its correlation with y_{t+1} will stand far out from the rest.
Also sanity-checks the raw bars: is close[t+1] abnormally predictable from
day-t OHLC alone?
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from meridianalgo.unified_ml import UnifiedStockML  # noqa: E402

db_file, asset, cutoff = sys.argv[1], sys.argv[2], sys.argv[3]
conn = sqlite3.connect(db_file)
symbols = [
    r[0]
    for r in conn.execute(
        "SELECT DISTINCT symbol FROM market_data WHERE asset_type = ? AND interval = '1d'",
        (asset,),
    ).fetchall()
]
feat = UnifiedStockML.__new__(UnifiedStockML)
cut = pd.Timestamp(cutoff)

F, Y = [], []
ohlc_rows = []
for sym in symbols:
    rows = conn.execute(
        """SELECT date, open, high, low, close, volume FROM market_data
           WHERE asset_type = ? AND interval = '1d' AND symbol = ? ORDER BY date""",
        (asset, sym),
    ).fetchall()
    if len(rows) < 300:
        continue
    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = UnifiedStockML._add_indicators(feat, df)
    y1 = df["Close"].shift(-1) / df["Close"] - 1
    mask = (df["Date"] >= cut) & y1.notna()
    fm = df.loc[mask, UnifiedStockML.FEATURE_COLS].values.astype(np.float64)
    F.append(fm)
    Y.append(y1[mask].values)
    sub = df.loc[mask, ["Date", "Open", "High", "Low", "Close"]].copy()
    sub["y1"] = y1[mask].values
    ohlc_rows.append(sub)

F = np.concatenate(F)
Y = np.concatenate(Y)
print(f"n={len(Y)}")
print("\nfeature corr with next-day return (sorted by |corr|):")
corrs = []
for j, name in enumerate(UnifiedStockML.FEATURE_COLS):
    col = F[:, j]
    if np.std(col) < 1e-12:
        corrs.append((name, 0.0))
        continue
    corrs.append((name, float(np.corrcoef(col, Y)[0, 1])))
corrs.sort(key=lambda t: -abs(t[1]))
for name, c in corrs[:15]:
    print(f"  {name:20s} {c:+.4f}")

# Raw-bar probe: how much of y1 do day-t OHL ratios explain?
d = pd.concat(ohlc_rows)
x1 = (d["High"] / d["Close"] - 1).values
x2 = (d["Low"] / d["Close"] - 1).values
x3 = (d["Open"] / d["Close"] - 1).values
X = np.column_stack([x1, x2, x3, np.ones(len(d))])
yv = d["y1"].values
beta, *_ = np.linalg.lstsq(X, yv, rcond=None)
pred = X @ beta
print("\nraw-bar OLS (high/close, low/close, open/close -> y1):")
print(f"  corr(pred, y1) = {np.corrcoef(pred, yv)[0, 1]:+.4f}")
print(f"  sign acc       = {100 * np.mean((pred > 0) == (yv > 0)):.2f}%")
