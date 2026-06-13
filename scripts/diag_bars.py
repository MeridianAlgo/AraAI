#!/usr/bin/env python3
"""Probe raw bar alignment: where does close[t] sit relative to its own bar,
and how does close[t+1] relate to day-t OHLC?"""

import sqlite3
import sys

import numpy as np
import pandas as pd

db_file, asset = sys.argv[1], sys.argv[2]
conn = sqlite3.connect(db_file)
symbols = [
    r[0]
    for r in conn.execute(
        "SELECT DISTINCT symbol FROM market_data WHERE asset_type = ? AND interval = '1d'",
        (asset,),
    ).fetchall()
]

stats = []
for sym in symbols[:30]:
    df = pd.read_sql(
        "SELECT date, open, high, low, close FROM market_data "
        "WHERE asset_type = ? AND interval = '1d' AND symbol = ? ORDER BY date",
        conn,
        params=(asset, sym),
    )
    if len(df) < 500:
        continue
    o, h, lo, c = (df[k].values for k in ("open", "high", "low", "close"))
    c_next = np.roll(c, -1)[:-1]
    o_next = np.roll(o, -1)[:-1]
    h, lo, c, o = h[:-1], lo[:-1], c[:-1], o[:-1]
    rng = np.maximum(h - lo, 1e-12)
    stats.append(
        {
            "sym": sym,
            # where close sits in its own bar's range (0=low, 1=high)
            "close_pos_in_bar": float(np.mean((c - lo) / rng)),
            # is next close inside today's range?
            "next_close_in_bar": float(np.mean((c_next >= lo) & (c_next <= h))),
            # open[t+1] == close[t]?
            "o_next_eq_c": float(np.mean(np.abs(o_next - c) / c < 1e-4)),
            # open[t] == close[t]? (close sampled at bar start)
            "o_eq_c": float(np.mean(np.abs(o - c) / c < 1e-4)),
            # corr(next-day ret, today's range midpoint vs close)
            "corr_mid": float(
                np.corrcoef((h + lo) / 2 / c - 1, c_next / c - 1)[0, 1]
            ),
        }
    )

d = pd.DataFrame(stats)
print(f"asset={asset}  symbols={len(d)}")
print(d.drop(columns='sym').mean().round(4).to_string())
print("\nper-symbol (first 8):")
print(d.head(8).round(3).to_string(index=False))
