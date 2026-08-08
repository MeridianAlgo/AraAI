"""Ara.AI v8 feature engineering — vectorized over the whole panel.

v7 built features with a per-symbol Python loop and 30-day tensor windows
(44 floats x 30 steps per sample). v8 uses one row per (symbol, date) with
pandas groupby ops over the entire panel: ~40x less memory, seconds instead
of minutes, and the same information — a tree model reads lagged returns
directly, it does not need the raw window.

Every feature is scale-invariant (returns, ratios, z-scores, cross-sectional
ranks). No price or volume LEVELS: those encode symbol identity, which is
the leak that pinned v6 at coin-flip.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Trailing windows for momentum/reversal. 252 = 1 trading year.
_RET_WINDOWS = (2, 5, 10, 21, 63, 126, 252)
_SMA_WINDOWS = (5, 20, 50, 200)

# Features that also get a cross-sectional (within-day) rank. Ranking removes
# the market factor: what matters is whether a name is cheap RELATIVE to its
# peers today, not its absolute reading.
_RANKED = ("ret_1", "ret_5", "ret_21", "ret_252", "vol_21", "volume_z", "close_vs_sma_50", "rsi_14")


def _roll(s: pd.Series, sym: pd.Series, w: int, how: str = "mean") -> pd.Series:
    r = s.groupby(sym, sort=False).rolling(w, min_periods=w)
    return getattr(r, how)().reset_index(level=0, drop=True).sort_index()


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Panel in (symbol, date, open, high, low, close, volume) -> features + fwd_ret.

    Rows are sorted by (symbol, date). Every feature at row t uses bars up to
    and including t; `fwd_ret` is the t -> t+1 close-to-close return and is the
    ONLY column containing future information.
    """
    df = df.sort_values(["symbol", "date"], kind="stable").reset_index(drop=True)
    sym = df["symbol"]
    c, h, lo, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]
    g = df.groupby("symbol", sort=False)

    f = pd.DataFrame(index=df.index)
    r1 = g["close"].pct_change()
    f["ret_1"] = r1
    for k in _RET_WINDOWS:
        f[f"ret_{k}"] = g["close"].pct_change(k)

    # Realized volatility and volatility-normalized shock — the single most
    # useful transform on daily bars: a 2% move means different things in a
    # calm vs a stressed name.
    vol21 = _roll(r1, sym, 21, "std")
    vol63 = _roll(r1, sym, 63, "std")
    f["vol_21"] = vol21
    f["vol_63"] = vol63
    f["vol_ratio"] = vol21 / (vol63 + 1e-8)
    f["shock_1"] = r1 / (vol21 + 1e-8)
    f["shock_5"] = f["ret_5"] / (vol21 * np.sqrt(5) + 1e-8)

    for k in _SMA_WINDOWS:
        f[f"close_vs_sma_{k}"] = c / (_roll(c, sym, k) + 1e-10) - 1.0

    # Intraday structure: where the close sat in the day's range, the gap, and
    # the range itself (a cheap volatility/liquidity proxy).
    rng = (h - lo).replace(0, np.nan)
    f["close_pos"] = (c - lo) / rng
    f["hl_range"] = rng / c
    f["gap"] = o / (g["close"].shift(1) + 1e-10) - 1.0
    f["body"] = (c - o) / (rng + 1e-10)

    prev_c = g["close"].shift(1)
    tr = pd.concat([h - lo, (h - prev_c).abs(), (lo - prev_c).abs()], axis=1).max(axis=1)
    f["atr_pct"] = _roll(tr, sym, 14) / c

    # Wilder RSI via EWM (alpha = 1/period).
    up = r1.clip(lower=0)
    dn = (-r1).clip(lower=0)
    au = up.groupby(sym, sort=False).transform(lambda s: s.ewm(alpha=1 / 14, min_periods=14).mean())
    ad = dn.groupby(sym, sort=False).transform(lambda s: s.ewm(alpha=1 / 14, min_periods=14).mean())
    f["rsi_14"] = 100 - 100 / (1 + au / (ad + 1e-10))

    # Volume in units of its own trailing distribution.
    lv = np.log1p(v.astype("float64"))
    f["volume_z"] = (lv - _roll(lv, sym, 21)) / (_roll(lv, sym, 21, "std") + 1e-8)
    f["volume_ratio"] = _roll(lv, sym, 5) / (_roll(lv, sym, 21) + 1e-10) - 1.0
    f["dollar_vol_z"] = (
        np.log1p(v.astype("float64") * c) - _roll(np.log1p(v.astype("float64") * c), sym, 63)
    ) / (_roll(np.log1p(v.astype("float64") * c), sym, 63, "std") + 1e-8)

    f["dist_high_252"] = c / (_roll(c, sym, 252, "max") + 1e-10) - 1.0
    f["dist_low_252"] = c / (_roll(c, sym, 252, "min") + 1e-10) - 1.0
    f["zscore_20"] = (c - _roll(c, sym, 20)) / (_roll(c, sym, 20, "std") + 1e-10)

    # Calendar: weekday and month-phase carry real (small) daily seasonality.
    dt = pd.to_datetime(df["date"])
    f["dow"] = dt.dt.dayofweek.astype("float32")
    f["dom"] = dt.dt.day.astype("float32")

    f["date"] = df["date"].values
    f["symbol"] = sym.values

    # Cross-sectional ranks, centered on 0. This is the v8 idea: the model is
    # asked which names outperform TODAY'S universe, not where the market goes.
    for col in _RANKED:
        f[f"xs_{col}"] = f.groupby("date")[col].rank(pct=True) - 0.5

    f["fwd_ret"] = g["close"].shift(-1) / c - 1.0
    return f


NON_FEATURES = ("date", "symbol", "fwd_ret", "target")


def feature_cols(f: pd.DataFrame) -> list[str]:
    """Model inputs = every built column except the keys and the target."""
    return [c for c in f.columns if c not in NON_FEATURES]
