"""Ara.AI v8 — cross-sectional gradient-boosted stock model.

What changed from v7 and why
----------------------------
v7 was a 433K-parameter transformer (GQA + MoE + RoPE) trained to predict each
symbol's *absolute* next-day return. Honest out-of-sample result: 50.23%
direction accuracy against a 51.44% always-up baseline, with return MAE sitting
exactly on the zero-prediction floor. It learned nothing, in 7 minutes of CPU
per run, 24 runs a day.

Two things were wrong, and neither was the model size:

1. **The target.** Most of a stock's next-day return is the market's next-day
   return, which daily OHLCV cannot predict. v8 predicts the *cross-sectional
   residual* — return minus the universe mean that day — so the model spends
   its capacity on the part that is actually forecastable, and the tradable
   output is a dollar-neutral long/short ranking.
2. **The model class.** ~40 weak, noisy tabular features is the exact regime
   where gradient-boosted trees dominate deep nets. `HistGradientBoosting`
   trains on the full panel in seconds on two CI cores, with no torch, no
   accelerate, and no GPU.

The metric changed too. Direction accuracy on raw returns is dominated by
market drift (that is why always-up "wins"), so it measures almost nothing.
v8 reports **daily rank IC** and **long/short decile spread**, which are the
standard cross-sectional measures and cannot be gamed by drift.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .features import build_features, feature_cols

MODEL_VERSION = "8.0.0"
ARCHITECTURE_NAME = "AraAI-v8-CrossSectional"

# A day needs enough names for a cross-sectional rank to mean anything.
MIN_UNIVERSE = 10
# Winsorize the target at +/- N per-day sigma. Daily equity returns are fat
# tailed; without this a handful of earnings gaps drive the whole fit.
WINSOR_SIGMA = 4.0
# Boosters averaged per fit. 3 kills most of the seed lottery; more barely moves
# IC and multiplies CI time.
N_SEEDS = 3


@dataclass
class TrainResult:
    models: list[HistGradientBoostingRegressor]
    features: list[str]
    n_train: int
    train_seconds: float
    meta: dict = field(default_factory=dict)


def load_panel(db_file: str | Path, asset_type: str = "stock") -> pd.DataFrame:
    """Read daily bars straight out of the training DB."""
    with sqlite3.connect(str(db_file)) as conn:
        df = pd.read_sql_query(
            "SELECT symbol, date, open, high, low, close, volume FROM market_data "
            "WHERE asset_type = ? AND interval = '1d' ORDER BY symbol, date",
            conn,
            params=(asset_type,),
        )
    if df.empty:
        raise ValueError(f"no {asset_type} 1d rows in {db_file}")
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def make_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Features + cross-sectional residual target, cleaned and date-sorted."""
    f = build_features(df)
    cols = feature_cols(f)
    f = f[np.isfinite(f["fwd_ret"]) & f[cols].notna().all(axis=1)]

    # Drop thin days before demeaning — a 3-name "universe" mean is noise.
    counts = f.groupby("date")["symbol"].transform("size")
    f = f[counts >= MIN_UNIVERSE].copy()

    grp = f.groupby("date")["fwd_ret"]
    resid = f["fwd_ret"] - grp.transform("mean")
    sd = grp.transform("std")
    f["target"] = resid.clip(-WINSOR_SIGMA * sd, WINSOR_SIGMA * sd)
    return f.sort_values("date", kind="stable").reset_index(drop=True)


def train(
    data: pd.DataFrame, seed: int = 0, max_iter: int = 400, n_seeds: int = N_SEEDS
) -> TrainResult:
    """Fit the booster on every row of `data` (already time-filtered by caller).

    Trains `n_seeds` boosters and averages them at predict time. The signal here
    is weak enough that a single seed's IC swings +0.010 to +0.016 on the same
    data - the spread between seeds is ~60% of the effect. Averaging costs a few
    seconds and removes the lottery.
    """
    cols = feature_cols(data)
    t0 = time.time()
    X = data[cols].to_numpy(np.float32)
    # Target is ~1e-2; scale to percent so tree split gains are not denormal.
    y = data["target"].to_numpy(np.float64) * 100.0
    models = []
    for i in range(n_seeds):
        m = HistGradientBoostingRegressor(
            loss="squared_error",
            max_iter=max_iter,
            learning_rate=0.03,
            max_leaf_nodes=15,
            min_samples_leaf=200,
            l2_regularization=1.0,
            max_features=0.7,
            early_stopping=False,  # sklearn's split is random, which leaks across time
            random_state=seed + i,
        )
        m.fit(X, y)
        models.append(m)
    return TrainResult(
        models=models,
        features=cols,
        n_train=len(data),
        train_seconds=time.time() - t0,
        meta={
            "model_version": MODEL_VERSION,
            "architecture": ARCHITECTURE_NAME,
            "trained_at": pd.Timestamp.now("UTC").isoformat(),
            "train_start": str(data["date"].min().date()),
            "train_end": str(data["date"].max().date()),
            "n_symbols": int(data["symbol"].nunique()),
            "seed": seed,
            "n_seeds": n_seeds,
        },
    )


def predict(res: TrainResult, data: pd.DataFrame) -> np.ndarray:
    """Predicted residual return (in return units, not percent)."""
    X = data[res.features].to_numpy(np.float32)
    return np.mean([m.predict(X) for m in res.models], axis=0) / 100.0


# --------------------------------------------------------------------------
# Evaluation — cross-sectional metrics, plus the v7 metrics for continuity.
# --------------------------------------------------------------------------


def _daily_ic(dates: pd.Series, pred: np.ndarray, y: np.ndarray) -> pd.Series:
    d = pd.DataFrame({"d": dates.values, "p": pred, "y": y})
    r = d.groupby("d")[["p", "y"]].rank()
    r["d"] = d["d"].values
    return r.groupby("d").apply(lambda x: x["p"].corr(x["y"]), include_groups=False).dropna()


def _long_short(dates: pd.Series, pred: np.ndarray, y: np.ndarray, k: int = 5) -> pd.Series:
    """Daily P&L of longing the top-k and shorting the bottom-k by prediction."""
    d = pd.DataFrame({"d": dates.values, "p": pred, "y": y})

    def leg(x):
        if len(x) < 2 * k:
            return np.nan
        s = x.sort_values("p")
        return s["y"].tail(k).mean() - s["y"].head(k).mean()

    return d.groupby("d")[["p", "y"]].apply(leg).dropna()


def _t_stat(s: pd.Series) -> float:
    return float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))) if len(s) > 1 and s.std() else 0.0


def _sharpe(s: pd.Series) -> float:
    return float(s.mean() / s.std(ddof=1) * np.sqrt(252)) if len(s) > 1 and s.std() else 0.0


def evaluate(res: TrainResult, test: pd.DataFrame, k: int = 5) -> dict:
    pred = predict(res, test)
    y_resid = test["fwd_ret"].to_numpy() - test.groupby("date")["fwd_ret"].transform("mean")
    y_raw = test["fwd_ret"].to_numpy()

    ic = _daily_ic(test["date"], pred, y_resid.to_numpy())
    ls = _long_short(test["date"], pred, y_resid.to_numpy(), k=k)

    # Baseline: naive 1-day reversal (yesterday's loser is today's winner). If
    # the booster cannot beat one negated column, it has learned nothing worth
    # 400 trees.
    rev = -test["ret_1"].to_numpy()
    ic_rev = _daily_ic(test["date"], rev, y_resid.to_numpy())
    ls_rev = _long_short(test["date"], rev, y_resid.to_numpy(), k=k)

    ic_t = _t_stat(ic)
    sharpe = _sharpe(ls)

    return {
        "_ic": ic,  # daily series, pooled across folds by walk_forward
        "_ls": ls,
        "_ic_rev": ic_rev,
        "_ls_rev": ls_rev,
        "baseline_reversal_ic": float(ic_rev.mean()),
        "baseline_reversal_ls_daily": float(ls_rev.mean()),
        "n_test": int(len(test)),
        "n_days": int(len(ic)),
        "test_start": str(test["date"].min().date()),
        "test_end": str(test["date"].max().date()),
        # --- primary metrics ---
        "mean_ic": float(ic.mean()),
        "ic_t_stat": ic_t,
        "ic_hit_rate": float((ic > 0).mean()),
        "ls_mean_daily": float(ls.mean()),
        "ls_sharpe_annual": sharpe,
        "ls_win_rate": float((ls > 0).mean()),
        # --- continuity with the v7 report ---
        "direction_accuracy_residual": float(((pred > 0) == (y_resid > 0)).mean() * 100),
        "direction_accuracy_raw": float(((pred > 0) == (y_raw > 0)).mean() * 100),
        "baseline_always_up_raw": float((y_raw > 0).mean() * 100),
        "mae_residual": float(np.abs(pred - y_resid).mean()),
        "mae_zero_pred_residual": float(np.abs(y_resid).mean()),
        "pred_std": float(pred.std()),
    }


def walk_forward(
    data: pd.DataFrame,
    start: str,
    folds: int = 4,
    embargo_days: int = 1,
    seed: int = 0,
    max_iter: int = 400,
    n_seeds: int = N_SEEDS,
) -> dict:
    """Expanding-window backtest: retrain before each fold, never look ahead.

    Fold i trains on everything up to (fold start - embargo) and tests on the
    fold. This is the honest number; a single split can get lucky on one regime.
    """
    cut = pd.Timestamp(start)
    days = np.array(sorted(data.loc[data["date"] >= cut, "date"].unique()))
    if len(days) < folds * 5:
        raise ValueError(f"only {len(days)} test days on/after {start}")
    bounds = np.array_split(days, folds)

    per_fold, total_train_s = [], 0.0
    for i, chunk in enumerate(bounds):
        f0, f1 = chunk[0], chunk[-1]
        tr = data[data["date"] < f0 - pd.Timedelta(days=embargo_days)]
        te = data[(data["date"] >= f0) & (data["date"] <= f1)]
        if tr.empty or te.empty:
            continue
        res = train(tr, seed=seed, max_iter=max_iter, n_seeds=n_seeds)
        total_train_s += res.train_seconds
        m = evaluate(res, te)
        m["fold"] = i
        m["n_train"] = res.n_train
        m["train_seconds"] = round(res.train_seconds, 2)
        per_fold.append(m)

    # Pool the DAILY series across folds before computing t-stat/Sharpe. The
    # mean of four per-fold t-stats is not a t-stat; 250 pooled days is the
    # sample that actually decides whether the edge is real.
    ic = pd.concat([f.pop("_ic") for f in per_fold])
    ls = pd.concat([f.pop("_ls") for f in per_fold])
    ic_rev = pd.concat([f.pop("_ic_rev") for f in per_fold])
    ls_rev = pd.concat([f.pop("_ls_rev") for f in per_fold])

    return {
        "architecture": ARCHITECTURE_NAME,
        "model_version": MODEL_VERSION,
        # The universe defines what "cross-sectional" means, so it belongs in
        # the report: two runs on different symbol sets are not comparable, and
        # that is not visible from the metrics alone.
        "n_symbols": int(data["symbol"].nunique()),
        "n_samples": int(len(data)),
        "median_universe_per_day": float(data.groupby("date").size().median()),
        "folds": per_fold,
        "pooled": {
            "n_days": int(len(ic)),
            "mean_ic": float(ic.mean()),
            "ic_t_stat": _t_stat(ic),
            "ic_hit_rate": float((ic > 0).mean()),
            "ls_mean_daily": float(ls.mean()),
            "ls_sharpe_annual": _sharpe(ls),
            "ls_win_rate": float((ls > 0).mean()),
            "ls_t_stat": _t_stat(ls),
            "baseline_reversal_ic": float(ic_rev.mean()),
            "baseline_reversal_ic_t": _t_stat(ic_rev),
            "baseline_reversal_ls_daily": float(ls_rev.mean()),
            "baseline_reversal_ls_sharpe": _sharpe(ls_rev),
            "direction_accuracy_residual": float(
                np.mean([f["direction_accuracy_residual"] for f in per_fold])
            ),
            "direction_accuracy_raw": float(
                np.mean([f["direction_accuracy_raw"] for f in per_fold])
            ),
            "baseline_always_up_raw": float(
                np.mean([f["baseline_always_up_raw"] for f in per_fold])
            ),
        },
        "total_train_seconds": round(total_train_s, 2),
    }


# --------------------------------------------------------------------------
# Persistence — joblib payload, no custom checkpoint format.
# --------------------------------------------------------------------------


def save(res: TrainResult, path: str | Path, extra: dict | None = None) -> Path:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {**res.meta, "n_train": res.n_train, **(extra or {})}
    joblib.dump({"models": res.models, "features": res.features, "meta": meta}, path, compress=3)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return path


def load(path: str | Path) -> TrainResult:
    import joblib

    p = joblib.load(str(path))
    return TrainResult(
        models=p["models"],
        features=p["features"],
        n_train=p["meta"].get("n_train", 0),
        train_seconds=0.0,
        meta=p["meta"],
    )
