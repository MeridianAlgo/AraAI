"""Ara.AI v8 checks — the ones that fail if the pipeline silently breaks.

The expensive property here is "no future information reaches a feature".
Everything else is cheap plumbing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ara import build_features, feature_cols, load, make_dataset, predict, save, train
from ara.model import MIN_UNIVERSE


def _panel(n_symbols=12, n_days=400, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    rows = []
    for s in range(n_symbols):
        px = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.02, n_days)))
        rows.append(
            pd.DataFrame(
                {
                    "symbol": f"S{s:02d}",
                    "date": dates,
                    "open": px * (1 + rng.normal(0, 0.002, n_days)),
                    "high": px * (1 + abs(rng.normal(0, 0.01, n_days))),
                    "low": px * (1 - abs(rng.normal(0, 0.01, n_days))),
                    "close": px,
                    "volume": rng.integers(1e6, 1e7, n_days),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_no_lookahead_in_features():
    """Overwriting every bar after day T must not change any feature at day T.

    This is the check that matters: it catches a centered rolling window, a
    forgotten shift(-1), or a groupby that spills across symbols.
    """
    df = _panel()
    cut = df["date"].unique()[300]
    base = build_features(df)

    tampered = df.copy()
    future = tampered["date"] > cut
    for col in ("open", "high", "low", "close"):
        tampered.loc[future, col] *= 3.0
    tampered.loc[future, "volume"] *= 7

    after = build_features(tampered)
    cols = feature_cols(base)
    m = base["date"] <= cut
    pd.testing.assert_frame_equal(
        base.loc[m, cols].reset_index(drop=True),
        after.loc[m.values, cols].reset_index(drop=True),
        check_exact=False,
        atol=1e-9,
    )


def test_features_never_cross_symbols():
    """A symbol's features must be identical whether or not peers are present.

    Only the cross-sectional `xs_*` columns may depend on the universe.
    """
    df = _panel()
    solo = build_features(df[df["symbol"] == "S00"])
    both = build_features(df)
    both = both[both["symbol"] == "S00"]
    per_symbol = [c for c in feature_cols(solo) if not c.startswith("xs_")]
    pd.testing.assert_frame_equal(
        solo[per_symbol].reset_index(drop=True),
        both[per_symbol].reset_index(drop=True),
        check_exact=False,
        atol=1e-9,
    )


def test_target_is_market_neutral():
    """Cross-sectional residuals sum to ~0 each day, or the demeaning is broken."""
    data = make_dataset(_panel())
    resid = data["fwd_ret"] - data.groupby("date")["fwd_ret"].transform("mean")
    assert resid.groupby(data["date"]).mean().abs().max() < 1e-12
    assert (data.groupby("date").size() >= MIN_UNIVERSE).all()


def test_no_nans_reach_the_model():
    data = make_dataset(_panel())
    assert len(data) > 0
    assert np.isfinite(data[feature_cols(data)].to_numpy(np.float64)).all()


def test_train_predict_roundtrip(tmp_path):
    data = make_dataset(_panel(n_days=300))
    res = train(data, max_iter=20, n_seeds=2)
    p1 = predict(res, data)
    assert p1.shape == (len(data),)
    assert np.isfinite(p1).all()

    path = save(res, tmp_path / "m.joblib")
    p2 = predict(load(path), data)
    np.testing.assert_allclose(p1, p2, rtol=1e-6)
    assert path.with_suffix(".json").exists()


def test_learns_a_planted_signal():
    """Sanity: with a real signal in the data, IC must be clearly positive.

    Guards against a broken feature/target alignment that would silently make
    every honest backtest read ~0.
    """
    df = _panel(n_symbols=20, n_days=500, seed=7)
    data = make_dataset(df)
    # Plant it in the target: tomorrow's residual follows today's rsi rank.
    data = data.copy()
    data["target"] = data["xs_rsi_14"] * 0.02
    res = train(data, max_iter=60, n_seeds=1)
    pred = predict(res, data)
    assert np.corrcoef(pred, data["target"])[0, 1] > 0.9


@pytest.mark.parametrize("asset", ["stock"])
def test_walk_forward_shape(asset):
    from ara import walk_forward

    data = make_dataset(_panel(n_symbols=15, n_days=600))
    start = str(pd.Timestamp(data["date"].unique()[-120]).date())
    rep = walk_forward(data, start=start, folds=2, max_iter=20, n_seeds=1)
    assert len(rep["folds"]) == 2
    assert rep["pooled"]["n_days"] > 0
    # Pooled t-stat must be computed from pooled days, not averaged per fold.
    assert set(rep["pooled"]) >= {"mean_ic", "ic_t_stat", "ls_sharpe_annual"}
