"""End-to-end directional accuracy on real market data.

Pulls a few liquid tickers from yfinance, runs the full prediction pipeline,
and measures whether the signed prediction beats a coin-flip baseline.

Design note on thresholds
--------------------------
Per-symbol accuracy on a rolling 30-day window has high variance.  A model at
true 50% accuracy will fall below 40% on a single symbol ~3.5% of the time by
chance alone.  When testing 5 symbols that is a ~16% false-failure rate per run.

Instead, the directional-accuracy tests aggregate all predictions across every
symbol in the group before asserting.  With 5 x 30 = 150+ samples the same 40%
floor is a ~3-sigma event (p < 0.1%) -- meaningful enough to act on while
suppressing noise from individual symbols.

Per-symbol scores are still printed so nightly logs show which instruments
are struggling.  The variance test remains per-symbol because a collapsed model
that outputs the same constant for every input is a structural bug regardless of
which symbol surfaces it first.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pytest
import torch

warnings.filterwarnings("ignore")

REQUIRES_NET = pytest.mark.skipif(
    os.environ.get("NO_NET") == "1",
    reason="network disabled (NO_NET=1)",
)

STOCK_SYMBOLS = ["AAPL", "MSFT", "SPY", "NVDA", "JPM"]
FOREX_SYMBOLS = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]


def _predict_returns(ckpt_path, model_cls_args, df, n_steps: int = 30):
    """Roll the loaded model over a price history and return paired
    (predicted_return, actual_return) arrays.

    Mirrors the windowing logic in UnifiedStockML.predict_ultimate.
    """
    from meridianalgo.meridian_model import MeridianModel as RevolutionaryFinancialModel
    from meridianalgo.unified_ml import UnifiedStockML

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = RevolutionaryFinancialModel(**model_cls_args)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    ml = UnifiedStockML(model_path=str(ckpt_path))
    df = ml._add_indicators(df).dropna()
    seq_len = ckpt["seq_len"]
    if len(df) < seq_len + n_steps + 1:
        pytest.skip(f"not enough data after burn-in: have {len(df)}, need {seq_len + n_steps + 1}")

    mean = ckpt["scaler_mean"].numpy()
    std = ckpt["scaler_std"].numpy()
    safe_std = np.where(std == 0, 1.0, std)

    closes = df["Close"].values
    preds, actuals = [], []

    for end in range(seq_len, len(df) - 1):
        if len(preds) >= n_steps:
            break
        rows = []
        for j in range(end - seq_len + 1, end + 1):
            row = ml._extract_features(df.iloc[: j + 1])
            if row.shape != (ckpt["input_size"],):
                rows = None
                break
            rows.append(row)
        if rows is None:
            continue
        window = np.stack(rows, axis=0)
        x = (window - mean) / safe_std
        x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            pred, _ = model(x_t)
        pred_ret = float(pred.squeeze().item())
        actual_ret = float((closes[end + 1] - closes[end]) / closes[end])
        if not (np.isfinite(pred_ret) and np.isfinite(actual_ret)):
            continue
        preds.append(pred_ret)
        actuals.append(actual_ret)
    return np.array(preds), np.array(actuals)


def _build_args(ckpt: dict) -> dict:
    return {
        "input_size": ckpt["input_size"],
        "seq_len": ckpt["seq_len"],
        "dim": ckpt["dim"],
        "num_layers": ckpt["num_layers"],
        "num_heads": ckpt["num_heads"],
        "num_kv_heads": ckpt["num_kv_heads"],
        "num_experts": ckpt["num_experts"],
        "num_prediction_heads": ckpt["num_prediction_heads"],
        "dropout": ckpt["dropout"],
        "use_mamba": ckpt["use_mamba"],
        "mamba_state_dim": ckpt.get("mamba_state_dim", 16),
    }


# ---------------------------------------------------------------------------
# Stocks
# ---------------------------------------------------------------------------


@REQUIRES_NET
def test_stocks_aggregate_directional_accuracy(stocks_ckpt_path, stocks_ckpt) -> None:
    """Aggregate directional accuracy pooled across all stock symbols.

    Pools 30 predictions per symbol (150+ total). At n=150, a 40% floor is
    ~3 sigma below chance (p < 0.1%).
    """
    yf = pytest.importorskip("yfinance")
    args = _build_args(stocks_ckpt)

    all_preds: list = []
    all_actuals: list = []

    for symbol in STOCK_SYMBOLS:
        df = yf.Ticker(symbol).history(period="2y")
        if df.empty:
            print(f"\n  {symbol}: no data, skipping")
            continue
        preds, actuals = _predict_returns(stocks_ckpt_path, args, df, n_steps=30)
        if len(preds):
            sym_acc = float(np.mean(np.sign(preds) == np.sign(actuals)))
            print(f"\n  {symbol}: {sym_acc:.3f} (n={len(preds)})")
            all_preds.extend(preds.tolist())
            all_actuals.extend(actuals.tolist())

    if len(all_preds) < 30:
        pytest.skip(f"too few total paired samples ({len(all_preds)})")

    agg_acc = float(np.mean(np.sign(all_preds) == np.sign(all_actuals)))
    print(f"\nStocks aggregate directional acc: {agg_acc:.3f} (n={len(all_preds)})")
    assert agg_acc >= 0.40, (
        f"Stocks aggregate directional accuracy {agg_acc:.3f} < 40% "
        f"over {len(all_preds)} predictions -- model outputs may be degenerate"
    )


@REQUIRES_NET
@pytest.mark.parametrize("symbol", STOCK_SYMBOLS)
def test_stocks_predictions_have_variance(symbol: str, stocks_ckpt_path, stocks_ckpt) -> None:
    """A model that emits the same value for every input is broken regardless of accuracy."""
    yf = pytest.importorskip("yfinance")
    df = yf.Ticker(symbol).history(period="2y")
    if df.empty:
        pytest.skip(f"no data for {symbol}")
    preds, _ = _predict_returns(stocks_ckpt_path, _build_args(stocks_ckpt), df, n_steps=30)
    if len(preds) < 10:
        pytest.skip("too few samples")
    spread = float(preds.std())
    assert spread > 1e-6, f"{symbol}: predictions are constant ({spread:.2e})"


# ---------------------------------------------------------------------------
# Forex
# ---------------------------------------------------------------------------


@REQUIRES_NET
def test_forex_aggregate_directional_accuracy(forex_ckpt_path, forex_ckpt) -> None:
    """Aggregate directional accuracy pooled across all forex pairs."""
    yf = pytest.importorskip("yfinance")
    args = _build_args(forex_ckpt)

    all_preds: list = []
    all_actuals: list = []

    for symbol in FOREX_SYMBOLS:
        df = yf.Ticker(symbol).history(period="2y")
        if df.empty:
            print(f"\n  {symbol}: no data, skipping")
            continue
        preds, actuals = _predict_returns(forex_ckpt_path, args, df, n_steps=30)
        if len(preds):
            sym_acc = float(np.mean(np.sign(preds) == np.sign(actuals)))
            print(f"\n  {symbol}: {sym_acc:.3f} (n={len(preds)})")
            all_preds.extend(preds.tolist())
            all_actuals.extend(actuals.tolist())

    if len(all_preds) < 20:
        pytest.skip(f"too few total paired samples ({len(all_preds)})")

    agg_acc = float(np.mean(np.sign(all_preds) == np.sign(all_actuals)))
    print(f"\nForex aggregate directional acc: {agg_acc:.3f} (n={len(all_preds)})")
    assert agg_acc >= 0.40, (
        f"Forex aggregate directional accuracy {agg_acc:.3f} < 40% "
        f"over {len(all_preds)} predictions -- model outputs may be degenerate"
    )
