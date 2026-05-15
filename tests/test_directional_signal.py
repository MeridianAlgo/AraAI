"""End-to-end directional accuracy on real market data.

Pulls a few liquid tickers from yfinance, runs the full prediction pipeline,
and measures whether the signed prediction beats a coin-flip baseline.
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

    Mirrors the windowing logic in `UnifiedStockML.predict_ultimate`:
    `_extract_features(df.iloc[:i+1])` returns a single (44,) feature row
    computed on data ending at i. Stack `lookback` consecutive rows to form
    the (lookback, 44) input tensor.
    """
    from meridianalgo.revolutionary_model import RevolutionaryFinancialModel
    from meridianalgo.unified_ml import UnifiedStockML

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = RevolutionaryFinancialModel(**model_cls_args)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    ml = UnifiedStockML(model_path=str(ckpt_path))
    df = ml._add_indicators(df).dropna()
    seq_len = ckpt["seq_len"]
    if len(df) < seq_len + n_steps + 1:
        pytest.skip(
            f"not enough data after burn-in: have {len(df)}, " f"need {seq_len + n_steps + 1}"
        )

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


@REQUIRES_NET
@pytest.mark.parametrize("symbol", STOCK_SYMBOLS)
def test_stocks_directional_accuracy(symbol: str, stocks_ckpt_path, stocks_ckpt) -> None:
    yf = pytest.importorskip("yfinance")
    df = yf.Ticker(symbol).history(period="2y")
    if df.empty:
        pytest.skip(f"no data for {symbol}")
    args = {
        "input_size": stocks_ckpt["input_size"],
        "seq_len": stocks_ckpt["seq_len"],
        "dim": stocks_ckpt["dim"],
        "num_layers": stocks_ckpt["num_layers"],
        "num_heads": stocks_ckpt["num_heads"],
        "num_kv_heads": stocks_ckpt["num_kv_heads"],
        "num_experts": stocks_ckpt["num_experts"],
        "num_prediction_heads": stocks_ckpt["num_prediction_heads"],
        "dropout": stocks_ckpt["dropout"],
        "use_mamba": stocks_ckpt["use_mamba"],
        "mamba_state_dim": stocks_ckpt.get("mamba_state_dim", 16),
    }
    preds, actuals = _predict_returns(stocks_ckpt_path, args, df, n_steps=40)
    if len(preds) < 10:
        pytest.skip("too few paired samples")
    acc = float(np.mean(np.sign(preds) == np.sign(actuals)))
    print(f"\n{symbol} directional acc: {acc:.3f} (n={len(preds)})")
    assert (
        acc >= 0.50
    ), f"{symbol}: directional accuracy {acc:.3f} <= 50% - model not better than coin flip"


@REQUIRES_NET
@pytest.mark.parametrize("symbol", STOCK_SYMBOLS)
def test_stocks_predictions_have_variance(symbol: str, stocks_ckpt_path, stocks_ckpt) -> None:
    yf = pytest.importorskip("yfinance")
    df = yf.Ticker(symbol).history(period="2y")
    if df.empty:
        pytest.skip(f"no data for {symbol}")
    args = {
        "input_size": stocks_ckpt["input_size"],
        "seq_len": stocks_ckpt["seq_len"],
        "dim": stocks_ckpt["dim"],
        "num_layers": stocks_ckpt["num_layers"],
        "num_heads": stocks_ckpt["num_heads"],
        "num_kv_heads": stocks_ckpt["num_kv_heads"],
        "num_experts": stocks_ckpt["num_experts"],
        "num_prediction_heads": stocks_ckpt["num_prediction_heads"],
        "dropout": stocks_ckpt["dropout"],
        "use_mamba": stocks_ckpt["use_mamba"],
        "mamba_state_dim": stocks_ckpt.get("mamba_state_dim", 16),
    }
    preds, _ = _predict_returns(stocks_ckpt_path, args, df, n_steps=30)
    if len(preds) < 10:
        pytest.skip("too few samples")
    spread = float(preds.std())
    assert spread > 1e-6, f"{symbol}: predictions are constant ({spread:.2e})"


@REQUIRES_NET
@pytest.mark.parametrize("symbol", FOREX_SYMBOLS)
def test_forex_directional_accuracy(symbol: str, forex_ckpt_path, forex_ckpt) -> None:
    yf = pytest.importorskip("yfinance")
    df = yf.Ticker(symbol).history(period="2y")
    if df.empty:
        pytest.skip(f"no data for {symbol}")
    args = {
        "input_size": forex_ckpt["input_size"],
        "seq_len": forex_ckpt["seq_len"],
        "dim": forex_ckpt["dim"],
        "num_layers": forex_ckpt["num_layers"],
        "num_heads": forex_ckpt["num_heads"],
        "num_kv_heads": forex_ckpt["num_kv_heads"],
        "num_experts": forex_ckpt["num_experts"],
        "num_prediction_heads": forex_ckpt["num_prediction_heads"],
        "dropout": forex_ckpt["dropout"],
        "use_mamba": forex_ckpt["use_mamba"],
        "mamba_state_dim": forex_ckpt.get("mamba_state_dim", 16),
    }
    preds, actuals = _predict_returns(forex_ckpt_path, args, df, n_steps=40)
    if len(preds) < 10:
        pytest.skip("too few paired samples")
    acc = float(np.mean(np.sign(preds) == np.sign(actuals)))
    print(f"\n{symbol} directional acc: {acc:.3f} (n={len(preds)})")
    assert acc >= 0.50, f"{symbol}: directional accuracy {acc:.3f} <= 50%"
