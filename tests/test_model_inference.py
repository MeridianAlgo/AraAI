"""Forward-pass and numerical-stability tests on the loaded model."""

from __future__ import annotations

import time

import pytest
import torch

MODELS = [
    pytest.param("stocks_model", "stocks_ckpt", id="stocks"),
    pytest.param("forex_model", "forex_ckpt", id="forex"),
]


def _random_batch(ckpt: dict, batch: int = 4) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(batch, ckpt["seq_len"], ckpt["input_size"])


@pytest.mark.parametrize("model_fix,ckpt_fix", MODELS)
def test_forward_shape(model_fix: str, ckpt_fix: str, request) -> None:
    model = request.getfixturevalue(model_fix)
    ckpt = request.getfixturevalue(ckpt_fix)
    x = _random_batch(ckpt, batch=4)
    with torch.no_grad():
        final, all_preds = model(x)
    assert final.shape == (4, 1), f"final pred shape {final.shape}"
    assert all_preds.shape == (
        4,
        ckpt["num_prediction_heads"],
    ), f"all_preds shape {all_preds.shape}"


@pytest.mark.parametrize("model_fix,ckpt_fix", MODELS)
def test_forward_finite(model_fix: str, ckpt_fix: str, request) -> None:
    model = request.getfixturevalue(model_fix)
    ckpt = request.getfixturevalue(ckpt_fix)
    x = _random_batch(ckpt, batch=8)
    with torch.no_grad():
        final, all_preds = model(x)
    assert torch.isfinite(final).all(), "final prediction has NaN/Inf"
    assert torch.isfinite(all_preds).all(), "head predictions have NaN/Inf"


@pytest.mark.parametrize("model_fix,ckpt_fix", MODELS)
def test_determinism(model_fix: str, ckpt_fix: str, request) -> None:
    model = request.getfixturevalue(model_fix)
    ckpt = request.getfixturevalue(ckpt_fix)
    x = _random_batch(ckpt)
    with torch.no_grad():
        a, _ = model(x)
        b, _ = model(x)
    assert torch.allclose(a, b, atol=1e-6), "model is non-deterministic in eval mode"


@pytest.mark.parametrize("model_fix,ckpt_fix", MODELS)
def test_predictions_not_collapsed(model_fix: str, ckpt_fix: str, request) -> None:
    """If the model emits the same value for every input it learned nothing.

    Use a varied batch (different scales + signs) and assert the std across the
    batch is non-trivial.
    """
    model = request.getfixturevalue(model_fix)
    ckpt = request.getfixturevalue(ckpt_fix)
    torch.manual_seed(1)
    base = torch.randn(16, ckpt["seq_len"], ckpt["input_size"])
    scales = torch.linspace(0.1, 5.0, 16).view(16, 1, 1)
    x = base * scales
    with torch.no_grad():
        final, _ = model(x)
    spread = final.std().item()
    assert spread > 1e-5, f"prediction std across varied batch is {spread:.2e} — model is collapsed"


@pytest.mark.parametrize("model_fix,ckpt_fix", MODELS)
def test_batch_invariance(model_fix: str, ckpt_fix: str, request) -> None:
    """Sample i in a batched call must equal the same sample run alone."""
    model = request.getfixturevalue(model_fix)
    ckpt = request.getfixturevalue(ckpt_fix)
    torch.manual_seed(2)
    x = torch.randn(4, ckpt["seq_len"], ckpt["input_size"])
    with torch.no_grad():
        batched, _ = model(x)
        single, _ = model(x[2:3])
    assert torch.allclose(
        batched[2], single[0], atol=1e-4
    ), "batched output differs from single-sample output (batch leakage?)"


@pytest.mark.parametrize("model_fix,ckpt_fix", MODELS)
def test_inference_latency(model_fix: str, ckpt_fix: str, request) -> None:
    """Single-sample inference should fit in a generous CPU budget."""
    model = request.getfixturevalue(model_fix)
    ckpt = request.getfixturevalue(ckpt_fix)
    x = torch.randn(1, ckpt["seq_len"], ckpt["input_size"])
    with torch.no_grad():
        model(x)
        start = time.perf_counter()
        for _ in range(5):
            model(x)
        elapsed = (time.perf_counter() - start) / 5
    assert elapsed < 5.0, f"single-sample inference took {elapsed:.2f}s"


@pytest.mark.parametrize("model_fix,ckpt_fix", MODELS)
def test_state_dict_loads_strictly(model_fix: str, ckpt_fix: str, request) -> None:
    """Re-build the model and load weights with strict=True to catch silent
    architecture drift between training and inference."""
    from meridianalgo.revolutionary_model import RevolutionaryFinancialModel

    ckpt = request.getfixturevalue(ckpt_fix)
    fresh = RevolutionaryFinancialModel(
        input_size=ckpt["input_size"],
        seq_len=ckpt["seq_len"],
        dim=ckpt["dim"],
        num_layers=ckpt["num_layers"],
        num_heads=ckpt["num_heads"],
        num_kv_heads=ckpt["num_kv_heads"],
        num_experts=ckpt["num_experts"],
        num_prediction_heads=ckpt["num_prediction_heads"],
        dropout=ckpt["dropout"],
        use_mamba=ckpt["use_mamba"],
        mamba_state_dim=ckpt.get("mamba_state_dim", 16),
    )
    missing, unexpected = fresh.load_state_dict(ckpt["model_state_dict"], strict=False)
    assert not missing, f"missing keys: {missing[:5]}"
    assert not unexpected, f"unexpected keys: {unexpected[:5]}"
