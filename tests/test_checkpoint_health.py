"""Health checks on the saved checkpoint metadata.

These catch silent training failures (Inf loss, zero direction accuracy,
extreme outliers in target distribution) BEFORE we trust the model's
predictions for anything.
"""

from __future__ import annotations

import math

import pytest

CKPTS = [
    pytest.param("stocks_ckpt", id="stocks"),
    pytest.param("forex_ckpt", id="forex"),
]


@pytest.mark.parametrize("ckpt_fix", CKPTS)
def test_checkpoint_has_required_keys(ckpt_fix: str, request: pytest.FixtureRequest) -> None:
    ckpt = request.getfixturevalue(ckpt_fix)
    required = {
        "model_state_dict",
        "model_type",
        "scaler_mean",
        "scaler_std",
        "metadata",
        "architecture",
        "version",
        "input_size",
        "seq_len",
        "dim",
        "num_layers",
        "num_heads",
        "num_kv_heads",
        "num_experts",
        "num_prediction_heads",
        "dropout",
        "use_mamba",
    }
    missing = required - set(ckpt.keys())
    assert not missing, f"checkpoint missing keys: {missing}"


@pytest.mark.parametrize("ckpt_fix", CKPTS)
def test_architecture_matches_readme(ckpt_fix: str, request: pytest.FixtureRequest) -> None:
    ckpt = request.getfixturevalue(ckpt_fix)
    # Fixed: tied to the 44-feature pipeline and 30-day lookback window
    assert ckpt["input_size"] == 44, "feature count must be 44"
    assert ckpt["seq_len"] == 30, "lookback window must be 30"
    assert ckpt["num_kv_heads"] == 2

    # Flexible: may change across training runs / architecture versions
    assert 64 <= ckpt["dim"] <= 2048, f"dim={ckpt['dim']} out of expected range"
    assert 1 <= ckpt["num_layers"] <= 24
    assert 1 <= ckpt["num_heads"] <= 32
    assert ckpt["num_experts"] >= 1
    assert ckpt["num_prediction_heads"] >= 1

    # Accept v4.1 (old HuggingFace models) and v5.0+ (new MeridianModel)
    assert str(ckpt.get("version", "0")) >= "4.1", f"model version {ckpt.get('version')} too old"
    arch = ckpt.get("architecture", "")
    assert arch in (
        "MeridianModel-2026",
        "RevolutionaryFinancialModel-2026",
    ), f"unknown architecture: {arch}"


@pytest.mark.parametrize("ckpt_fix", CKPTS)
def test_validation_loss_is_finite(ckpt_fix: str, request: pytest.FixtureRequest) -> None:
    """A trained model must report a finite best validation loss."""
    ckpt = request.getfixturevalue(ckpt_fix)
    md = ckpt["metadata"]
    best = md.get("best_val_loss")
    assert best is not None, "metadata.best_val_loss missing"
    assert math.isfinite(
        float(best)
    ), f"best_val_loss is {best} - training diverged or never recorded a real loss"


@pytest.mark.parametrize("ckpt_fix", CKPTS)
def test_training_history_has_finite_losses(ckpt_fix: str, request: pytest.FixtureRequest) -> None:
    ckpt = request.getfixturevalue(ckpt_fix)
    history = ckpt["metadata"].get("training_history", [])
    assert history, "training_history empty"
    bad = [
        i
        for i, h in enumerate(history)
        if not math.isfinite(float(h.get("val_loss", float("inf"))))
    ]
    assert not bad, (
        f"{len(bad)}/{len(history)} training runs recorded non-finite val_loss "
        f"(first bad index: {bad[0]})"
    )


@pytest.mark.parametrize("ckpt_fix", CKPTS)
def test_direction_accuracy_above_chance(ckpt_fix: str, request: pytest.FixtureRequest) -> None:
    """If the model can't beat 50/50 on the validation set, it learned nothing.

    `calculate_direction_metrics` reports accuracy as a percent (0..100),
    so the chance threshold is 50.0, not 0.50.
    """
    ckpt = request.getfixturevalue(ckpt_fix)
    acc = float(ckpt["metadata"].get("direction_accuracy", 0))
    assert acc >= 50.0, f"direction_accuracy={acc:.2f}% - model did not beat random chance"


@pytest.mark.parametrize(
    "ckpt_fix,limit",
    [
        pytest.param("stocks_ckpt", 1.0, id="stocks"),
        pytest.param("forex_ckpt", 0.20, id="forex"),
    ],
)
def test_target_range_sane(ckpt_fix: str, limit: float, request: pytest.FixtureRequest) -> None:
    """Daily returns above `limit` (e.g. 100% for stocks, 20% for forex) are
    almost certainly bad data, not real moves. Outliers this size will dominate
    MSE and break the loss surface.
    """
    ckpt = request.getfixturevalue(ckpt_fix)
    md = ckpt["metadata"]
    tmin = float(md["target_min"])
    tmax = float(md["target_max"])
    assert abs(tmin) <= limit, f"target_min={tmin} exceeds sanity limit {limit}"
    assert abs(tmax) <= limit, f"target_max={tmax} exceeds sanity limit {limit}"


@pytest.mark.parametrize("ckpt_fix", CKPTS)
def test_scaler_stats_finite(ckpt_fix: str, request: pytest.FixtureRequest) -> None:
    import torch

    ckpt = request.getfixturevalue(ckpt_fix)
    mean = ckpt["scaler_mean"]
    std = ckpt["scaler_std"]
    assert torch.isfinite(mean).all(), "scaler_mean has NaN/Inf"
    assert torch.isfinite(std).all(), "scaler_std has NaN/Inf"
    assert (std > 0).all(), "scaler_std has zero or negative entries (will divide-by-zero)"
