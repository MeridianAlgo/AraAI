"""Regression tests for the predict() denormalization path.

Old checkpoints saved with broken target stats (e.g. target_max=375 from a
single bad row) used to remap a near-zero prediction to a -99% return. The
predict path must:

  * skip remapping for raw-scale checkpoints (target_normalization="raw")
  * skip remapping for legacy checkpoints whose stored target range is
    obviously corrupted (e.g. |max| > 5.0 — 500% daily return)
  * still remap for legitimate legacy checkpoints in a reasonable range
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch


def _make_system(metadata: dict, raw_pred: float = 0.01):
    from meridianalgo.large_torch_model import AdvancedMLSystem

    sys_obj = AdvancedMLSystem.__new__(AdvancedMLSystem)
    sys_obj.metadata = metadata
    sys_obj.scaler_mean = torch.zeros(44)
    sys_obj.scaler_std = torch.ones(44)
    sys_obj.device = torch.device("cpu")

    fake_model = MagicMock()
    pred_t = torch.tensor([[raw_pred]])
    fake_model.return_value = (pred_t, pred_t)
    fake_model.eval = MagicMock()
    sys_obj.model = fake_model
    return sys_obj


def test_predict_skips_remap_for_raw_targets() -> None:
    sys_obj = _make_system(
        metadata={
            "target_normalization": "raw",
            "target_min": -0.05,
            "target_max": 0.05,
        },
        raw_pred=0.012,
    )
    X = np.zeros((1, 30, 44), dtype=np.float32)
    pred, _ = sys_obj.predict(X)
    assert float(np.asarray(pred).flatten()[0]) == pytest.approx(0.012, abs=1e-6)


def test_predict_skips_remap_for_corrupted_legacy_range() -> None:
    """Legacy checkpoint with the historical bug: target_max=375 from a single
    bad data row. predict() must NOT remap or it produces a nonsense -99%
    return for a healthy ~0 prediction."""
    sys_obj = _make_system(
        metadata={"target_min": -0.99, "target_max": 375.7},  # no normalization tag
        raw_pred=0.0,
    )
    X = np.zeros((1, 30, 44), dtype=np.float32)
    pred, _ = sys_obj.predict(X)
    out = float(np.asarray(pred).flatten()[0])
    assert abs(out) < 1.0, (
        f"corrupted-range remap produced {out:.4f}; should have been clamped"
    )


def test_predict_remaps_legitimate_legacy_range() -> None:
    sys_obj = _make_system(
        metadata={"target_min": -0.05, "target_max": 0.05},  # legacy, no tag
        raw_pred=0.5,  # midpoint of normalized [0,1] => 0% return
    )
    X = np.zeros((1, 30, 44), dtype=np.float32)
    pred, _ = sys_obj.predict(X)
    out = float(np.asarray(pred).flatten()[0])
    assert out == pytest.approx(0.0, abs=1e-6), (
        f"legitimate legacy remap broken: got {out}"
    )
