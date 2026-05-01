"""Pytest fixtures for Meridian.AI model tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
STOCKS_PT = REPO_ROOT / "models_hf" / "models" / "Meridian.AI_Stocks.pt"
FOREX_PT = REPO_ROOT / "models_hf" / "models" / "Meridian.AI_Forex.pt"


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"checkpoint missing: {path}")
    return path


@pytest.fixture(scope="session")
def stocks_ckpt_path() -> Path:
    return _require(STOCKS_PT)


@pytest.fixture(scope="session")
def forex_ckpt_path() -> Path:
    return _require(FOREX_PT)


@pytest.fixture(scope="session")
def stocks_ckpt(stocks_ckpt_path: Path) -> dict:
    return torch.load(stocks_ckpt_path, map_location="cpu", weights_only=False)


@pytest.fixture(scope="session")
def forex_ckpt(forex_ckpt_path: Path) -> dict:
    return torch.load(forex_ckpt_path, map_location="cpu", weights_only=False)


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")


def _build_model(ckpt: dict) -> torch.nn.Module:
    from meridianalgo.revolutionary_model import RevolutionaryFinancialModel

    model = RevolutionaryFinancialModel(
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
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model


@pytest.fixture(scope="session")
def stocks_model(stocks_ckpt: dict):
    return _build_model(stocks_ckpt)


@pytest.fixture(scope="session")
def forex_model(forex_ckpt: dict):
    return _build_model(forex_ckpt)
