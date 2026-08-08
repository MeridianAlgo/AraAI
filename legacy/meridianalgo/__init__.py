"""
Meridian.AI - Real-time financial prediction engine.

A compact transformer (MeridianModel: GQA + MoE + optional Mamba SSM) that
forecasts next-day price movement and a direction signal for stocks and forex
from 44 technical indicators. Checkpoints retrain hourly via GitHub Actions and
publish to Hugging Face (meridianal/ARA.AI).
"""

# Product / package release version. This is distinct from the checkpoint
# architecture version (MODEL_VERSION / _MIN_LOADABLE in large_torch_model.py),
# which gates loadable checkpoint formats (7.x since v1.2.0). v1.0.0 was the
# first production release; v1.2.0 ships the v7 checkpoint format.
__version__ = "1.2.1"
__author__ = "MeridianAlgo Team"
__email__ = "support@meridianalgo.com"
__license__ = "MIT"

# Delayed imports are recommended to avoid pulling optional/legacy modules at import time

# Version info
VERSION_INFO = {
    "version": __version__,
    "features": [
        "MeridianModel transformer (Grouped Query Attention + Mixture of Experts)",
        "Next-day price forecast and direction signal in one forward pass",
        "44 scale-invariant technical indicators over a 30-step window",
        "Stock and forex prediction APIs (forex trained with a 1-day embargo)",
        "Hourly retraining via GitHub Actions, checkpoints published to Hugging Face",
        "CPU-first; optional GPU acceleration (NVIDIA/AMD/Intel/Apple)",
        "Sanity gate that blocks degenerate checkpoints from publishing",
    ],
    "gpu_support": ["NVIDIA CUDA", "AMD ROCm/DirectML", "Intel XPU", "Apple MPS"],
    "python_versions": ["3.9+", "3.10+", "3.11+", "3.12+"],
}


def get_version_info():
    return VERSION_INFO


def check_gpu_support():
    try:
        from .utils import GPUManager

        return GPUManager().detect_gpu_vendor()
    except Exception:
        return "unknown"


def analyze_accuracy(symbol=None):
    try:
        from .utils import AccuracyTracker

        tracker = AccuracyTracker()
        return tracker.analyze_accuracy(symbol)
    except Exception:
        return {"error": "accuracy tracker unavailable"}


# Package metadata
__all__ = [
    "__version__",
    "VERSION_INFO",
    "get_version_info",
    "check_gpu_support",
    "analyze_accuracy",
]
