"""Ara.AI v8 — cross-sectional gradient-boosted daily stock model.

Predicts each name's next-day return *relative to the day's universe* and
ranks the universe long/short. See `ara/model.py` for why v7's absolute-return
transformer could not work, and `docs/ARA_V8.md` for the measured numbers.

    python -m ara eval  --db-file training.db --holdout-start 2025-06-01
    python -m ara train --db-file training.db --output models/ara_v8_stocks.joblib
    python -m ara predict --model-path models/ara_v8_stocks.joblib
"""

from .features import build_features, feature_cols
from .model import (
    ARCHITECTURE_NAME,
    MODEL_VERSION,
    load,
    load_panel,
    make_dataset,
    predict,
    save,
    train,
    walk_forward,
)

__version__ = MODEL_VERSION
__all__ = [
    "ARCHITECTURE_NAME",
    "MODEL_VERSION",
    "build_features",
    "feature_cols",
    "load",
    "load_panel",
    "make_dataset",
    "predict",
    "save",
    "train",
    "walk_forward",
]
