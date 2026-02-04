"""
Ensemble model implementation for ARA AI
Wraps the core MeridianAlgo unified ML system
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from ara.models.base_model import BaseModel
from meridianalgo.large_torch_model import AdvancedMLSystem


class EnhancedEnsemble(BaseModel):
    """
    Enhanced ensemble model that bridges ARA AI API to MeridianAlgo's ML system
    """

    def __init__(self, symbol: str, model_type: str = "stock", model_path: Optional[str] = None):
        self.symbol = symbol
        self.model_type = model_type

        # Default paths to unified models
        if model_path is None:
            if model_type == "stock":
                model_path = "models/stock_model.pt"
            else:
                model_path = "models/forex_model.pt"

        self.model_path = Path(model_path)
        self.ml_system = AdvancedMLSystem(self.model_path, model_type=model_type)

    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the model on given data"""
        return self.ml_system.train(X, y, self.symbol, **kwargs)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence scores"""
        pred, individuals = self.ml_system.predict(X)
        # Calculate confidence based on ensemble agreement
        if individuals.shape[1] > 1:
            confidence = 1.0 - np.std(individuals, axis=1) / (
                np.abs(np.mean(individuals, axis=1)) + 1e-6
            )
            confidence = np.clip(confidence, 0.5, 0.99)
        else:
            confidence = np.full(len(pred), 0.85)

        return pred, confidence

    def save(self, path: Path) -> None:
        """Save model is handled by ml_system internally but bridged here"""
        self.ml_system._save_model()

    def load(self, path: Path) -> None:
        """Load model is handled by ml_system internally but bridged here"""
        self.ml_system._load_model()

    def is_trained(self) -> bool:
        """Check if model is trained"""
        return self.ml_system.is_trained()
