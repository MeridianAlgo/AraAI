"""
Base model interface for ARA AI
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple
import numpy as np


class BaseModel(ABC):
    """Abstract base class for all prediction models"""

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict[str, Any]:
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions"""
        pass

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk"""
        pass

    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk"""
        pass

    @abstractmethod
    def is_trained(self) -> bool:
        """Check if model is trained"""
        pass
