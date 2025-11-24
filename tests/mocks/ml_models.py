"""
Mock ML models for testing.

These mocks provide fast, deterministic predictions without
requiring actual model training or inference.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from ara.core.interfaces import IMLModel


class MockMLModel(IMLModel):
    """Mock ML model for fast testing."""
    
    def __init__(self, accuracy: float = 0.85, prediction_bias: float = 0.0):
        self.accuracy = accuracy
        self.prediction_bias = prediction_bias
        self._is_trained = False
        self._training_history = []
        self._prediction_count = 0
        
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "name": "MockModel",
            "type": "mock",
            "accuracy": self.accuracy,
            "is_trained": self._is_trained
        }
        
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """Mock training."""
        self._is_trained = True
        self._training_history.append({
            "samples": len(X),
            "features": X.shape[1] if len(X.shape) > 1 else 1,
            "accuracy": self.accuracy
        })
        
        return {
            "loss": 0.1,
            "accuracy": self.accuracy,
            "epochs": kwargs.get("epochs", 10)
        }
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Mock predictions with confidence scores."""
        self._prediction_count += 1
        
        n_samples = len(X)
        
        # Generate deterministic predictions based on input
        predictions = np.mean(X, axis=1) + self.prediction_bias
        
        # Generate confidence scores
        confidences = np.full(n_samples, self.accuracy)
        
        return predictions, confidences
        
    def explain(self, X: np.ndarray) -> Dict:
        """Mock explanations."""
        n_features = X.shape[1] if len(X.shape) > 1 else 1
        
        return {
            "feature_importance": {
                f"feature_{i}": 1.0 / n_features
                for i in range(n_features)
            },
            "top_features": [f"feature_{i}" for i in range(min(5, n_features))]
        }
        
    def save(self, path: Path) -> None:
        """Mock save."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"MockModel(accuracy={self.accuracy})")
        
    def load(self, path: Path) -> None:
        """Mock load."""
        self._is_trained = True
        
    def is_trained(self) -> bool:
        """Check if model is trained."""
        return self._is_trained
        
    def get_prediction_count(self) -> int:
        """Get number of predictions made."""
        return self._prediction_count


class MockTransformer(MockMLModel):
    """Mock Transformer model."""
    
    def __init__(self, accuracy: float = 0.88, num_heads: int = 8):
        super().__init__(accuracy)
        self.num_heads = num_heads
        self._attention_weights = None
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Mock predictions with attention weights."""
        predictions, confidences = super().predict(X)
        
        # Generate mock attention weights
        seq_len = X.shape[0] if len(X.shape) > 1 else 1
        self._attention_weights = np.random.rand(self.num_heads, seq_len, seq_len)
        
        return predictions, confidences
        
    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Get attention weights from last prediction."""
        return self._attention_weights
        
    def explain(self, X: np.ndarray) -> Dict:
        """Mock explanations with attention."""
        base_explanation = super().explain(X)
        base_explanation["attention_weights"] = self._attention_weights
        return base_explanation


class MockEnsemble(IMLModel):
    """Mock ensemble model."""
    
    def __init__(self, num_models: int = 5, accuracy: float = 0.90):
        self.num_models = num_models
        self.accuracy = accuracy
        self._models = [
            MockMLModel(accuracy=accuracy - 0.05 + i * 0.02)
            for i in range(num_models)
        ]
        self._is_trained = False
        
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "name": "MockEnsemble",
            "type": "ensemble",
            "num_models": self.num_models,
            "accuracy": self.accuracy,
            "is_trained": self._is_trained
        }
        
    def train(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """Train all models in ensemble."""
        results = []
        for model in self._models:
            result = model.train(X, y, **kwargs)
            results.append(result)
            
        self._is_trained = True
        
        return {
            "models_trained": len(self._models),
            "average_accuracy": np.mean([r["accuracy"] for r in results]),
            "individual_results": results
        }
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ensemble predictions."""
        all_predictions = []
        all_confidences = []
        
        for model in self._models:
            preds, confs = model.predict(X)
            all_predictions.append(preds)
            all_confidences.append(confs)
            
        # Average predictions
        predictions = np.mean(all_predictions, axis=0)
        
        # Confidence based on model agreement
        confidences = np.mean(all_confidences, axis=0)
        
        return predictions, confidences
        
    def explain(self, X: np.ndarray) -> Dict:
        """Aggregate explanations from all models."""
        explanations = [model.explain(X) for model in self._models]
        
        # Aggregate feature importance
        all_features = set()
        for exp in explanations:
            all_features.update(exp["feature_importance"].keys())
            
        aggregated_importance = {}
        for feature in all_features:
            importances = [
                exp["feature_importance"].get(feature, 0.0)
                for exp in explanations
            ]
            aggregated_importance[feature] = np.mean(importances)
            
        return {
            "feature_importance": aggregated_importance,
            "model_count": len(self._models),
            "individual_explanations": explanations
        }
        
    def save(self, path: Path) -> None:
        """Save ensemble."""
        path.parent.mkdir(parents=True, exist_ok=True)
        for i, model in enumerate(self._models):
            model.save(path.parent / f"{path.stem}_model_{i}{path.suffix}")
            
    def load(self, path: Path) -> None:
        """Load ensemble."""
        for i, model in enumerate(self._models):
            model_path = path.parent / f"{path.stem}_model_{i}{path.suffix}"
            if model_path.exists():
                model.load(model_path)
        self._is_trained = True


class FastMLModel(MockMLModel):
    """Very fast mock model for performance testing."""
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Ultra-fast predictions."""
        n_samples = len(X)
        predictions = np.ones(n_samples)
        confidences = np.full(n_samples, self.accuracy)
        return predictions, confidences


class UntrainedModel(MockMLModel):
    """Model that hasn't been trained - for testing error handling."""
    
    def __init__(self):
        super().__init__()
        self._is_trained = False
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Raise error if not trained."""
        if not self._is_trained:
            raise RuntimeError("Model must be trained before prediction")
        return super().predict(X)
