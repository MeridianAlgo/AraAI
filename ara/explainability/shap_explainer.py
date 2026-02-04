"""
SHAP (SHapley Additive exPlanations) Explainer
Provides model-agnostic explanations using SHAP values
"""

import numpy as np
from typing import Dict, List, Any, Optional
import warnings

try:
    import shap

    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")

from ara.core.exceptions import ModelError, ValidationError


class SHAPExplainer:
    """
    SHAP-based explainer for ML models
    Calculates SHAP values to explain individual predictions
    """

    def __init__(
        self,
        model: Any,
        background_data: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
        explainer_type: str = "auto",
    ):
        """
        Initialize SHAP explainer

        Args:
            model: Trained ML model (must have predict method)
            background_data: Background dataset for SHAP (optional)
            feature_names: Names of features (optional)
            explainer_type: Type of SHAP explainer ('auto', 'tree', 'kernel', 'deep', 'linear')
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP library not available. Install with: pip install shap")

        self.model = model
        self.background_data = background_data
        self.feature_names = feature_names
        self.explainer_type = explainer_type
        self.explainer = None

        # Initialize explainer
        self._initialize_explainer()

    def _initialize_explainer(self):
        """Initialize the appropriate SHAP explainer"""
        if self.explainer_type == "auto":
            # Try to detect the best explainer type
            self.explainer = self._create_auto_explainer()
        elif self.explainer_type == "tree":
            self.explainer = shap.TreeExplainer(self.model)
        elif self.explainer_type == "kernel":
            if self.background_data is None:
                raise ValidationError(
                    "background_data required for KernelExplainer",
                    {"explainer_type": "kernel"},
                )
            self.explainer = shap.KernelExplainer(
                self.model.predict if hasattr(self.model, "predict") else self.model,
                self.background_data,
            )
        elif self.explainer_type == "linear":
            self.explainer = shap.LinearExplainer(self.model, self.background_data)
        elif self.explainer_type == "deep":
            self.explainer = shap.DeepExplainer(self.model, self.background_data)
        else:
            raise ValidationError(
                f"Unknown explainer type: {self.explainer_type}",
                {"valid_types": ["auto", "tree", "kernel", "deep", "linear"]},
            )

    def _create_auto_explainer(self):
        """Automatically select the best explainer type"""
        # Check if it's a tree-based model
        tree_models = [
            "XGBRegressor",
            "LGBMRegressor",
            "CatBoostRegressor",
            "RandomForestRegressor",
            "ExtraTreesRegressor",
            "GradientBoostingRegressor",
        ]

        model_name = type(self.model).__name__

        if any(tree_model in model_name for tree_model in tree_models):
            try:
                return shap.TreeExplainer(self.model)
            except:
                pass

        # Check if it's a linear model
        linear_models = ["Ridge", "Lasso", "ElasticNet", "LinearRegression"]
        if any(linear_model in model_name for linear_model in linear_models):
            try:
                return shap.LinearExplainer(self.model, self.background_data)
            except:
                pass

        # Fall back to KernelExplainer (model-agnostic but slower)
        if self.background_data is not None:
            return shap.KernelExplainer(
                self.model.predict if hasattr(self.model, "predict") else self.model,
                self.background_data,
            )
        else:
            raise ModelError(
                "Cannot create auto explainer without background_data. "
                "Please provide background_data or specify explainer_type."
            )

    def explain(self, X: np.ndarray, check_additivity: bool = False) -> Dict[str, Any]:
        """
        Calculate SHAP values for predictions

        Args:
            X: Feature matrix to explain (n_samples, n_features)
            check_additivity: Whether to check SHAP value additivity

        Returns:
            Dict containing SHAP values and related information
        """
        if self.explainer is None:
            raise ModelError("Explainer not initialized")

        # Calculate SHAP values
        try:
            shap_values = self.explainer.shap_values(X, check_additivity=check_additivity)
        except TypeError:
            # Some explainers don't support check_additivity
            shap_values = self.explainer.shap_values(X)

        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # Multi-output model - take first output
            shap_values = shap_values[0]

        # Get base value (expected value)
        if hasattr(self.explainer, "expected_value"):
            base_value = self.explainer.expected_value
            if isinstance(base_value, (list, np.ndarray)):
                base_value = base_value[0] if len(base_value) > 0 else 0.0
        else:
            base_value = 0.0

        return {
            "shap_values": shap_values,
            "base_value": float(base_value),
            "feature_names": self.feature_names,
            "data": X,
        }

    def get_feature_importance(self, X: np.ndarray, method: str = "mean_abs") -> Dict[str, float]:
        """
        Calculate feature importance from SHAP values

        Args:
            X: Feature matrix
            method: Method for aggregating SHAP values ('mean_abs', 'mean', 'max')

        Returns:
            Dict mapping feature names to importance scores
        """
        explanation = self.explain(X)
        shap_values = explanation["shap_values"]

        # Aggregate SHAP values
        if method == "mean_abs":
            importance = np.mean(np.abs(shap_values), axis=0)
        elif method == "mean":
            importance = np.mean(shap_values, axis=0)
        elif method == "max":
            importance = np.max(np.abs(shap_values), axis=0)
        else:
            raise ValidationError(
                f"Unknown aggregation method: {method}",
                {"valid_methods": ["mean_abs", "mean", "max"]},
            )

        # Create feature importance dict
        if self.feature_names:
            feature_importance = {
                name: float(imp) for name, imp in zip(self.feature_names, importance)
            }
        else:
            feature_importance = {f"feature_{i}": float(imp) for i, imp in enumerate(importance)}

        # Sort by importance
        feature_importance = dict(
            sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        )

        return feature_importance

    def get_top_features(
        self, X: np.ndarray, n: int = 5, sample_idx: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Get top N contributing features for a specific prediction

        Args:
            X: Feature matrix
            n: Number of top features to return
            sample_idx: Index of sample to explain

        Returns:
            List of dicts with feature name, value, and SHAP value
        """
        explanation = self.explain(X)
        shap_values = explanation["shap_values"]

        if sample_idx >= len(shap_values):
            raise ValidationError(
                f"sample_idx {sample_idx} out of range",
                {"max_idx": len(shap_values) - 1},
            )

        # Get SHAP values for this sample
        sample_shap = shap_values[sample_idx]
        sample_data = X[sample_idx]

        # Get indices of top features by absolute SHAP value
        top_indices = np.argsort(np.abs(sample_shap))[-n:][::-1]

        # Build result
        top_features = []
        for idx in top_indices:
            feature_name = self.feature_names[idx] if self.feature_names else f"feature_{idx}"

            top_features.append(
                {
                    "feature_name": feature_name,
                    "feature_value": float(sample_data[idx]),
                    "shap_value": float(sample_shap[idx]),
                    "contribution": float(sample_shap[idx]),
                    "abs_contribution": float(abs(sample_shap[idx])),
                }
            )

        return top_features

    def explain_prediction(
        self, X: np.ndarray, sample_idx: int = 0, n_features: int = 5
    ) -> Dict[str, Any]:
        """
        Comprehensive explanation for a single prediction

        Args:
            X: Feature matrix
            sample_idx: Index of sample to explain
            n_features: Number of top features to include

        Returns:
            Dict with comprehensive explanation
        """
        explanation = self.explain(X)
        top_features = self.get_top_features(X, n=n_features, sample_idx=sample_idx)

        # Calculate prediction
        if hasattr(self.model, "predict"):
            prediction = self.model.predict(X[sample_idx : sample_idx + 1])[0]
        else:
            prediction = None

        # Calculate total contribution
        shap_values = explanation["shap_values"][sample_idx]
        total_contribution = np.sum(shap_values)
        base_value = explanation["base_value"]

        return {
            "sample_idx": sample_idx,
            "prediction": float(prediction) if prediction is not None else None,
            "base_value": base_value,
            "total_shap_contribution": float(total_contribution),
            "expected_prediction": float(base_value + total_contribution),
            "top_features": top_features,
            "n_features_total": len(shap_values),
        }

    def get_interaction_values(
        self, X: np.ndarray, feature_idx1: int, feature_idx2: int
    ) -> Optional[np.ndarray]:
        """
        Calculate SHAP interaction values between two features

        Args:
            X: Feature matrix
            feature_idx1: Index of first feature
            feature_idx2: Index of second feature

        Returns:
            Interaction values or None if not supported
        """
        # Only TreeExplainer supports interaction values
        if not isinstance(self.explainer, shap.TreeExplainer):
            warnings.warn("Interaction values only supported for TreeExplainer", UserWarning)
            return None

        try:
            interaction_values = self.explainer.shap_interaction_values(X)
            return interaction_values[:, feature_idx1, feature_idx2]
        except Exception as e:
            warnings.warn(f"Failed to calculate interaction values: {e}", UserWarning)
            return None
