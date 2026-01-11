"""
Adversarial Robustness Module

Provides protection against adversarial attacks on ML models through
input validation, anomaly detection, and model integrity verification.
"""

import hashlib
import json
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime


class AdversarialProtection:
    """Protects ML models from adversarial attacks"""

    def __init__(
        self, confidence_threshold: float = 0.3, anomaly_threshold: float = 3.0
    ):
        """
        Initialize adversarial protection

        Args:
            confidence_threshold: Minimum confidence for predictions
            anomaly_threshold: Z-score threshold for anomaly detection
        """
        self.confidence_threshold = confidence_threshold
        self.anomaly_threshold = anomaly_threshold

        # Statistics for anomaly detection
        self.feature_stats = {}
        self.prediction_history = []

    def validate_input(
        self, features: np.ndarray, feature_names: Optional[List[str]] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate input features for adversarial patterns

        Args:
            features: Input feature array
            feature_names: Names of features (optional)

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check for NaN or Inf values
        if np.any(np.isnan(features)):
            return False, "Input contains NaN values"

        if np.any(np.isinf(features)):
            return False, "Input contains infinite values"

        # Check for extreme values (potential adversarial perturbations)
        if len(self.feature_stats) > 0:
            is_anomaly, anomaly_features = self._detect_anomalies(features)
            if is_anomaly:
                return (
                    False,
                    f"Anomalous values detected in features: {anomaly_features}",
                )

        # Check feature value ranges
        if np.any(np.abs(features) > 1e6):
            return False, "Feature values exceed reasonable bounds"

        return True, None

    def _detect_anomalies(self, features: np.ndarray) -> Tuple[bool, List[int]]:
        """
        Detect anomalous feature values using statistical methods

        Args:
            features: Input feature array

        Returns:
            Tuple of (is_anomaly, list of anomalous feature indices)
        """
        anomalous_features = []

        for i, value in enumerate(features):
            if i in self.feature_stats:
                mean = self.feature_stats[i]["mean"]
                std = self.feature_stats[i]["std"]

                # Calculate Z-score
                if std > 0:
                    z_score = abs((value - mean) / std)
                    if z_score > self.anomaly_threshold:
                        anomalous_features.append(i)

        is_anomaly = len(anomalous_features) > 0
        return is_anomaly, anomalous_features

    def update_feature_statistics(self, features: np.ndarray) -> None:
        """
        Update running statistics for anomaly detection

        Args:
            features: Input feature array
        """
        for i, value in enumerate(features):
            if i not in self.feature_stats:
                self.feature_stats[i] = {
                    "mean": value,
                    "std": 0.0,
                    "count": 1,
                    "sum": value,
                    "sum_sq": value**2,
                }
            else:
                stats = self.feature_stats[i]
                stats["count"] += 1
                stats["sum"] += value
                stats["sum_sq"] += value**2

                # Update mean and std
                stats["mean"] = stats["sum"] / stats["count"]
                variance = (stats["sum_sq"] / stats["count"]) - (stats["mean"] ** 2)
                stats["std"] = np.sqrt(max(0, variance))

    def validate_prediction(
        self,
        prediction: float,
        confidence: float,
        current_price: Optional[float] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate prediction for suspicious patterns

        Args:
            prediction: Predicted value
            confidence: Prediction confidence (0-1)
            current_price: Current price (for sanity check)

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check confidence threshold
        if confidence < self.confidence_threshold:
            return False, f"Prediction confidence too low: {confidence:.2f}"

        # Check for extreme predictions
        if current_price is not None:
            # Prediction should not be more than 10x or less than 0.1x current price
            if prediction > current_price * 10:
                return False, "Prediction unreasonably high (>10x current price)"
            if prediction < current_price * 0.1:
                return False, "Prediction unreasonably low (<0.1x current price)"

        # Check prediction history for sudden changes
        if len(self.prediction_history) > 0:
            recent_predictions = self.prediction_history[-10:]
            avg_prediction = np.mean(recent_predictions)

            # Check for sudden large deviation
            if abs(prediction - avg_prediction) > avg_prediction * 2:
                return False, "Prediction deviates significantly from recent history"

        return True, None

    def record_prediction(self, prediction: float, confidence: float) -> None:
        """
        Record prediction for history tracking

        Args:
            prediction: Predicted value
            confidence: Prediction confidence
        """
        self.prediction_history.append(prediction)

        # Keep only recent history (last 100 predictions)
        if len(self.prediction_history) > 100:
            self.prediction_history = self.prediction_history[-100:]

    @staticmethod
    def calculate_model_checksum(model_path: Path) -> str:
        """
        Calculate checksum of model file for integrity verification

        Args:
            model_path: Path to model file

        Returns:
            SHA256 checksum
        """
        sha256_hash = hashlib.sha256()

        with open(model_path, "rb") as f:
            # Read in chunks for large files
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()

    @staticmethod
    def verify_model_integrity(model_path: Path, expected_checksum: str) -> bool:
        """
        Verify model file integrity using checksum

        Args:
            model_path: Path to model file
            expected_checksum: Expected SHA256 checksum

        Returns:
            True if checksum matches
        """
        actual_checksum = AdversarialProtection.calculate_model_checksum(model_path)
        return actual_checksum == expected_checksum

    @staticmethod
    def save_model_metadata(model_path: Path, metadata: Dict[str, Any]) -> None:
        """
        Save model metadata including checksum

        Args:
            model_path: Path to model file
            metadata: Metadata dictionary
        """
        # Calculate checksum
        checksum = AdversarialProtection.calculate_model_checksum(model_path)

        # Add checksum and timestamp to metadata
        metadata["checksum"] = checksum
        metadata["created_at"] = datetime.utcnow().isoformat()

        # Save metadata
        metadata_path = model_path.with_suffix(".meta.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @staticmethod
    def load_model_metadata(model_path: Path) -> Optional[Dict[str, Any]]:
        """
        Load model metadata

        Args:
            model_path: Path to model file

        Returns:
            Metadata dictionary or None if not found
        """
        metadata_path = model_path.with_suffix(".meta.json")

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            return json.load(f)

    @staticmethod
    def verify_model_before_load(model_path: Path) -> Tuple[bool, Optional[str]]:
        """
        Verify model integrity before loading

        Args:
            model_path: Path to model file

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if model file exists
        if not model_path.exists():
            return False, f"Model file not found: {model_path}"

        # Load metadata
        metadata = AdversarialProtection.load_model_metadata(model_path)

        if metadata is None:
            return False, "Model metadata not found"

        # Verify checksum
        expected_checksum = metadata.get("checksum")
        if not expected_checksum:
            return False, "Model checksum not found in metadata"

        if not AdversarialProtection.verify_model_integrity(
            model_path, expected_checksum
        ):
            return (
                False,
                "Model checksum verification failed - file may be corrupted or tampered",
            )

        return True, None


class ModelVersionManager:
    """Manages model versions and rollback capability"""

    def __init__(self, models_dir: Path):
        """
        Initialize model version manager

        Args:
            models_dir: Directory for storing model versions
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.versions_file = self.models_dir / "versions.json"
        self.versions = self._load_versions()

    def _load_versions(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load version history from file"""
        if self.versions_file.exists():
            with open(self.versions_file, "r") as f:
                return json.load(f)
        return {}

    def _save_versions(self) -> None:
        """Save version history to file"""
        with open(self.versions_file, "w") as f:
            json.dump(self.versions, f, indent=2)

    def register_version(
        self,
        model_name: str,
        model_path: Path,
        version: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register a new model version

        Args:
            model_name: Name of the model
            model_path: Path to model file
            version: Version string (e.g., "v1.0.0")
            metadata: Additional metadata
        """
        if model_name not in self.versions:
            self.versions[model_name] = []

        # Calculate checksum
        checksum = AdversarialProtection.calculate_model_checksum(model_path)

        version_info = {
            "version": version,
            "path": str(model_path),
            "checksum": checksum,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        }

        self.versions[model_name].append(version_info)
        self._save_versions()

    def get_version(
        self, model_name: str, version: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get specific model version (or latest if version not specified)

        Args:
            model_name: Name of the model
            version: Version string (None = latest)

        Returns:
            Version info dictionary or None
        """
        if model_name not in self.versions:
            return None

        versions = self.versions[model_name]

        if not versions:
            return None

        if version is None:
            # Return latest version
            return versions[-1]

        # Find specific version
        for v in versions:
            if v["version"] == version:
                return v

        return None

    def rollback(
        self, model_name: str, target_version: Optional[str] = None
    ) -> Optional[Path]:
        """
        Rollback to a previous model version

        Args:
            model_name: Name of the model
            target_version: Target version (None = previous version)

        Returns:
            Path to rolled back model or None
        """
        if model_name not in self.versions:
            return None

        versions = self.versions[model_name]

        if len(versions) < 2:
            return None  # No previous version to rollback to

        if target_version is None:
            # Rollback to previous version
            target_info = versions[-2]
        else:
            # Find target version
            target_info = None
            for v in versions:
                if v["version"] == target_version:
                    target_info = v
                    break

            if target_info is None:
                return None

        model_path = Path(target_info["path"])

        # Verify integrity before rollback
        is_valid, error = AdversarialProtection.verify_model_before_load(model_path)

        if not is_valid:
            raise ValueError(
                f"Cannot rollback to version {target_info['version']}: {error}"
            )

        return model_path

    def list_versions(self, model_name: str) -> List[Dict[str, Any]]:
        """
        List all versions of a model

        Args:
            model_name: Name of the model

        Returns:
            List of version info dictionaries
        """
        return self.versions.get(model_name, [])

    def delete_old_versions(self, model_name: str, keep_count: int = 5) -> int:
        """
        Delete old model versions, keeping only the most recent ones

        Args:
            model_name: Name of the model
            keep_count: Number of recent versions to keep

        Returns:
            Number of versions deleted
        """
        if model_name not in self.versions:
            return 0

        versions = self.versions[model_name]

        if len(versions) <= keep_count:
            return 0

        # Delete old versions
        versions_to_delete = versions[:-keep_count]
        deleted_count = 0

        for version_info in versions_to_delete:
            model_path = Path(version_info["path"])
            if model_path.exists():
                model_path.unlink()
                deleted_count += 1

            # Delete metadata file
            metadata_path = model_path.with_suffix(".meta.json")
            if metadata_path.exists():
                metadata_path.unlink()

        # Update versions list
        self.versions[model_name] = versions[-keep_count:]
        self._save_versions()

        return deleted_count
