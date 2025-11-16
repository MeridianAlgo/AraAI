"""
Tests for model retraining and registry system.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from ara.models.model_registry import ModelRegistry, ModelMetadata, ModelComparison
from ara.models.retraining_scheduler import (
    ModelRetrainingScheduler,
    RetrainingConfig,
    RetrainingTrigger
)
from ara.models.ensemble import EnhancedEnsemble
from ara.models.base_model import BaseModel
from sklearn.ensemble import RandomForestRegressor


class SimpleTestModel(BaseModel):
    """Simple model for testing."""
    
    def __init__(self, model_name: str = "test_model"):
        super().__init__(model_name)
        self.model = None
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2, **kwargs):
        self.validate_input(X, y)
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        self.is_trained = True
        self.training_date = datetime.now()
        return {"status": "trained"}
    
    def predict(self, X: np.ndarray):
        if not self.is_trained:
            raise Exception("Model not trained")
        predictions = self.model.predict(X)
        confidence = np.ones_like(predictions) * 0.8
        return predictions, confidence
    
    def explain(self, X: np.ndarray):
        return {"feature_importance": {}}
    
    def save(self, path: Path) -> None:
        """Save model using BaseModel's save method."""
        super().save(path)
    
    def load(self, path: Path) -> None:
        """Load model using BaseModel's load method."""
        super().load(path)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    # Cleanup
    if temp_path.exists():
        shutil.rmtree(temp_path)


@pytest.fixture
def sample_data():
    """Create sample training data."""
    n_samples = 500
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='D')
    
    data = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
        'target': np.random.randn(n_samples) * 0.02,
        'close': 100 * np.exp(np.cumsum(np.random.randn(n_samples) * 0.02))
    }, index=dates)
    
    return data


@pytest.fixture
def sample_model(temp_dir, sample_data):
    """Create and save a sample model."""
    feature_cols = ['feature_1', 'feature_2', 'feature_3']
    X = sample_data[feature_cols].values
    y = sample_data['target'].values
    
    model = SimpleTestModel(model_name="test_model")
    model.train(X, y, validation_split=0.2)
    
    model_path = temp_dir / "test_model.pkl"
    model.save(model_path)
    
    return model_path, model


@pytest.fixture
def sample_metadata(sample_data):
    """Create sample model metadata."""
    return ModelMetadata(
        model_id="",
        model_name="test_predictor",
        version="1.0.0",
        created_at=datetime.now(),
        training_date=datetime.now(),
        accuracy=0.80,
        directional_accuracy=0.75,
        mae=0.015,
        rmse=0.022,
        sharpe_ratio=1.5,
        max_drawdown=-0.12,
        training_samples=400,
        training_period_start=sample_data.index[0],
        training_period_end=sample_data.index[-1],
        feature_count=3,
        model_type="ensemble",
        hyperparameters={"n_estimators": 3},
        data_sources=["test"],
        data_hash="test123",
        status="active",
        deployed=False,
        tags=["test"],
        notes="Test model"
    )


class TestModelRegistry:
    """Tests for ModelRegistry."""
    
    def test_registry_initialization(self, temp_dir):
        """Test registry initialization."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        
        assert registry.registry_dir.exists()
        assert registry.models_dir.exists()
        assert registry.metadata_dir.exists()
        assert registry.archive_dir.exists()
        assert registry.index_file.exists()
    
    def test_register_model(self, temp_dir, sample_model, sample_metadata):
        """Test model registration."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        model_path, _ = sample_model
        
        model_id = registry.register_model(model_path, sample_metadata)
        
        assert model_id is not None
        assert model_id in registry.index
        assert (registry.models_dir / f"{model_id}.pkl").exists()
        assert (registry.metadata_dir / f"{model_id}.json").exists()
    
    def test_get_model(self, temp_dir, sample_model, sample_metadata):
        """Test retrieving model."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        model_path, _ = sample_model
        
        model_id = registry.register_model(model_path, sample_metadata)
        
        retrieved_path, metadata = registry.get_model(model_id)
        
        assert retrieved_path.exists()
        assert metadata.model_id == model_id
        assert metadata.model_name == sample_metadata.model_name
    
    def test_list_models(self, temp_dir, sample_model, sample_metadata):
        """Test listing models."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        model_path, _ = sample_model
        
        # Register multiple models
        model_id_1 = registry.register_model(model_path, sample_metadata)
        
        sample_metadata.version = "2.0.0"
        sample_metadata.model_id = ""
        model_id_2 = registry.register_model(model_path, sample_metadata)
        
        models = registry.list_models()
        
        assert len(models) == 2
        assert any(m.model_id == model_id_1 for m in models)
        assert any(m.model_id == model_id_2 for m in models)
    
    def test_deploy_model(self, temp_dir, sample_model, sample_metadata):
        """Test model deployment."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        model_path, _ = sample_model
        
        model_id = registry.register_model(model_path, sample_metadata)
        registry.deploy_model(model_id)
        
        deployed = registry.get_deployed_model("test_predictor")
        
        assert deployed is not None
        assert deployed.model_id == model_id
        assert deployed.deployed is True
    
    def test_compare_models(self, temp_dir, sample_model, sample_metadata):
        """Test model comparison."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        model_path, _ = sample_model
        
        # Register two models with different performance
        model_id_1 = registry.register_model(model_path, sample_metadata)
        
        sample_metadata.version = "2.0.0"
        sample_metadata.model_id = ""
        sample_metadata.directional_accuracy = 0.82
        sample_metadata.sharpe_ratio = 1.8
        model_id_2 = registry.register_model(model_path, sample_metadata)
        
        comparison = registry.compare_models(model_id_1, model_id_2)
        
        assert comparison.model_a_id == model_id_1
        assert comparison.model_b_id == model_id_2
        assert comparison.accuracy_diff > 0  # Model B is better
        assert comparison.recommended_model == model_id_2
    
    def test_select_best_model(self, temp_dir, sample_model, sample_metadata):
        """Test selecting best model."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        model_path, _ = sample_model
        
        # Register models with different performance
        registry.register_model(model_path, sample_metadata)
        
        sample_metadata.version = "2.0.0"
        sample_metadata.model_id = ""
        sample_metadata.directional_accuracy = 0.85
        model_id_2 = registry.register_model(model_path, sample_metadata)
        
        best = registry.select_best_model("test_predictor", criteria="accuracy")
        
        assert best is not None
        assert best.model_id == model_id_2
        assert best.directional_accuracy == 0.85
    
    def test_archive_model(self, temp_dir, sample_model, sample_metadata):
        """Test model archival."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        model_path, _ = sample_model
        
        model_id = registry.register_model(model_path, sample_metadata)
        registry.archive_model(model_id)
        
        metadata = registry.get_metadata(model_id)
        
        assert metadata.status == "archived"
        assert (registry.archive_dir / f"{model_id}.pkl").exists()
    
    def test_cleanup_old_models(self, temp_dir, sample_model, sample_metadata):
        """Test cleaning up old models."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        model_path, _ = sample_model
        
        # Register multiple versions
        for i in range(5):
            sample_metadata.version = f"{i+1}.0.0"
            sample_metadata.model_id = ""
            registry.register_model(model_path, sample_metadata)
        
        archived = registry.cleanup_old_models("test_predictor", keep_latest=2)
        
        assert len(archived) == 3
        active_models = registry.list_models(model_name="test_predictor", status="active")
        assert len(active_models) == 2


class TestModelRetrainingScheduler:
    """Tests for ModelRetrainingScheduler."""
    
    def test_scheduler_initialization(self, temp_dir):
        """Test scheduler initialization."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        config = RetrainingConfig()
        
        scheduler = ModelRetrainingScheduler(
            registry=registry,
            config=config,
            output_dir=temp_dir / "retraining"
        )
        
        assert scheduler.registry == registry
        assert scheduler.config == config
        assert scheduler.output_dir.exists()
    
    def test_check_retraining_needed_accuracy(self, temp_dir, sample_model, sample_metadata):
        """Test retraining check based on accuracy."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        model_path, _ = sample_model
        
        # Register and deploy model
        model_id = registry.register_model(model_path, sample_metadata)
        registry.deploy_model(model_id)
        
        config = RetrainingConfig(accuracy_threshold=0.80)
        scheduler = ModelRetrainingScheduler(registry, config)
        
        # Simulate poor predictions
        n_recent = 30
        recent_predictions = np.random.randn(n_recent) * 0.02
        recent_actuals = -recent_predictions  # Opposite direction
        recent_dates = [datetime.now() - timedelta(days=i) for i in range(n_recent, 0, -1)]
        
        needs_retraining, trigger, reason = scheduler.check_retraining_needed(
            model_name="test_predictor",
            recent_predictions=recent_predictions,
            recent_actuals=recent_actuals,
            recent_dates=recent_dates
        )
        
        assert needs_retraining is True
        assert trigger in [RetrainingTrigger.ACCURACY_DROP, RetrainingTrigger.DEGRADATION]
    
    def test_retrain_model(self, temp_dir, sample_model, sample_metadata, sample_data):
        """Test model retraining."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        model_path, _ = sample_model
        
        # Register initial model
        registry.register_model(model_path, sample_metadata)
        
        # Use lower rollback threshold for test
        config = RetrainingConfig(rollback_accuracy_threshold=0.40)
        scheduler = ModelRetrainingScheduler(
            registry,
            config,
            output_dir=temp_dir / "retraining"
        )
        
        # Define training function
        def train_fn(X, y):
            model = SimpleTestModel(model_name="test_model")
            model.train(X, y, validation_split=0.2)
            return model
        
        # Retrain model
        feature_cols = ['feature_1', 'feature_2', 'feature_3']
        result = scheduler.retrain_model(
            model_name="test_predictor",
            train_fn=train_fn,
            training_data=sample_data,
            feature_columns=feature_cols,
            target_column='target',
            trigger=RetrainingTrigger.MANUAL,
            model_type="ensemble"
        )
        
        assert result.success is True
        assert result.new_model_id != ""
        assert result.new_version != ""
    
    def test_rollback_model(self, temp_dir, sample_model, sample_metadata):
        """Test model rollback."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        model_path, _ = sample_model
        
        # Register two versions
        model_id_1 = registry.register_model(model_path, sample_metadata)
        registry.deploy_model(model_id_1)
        
        sample_metadata.version = "2.0.0"
        sample_metadata.model_id = ""
        model_id_2 = registry.register_model(model_path, sample_metadata)
        registry.deploy_model(model_id_2)
        
        config = RetrainingConfig()
        scheduler = ModelRetrainingScheduler(registry, config)
        
        # Rollback
        success = scheduler.rollback_model("test_predictor")
        
        assert success is True
        
        deployed = registry.get_deployed_model("test_predictor")
        assert deployed.model_id == model_id_1
    
    def test_retraining_statistics(self, temp_dir, sample_model, sample_metadata, sample_data):
        """Test retraining statistics."""
        registry = ModelRegistry(registry_dir=temp_dir / "registry")
        model_path, _ = sample_model
        
        registry.register_model(model_path, sample_metadata)
        
        # Use lower rollback threshold for test
        config = RetrainingConfig(rollback_accuracy_threshold=0.40)
        scheduler = ModelRetrainingScheduler(
            registry,
            config,
            output_dir=temp_dir / "retraining"
        )
        
        # Perform retraining
        def train_fn(X, y):
            model = SimpleTestModel(model_name="test_model")
            model.train(X, y, validation_split=0.2)
            return model
        
        feature_cols = ['feature_1', 'feature_2', 'feature_3']
        scheduler.retrain_model(
            model_name="test_predictor",
            train_fn=train_fn,
            training_data=sample_data,
            feature_columns=feature_cols,
            target_column='target',
            trigger=RetrainingTrigger.MANUAL
        )
        
        # Get statistics
        stats = scheduler.get_retraining_statistics("test_predictor")
        
        assert 'total_retrainings' in stats
        assert stats['total_retrainings'] >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
