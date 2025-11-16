"""
Model Retraining and Registry Demo

This example demonstrates:
1. Model registry and versioning
2. Automated retraining triggers
3. Model comparison and selection
4. Deployment workflow
5. Rollback capabilities
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

from ara.models.model_registry import ModelRegistry, ModelMetadata
from ara.models.retraining_scheduler import (
    ModelRetrainingScheduler,
    RetrainingConfig,
    RetrainingTrigger
)
from ara.models.ensemble import EnhancedEnsemble


def create_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample training data."""
    dates = pd.date_range(end=datetime.now(), periods=n_samples, freq='D')
    
    # Generate synthetic features
    data = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.randn(n_samples),
        'feature_4': np.random.randn(n_samples),
        'feature_5': np.random.randn(n_samples),
        'target': np.random.randn(n_samples) * 0.02,  # 2% daily returns
        'close': 100 * np.exp(np.cumsum(np.random.randn(n_samples) * 0.02))
    }, index=dates)
    
    return data


def train_model_fn(X: np.ndarray, y: np.ndarray) -> EnhancedEnsemble:
    """Training function for ensemble model."""
    print(f"Training model with {len(X)} samples...")
    
    model = EnhancedEnsemble(model_name="demo_ensemble")
    
    model.train(X, y, validation_split=0.2)
    
    return model


def demo_model_registry():
    """Demonstrate model registry functionality."""
    print("\n" + "="*60)
    print("DEMO 1: Model Registry and Versioning")
    print("="*60)
    
    # Initialize registry
    registry = ModelRegistry(registry_dir=Path("models/demo_registry"))
    
    # Create sample training data
    data = create_sample_data(500)
    feature_cols = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    
    # Train a model
    X = data[feature_cols].values
    y = data['target'].values
    
    model = train_model_fn(X, y)
    
    # Save model
    model_path = Path("models/demo_model_v1.pkl")
    model.save(model_path)
    
    # Create metadata
    metadata = ModelMetadata(
        model_id="",
        model_name="demo_predictor",
        version="1.0.0",
        created_at=datetime.now(),
        training_date=datetime.now(),
        accuracy=0.82,
        directional_accuracy=0.78,
        mae=0.015,
        rmse=0.022,
        sharpe_ratio=1.5,
        max_drawdown=-0.12,
        training_samples=len(X),
        training_period_start=data.index[0],
        training_period_end=data.index[-1],
        feature_count=len(feature_cols),
        model_type="ensemble",
        hyperparameters={},
        data_sources=["synthetic"],
        data_hash="abc123",
        status="active",
        deployed=False,
        tags=["demo", "v1"],
        notes="Initial model version"
    )
    
    # Register model
    model_id = registry.register_model(model_path, metadata)
    print(f"\n✓ Registered model: {model_id}")
    
    # Deploy model
    registry.deploy_model(model_id)
    print(f"✓ Deployed model: {model_id}")
    
    # List models
    models = registry.list_models()
    print(f"\n✓ Total models in registry: {len(models)}")
    
    for m in models:
        print(f"  - {m.model_name} v{m.version}: "
              f"Accuracy={m.directional_accuracy:.2%}, "
              f"Deployed={m.deployed}")
    
    # Get statistics
    stats = registry.get_statistics()
    print(f"\n✓ Registry Statistics:")
    print(f"  - Total models: {stats['total_models']}")
    print(f"  - Deployed models: {stats['deployed_models']}")
    print(f"  - Status counts: {stats['status_counts']}")
    
    return registry, model_id


def demo_model_comparison(registry: ModelRegistry, model_a_id: str):
    """Demonstrate model comparison."""
    print("\n" + "="*60)
    print("DEMO 2: Model Comparison")
    print("="*60)
    
    # Train a second model with different performance
    data = create_sample_data(500)
    feature_cols = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
    
    X = data[feature_cols].values
    y = data['target'].values
    
    model = train_model_fn(X, y)
    
    # Save model
    model_path = Path("models/demo_model_v2.pkl")
    model.save(model_path)
    
    # Create metadata with better performance
    metadata = ModelMetadata(
        model_id="",
        model_name="demo_predictor",
        version="2.0.0",
        created_at=datetime.now(),
        training_date=datetime.now(),
        accuracy=0.85,
        directional_accuracy=0.82,
        mae=0.012,
        rmse=0.018,
        sharpe_ratio=1.8,
        max_drawdown=-0.10,
        training_samples=len(X),
        training_period_start=data.index[0],
        training_period_end=data.index[-1],
        feature_count=len(feature_cols),
        model_type="ensemble",
        hyperparameters={},
        data_sources=["synthetic"],
        data_hash="def456",
        status="testing",
        deployed=False,
        tags=["demo", "v2"],
        notes="Improved model version"
    )
    
    # Register second model
    model_b_id = registry.register_model(model_path, metadata)
    print(f"\n✓ Registered second model: {model_b_id}")
    
    # Compare models
    comparison = registry.compare_models(model_a_id, model_b_id)
    
    print(f"\n✓ Model Comparison Results:")
    print(f"  - Model A: {model_a_id}")
    print(f"  - Model B: {model_b_id}")
    print(f"  - Accuracy difference: {comparison.accuracy_diff:+.2%}")
    print(f"  - Sharpe difference: {comparison.sharpe_diff:+.2f}")
    print(f"  - Drawdown difference: {comparison.drawdown_diff:+.2%}")
    print(f"  - Recommended: {comparison.recommended_model}")
    print(f"  - Confidence: {comparison.confidence:.2%}")
    print(f"\n  Reasoning:")
    for reason in comparison.reasoning:
        print(f"    • {reason}")
    
    # Select best model
    best_model = registry.select_best_model("demo_predictor", criteria="overall")
    print(f"\n✓ Best model (overall): {best_model.model_id} v{best_model.version}")
    
    return model_b_id


def demo_automated_retraining(registry: ModelRegistry):
    """Demonstrate automated retraining."""
    print("\n" + "="*60)
    print("DEMO 3: Automated Retraining")
    print("="*60)
    
    # Initialize retraining scheduler
    config = RetrainingConfig(
        accuracy_threshold=0.75,
        degradation_threshold=0.10,
        periodic_retraining_days=90,
        min_days_between_retraining=7,
        enable_auto_rollback=True,
        rollback_accuracy_threshold=0.70,
        max_versions_to_keep=3
    )
    
    scheduler = ModelRetrainingScheduler(
        registry=registry,
        config=config,
        output_dir=Path("retraining_logs")
    )
    
    print(f"\n✓ Initialized retraining scheduler")
    print(f"  - Accuracy threshold: {config.accuracy_threshold:.2%}")
    print(f"  - Degradation threshold: {config.degradation_threshold:.2%}")
    print(f"  - Periodic retraining: {config.periodic_retraining_days} days")
    
    # Simulate recent predictions with poor performance
    n_recent = 30
    recent_predictions = np.random.randn(n_recent) * 0.02
    recent_actuals = np.random.randn(n_recent) * 0.02
    recent_dates = [datetime.now() - timedelta(days=i) for i in range(n_recent, 0, -1)]
    
    # Check if retraining is needed
    needs_retraining, trigger, reason = scheduler.check_retraining_needed(
        model_name="demo_predictor",
        recent_predictions=recent_predictions,
        recent_actuals=recent_actuals,
        recent_dates=recent_dates
    )
    
    print(f"\n✓ Retraining Check:")
    print(f"  - Needs retraining: {needs_retraining}")
    print(f"  - Trigger: {trigger.value if trigger else 'None'}")
    print(f"  - Reason: {reason}")
    
    # Perform retraining
    if needs_retraining or True:  # Force retraining for demo
        print(f"\n✓ Starting retraining...")
        
        # Create training data
        training_data = create_sample_data(1000)
        feature_cols = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        
        # Retrain model
        result = scheduler.retrain_model(
            model_name="demo_predictor",
            train_fn=train_model_fn,
            training_data=training_data,
            feature_columns=feature_cols,
            target_column='target',
            trigger=trigger or RetrainingTrigger.MANUAL,
            model_type="ensemble",
            hyperparameters={}
        )
        
        print(f"\n✓ Retraining Result:")
        print(f"  - Success: {result.success}")
        print(f"  - New model ID: {result.new_model_id}")
        print(f"  - New version: {result.new_version}")
        print(f"  - New accuracy: {result.new_accuracy:.2%}")
        print(f"  - Old accuracy: {result.old_accuracy:.2%}" if result.old_accuracy else "  - Old accuracy: N/A")
        print(f"  - Deployed: {result.deployed}")
        print(f"  - Rolled back: {result.rolled_back}")
        
        if result.error_message:
            print(f"  - Error: {result.error_message}")
    
    # Get retraining statistics
    stats = scheduler.get_retraining_statistics("demo_predictor")
    print(f"\n✓ Retraining Statistics:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")
    
    return scheduler


def demo_rollback(scheduler: ModelRetrainingScheduler):
    """Demonstrate model rollback."""
    print("\n" + "="*60)
    print("DEMO 4: Model Rollback")
    print("="*60)
    
    # Attempt rollback
    success = scheduler.rollback_model("demo_predictor")
    
    if success:
        print(f"\n✓ Successfully rolled back to previous version")
        
        # Check deployed model
        deployed = scheduler.registry.get_deployed_model("demo_predictor")
        if deployed:
            print(f"  - Current deployed: {deployed.model_id} v{deployed.version}")
            print(f"  - Accuracy: {deployed.directional_accuracy:.2%}")
    else:
        print(f"\n✗ Rollback failed or no previous version available")


def demo_cleanup(registry: ModelRegistry):
    """Demonstrate model cleanup."""
    print("\n" + "="*60)
    print("DEMO 5: Model Cleanup")
    print("="*60)
    
    # List all models before cleanup
    models_before = registry.list_models(model_name="demo_predictor")
    print(f"\n✓ Models before cleanup: {len(models_before)}")
    
    # Cleanup old models
    archived = registry.cleanup_old_models(
        model_name="demo_predictor",
        keep_latest=2,
        keep_deployed=True
    )
    
    print(f"\n✓ Archived {len(archived)} old models")
    
    # List models after cleanup
    models_after = registry.list_models(model_name="demo_predictor", status="active")
    print(f"✓ Active models after cleanup: {len(models_after)}")
    
    for m in models_after:
        print(f"  - {m.model_id} v{m.version}: "
              f"Status={m.status}, Deployed={m.deployed}")


def main():
    """Run all demos."""
    print("\n" + "="*60)
    print("MODEL RETRAINING AND REGISTRY DEMO")
    print("="*60)
    
    try:
        # Demo 1: Model Registry
        registry, model_id = demo_model_registry()
        
        # Demo 2: Model Comparison
        model_b_id = demo_model_comparison(registry, model_id)
        
        # Demo 3: Automated Retraining
        scheduler = demo_automated_retraining(registry)
        
        # Demo 4: Rollback
        demo_rollback(scheduler)
        
        # Demo 5: Cleanup
        demo_cleanup(registry)
        
        print("\n" + "="*60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
