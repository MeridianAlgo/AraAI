"""
Enhanced Ensemble System Demo
Demonstrates 12+ model ensemble with regime-adaptive weighting
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ara.models.adaptive_ensemble import AdaptiveEnsembleSystem
from ara.models.regime_adaptive import MarketRegime


def generate_sample_data(n_samples=1000, n_features=20, regime='bull'):
    """
    Generate synthetic financial data for different market regimes
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        regime: Market regime ('bull', 'bear', 'sideways', 'volatile')
    """
    np.random.seed(42)
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate prices based on regime
    if regime == 'bull':
        # Upward trend with low volatility
        trend = np.linspace(100, 150, n_samples)
        noise = np.random.randn(n_samples) * 2
    elif regime == 'bear':
        # Downward trend with moderate volatility
        trend = np.linspace(100, 70, n_samples)
        noise = np.random.randn(n_samples) * 3
    elif regime == 'sideways':
        # Range-bound with low volatility
        trend = np.ones(n_samples) * 100 + np.sin(np.linspace(0, 4*np.pi, n_samples)) * 5
        noise = np.random.randn(n_samples) * 1.5
    else:  # volatile
        # High volatility
        trend = np.ones(n_samples) * 100
        noise = np.random.randn(n_samples) * 8
    
    prices = trend + noise
    
    # Calculate returns as target
    returns = np.diff(prices) / prices[:-1]
    
    # Align X with returns
    X = X[:-1]
    
    return X, returns, prices


def demo_basic_ensemble():
    """Demonstrate basic ensemble training and prediction"""
    print("=" * 80)
    print("DEMO 1: Basic Enhanced Ensemble")
    print("=" * 80)
    
    # Generate training data
    print("\n1. Generating training data...")
    X_train, y_train, prices_train = generate_sample_data(1000, 20, 'bull')
    print(f"   Training samples: {len(X_train)}")
    print(f"   Features: {X_train.shape[1]}")
    
    # Create ensemble
    print("\n2. Creating adaptive ensemble system...")
    ensemble = AdaptiveEnsembleSystem("demo_ensemble")
    
    # Train
    print("\n3. Training ensemble (this may take a minute)...")
    results = ensemble.train(
        X_train, 
        y_train, 
        prices=prices_train,
        validation_split=0.2,
        n_estimators=50,  # Reduced for demo speed
        max_depth=5
    )
    
    print(f"\n   Training completed!")
    print(f"   Detected regime: {results['regime']}")
    print(f"   Active models: {len(ensemble.ensemble.get_active_models())}")
    
    # Show model weights
    print("\n4. Model weights:")
    for model_name, weight in sorted(results['model_weights'].items(), 
                                     key=lambda x: x[1], reverse=True):
        if weight > 0.01:
            print(f"   {model_name:20s}: {weight:.4f}")
    
    # Make predictions
    print("\n5. Making predictions...")
    X_test, y_test, prices_test = generate_sample_data(100, 20, 'bull')
    predictions, confidence = ensemble.predict(X_test, prices=prices_test)
    
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Mean prediction: {np.mean(predictions):.6f}")
    print(f"   Mean confidence: {np.mean(confidence):.4f}")
    
    # Calculate accuracy
    pred_direction = np.sign(predictions)
    actual_direction = np.sign(y_test)
    accuracy = np.mean(pred_direction == actual_direction)
    print(f"   Directional accuracy: {accuracy:.2%}")
    
    return ensemble


def demo_regime_adaptation():
    """Demonstrate regime-adaptive weighting"""
    print("\n" + "=" * 80)
    print("DEMO 2: Regime-Adaptive Weighting")
    print("=" * 80)
    
    # Create ensemble
    ensemble = AdaptiveEnsembleSystem("regime_demo")
    
    # Train on different regimes
    regimes = ['bull', 'bear', 'sideways', 'volatile']
    
    for regime in regimes:
        print(f"\n--- Training on {regime.upper()} market ---")
        
        # Generate regime-specific data
        X_train, y_train, prices_train = generate_sample_data(500, 20, regime)
        
        # Train
        results = ensemble.train(
            X_train, 
            y_train, 
            prices=prices_train,
            validation_split=0.2,
            n_estimators=30
        )
        
        print(f"Detected regime: {results['regime']}")
        
        # Show top 5 models for this regime
        top_models = ensemble.get_best_models_for_regime(5)
        print(f"Top 5 models: {', '.join(top_models)}")
        
        # Test on same regime
        X_test, y_test, prices_test = generate_sample_data(100, 20, regime)
        predictions, confidence = ensemble.predict(X_test, prices=prices_test)
        
        pred_direction = np.sign(predictions)
        actual_direction = np.sign(y_test)
        accuracy = np.mean(pred_direction == actual_direction)
        print(f"Accuracy: {accuracy:.2%}")


def demo_performance_tracking():
    """Demonstrate performance tracking and weight optimization"""
    print("\n" + "=" * 80)
    print("DEMO 3: Performance Tracking and Optimization")
    print("=" * 80)
    
    # Create ensemble
    ensemble = AdaptiveEnsembleSystem("tracking_demo")
    
    # Initial training
    print("\n1. Initial training...")
    X_train, y_train, prices_train = generate_sample_data(800, 20, 'bull')
    ensemble.train(X_train, y_train, prices=prices_train, n_estimators=30)
    
    # Simulate multiple predictions with tracking
    print("\n2. Making predictions with performance tracking...")
    
    for i in range(5):
        # Generate test data
        X_test, y_test, prices_test = generate_sample_data(50, 20, 'bull')
        
        # Predict with tracking
        result = ensemble.predict_with_tracking(
            X_test, 
            prices=prices_test,
            actual=y_test
        )
        
        print(f"\n   Batch {i+1}:")
        print(f"   Regime: {result['regime']}")
        print(f"   Active models: {len(result['active_models'])}")
        print(f"   Mean confidence: {np.mean(result['confidence']):.4f}")
    
    # Optimize weights based on tracked performance
    print("\n3. Optimizing weights based on performance...")
    ensemble.optimize_for_regime()
    
    # Show performance report
    print("\n4. Performance report:")
    report = ensemble.get_performance_report()
    
    print(f"   Current regime: {report['current_regime']}")
    print(f"   Total models: {report['model_count']}")
    print(f"   Active models: {len(report['active_models'])}")
    
    # Show regime statistics
    print("\n5. Regime statistics:")
    for regime, stats in report['regime_statistics'].items():
        if stats['model_count'] > 0:
            print(f"\n   {regime.upper()}:")
            print(f"   Models tracked: {stats['model_count']}")
            
            # Show top 3 models by performance
            if stats['models']:
                sorted_models = sorted(
                    stats['models'].items(),
                    key=lambda x: x[1]['mae']
                )[:3]
                
                for model_name, perf in sorted_models:
                    print(f"     {model_name:20s}: MAE={perf['mae']:.6f}, Count={perf['count']}")


def demo_model_explanations():
    """Demonstrate model explanations"""
    print("\n" + "=" * 80)
    print("DEMO 4: Model Explanations")
    print("=" * 80)
    
    # Create and train ensemble
    print("\n1. Training ensemble...")
    ensemble = AdaptiveEnsembleSystem("explain_demo")
    X_train, y_train, prices_train = generate_sample_data(500, 20, 'bull')
    ensemble.train(X_train, y_train, prices=prices_train, n_estimators=30)
    
    # Generate test sample
    X_test, _, _ = generate_sample_data(1, 20, 'bull')
    
    # Get explanations
    print("\n2. Generating explanations...")
    explanations = ensemble.explain(X_test)
    
    print(f"\n   Current regime: {explanations['current_regime']}")
    
    print("\n   Individual model predictions:")
    for model_name, pred_info in explanations['individual_predictions'].items():
        if pred_info['weight'] > 0.01:
            print(f"     {model_name:20s}: pred={pred_info['prediction']:8.6f}, "
                  f"weight={pred_info['weight']:.4f}")
    
    print("\n   Best models for current regime:")
    for i, model_name in enumerate(explanations['best_models'], 1):
        print(f"     {i}. {model_name}")
    
    # Show feature importance if available
    if explanations['feature_importance']:
        print("\n   Feature importance (from tree models):")
        for model_name, importance in list(explanations['feature_importance'].items())[:2]:
            top_features = np.argsort(importance)[-5:][::-1]
            print(f"\n     {model_name}:")
            for idx in top_features:
                print(f"       Feature {idx}: {importance[idx]:.4f}")


def demo_save_load():
    """Demonstrate saving and loading"""
    print("\n" + "=" * 80)
    print("DEMO 5: Save and Load")
    print("=" * 80)
    
    # Create and train ensemble
    print("\n1. Training ensemble...")
    ensemble = AdaptiveEnsembleSystem("save_demo")
    X_train, y_train, prices_train = generate_sample_data(500, 20, 'bull')
    ensemble.train(X_train, y_train, prices=prices_train, n_estimators=30)
    
    # Make prediction
    X_test, _, _ = generate_sample_data(10, 20, 'bull')
    pred_before, conf_before = ensemble.predict(X_test)
    
    # Save
    print("\n2. Saving ensemble...")
    save_path = Path("models/demo_ensemble")
    ensemble.save(save_path)
    print(f"   Saved to {save_path}")
    
    # Load into new instance
    print("\n3. Loading ensemble...")
    new_ensemble = AdaptiveEnsembleSystem("loaded_demo")
    new_ensemble.load(save_path)
    print("   Loaded successfully")
    
    # Make prediction with loaded model
    pred_after, conf_after = new_ensemble.predict(X_test)
    
    # Verify predictions match
    print("\n4. Verifying predictions match...")
    pred_diff = np.abs(pred_before - pred_after).max()
    conf_diff = np.abs(conf_before - conf_after).max()
    
    print(f"   Max prediction difference: {pred_diff:.10f}")
    print(f"   Max confidence difference: {conf_diff:.10f}")
    
    if pred_diff < 1e-6 and conf_diff < 1e-6:
        print("   ✓ Predictions match perfectly!")
    else:
        print("   ✗ Warning: Predictions differ")


def main():
    """Run all demos"""
    print("\n" + "=" * 80)
    print("ENHANCED ENSEMBLE SYSTEM DEMONSTRATION")
    print("12+ Models with Regime-Adaptive Weighting")
    print("=" * 80)
    
    try:
        # Run demos
        demo_basic_ensemble()
        demo_regime_adaptation()
        demo_performance_tracking()
        demo_model_explanations()
        demo_save_load()
        
        print("\n" + "=" * 80)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
