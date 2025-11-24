"""
Financial Transformer Model Demo
Demonstrates training and inference with the state-of-the-art transformer model
"""

import numpy as np
import torch
from pathlib import Path

from ara.models.transformer import FinancialTransformerModel
from ara.models.transformer_training import TransformerTrainer
from ara.models.transformer_inference import TransformerInference


def generate_synthetic_data(n_samples=1000, n_features=50, seq_length=60):
    """Generate synthetic time series data for demonstration"""
    print("Generating synthetic financial data...")
    
    # Generate features with temporal patterns
    X = []
    y = []
    
    for i in range(n_samples):
        # Create a sample with trend and noise
        trend = np.linspace(0, 1, n_features) * (i / n_samples)
        noise = np.random.randn(n_features) * 0.1
        sample = trend + noise
        
        # Target is a function of the features
        target = sample.mean() + np.random.randn() * 0.05
        
        X.append(sample)
        y.append(target)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Generated {n_samples} samples with {n_features} features")
    return X, y


def demo_basic_training():
    """Demonstrate basic transformer training"""
    print("\n" + "="*60)
    print("DEMO 1: Basic Transformer Training")
    print("="*60)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=500, n_features=50)
    
    # Create model
    model = FinancialTransformerModel(
        input_dim=50,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        prediction_horizons=1,
        device='cpu'
    )
    
    print(f"\nModel created with {model.model.count_parameters():,} parameters")
    
    # Train model
    history = model.train(
        X=X,
        y=y,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        seq_length=30
    )
    
    print(f"\nTraining completed!")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    
    # Make predictions
    predictions, confidence = model.predict(X[:10], seq_length=30)
    
    print(f"\nPredictions shape: {predictions.shape}")
    print(f"Confidence shape: {confidence.shape}")
    print(f"Sample predictions: {predictions[:3, 0]}")
    print(f"Sample confidence: {confidence[:3, 0]}")
    
    return model


def demo_advanced_training():
    """Demonstrate advanced training with TransformerTrainer"""
    print("\n" + "="*60)
    print("DEMO 2: Advanced Training Pipeline")
    print("="*60)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=800, n_features=50)
    
    # Prepare sequences
    seq_length = 40
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X) - seq_length):
        X_sequences.append(X[i:i+seq_length])
        y_sequences.append(y[i+seq_length])
    
    X_seq = np.array(X_sequences)
    y_seq = np.array(y_sequences)
    
    print(f"Prepared {len(X_seq)} sequences of length {seq_length}")
    
    # Normalize
    X_mean = X_seq.mean(axis=(0, 1))
    X_std = X_seq.std(axis=(0, 1)) + 1e-8
    X_normalized = (X_seq - X_mean) / X_std
    
    y_mean = y_seq.mean()
    y_std = y_seq.std() + 1e-8
    y_normalized = (y_seq - y_mean) / y_std
    
    # Split data
    n_train = int(len(X_normalized) * 0.8)
    X_train = torch.FloatTensor(X_normalized[:n_train])
    y_train = torch.FloatTensor(y_normalized[:n_train])
    X_val = torch.FloatTensor(X_normalized[n_train:])
    y_val = torch.FloatTensor(y_normalized[n_train:])
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    # Create model
    from ara.models.transformer import FinancialTransformer
    model = FinancialTransformer(
        input_dim=50,
        d_model=128,
        nhead=8,
        num_encoder_layers=4,
        dim_feedforward=512,
        dropout=0.1,
        prediction_horizons=1
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create trainer
    trainer = TransformerTrainer(
        model=model,
        device='cpu',
        use_mixed_precision=False
    )
    
    # Train with advanced features
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,
        learning_rate=0.001,
        weight_decay=0.01,
        warmup_epochs=5,
        gradient_clip_norm=1.0,
        early_stopping_patience=10,
        checkpoint_dir=Path("models/checkpoints"),
        verbose=True
    )
    
    # Get training summary
    summary = trainer.get_training_summary()
    print(f"\nTraining Summary:")
    print(f"  Total epochs: {summary['total_epochs']}")
    print(f"  Best val loss: {summary['best_val_loss']:.6f}")
    print(f"  Final train loss: {summary['final_train_loss']:.6f}")
    
    return model, trainer


def demo_inference_and_uncertainty():
    """Demonstrate inference with uncertainty quantification"""
    print("\n" + "="*60)
    print("DEMO 3: Inference and Uncertainty Quantification")
    print("="*60)
    
    # Generate data
    X, y = generate_synthetic_data(n_samples=500, n_features=50)
    
    # Train a simple model
    model = FinancialTransformerModel(
        input_dim=50,
        d_model=128,
        nhead=8,
        num_encoder_layers=3,
        prediction_horizons=5,  # Multi-horizon
        device='cpu'
    )
    
    print("Training model...")
    model.train(X, y, epochs=30, batch_size=32, seq_length=30)
    
    # Prepare test sequences
    seq_length = 30
    X_test = []
    for i in range(10):
        X_test.append(X[i:i+seq_length])
    X_test = np.array(X_test)
    
    # Create inference engine
    inference = TransformerInference(
        model=model.model,
        device='cpu',
        feature_mean=model.feature_mean,
        feature_std=model.feature_std,
        target_mean=model.target_mean,
        target_std=model.target_std
    )
    
    # Standard inference
    print("\n1. Standard Batched Inference:")
    result = inference.predict_batch(X_test, batch_size=4)
    print(f"   Predictions shape: {result.predictions.shape}")
    print(f"   Confidence scores: {result.confidence_scores[0]}")
    print(f"   95% CI: [{result.lower_bounds[0, 0]:.4f}, {result.upper_bounds[0, 0]:.4f}]")
    
    # Monte Carlo dropout
    print("\n2. Monte Carlo Dropout (30 samples):")
    mc_result = inference.predict_with_mc_dropout(X_test, n_samples=30)
    print(f"   Predictions: {mc_result.predictions[0]}")
    print(f"   Uncertainties: {mc_result.uncertainties[0]}")
    print(f"   Confidence: {mc_result.confidence_scores[0]}")
    
    # Multi-horizon predictions
    print("\n3. Multi-Horizon Predictions:")
    horizons_result = inference.predict_multi_horizon(
        X_test[:1],
        horizons=[1, 3, 5],
        use_mc_dropout=True,
        mc_samples=20
    )
    
    for horizon, result in horizons_result.items():
        print(f"   {horizon}-day: {result.predictions[0, 0]:.4f} "
              f"(confidence: {result.confidence_scores[0, 0]:.3f})")
    
    # Benchmark inference speed
    print("\n4. Inference Speed Benchmark:")
    benchmark = inference.benchmark_inference_speed(
        input_shape=(32, 30, 50),
        n_iterations=50
    )
    print(f"   Mean time: {benchmark['mean_time_ms']:.2f} ms")
    print(f"   Throughput: {benchmark['throughput_samples_per_sec']:.1f} samples/sec")
    
    return inference


def demo_explainability():
    """Demonstrate explainability features"""
    print("\n" + "="*60)
    print("DEMO 4: Explainability and Attention Visualization")
    print("="*60)
    
    # Generate data with named features
    X, y = generate_synthetic_data(n_samples=300, n_features=20)
    
    feature_names = [
        f"feature_{i}" for i in range(20)
    ]
    
    # Train model
    model = FinancialTransformerModel(
        input_dim=20,
        d_model=64,
        nhead=4,
        num_encoder_layers=2,
        prediction_horizons=1,
        device='cpu'
    )
    
    print("Training model...")
    model.train(X, y, epochs=20, batch_size=16, seq_length=20)
    
    # Prepare test sequence
    X_test = X[:20].reshape(1, 20, 20)
    
    # Create inference engine
    inference = TransformerInference(
        model=model.model,
        device='cpu',
        feature_mean=model.feature_mean,
        feature_std=model.feature_std,
        target_mean=model.target_mean,
        target_std=model.target_std
    )
    
    # Get explanation
    print("\nGenerating prediction explanation...")
    explanation = inference.explain_prediction(X_test, feature_names=feature_names)
    
    print(f"\nPrediction: {explanation['predictions'][0]:.4f}")
    print(f"Confidence: {explanation['confidence_scores'][0]:.3f}")
    print(f"Uncertainty: {explanation['uncertainties'][0]:.4f}")
    
    print("\nTop 5 Important Features:")
    for i, feature in enumerate(explanation['top_features'][:5], 1):
        importance = explanation['feature_importance'][feature]
        print(f"  {i}. {feature}: {importance:.4f}")
    
    return explanation


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("Financial Transformer Model Demonstrations")
    print("="*60)
    
    # Check PyTorch availability
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Run demos
    try:
        # Demo 1: Basic training
        model1 = demo_basic_training()
        
        # Demo 2: Advanced training
        model2, trainer = demo_advanced_training()
        
        # Demo 3: Inference and uncertainty
        inference = demo_inference_and_uncertainty()
        
        # Demo 4: Explainability
        explanation = demo_explainability()
        
        print("\n" + "="*60)
        print("All demos completed successfully!")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
