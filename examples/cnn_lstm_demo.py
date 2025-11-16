"""
CNN-LSTM Hybrid Model Demo
Demonstrates the usage of CNN-LSTM model for financial prediction
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ara.models.cnn_lstm import CNNLSTMModel
from ara.models.cnn_lstm_training import (
    AdvancedCNNLSTMTrainer,
    ModelPruner,
    ModelQuantizer
)


def generate_synthetic_data(n_samples=500, n_features=50):
    """Generate synthetic financial time series data"""
    np.random.seed(42)
    
    # Generate price-like data with trend and noise
    t = np.linspace(0, 10, n_samples)
    trend = 100 + 5 * t
    seasonality = 10 * np.sin(2 * np.pi * t / 2)
    noise = np.random.randn(n_samples) * 5
    
    prices = trend + seasonality + noise
    
    # Generate features (technical indicators, etc.)
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    
    # Add some correlation with price
    X[:, 0] = prices / 100  # Normalized price
    X[:, 1] = np.gradient(prices)  # Price change
    X[:, 2] = np.gradient(np.gradient(prices))  # Acceleration
    
    # Target: next day price change
    y = np.diff(prices, prepend=prices[0]).astype(np.float32)
    
    return X, y, prices


def demo_basic_training():
    """Demo 1: Basic CNN-LSTM training"""
    print("=" * 60)
    print("Demo 1: Basic CNN-LSTM Training")
    print("=" * 60)
    
    # Generate data
    X, y, prices = generate_synthetic_data(n_samples=500, n_features=50)
    
    print(f"\nData shape: X={X.shape}, y={y.shape}")
    
    # Create model
    model = CNNLSTMModel(
        input_dim=50,
        cnn_channels=[64, 128, 256],
        kernel_sizes=[3, 5, 7],
        lstm_hidden_dim=256,
        lstm_num_layers=2,
        dropout=0.2,
        prediction_horizons=5,
        device='cpu'
    )
    
    print(f"\nModel created with {model.model.count_parameters():,} parameters")
    
    # Train
    print("\nTraining model...")
    history = model.train(
        X, y,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        learning_rate=0.001,
        seq_length=60
    )
    
    # Plot training history
    plt.figure(figsize=(10, 4))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CNN-LSTM Training History')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cnn_lstm_training_history.png')
    print("\nTraining history saved to 'cnn_lstm_training_history.png'")
    
    # Make predictions
    print("\nMaking predictions...")
    predictions, confidence = model.predict(X, seq_length=60)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Confidence shape: {confidence.shape}")
    print(f"Average confidence: {confidence.mean():.4f}")
    
    return model, X, y, predictions, confidence


def demo_advanced_training():
    """Demo 2: Advanced training with optimizations"""
    print("\n" + "=" * 60)
    print("Demo 2: Advanced Training with Optimizations")
    print("=" * 60)
    
    # Generate data
    X, y, prices = generate_synthetic_data(n_samples=500, n_features=50)
    
    # Create model
    model = CNNLSTMModel(
        input_dim=50,
        cnn_channels=[64, 128],
        kernel_sizes=[3, 5],
        lstm_hidden_dim=128,
        lstm_num_layers=2,
        dropout=0.2,
        prediction_horizons=5,
        device='cpu'
    )
    
    # Create advanced trainer
    trainer = AdvancedCNNLSTMTrainer(
        model=model,
        use_augmentation=True,
        use_curriculum=True,
        use_gradient_accumulation=False,
        augmentation_prob=0.5
    )
    
    print("\nTraining with advanced techniques:")
    print("- Data augmentation")
    print("- Curriculum learning")
    
    # Train
    history = trainer.train_with_optimization(
        X, y,
        validation_split=0.2,
        epochs=20,
        batch_size=32,
        learning_rate=0.001,
        seq_length=60
    )
    
    print(f"\nFinal train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    
    return model, trainer


def demo_model_optimization():
    """Demo 3: Model pruning and quantization"""
    print("\n" + "=" * 60)
    print("Demo 3: Model Pruning and Quantization")
    print("=" * 60)
    
    # Generate data
    X, y, prices = generate_synthetic_data(n_samples=300, n_features=30)
    
    # Create and train model
    model = CNNLSTMModel(
        input_dim=30,
        cnn_channels=[32, 64],
        lstm_hidden_dim=64,
        lstm_num_layers=1,
        device='cpu'
    )
    
    print("\nTraining base model...")
    model.train(X, y, epochs=10, batch_size=16, seq_length=40)
    
    original_params = model.model.count_parameters()
    print(f"\nOriginal model parameters: {original_params:,}")
    
    # Create trainer for optimization
    trainer = AdvancedCNNLSTMTrainer(model=model)
    
    # Apply pruning and quantization
    print("\nApplying pruning and quantization...")
    trainer.prune_and_quantize(
        pruning_amount=0.3,
        quantization_type='fp16'
    )
    
    optimized_params = model.model.count_parameters()
    print(f"Optimized model parameters: {optimized_params:,}")
    
    return model


def demo_attention_visualization():
    """Demo 4: Attention mechanism visualization"""
    print("\n" + "=" * 60)
    print("Demo 4: Attention Mechanism Visualization")
    print("=" * 60)
    
    # Generate data
    X, y, prices = generate_synthetic_data(n_samples=300, n_features=30)
    
    # Create and train model
    model = CNNLSTMModel(
        input_dim=30,
        cnn_channels=[32, 64],
        lstm_hidden_dim=64,
        device='cpu'
    )
    
    print("\nTraining model...")
    model.train(X, y, epochs=10, batch_size=16, seq_length=40)
    
    # Get explanations with attention weights
    print("\nGenerating explanations...")
    explanations = model.explain(X[:100], seq_length=40)
    
    if explanations['attention_weights'] is not None:
        attention = explanations['attention_weights']
        print(f"Attention weights shape: {attention.shape}")
        
        # Visualize attention for first sample
        plt.figure(figsize=(12, 4))
        plt.imshow(attention[0:10].T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Attention Weight')
        plt.xlabel('Sample')
        plt.ylabel('Time Step')
        plt.title('CNN-LSTM Attention Weights')
        plt.tight_layout()
        plt.savefig('cnn_lstm_attention.png')
        print("Attention visualization saved to 'cnn_lstm_attention.png'")
    else:
        print("Attention weights not available")
    
    # Show uncertainties
    uncertainties = explanations['uncertainties']
    print(f"\nUncertainty statistics:")
    print(f"  Mean: {uncertainties.mean():.4f}")
    print(f"  Std: {uncertainties.std():.4f}")
    print(f"  Min: {uncertainties.min():.4f}")
    print(f"  Max: {uncertainties.max():.4f}")


def demo_save_load():
    """Demo 5: Model persistence"""
    print("\n" + "=" * 60)
    print("Demo 5: Model Save and Load")
    print("=" * 60)
    
    # Generate data
    X, y, prices = generate_synthetic_data(n_samples=300, n_features=30)
    
    # Create and train model
    model = CNNLSTMModel(
        input_dim=30,
        cnn_channels=[32, 64],
        lstm_hidden_dim=64,
        device='cpu'
    )
    
    print("\nTraining model...")
    model.train(X, y, epochs=5, batch_size=16, seq_length=40)
    
    # Make predictions before saving
    pred_before, conf_before = model.predict(X, seq_length=40)
    
    # Save model
    save_path = Path("models/cnn_lstm_demo")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to {save_path}")
    
    # Load model
    loaded_model = CNNLSTMModel(
        input_dim=30,
        cnn_channels=[32, 64],
        lstm_hidden_dim=64,
        device='cpu'
    )
    loaded_model.load(save_path)
    print(f"Model loaded from {save_path}")
    
    # Make predictions after loading
    pred_after, conf_after = loaded_model.predict(X, seq_length=40)
    
    # Verify predictions are identical
    pred_diff = np.abs(pred_before - pred_after).max()
    print(f"\nMax prediction difference: {pred_diff:.10f}")
    
    if pred_diff < 1e-5:
        print("✓ Model save/load successful - predictions match!")
    else:
        print("✗ Warning: Predictions differ after load")


def demo_comparison_with_transformer():
    """Demo 6: Compare CNN-LSTM with Transformer"""
    print("\n" + "=" * 60)
    print("Demo 6: CNN-LSTM vs Transformer Comparison")
    print("=" * 60)
    
    # Generate data
    X, y, prices = generate_synthetic_data(n_samples=400, n_features=40)
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Train CNN-LSTM
    print("\nTraining CNN-LSTM...")
    cnn_lstm = CNNLSTMModel(
        input_dim=40,
        cnn_channels=[64, 128],
        lstm_hidden_dim=128,
        device='cpu'
    )
    
    cnn_lstm_history = cnn_lstm.train(
        X_train, y_train,
        epochs=15,
        batch_size=32,
        seq_length=50
    )
    
    # Make predictions
    cnn_lstm_pred, cnn_lstm_conf = cnn_lstm.predict(X_test, seq_length=50)
    
    print(f"\nCNN-LSTM Results:")
    print(f"  Parameters: {cnn_lstm.model.count_parameters():,}")
    print(f"  Final train loss: {cnn_lstm_history['train_loss'][-1]:.6f}")
    print(f"  Final val loss: {cnn_lstm_history['val_loss'][-1]:.6f}")
    print(f"  Average confidence: {cnn_lstm_conf.mean():.4f}")
    
    # Note: Transformer comparison would require importing transformer model
    print("\nNote: For full comparison, import and train FinancialTransformerModel")


def main():
    """Run all demos"""
    print("\n" + "=" * 60)
    print("CNN-LSTM Hybrid Model Demonstration")
    print("=" * 60)
    
    try:
        # Run demos
        demo_basic_training()
        demo_advanced_training()
        demo_model_optimization()
        demo_attention_visualization()
        demo_save_load()
        demo_comparison_with_transformer()
        
        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
