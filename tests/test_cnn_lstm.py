"""
Tests for CNN-LSTM Hybrid Model
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile

from ara.models.cnn_lstm import CNNLSTMModel, CNNLSTMHybrid, AttentionLayer
from ara.models.cnn_lstm_training import (
    AdvancedCNNLSTMTrainer,
    DataAugmenter,
    CurriculumLearning,
    GradientAccumulator,
    ModelPruner,
    ModelQuantizer
)
from ara.core.exceptions import ModelError, ValidationError


@pytest.fixture
def sample_data():
    """Generate sample time series data"""
    np.random.seed(42)
    n_samples = 200
    n_features = 20
    
    # Generate synthetic price-like data
    X = np.random.randn(n_samples, n_features).astype(np.float32)
    y = np.random.randn(n_samples).astype(np.float32)
    
    return X, y


@pytest.fixture
def cnn_lstm_model():
    """Create CNN-LSTM model instance"""
    return CNNLSTMModel(
        input_dim=20,
        cnn_channels=[32, 64],
        kernel_sizes=[3, 5],
        lstm_hidden_dim=128,
        lstm_num_layers=2,
        dropout=0.2,
        prediction_horizons=5,
        device='cpu'
    )


class TestCNNLSTMHybrid:
    """Test CNN-LSTM hybrid architecture"""
    
    def test_model_initialization(self):
        """Test model can be initialized"""
        model = CNNLSTMHybrid(
            input_dim=20,
            cnn_channels=[32, 64],
            kernel_sizes=[3, 5],
            lstm_hidden_dim=128,
            lstm_num_layers=2,
            prediction_horizons=5
        )
        
        assert model is not None
        assert model.input_dim == 20
        assert model.lstm_hidden_dim == 128
        assert model.prediction_horizons == 5
    
    def test_forward_pass(self):
        """Test forward pass produces correct output shape"""
        model = CNNLSTMHybrid(
            input_dim=20,
            cnn_channels=[32, 64],
            kernel_sizes=[3, 5],
            lstm_hidden_dim=128,
            prediction_horizons=5
        )
        
        # Create sample input
        batch_size = 4
        seq_len = 60
        x = torch.randn(batch_size, seq_len, 20)
        
        # Forward pass
        predictions, uncertainties, attention = model(x, return_attention=False)
        
        assert predictions.shape == (batch_size, 5)
        assert uncertainties.shape == (batch_size, 5)
        assert attention is None
    
    def test_attention_mechanism(self):
        """Test attention mechanism"""
        attention_layer = AttentionLayer(hidden_dim=128)
        
        # Create sample LSTM output
        batch_size = 4
        seq_len = 60
        hidden_dim = 128
        lstm_output = torch.randn(batch_size, seq_len, hidden_dim)
        
        # Apply attention
        context, attention_weights = attention_layer(lstm_output)
        
        assert context.shape == (batch_size, hidden_dim)
        assert attention_weights.shape == (batch_size, seq_len)
        
        # Check attention weights sum to 1
        assert torch.allclose(attention_weights.sum(dim=1), torch.ones(batch_size), atol=1e-5)
    
    def test_residual_connections(self):
        """Test model with and without residual connections"""
        model_with_residual = CNNLSTMHybrid(
            input_dim=20,
            lstm_hidden_dim=128,
            use_residual=True
        )
        
        model_without_residual = CNNLSTMHybrid(
            input_dim=20,
            lstm_hidden_dim=128,
            use_residual=False
        )
        
        assert model_with_residual.use_residual == True
        assert model_without_residual.use_residual == False
        
        # Both should work
        x = torch.randn(2, 60, 20)
        pred1, _, _ = model_with_residual(x)
        pred2, _, _ = model_without_residual(x)
        
        assert pred1.shape == pred2.shape
    
    def test_parameter_count(self):
        """Test parameter counting"""
        model = CNNLSTMHybrid(
            input_dim=20,
            cnn_channels=[32, 64],
            lstm_hidden_dim=128,
            lstm_num_layers=2
        )
        
        param_count = model.count_parameters()
        assert param_count > 0
        assert isinstance(param_count, int)


class TestCNNLSTMModel:
    """Test CNN-LSTM model wrapper"""
    
    def test_model_creation(self, cnn_lstm_model):
        """Test model can be created"""
        assert cnn_lstm_model is not None
        assert cnn_lstm_model.model_name == "CNNLSTMHybrid"
        assert not cnn_lstm_model.is_trained
    
    def test_training(self, cnn_lstm_model, sample_data):
        """Test model training"""
        X, y = sample_data
        
        # Train with minimal epochs for testing
        history = cnn_lstm_model.train(
            X, y,
            validation_split=0.2,
            epochs=5,
            batch_size=16,
            seq_length=30
        )
        
        assert cnn_lstm_model.is_trained
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) > 0
    
    def test_prediction(self, cnn_lstm_model, sample_data):
        """Test model prediction"""
        X, y = sample_data
        
        # Train first
        cnn_lstm_model.train(X, y, epochs=3, batch_size=16, seq_length=30)
        
        # Predict
        predictions, confidence = cnn_lstm_model.predict(X, seq_length=30)
        
        assert predictions.shape[0] > 0
        assert confidence.shape[0] > 0
        assert predictions.shape == confidence.shape
    
    def test_prediction_without_training(self, cnn_lstm_model, sample_data):
        """Test prediction fails without training"""
        X, _ = sample_data
        
        with pytest.raises(ModelError):
            cnn_lstm_model.predict(X)
    
    def test_save_load(self, cnn_lstm_model, sample_data):
        """Test model save and load"""
        X, y = sample_data
        
        # Train
        cnn_lstm_model.train(X, y, epochs=3, batch_size=16, seq_length=30)
        
        # Save
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            cnn_lstm_model.save(save_path)
            
            # Create new model and load
            new_model = CNNLSTMModel(
                input_dim=20,
                cnn_channels=[32, 64],
                kernel_sizes=[3, 5],
                lstm_hidden_dim=128,
                device='cpu'
            )
            new_model.load(save_path)
            
            assert new_model.is_trained
            assert new_model.metadata['model_type'] == 'CNNLSTMHybrid'
    
    def test_explain(self, cnn_lstm_model, sample_data):
        """Test model explanation"""
        X, y = sample_data
        
        # Train
        cnn_lstm_model.train(X, y, epochs=3, batch_size=16, seq_length=30)
        
        # Get explanations
        explanations = cnn_lstm_model.explain(X[:50], seq_length=30)
        
        assert 'attention_weights' in explanations
        assert 'uncertainties' in explanations
        assert 'predictions' in explanations
        assert explanations['model_type'] == 'CNN-LSTM Hybrid'


class TestDataAugmentation:
    """Test data augmentation techniques"""
    
    def test_augmenter_creation(self):
        """Test augmenter can be created"""
        augmenter = DataAugmenter(
            noise_level=0.01,
            scaling_range=(0.95, 1.05)
        )
        assert augmenter is not None
    
    def test_noise_augmentation(self):
        """Test noise augmentation"""
        augmenter = DataAugmenter(noise_level=0.1)
        
        X = torch.randn(4, 60, 20)
        X_aug = augmenter.add_noise(X)
        
        assert X_aug.shape == X.shape
        assert not torch.allclose(X, X_aug)
    
    def test_scaling_augmentation(self):
        """Test scaling augmentation"""
        augmenter = DataAugmenter(scaling_range=(0.9, 1.1))
        
        X = torch.randn(4, 60, 20)
        X_aug = augmenter.scale(X)
        
        assert X_aug.shape == X.shape
    
    def test_time_warp_augmentation(self):
        """Test time warping"""
        augmenter = DataAugmenter(time_warp_strength=0.1)
        
        X = torch.randn(4, 60, 20)
        X_aug = augmenter.time_warp(X)
        
        assert X_aug.shape == X.shape
    
    def test_combined_augmentation(self):
        """Test combined augmentation"""
        augmenter = DataAugmenter()
        
        X = torch.randn(4, 60, 20)
        X_aug = augmenter.augment(X, 'all')
        
        assert X_aug.shape == X.shape
        assert not torch.allclose(X, X_aug)


class TestCurriculumLearning:
    """Test curriculum learning"""
    
    def test_curriculum_creation(self):
        """Test curriculum learning can be created"""
        curriculum = CurriculumLearning(difficulty_metric='volatility')
        assert curriculum is not None
    
    def test_difficulty_calculation(self):
        """Test difficulty calculation"""
        curriculum = CurriculumLearning(difficulty_metric='volatility')
        
        X = torch.randn(100, 60, 20)
        y = torch.randn(100)
        
        difficulty = curriculum.calculate_difficulty(X, y)
        
        assert difficulty.shape == (100,)
        assert torch.all(difficulty >= 0)
    
    def test_curriculum_batches(self):
        """Test curriculum batch generation"""
        curriculum = CurriculumLearning()
        
        X = torch.randn(100, 60, 20)
        y = torch.randn(100)
        
        # Early epoch - should get easier samples
        batches_early = curriculum.get_curriculum_batches(
            X, y, batch_size=16, epoch=0, total_epochs=100
        )
        
        # Late epoch - should get all samples
        batches_late = curriculum.get_curriculum_batches(
            X, y, batch_size=16, epoch=99, total_epochs=100
        )
        
        assert len(batches_early) > 0
        assert len(batches_late) > 0


class TestGradientAccumulation:
    """Test gradient accumulation"""
    
    def test_accumulator_creation(self):
        """Test accumulator can be created"""
        accumulator = GradientAccumulator(accumulation_steps=4)
        assert accumulator.accumulation_steps == 4
    
    def test_update_logic(self):
        """Test update logic"""
        accumulator = GradientAccumulator(accumulation_steps=4)
        
        # First 3 steps should not update
        assert not accumulator.should_update()
        assert not accumulator.should_update()
        assert not accumulator.should_update()
        
        # 4th step should update
        assert accumulator.should_update()
        
        # Reset and repeat
        assert not accumulator.should_update()
    
    def test_loss_scaling(self):
        """Test loss scaling"""
        accumulator = GradientAccumulator(accumulation_steps=4)
        
        loss = torch.tensor(1.0)
        scaled_loss = accumulator.scale_loss(loss)
        
        assert scaled_loss == 0.25


class TestModelPruning:
    """Test model pruning"""
    
    def test_pruner_creation(self):
        """Test pruner can be created"""
        pruner = ModelPruner(pruning_amount=0.3)
        assert pruner.pruning_amount == 0.3
    
    def test_model_pruning(self):
        """Test model can be pruned"""
        model = CNNLSTMHybrid(input_dim=20, lstm_hidden_dim=64)
        pruner = ModelPruner(pruning_amount=0.2)
        
        params_before = model.count_parameters()
        pruned_model = pruner.prune_model(model, pruning_method='l1_unstructured')
        params_after = pruned_model.count_parameters()
        
        # Model should still work after pruning
        x = torch.randn(2, 60, 20)
        pred, _, _ = pruned_model(x)
        assert pred.shape[0] == 2


class TestModelQuantization:
    """Test model quantization"""
    
    def test_quantizer_creation(self):
        """Test quantizer can be created"""
        quantizer = ModelQuantizer(quantization_type='fp16')
        assert quantizer.quantization_type == 'fp16'
    
    def test_fp16_quantization(self):
        """Test FP16 quantization"""
        model = CNNLSTMHybrid(input_dim=20, lstm_hidden_dim=64)
        quantizer = ModelQuantizer(quantization_type='fp16')
        
        quantized_model = quantizer.quantize_to_fp16(model)
        
        # Check model is in half precision
        for param in quantized_model.parameters():
            assert param.dtype == torch.float16


class TestAdvancedTrainer:
    """Test advanced training pipeline"""
    
    def test_trainer_creation(self, cnn_lstm_model):
        """Test trainer can be created"""
        trainer = AdvancedCNNLSTMTrainer(
            model=cnn_lstm_model,
            use_augmentation=True,
            use_curriculum=True
        )
        assert trainer is not None
    
    def test_training_with_optimizations(self, sample_data):
        """Test training with all optimizations"""
        X, y = sample_data
        
        model = CNNLSTMModel(
            input_dim=20,
            cnn_channels=[32, 64],
            lstm_hidden_dim=64,
            lstm_num_layers=1,
            device='cpu'
        )
        
        trainer = AdvancedCNNLSTMTrainer(
            model=model,
            use_augmentation=True,
            use_curriculum=True,
            use_gradient_accumulation=False
        )
        
        history = trainer.train_with_optimization(
            X, y,
            epochs=3,
            batch_size=16,
            seq_length=30
        )
        
        assert model.is_trained
        assert 'train_loss' in history
        assert len(history['train_loss']) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
