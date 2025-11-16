"""
Tests for Financial Transformer Model
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile

from ara.models.transformer import (
    FinancialTransformer,
    FinancialTransformerModel,
    PositionalEncoding
)
from ara.models.transformer_training import TransformerTrainer
from ara.models.transformer_inference import TransformerInference, PredictionResult
from ara.core.exceptions import ModelError, ValidationError


@pytest.fixture
def sample_data():
    """Generate sample data for testing"""
    np.random.seed(42)
    X = np.random.randn(100, 50).astype(np.float32)
    y = np.random.randn(100).astype(np.float32)
    return X, y


@pytest.fixture
def sample_sequences():
    """Generate sample sequences for testing"""
    np.random.seed(42)
    X = np.random.randn(10, 30, 50).astype(np.float32)
    y = np.random.randn(10).astype(np.float32)
    return X, y


class TestPositionalEncoding:
    """Test PositionalEncoding module"""
    
    def test_initialization(self):
        """Test positional encoding initialization"""
        pe = PositionalEncoding(d_model=64, max_len=100)
        assert pe.pe.shape == (100, 1, 64)
    
    def test_forward(self):
        """Test forward pass"""
        pe = PositionalEncoding(d_model=64)
        x = torch.randn(50, 8, 64)  # (seq_len, batch_size, d_model)
        output = pe(x)
        assert output.shape == x.shape


class TestFinancialTransformer:
    """Test FinancialTransformer model"""
    
    def test_initialization(self):
        """Test model initialization"""
        model = FinancialTransformer(
            input_dim=50,
            d_model=128,
            nhead=8,
            num_encoder_layers=4
        )
        assert model.input_dim == 50
        assert model.d_model == 128
        assert model.nhead == 8
    
    def test_parameter_count(self):
        """Test parameter counting"""
        model = FinancialTransformer(input_dim=50, d_model=128)
        param_count = model.count_parameters()
        assert param_count > 0
        assert isinstance(param_count, int)
    
    def test_forward_pass(self):
        """Test forward pass"""
        model = FinancialTransformer(
            input_dim=50,
            d_model=64,
            nhead=4,
            prediction_horizons=5
        )
        
        x = torch.randn(30, 8, 50)  # (seq_len, batch_size, input_dim)
        predictions, uncertainties, attention = model(x)
        
        assert predictions.shape == (8, 5)  # (batch_size, horizons)
        assert uncertainties.shape == (8, 5)
    
    def test_forward_with_attention(self):
        """Test forward pass with attention extraction"""
        model = FinancialTransformer(input_dim=50, d_model=64, nhead=4)
        x = torch.randn(30, 4, 50)
        
        predictions, uncertainties, attention = model(x, return_attention=True)
        
        assert predictions.shape[0] == 4
        assert uncertainties.shape[0] == 4


class TestFinancialTransformerModel:
    """Test FinancialTransformerModel wrapper"""
    
    def test_initialization(self):
        """Test model wrapper initialization"""
        model = FinancialTransformerModel(
            input_dim=50,
            d_model=128,
            nhead=8
        )
        assert model.input_dim == 50
        assert not model.is_trained
    
    def test_training(self, sample_data):
        """Test model training"""
        X, y = sample_data
        
        model = FinancialTransformerModel(
            input_dim=50,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            prediction_horizons=1
        )
        
        history = model.train(
            X=X,
            y=y,
            epochs=5,
            batch_size=16,
            seq_length=20
        )
        
        assert model.is_trained
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) > 0
    
    def test_prediction(self, sample_data):
        """Test model prediction"""
        X, y = sample_data
        
        model = FinancialTransformerModel(
            input_dim=50,
            d_model=64,
            nhead=4,
            num_encoder_layers=2
        )
        
        # Train model
        model.train(X, y, epochs=3, batch_size=16, seq_length=20)
        
        # Make predictions
        predictions, confidence = model.predict(X[:30], seq_length=20)
        
        assert predictions.shape[0] > 0
        assert confidence.shape[0] > 0
        assert np.all(confidence >= 0) and np.all(confidence <= 1)
    
    def test_save_load(self, sample_data):
        """Test model save and load"""
        X, y = sample_data
        
        model = FinancialTransformerModel(input_dim=50, d_model=64, nhead=4)
        model.train(X, y, epochs=3, batch_size=16, seq_length=20)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / "test_model"
            
            # Save model
            model.save(save_path)
            assert save_path.with_suffix('.pt').exists()
            
            # Load model
            new_model = FinancialTransformerModel(input_dim=50, d_model=64, nhead=4)
            new_model.load(save_path)
            
            assert new_model.is_trained
            
            # Compare predictions
            pred1, _ = model.predict(X[:30], seq_length=20)
            pred2, _ = new_model.predict(X[:30], seq_length=20)
            
            np.testing.assert_allclose(pred1, pred2, rtol=1e-5)
    
    def test_validation_error(self):
        """Test input validation"""
        model = FinancialTransformerModel(input_dim=50)
        
        # Invalid input type
        with pytest.raises(ValidationError):
            model.train([1, 2, 3], np.array([1, 2, 3]))
        
        # NaN values
        X_nan = np.array([[1, 2, np.nan], [4, 5, 6]])
        y = np.array([1, 2])
        with pytest.raises(ValidationError):
            model.train(X_nan, y)


class TestTransformerTrainer:
    """Test TransformerTrainer"""
    
    def test_initialization(self):
        """Test trainer initialization"""
        model = FinancialTransformer(input_dim=50, d_model=64, nhead=4)
        trainer = TransformerTrainer(model, device='cpu')
        
        assert trainer.model is model
        assert trainer.device.type == 'cpu'
    
    def test_training_setup(self):
        """Test training setup"""
        model = FinancialTransformer(input_dim=50, d_model=64, nhead=4)
        trainer = TransformerTrainer(model)
        
        trainer.setup_training(
            learning_rate=0.001,
            warmup_epochs=5,
            total_epochs=50
        )
        
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.criterion is not None
    
    def test_full_training(self, sample_sequences):
        """Test full training loop"""
        X, y = sample_sequences
        
        # Create data loaders
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        train_dataset = torch.utils.data.TensorDataset(X_tensor[:8], y_tensor[:8])
        val_dataset = torch.utils.data.TensorDataset(X_tensor[8:], y_tensor[8:])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=2)
        
        # Create and train
        model = FinancialTransformer(input_dim=50, d_model=64, nhead=4)
        trainer = TransformerTrainer(model)
        
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=5,
            verbose=False
        )
        
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) > 0


class TestTransformerInference:
    """Test TransformerInference"""
    
    @pytest.fixture
    def trained_model_and_data(self, sample_data):
        """Create a trained model for inference testing"""
        X, y = sample_data
        
        model = FinancialTransformerModel(
            input_dim=50,
            d_model=64,
            nhead=4,
            num_encoder_layers=2,
            prediction_horizons=3
        )
        
        model.train(X, y, epochs=3, batch_size=16, seq_length=20)
        
        # Prepare test sequences
        X_test = []
        for i in range(5):
            X_test.append(X[i:i+20])
        X_test = np.array(X_test)
        
        return model, X_test
    
    def test_initialization(self, trained_model_and_data):
        """Test inference engine initialization"""
        model, _ = trained_model_and_data
        
        inference = TransformerInference(
            model=model.model,
            device='cpu',
            feature_mean=model.feature_mean,
            feature_std=model.feature_std
        )
        
        assert inference.model is model.model
    
    def test_batch_prediction(self, trained_model_and_data):
        """Test batched inference"""
        model, X_test = trained_model_and_data
        
        inference = TransformerInference(
            model=model.model,
            device='cpu',
            feature_mean=model.feature_mean,
            feature_std=model.feature_std,
            target_mean=model.target_mean,
            target_std=model.target_std
        )
        
        result = inference.predict_batch(X_test, batch_size=2)
        
        assert isinstance(result, PredictionResult)
        assert result.predictions.shape[0] == len(X_test)
        assert result.uncertainties.shape[0] == len(X_test)
        assert result.confidence_scores.shape[0] == len(X_test)
    
    def test_mc_dropout(self, trained_model_and_data):
        """Test Monte Carlo dropout"""
        model, X_test = trained_model_and_data
        
        inference = TransformerInference(
            model=model.model,
            device='cpu',
            feature_mean=model.feature_mean,
            feature_std=model.feature_std,
            target_mean=model.target_mean,
            target_std=model.target_std
        )
        
        result = inference.predict_with_mc_dropout(X_test, n_samples=10)
        
        assert isinstance(result, PredictionResult)
        assert result.predictions.shape[0] == len(X_test)
        assert result.uncertainties.shape[0] == len(X_test)
    
    def test_multi_horizon(self, trained_model_and_data):
        """Test multi-horizon predictions"""
        model, X_test = trained_model_and_data
        
        inference = TransformerInference(
            model=model.model,
            device='cpu',
            feature_mean=model.feature_mean,
            feature_std=model.feature_std,
            target_mean=model.target_mean,
            target_std=model.target_std
        )
        
        results = inference.predict_multi_horizon(
            X_test[:1],
            horizons=[1, 2, 3],
            use_mc_dropout=False
        )
        
        assert len(results) == 3
        assert 1 in results
        assert 2 in results
        assert 3 in results
    
    def test_explainability(self, trained_model_and_data):
        """Test prediction explanation"""
        model, X_test = trained_model_and_data
        
        inference = TransformerInference(
            model=model.model,
            device='cpu',
            feature_mean=model.feature_mean,
            feature_std=model.feature_std,
            target_mean=model.target_mean,
            target_std=model.target_std
        )
        
        explanation = inference.explain_prediction(X_test[:1])
        
        assert 'predictions' in explanation
        assert 'confidence_scores' in explanation
        assert 'feature_importance' in explanation
        assert 'top_features' in explanation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
