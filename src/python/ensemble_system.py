"""
Ensemble System for Stock Prediction
Combines multiple models with weighted averaging and uncertainty quantification
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import json
from collections import defaultdict

from advanced_models import (
    LSTMAttentionPredictor, TransformerPredictor, CNNLSTMPredictor,
    BayesianPredictor, EnhancedFeedforward
)

logger = logging.getLogger(__name__)

class EnsemblePredictor:
    """Manages multiple models and combines their predictions"""
    
    def __init__(self, input_size: int = 22, device: str = 'cpu'):
        self.input_size = input_size
        self.device = torch.device(device)
        
        # Initialize models
        self.models = {
            'lstm_attention': LSTMAttentionPredictor(input_size).to(self.device),
            'transformer': TransformerPredictor(input_size, d_model=128, num_layers=4).to(self.device),
            'cnn_lstm': CNNLSTMPredictor(input_size).to(self.device),
            'bayesian': BayesianPredictor(input_size, hidden_sizes=[64, 32]).to(self.device),
            'enhanced_ffn': EnhancedFeedforward(input_size).to(self.device)
        }
        
        # Performance tracking
        self.performance_history = {name: [] for name in self.models.keys()}
        self.weights = {name: 1.0 / len(self.models) for name in self.models.keys()}  # Equal weights initially
        self.training_history = []
        
        # Uncertainty quantification
        self.mc_samples = 100
        self.confidence_levels = [0.90, 0.95, 0.99]
        
        logger.info(f"Ensemble initialized with {len(self.models)} models")
        
    def train_ensemble(self, train_loader, val_loader, epochs: int = 100, 
                      learning_rate: float = 0.001, patience: int = 15):
        """Train all models in the ensemble"""
        
        ensemble_results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Train individual model
            results = self._train_single_model(
                model, train_loader, val_loader, epochs, learning_rate, patience
            )
            
            ensemble_results[name] = results
            self.performance_history[name].append(results['best_val_loss'])
            
            logger.info(f"{name} - Best Val Loss: {results['best_val_loss']:.6f}")
            
        # Update ensemble weights based on performance
        self._update_ensemble_weights()
        
        # Store training history
        self.training_history.append({
            'timestamp': datetime.now().isoformat(),
            'results': ensemble_results,
            'weights': self.weights.copy()
        })
        
        return ensemble_results
    
    def _train_single_model(self, model, train_loader, val_loader, 
                           epochs, learning_rate, patience):
        """Train a single model with early stopping"""
        
        # Special handling for Bayesian model
        if isinstance(model, BayesianPredictor):
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            criterion = self._bayesian_loss
        else:
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
            criterion = nn.MSELoss()
            
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=patience//2
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            num_batches = 0
            
            for batch_features, batch_targets in train_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                optimizer.zero_grad()
                
                if isinstance(model, BayesianPredictor):
                    outputs = model(batch_features, sample=True)
                    loss = criterion(outputs, batch_targets, model)
                else:
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                num_batches += 1
                
            avg_train_loss = train_loss / num_batches if num_batches > 0 else 0
            train_losses.append(avg_train_loss)
            
            # Validation phase
            val_loss = self._validate_model(model, val_loader, criterion)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
        return {
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'epochs_trained': len(train_losses)
        }
    
    def _bayesian_loss(self, outputs, targets, model):
        """Loss function for Bayesian neural network"""
        mse_loss = nn.MSELoss()(outputs, targets)
        kl_loss = model.kl_divergence()
        
        # Scale KL loss by number of batches (approximate)
        kl_weight = 1.0 / len(targets)
        
        return mse_loss + kl_weight * kl_loss
    
    def _validate_model(self, model, val_loader, criterion):
        """Validate a single model"""
        model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                if isinstance(model, BayesianPredictor):
                    outputs = model(batch_features, sample=False)  # Use mean weights for validation
                    loss = nn.MSELoss()(outputs, batch_targets)
                else:
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_targets)
                
                val_loss += loss.item()
                num_batches += 1
                
        return val_loss / num_batches if num_batches > 0 else 0
    
    def _update_ensemble_weights(self):
        """Update ensemble weights based on recent performance"""
        if not any(self.performance_history.values()):
            return
            
        # Use inverse of validation loss as weight (lower loss = higher weight)
        recent_performance = {}
        for name, history in self.performance_history.items():
            if history:
                # Use recent performance (last 3 results)
                recent_losses = history[-3:]
                avg_loss = np.mean(recent_losses)
                recent_performance[name] = 1.0 / (avg_loss + 1e-8)  # Avoid division by zero
                
        if recent_performance:
            # Normalize weights
            total_weight = sum(recent_performance.values())
            self.weights = {name: weight / total_weight 
                          for name, weight in recent_performance.items()}
            
            # Apply softmax for smoother weight distribution
            weight_values = np.array(list(self.weights.values()))
            softmax_weights = np.exp(weight_values) / np.sum(np.exp(weight_values))
            
            for i, name in enumerate(self.weights.keys()):
                self.weights[name] = softmax_weights[i]
                
        logger.info(f"Updated ensemble weights: {self.weights}")
    
    def predict_with_ensemble(self, features: torch.Tensor) -> Dict[str, Any]:
        """Generate ensemble prediction with comprehensive uncertainty analysis"""
        
        if len(features.shape) == 1:
            features = features.unsqueeze(0)
            
        features = features.to(self.device)
        
        # Get predictions from all models
        individual_predictions = {}
        individual_uncertainties = {}
        
        for name, model in self.models.items():
            model.eval()
            
            if isinstance(model, BayesianPredictor):
                # Bayesian uncertainty
                pred, uncertainty = model.predict_with_uncertainty(features, self.mc_samples)
                individual_predictions[name] = pred.cpu().numpy().flatten()
                individual_uncertainties[name] = uncertainty.cpu().numpy().flatten()
            else:
                # Monte Carlo Dropout uncertainty
                pred, uncertainty = self._mc_dropout_prediction(model, features)
                individual_predictions[name] = pred
                individual_uncertainties[name] = uncertainty
                
        # Weighted ensemble prediction
        ensemble_pred = self._calculate_weighted_ensemble(individual_predictions)
        
        # Ensemble uncertainty (combines model uncertainty and disagreement)
        ensemble_uncertainty = self._calculate_ensemble_uncertainty(
            individual_predictions, individual_uncertainties
        )
        
        # Prediction intervals
        prediction_intervals = self._calculate_prediction_intervals(
            individual_predictions, ensemble_pred, ensemble_uncertainty
        )
        
        # Model agreement metrics
        agreement_metrics = self._calculate_agreement_metrics(individual_predictions)
        
        return {
            'ensemble_prediction': float(ensemble_pred[0]),
            'individual_predictions': {k: float(v[0]) for k, v in individual_predictions.items()},
            'model_weights': self.weights.copy(),
            'total_uncertainty': float(ensemble_uncertainty[0]),
            'individual_uncertainties': {k: float(v[0]) for k, v in individual_uncertainties.items()},
            'prediction_intervals': prediction_intervals,
            'model_agreement': agreement_metrics,
            'ensemble_confidence': self._calculate_ensemble_confidence(agreement_metrics, ensemble_uncertainty[0])
        }
    
    def _mc_dropout_prediction(self, model, features, n_samples=None):
        """Monte Carlo Dropout for uncertainty estimation"""
        if n_samples is None:
            n_samples = self.mc_samples
            
        # Enable dropout during inference
        def enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()
                
        model.apply(enable_dropout)
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = model(features)
                predictions.append(pred.cpu().numpy())
                
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions, axis=0).flatten()
        uncertainty = np.std(predictions, axis=0).flatten()
        
        return mean_pred, uncertainty
    
    def _calculate_weighted_ensemble(self, individual_predictions):
        """Calculate weighted ensemble prediction"""
        ensemble_pred = np.zeros_like(list(individual_predictions.values())[0])
        
        for name, pred in individual_predictions.items():
            ensemble_pred += self.weights[name] * pred
            
        return ensemble_pred
    
    def _calculate_ensemble_uncertainty(self, individual_predictions, individual_uncertainties):
        """Calculate ensemble uncertainty combining model uncertainty and disagreement"""
        
        # Model disagreement (variance of predictions)
        predictions_array = np.array(list(individual_predictions.values()))
        model_disagreement = np.var(predictions_array, axis=0)
        
        # Weighted average of individual uncertainties
        avg_uncertainty = np.zeros_like(list(individual_uncertainties.values())[0])
        for name, uncertainty in individual_uncertainties.items():
            avg_uncertainty += self.weights[name] * uncertainty**2
        avg_uncertainty = np.sqrt(avg_uncertainty)
        
        # Total uncertainty (epistemic + aleatoric)
        total_uncertainty = np.sqrt(model_disagreement + avg_uncertainty**2)
        
        return total_uncertainty
    
    def _calculate_prediction_intervals(self, individual_predictions, ensemble_pred, ensemble_uncertainty):
        """Calculate prediction intervals for different confidence levels"""
        
        intervals = {}
        
        for confidence_level in self.confidence_levels:
            # Use normal approximation for intervals
            z_score = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}[confidence_level]
            
            margin = z_score * ensemble_uncertainty
            lower_bound = ensemble_pred - margin
            upper_bound = ensemble_pred + margin
            
            intervals[f"{int(confidence_level*100)}%"] = {
                'lower': float(lower_bound[0]),
                'upper': float(upper_bound[0]),
                'width': float(2 * margin[0])
            }
            
        return intervals
    
    def _calculate_agreement_metrics(self, individual_predictions):
        """Calculate model agreement metrics"""
        
        predictions_array = np.array(list(individual_predictions.values()))
        
        # Coefficient of variation
        mean_pred = np.mean(predictions_array, axis=0)
        std_pred = np.std(predictions_array, axis=0)
        cv = std_pred / (np.abs(mean_pred) + 1e-8)
        
        # Pairwise correlations
        correlations = []
        pred_list = list(individual_predictions.values())
        for i in range(len(pred_list)):
            for j in range(i+1, len(pred_list)):
                corr = np.corrcoef(pred_list[i], pred_list[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
                    
        avg_correlation = np.mean(correlations) if correlations else 0.0
        
        # Agreement score (inverse of coefficient of variation)
        agreement_score = 1.0 / (1.0 + cv[0])
        
        return {
            'coefficient_of_variation': float(cv[0]),
            'average_correlation': float(avg_correlation),
            'agreement_score': float(agreement_score),
            'prediction_spread': float(std_pred[0])
        }
    
    def _calculate_ensemble_confidence(self, agreement_metrics, uncertainty):
        """Calculate overall ensemble confidence"""
        
        # Combine agreement and uncertainty into confidence score
        agreement_score = agreement_metrics['agreement_score']
        uncertainty_score = 1.0 / (1.0 + uncertainty)
        
        # Weighted combination
        confidence = 0.6 * agreement_score + 0.4 * uncertainty_score
        
        return float(np.clip(confidence, 0.0, 1.0))
    
    def get_ensemble_diagnostics(self):
        """Get comprehensive ensemble diagnostics"""
        
        diagnostics = {
            'models': list(self.models.keys()),
            'current_weights': self.weights.copy(),
            'performance_history': self.performance_history.copy(),
            'training_history': self.training_history.copy()
        }
        
        # Model complexity analysis
        model_info = {}
        for name, model in self.models.items():
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            model_info[name] = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024)
            }
            
        diagnostics['model_info'] = model_info
        
        return diagnostics
    
    def save_ensemble(self, filepath: str):
        """Save ensemble models and metadata"""
        
        save_data = {
            'models': {},
            'weights': self.weights,
            'performance_history': self.performance_history,
            'training_history': self.training_history,
            'input_size': self.input_size,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save model state dicts
        for name, model in self.models.items():
            save_data['models'][name] = {
                'state_dict': model.state_dict(),
                'model_class': model.__class__.__name__
            }
            
        torch.save(save_data, filepath)
        logger.info(f"Ensemble saved to {filepath}")
        
    def load_ensemble(self, filepath: str):
        """Load ensemble models and metadata"""
        save_data = torch.load(filepath, map_location=self.device)
        # Check input size compatibility
        saved_input_size = save_data.get('input_size', None)
        if saved_input_size is not None and saved_input_size != self.input_size:
            raise ValueError(f"Saved ensemble input size ({saved_input_size}) does not match current input size ({self.input_size})!")
        self.weights = save_data['weights']
        self.performance_history = save_data['performance_history']
        self.training_history = save_data['training_history']
        # Load model state dicts
        for name, model_data in save_data['models'].items():
            if name in self.models:
                self.models[name].load_state_dict(model_data['state_dict'])
        logger.info(f"Ensemble loaded from {filepath}")

class MCDropoutPredictor:
    """Enhanced Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, model, dropout_rate=0.3, n_samples=100):
        self.model = model
        self.dropout_rate = dropout_rate
        self.n_samples = n_samples
        
    def enable_dropout(self):
        """Enable dropout during inference"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
                
    def predict_with_uncertainty(self, x, confidence_levels=[0.90, 0.95, 0.99]):
        """Generate prediction intervals using MC Dropout"""
        
        self.enable_dropout()
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(x)
                predictions.append(pred.cpu().numpy())
                
        predictions = np.array(predictions)
        
        # Calculate statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Prediction intervals
        intervals = {}
        for confidence_level in confidence_levels:
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(predictions, upper_percentile, axis=0)
            
            intervals[f"{int(confidence_level*100)}%"] = {
                'lower': lower_bound.flatten(),
                'upper': upper_bound.flatten(),
                'width': (upper_bound - lower_bound).flatten()
            }
            
        return {
            'prediction': mean_pred.flatten(),
            'uncertainty': std_pred.flatten(),
            'prediction_intervals': intervals,
            'raw_predictions': predictions
        }

if __name__ == "__main__":
    # Test ensemble system
    print("Testing Ensemble System...")
    
    # Create test data
    input_size = 22
    batch_size = 32
    test_input = torch.randn(batch_size, input_size)
    
    # Initialize ensemble
    ensemble = EnsemblePredictor(input_size)
    
    # Test prediction
    result = ensemble.predict_with_ensemble(test_input[0])
    
    print("Ensemble Prediction Results:")
    print(f"Ensemble Prediction: {result['ensemble_prediction']:.6f}")
    print(f"Total Uncertainty: {result['total_uncertainty']:.6f}")
    print(f"Ensemble Confidence: {result['ensemble_confidence']:.3f}")
    print(f"Model Agreement Score: {result['model_agreement']['agreement_score']:.3f}")
    
    print("\nIndividual Model Predictions:")
    for name, pred in result['individual_predictions'].items():
        weight = result['model_weights'][name]
        uncertainty = result['individual_uncertainties'][name]
        print(f"  {name}: {pred:.6f} (weight: {weight:.3f}, uncertainty: {uncertainty:.6f})")
        
    print("\nPrediction Intervals:")
    for level, interval in result['prediction_intervals'].items():
        print(f"  {level}: [{interval['lower']:.6f}, {interval['upper']:.6f}] (width: {interval['width']:.6f})")
        
    print("\nEnsemble system test completed successfully!")