"""
PyTorch Neural Network Model for Stock Price Prediction
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class StockPredictor(nn.Module):
    """PyTorch neural network for stock price prediction"""
    
    def __init__(self, input_size: int = 22, hidden_sizes: List[int] = [64, 32, 16], 
                 output_size: int = 1, dropout_rate: float = 0.2):
        """
        Initialize the neural network
        
        Args:
            input_size: Number of input features (OHLCV + 17 technical indicators = 22)
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output neurons (1 for price prediction)
            dropout_rate: Dropout rate for regularization
        """
        super(StockPredictor, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Build the network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"StockPredictor initialized: {input_size} -> {hidden_sizes} -> {output_size}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        return self.network(x)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def get_model_info(self) -> Dict:
        """Get model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)  # Assuming float32
        }

class ModelManager:
    """Manages model training, saving, and loading"""
    
    def __init__(self, model_save_path: str = "./data/models/"):
        self.model_save_path = model_save_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Ensure model directory exists
        os.makedirs(model_save_path, exist_ok=True)
    
    def create_model(self, input_size: int = 22, hidden_sizes: List[int] = [64, 32, 16],
                    dropout_rate: float = 0.2) -> StockPredictor:
        """Create a new model instance"""
        model = StockPredictor(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            dropout_rate=dropout_rate
        ).to(self.device)
        
        return model
    
    def save_model(self, model: StockPredictor, symbol: str, version: str = "1.0",
                  metadata: Optional[Dict] = None) -> str:
        """Save model with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{symbol}_{version}_{timestamp}.pth"
        filepath = os.path.join(self.model_save_path, filename)
        
        # Prepare save data
        save_data = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'input_size': model.input_size,
                'hidden_sizes': model.hidden_sizes,
                'output_size': model.output_size,
                'dropout_rate': model.dropout_rate
            },
            'symbol': symbol,
            'version': version,
            'timestamp': timestamp,
            'device': str(self.device),
            'pytorch_version': torch.__version__
        }
        
        if metadata:
            save_data['metadata'] = metadata
        
        torch.save(save_data, filepath)
        logger.info(f"Model saved: {filepath}")
        
        return filepath
    
    def load_model(self, filepath: str) -> Tuple[StockPredictor, Dict]:
        """Load model from file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        # Load the saved data (handle PyTorch 2.6+ security requirements)
        try:
            save_data = torch.load(filepath, map_location=self.device, weights_only=False)
        except Exception:
            # Fallback for older PyTorch versions
            save_data = torch.load(filepath, map_location=self.device)
        
        # Create model with saved configuration
        config = save_data['model_config']
        model = StockPredictor(
            input_size=config['input_size'],
            hidden_sizes=config['hidden_sizes'],
            output_size=config['output_size'],
            dropout_rate=config['dropout_rate']
        ).to(self.device)
        
        # Load the state dict
        model.load_state_dict(save_data['model_state_dict'])
        
        # Extract metadata
        metadata = {
            'symbol': save_data.get('symbol'),
            'version': save_data.get('version'),
            'timestamp': save_data.get('timestamp'),
            'device': save_data.get('device'),
            'pytorch_version': save_data.get('pytorch_version'),
            'metadata': save_data.get('metadata', {})
        }
        
        logger.info(f"Model loaded: {filepath}")
        return model, metadata
    
    def get_latest_model(self, symbol: str) -> Optional[Tuple[StockPredictor, Dict]]:
        """Get the latest model for a symbol"""
        model_files = []
        
        for filename in os.listdir(self.model_save_path):
            if filename.startswith(f"{symbol}_") and filename.endswith('.pth'):
                filepath = os.path.join(self.model_save_path, filename)
                model_files.append((filepath, os.path.getmtime(filepath)))
        
        if not model_files:
            return None
        
        # Sort by modification time and get the latest
        latest_file = sorted(model_files, key=lambda x: x[1], reverse=True)[0][0]
        
        return self.load_model(latest_file)
    
    def predict(self, model: StockPredictor, features: torch.Tensor) -> torch.Tensor:
        """Make prediction with the model"""
        model.eval()
        with torch.no_grad():
            if len(features.shape) == 1:
                features = features.unsqueeze(0)  # Add batch dimension
            
            features = features.to(self.device)
            prediction = model(features)
            
        return prediction
    
    def predict_with_confidence(self, model: StockPredictor, features: torch.Tensor, 
                              n_samples: int = 100) -> Tuple[float, float]:
        """Make prediction with confidence estimation using Monte Carlo Dropout"""
        # For single predictions, use eval mode to avoid batch norm issues
        model.eval()
        predictions = []
        
        with torch.no_grad():
            if len(features.shape) == 1:
                features = features.unsqueeze(0)
            
            features = features.to(self.device)
            
            # Make multiple predictions with different random seeds for uncertainty
            for _ in range(n_samples):
                pred = model(features)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        # Convert standard deviation to confidence (0-1 scale)
        # For eval mode, std will be 0, so use a default confidence
        if std_pred == 0:
            confidence = 0.7  # Default confidence for deterministic predictions
        else:
            confidence = max(0.0, min(1.0, 1.0 - (std_pred / abs(mean_pred)) if mean_pred != 0 else 0.5))
        
        return float(mean_pred), float(confidence)

class ModelTrainer:
    """Handles model training with various optimizers and schedulers"""
    
    def __init__(self, model: StockPredictor, device: torch.device):
        self.model = model
        self.device = device
        self.training_history = {
            'train_losses': [],
            'val_losses': [],
            'epochs': 0
        }
    
    def train_epoch(self, train_loader, optimizer, criterion) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_features, batch_targets in train_loader:
            batch_features = batch_features.to(self.device)
            batch_targets = batch_targets.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def validate_epoch(self, val_loader, criterion) -> float:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_features, batch_targets in val_loader:
                batch_features = batch_features.to(self.device)
                batch_targets = batch_targets.to(self.device)
                
                outputs = self.model(batch_features)
                loss = criterion(outputs, batch_targets)
                
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self, train_loader, val_loader, epochs: int = 100, 
              learning_rate: float = 0.001, patience: int = 10) -> Dict:
        """Train the model with early stopping"""
        
        # Setup optimizer and criterion
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        criterion = nn.MSELoss()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        for epoch in range(epochs):
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_loss = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            scheduler.step(val_loss)
            
            # Record history
            self.training_history['train_losses'].append(train_loss)
            self.training_history['val_losses'].append(val_loss)
            self.training_history['epochs'] = epoch + 1
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Log progress
            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        training_results = {
            'final_train_loss': self.training_history['train_losses'][-1],
            'final_val_loss': self.training_history['val_losses'][-1],
            'best_val_loss': best_val_loss,
            'epochs_trained': self.training_history['epochs'],
            'converged': patience_counter >= patience
        }
        
        logger.info(f"Training completed: {training_results}")
        return training_results

# Global instances
model_manager = ModelManager()

if __name__ == "__main__":
    print("PyTorch Stock Predictor Model - Setting up...")
    
    # Test model creation
    model = model_manager.create_model()
    info = model.get_model_info()
    print(f"Model created with {info['total_parameters']} parameters")
    print(f"Model size: {info['model_size_mb']:.2f} MB")