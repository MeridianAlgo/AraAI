"""
PyTorch Ensemble Model - Single .pt file for stocks and forex
Combines all 9 models into one PyTorch model
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

class EnsembleModel(nn.Module):
    """
    PyTorch ensemble combining 9 ML models
    """
    def __init__(self, input_size=44, hidden_size=128):
        super(EnsembleModel, self).__init__()
        
        # Shared feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 9 model heads (simulating XGBoost, LightGBM, RF, ET, GB, AdaBoost, Ridge, Elastic, Lasso)
        self.xgb_head = nn.Linear(hidden_size, 1)
        self.lgb_head = nn.Linear(hidden_size, 1)
        self.rf_head = nn.Linear(hidden_size, 1)
        self.et_head = nn.Linear(hidden_size, 1)
        self.gb_head = nn.Linear(hidden_size, 1)
        self.adaboost_head = nn.Linear(hidden_size, 1)
        self.ridge_head = nn.Linear(hidden_size, 1)
        self.elastic_head = nn.Linear(hidden_size, 1)
        self.lasso_head = nn.Linear(hidden_size, 1)
        
        # Model weights
        self.model_weights = torch.tensor([0.25, 0.20, 0.15, 0.10, 0.10, 0.08, 0.04, 0.04, 0.04])
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Get predictions from all heads
        preds = torch.stack([
            self.xgb_head(features).squeeze(),
            self.lgb_head(features).squeeze(),
            self.rf_head(features).squeeze(),
            self.et_head(features).squeeze(),
            self.gb_head(features).squeeze(),
            self.adaboost_head(features).squeeze(),
            self.ridge_head(features).squeeze(),
            self.elastic_head(features).squeeze(),
            self.lasso_head(features).squeeze()
        ], dim=-1)
        
        # Weighted ensemble
        weights = self.model_weights.to(x.device)
        ensemble_pred = (preds * weights).sum(dim=-1)
        
        return ensemble_pred, preds


class TorchMLSystem:
    """
    PyTorch-based ML system with single .pt model file
    """
    def __init__(self, model_path, device='cpu'):
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.metadata = {}
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self):
        """Load model from .pt file"""
        if self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                self.model = EnsembleModel(
                    input_size=checkpoint.get('input_size', 44),
                    hidden_size=checkpoint.get('hidden_size', 128)
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                self.scaler_mean = checkpoint['scaler_mean']
                self.scaler_std = checkpoint['scaler_std']
                self.metadata = checkpoint.get('metadata', {})
                
                print(f"Loaded model from {self.model_path}")
                print(f"Training date: {self.metadata.get('training_date', 'Unknown')}")
                print(f"Trained on: {self.metadata.get('symbol', 'Unknown')}")
                
            except Exception as e:
                print(f"Could not load model: {e}")
                self.model = None
    
    def _save_model(self):
        """Save model to .pt file"""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'input_size': 44,
                'hidden_size': 128,
                'scaler_mean': self.scaler_mean,
                'scaler_std': self.scaler_std,
                'metadata': self.metadata
            }
            
            torch.save(checkpoint, self.model_path)
            print(f"Saved model to {self.model_path}")
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def train(self, X, y, symbol, epochs=100, batch_size=32, lr=0.001):
        """Train the ensemble model"""
        try:
            print(f"Training PyTorch ensemble on {symbol}...")
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            
            # Calculate scaler parameters
            self.scaler_mean = X_tensor.mean(dim=0)
            self.scaler_std = X_tensor.std(dim=0) + 1e-8
            
            # Normalize
            X_normalized = (X_tensor - self.scaler_mean) / self.scaler_std
            
            # Create model
            self.model = EnsembleModel(input_size=X.shape[1], hidden_size=128)
            self.model.to(self.device)
            
            # Training setup
            optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            # Training loop
            self.model.train()
            dataset = torch.utils.data.TensorDataset(X_normalized, y_tensor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            for epoch in range(epochs):
                total_loss = 0
                for batch_X, batch_y in dataloader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    pred, _ = self.model(batch_X)
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 20 == 0:
                    avg_loss = total_loss / len(dataloader)
                    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            # Update metadata
            self.metadata = {
                'training_date': datetime.now().isoformat(),
                'symbol': symbol,
                'data_points': len(X),
                'epochs': epochs,
                'input_size': X.shape[1]
            }
            
            # Save model
            self._save_model()
            
            print("Training complete!")
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            X_normalized = (X_tensor - self.scaler_mean) / self.scaler_std
            X_normalized = X_normalized.to(self.device)
            
            pred, individual_preds = self.model(X_normalized)
            
            # Return as numpy arrays, ensure proper shape
            pred_np = pred.cpu().numpy()
            individual_np = individual_preds.cpu().numpy()
            
            # If single prediction, ensure it's a 1D array
            if pred_np.ndim == 0:
                pred_np = np.array([pred_np])
            
            return pred_np, individual_np
    
    def is_trained(self):
        """Check if model is trained"""
        return self.model is not None
    
    def get_metadata(self):
        """Get training metadata"""
        return self.metadata
