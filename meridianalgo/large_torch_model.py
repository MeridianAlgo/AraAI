"""
Large PyTorch Model - 1M+ parameters for high accuracy
Separate models for stocks and forex with proper data categorization
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
import json


class LargeEnsembleModel(nn.Module):
    """
    Intelligent ensemble model with up to 1M parameters
    Deep architecture with attention and residual connections
    """
    def __init__(self, input_size=44, hidden_sizes=[1024, 768, 512, 384, 256, 128], dropout=0.2):
        super(LargeEnsembleModel, self).__init__()
        
        # Deep feature extraction network
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Multiple prediction heads (ensemble)
        final_hidden = hidden_sizes[-1]
        
        # Primary heads (tree-based model simulation) - larger for more intelligence
        self.xgb_head = nn.Sequential(
            nn.Linear(final_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.lgb_head = nn.Sequential(
            nn.Linear(final_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.rf_head = nn.Sequential(
            nn.Linear(final_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.gb_head = nn.Sequential(
            nn.Linear(final_hidden, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Secondary heads (linear model simulation) - also larger
        self.ridge_head = nn.Sequential(
            nn.Linear(final_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.elastic_head = nn.Sequential(
            nn.Linear(final_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Attention mechanism for ensemble weighting - larger
        self.attention = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # Extract features
        features = self.feature_extractor(x)
        
        # Get predictions from all heads
        preds = torch.stack([
            self.xgb_head(features).squeeze(-1),
            self.lgb_head(features).squeeze(-1),
            self.rf_head(features).squeeze(-1),
            self.gb_head(features).squeeze(-1),
            self.ridge_head(features).squeeze(-1),
            self.elastic_head(features).squeeze(-1)
        ], dim=-1)
        
        # Dynamic attention-based weighting
        weights = self.attention(preds)
        ensemble_pred = (preds * weights).sum(dim=-1)
        
        return ensemble_pred, preds
    
    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdvancedMLSystem:
    """
    Advanced ML system with large models and proper data categorization
    """
    def __init__(self, model_path, model_type='stock', device='cpu'):
        self.model_path = Path(model_path)
        self.model_type = model_type  # 'stock' or 'forex'
        self.device = device
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.metadata = {
            'model_type': model_type,
            'trained_symbols': [],
            'training_history': []
        }
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self):
        """Load model from .pt file"""
        if self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Verify model type matches
                if checkpoint.get('model_type') != self.model_type:
                    print(f"Warning: Model type mismatch. Expected {self.model_type}, got {checkpoint.get('model_type')}")
                    return
                
                self.model = LargeEnsembleModel(
                    input_size=checkpoint.get('input_size', 44),
                    hidden_sizes=checkpoint.get('hidden_sizes', [1024, 768, 512, 384, 256, 128]),
                    dropout=checkpoint.get('dropout', 0.2)
                )
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.model.eval()
                
                self.scaler_mean = checkpoint['scaler_mean']
                self.scaler_std = checkpoint['scaler_std']
                self.metadata = checkpoint.get('metadata', self.metadata)
                
                param_count = self.model.count_parameters()
                print(f"Loaded {self.model_type} model from {self.model_path}")
                print(f"Parameters: {param_count:,}")
                print(f"Training date: {self.metadata.get('training_date', 'Unknown')}")
                print(f"Trained symbols: {len(self.metadata.get('trained_symbols', []))}")
                
            except Exception as e:
                print(f"Could not load model: {e}")
                self.model = None
    
    def _save_model(self):
        """Save model to .pt file"""
        try:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'model_type': self.model_type,
                'input_size': 44,
                'hidden_sizes': [1024, 768, 512, 384, 256, 128],
                'dropout': 0.2,
                'scaler_mean': self.scaler_mean,
                'scaler_std': self.scaler_std,
                'metadata': self.metadata
            }
            
            torch.save(checkpoint, self.model_path)
            param_count = self.model.count_parameters()
            print(f"Saved {self.model_type} model to {self.model_path}")
            print(f"Parameters: {param_count:,}")
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def train(self, X, y, symbol, epochs=200, batch_size=64, lr=0.0005, validation_split=0.2):
        """
        Train the model with validation and early stopping
        
        Args:
            X: Features
            y: Targets
            symbol: Symbol name
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            validation_split: Validation data split
        """
        try:
            print(f"\nTraining {self.model_type} model on {symbol}...")
            print(f"Training samples: {len(X)}")
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            
            # Calculate scaler parameters
            self.scaler_mean = X_tensor.mean(dim=0)
            self.scaler_std = X_tensor.std(dim=0) + 1e-8
            
            # Normalize
            X_normalized = (X_tensor - self.scaler_mean) / self.scaler_std
            
            # Train/validation split
            n_val = int(len(X) * validation_split)
            n_train = len(X) - n_val
            
            X_train = X_normalized[:n_train]
            y_train = y_tensor[:n_train]
            X_val = X_normalized[n_train:]
            y_val = y_tensor[n_train:]
            
            print(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")
            
            # Create model
            self.model = LargeEnsembleModel(
                input_size=X.shape[1],
                hidden_sizes=[1024, 768, 512, 384, 256, 128],
                dropout=0.2
            )
            self.model.to(self.device)
            
            param_count = self.model.count_parameters()
            print(f"Model parameters: {param_count:,}")
            
            # Training setup
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            criterion = nn.MSELoss()
            
            # Training loop with early stopping
            best_val_loss = float('inf')
            patience = 500  # Much higher patience for 2500 steps
            patience_counter = 0
            
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            for epoch in range(epochs):
                # Training
                self.model.train()
                train_loss = 0
                for batch_X, batch_y in train_loader:
                    batch_X = batch_X.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    pred, _ = self.model(batch_X)
                    loss = criterion(pred, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                
                train_loss /= len(train_loader)
                
                # Validation
                self.model.eval()
                with torch.no_grad():
                    X_val_device = X_val.to(self.device)
                    y_val_device = y_val.to(self.device)
                    val_pred, _ = self.model(X_val_device)
                    val_loss = criterion(val_pred, y_val_device).item()
                
                scheduler.step(val_loss)
                
                # Print progress
                if (epoch + 1) % 100 == 0 or (epoch + 1) % 20 == 0 and (epoch + 1) <= 100:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Update metadata
            if symbol not in self.metadata['trained_symbols']:
                self.metadata['trained_symbols'].append(symbol)
            
            self.metadata['training_date'] = datetime.now().isoformat()
            self.metadata['last_symbol'] = symbol
            self.metadata['data_points'] = len(X)
            self.metadata['best_val_loss'] = best_val_loss
            self.metadata['training_history'].append({
                'symbol': symbol,
                'date': datetime.now().isoformat(),
                'samples': len(X),
                'val_loss': best_val_loss
            })
            
            # Save model
            self._save_model()
            
            print(f"\nTraining complete!")
            print(f"Best validation loss: {best_val_loss:.6f}")
            print(f"Total symbols trained: {len(self.metadata['trained_symbols'])}")
            
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
            
            # Return as numpy arrays
            pred_np = pred.cpu().numpy()
            individual_np = individual_preds.cpu().numpy()
            
            # Ensure proper shape
            if pred_np.ndim == 0:
                pred_np = np.array([pred_np])
            
            return pred_np, individual_np
    
    def is_trained(self):
        """Check if model is trained"""
        return self.model is not None
    
    def get_metadata(self):
        """Get training metadata"""
        return self.metadata
    
    def can_predict_symbol(self, symbol):
        """Check if model can predict this symbol"""
        # Model can predict any symbol of the same type (stock/forex)
        # But warn if symbol wasn't in training set
        if symbol not in self.metadata.get('trained_symbols', []):
            print(f"Warning: {symbol} was not in training set. Predictions may be less accurate.")
        return True
