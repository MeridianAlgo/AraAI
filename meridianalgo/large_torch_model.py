"""
Large PyTorch Model - 1M+ parameters for high accuracy
Separate models for stocks and forex with proper data categorization
Optimized with Hugging Face Accelerate and CPU limiting
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
import time
import psutil
from pathlib import Path
from datetime import datetime
from accelerate import Accelerator

class ResidualBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout)
        
        # Shortcut connection if dimensions change
        self.shortcut = nn.Sequential()
        if in_features != out_features:
            self.shortcut = nn.Sequential(
                nn.Linear(in_features, out_features),
                nn.BatchNorm1d(out_features)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.linear1(x)))
        out = self.dropout(out)
        out = self.bn2(self.linear2(out))
        out += residual
        out = F.relu(out)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value):
        batch_size = query.size(0)
        
        # Linear projections
        q = self.q_linear(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled Dot-Product Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        return self.out_proj(output), attn_weights

class PredictionHead(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.2):
        super(PredictionHead, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
            
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class LargeEnsembleModel(nn.Module):
    def __init__(self, input_size=44, hidden_sizes=[1024, 768, 512, 384, 256, 128], dropout=0.2):
        super(LargeEnsembleModel, self).__init__()
        
        # Input projection with residual
        self.input_proj = nn.Linear(input_size, hidden_sizes[0])
        self.input_norm = nn.LayerNorm(hidden_sizes[0])
        
        # Deep feature extraction with residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.residual_blocks.append(
                ResidualBlock(hidden_sizes[i], hidden_sizes[i+1], dropout)
            )
        
        # Multi-head attention for feature importance
        final_hidden = hidden_sizes[-1]
        self.attention = MultiHeadAttention(final_hidden, num_heads=4, dropout=dropout)
        
        # Multiple specialized prediction heads (ensemble)
        # 1. XGBoost-style head (Gradient Boosting focus)
        self.xgb_head = PredictionHead(final_hidden, [64, 32], dropout)
        
        # 2. LightGBM-style head (Leaf-wise focus)
        self.lgb_head = PredictionHead(final_hidden, [64, 32], dropout)
        
        # 3. Random Forest-style head (Bagging focus)
        self.rf_head = PredictionHead(final_hidden, [64, 32], dropout)
        
        # 4. Gradient Boosting-style head (Residual focus)
        self.gb_head = PredictionHead(final_hidden, [64, 32], dropout)
        
        # 5. Ridge Regression-style head (L2 focus)
        self.ridge_head = nn.Sequential(
            nn.Linear(final_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # 6. Elastic Net-style head (L1+L2 focus)
        self.elastic_head = nn.Sequential(
            nn.Linear(final_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Attention weights for ensemble
        self.attention_weights = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 6),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = F.relu(x)
        
        # Deep residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Attention mechanism
        x, _ = self.attention(x, x, x)
        if x.dim() == 3:
            x = x.squeeze(1)
        
        # Get predictions from all heads
        pred_xgb = self.xgb_head(x)
        pred_lgb = self.lgb_head(x)
        pred_rf = self.rf_head(x)
        pred_gb = self.gb_head(x)
        pred_ridge = self.ridge_head(x)
        pred_elastic = self.elastic_head(x)
        
        # Stack predictions
        predictions = torch.cat([
            pred_xgb, pred_lgb, pred_rf, pred_gb, pred_ridge, pred_elastic
        ], dim=1)
        
        # Debug shapes if something looks wrong
        if predictions.shape[1] != 6:
            print(f"DEBUG: predictions shape: {predictions.shape}")
            print(f"DEBUG: pred_xgb shape: {pred_xgb.shape}")
        
        # Calculate ensemble weights
        weights = self.attention_weights(predictions)
        
        # Weighted average
        final_pred = torch.sum(predictions * weights, dim=1, keepdim=True)
        
        return final_pred, predictions

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
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.metadata = {
            'model_type': model_type,
            'trained_symbols': [],
            'training_history': []
        }
        
        # Initialize Accelerator
        self.accelerator = Accelerator(mixed_precision="fp16", cpu=True)
        self.device = self.accelerator.device
        print(f"Using device: {self.device} with Accelerate")
        
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
                
                self.scaler_mean = checkpoint['scaler_mean'].to(self.device)
                self.scaler_std = checkpoint['scaler_std'].to(self.device)
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
            
            # Unwrap model for saving
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            
            checkpoint = {
                'model_state_dict': unwrapped_model.state_dict(),
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
    
    def train(self, X, y, symbol, epochs=200, batch_size=64, lr=0.0005, validation_split=0.2, cpu_limit=80):
        """
        Train the model with validation, early stopping, and CPU limiting
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
            
            param_count = self.model.count_parameters()
            print(f"Model parameters: {param_count:,}")
            
            # Training setup
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            criterion = nn.MSELoss()
            
            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            
            # Prepare with Accelerate
            self.model, optimizer, train_loader = self.accelerator.prepare(
                self.model, optimizer, train_loader
            )
            
            # Training loop with early stopping
            best_val_loss = float('inf')
            patience = 500
            patience_counter = 0
            
            for epoch in range(epochs):
                self.model.train()
                train_loss = 0
                
                for batch_X, batch_y in train_loader:
                    # CPU Limiter - Check every batch for smoother control
                    if psutil.cpu_percent(interval=None) > cpu_limit:
                        time.sleep(0.05)  # Short sleep to let CPU cool down
                        
                    optimizer.zero_grad()
                    pred, _ = self.model(batch_X)
                    loss = criterion(pred, batch_y)
                    self.accelerator.backward(loss)
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
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
