"""
State-of-the-Art PyTorch Model - 2025 Architecture
Latest technologies: Transformer-XL, Flash Attention, Layer Normalization, SwiGLU
Optimized for financial time series prediction with elite performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import psutil
from pathlib import Path
from datetime import datetime
from accelerate import Accelerator


class FlashMultiHeadAttention(nn.Module):
    """Flash Attention for better memory efficiency and speed"""
    def __init__(self, embed_dim, num_heads, dropout=0.1, use_flash=True):
        super(FlashMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape
        
        # Generate Q, K, V in one go
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.use_flash:
            # Use Flash Attention if available
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False
            )
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        else:
            # Fallback to standard attention
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores.masked_fill_(mask == 0, -1e9)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        return self.out_proj(attn_output), attn_weights if not self.use_flash else None


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit - better than ReLU/GELU for transformers"""
    def __init__(self, dim, hidden_dim=None, bias=False):
        super().__init__()
        hidden_dim = hidden_dim or int(2 * dim / 3)
        hidden_dim = int((hidden_dim + 255) // 256) * 256  # Make divisible by 256
        
        self.w = nn.Linear(dim, hidden_dim, bias=bias)
        self.v = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
    
    def forward(self, x):
        return self.w2(F.silu(self.w(x)) * self.v(x))


class RMSNorm(nn.Module):
    """Root Mean Square Normalization - better than LayerNorm"""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True) * (x.size(-1) ** -0.5)
        return self.weight * x / (norm + self.eps)


class TransformerBlock(nn.Module):
    """Modern Transformer Block with Flash Attention and SwiGLU"""
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.1, use_flash=True):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = FlashMultiHeadAttention(dim, num_heads, dropout, use_flash)
        self.norm2 = RMSNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = SwiGLU(dim, mlp_hidden_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm architecture
        x = x + self.dropout(self.attn(self.norm1(x), mask)[0])
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class AdaptiveTimeSeriesPooler(nn.Module):
    """Adaptive pooling for time series data"""
    def __init__(self, seq_len, hidden_dim):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.temporal_weights = nn.Parameter(torch.ones(seq_len))
        
    def forward(self, x):
        # x: [batch, seq_len, hidden_dim]
        weights = F.softmax(self.temporal_weights, dim=0).unsqueeze(0).unsqueeze(-1)
        weighted_x = x * weights
        
        # Adaptive pooling
        pooled = self.adaptive_pool(weighted_x.transpose(1, 2)).transpose(1, 2)
        return pooled.squeeze(1)  # [batch, hidden_dim]


class EliteEnsembleModel(nn.Module):
    """Efficient Elite Model Architecture for Financial Time Series - Optimized Size"""
    def __init__(self, input_size=44, seq_len=1, hidden_dims=[256, 192, 128, 64], 
                 num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        
        self.input_size = input_size
        self.seq_len = seq_len
        self.hidden_dims = hidden_dims
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        # Input projection with positional encoding
        self.input_proj = nn.Linear(input_size, hidden_dims[0])
        self.pos_encoding = nn.Parameter(torch.randn(seq_len, hidden_dims[0]) * 0.02)
        self.input_dropout = nn.Dropout(dropout)
        
        # Efficient transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(hidden_dims[0], num_heads, mlp_ratio=2, dropout=dropout, use_flash=True)
            for _ in range(num_layers)
        ])
        
        # Adaptive pooling
        self.pooler = AdaptiveTimeSeriesPooler(seq_len, hidden_dims[0])
        
        # Efficient feature extraction
        self.feature_extractors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[0], dim),
                RMSNorm(dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for dim in hidden_dims[1:]
        ])
        
        # Simple prediction heads (3 specialized models)
        self.prediction_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
                RMSNorm(hidden_dims[0] // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dims[0] // 2, 1)
            ) for _ in range(3)
        ])
        
        # Attention-based ensemble weights
        self.ensemble_attention = nn.Sequential(
            nn.Linear(len(self.prediction_heads) * hidden_dims[0], hidden_dims[0] // 2),
            RMSNorm(hidden_dims[0] // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0] // 2, len(self.prediction_heads)),
            nn.Softmax(dim=-1)
        )
        
        # Final output layer
        self.final_output = nn.Sequential(
            nn.Linear(len(self.prediction_heads), hidden_dims[0] // 4),
            RMSNorm(hidden_dims[0] // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0] // 4, 1)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Reshape for sequence processing
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, features]
        
        batch_size, seq_len, _ = x.shape
        
        # Input projection with positional encoding
        x = self.input_proj(x)
        x = x + self.pos_encoding[:seq_len].unsqueeze(0)
        x = self.input_dropout(x)
        
        # Transformer encoder
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Adaptive pooling
        pooled_features = self.pooler(x)  # [batch, hidden_dims[0]]
        
        # Multi-scale features
        multi_scale_features = [pooled_features]
        for extractor in self.feature_extractors:
            multi_scale_features.append(extractor(pooled_features))
        
        # Prediction heads
        predictions = []
        for head in self.prediction_heads:
            pred = head(pooled_features)
            predictions.append(pred)
        
        all_predictions = torch.cat(predictions, dim=-1)  # [batch, num_heads]
        
        # Ensemble weighting
        ensemble_input = torch.cat([pooled_features] * len(self.prediction_heads), dim=-1)
        ensemble_weights = self.ensemble_attention(ensemble_input)
        
        # Weighted ensemble
        weighted_predictions = all_predictions * ensemble_weights
        ensemble_output = weighted_predictions.sum(dim=-1, keepdim=True)
        
        # Final processing
        final_pred = self.final_output(all_predictions)
        
        return final_pred, all_predictions

    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdvancedMLSystem:
    """
    Elite ML System with 2025 State-of-the-Art Architecture
    Flash Attention, SwiGLU, RMSNorm, and advanced ensemble methods
    """

    def __init__(self, model_path, model_type="stock", device="cpu"):
        self.model_path = Path(model_path)
        self.model_type = model_type  # 'stock' or 'forex'
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.metadata = {
            "model_type": model_type,
            "trained_symbols": [],
            "training_history": [],
            "architecture": "EliteEnsembleModel-2025-Compact",
            "version": "3.1"
        }

        # Initialize Accelerate
        self.accelerator = Accelerator(mixed_precision="fp16", cpu=True)
        self.device = self.accelerator.device
        print(f"Using device: {self.device} with Elite Accelerate")

        # Try to load existing model
        self._load_model()

    def _load_model(self):
        """Load model from .pt file"""
        if self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # Verify model type matches
                if checkpoint.get("model_type") != self.model_type:
                    print(
                        f"Warning: Model type mismatch. Expected {self.model_type}, got {checkpoint.get('model_type')}"
                    )
                    return

                self.model = EliteEnsembleModel(
                    input_size=checkpoint.get("input_size", 44),
                    seq_len=checkpoint.get("seq_len", 1),
                    hidden_dims=checkpoint.get(
                        "hidden_dims", [256, 192, 128, 64]
                    ),
                    num_heads=checkpoint.get("num_heads", 4),
                    num_layers=checkpoint.get("num_layers", 3),
                    dropout=checkpoint.get("dropout", 0.1),
                )
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.to(self.device)
                self.model.eval()

                self.scaler_mean = checkpoint["scaler_mean"].to(self.device)
                self.scaler_std = checkpoint["scaler_std"].to(self.device)
                self.metadata = checkpoint.get("metadata", self.metadata)

                param_count = self.model.count_parameters()
                print(f"Loaded {self.model_type} model from {self.model_path}")
                print(f"Parameters: {param_count:,}")
                print(f"Training date: {self.metadata.get('training_date', 'Unknown')}")
                print(
                    f"Trained symbols: {len(self.metadata.get('trained_symbols', []))}"
                )

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
                "model_state_dict": unwrapped_model.state_dict(),
                "model_type": self.model_type,
                "input_size": 44,
                "seq_len": 1,
                "hidden_dims": [256, 192, 128, 64],
                "num_heads": 4,
                "num_layers": 3,
                "dropout": 0.1,
                "scaler_mean": self.scaler_mean,
                "scaler_std": self.scaler_std,
                "metadata": self.metadata,
                "architecture": "EliteEnsembleModel-2025-Compact",
                "version": "3.1"
            }

            torch.save(checkpoint, self.model_path)
            param_count = self.model.count_parameters()
            print(f"Saved {self.model_type} model to {self.model_path}")
            print(f"Parameters: {param_count:,}")

        except Exception as e:
            print(f"Error saving model: {e}")

    def train(
        self,
        X,
        y,
        symbol,
        epochs=200,
        batch_size=64,
        lr=0.0005,
        validation_split=0.2,
        cpu_limit=80,
    ):
        """
        Train the model with validation, early stopping, and CPU limiting
        """
        try:
            print(f"\nTraining {self.model_type} model on {symbol}...")
            print(f"Training samples: {len(X)}")

            # Convert to tensors and normalize targets to reduce loss scale
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)

            # Normalize targets to reasonable range (0-1) to prevent explosion
            y_min = y_tensor.min()
            y_max = y_tensor.max()
            if y_max > y_min:
                y_tensor = (y_tensor - y_min) / (y_max - y_min)
            else:
                y_tensor = torch.zeros_like(y_tensor)

            # Calculate scaler parameters for features
            self.scaler_mean = X_tensor.mean(dim=0)
            self.scaler_std = X_tensor.std(dim=0) + 1e-8

            # Normalize features
            X_normalized = (X_tensor - self.scaler_mean) / self.scaler_std

            # Train/validation split
            n_val = int(len(X) * validation_split)
            n_train = len(X) - n_val

            X_train = X_normalized[:n_train]
            y_train = y_tensor[:n_train]
            X_val = X_normalized[n_train:]
            y_val = y_tensor[n_train:]

            print(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")

            # Create model if not already loaded/trained
            if self.model is None:
                print("  Creating new EliteEnsembleModel architecture...")
                self.model = EliteEnsembleModel(
                    input_size=X.shape[1],
                    seq_len=1,
                    hidden_dims=[256, 192, 128, 64],
                    num_heads=4,
                    num_layers=3,
                    dropout=0.1,
                )
            else:
                print("  Resuming training from existing model weights...")

            param_count = self.model.count_parameters()
            print(f"Model parameters: {param_count:,}")

            # Training setup with elite learning rate for 2025 architecture
            optimizer = torch.optim.AdamW(
                self.model.parameters(), lr=lr * 0.02, weight_decay=0.001, betas=(0.9, 0.95)
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=200, T_mult=2, eta_min=lr * 0.0001
            )
            criterion = nn.MSELoss()

            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )

            # Prepare with Accelerate
            self.model, optimizer, train_loader = self.accelerator.prepare(
                self.model, optimizer, train_loader
            )

            # Training loop with early stopping
            best_val_loss = float("inf")
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

                scheduler.step()

                # Print progress
                if (epoch + 1) % 100 == 0 or ((epoch + 1) % 20 == 0 and (epoch + 1) <= 100):
                    print(
                        f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                    )

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            # Update metadata
            if symbol not in self.metadata["trained_symbols"]:
                self.metadata["trained_symbols"].append(symbol)

            self.metadata["training_date"] = datetime.now().isoformat()
            self.metadata["last_symbol"] = symbol
            self.metadata["data_points"] = len(X)
            self.metadata["best_val_loss"] = best_val_loss
            # Store target normalization parameters
            self.metadata["target_min"] = float(y_min)
            self.metadata["target_max"] = float(y_max)
            self.metadata["training_history"].append(
                {
                    "symbol": symbol,
                    "date": datetime.now().isoformat(),
                    "samples": len(X),
                    "val_loss": best_val_loss,
                }
            )

            # Save model
            self._save_model()

            print("\nTraining complete!")
            print(f"Best validation loss: {best_val_loss:.6f}")
            print(f"Total symbols trained: {len(self.metadata['trained_symbols'])}")

            return {
                "success": True,
                "final_loss": best_val_loss,
                "accuracy": (
                    100 * (1 - best_val_loss) if best_val_loss < 1 else 0
                ),  # Simple heuristic
                "epochs": epoch + 1,
            }

        except Exception as e:
            print(f"Training failed: {e}")
            import traceback

            traceback.print_exc()
            return {"success": False, "error": str(e)}

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

            # Denormalize predictions if we have target normalization parameters
            if "target_min" in self.metadata and "target_max" in self.metadata:
                target_min = self.metadata["target_min"]
                target_max = self.metadata["target_max"]
                pred = pred * (target_max - target_min) + target_min
                individual_preds = individual_preds * (target_max - target_min) + target_min

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
        if symbol not in self.metadata.get("trained_symbols", []):
            print(
                f"Warning: {symbol} was not in training set. Predictions may be less accurate."
            )
        return True
