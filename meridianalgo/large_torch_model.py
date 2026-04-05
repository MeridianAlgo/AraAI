"""
Revolutionary 2026 PyTorch Model Architecture
Latest technologies: Mamba SSM, RoPE, GQA, MoE, SwiGLU, RMSNorm, Flash Attention 2
Optimized for financial time series prediction with revolutionary performance
"""

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator

# Import revolutionary model
try:
    from .revolutionary_model import RevolutionaryFinancialModel

    REVOLUTIONARY_MODEL_AVAILABLE = True
except ImportError:
    REVOLUTIONARY_MODEL_AVAILABLE = False
    print("Warning: Revolutionary model not available, using fallback")


class FlashMultiHeadAttention(nn.Module):
    """Flash Attention for better memory efficiency and speed"""

    def __init__(self, embed_dim, num_heads, dropout=0.1, use_flash=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.shape

        # Generate Q, K, V in one go
        qkv = self.qkv_proj(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_flash:
            # Use Flash Attention if available
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
            attn_output = (
                attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
            )
        else:
            # Fallback to standard attention
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores.masked_fill_(mask == 0, -1e9)
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
            attn_output = (
                attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
            )

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

    def __init__(
        self,
        input_size=44,
        seq_len=1,
        hidden_dims=None,
        num_heads=4,
        num_layers=3,
        dropout=0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 192, 128, 64]

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
        self.transformer_layers = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_dims[0],
                    num_heads,
                    mlp_ratio=2,
                    dropout=dropout,
                    use_flash=True,
                )
                for _ in range(num_layers)
            ]
        )

        # Adaptive pooling
        self.pooler = AdaptiveTimeSeriesPooler(seq_len, hidden_dims[0])

        # Efficient feature extraction
        self.feature_extractors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dims[0], dim),
                    RMSNorm(dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for dim in hidden_dims[1:]
            ]
        )

        # Simple prediction heads (3 specialized models)
        self.prediction_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dims[0], hidden_dims[0] // 2),
                    RMSNorm(hidden_dims[0] // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dims[0] // 2, 1),
                )
                for _ in range(3)
            ]
        )

        # Attention-based ensemble weights
        self.ensemble_attention = nn.Sequential(
            nn.Linear(len(self.prediction_heads) * hidden_dims[0], hidden_dims[0] // 2),
            RMSNorm(hidden_dims[0] // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0] // 2, len(self.prediction_heads)),
            nn.Softmax(dim=-1),
        )

        # Final output layer
        self.final_output = nn.Sequential(
            nn.Linear(len(self.prediction_heads), hidden_dims[0] // 4),
            RMSNorm(hidden_dims[0] // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0] // 4, 1),
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
        _ = weighted_predictions.sum(dim=-1, keepdim=True)  # ensemble_output not used

        # Final processing
        final_pred = self.final_output(all_predictions)

        return final_pred, all_predictions

    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class AdvancedMLSystem:
    """
    Revolutionary ML System with 2026 State-of-the-Art Architecture
    Mamba SSM, RoPE, GQA, MoE, SwiGLU, RMSNorm, Flash Attention 2
    """

    def __init__(self, model_path, model_type="stock", device="cpu", use_revolutionary=True):
        self.model_path = Path(model_path)
        self.model_type = model_type  # 'stock' or 'forex'
        self.use_revolutionary = use_revolutionary and REVOLUTIONARY_MODEL_AVAILABLE
        self.model = None
        self.scaler_mean = None
        self.scaler_std = None
        self.metadata = {
            "model_type": model_type,
            "trained_symbols": [],
            "training_history": [],
            "architecture": (
                "RevolutionaryFinancialModel-2026"
                if self.use_revolutionary
                else "EliteEnsembleModel-2025-Compact"
            ),
            "version": "4.1" if self.use_revolutionary else "3.1",
        }

        # Initialize Accelerate
        self.accelerator = Accelerator(mixed_precision="fp16", cpu=True)
        self.device = self.accelerator.device
        print(
            f"Using device: {self.device} with {'Revolutionary 2026' if self.use_revolutionary else 'Elite'} Architecture"
        )

        # Try to load existing model
        self._load_model()

    def _load_model(self):
        """Load model from .pt file - only loads v4.1+ models, skips old ones"""
        if self.model_path.exists():
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)

                # Verify model type matches
                if checkpoint.get("model_type") != self.model_type:
                    print(
                        f"Warning: Model type mismatch. Expected {self.model_type}, "
                        f"got {checkpoint.get('model_type')}. Training fresh."
                    )
                    return

                # Skip old model versions - train fresh with new architecture
                version = checkpoint.get("version", "0")
                if str(version) < "4.1":
                    print(
                        f"  Skipping old model (v{version}) - training fresh with v4.1 architecture"
                    )
                    self.model = None
                    return

                model_state_dict = checkpoint.get("model_state_dict")
                if not isinstance(model_state_dict, dict) or not model_state_dict:
                    print(f"Could not load model: missing model_state_dict in {self.model_path}")
                    self.model = None
                    return

                architecture = checkpoint.get("architecture")

                # Check if this is a v4.1 revolutionary model
                if (
                    architecture == "RevolutionaryFinancialModel-2026"
                    and REVOLUTIONARY_MODEL_AVAILABLE
                ):
                    self.model = RevolutionaryFinancialModel(
                        input_size=int(checkpoint.get("input_size", 44)),
                        seq_len=int(checkpoint.get("seq_len", 30)),
                        dim=int(checkpoint.get("dim", 512)),
                        num_layers=int(checkpoint.get("num_layers", 6)),
                        num_heads=int(checkpoint.get("num_heads", 8)),
                        num_kv_heads=int(checkpoint.get("num_kv_heads", 2)),
                        num_experts=int(checkpoint.get("num_experts", 6)),
                        num_prediction_heads=int(checkpoint.get("num_prediction_heads", 6)),
                        dropout=float(checkpoint.get("dropout", 0.1)),
                        use_mamba=bool(checkpoint.get("use_mamba", True)),
                    )
                    self.use_revolutionary = True
                elif architecture == "EliteEnsembleModel-2025-Compact":
                    hidden_dims = checkpoint.get("hidden_dims", [256, 192, 128, 64])
                    self.model = EliteEnsembleModel(
                        input_size=int(checkpoint.get("input_size", 44)),
                        seq_len=int(checkpoint.get("seq_len", 1)),
                        hidden_dims=list(hidden_dims),
                        num_heads=int(checkpoint.get("num_heads", 4)),
                        num_layers=int(checkpoint.get("num_layers", 3)),
                        dropout=float(checkpoint.get("dropout", 0.1)),
                    )
                    self.use_revolutionary = False
                else:
                    print(f"  Unknown architecture: {architecture}. Training fresh.")
                    self.model = None
                    return

                try:
                    self.model.load_state_dict(model_state_dict, strict=True)
                except RuntimeError as e:
                    print(f"Could not load model state dict (architecture changed): {e}")
                    print("  Training fresh with new architecture.")
                    self.model = None
                    return

                self.model.to(self.device)
                self.model.eval()

                self.scaler_mean = checkpoint["scaler_mean"].to(self.device)
                self.scaler_std = checkpoint["scaler_std"].to(self.device)
                self.metadata = checkpoint.get("metadata", self.metadata)

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
                "model_state_dict": unwrapped_model.state_dict(),
                "model_type": self.model_type,
                "scaler_mean": self.scaler_mean,
                "scaler_std": self.scaler_std,
                "metadata": self.metadata,
            }

            # Add model-specific parameters
            if self.use_revolutionary and isinstance(unwrapped_model, RevolutionaryFinancialModel):
                checkpoint.update(
                    {
                        "architecture": "RevolutionaryFinancialModel-2026",
                        "version": "4.1",
                        "input_size": int(getattr(unwrapped_model, "input_size", 44)),
                        "seq_len": int(getattr(unwrapped_model, "seq_len", 30)),
                        "dim": int(getattr(unwrapped_model, "dim", 512)),
                        "num_layers": int(len(unwrapped_model.layers)),
                        "num_heads": int(getattr(unwrapped_model, "num_heads", 8)),
                        "num_kv_heads": int(getattr(unwrapped_model, "num_kv_heads", 2)),
                        "num_experts": int(getattr(unwrapped_model, "num_experts", 6)),
                        "num_prediction_heads": len(unwrapped_model.prediction_heads),
                        "dropout": 0.15,
                        "use_mamba": True,
                    }
                )
            else:
                checkpoint.update(
                    {
                        "architecture": "EliteEnsembleModel-2025-Compact",
                        "version": "3.2",
                        "input_size": int(getattr(unwrapped_model, "input_size", 44)),
                        "seq_len": int(getattr(unwrapped_model, "seq_len", 1)),
                        "hidden_dims": list(
                            getattr(unwrapped_model, "hidden_dims", [256, 192, 128, 64])
                        ),
                        "num_heads": int(getattr(unwrapped_model, "num_heads", 4)),
                        "num_layers": int(getattr(unwrapped_model, "num_layers", 3)),
                        "dropout": float(
                            getattr(
                                getattr(unwrapped_model, "input_dropout", None),
                                "p",
                                0.1,
                            )
                        ),
                    }
                )

            torch.save(checkpoint, self.model_path)
            param_count = self.model.count_parameters()
            print(f"Saved {self.model_type} model to {self.model_path}")
            print(f"Parameters: {param_count:,}")

        except Exception as e:
            print(f"Error saving model: {e}")

    @staticmethod
    def _augment_batch(X, y, noise_std=0.01, time_mask_prob=0.1):
        """
        Lightweight data augmentation for financial time series:
        1. Gaussian noise injection on features
        2. Random timestep masking (dropout entire timesteps)
        """
        # Gaussian noise on features
        noise = torch.randn_like(X) * noise_std
        X_aug = X + noise

        # Random timestep masking: zero out ~10% of timesteps per sample
        if X_aug.dim() == 3 and time_mask_prob > 0:
            mask = torch.rand(X_aug.shape[0], X_aug.shape[1], 1, device=X.device) > time_mask_prob
            X_aug = X_aug * mask

        return X_aug, y

    def train(
        self,
        X,
        y,
        symbol,
        epochs=60,
        batch_size=64,
        lr=0.0005,
        validation_split=0.2,
        cpu_limit=80,
        comet_experiment=None,
        max_time_seconds=None,
        max_steps=None,
        global_start_time=None,
    ):
        """
        Train with data augmentation, gradient accumulation, cosine warm restarts,
        and EMA model averaging for maximum accuracy from a small model.

        max_time_seconds: If set, stop training gracefully when elapsed time exceeds
                          this limit (saves the best model seen so far). Useful for
                          CI/CD environments with strict time limits.
        max_steps: If set, stop after exactly this many optimizer steps (for benchmarking).
        global_start_time: Wall-clock time when the overall pipeline started (before data
                           prep). When provided, max_time_seconds is measured from this
                           point so data-prep overhead is accounted for and CI timeouts
                           are not exceeded.
        """
        try:
            train_start = time.time()

            # Use global start time for the deadline so data-prep overhead counts
            # toward the time budget and we never overshoot CI timeout.
            deadline_start = global_start_time if global_start_time is not None else train_start
            if global_start_time is not None and max_time_seconds is not None:
                prep_elapsed = train_start - global_start_time
                effective_remaining = max_time_seconds - prep_elapsed
                print(
                    f"Data prep took {prep_elapsed:.0f}s — {effective_remaining/60:.1f}min left for training"
                )

            print(f"\nTraining {self.model_type} model on {symbol}...")
            print(f"Training samples: {len(X)}")
            if max_time_seconds:
                print(f"Time budget: {max_time_seconds/60:.0f} minutes (from pipeline start)")
            if max_steps:
                print(f"Step limit: {max_steps} steps")

            # === Gradient accumulation config ===
            # Simulate effective batch of 256 on small GPU/CPU memory
            grad_accum_steps = max(1, 256 // batch_size)
            effective_batch = batch_size * grad_accum_steps
            print(f"Effective batch size: {effective_batch} (accum {grad_accum_steps} steps)")

            # Convert to tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)

            if X_tensor.dim() == 3:
                seq_len = int(X_tensor.shape[1])
                input_size = int(X_tensor.shape[2])
            else:
                seq_len = 1
                input_size = int(X_tensor.shape[1])

            # Normalize targets to (0, 1) to prevent explosion
            y_min = y_tensor.min()
            y_max = y_tensor.max()
            if y_max > y_min:
                y_tensor = (y_tensor - y_min) / (y_max - y_min)
            else:
                y_tensor = torch.zeros_like(y_tensor)

            # Scaler for features
            self.scaler_mean = X_tensor.mean(dim=0)
            self.scaler_std = X_tensor.std(dim=0) + 1e-8
            X_normalized = (X_tensor - self.scaler_mean) / self.scaler_std

            # Train/validation split
            n_val = int(len(X) * validation_split)
            n_train = len(X) - n_val

            X_train = X_normalized[:n_train]
            y_train = y_tensor[:n_train]
            X_val = X_normalized[n_train:]
            y_val = y_tensor[n_train:]

            print(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")

            # Create model if not already loaded
            if self.model is None:
                if self.use_revolutionary and REVOLUTIONARY_MODEL_AVAILABLE:
                    print("  Creating new Revolutionary v4.1 architecture (~45M Parameters)...")
                    self.model = RevolutionaryFinancialModel(
                        input_size=input_size,
                        seq_len=seq_len,
                        dim=384,
                        num_layers=6,
                        num_heads=6,
                        num_kv_heads=2,
                        num_experts=4,
                        num_prediction_heads=4,
                        dropout=0.15,
                        use_mamba=True,
                        drop_path_rate=0.1,
                    )
                else:
                    print("  Creating new EliteEnsembleModel architecture...")
                    self.model = EliteEnsembleModel(
                        input_size=input_size,
                        seq_len=seq_len,
                        hidden_dims=[256, 192, 128, 64],
                        num_heads=4,
                        num_layers=3,
                        dropout=0.1,
                    )
            else:
                print("  Resuming training from existing model weights...")

            param_count = self.model.count_parameters()
            print(f"Model parameters: {param_count:,}")

            # === EMA (Exponential Moving Average) of model weights ===
            # Keeps a smoothed copy — often generalizes better than final weights
            import copy

            ema_model = copy.deepcopy(self.model)
            ema_model.eval()
            ema_decay = 0.999

            def update_ema(ema_m, model_m, decay):
                with torch.no_grad():
                    for ema_p, model_p in zip(ema_m.parameters(), model_m.parameters()):
                        ema_p.data.mul_(decay).add_(model_p.data, alpha=1.0 - decay)

            # === Optimizer: AdamW with higher weight decay ===
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=lr,
                weight_decay=0.02,
                betas=(0.9, 0.95),
            )

            # === LR schedule: Cosine Annealing with Warm Restarts ===
            # Restarts every T_0 epochs — escapes local minima better than single cosine
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=max(3, epochs // 3),  # Restart every ~1/3 of training
                T_mult=1,
                eta_min=lr * 0.01,
            )

            # === Loss with label smoothing on direction component ===
            from .direction_loss import BalancedDirectionLoss, calculate_direction_metrics

            criterion = BalancedDirectionLoss(alpha=0.6, beta=0.4)

            train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
            )

            # Prepare with Accelerate
            self.model, optimizer, train_loader = self.accelerator.prepare(
                self.model, optimizer, train_loader
            )
            ema_model = ema_model.to(self.device)

            # === Training loop ===
            best_val_loss = float("inf")
            best_ema_state = None
            patience = 15
            patience_counter = 0
            global_step = 0
            step_limit_reached = False
            direction_metrics = {
                "direction_accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1_score": 0,
            }

            for epoch in range(epochs):
                # === Time-based stop: halt before CI timeout kills the job ===
                if max_time_seconds is not None:
                    elapsed = time.time() - deadline_start
                    remaining = max_time_seconds - elapsed
                    if remaining < 90:  # Less than 1.5 min left → stop now (save buffer)
                        print(
                            f"Time limit reached after {elapsed/60:.1f}min — stopping at epoch {epoch}"
                        )
                        break
                    # If one more epoch would likely exceed the limit, stop
                    train_elapsed = time.time() - train_start
                    if epoch > 0:
                        avg_epoch_time = train_elapsed / epoch
                        if remaining < avg_epoch_time * 1.5:  # 50% buffer
                            print(
                                f"Estimated epoch time {avg_epoch_time:.0f}s > remaining {remaining:.0f}s — stopping"
                            )
                            break

                self.model.train()
                epoch_start = time.time()
                train_loss = 0
                optimizer.zero_grad()

                for step, (batch_X, batch_y) in enumerate(train_loader):
                    # === Step limit (for benchmarking) ===
                    if max_steps is not None and global_step >= max_steps:
                        step_limit_reached = True
                        break

                    # === Time-based stop within step loop ===
                    if max_time_seconds is not None and step % 5 == 0:
                        elapsed = time.time() - deadline_start
                        if elapsed >= max_time_seconds - 90:
                            print(
                                f"Time limit reached at step {global_step} "
                                f"({elapsed/60:.1f}min) — stopping mid-epoch"
                            )
                            step_limit_reached = True
                            break

                    # CPU limiter
                    if step % 5 == 0 and psutil.cpu_percent(interval=None) > cpu_limit:
                        time.sleep(0.05)

                    # === Data augmentation ===
                    batch_X_aug, batch_y_aug = self._augment_batch(
                        batch_X,
                        batch_y,
                        noise_std=0.005,  # Small gaussian noise
                        time_mask_prob=0.05,  # 5% timestep dropout
                    )

                    predictions, _ = self.model(batch_X_aug)
                    loss, loss_dict = criterion(predictions, batch_y_aug)

                    # Scale loss for gradient accumulation
                    scaled_loss = loss / grad_accum_steps
                    self.accelerator.backward(scaled_loss)

                    # Step optimizer every grad_accum_steps
                    if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        optimizer.step()
                        optimizer.zero_grad()
                        global_step += 1

                        # Update EMA model
                        unwrapped = self.accelerator.unwrap_model(self.model)
                        update_ema(ema_model, unwrapped, ema_decay)

                    train_loss += loss.item()

                if step_limit_reached:
                    elapsed = time.time() - train_start
                    if max_steps is not None:
                        print(
                            f"Step limit ({max_steps} steps) reached in {elapsed:.2f}s "
                            f"({elapsed/max_steps:.3f}s/step)"
                        )
                    else:
                        print(f"Time limit reached — stopping after {elapsed/60:.1f}min of training")
                    break

                # === Validation using EMA model (better generalization) ===
                ema_model.eval()
                self.model.eval()
                with torch.no_grad():
                    X_val_device = X_val.to(self.device)
                    y_val_device = y_val.to(self.device)

                    # Validate with EMA weights
                    ema_pred, _ = ema_model(X_val_device)
                    ema_val_loss, _ = criterion(ema_pred, y_val_device)
                    ema_val_loss_val = ema_val_loss.item()

                    # Also get regular model val loss for logging
                    val_pred, _ = self.model(X_val_device)
                    val_loss, _ = criterion(val_pred, y_val_device)
                    val_loss = val_loss.item()

                    direction_metrics = calculate_direction_metrics(ema_pred, y_val_device)

                scheduler.step()

                current_lr = optimizer.param_groups[0]["lr"]
                epoch_time = time.time() - epoch_start
                elapsed_total = time.time() - train_start
                print(
                    f"Epoch {epoch + 1}/{epochs} [{epoch_time:.0f}s, total {elapsed_total/60:.1f}min] - "
                    f"Train: {train_loss:.4f}, Val: {val_loss:.4f}, EMA Val: {ema_val_loss_val:.4f}, "
                    f"Dir Acc: {direction_metrics['direction_accuracy']:.1f}%, LR: {current_lr:.2e}"
                )

                # Log to Comet ML
                if comet_experiment:
                    comet_experiment.log_metrics(
                        {
                            "train_loss": train_loss,
                            "val_loss": val_loss,
                            "ema_val_loss": ema_val_loss_val,
                            "learning_rate": current_lr,
                            "direction_accuracy": direction_metrics["direction_accuracy"],
                            "precision": direction_metrics["precision"],
                            "recall": direction_metrics["recall"],
                            "f1_score": direction_metrics["f1_score"],
                        },
                        step=epoch + 1,
                        epoch=epoch + 1,
                    )

                # Early stopping based on EMA val loss
                if ema_val_loss_val < best_val_loss:
                    best_val_loss = ema_val_loss_val
                    patience_counter = 0
                    # Save best EMA state
                    best_ema_state = copy.deepcopy(ema_model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

            # === Load best EMA weights into the main model for saving ===
            if best_ema_state is not None:
                unwrapped = self.accelerator.unwrap_model(self.model)
                unwrapped.load_state_dict(best_ema_state)
                print("Loaded best EMA weights into model")

            # Update metadata
            if symbol not in self.metadata["trained_symbols"]:
                self.metadata["trained_symbols"].append(symbol)

            self.metadata["training_date"] = datetime.now().isoformat()
            self.metadata["last_symbol"] = symbol
            self.metadata["data_points"] = len(X)
            self.metadata["best_val_loss"] = best_val_loss
            self.metadata["direction_accuracy"] = direction_metrics.get("direction_accuracy", 0)
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
            print(f"Best validation loss (EMA): {best_val_loss:.6f}")
            print(f"Total symbols trained: {len(self.metadata['trained_symbols'])}")

            # Final direction accuracy
            self.model.eval()
            with torch.no_grad():
                X_val_device = X_val.to(self.device)
                y_val_device = y_val.to(self.device)
                final_pred, _ = self.model(X_val_device)
                final_metrics = calculate_direction_metrics(final_pred, y_val_device)
                real_accuracy = final_metrics["direction_accuracy"]

            return {
                "success": True,
                "final_loss": best_val_loss,
                "accuracy": real_accuracy,
                "direction_accuracy": real_accuracy,
                "f1_score": final_metrics.get("f1_score", 0),
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

            if X_tensor.dim() == 1:
                X_tensor = X_tensor.unsqueeze(0)

            if X_tensor.dim() == 2 and getattr(self.scaler_mean, "dim", lambda: 0)() == 2:
                seq_len = int(self.scaler_mean.shape[0])
                X_tensor = X_tensor.unsqueeze(1).repeat(1, seq_len, 1)

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
            print(f"Warning: {symbol} was not in training set. Predictions may be less accurate.")
        return True
