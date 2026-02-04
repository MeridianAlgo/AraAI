"""
Enhanced Large PyTorch Model - Up to 1M parameters
More intelligent architecture with attention and residual connections
"""

import torch
import torch.nn as nn
import numpy as np


class IntelligentEnsembleModel(nn.Module):
    """
    Intelligent ensemble model with up to 1M parameters
    Features:
    - Multi-head attention
    - Residual connections
    - Layer normalization
    - Advanced dropout strategies
    """

    def __init__(self, input_size=44, hidden_sizes=[768, 512, 384, 256, 128], dropout=0.2):
        super(IntelligentEnsembleModel, self).__init__()

        # Input projection with residual
        self.input_proj = nn.Linear(input_size, hidden_sizes[0])
        self.input_norm = nn.LayerNorm(hidden_sizes[0])

        # Deep feature extraction with residual blocks
        self.residual_blocks = nn.ModuleList()
        for i in range(len(hidden_sizes) - 1):
            self.residual_blocks.append(
                ResidualBlock(hidden_sizes[i], hidden_sizes[i + 1], dropout)
            )

        # Multi-head attention for feature importance
        final_hidden = hidden_sizes[-1]
        self.attention = MultiHeadAttention(final_hidden, num_heads=4, dropout=dropout)

        # Multiple specialized prediction heads (ensemble)
        # Trend prediction head
        self.trend_head = nn.Sequential(
            nn.Linear(final_hidden, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Volatility-aware head
        self.volatility_head = nn.Sequential(
            nn.Linear(final_hidden, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Momentum head
        self.momentum_head = nn.Sequential(
            nn.Linear(final_hidden, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Mean reversion head
        self.reversion_head = nn.Sequential(
            nn.Linear(final_hidden, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Pattern recognition head
        self.pattern_head = nn.Sequential(
            nn.Linear(final_hidden, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Fundamental analysis head
        self.fundamental_head = nn.Sequential(
            nn.Linear(final_hidden, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

        # Advanced ensemble weighting with context
        self.ensemble_weights = nn.Sequential(
            nn.Linear(final_hidden + 6, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 6),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = torch.relu(x)

        # Pass through residual blocks
        for block in self.residual_blocks:
            x = block(x)

        # Apply attention
        x = self.attention(x)

        # Get predictions from all specialized heads
        preds = torch.stack(
            [
                self.trend_head(x).squeeze(-1),
                self.volatility_head(x).squeeze(-1),
                self.momentum_head(x).squeeze(-1),
                self.reversion_head(x).squeeze(-1),
                self.pattern_head(x).squeeze(-1),
                self.fundamental_head(x).squeeze(-1),
            ],
            dim=-1,
        )

        # Context-aware ensemble weighting
        context = torch.cat([x, preds], dim=-1)
        weights = self.ensemble_weights(context)

        # Weighted ensemble prediction
        ensemble_pred = (preds * weights).sum(dim=-1)

        return ensemble_pred, preds, weights

    def count_parameters(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ResidualBlock(nn.Module):
    """Residual block with layer normalization"""

    def __init__(self, in_features, out_features, dropout=0.2):
        super(ResidualBlock, self).__init__()

        self.linear1 = nn.Linear(in_features, out_features)
        self.norm1 = nn.LayerNorm(out_features)
        self.linear2 = nn.Linear(out_features, out_features)
        self.norm2 = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

        # Projection for residual if dimensions don't match
        self.projection = None
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)

    def forward(self, x):
        identity = x

        # First layer
        out = self.linear1(x)
        out = self.norm1(out)
        out = torch.relu(out)
        out = self.dropout(out)

        # Second layer
        out = self.linear2(out)
        out = self.norm2(out)

        # Residual connection
        if self.projection is not None:
            identity = self.projection(identity)

        out = out + identity
        out = torch.relu(out)

        return out


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention for feature importance"""

    def __init__(self, hidden_size, num_heads=4, dropout=0.2):
        super(MultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.head_dim = hidden_size // num_heads

        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        batch_size = x.size(0)
        identity = x

        # Expand for multi-head attention (treat as sequence of length 1)
        x = x.unsqueeze(1)  # [batch, 1, hidden]

        # Linear projections
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        # Reshape for multi-head
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, 1, self.hidden_size)

        # Output projection
        out = self.out(context).squeeze(1)

        # Residual connection and normalization
        out = self.norm(out + identity)

        return out
