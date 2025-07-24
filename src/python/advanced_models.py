"""
Advanced Neural Network Architectures for Stock Prediction
Includes LSTM+Attention, Transformer, CNN-LSTM, and Bayesian models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        return self.w_o(attention_output)
    
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        return torch.matmul(attention_weights, V)

class LSTMAttentionPredictor(nn.Module):
    """LSTM with self-attention mechanism for temporal pattern recognition"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_layers: int = 2, attention_heads: int = 8, 
                 dropout: float = 0.2, sequence_length: int = 30):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        # Attention mechanism
        self.attention = MultiHeadAttention(hidden_size, attention_heads, dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
        
        self._initialize_weights()
        
    def forward(self, x):
        # Reshape if needed for sequence processing
        if len(x.shape) == 2:
            # Convert flat features to sequence
            batch_size = x.size(0)
            x = x.view(batch_size, 1, -1)  # Single timestep
            
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Self-attention mechanism
        attended = self.attention(lstm_out, lstm_out, lstm_out)
        attended = self.layer_norm1(attended + lstm_out)  # Residual connection
        
        # Feed-forward network
        ffn_out = self.ffn(attended)
        ffn_out = self.layer_norm2(ffn_out + attended)  # Residual connection
        
        # Use last timestep for prediction
        output = self.dropout(ffn_out[:, -1, :])
        return self.output_layer(output)
    
    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
            elif 'bias' in name:
                nn.init.constant_(param, 0)

class TransformerPredictor(nn.Module):
    """Transformer architecture for stock prediction"""
    
    def __init__(self, input_size: int, d_model: int = 256, 
                 nhead: int = 8, num_layers: int = 6, 
                 dropout: float = 0.1, sequence_length: int = 30):
        super().__init__()
        
        self.input_size = input_size
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # Input projection
        self.input_projection = nn.Linear(input_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout,
            activation='gelu', batch_first=True,
            dim_feedforward=d_model * 4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self._initialize_weights()
        
    def forward(self, x):
        # Reshape if needed for sequence processing
        if len(x.shape) == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, 1, -1)
            
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer processing
        transformer_out = self.transformer(x)
        
        # Use last timestep for prediction
        output = transformer_out[:, -1, :]
        return self.output_layer(output)
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

class CNNLSTMPredictor(nn.Module):
    """Hybrid CNN-LSTM for pattern recognition and temporal modeling"""
    
    def __init__(self, input_size: int, sequence_length: int = 30, 
                 cnn_channels: List[int] = [64, 128, 256],
                 lstm_hidden: int = 128, lstm_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        
        # 1D CNN for pattern extraction
        conv_layers = []
        in_channels = input_size
        
        for out_channels in cnn_channels:
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.GroupNorm(min(32, out_channels), out_channels),  # Use GroupNorm instead of BatchNorm
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
            
        self.conv_layers = nn.Sequential(*conv_layers)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(sequence_length)
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(
            cnn_channels[-1], lstm_hidden, lstm_layers,
            batch_first=True, dropout=dropout if lstm_layers > 1 else 0
        )
        
        # Output layers
        self.output_layer = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, 1)
        )
        
        self._initialize_weights()
        
    def forward(self, x):
        # Reshape for CNN processing
        if len(x.shape) == 2:
            batch_size = x.size(0)
            x = x.view(batch_size, self.input_size, 1)
        elif len(x.shape) == 3:
            x = x.transpose(1, 2)  # (batch, seq, features) -> (batch, features, seq)
            
        # CNN feature extraction
        conv_out = self.conv_layers(x)
        conv_out = self.adaptive_pool(conv_out)
        
        # Prepare for LSTM
        conv_out = conv_out.transpose(1, 2)  # (batch, features, seq) -> (batch, seq, features)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(conv_out)
        
        # Use last timestep
        output = lstm_out[:, -1, :]
        return self.output_layer(output)
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight uncertainty"""
    
    def __init__(self, in_features: int, out_features: int, prior_std: float = 1.0):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        # Initialize parameters
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        nn.init.constant_(self.weight_rho, -3)
        
        fan_in = self.weight_mu.size(1)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.bias_rho, -3)
        
    def forward(self, x, sample=True):
        if sample:
            # Sample weights from posterior
            weight_std = torch.log1p(torch.exp(self.weight_rho))
            weight = self.weight_mu + weight_std * torch.randn_like(weight_std)
            
            bias_std = torch.log1p(torch.exp(self.bias_rho))
            bias = self.bias_mu + bias_std * torch.randn_like(bias_std)
        else:
            # Use mean weights
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)
    
    def kl_divergence(self):
        """Calculate KL divergence for regularization"""
        weight_std = torch.log1p(torch.exp(self.weight_rho))
        bias_std = torch.log1p(torch.exp(self.bias_rho))
        # KL divergence between posterior and prior
        kl_weight = self._kl_divergence_normal(self.weight_mu, weight_std, 0, self.prior_std)
        kl_bias = self._kl_divergence_normal(self.bias_mu, bias_std, 0, self.prior_std)
        return kl_weight + kl_bias
    
    def _kl_divergence_normal(self, mu1, std1, mu2, std2):
        """KL divergence between two normal distributions"""
        if not torch.is_tensor(mu2):
            mu2 = torch.zeros_like(mu1)
        if not torch.is_tensor(std2):
            std2 = torch.ones_like(std1) * float(std2)
        kl = torch.log(std2 / std1) + (std1**2 + (mu1 - mu2)**2) / (2 * std2**2) - 0.5
        return kl.sum()

class BayesianPredictor(nn.Module):
    """Bayesian neural network for uncertainty quantification"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [128, 64, 32],
                 prior_std: float = 1.0, dropout: float = 0.1):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        
        # Build Bayesian layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(BayesianLinear(prev_size, hidden_size, prior_std))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
            
        layers.append(BayesianLinear(prev_size, 1, prior_std))
        
        self.layers = nn.ModuleList([layer for layer in layers if isinstance(layer, BayesianLinear)])
        self.activations = nn.ModuleList([layer for layer in layers if not isinstance(layer, BayesianLinear)])
        
    def forward(self, x, sample=True):
        """Forward pass with optional weight sampling"""
        layer_idx = 0
        activation_idx = 0
        
        for i in range(len(self.layers)):
            x = self.layers[i](x, sample)
            
            # Apply activations except for last layer
            if i < len(self.layers) - 1:
                x = F.gelu(x)
                if activation_idx < len(self.activations) and isinstance(self.activations[activation_idx], nn.Dropout):
                    x = self.activations[activation_idx](x)
                    activation_idx += 1
                    
        return x
    
    def kl_divergence(self):
        """Total KL divergence for all Bayesian layers"""
        return sum(layer.kl_divergence() for layer in self.layers)
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """Generate predictions with epistemic uncertainty"""
        self.train()  # Enable sampling
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x, sample=True)
                predictions.append(pred)
                
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.std(dim=0)
        
        return mean_pred, uncertainty

class EnhancedFeedforward(nn.Module):
    """Enhanced feedforward network with residual connections and advanced activations"""
    
    def __init__(self, input_size: int = 22, hidden_sizes: List[int] = [256, 128, 64],
                 dropout_rate: float = 0.3, use_residual: bool = True):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.use_residual = use_residual
        
        # Build network with residual connections
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            self.norms.append(nn.LayerNorm(hidden_size))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
            
        # Output layer
        self.output_layer = nn.Linear(prev_size, 1)
        
        # Projection layers for residual connections
        self.projections = nn.ModuleList()
        prev_size = input_size
        for hidden_size in hidden_sizes:
            if prev_size != hidden_size and use_residual:
                self.projections.append(nn.Linear(prev_size, hidden_size))
            else:
                self.projections.append(None)
            prev_size = hidden_size
            
        self._initialize_weights()
        
    def forward(self, x):
        residual = x
        
        for i, (layer, norm, dropout, projection) in enumerate(zip(
            self.layers, self.norms, self.dropouts, self.projections
        )):
            # Linear transformation
            out = layer(x)
            
            # Residual connection
            if self.use_residual:
                if projection is not None:
                    residual = projection(residual)
                if residual.shape == out.shape:
                    out = out + residual
                    
            # Activation and normalization
            out = F.silu(out)  # SiLU activation (same as Swish)
            out = norm(out)
            out = dropout(out)
            
            x = out
            residual = out
            
        return self.output_layer(x)
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

def get_model_info(model):
    """Get comprehensive model information"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'model_type': model.__class__.__name__,
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),
        'architecture': str(model)
    }

if __name__ == "__main__":
    # Test all models
    input_size = 22
    batch_size = 32
    test_input = torch.randn(batch_size, input_size)
    
    models = {
        'LSTM+Attention': LSTMAttentionPredictor(input_size),
        'Transformer': TransformerPredictor(input_size),
        'CNN-LSTM': CNNLSTMPredictor(input_size),
        'Bayesian': BayesianPredictor(input_size),
        'Enhanced FFN': EnhancedFeedforward(input_size)
    }
    
    print("Testing Advanced Models:")
    print("=" * 50)
    
    for name, model in models.items():
        try:
            output = model(test_input)
            info = get_model_info(model)
            print(f"{name}:")
            print(f"  Output shape: {output.shape}")
            print(f"  Parameters: {info['total_parameters']:,}")
            print(f"  Size: {info['model_size_mb']:.2f} MB")
            print()
        except Exception as e:
            print(f"{name}: Error - {e}")
            print()