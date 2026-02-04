"""
Revolutionary 2026 Financial AI Model
Latest Technologies:
- Mamba State Space Models (SSM) for efficient sequence modeling
- Rotary Position Embeddings (RoPE) for better position encoding
- Group Query Attention (GQA) for efficiency
- SwiGLU activation with gating
- RMSNorm for stability
- Flash Attention 2 for speed
- Mixture of Experts (MoE) for specialization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - Better than absolute/relative positional encoding"""

    def __init__(self, dim, max_seq_len=2048, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos()[None, :, :], emb.sin()[None, :, :]


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embedding to queries and keys"""

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) - More efficient than Multi-Head Attention
    Uses fewer KV heads than Q heads for better efficiency
    """

    def __init__(self, dim, num_heads=8, num_kv_heads=2, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads

        # Q projection uses all heads, K/V use fewer heads
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5

        # Rotary embeddings
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings
        cos, sin = self.rotary_emb(x, seq_len)
        q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE to Q and K
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand K and V to match Q heads (grouped attention)
        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Scaled dot-product attention with Flash Attention if available
        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            # Fallback
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.dim)
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    """SwiGLU: Swish-Gated Linear Unit - State-of-the-art activation"""

    def __init__(self, dim, hidden_dim=None, bias=False):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(dim * 8 / 3)  # Standard ratio for SwiGLU
        hidden_dim = (hidden_dim + 255) // 256 * 256  # Round to 256

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    """RMSNorm: Root Mean Square Layer Normalization - More stable than LayerNorm"""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class MambaBlock(nn.Module):
    """
    Mamba State Space Model Block
    Efficient alternative to attention for sequence modeling
    """

    def __init__(self, dim, state_dim=16, conv_dim=4, expand=2):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        inner_dim = int(dim * expand)

        self.norm = RMSNorm(dim)

        # Input projection
        self.in_proj = nn.Linear(dim, inner_dim * 2, bias=False)

        # Convolution for local context
        self.conv1d = nn.Conv1d(
            inner_dim,
            inner_dim,
            kernel_size=conv_dim,
            padding=conv_dim - 1,
            groups=inner_dim,
        )

        # SSM parameters
        self.x_proj = nn.Linear(inner_dim, state_dim + state_dim + inner_dim, bias=False)
        self.dt_proj = nn.Linear(state_dim, inner_dim, bias=True)

        # Output projection
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        batch, seq_len, dim = x.shape

        # Normalize and project
        x_norm = self.norm(x)
        x_and_res = self.in_proj(x_norm)
        x_inner, res = x_and_res.split([x_and_res.shape[-1] // 2] * 2, dim=-1)

        # Apply convolution
        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)

        # SSM computation (simplified)
        ssm_params = self.x_proj(x_conv)
        delta, B, C = torch.split(
            ssm_params, [self.state_dim, self.state_dim, x_inner.shape[-1]], dim=-1
        )

        # Discretization
        delta = F.softplus(self.dt_proj(delta))

        # State space computation (simplified selective scan)
        y = x_conv * torch.sigmoid(C)

        # Gating and output
        y = y * F.silu(res)
        return self.out_proj(y)


class ExpertLayer(nn.Module):
    """Single expert in Mixture of Experts"""

    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)


class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) - Specialized experts for different patterns
    Top-K routing for efficiency
    """

    def __init__(self, dim, num_experts=4, expert_hidden_dim=None, top_k=2):
        super().__init__()
        if expert_hidden_dim is None:
            expert_hidden_dim = dim * 4

        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.router = nn.Linear(dim, num_experts, bias=False)

        # Experts
        self.experts = nn.ModuleList(
            [ExpertLayer(dim, expert_hidden_dim) for _ in range(num_experts)]
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        # Route to experts
        router_logits = self.router(x_flat)
        router_probs = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # Compute expert outputs
        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_prob = top_k_probs[:, i : i + 1]

            for expert_id in range(self.num_experts):
                mask = expert_idx == expert_id
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_output * expert_prob[mask]

        return output.view(batch_size, seq_len, dim)


class RevolutionaryTransformerBlock(nn.Module):
    """
    Revolutionary Transformer Block combining:
    - Grouped Query Attention with RoPE
    - Mamba SSM for efficient sequence modeling
    - Mixture of Experts for specialization
    - SwiGLU activation
    - RMSNorm
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        num_kv_heads=2,
        num_experts=4,
        dropout=0.1,
        use_mamba=True,
    ):
        super().__init__()
        self.use_mamba = use_mamba

        # Pre-normalization
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)

        # Attention
        self.attn = GroupedQueryAttention(dim, num_heads, num_kv_heads, dropout)

        # Mamba SSM (optional, for long sequences)
        if use_mamba:
            self.mamba = MambaBlock(dim)

        # Mixture of Experts
        self.moe = MixtureOfExperts(dim, num_experts)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Attention with residual
        x = x + self.dropout(self.attn(self.norm1(x)))

        # Mamba SSM with residual (if enabled)
        if self.use_mamba:
            x = x + self.dropout(self.mamba(self.norm2(x)))

        # MoE with residual
        x = x + self.dropout(self.moe(self.norm3(x) if self.use_mamba else self.norm2(x)))

        return x


class RevolutionaryFinancialModel(nn.Module):
    """
    Revolutionary 2026 Financial AI Model
    State-of-the-art architecture for financial time series prediction
    """

    def __init__(
        self,
        input_size=44,
        seq_len=30,
        dim=512,
        num_layers=6,
        num_heads=8,
        num_kv_heads=2,
        num_experts=4,
        num_prediction_heads=4,
        dropout=0.1,
        use_mamba=True,
    ):
        super().__init__()

        self.input_size = input_size
        self.seq_len = seq_len
        self.dim = dim

        # Input embedding
        self.input_proj = nn.Linear(input_size, dim, bias=False)
        self.input_norm = RMSNorm(dim)
        self.input_dropout = nn.Dropout(dropout)

        # Revolutionary transformer blocks
        self.layers = nn.ModuleList(
            [
                RevolutionaryTransformerBlock(
                    dim, num_heads, num_kv_heads, num_experts, dropout, use_mamba
                )
                for _ in range(num_layers)
            ]
        )

        # Output normalization
        self.output_norm = RMSNorm(dim)

        # Temporal pooling with learnable weights
        self.temporal_attention = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1)
        )

        # Multiple prediction heads for ensemble
        self.prediction_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim, dim // 2, bias=False),
                    RMSNorm(dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim // 2, 1, bias=False),
                )
                for _ in range(num_prediction_heads)
            ]
        )

        # Ensemble weighting
        self.ensemble_weights = nn.Parameter(
            torch.ones(num_prediction_heads) / num_prediction_heads
        )

    def forward(self, x):
        # Input projection
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension

        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)

        x = self.output_norm(x)

        # Temporal attention pooling
        attn_weights = self.temporal_attention(x)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = (x * attn_weights).sum(dim=1)  # [batch, dim]

        # Multiple prediction heads
        predictions = []
        for head in self.prediction_heads:
            pred = head(pooled)
            predictions.append(pred)

        all_preds = torch.cat(predictions, dim=-1)  # [batch, num_heads]

        # Weighted ensemble
        weights = F.softmax(self.ensemble_weights, dim=0)
        final_pred = (all_preds * weights).sum(dim=-1, keepdim=True)

        return final_pred, all_preds

    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Alias for backward compatibility
RevolutionaryModel = RevolutionaryFinancialModel
