"""
Meridian.AI Financial Model v5.0
Architecture:
- Grouped Query Attention (GQA) with QK-Norm and RoPE
- Mamba-2 State Space Models (SSM) with selective scan
- Mixture of Experts (MoE) with SwiGLU
- RMSNorm, stochastic depth, scaled residual init
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RotaryEmbedding(nn.Module):
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
    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class GroupedQueryAttention(nn.Module):
    """GQA with QK-Norm (Gemma 2 style) and RoPE."""

    def __init__(self, dim, num_heads=8, num_kv_heads=2, dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = dim // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim**-0.5
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)

        cos, sin = self.rotary_emb(x, seq_len)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k = k.repeat_interleave(self.num_queries_per_kv, dim=1)
        v = v.repeat_interleave(self.num_queries_per_kv, dim=1)

        if hasattr(F, "scaled_dot_product_attention"):
            attn_output = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.dropout.p if self.training else 0.0,
                is_causal=False,
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim=None, bias=False):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(dim * 8 / 3)
        hidden_dim = (hidden_dim + 255) // 256 * 256
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MambaBlock(nn.Module):
    """Mamba-2 SSM with vectorized selective scan (precomputes dA/dB outside the loop)."""

    def __init__(self, dim, state_dim=16, conv_dim=4, expand=2):
        super().__init__()
        self.dim = dim
        self.state_dim = state_dim
        inner_dim = int(dim * expand)
        self.inner_dim = inner_dim

        self.norm = RMSNorm(dim)
        self.in_proj = nn.Linear(dim, inner_dim * 2, bias=False)
        self.conv1d = nn.Conv1d(
            inner_dim, inner_dim, kernel_size=conv_dim, padding=conv_dim - 1, groups=inner_dim
        )
        self.dt_proj = nn.Linear(inner_dim, inner_dim, bias=True)
        self.B_proj = nn.Linear(inner_dim, state_dim, bias=False)
        self.C_proj = nn.Linear(inner_dim, state_dim, bias=False)

        A = torch.arange(1, state_dim + 1, dtype=torch.float32).unsqueeze(0).expand(inner_dim, -1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(inner_dim))
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x):
        batch, seq_len, dim = x.shape

        x_norm = self.norm(x)
        x_inner, res = self.in_proj(x_norm).split([self.inner_dim, self.inner_dim], dim=-1)

        x_conv = self.conv1d(x_inner.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        x_conv = F.silu(x_conv)

        dt = F.softplus(self.dt_proj(x_conv))
        B = self.B_proj(x_conv)
        C = self.C_proj(x_conv)
        A = -torch.exp(self.A_log)

        # Precompute dA[B,T,I,S] and dB[B,T,I,S] so each loop iteration is 2 ops.
        dA = torch.exp(A[None, None] * dt.unsqueeze(-1))
        dB = (dt * x_conv).unsqueeze(-1) * B.unsqueeze(2)

        h = x.new_zeros(batch, self.inner_dim, self.state_dim)
        y = x.new_empty(batch, seq_len, self.inner_dim)
        for t in range(seq_len):
            h = dA[:, t] * h + dB[:, t]
            y[:, t] = (h * C[:, t, None, :]).sum(-1) + self.D * x_conv[:, t]

        return self.out_proj(y * F.silu(res))


class ExpertLayer(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MixtureOfExperts(nn.Module):
    """Top-k sparse MoE with SwiGLU experts."""

    def __init__(self, dim, num_experts=4, expert_hidden_dim=None, top_k=2):
        super().__init__()
        if expert_hidden_dim is None:
            expert_hidden_dim = dim * 2
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [ExpertLayer(dim, expert_hidden_dim) for _ in range(num_experts)]
        )

    def forward(self, x):
        batch_size, seq_len, dim = x.shape
        x_flat = x.view(-1, dim)

        router_probs = F.softmax(self.router(x_flat), dim=-1)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            expert_prob = top_k_probs[:, i : i + 1]
            for expert_id in range(self.num_experts):
                mask = expert_idx == expert_id
                if mask.any():
                    output[mask] += self.experts[expert_id](x_flat[mask]) * expert_prob[mask]

        return output.view(batch_size, seq_len, dim)


class StochasticDepth(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.floor(torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob)
        return x * random_tensor / keep_prob


class MeridianBlock(nn.Module):
    """Transformer block: GQA + optional Mamba SSM + MoE with stochastic depth."""

    def __init__(
        self,
        dim,
        num_heads=8,
        num_kv_heads=2,
        num_experts=4,
        dropout=0.1,
        use_mamba=True,
        drop_path=0.0,
        layer_scale_init=1e-2,
        mamba_state_dim=16,
    ):
        super().__init__()
        self.use_mamba = use_mamba

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)

        self.attn = GroupedQueryAttention(dim, num_heads, num_kv_heads, dropout)

        if use_mamba:
            self.mamba = MambaBlock(dim, state_dim=mamba_state_dim)

        self.moe = MixtureOfExperts(dim, num_experts)
        self.dropout = nn.Dropout(dropout)
        self.drop_path = StochasticDepth(drop_path)

        self.attn_scale = nn.Parameter(torch.ones(dim) * layer_scale_init)
        self.moe_scale = nn.Parameter(torch.ones(dim) * layer_scale_init)
        if use_mamba:
            self.mamba_scale = nn.Parameter(torch.ones(dim) * layer_scale_init)

    def forward(self, x):
        x = x + self.drop_path(self.attn_scale * self.dropout(self.attn(self.norm1(x))))

        if self.use_mamba:
            # MambaBlock applies its own internal norm; do NOT pre-normalize here.
            x = x + self.drop_path(self.mamba_scale * self.dropout(self.mamba(x)))

        moe_norm = self.norm3 if self.use_mamba else self.norm2
        x = x + self.drop_path(self.moe_scale * self.dropout(self.moe(moe_norm(x))))

        return x


class MeridianModel(nn.Module):
    """
    Meridian.AI Financial Model v5.0
    GQA + optional Mamba SSM + MoE, ~11M params at default CPU config.
    """

    def __init__(
        self,
        input_size=44,
        seq_len=30,
        dim=256,
        num_layers=6,
        num_heads=4,
        num_kv_heads=2,
        num_experts=4,
        num_prediction_heads=4,
        dropout=0.1,
        use_mamba=False,
        drop_path_rate=0.1,
        mamba_state_dim=4,
    ):
        super().__init__()

        self.input_size = input_size
        self.seq_len = seq_len
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_experts = num_experts
        self.mamba_state_dim = mamba_state_dim

        self.input_proj = nn.Linear(input_size, dim, bias=False)
        self.input_norm = RMSNorm(dim)
        self.input_dropout = nn.Dropout(dropout)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, num_layers)]
        self.layers = nn.ModuleList(
            [
                MeridianBlock(
                    dim,
                    num_heads,
                    num_kv_heads,
                    num_experts,
                    dropout,
                    use_mamba,
                    drop_path=dpr[i],
                    mamba_state_dim=mamba_state_dim,
                )
                for i in range(num_layers)
            ]
        )

        self.output_norm = RMSNorm(dim)
        self.temporal_attention = nn.Sequential(
            nn.Linear(dim, dim // 4), nn.Tanh(), nn.Linear(dim // 4, 1)
        )
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
        self.ensemble_weights = nn.Parameter(
            torch.ones(num_prediction_heads) / num_prediction_heads
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

        for layer in self.layers:
            x = layer(x)

        x = self.output_norm(x)

        attn_weights = F.softmax(self.temporal_attention(x), dim=1)
        pooled = (x * attn_weights).sum(dim=1)

        all_preds = torch.cat([head(pooled) for head in self.prediction_heads], dim=-1)
        weights = F.softmax(self.ensemble_weights, dim=0)
        final_pred = (all_preds * weights).sum(dim=-1, keepdim=True)

        return final_pred, all_preds

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Backward-compatibility aliases — old code importing RevolutionaryFinancialModel still works.
RevolutionaryTransformerBlock = MeridianBlock
RevolutionaryFinancialModel = MeridianModel
RevolutionaryModel = MeridianModel
