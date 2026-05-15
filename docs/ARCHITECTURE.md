# MeridianModel Architecture

A technical reference for MeridianModel v5.0 — the hybrid transformer backbone used for financial time-series prediction.

---

## Overview

MeridianModel processes a sliding window of 30 timesteps × 44 features and outputs a single predicted return for the next day. The architecture stacks 6 `MeridianBlock` layers, each combining grouped query attention, optional Mamba SSM, and a mixture-of-experts feed-forward network.

```
Input: (batch, seq_len=30, input_size=44)
       |
       v
Linear projection → (batch, 30, dim)
       |
       v
┌──────────────────────────┐
│       MeridianBlock      │  × num_layers
│  ┌────────────────────┐  │
│  │ RMSNorm            │  │
│  │ GroupedQueryAttn   │  │
│  │ (+ RoPE, QK-Norm)  │  │
│  └────────────────────┘  │
│  ┌────────────────────┐  │
│  │ MambaBlock         │  │  (optional — disabled on CPU)
│  └────────────────────┘  │
│  LayerScale + StochDepth │
│  ┌────────────────────┐  │
│  │ RMSNorm            │  │
│  │ MixtureOfExperts   │  │
│  │ (SwiGLU, top-2)    │  │
│  └────────────────────┘  │
│  LayerScale + StochDepth │
└──────────────────────────┘
       |
       v
RMSNorm → mean pool over sequence
       |
       v
num_prediction_heads Linear(dim, 1) → mean → scalar output
```

---

## Components

### Input Projection

A single `nn.Linear(input_size, dim)` maps each of the 44 feature columns into the model's hidden dimension. No positional embeddings are added at this stage — position is handled by RoPE inside attention.

---

### RMSNorm

Root Mean Square Layer Normalisation — a simpler, faster alternative to LayerNorm that omits the mean-centering step:

```
RMSNorm(x) = x / RMS(x) * weight
RMS(x) = sqrt(mean(x²) + eps)
```

Used as pre-norm before both the attention sublayer and the MoE sublayer within each block. Also applied once at the very end of the stack before pooling.

---

### GroupedQueryAttention (GQA)

Standard multi-head attention is expensive: it maintains a separate key and value projection for every head. GQA reduces the number of KV heads while keeping the query heads full:

```
Q: (batch, seq, num_heads, head_dim)      # num_heads projections
K: (batch, seq, num_kv_heads, head_dim)   # num_kv_heads projections  
V: (batch, seq, num_kv_heads, head_dim)   # num_kv_heads projections
```

Each KV head is shared across `num_heads / num_kv_heads` query heads. At the CPU default (`num_heads=4`, `num_kv_heads=2`), each KV head serves 2 query heads, halving the KV computation.

**QK-Norm**: Both the query and key projections are normalised with RMSNorm before computing attention scores. This stabilises training — without it, dot-product scores can grow large at depth, causing vanishing softmax gradients.

**RoPE (Rotary Position Embeddings)**: Applied to Q and K after QK-Norm. Rotates each head dimension by an angle proportional to its position index:

```
θ_i = position / 10000^(2i/head_dim)
[q_2i, q_{2i+1}] → [q_2i cos θ - q_{2i+1} sin θ, q_2i sin θ + q_{2i+1} cos θ]
```

RoPE encodes relative rather than absolute positions — the attention score between two timesteps depends only on their distance, not their absolute indices. This generalises better to sequence lengths not seen during training.

---

### MambaBlock (optional)

A selective state space model (SSM) operating in O(N) time over the sequence. The selective scan gate chooses which timesteps to retain in the hidden state:

```
h_t = A * h_{t-1} + B_t * x_t      # state update
y_t = C_t * h_t + D * x_t          # output
```

Where `B_t`, `C_t` are input-dependent (the "selective" part). In v5.0, the scan is vectorised: `dA` and `dB` are precomputed for all timesteps before the scan loop, and `h` is reset to zero at the start of each sample.

**CPU default**: `use_mamba=False`. The Mamba block is an identity pass-through when disabled — no overhead. GQA alone handles temporal dependencies on CPU. Mamba adds value on GPU where the SSM can be parallelised efficiently.

---

### Layer Scale + Stochastic Depth

Each sublayer (attention + Mamba, then MoE) is wrapped in two regularisation mechanisms:

**Layer Scale**: Multiplies the sublayer output by a learnable scalar initialised to 0.1. This lets the model start near-identity (residual connections dominate early training) and gradually learn to use each component:

```
output = x + layer_scale * sublayer(x)
```

**Stochastic Depth (Drop Path)**: During training, randomly drops entire sublayer outputs at rate `drop_prob`, replacing them with zero. The residual connection still passes through. This acts as a data-dependent regulariser — the model learns to be robust when any individual sublayer is absent.

---

### MixtureOfExperts (MoE)

The feed-forward sublayer is replaced by a pool of `num_experts` independent expert networks with top-2 routing:

```
gate_logits = x @ W_gate                     # (batch*seq, num_experts)
top2_weights, top2_idx = softmax(gate_logits).topk(2)
output = sum(weight_i * Expert_i(x))         # over top-2 experts
```

Each expert is a **SwiGLU** network:

```
SwiGLU(x) = Linear_down(SiLU(Linear_gate(x)) * Linear_up(x))
```

The gating mechanism (`SiLU(gate) * up`) lets each expert selectively amplify or suppress parts of the input, which is more expressive than a standard FFN. With 4 experts and top-2 routing, the model routes to 2 experts per token — different market regimes (trending, volatile, mean-reverting) tend to activate different experts.

---

### Output heads

After the final RMSNorm, the sequence dimension is mean-pooled to a single vector. This is then passed through `num_prediction_heads` independent `Linear(dim, 1)` heads. Their outputs are averaged:

```
x_pooled = x.mean(dim=1)                            # (batch, dim)
preds = [head(x_pooled) for head in self.heads]     # num_heads × (batch, 1)
output = torch.stack(preds, dim=-1).mean(dim=-1)    # (batch, 1)
```

Multiple heads act as an implicit ensemble — each learns a slightly different projection of the pooled state. Averaging reduces variance in the final prediction.

The model also returns a `confidence` signal (the standard deviation across heads):

```python
pred, confidence = model(x)   # pred: (batch, 1), confidence: (batch,)
```

---

## Hyperparameter defaults

| Parameter | CPU Default | Description |
|-----------|-------------|-------------|
| `input_size` | 44 | Number of input features |
| `seq_len` | 30 | Lookback window (timesteps) |
| `dim` | 256 | Hidden dimension |
| `num_layers` | 6 | Number of MeridianBlocks |
| `num_heads` | 4 | Query attention heads |
| `num_kv_heads` | 2 | Key/value heads (GQA) |
| `num_experts` | 4 | MoE expert count |
| `num_prediction_heads` | 4 | Output heads (averaged) |
| `dropout` | 0.1 | Dropout rate |
| `use_mamba` | False | Enable Mamba SSM sublayer |
| `mamba_state_dim` | 4 | Mamba hidden state size |

---

## Parameter count

At CPU default config (`dim=256`, `num_layers=6`, `num_heads=4`, `num_experts=4`, `use_mamba=False`):

| Component | Params |
|-----------|--------|
| Input projection | 256 × 44 = 11K |
| GQA per layer | Q: 256×256, KV: 256×2×64, out: 256×256 ≈ 230K |
| MoE per layer (4 experts) | 4 × SwiGLU(256→1024→256) ≈ 2M |
| RMSNorm, LayerScale | negligible |
| Output heads (4) | 4 × 256 = 1K |
| **Total** | **~11M** |

At GPU/large config (`dim=384`, same depth):

| Total | **~45M** |
|-------|---------|

---

## Backward compatibility

The canonical implementation lives in `meridianalgo/meridian_model.py`. The old `meridianalgo/revolutionary_model.py` is now a shim that re-exports everything with the old names:

```python
from .meridian_model import (
    MeridianModel,
    MeridianModel as RevolutionaryFinancialModel,
    MeridianModel as RevolutionaryModel,
    MeridianBlock as RevolutionaryTransformerBlock,
    ...
)
```

Old checkpoints saved with `"RevolutionaryFinancialModel-2026"` load transparently — the loader accepts both architecture strings.

---

## See also

- [Training guide](TRAINING.md) — data pipeline, loss function, LR schedule, CI
- [Model card](MODEL_CARD.md) — specs, checkpoint format, Hugging Face usage
- [Quick start](QUICK_START.md) — get predictions running in 5 minutes
