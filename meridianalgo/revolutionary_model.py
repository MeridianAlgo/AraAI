# Backward-compatibility shim. New code should import from .meridian_model.
from .meridian_model import (  # noqa: F401
    GroupedQueryAttention,
    MambaBlock,
    MeridianBlock,
    MeridianModel,
    MixtureOfExperts,
    RMSNorm,
    RotaryEmbedding,
    StochasticDepth,
    SwiGLU,
    apply_rotary_pos_emb,
)
