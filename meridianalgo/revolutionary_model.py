# Backward-compatibility shim. New code should import from .meridian_model.
from .meridian_model import (  # noqa: F401
    MeridianBlock,
    MeridianBlock as RevolutionaryTransformerBlock,
    MeridianModel,
    MeridianModel as RevolutionaryFinancialModel,
    MeridianModel as RevolutionaryModel,
    GroupedQueryAttention,
    MambaBlock,
    MixtureOfExperts,
    RMSNorm,
    RotaryEmbedding,
    StochasticDepth,
    SwiGLU,
    apply_rotary_pos_emb,
)
