"""DEPRECATED — import from meridianalgo.meridian_model instead.

This module is a backward-compatibility shim kept so older imports keep working.
New code should import directly from `meridianalgo.meridian_model`. The shim
will be removed in v6.0.
"""

import warnings

warnings.warn(
    "meridianalgo.revolutionary_model is deprecated; use "
    "meridianalgo.meridian_model. This shim will be removed in v6.0.",
    DeprecationWarning,
    stacklevel=2,
)

from .meridian_model import (  # noqa: F401, E402
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

# Pre-v5 names kept for back-compat — please migrate to MeridianModel.
RevolutionaryTransformerBlock = MeridianBlock
RevolutionaryFinancialModel = MeridianModel
RevolutionaryModel = MeridianModel
