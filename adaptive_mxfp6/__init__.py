"""
adaptive_mxfp6 — per-block grid selection for 6-bit quantization.

For each block of `block_size` elements, the quantizer evaluates a user-supplied
list of strategies (drawn from `AVAILABLE_STRATEGIES`) and picks the one that
minimizes per-block reconstruction MSE. The number of strategies must be a
power of 2.

Default behavior reproduces the original 2-grid adaptive MXFP6 scheme by
selecting between fp6_e3m2 and fp6_e2m3.
"""
from .quantize import (
    adaptive_quantize_dequantize,
    fixed_quantize_dequantize,
    AdaptiveResult,
    DEFAULT_STRATEGIES,
)
from .layers import (
    AdaptiveMXFP6Linear,
    quantize_linear_layers_,
    SCHEME_PRESETS,
)
from .strategies import (
    AVAILABLE_STRATEGIES,
    StrategyInfo,
    validate_strategies,
)

__all__ = [
    # Core quantization
    "adaptive_quantize_dequantize",
    "fixed_quantize_dequantize",
    "AdaptiveResult",
    "DEFAULT_STRATEGIES",
    # Layer / model API
    "AdaptiveMXFP6Linear",
    "quantize_linear_layers_",
    "SCHEME_PRESETS",
    # Strategy catalog
    "AVAILABLE_STRATEGIES",
    "StrategyInfo",
    "validate_strategies",
]
