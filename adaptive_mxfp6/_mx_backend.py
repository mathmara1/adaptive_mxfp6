"""
Quantization backend dispatcher.

Routes each strategy to the right implementation:
  - microxcaling-supported formats (fp6_e3m2, fp6_e2m3, fp8 variants, fp4_e2m1,
    int4, int8) → microxcaling's `quantize_mx_op`.
  - Everything else (int6, nf6, split6, learned grids, shifted variants) →
    our own codebook quantizer in `_codebooks.py`.

This is the only file outside `_codebooks.py` that imports microxcaling, so
if we ever swap the backend we touch one place.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add microxcaling to sys.path without pip-installing (avoids torch==2.2.0 pin)
_MICROXCALING_DIR = Path(__file__).resolve().parent.parent / "microxcaling"
if str(_MICROXCALING_DIR) not in sys.path:
    sys.path.insert(0, str(_MICROXCALING_DIR))

import torch
from mx.mx_ops import quantize_mx_op
from mx.specs import finalize_mx_specs

from . import _codebooks


# Formats microxcaling implements natively. Anything not in this set is assumed
# to be a custom-codebook strategy and routed to _codebooks.codebook_quantize_dequantize.
_MICROXCALING_FORMATS = frozenset({
    "fp8_e5m2", "fp8_e4m3",
    "fp6_e3m2", "fp6_e2m3",
    "fp4_e2m1",
    "int8", "int4",
})


def _make_specs(elem_format: str, block_size: int = 32) -> dict:
    return finalize_mx_specs({
        "w_elem_format": elem_format,
        "a_elem_format": elem_format,
        "block_size": block_size,
        "scale_bits": 8,
        "bfloat": 16,
        "custom_cuda": False,
    })


def quantize_dequantize_mx(
    x: torch.Tensor, elem_format: str, block_size: int = 32
) -> torch.Tensor:
    """
    Round-trip a tensor through the appropriate quantization backend.
    Routes microxcaling-native formats to microxcaling; custom-codebook
    strategies (int6, nf6, etc.) to our codebook quantizer.
    """
    if elem_format in _MICROXCALING_FORMATS:
        specs = _make_specs(elem_format, block_size)
        return quantize_mx_op(
            x, specs, elem_format=elem_format, block_size=block_size, axes=[-1]
        )

    if elem_format in _codebooks.CODEBOOKS:
        return _codebooks.codebook_quantize_dequantize(
            x, _codebooks.CODEBOOKS[elem_format], block_size=block_size
        )

    raise ValueError(
        f"Unknown elem_format: {elem_format!r}. "
        f"microxcaling formats: {sorted(_MICROXCALING_FORMATS)}; "
        f"registered codebooks: {sorted(_codebooks.CODEBOOKS)}"
    )
