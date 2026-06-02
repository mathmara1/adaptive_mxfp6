"""
Thin wrapper around microxcaling. Isolates the dependency so the rest of
the package only talks to microxcaling through this module.
"""
import sys
from pathlib import Path

# Add microxcaling to sys.path without pip-installing (avoids torch==2.2.0 pin)
_MICROXCALING_DIR = Path(__file__).resolve().parent.parent / "microxcaling"
if str(_MICROXCALING_DIR) not in sys.path:
    sys.path.insert(0, str(_MICROXCALING_DIR))

import torch
from mx.mx_ops import quantize_mx_op
from mx.specs import finalize_mx_specs


def _make_specs(elem_format: str, block_size: int = 32) -> dict:
    return finalize_mx_specs({
        "w_elem_format": elem_format,
        "a_elem_format": elem_format,
        "block_size": block_size,
        "scale_bits": 8,
        "bfloat": 16,
        "custom_cuda": False,
    })


def quantize_dequantize_mx(x: torch.Tensor, elem_format: str, block_size: int = 32) -> torch.Tensor:
    """
    Round-trip a tensor through OCP MXFP6 (or any other supported elem format).
    Blocks are taken along the last dimension.
    """
    specs = _make_specs(elem_format, block_size)
    return quantize_mx_op(x, specs, elem_format=elem_format, block_size=block_size, axes=[-1])
