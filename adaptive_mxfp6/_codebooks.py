"""
Generic block-scaled quantization for arbitrary codebooks.

This module is the quantization backend for strategies that aren't supported
by microxcaling (int6, nf6, split6, learned grids, shifted variants). The math
is the same block-scaled snap-to-nearest pattern as OCP MX formats; only the
codebook (the list of representable values) differs per strategy.

To add a new codebook strategy:
  1. Define its codebook (1-D tensor of representable values) here.
  2. Register it in `CODEBOOKS` keyed by strategy name.
  3. Flip `implemented=True` on its entry in `strategies.py`.

The `_mx_backend.py` dispatcher routes any non-microxcaling strategy through
this module automatically based on its presence in `CODEBOOKS`.
"""
from __future__ import annotations

from typing import Dict

import torch


def _make_int6_codebook() -> torch.Tensor:
    """
    Uniform 6-bit signed integer grid: 64 values, two's complement style.

    Codebook = [-32, -31, ..., 30, 31] / 32  →  [-1.0, -31/32, ..., 30/32, 31/32]

    The "/32" normalization makes the most-negative value exactly -1, matching
    the convention the codebook quantizer expects (block absmax maps to codebook
    absmax). The 64 levels are evenly spaced — uniform precision across the range.

    Pairs naturally with `fp6_e3m2` or `fp6_e2m3` to form the 6-bit analog of
    the IF4 scheme (IF6).
    """
    return torch.arange(-32, 32, dtype=torch.float32) / 32.0


def _make_nf6_codebook(offset: float = 0.9925) -> torch.Tensor:
    """
    Normal-distribution prior grid (NormalFloat 6, the 6-bit analog of QLoRA's NF4).

    Codebook is 64 values placed at equal-probability quantiles of N(0, 1):
      - 32 positive values from inverse-CDF at probabilities [0.5, offset]
      - 1 explicit zero
      - 31 negative values mirroring the positive side (one fewer to make 64 total)

    All values are normalized so the maximum absolute value is exactly 1. The
    `offset` parameter controls how much probability mass falls outside the
    representable range; the default 0.9925 puts ~0.75% of mass in each tail
    (chosen to match NF4's design pattern, scaled for 64 levels instead of 16).

    Pairs naturally with a learned residual grid to form PO2(nf6), the 6-bit
    analog of Grid Games' PO2(NF4) — their strongest 4-bit single-grid baseline.
    """
    # 32 positive values from probit at probabilities [offset, 0.5], excluding 0.5
    pos_probs = torch.linspace(offset, 0.5, 33, dtype=torch.float64)[:-1]
    pos_values = torch.special.ndtri(pos_probs)  # 32 positive probit values, descending

    # 31 negative values (mirror of positive, one fewer for 64 total)
    neg_probs = torch.linspace(offset, 0.5, 32, dtype=torch.float64)[:-1]
    neg_values = -torch.special.ndtri(neg_probs)  # 31 negative values

    values = torch.cat([neg_values, torch.tensor([0.0], dtype=torch.float64), pos_values])
    values, _ = values.sort()
    values = (values / values.abs().max()).to(torch.float32)
    return values


# Codebook tensors keyed by strategy name. Each entry is a 1-D float tensor
# listing every representable value of that strategy. For 6-bit formats this
# is typically 64 entries. Populated incrementally as strategies are added.
CODEBOOKS: Dict[str, torch.Tensor] = {
    "int6": _make_int6_codebook(),
    "nf6":  _make_nf6_codebook(),
}


def codebook_quantize_dequantize(
    x: torch.Tensor,
    codebook: torch.Tensor,
    block_size: int = 32,
) -> torch.Tensor:
    """
    Block-scaled quantization onto an arbitrary codebook.

    For each block of `block_size` elements along the last dimension:
      1. Compute the block's absmax.
      2. Scale so the block's absmax maps to the codebook's absmax.
      3. Snap each scaled value to the nearest codebook entry.
      4. Dequantize by multiplying back by the scale.

    The "snap to nearest" step uses `torch.searchsorted` for O(N log K) cost
    rather than the naive O(N*K) broadcast — important for large weight tensors.

    Args:
        x: input tensor. Last dim must be divisible by block_size.
        codebook: 1-D tensor of representable values. Will be sorted internally
            and moved to the device/dtype of x as needed.
        block_size: number of elements sharing one block scale (OCP default 32).

    Returns:
        Reconstructed tensor with the same shape and dtype as x.
    """
    *lead, n = x.shape
    assert n % block_size == 0, (
        f"Last dim ({n}) must be divisible by block_size ({block_size})."
    )

    in_dtype = x.dtype
    work_dtype = torch.float32  # do quantization math in fp32 for precision

    # Sort once, move to the input's device, fp32 for math
    codebook = codebook.to(device=x.device, dtype=work_dtype)
    codebook, _ = codebook.sort()
    code_absmax = codebook.abs().max()

    x_work = x.to(work_dtype)
    n_blocks = n // block_size
    x_blocks = x_work.reshape(*lead, n_blocks, block_size)

    # Per-block absmax → (..., n_blocks, 1). Clamp avoids div-by-zero on all-zero blocks.
    block_absmax = x_blocks.abs().amax(dim=-1, keepdim=True)
    scale = (block_absmax / code_absmax).clamp(min=1e-12)

    # Normalize each block to roughly [-1, 1] relative to the codebook range
    x_norm = x_blocks / scale

    # Snap to nearest codebook entry via searchsorted
    flat = x_norm.contiguous().reshape(-1)
    idx_right = torch.searchsorted(codebook, flat)
    idx_right = idx_right.clamp(max=len(codebook) - 1)
    idx_left = (idx_right - 1).clamp(min=0)

    left_vals = codebook[idx_left]
    right_vals = codebook[idx_right]
    # Strict less-than means ties (equidistant) snap to the lower-magnitude side.
    use_right = (right_vals - flat).abs() < (flat - left_vals).abs()
    snapped = torch.where(use_right, right_vals, left_vals)

    # Reshape back and dequantize
    snapped = snapped.reshape(x_blocks.shape)
    x_quant = snapped * scale

    return x_quant.reshape(*lead, n).to(in_dtype)
