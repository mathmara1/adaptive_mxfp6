"""
Core adaptive 6-bit quantization logic.

`adaptive_quantize_dequantize` quantizes a tensor by evaluating every strategy
in the user-provided list and, for each block independently, keeping the
reconstruction with the lowest MSE. The set of strategies must be drawn from
`AVAILABLE_STRATEGIES` and its length must be a power of two.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Sequence, Tuple

import torch

from ._mx_backend import quantize_dequantize_mx
from .strategies import AVAILABLE_STRATEGIES, validate_strategies

DEFAULT_STRATEGIES: Tuple[str, str] = ("fp6_e3m2", "fp6_e2m3")
"""The default 2-grid adaptive scheme — the two OCP MXFP6 variants."""


@dataclass
class AdaptiveResult:
    """
    Output of adaptive quantization.

    Attributes:
        recon: the reconstructed (dequantized) tensor, same shape as input.
        choices: int8 tensor with shape (..., n_blocks). Each value is an
            index into `strategy_names`: 0 for the first strategy, 1 for the
            second, etc.
        per_block_mse: float tensor with shape (..., n_blocks); MSE of the
            chosen strategy for each block.
        strategy_names: tuple of strategy identifiers; `strategy_names[i]` is
            the strategy whose index is `i` in `choices`.
    """
    recon: torch.Tensor
    choices: torch.Tensor
    per_block_mse: torch.Tensor
    strategy_names: Tuple[str, ...]

    @property
    def fractions(self) -> Dict[str, float]:
        """Fraction of blocks that selected each strategy. Sums to 1."""
        return {
            name: float((self.choices == i).float().mean())
            for i, name in enumerate(self.strategy_names)
        }


def _block_mse(x: torch.Tensor, x_recon: torch.Tensor, block_size: int) -> torch.Tensor:
    """Per-block MSE along the last dimension. Returns shape (..., n_blocks)."""
    *lead, n = x.shape
    assert n % block_size == 0, (
        f"Last dim ({n}) must be divisible by block_size ({block_size}). "
        "Padding is not yet implemented."
    )
    n_blocks = n // block_size
    sqerr = (x - x_recon).pow(2)
    sqerr = sqerr.reshape(*lead, n_blocks, block_size)
    return sqerr.mean(dim=-1)


def fixed_quantize_dequantize(
    x: torch.Tensor,
    elem_format: Literal["fp6_e3m2", "fp6_e2m3"],
    block_size: int = 32,
) -> torch.Tensor:
    """Round-trip with a single fixed strategy. Baseline for comparison."""
    if elem_format not in AVAILABLE_STRATEGIES:
        raise ValueError(
            f"Unknown strategy {elem_format!r}. "
            f"Available: {sorted(AVAILABLE_STRATEGIES.keys())}"
        )
    return quantize_dequantize_mx(x, elem_format, block_size)


def adaptive_quantize_dequantize(
    x: torch.Tensor,
    block_size: int = 32,
    strategies: Sequence[str] = DEFAULT_STRATEGIES,
) -> AdaptiveResult:
    """
    Adaptive quantization with per-block strategy selection.

    For each block of `block_size` elements along the last dim, the function
    evaluates every strategy in `strategies` and picks the one that minimizes
    per-block reconstruction MSE.

    Args:
        x: input tensor. Last dim must be divisible by block_size.
        block_size: number of elements sharing a scale and a strategy choice.
        strategies: ordered list of strategy names to choose between.
            Length must be a power of 2; all must share a bit width.

    Returns:
        AdaptiveResult with the reconstructed tensor, per-block choices, the
        per-block MSE of the chosen strategy, and the strategy-name list for
        decoding choice indices.
    """
    validate_strategies(strategies)
    strategies = tuple(strategies)
    k = len(strategies)

    # Round-trip with each strategy
    recons = [quantize_dequantize_mx(x, fmt, block_size) for fmt in strategies]

    # Per-block MSE for each strategy: shape (k, ..., n_blocks)
    mses = torch.stack([_block_mse(x, recon, block_size) for recon in recons], dim=0)

    # Pick best strategy per block
    choices = mses.argmin(dim=0)                                # (..., n_blocks), values 0..k-1
    chosen_mse = mses.gather(0, choices.unsqueeze(0)).squeeze(0)

    # Assemble the final reconstruction by gathering per-block from the right recon
    *lead, n = x.shape
    n_blocks = n // block_size
    if k == 1:
        recon = recons[0]
    else:
        # Stack reconstructions in block layout: (k, ..., n_blocks, block_size)
        recons_b = torch.stack(
            [r.reshape(*lead, n_blocks, block_size) for r in recons], dim=0
        )
        # Expand choices to (1, ..., n_blocks, block_size) for gather, broadcasting
        # the per-block index across all block_size elements of that block.
        idx = choices.unsqueeze(0).unsqueeze(-1).expand(
            1, *choices.shape, block_size
        )
        recon_b = recons_b.gather(0, idx).squeeze(0)            # (..., n_blocks, block_size)
        recon = recon_b.reshape(*lead, n)

    return AdaptiveResult(
        recon=recon,
        choices=choices.to(torch.int8),
        per_block_mse=chosen_mse,
        strategy_names=strategies,
    )
