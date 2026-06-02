"""
Drop-in nn.Linear replacement that quantizes its weight via adaptive 6-bit
per-block selection over a configurable list of strategies.

Weight-only PTQ simulator: weights are quantized-then-dequantized once at load
time, and the cached FP32 reconstruction is used for inference. Activations
stay in their original dtype.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantize import (
    AdaptiveResult,
    DEFAULT_STRATEGIES,
    adaptive_quantize_dequantize,
    fixed_quantize_dequantize,
)
from .strategies import AVAILABLE_STRATEGIES, validate_strategies

# Preset shorthand for the four common configurations. None means "no
# quantization" (a passthrough scheme for sanity checks).
SCHEME_PRESETS: Dict[str, Optional[Tuple[str, ...]]] = {
    "adaptive":   ("fp6_e3m2", "fp6_e2m3"),
    "fixed_e3m2": ("fp6_e3m2",),
    "fixed_e2m3": ("fp6_e2m3",),
    "fp16":       None,  # bypass quantization entirely
}


def _resolve_strategies(
    strategies: Optional[Sequence[str]],
    scheme: Optional[str],
) -> Optional[Tuple[str, ...]]:
    """
    Resolve user input to a strategies tuple (or None for passthrough).

    Exactly one of {strategies, scheme} should be set. If both are given,
    `strategies` wins and `scheme` is ignored.
    """
    if strategies is not None:
        return tuple(strategies)
    if scheme is None:
        scheme = "adaptive"
    if scheme not in SCHEME_PRESETS:
        raise ValueError(
            f"Unknown scheme {scheme!r}. Known presets: {sorted(SCHEME_PRESETS)}. "
            f"Alternatively pass `strategies=[...]` explicitly."
        )
    return SCHEME_PRESETS[scheme]


class AdaptiveMXFP6Linear(nn.Module):
    """
    nn.Linear replacement whose weight is quantized per-block using adaptive
    selection over a user-supplied set of strategies (or a single fixed
    strategy, when the list has length 1).

    Args:
        in_features, out_features: as in nn.Linear.
        bias: whether to include a bias term (kept in FP32 — biases are tiny).
        block_size: MX block size (default 32, the OCP standard).
        strategies: ordered list of strategy names from `AVAILABLE_STRATEGIES`.
            Length must be a power of 2. If None, `scheme` is used instead.
        scheme: convenience shorthand for the four common cases — "adaptive"
            (default), "fixed_e3m2", "fixed_e2m3", "fp16" (no quantization).
            Ignored if `strategies` is given.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        block_size: int = 32,
        strategies: Optional[Sequence[str]] = None,
        scheme: Optional[str] = None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size

        resolved = _resolve_strategies(strategies, scheme)
        if resolved is not None:
            validate_strategies(resolved)
        self.strategies: Optional[Tuple[str, ...]] = resolved

        # Cache the dequantized weight as a non-parameter buffer
        # (it's derived from the original weight, not separately trainable)
        self.register_buffer("weight", torch.empty(out_features, in_features))
        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        # Quantization diagnostics, populated on quantize_from()
        self.fractions: Optional[Dict[str, float]] = None
        self.choices: Optional[torch.Tensor] = None  # (out_features, n_blocks) if adaptive

    # --------------------------------------------------------------------- #
    # Construction helpers                                                  #
    # --------------------------------------------------------------------- #

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        block_size: int = 32,
        strategies: Optional[Sequence[str]] = None,
        scheme: Optional[str] = None,
    ) -> "AdaptiveMXFP6Linear":
        """Wrap an existing nn.Linear, quantizing its weight."""
        new = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=(linear.bias is not None),
            block_size=block_size,
            strategies=strategies,
            scheme=scheme,
        )
        new.quantize_from(
            linear.weight.detach(),
            linear.bias.detach() if linear.bias is not None else None,
        )
        return new

    def quantize_from(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> None:
        """Apply the configured quantization to a weight tensor."""
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"weight shape {tuple(weight.shape)} does not match "
                f"({self.out_features}, {self.in_features})"
            )
        if self.in_features % self.block_size != 0:
            raise ValueError(
                f"in_features ({self.in_features}) must be divisible by block_size "
                f"({self.block_size}); padding is not yet supported."
            )

        w = weight.to(torch.float32)
        if self.strategies is None:
            # fp16 / passthrough mode — no quantization
            self.weight.data.copy_(w)
            self.fractions = None
            self.choices = None
        elif len(self.strategies) == 1:
            # Fixed (single-strategy) mode
            (only,) = self.strategies
            self.weight.data.copy_(fixed_quantize_dequantize(w, only, self.block_size))
            self.fractions = {only: 1.0}
            self.choices = None
        else:
            # Adaptive (multi-strategy) mode
            res: AdaptiveResult = adaptive_quantize_dequantize(
                w, block_size=self.block_size, strategies=self.strategies
            )
            self.weight.data.copy_(res.recon)
            self.fractions = res.fractions
            self.choices = res.choices  # (out_features, n_blocks)

        if bias is not None and self.bias is not None:
            self.bias.data.copy_(bias.to(self.bias.dtype))

    # --------------------------------------------------------------------- #
    # Forward pass                                                          #
    # --------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        if self.strategies is None:
            cfg = "scheme='fp16'"
        elif len(self.strategies) == 1:
            cfg = f"strategies={self.strategies}"
        else:
            cfg = f"strategies={self.strategies} (adaptive, k={len(self.strategies)})"
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, block_size={self.block_size}, {cfg}"
        )


# --------------------------------------------------------------------- #
# Model-wide swap helper                                                #
# --------------------------------------------------------------------- #

def quantize_linear_layers_(
    model: nn.Module,
    strategies: Optional[Sequence[str]] = None,
    scheme: Optional[str] = None,
    block_size: int = 32,
    exclude_name_substrings: Tuple[str, ...] = ("lm_head", "embed"),
) -> dict:
    """
    In-place: replace every nn.Linear in `model` with AdaptiveMXFP6Linear.

    Args:
        model: any nn.Module containing nn.Linear children.
        strategies: ordered list of strategy names; length must be a power
            of 2. If None, `scheme` resolves to a preset.
        scheme: convenience preset ("adaptive", "fixed_e3m2", "fixed_e2m3",
            "fp16"). Ignored if `strategies` is given.
        block_size: per-block scale group size (OCP standard is 32).
        exclude_name_substrings: skip Linears whose dotted name contains any
            of these substrings. Default skips `lm_head` and `embed`.

    Returns:
        A summary dict keyed by layer name with:
            - "strategies": tuple of strategy names used (or None for fp16)
            - "fractions": dict {strategy_name: fraction_of_blocks}
            - For backward compatibility, "fraction_e3m2" / "fraction_e2m3"
              entries are populated when those strategies are in use.
    """
    summary: dict = {}
    # Collect first to avoid mutating while iterating
    targets: List[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(s in name for s in exclude_name_substrings):
            continue
        if module.in_features % block_size != 0:
            # silently skip layers we can't block evenly
            continue
        targets.append(name)

    for name in targets:
        parent, attr = _find_parent(model, name)
        old: nn.Linear = getattr(parent, attr)
        new = AdaptiveMXFP6Linear.from_linear(
            old,
            block_size=block_size,
            strategies=strategies,
            scheme=scheme,
        )
        setattr(parent, attr, new)

        entry: dict = {
            "strategies": new.strategies,
            "fractions": new.fractions,
        }
        # Backward-compat keys for benchmark code that read these directly
        if new.fractions is not None:
            entry["fraction_e3m2"] = new.fractions.get("fp6_e3m2", 0.0)
            entry["fraction_e2m3"] = new.fractions.get("fp6_e2m3", 0.0)
        summary[name] = entry

    return summary


def _find_parent(model: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]
