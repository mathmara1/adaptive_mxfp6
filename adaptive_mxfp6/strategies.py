"""
Catalog of 6-bit quantization strategies and validation helpers.

The library is currently scoped to 6-bit quantization. The user picks a subset
of strategies from `AVAILABLE_STRATEGIES`; per-block selection then chooses
among them to minimize per-block MSE. The number of strategies must be a power
of two, matching Grid Games' "PO2" framework: 1 grid (fixed), 2 grids (1
selector bit, free in the scale's sign bit), 4 grids (2 selector bits), etc.

The catalog tracks both *implemented* strategies (currently the two OCP MXFP6
variants, backed by microxcaling) and *planned* strategies. Planned entries
appear in the catalog so they are discoverable, but `validate_strategies`
rejects them with a clear error until their implementation lands. To enable a
planned strategy, implement its quantizer and flip `implemented=True`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence


@dataclass(frozen=True)
class StrategyInfo:
    """Static metadata for one quantization strategy in the catalog."""
    name: str           # the format identifier (passed to microxcaling, or used as a key)
    bits: int           # bits per element (excluding the shared block scale)
    implemented: bool   # whether the quantizer actually exists yet
    description: str    # human-readable summary; for planned strategies, includes the TODO


# Catalog of supported strategies. v1 is 6-bit only.
AVAILABLE_STRATEGIES: Dict[str, StrategyInfo] = {

    # ===================================================================
    # Implemented (microxcaling-backed OCP MXFP6 variants)
    # ===================================================================

    "fp6_e3m2": StrategyInfo(
        name="fp6_e3m2",
        bits=6,
        implemented=True,
        description=(
            "OCP MXFP6, 3 exponent bits + 2 mantissa bits — wider range, coarser steps. "
            "Tends to win on blocks with outlier-dominated dynamic range."
        ),
    ),
    "fp6_e2m3": StrategyInfo(
        name="fp6_e2m3",
        bits=6,
        implemented=True,
        description=(
            "OCP MXFP6, 2 exponent bits + 3 mantissa bits — narrower range, finer steps. "
            "Tends to win on blocks where values are clustered close in magnitude."
        ),
    ),

    # ===================================================================
    # Planned (catalog placeholders; implementation TODO)
    # ===================================================================

    "int6": StrategyInfo(
        name="int6",
        bits=6,
        implemented=True,
        description=(
            "Uniform 6-bit signed integer grid (64 levels, two's complement: "
            "[-32, -31, ..., 30, 31] / 32 after normalization). Evenly spaced — "
            "uniform precision across the range. Pairs with an FP6 variant to "
            "form the 6-bit analog of the IF4 scheme (IF6)."
        ),
    ),
    "nf6": StrategyInfo(
        name="nf6",
        bits=6,
        implemented=True,
        description=(
            "Normal-distribution prior grid: 64 values placed at equal-probability "
            "quantiles of N(0, 1) (32 positive + 1 zero + 31 negative). The 6-bit "
            "analog of QLoRA's NF4. Pairs naturally with a learned residual to form "
            "PO2(nf6) — the 6-bit analog of Grid Games' PO2(NF4)."
        ),
    ),
    "split6": StrategyInfo(
        name="split6",
        bits=6,
        implemented=False,
        description=(
            "Asymmetric 6-bit grid with explicit zero: 32 negative + 1 zero + 31 positive "
            "levels (a 32+0+31 split). The 6-bit analog of Grid Games' Split87. "
            "MSE-optimized via coordinate descent on absmax-normalized calibration data. "
            "TODO: train codebook on a pool of normalized blocks + implement codebook quantizer."
        ),
    ),
    "sfp6_shift_pos": StrategyInfo(
        name="sfp6_shift_pos",
        bits=6,
        implemented=False,
        description=(
            "fp6_e3m2 grid shifted by +c*scale (center of mass above zero). "
            "6-bit analog of Grid Games' SFP4 B+ grid. Hardware-implementable as "
            "standard MXFP6 GEMM plus a per-block correction term. "
            "TODO: implement shifted-FP6 quantizer with configurable shift constant c."
        ),
    ),
    "sfp6_shift_neg": StrategyInfo(
        name="sfp6_shift_neg",
        bits=6,
        implemented=False,
        description=(
            "fp6_e3m2 grid shifted by -c*scale (center of mass below zero). "
            "6-bit analog of Grid Games' SFP4 B- grid. Paired with the standard "
            "fp6_e3m2 and sfp6_shift_pos to form a 3- or 4-grid adaptive scheme. "
            "TODO: implement shifted-FP6 quantizer with configurable shift constant c."
        ),
    ),
    "learned_residual": StrategyInfo(
        name="learned_residual",
        bits=6,
        implemented=False,
        description=(
            "Learned 64-entry codebook trained via Lloyd-Max iteration on blocks "
            "where a primary grid (e.g. fp6_e3m2 or nf6) has high reconstruction "
            "error. Implements the residual learning step of Grid Games' PO2 "
            "algorithm at 6-bit; intended as the secondary in PO2(primary). "
            "TODO: implement (a) calibration block sampling, (b) Lloyd-Max codebook "
            "learning with residual-pool initialization, (c) codebook quantizer."
        ),
    ),
}


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def validate_strategies(strategies: Sequence[str]) -> int:
    """
    Validate a strategy list and return the common bit width.

    Checks (in order):
      1. List is non-empty.
      2. Length is a power of 2.
      3. All names are in AVAILABLE_STRATEGIES.
      4. All strategies share a bit width.
      5. All strategies are actually implemented.

    Raises:
        ValueError: for problems #1–#4.
        NotImplementedError: if any strategy is in the catalog but not yet implemented.
    """
    if not strategies:
        raise ValueError("strategies must be a non-empty sequence.")

    k = len(strategies)
    if not _is_power_of_two(k):
        raise ValueError(
            f"Number of strategies must be a power of 2; got {k}. "
            f"Allowed: 1, 2, 4, 8, 16."
        )

    unknown = [s for s in strategies if s not in AVAILABLE_STRATEGIES]
    if unknown:
        raise ValueError(
            f"Unknown strategies: {unknown}. "
            f"Available: {sorted(AVAILABLE_STRATEGIES.keys())}"
        )

    bit_widths = {AVAILABLE_STRATEGIES[s].bits for s in strategies}
    if len(bit_widths) > 1:
        raise ValueError(
            f"All strategies must share a bit width; got widths {sorted(bit_widths)} "
            f"from strategies {[(s, AVAILABLE_STRATEGIES[s].bits) for s in strategies]}. "
            f"Adaptive selection only makes sense within a fixed bit budget."
        )

    unimplemented = [s for s in strategies if not AVAILABLE_STRATEGIES[s].implemented]
    if unimplemented:
        details = "\n".join(
            f"  - {s}: {AVAILABLE_STRATEGIES[s].description}"
            for s in unimplemented
        )
        raise NotImplementedError(
            f"The following strategies are in the catalog but not yet implemented:\n"
            f"{details}\n"
            f"Implemented strategies: "
            f"{[s for s, info in AVAILABLE_STRATEGIES.items() if info.implemented]}"
        )

    return bit_widths.pop()
