"""
Tests for the strategies catalog and the generalized k-way selector.

Covers:
  - validate_strategies error cases (empty, non-power-of-2, unknown, mixed bits)
  - 1-strategy (fixed) path via adaptive_quantize_dequantize
  - 2-strategy default path still behaves identically to the original library
  - AdaptiveResult.fractions structure and consistency
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from adaptive_mxfp6 import (
    AVAILABLE_STRATEGIES,
    adaptive_quantize_dequantize,
    fixed_quantize_dequantize,
    validate_strategies,
)


def _passed(label: str):
    print(f"  PASS  {label}")


# ----------------------------- validation tests ---------------------------

def test_validate_rejects_empty():
    try:
        validate_strategies([])
    except ValueError as e:
        assert "non-empty" in str(e)
        _passed("validate_strategies rejects empty list")
        return
    raise AssertionError("expected ValueError on empty strategies list")


def test_validate_rejects_non_power_of_two():
    for bad in [3, 5, 6, 7]:
        try:
            validate_strategies(["fp6_e3m2"] * bad)
        except ValueError as e:
            assert "power of 2" in str(e), f"unexpected message: {e}"
        else:
            raise AssertionError(f"expected ValueError for length {bad}")
    _passed("validate_strategies rejects non-power-of-2 lengths (3, 5, 6, 7)")


def test_validate_rejects_unknown_strategy():
    try:
        validate_strategies(["fp6_e3m2", "fp4_e2m1"])  # fp4 not in 6-bit-only catalog
    except ValueError as e:
        assert "Unknown" in str(e)
        _passed("validate_strategies rejects unknown strategy")
        return
    raise AssertionError("expected ValueError for unknown strategy")


def test_validate_returns_bit_width():
    bits = validate_strategies(["fp6_e3m2", "fp6_e2m3"])
    assert bits == 6, f"expected bits=6, got {bits}"
    _passed("validate_strategies returns common bit width (6)")


# ----------------------------- selector tests -----------------------------

def test_single_strategy_matches_fixed():
    """Calling adaptive with strategies=[only] should equal calling fixed with the same one."""
    torch.manual_seed(0)
    x = torch.randn(4, 256)
    for fmt in ("fp6_e3m2", "fp6_e2m3"):
        adapt = adaptive_quantize_dequantize(x, strategies=[fmt], block_size=32)
        fixed = fixed_quantize_dequantize(x, fmt, block_size=32)
        assert torch.allclose(adapt.recon, fixed, atol=0.0), \
            f"single-strategy adaptive should equal fixed {fmt}"
        assert adapt.strategy_names == (fmt,)
        assert adapt.fractions == {fmt: 1.0}
    _passed("single-strategy adaptive equals fixed (both variants)")


def test_default_strategies_unchanged():
    """Default 2-grid behavior is preserved (no strategies kwarg required)."""
    torch.manual_seed(0)
    x = torch.randn(8, 256)
    a = adaptive_quantize_dequantize(x, block_size=32)
    b = adaptive_quantize_dequantize(
        x, block_size=32, strategies=("fp6_e3m2", "fp6_e2m3")
    )
    assert torch.allclose(a.recon, b.recon, atol=0.0)
    assert a.strategy_names == ("fp6_e3m2", "fp6_e2m3")
    _passed("default strategies preserve original 2-grid behavior")


def test_fractions_sum_to_one():
    torch.manual_seed(0)
    x = torch.randn(4, 32 * 100)
    adapt = adaptive_quantize_dequantize(x, block_size=32)
    total = sum(adapt.fractions.values())
    assert abs(total - 1.0) < 1e-6, f"fractions should sum to 1, got {total}"
    assert set(adapt.fractions.keys()) == set(adapt.strategy_names)
    _passed("fractions dict sums to 1 and covers all strategies")


def test_choices_in_valid_range():
    """All choices must be valid strategy indices."""
    torch.manual_seed(0)
    x = torch.randn(4, 256)
    strategies = ("fp6_e3m2", "fp6_e2m3")
    adapt = adaptive_quantize_dequantize(x, block_size=32, strategies=strategies)
    k = len(strategies)
    assert adapt.choices.min().item() >= 0
    assert adapt.choices.max().item() < k
    _passed(f"choices stay within [0, {k - 1}]")


def test_catalog_has_known_entries():
    assert "fp6_e3m2" in AVAILABLE_STRATEGIES
    assert "fp6_e2m3" in AVAILABLE_STRATEGIES
    assert AVAILABLE_STRATEGIES["fp6_e3m2"].bits == 6
    assert AVAILABLE_STRATEGIES["fp6_e2m3"].bits == 6
    _passed("catalog contains both 6-bit MXFP6 variants")


def test_planned_strategies_present_in_catalog():
    """Planned 6-bit strategies appear in the catalog (so they're discoverable)
    but are flagged as not yet implemented."""
    planned = ["int6", "nf6", "split6", "sfp6_shift_pos", "sfp6_shift_neg", "learned_residual"]
    for name in planned:
        assert name in AVAILABLE_STRATEGIES, f"missing planned entry: {name}"
        info = AVAILABLE_STRATEGIES[name]
        assert info.bits == 6, f"{name} should be 6-bit, got {info.bits}"
        assert info.implemented is False, f"{name} should be marked not implemented"
        assert "TODO" in info.description, f"{name} description should contain TODO note"
    _passed(f"all {len(planned)} planned 6-bit strategies are catalogued with implemented=False")


def test_validate_rejects_unimplemented():
    """Selecting a planned-but-unimplemented strategy raises NotImplementedError."""
    try:
        validate_strategies(["int6"])
    except NotImplementedError as e:
        assert "int6" in str(e)
        assert "TODO" in str(e)
    else:
        raise AssertionError("expected NotImplementedError for int6")

    # Also rejects when paired with an implemented one
    try:
        validate_strategies(["fp6_e3m2", "int6"])
    except NotImplementedError as e:
        assert "int6" in str(e)
    else:
        raise AssertionError("expected NotImplementedError for [fp6_e3m2, int6]")

    _passed("validate_strategies rejects unimplemented strategies with NotImplementedError")


def test_implemented_strategies_still_pass_validation():
    """Sanity: after the catalog expansion, the implemented set still validates."""
    bits = validate_strategies(["fp6_e3m2", "fp6_e2m3"])
    assert bits == 6
    bits = validate_strategies(["fp6_e3m2"])
    assert bits == 6
    _passed("implemented strategies still validate after catalog expansion")


# ----------------------------- runner -------------------------------------

def main():
    print("Running strategies / k-way selector tests:")
    tests = [
        test_validate_rejects_empty,
        test_validate_rejects_non_power_of_two,
        test_validate_rejects_unknown_strategy,
        test_validate_returns_bit_width,
        test_single_strategy_matches_fixed,
        test_default_strategies_unchanged,
        test_fractions_sum_to_one,
        test_choices_in_valid_range,
        test_catalog_has_known_entries,
        test_planned_strategies_present_in_catalog,
        test_validate_rejects_unimplemented,
        test_implemented_strategies_still_pass_validation,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests PASSED.")


if __name__ == "__main__":
    main()
