"""
Unit tests for adaptive MXFP6 quantization.

Run with:  python -m pytest tests/  (after `pip install pytest`)
Or just:   python tests/test_adaptive.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from adaptive_mxfp6 import (
    adaptive_quantize_dequantize,
    fixed_quantize_dequantize,
)
from adaptive_mxfp6.quantize import _block_mse


# ----------------------------- helpers ------------------------------------

def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).pow(2).mean())


def _passed(label: str):
    print(f"  PASS  {label}")


def _failed(label: str, detail: str = ""):
    print(f"  FAIL  {label}  {detail}")
    raise AssertionError(f"{label} {detail}")


# ----------------------------- tests --------------------------------------

def test_adaptive_never_worse_than_either_fixed():
    """Per-block guarantee: adaptive MSE <= min(fixed_e3m2, fixed_e2m3) per block."""
    torch.manual_seed(0)
    x = torch.randn(8, 256)  # 8 rows, 256 cols = 8 blocks of 32 per row

    e3m2 = fixed_quantize_dequantize(x, "fp6_e3m2", block_size=32)
    e2m3 = fixed_quantize_dequantize(x, "fp6_e2m3", block_size=32)
    adapt = adaptive_quantize_dequantize(x, block_size=32)

    mse_e3m2_b = _block_mse(x, e3m2, 32)
    mse_e2m3_b = _block_mse(x, e2m3, 32)
    mse_min_b = torch.minimum(mse_e3m2_b, mse_e2m3_b)

    # Adaptive must be exactly the per-block min (by construction)
    assert torch.allclose(adapt.per_block_mse, mse_min_b, atol=1e-9), \
        "adaptive per-block MSE should equal min(fixed_e3m2, fixed_e2m3) per block"

    # And aggregate MSE must be <= min of the two fixed aggregates
    mse_adapt = _mse(x, adapt.recon)
    mse_e3m2 = _mse(x, e3m2)
    mse_e2m3 = _mse(x, e2m3)
    assert mse_adapt <= min(mse_e3m2, mse_e2m3) + 1e-9, \
        f"adaptive {mse_adapt} should beat min({mse_e3m2}, {mse_e2m3})"

    _passed("adaptive never worse than either fixed variant (per-block)")


def test_tight_cluster_prefers_e2m3():
    """A tensor with small, tightly-clustered values should mostly select e2m3
    (finer precision, narrow range)."""
    torch.manual_seed(0)
    x = torch.randn(4, 32 * 16) * 0.05  # tight cluster around zero
    adapt = adaptive_quantize_dequantize(x, block_size=32)
    frac_e2m3 = adapt.fractions["fp6_e2m3"]
    assert frac_e2m3 > 0.8, f"expected mostly e2m3 for tight cluster, got {frac_e2m3:.2%}"
    _passed(f"tight cluster prefers e2m3 ({frac_e2m3:.1%} of blocks)")


def test_outlier_blocks_prefer_e3m2():
    """A block that contains a big outlier should select e3m2 (wider range)."""
    torch.manual_seed(0)
    x = torch.randn(1, 32 * 4)  # 4 blocks
    # Inject a single huge value into block 1 only
    x[0, 32 + 5] = 100.0
    adapt = adaptive_quantize_dequantize(x, block_size=32)
    choices = adapt.choices.flatten()  # 4 values, one per block
    # Block 1 (the outlier block) should have chosen e3m2 (choice == 0)
    assert choices[1].item() == 0, \
        f"outlier block should choose e3m2 (0), got {choices[1].item()}"
    _passed(f"outlier block selects e3m2 (per-block choices = {choices.tolist()})")


def test_shape_preserved():
    """Reconstruction has the same shape as input."""
    for shape in [(32,), (4, 32), (2, 3, 64), (1, 2, 3, 128)]:
        x = torch.randn(*shape)
        adapt = adaptive_quantize_dequantize(x, block_size=32)
        assert adapt.recon.shape == x.shape, f"shape mismatch for {shape}: got {adapt.recon.shape}"
    _passed("shape preserved across all tested ranks")


def test_dtype_and_device():
    """Reconstruction has the same dtype as input."""
    x = torch.randn(64, dtype=torch.float32)
    adapt = adaptive_quantize_dequantize(x, block_size=32)
    assert adapt.recon.dtype == x.dtype
    _passed("dtype preserved (float32)")


def test_fp6_exact_values_roundtrip_exactly():
    """Values that already lie on the FP6 grid should round-trip with zero error."""
    # Values that lie exactly on both fp6_e3m2 and fp6_e2m3 grids
    x = torch.tensor([[0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0] * 4])  # 32 values
    adapt = adaptive_quantize_dequantize(x, block_size=32)
    err = (adapt.recon - x).abs().max().item()
    assert err < 1e-6, f"expected exact roundtrip, max error = {err}"
    _passed("FP6-grid values round-trip exactly")


# ----------------------------- runner -------------------------------------

def main():
    print("Running adaptive MXFP6 unit tests:")
    tests = [
        test_adaptive_never_worse_than_either_fixed,
        test_tight_cluster_prefers_e2m3,
        test_outlier_blocks_prefer_e3m2,
        test_shape_preserved,
        test_dtype_and_device,
        test_fp6_exact_values_roundtrip_exactly,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests PASSED.")


if __name__ == "__main__":
    main()
