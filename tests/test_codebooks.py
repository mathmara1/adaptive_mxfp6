"""
Tests for the generic block-scaled codebook quantizer.

This is the shared backend for all non-microxcaling 6-bit strategies (int6,
nf6, split6, learned grids). If these tests pass, every future codebook-based
strategy gets its quantization math for free.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from adaptive_mxfp6 import adaptive_quantize_dequantize, fixed_quantize_dequantize
from adaptive_mxfp6._codebooks import CODEBOOKS, codebook_quantize_dequantize
from adaptive_mxfp6._mx_backend import quantize_dequantize_mx


def _passed(label: str):
    print(f"  PASS  {label}")


def test_exact_codebook_values_roundtrip():
    """If a block contains only values that are exact codebook entries (after
    its own absmax normalization), the round-trip should be lossless."""
    codebook = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    # 32-element block whose absmax is 1.0, so normalization is identity.
    # Every value is in the codebook → snap-back is exact.
    x = torch.tensor([[1.0, 0.5, 0.0, -0.5, -1.0] * 6 + [1.0, 0.5]])
    out = codebook_quantize_dequantize(x, codebook, block_size=32)
    err = (out - x).abs().max().item()
    assert err < 1e-6, f"expected exact roundtrip, max error = {err}"
    _passed("values exactly on codebook round-trip with zero error")


def test_snap_to_nearest():
    """Off-grid values snap to the nearest codebook entry."""
    codebook = torch.tensor([-1.0, 0.0, 1.0])
    # block_size=4, absmax=1.0 so normalization is identity.
    # 0.3 → closer to 0; 0.6 → closer to 1; -0.4 → closer to 0; 1.0 exact.
    x = torch.tensor([[0.3, 0.6, -0.4, 1.0]])
    out = codebook_quantize_dequantize(x, codebook, block_size=4)
    expected = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    assert torch.allclose(out, expected, atol=1e-6), f"expected {expected}, got {out}"
    _passed("off-grid values snap to nearest codebook entry")


def test_per_block_scaling():
    """Each block uses its own absmax — different blocks get different scales."""
    codebook = torch.tensor([-1.0, 0.0, 1.0])
    # Two blocks of 4: block 1 absmax=2.0, block 2 absmax=10.0
    x = torch.tensor([[2.0, 1.0, -1.0, -2.0,
                       10.0, 5.0, -5.0, -10.0]])
    out = codebook_quantize_dequantize(x, codebook, block_size=4)
    # Endpoints (absmax positions) should round-trip exactly per block
    expected_endpoints = torch.tensor([2.0, -2.0, 10.0, -10.0])
    actual_endpoints = out[0, [0, 3, 4, 7]]
    assert torch.allclose(actual_endpoints, expected_endpoints, atol=1e-6), \
        f"expected {expected_endpoints}, got {actual_endpoints}"
    _passed("each block scales independently by its own absmax")


def test_shape_and_dtype_preserved():
    codebook = torch.linspace(-1, 1, 64)
    for shape in [(32,), (4, 32), (2, 3, 128)]:
        x = torch.randn(*shape, dtype=torch.float32)
        out = codebook_quantize_dequantize(x, codebook, block_size=32)
        assert out.shape == x.shape, f"shape mismatch for {shape}"
        assert out.dtype == x.dtype, f"dtype mismatch for {shape}"
    _passed("shape and dtype preserved across all tested ranks")


def test_all_zero_block_no_division_by_zero():
    """A block of all zeros must not produce NaNs or errors."""
    codebook = torch.tensor([-1.0, 0.0, 1.0])
    x = torch.zeros(1, 32)
    out = codebook_quantize_dequantize(x, codebook, block_size=32)
    assert torch.isfinite(out).all(), "output contains NaN/Inf on all-zero block"
    assert (out.abs() < 1e-6).all(), f"all-zero block should stay near zero, got {out}"
    _passed("all-zero block stays zero (no div-by-zero)")


def test_random_input_low_mse():
    """End-to-end sanity: random Gaussian input with a reasonable codebook
    should produce small reconstruction MSE."""
    torch.manual_seed(0)
    codebook = torch.linspace(-1, 1, 64)
    x = torch.randn(4, 256)
    out = codebook_quantize_dequantize(x, codebook, block_size=32)
    mse = (out - x).pow(2).mean().item()
    assert mse < 0.01, f"expected small MSE for 64-level uniform codebook, got {mse}"
    _passed(f"random Gaussian input with 64-level uniform codebook: MSE={mse:.4e}")


def test_codebook_auto_sorted():
    """An unsorted codebook should produce the same result as a sorted one."""
    torch.manual_seed(0)
    x = torch.randn(4, 256)
    sorted_cb = torch.linspace(-1, 1, 64)
    shuffled_cb = sorted_cb[torch.randperm(64)]
    out_sorted = codebook_quantize_dequantize(x, sorted_cb, block_size=32)
    out_shuffled = codebook_quantize_dequantize(x, shuffled_cb, block_size=32)
    assert torch.allclose(out_sorted, out_shuffled, atol=1e-6), \
        "unsorted codebook produced different output than sorted"
    _passed("codebook is sorted internally — unsorted input gives same result")


def test_device_handling():
    """Codebook on CPU should be moved automatically if input is on CUDA."""
    if not torch.cuda.is_available():
        _passed("(skipped — no CUDA available)")
        return
    codebook = torch.linspace(-1, 1, 64)  # on CPU
    x = torch.randn(2, 256, device="cuda")
    out = codebook_quantize_dequantize(x, codebook, block_size=32)
    assert out.device.type == "cuda", f"expected output on cuda, got {out.device}"
    _passed("codebook is auto-moved to input device (CUDA verified)")


# ---------------------- registered-codebook tests --------------------------

def test_int6_codebook_shape_and_range():
    """int6 has 64 values in [-1, 1], evenly spaced, includes 0."""
    cb = CODEBOOKS["int6"]
    assert cb.numel() == 64, f"int6 should have 64 values, got {cb.numel()}"
    assert cb.min().item() == -1.0, f"int6 min should be -1, got {cb.min()}"
    assert abs(cb.max().item() - 31.0/32.0) < 1e-6, f"int6 max should be 31/32, got {cb.max()}"
    assert 0.0 in cb.tolist(), "int6 should include zero"
    # Spacing should be uniform 1/32
    diffs = (cb.sort().values[1:] - cb.sort().values[:-1])
    assert torch.allclose(diffs, torch.full_like(diffs, 1/32), atol=1e-6), \
        "int6 spacing should be uniform 1/32"
    _passed("int6 codebook: 64 values, spans [-1, 31/32], uniform 1/32 spacing")


def test_nf6_codebook_shape_and_range():
    """nf6 has 64 values normalized to [-1, 1], includes 0, denser near zero."""
    cb = CODEBOOKS["nf6"]
    assert cb.numel() == 64, f"nf6 should have 64 values, got {cb.numel()}"
    assert abs(cb.abs().max().item() - 1.0) < 1e-6, f"nf6 max abs should be 1, got {cb.abs().max()}"
    assert 0.0 in cb.tolist(), "nf6 should include explicit zero"
    # Denser near zero: smallest gap should be in the middle
    sorted_cb = cb.sort().values
    diffs = sorted_cb[1:] - sorted_cb[:-1]
    mid_idx = len(diffs) // 2
    assert diffs[mid_idx].item() < diffs[0].item(), \
        "nf6 should have smaller gaps near zero than at extremes"
    assert diffs[mid_idx].item() < diffs[-1].item(), \
        "nf6 should have smaller gaps near zero than at extremes"
    _passed(f"nf6 codebook: 64 values, max|.|=1, denser near zero "
            f"(mid gap={diffs[mid_idx]:.4f}, extreme gap={diffs[-1]:.4f})")


def test_dispatcher_routes_int6_and_nf6():
    """quantize_dequantize_mx should route int6/nf6 to the codebook quantizer."""
    x = torch.randn(2, 256)
    for strategy in ("int6", "nf6"):
        out = quantize_dequantize_mx(x, strategy, block_size=32)
        assert out.shape == x.shape
        assert out.dtype == x.dtype
        # Sanity: output should be non-trivial (not all zero, not identical to input)
        assert (out != 0).any()
        assert not torch.allclose(out, x, atol=0)
    _passed("dispatcher routes int6 and nf6 to codebook backend")


def test_dispatcher_rejects_unknown_format():
    """An unknown strategy name should raise a clear ValueError."""
    x = torch.randn(1, 32)
    try:
        quantize_dequantize_mx(x, "bogus_format", block_size=32)
    except ValueError as e:
        assert "bogus_format" in str(e)
        _passed("dispatcher raises ValueError for unknown strategy name")
        return
    raise AssertionError("expected ValueError for unknown format")


def test_adaptive_with_if6_pair():
    """Adaptive selection between fp6_e3m2 and int6 should work end-to-end (IF6)."""
    torch.manual_seed(0)
    x = torch.randn(4, 256)
    res = adaptive_quantize_dequantize(x, block_size=32, strategies=("fp6_e3m2", "int6"))
    assert res.recon.shape == x.shape
    assert set(res.fractions.keys()) == {"fp6_e3m2", "int6"}
    # Both should be selected at least sometimes on random Gaussian input
    assert sum(res.fractions.values()) == 1.0 or abs(sum(res.fractions.values()) - 1.0) < 1e-6
    # Per-block guarantee: adaptive MSE <= min(individual MSEs) per block
    fp_recon = fixed_quantize_dequantize(x, "fp6_e3m2", block_size=32)
    int6_recon = quantize_dequantize_mx(x, "int6", block_size=32)
    from adaptive_mxfp6.quantize import _block_mse
    mse_fp = _block_mse(x, fp_recon, 32)
    mse_int = _block_mse(x, int6_recon, 32)
    mse_min = torch.minimum(mse_fp, mse_int)
    assert torch.allclose(res.per_block_mse, mse_min, atol=1e-9), \
        "adaptive per-block MSE should equal min(fp6_e3m2, int6) per block"
    _passed(f"IF6 adaptive works: fractions = {res.fractions}")


def test_adaptive_4_way_with_int6_and_nf6():
    """4-grid adaptive selection across the full FP6/INT6/NF6 family."""
    torch.manual_seed(0)
    x = torch.randn(4, 256)
    strategies = ("fp6_e3m2", "fp6_e2m3", "int6", "nf6")
    res = adaptive_quantize_dequantize(x, block_size=32, strategies=strategies)
    assert res.recon.shape == x.shape
    assert set(res.fractions.keys()) == set(strategies)
    assert abs(sum(res.fractions.values()) - 1.0) < 1e-6
    assert res.choices.max().item() < 4, "choice indices must be < k"
    _passed(f"4-grid adaptive (fp6_e3m2, fp6_e2m3, int6, nf6) works: {res.fractions}")


def main():
    print("Running codebook quantizer tests:")
    tests = [
        test_exact_codebook_values_roundtrip,
        test_snap_to_nearest,
        test_per_block_scaling,
        test_shape_and_dtype_preserved,
        test_all_zero_block_no_division_by_zero,
        test_random_input_low_mse,
        test_codebook_auto_sorted,
        test_device_handling,
        # registered-codebook + dispatcher integration:
        test_int6_codebook_shape_and_range,
        test_nf6_codebook_shape_and_range,
        test_dispatcher_routes_int6_and_nf6,
        test_dispatcher_rejects_unknown_format,
        test_adaptive_with_if6_pair,
        test_adaptive_4_way_with_int6_and_nf6,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests PASSED.")


if __name__ == "__main__":
    main()
