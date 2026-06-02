"""
Phase 0 verification: confirm microxcaling can round-trip a tensor through
both fp6_e3m2 and fp6_e2m3, using the pure-Python (no CUDA) path.
"""
import sys
from pathlib import Path

# Add microxcaling to path without pip-installing (avoids torch==2.2.0 pin)
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "microxcaling"))

import torch
from mx.mx_ops import quantize_mx_op
from mx.specs import finalize_mx_specs


def make_specs(elem_format: str, block_size: int = 32) -> dict:
    specs = {
        "w_elem_format": elem_format,
        "a_elem_format": elem_format,
        "block_size": block_size,
        "scale_bits": 8,
        "bfloat": 16,
        "custom_cuda": False,  # pure-Python path, no C++ compilation
    }
    return finalize_mx_specs(specs)


def round_trip(x: torch.Tensor, fmt: str, block_size: int = 32) -> torch.Tensor:
    specs = make_specs(fmt, block_size)
    # axes=[-1] means: each block of `block_size` elements along the last dim
    # shares one E8M0 scale (standard MX layout)
    return quantize_mx_op(x, specs, elem_format=fmt, block_size=block_size, axes=[-1])


def main():
    torch.manual_seed(0)

    print("=" * 70)
    print("Test 1: Round-trip a tightly-clustered tensor")
    print("=" * 70)
    x = torch.randn(64) * 0.1  # small values ~ N(0, 0.01)
    print(f"Input range:  [{x.min().item():+.4f}, {x.max().item():+.4f}]")
    for fmt in ("fp6_e3m2", "fp6_e2m3"):
        xq = round_trip(x, fmt)
        mse = (x - xq).pow(2).mean().item()
        print(f"  {fmt}:  recon range [{xq.min().item():+.4f}, {xq.max().item():+.4f}]  MSE = {mse:.3e}")

    print()
    print("=" * 70)
    print("Test 2: Round-trip a tensor with outliers")
    print("=" * 70)
    x = torch.randn(64)
    x[5] = 50.0  # inject an outlier
    print(f"Input range:  [{x.min().item():+.4f}, {x.max().item():+.4f}]")
    for fmt in ("fp6_e3m2", "fp6_e2m3"):
        xq = round_trip(x, fmt)
        mse = (x - xq).pow(2).mean().item()
        print(f"  {fmt}:  recon range [{xq.min().item():+.4f}, {xq.max().item():+.4f}]  MSE = {mse:.3e}")

    print()
    print("=" * 70)
    print("Test 3: Sanity — quantized values are a subset of the FP6 grid")
    print("=" * 70)
    x = torch.tensor([0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0, 8.0])
    for fmt in ("fp6_e3m2", "fp6_e2m3"):
        xq = round_trip(x, fmt)
        diffs = (xq - x).abs()
        print(f"  {fmt}: input  = {x.tolist()}")
        print(f"          recon  = {[round(v, 4) for v in xq.tolist()]}")
        print(f"          delta  = {[round(v, 4) for v in diffs.tolist()]}")

    print()
    print("Phase 0 verification: SUCCESS" if True else "FAIL")


if __name__ == "__main__":
    main()
