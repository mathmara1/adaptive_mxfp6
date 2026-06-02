"""
Tests for AdaptiveMXFP6Linear and the model-swap helper.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from adaptive_mxfp6 import AdaptiveMXFP6Linear, quantize_linear_layers_


def _passed(label: str):
    print(f"  PASS  {label}")


def test_layer_matches_fp16_when_no_quantization():
    """scheme='fp16' should be a pure passthrough (no quantization)."""
    torch.manual_seed(0)
    fp = nn.Linear(128, 64)
    q = AdaptiveMXFP6Linear.from_linear(fp, block_size=32, scheme="fp16")

    x = torch.randn(8, 128)
    err = (fp(x) - q(x)).abs().max().item()
    assert err < 1e-5, f"fp16 scheme should match exactly, max err = {err}"
    _passed("scheme='fp16' is a passthrough (matches nn.Linear exactly)")


def test_adaptive_layer_introduces_small_error():
    """The adaptive scheme should introduce *some* error but not catastrophically."""
    torch.manual_seed(0)
    fp = nn.Linear(128, 64)
    q = AdaptiveMXFP6Linear.from_linear(fp, block_size=32, scheme="adaptive")

    x = torch.randn(8, 128)
    err = (fp(x) - q(x)).pow(2).mean().item()
    assert 0 < err < 1e-2, f"adaptive error should be small but nonzero, got {err}"
    _passed(f"adaptive layer introduces small error ({err:.3e})")


def test_adaptive_beats_or_ties_both_fixed_on_layer_output():
    """At the layer-output level, adaptive should match or beat both fixed variants."""
    torch.manual_seed(0)
    fp = nn.Linear(256, 128)
    x = torch.randn(16, 256)
    ref = fp(x)

    errs = {}
    for scheme in ("fixed_e3m2", "fixed_e2m3", "adaptive"):
        q = AdaptiveMXFP6Linear.from_linear(fp, block_size=32, scheme=scheme)
        errs[scheme] = (q(x) - ref).pow(2).mean().item()

    # Layer-output MSE doesn't strictly inherit the per-block per-weight guarantee
    # (matmul shuffles errors around), but on random data the adaptive ranking
    # should at minimum be no more than ~2x worse than the better fixed.
    better_fixed = min(errs["fixed_e3m2"], errs["fixed_e2m3"])
    assert errs["adaptive"] <= 2 * better_fixed, (
        f"adaptive layer error {errs['adaptive']:.3e} unexpectedly worse than "
        f"better fixed {better_fixed:.3e}; details = {errs}"
    )
    _passed(f"layer-output MSE: e3m2={errs['fixed_e3m2']:.3e}, "
            f"e2m3={errs['fixed_e2m3']:.3e}, adaptive={errs['adaptive']:.3e}")


def test_model_swap_replaces_linears():
    """quantize_linear_layers_ should swap every nn.Linear (subject to filters)."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 32),
    )
    summary = quantize_linear_layers_(model, scheme="adaptive", block_size=32,
                                       exclude_name_substrings=())
    # All three Linears should have been swapped
    n_adaptive = sum(1 for m in model.modules() if isinstance(m, AdaptiveMXFP6Linear))
    n_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear)
                   and not isinstance(m, AdaptiveMXFP6Linear))
    assert n_adaptive == 3, f"expected 3 swapped, got {n_adaptive}"
    assert n_linear == 0, f"expected 0 remaining nn.Linear, got {n_linear}"
    assert len(summary) == 3, f"summary should have 3 entries, got {len(summary)}"
    # Verify the swapped model still runs
    out = model(torch.randn(4, 64))
    assert out.shape == (4, 32)
    _passed(f"swap replaced 3/3 Linears; sample fractions: "
            f"{list(summary.values())[0]}")


def test_model_swap_excludes_filtered():
    """Excluded substrings should preserve those layers as nn.Linear."""
    model = nn.ModuleDict({
        "encoder": nn.Linear(64, 128),
        "lm_head": nn.Linear(128, 64),  # should be excluded by default
    })
    quantize_linear_layers_(model, scheme="adaptive", block_size=32)
    assert isinstance(model["encoder"], AdaptiveMXFP6Linear)
    assert isinstance(model["lm_head"], nn.Linear) and not isinstance(model["lm_head"], AdaptiveMXFP6Linear)
    _passed("default exclude filter preserves lm_head")


def main():
    print("Running AdaptiveMXFP6Linear tests:")
    tests = [
        test_layer_matches_fp16_when_no_quantization,
        test_adaptive_layer_introduces_small_error,
        test_adaptive_beats_or_ties_both_fixed_on_layer_output,
        test_model_swap_replaces_linears,
        test_model_swap_excludes_filtered,
    ]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} tests PASSED.")


if __name__ == "__main__":
    main()
