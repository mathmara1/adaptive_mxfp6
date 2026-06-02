# Adaptive MXFP6 — Full Walkthrough

A first-principles explanation of every file, every decision, and every line in
this repository. Read top-to-bottom on first pass, then use as a reference.

---

## Table of Contents

1. [The Problem We're Solving](#1-the-problem-were-solving)
2. [The Big Idea in One Page](#2-the-big-idea-in-one-page)
3. [Repository Layout](#3-repository-layout)
4. [How the Pieces Fit Together](#4-how-the-pieces-fit-together)
5. [File-by-File Walkthrough](#5-file-by-file-walkthrough)
   - [verify_mxfp6.py](#51-verify_mxfp6py--phase-0-sanity-check)
   - [adaptive_mxfp6/_mx_backend.py](#52-adaptive_mxfp6_mx_backendpy)
   - [adaptive_mxfp6/quantize.py](#53-adaptive_mxfp6quantizepy--the-core-novelty)
   - [adaptive_mxfp6/layers.py](#54-adaptive_mxfp6layerspy)
   - [adaptive_mxfp6/__init__.py](#55-adaptive_mxfp6__init__py)
   - [tests/test_adaptive.py](#56-teststest_adaptivepy)
   - [tests/test_layers.py](#57-teststest_layerspy)
6. [How to Use It Yourself](#6-how-to-use-it-yourself)
7. [Design Decisions and Why](#7-design-decisions-and-why)
8. [What's Missing (Next Steps)](#8-whats-missing-next-steps)

---

## 1. The Problem We're Solving

### Quantization in one paragraph

Modern LLMs store weights in 16-bit floats (FP16 / BF16). For inference, we'd
like to use **fewer bits per weight** — 4, 6, or 8 — to save memory and run
faster on dedicated low-precision hardware (NVIDIA Blackwell, AMD MI350).
Reducing bits means we can only represent a small set of "grid points" — every
weight must be rounded to its nearest grid point, introducing error. The art
is choosing a grid that minimizes that error for your data.

### MXFP6 in one paragraph

The OCP Microscaling (MX) standard defines a family of block-scaled formats.
**MXFP6** groups 32 weights into a block, stores one shared 8-bit exponent per
block (the "scale"), and stores each weight as a 6-bit floating-point number.
The 6-bit element format comes in two variants:

- **`fp6_e3m2`** — 3 exponent bits, 2 mantissa bits → wider range, coarser steps
- **`fp6_e2m3`** — 2 exponent bits, 3 mantissa bits → narrower range, finer steps

Both variants can represent 64 values. Neither dominates the other — which is
better depends on what the block's actual data looks like.

### The gap this project fills

In the standard OCP spec, you pick **one** variant globally and use it for
every block. But:

- A block with a few extreme outliers needs `e3m2`'s wider range.
- A block where everything is similar in magnitude benefits from `e2m3`'s finer
  precision.

A single global choice forces a compromise. **What if each block could pick
its own variant?** That's *adaptive MXFP6* — the contribution this repository
prototypes. It costs one extra bit per block to encode the choice (so storage
goes from 200 to 201 bits per block, < 0.5% overhead) and is guaranteed by
construction to be at least as accurate as either fixed variant.

This idea is inspired by two recent papers (IF4 from arxiv 2603.28765 and Grid
Games from arxiv 2605.12327) which showed that per-block grid selection beats
single-grid formats at 4 bits. Nobody has published the equivalent result at
6 bits using the two existing OCP MXFP6 variants — that's the gap.

---

## 2. The Big Idea in One Page

The whole adaptive scheme is one decision per block:

```
For each block of 32 weights in your tensor:
    1. Quantize the block with fp6_e3m2 → get reconstruction → measure MSE
    2. Quantize the block with fp6_e2m3 → get reconstruction → measure MSE
    3. Keep whichever reconstruction has lower MSE
    4. Record which variant was chosen (1 bit per block)
```

That's it. The "library" is the machinery to make this work end-to-end:

- **A quantization primitive** that does steps 1–4 on a tensor
- **A drop-in `nn.Linear` replacement** that uses the primitive
- **A helper to swap every Linear in a model**
- **Tests** to prove the primitive works as advertised

Once you have those four things, you can quantize any PyTorch model and
measure accuracy. That's what we built today.

---

## 3. Repository Layout

```
adaptive-mxfp6/
├── .venv/                          # Python 3.14 virtualenv (torch 2.12, numpy)
├── microxcaling/                   # cloned from microsoft/microxcaling (untouched)
├── adaptive_mxfp6/                 # OUR LIBRARY
│   ├── __init__.py                 # public API exports
│   ├── _mx_backend.py              # thin wrapper around microxcaling
│   ├── quantize.py                 # the core adaptive selector
│   └── layers.py                   # AdaptiveMXFP6Linear + model-swap helper
├── tests/
│   ├── test_adaptive.py            # tests for the quantizer (6 tests)
│   └── test_layers.py              # tests for the Linear layer (5 tests)
├── verify_mxfp6.py                 # Phase 0 sanity script
└── WALKTHROUGH.md                  # this file
```

**Why `_mx_backend.py` starts with an underscore:** Python convention — the
leading underscore signals "internal, don't import this from outside." Users
should only touch the `adaptive_mxfp6.*` public symbols listed in
`__init__.py`. If we ever swap microxcaling for a different backend, only
`_mx_backend.py` changes.

---

## 4. How the Pieces Fit Together

Here's the call graph from a user's perspective:

```
User code:
    quantize_linear_layers_(model, scheme="adaptive")
            │
            ▼
    AdaptiveMXFP6Linear.from_linear(old_layer, scheme="adaptive")
            │
            ▼
    adaptive_quantize_dequantize(weight, block_size=32)
            │
            ├──► quantize_dequantize_mx(weight, "fp6_e3m2", 32) ──► microxcaling
            ├──► quantize_dequantize_mx(weight, "fp6_e2m3", 32) ──► microxcaling
            │
            ├──► _block_mse(weight, recon_e3m2)
            ├──► _block_mse(weight, recon_e2m3)
            │
            ├──► torch.where(...)   ← pick per-block winner
            │
            └──► return AdaptiveResult(recon, choices, per_block_mse)
```

The flow is one-way: layers ask the quantizer for a reconstruction; the
quantizer asks the backend (microxcaling) for the per-format
quantize-then-dequantize round-trips; the quantizer compares MSEs and assembles
the adaptive reconstruction.

There is no "dequant at inference time" complexity because we **dequantize
once at load** and store the reconstruction as a regular FP32 tensor. This is
sufficient for accuracy studies (we're simulating, not deploying to real
hardware).

---

## 5. File-by-File Walkthrough

### 5.1 `verify_mxfp6.py` — Phase 0 sanity check

This was the first script we wrote, before any of the library code existed.
Its job: prove microxcaling could quantize a tensor through both FP6 variants
on this machine. If this script worked, we were unblocked.

```python
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "microxcaling"))
```

**What's happening:** Python finds modules by walking through directories
listed in `sys.path`. The microxcaling repo lives in our project under
`./microxcaling/`, but it's not installed via `pip` (more on why below). We
manually prepend it to `sys.path` so `import mx` finds it.

```python
import torch
from mx.mx_ops import quantize_mx_op
from mx.specs import finalize_mx_specs
```

These imports succeed only because of the `sys.path` hack above. `mx` is
microxcaling's package, `mx_ops.quantize_mx_op` is the function that does
block-scaled quantization, and `finalize_mx_specs` builds the config dict
microxcaling needs.

```python
def make_specs(elem_format: str, block_size: int = 32) -> dict:
    specs = {
        "w_elem_format": elem_format,
        "a_elem_format": elem_format,
        "block_size": block_size,
        "scale_bits": 8,
        "bfloat": 16,
        "custom_cuda": False,
    }
    return finalize_mx_specs(specs)
```

The spec dict tells microxcaling:
- `w_elem_format`, `a_elem_format`: use this FP6 variant for both weights and
  activations (we'll only use the weight half, but microxcaling requires both)
- `block_size`: 32 elements share one scale (OCP standard)
- `scale_bits`: 8 bits for the shared exponent (E8M0 — OCP standard)
- `bfloat`: ambient precision for non-quantized ops (irrelevant here)
- `custom_cuda: False`: **the critical line** — skip microxcaling's
  hand-written C++/CUDA kernels and use pure PyTorch. Without this,
  microxcaling tries to JIT-compile `.cpp` and `.cu` files, which needs an
  MSVC toolchain and CUDA toolkit we don't have on this Windows laptop.

`finalize_mx_specs` fills in defaults for the dozens of other fields
microxcaling supports.

```python
def round_trip(x: torch.Tensor, fmt: str, block_size: int = 32) -> torch.Tensor:
    specs = make_specs(fmt, block_size)
    return quantize_mx_op(x, specs, elem_format=fmt, block_size=block_size, axes=[-1])
```

`quantize_mx_op` does **quantize-then-dequantize in one call**. You give it a
tensor; it returns a tensor of the same shape where each value has been
rounded to the nearest representable FP6 value, then scaled back. The
`axes=[-1]` argument means "blocks are formed along the last dimension" —
i.e., for a tensor of shape `(M, K)`, you get `M * (K/32)` blocks of 32
elements each, each block sharing one scale.

The remainder of the script runs three tests and prints results. The output
we observed:

```
Test 1: tight cluster (small values ~N(0, 0.01))
  fp6_e3m2:  MSE = 3.2e-05
  fp6_e2m3:  MSE = 6.4e-06    ← e2m3 wins (finer precision)

Test 2: tensor with a single outlier of magnitude 50
  fp6_e3m2:  MSE = 6.6e-02    ← e3m2 wins (wider range)
  fp6_e2m3:  MSE = 1.0e-01

Test 3: values exactly on the FP6 grid
  both formats reproduce them exactly
```

This was the moment we knew the project was feasible — both variants worked,
both showed the expected trade-off, and the adaptive scheme had something
real to exploit.

### 5.2 `adaptive_mxfp6/_mx_backend.py`

This is a 30-line file whose only job is to **hide microxcaling** behind a
clean function. Future-you should never have to think about microxcaling's API
in any other file.

```python
import sys
from pathlib import Path

_MICROXCALING_DIR = Path(__file__).resolve().parent.parent / "microxcaling"
if str(_MICROXCALING_DIR) not in sys.path:
    sys.path.insert(0, str(_MICROXCALING_DIR))
```

The `sys.path` hack again — same reason as `verify_mxfp6.py`. The `if` check
prevents re-adding the path on every import.

**Why not `pip install -e ./microxcaling`?** Because microxcaling's
`pyproject.toml` declares `torch==2.2.0` as a hard pin. `pip install`-ing it
would try to *downgrade* our `torch==2.12.0` to 2.2.0, which has no wheels
for Python 3.14. The install would fail. By bypassing pip and just adding the
directory to `sys.path`, we use microxcaling's code without invoking its
metadata.

```python
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
```

Same spec dict as the verification script, now wrapped in a helper. Leading
underscore on `_make_specs` says "internal".

```python
def quantize_dequantize_mx(x: torch.Tensor, elem_format: str, block_size: int = 32) -> torch.Tensor:
    """
    Round-trip a tensor through OCP MXFP6 (or any other supported elem format).
    Blocks are taken along the last dimension.
    """
    specs = _make_specs(elem_format, block_size)
    return quantize_mx_op(x, specs, elem_format=elem_format, block_size=block_size, axes=[-1])
```

The single public function this module exposes. It's a one-liner that:
1. Builds a spec dict for the requested format
2. Calls microxcaling's `quantize_mx_op` with `axes=[-1]` (block along last dim)

Everything above this layer talks to `quantize_dequantize_mx` and ignores
microxcaling entirely. If we ever wanted to replace microxcaling (e.g., with
a hand-written MXFP6 implementation, or AMD's Quark), only this file would
change.

### 5.3 `adaptive_mxfp6/quantize.py` — the core novelty

This is the file that implements the actual research idea. Everything else in
the library is plumbing around this.

```python
from dataclasses import dataclass
from typing import Literal, Tuple

import torch

from ._mx_backend import quantize_dequantize_mx
```

Standard imports. `Literal` lets us type-annotate the variant strings so
linters/IDEs can catch typos like `"fp6_3m2"`.

```python
VARIANTS: Tuple[str, str] = ("fp6_e3m2", "fp6_e2m3")
"""The two MXFP6 grids the adaptive selector chooses between."""
```

A module-level constant declaring the universe of choices. If we ever extend
to 3+ grids (e.g., for the IF6 stretch goal), this is the line to change.

```python
@dataclass
class AdaptiveResult:
    recon: torch.Tensor
    choices: torch.Tensor
    per_block_mse: torch.Tensor
```

A structured return type. `@dataclass` auto-generates `__init__`, `__repr__`,
and `__eq__`. The three fields are:

- **`recon`**: the dequantized tensor, same shape as input. This is what
  you'd use as the "quantized weight" in a forward pass.
- **`choices`**: an `int8` tensor of shape `(..., n_blocks)` where
  `0 = e3m2` and `1 = e2m3`. One bit of information per block. We store
  it as int8 (a byte) for convenience — in real hardware this would be 1
  bit packed into the scale's metadata.
- **`per_block_mse`**: the MSE of the winning variant for each block.
  Useful for diagnostics ("which blocks were hardest to quantize?") and
  proves the per-block guarantee in tests.

```python
    @property
    def fraction_e3m2(self) -> float:
        return float((self.choices == 0).float().mean())

    @property
    def fraction_e2m3(self) -> float:
        return float((self.choices == 1).float().mean())
```

Convenience properties for inspection. `(self.choices == 0)` produces a bool
tensor, `.float().mean()` computes the fraction that is True. So
`result.fraction_e2m3` tells you "what fraction of blocks chose e2m3?"

```python
def _block_mse(x: torch.Tensor, x_recon: torch.Tensor, block_size: int) -> torch.Tensor:
    *lead, n = x.shape
    assert n % block_size == 0, (
        f"Last dim ({n}) must be divisible by block_size ({block_size}). "
        "Padding is not yet implemented."
    )
    n_blocks = n // block_size
    sqerr = (x - x_recon).pow(2)
    sqerr = sqerr.reshape(*lead, n_blocks, block_size)
    return sqerr.mean(dim=-1)
```

This is the function that converts a flat per-element error into per-block
errors. Walking through it:

- **`*lead, n = x.shape`**: unpack the shape. For a tensor of shape
  `(8, 256)`, this gives `lead = [8]` and `n = 256`. For a tensor of shape
  `(2, 3, 128)`, `lead = [2, 3]` and `n = 128`. The `*` is Python's "rest"
  unpacking.

- **`assert n % block_size == 0`**: blocks have to tile the last dimension
  exactly. If `n=33` and `block_size=32`, we'd have a fractional block. Real
  implementations handle this with padding — we punt on it for the prototype.

- **`n_blocks = n // block_size`**: how many full blocks fit in the last dim.
  E.g., 256 / 32 = 8 blocks.

- **`sqerr = (x - x_recon).pow(2)`**: per-element squared error, same shape
  as `x`.

- **`sqerr = sqerr.reshape(*lead, n_blocks, block_size)`**: reshape so the
  last dim is split into (n_blocks, block_size). For `(8, 256)`, this
  becomes `(8, 8, 32)`.

- **`return sqerr.mean(dim=-1)`**: average over the last axis (the 32
  elements in each block). Returns `(*lead, n_blocks)`, e.g., `(8, 8)` —
  one MSE per block.

```python
def fixed_quantize_dequantize(
    x: torch.Tensor,
    elem_format: Literal["fp6_e3m2", "fp6_e2m3"],
    block_size: int = 32,
) -> torch.Tensor:
    return quantize_dequantize_mx(x, elem_format, block_size)
```

A trivial passthrough to the backend. It exists as a public symbol because
"fixed e3m2" and "fixed e2m3" are the two **baselines** we compare adaptive
against. Calling them out as named API operations makes user code self-documenting:

```python
baseline = fixed_quantize_dequantize(x, "fp6_e2m3")    # clear intent
baseline = quantize_dequantize_mx(x, "fp6_e2m3")       # works but reaches into internals
```

```python
def adaptive_quantize_dequantize(
    x: torch.Tensor,
    block_size: int = 32,
) -> AdaptiveResult:
    # Round-trip with each variant
    recon_e3m2 = quantize_dequantize_mx(x, "fp6_e3m2", block_size)
    recon_e2m3 = quantize_dequantize_mx(x, "fp6_e2m3", block_size)
```

The core function. First, get reconstructions from each variant. These are
two full-tensor round-trips, each touching every element.

**Compute cost note:** this means quantizing with adaptive does ~2× the work
of fixed. For an offline calibration step (which is what we're doing), this
is fine. In a deployment scenario you'd have hardware that does this in
parallel or pre-computes the choice once.

```python
    mse_e3m2 = _block_mse(x, recon_e3m2, block_size)  # (..., n_blocks)
    mse_e2m3 = _block_mse(x, recon_e2m3, block_size)
```

Compute per-block MSE for each variant. After this, we have two tensors of
shape `(*lead, n_blocks)`, one MSE-per-block from each variant.

```python
    choose_e2m3 = (mse_e2m3 < mse_e3m2)  # bool, shape (..., n_blocks)
    choices = choose_e2m3.to(torch.int8)
    chosen_mse = torch.where(choose_e2m3, mse_e2m3, mse_e3m2)
```

The selection logic:
- `choose_e2m3` is `True` for blocks where e2m3 had lower MSE.
- `choices` converts that bool to `int8` (0 = e3m2, 1 = e2m3) for storage.
- `chosen_mse` records the MSE of whichever variant we kept. By construction
  this equals `min(mse_e3m2, mse_e2m3)` per block — that's our guarantee.

```python
    *lead, n = x.shape
    n_blocks = n // block_size
    recon_e3m2_b = recon_e3m2.reshape(*lead, n_blocks, block_size)
    recon_e2m3_b = recon_e2m3.reshape(*lead, n_blocks, block_size)
    mask = choose_e2m3.unsqueeze(-1)  # (..., n_blocks, 1)
    recon_b = torch.where(mask, recon_e2m3_b, recon_e3m2_b)
    recon = recon_b.reshape(*lead, n)
```

Now build the final reconstruction by picking each block from the winning
variant.

- Reshape both reconstructions so the last dim is `(n_blocks, block_size)`.
- Reshape the selection mask to `(..., n_blocks, 1)` so it broadcasts across
  all 32 elements of each block.
- `torch.where(mask, a, b)` picks `a` where mask is `True`, `b` where
  `False`. So whole-block-by-whole-block, we pick the right reconstruction.
- Reshape back to the original flat layout.

```python
    return AdaptiveResult(recon=recon, choices=choices, per_block_mse=chosen_mse)
```

Bundle and return. The caller gets the reconstructed tensor *and* the
diagnostic metadata (which variant each block chose, per-block MSE).

### 5.4 `adaptive_mxfp6/layers.py`

This file wraps `adaptive_quantize_dequantize` in something PyTorch users
recognize: a Module.

```python
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantize import (
    AdaptiveResult,
    adaptive_quantize_dequantize,
    fixed_quantize_dequantize,
)

Scheme = Literal["adaptive", "fixed_e3m2", "fixed_e2m3", "fp16"]
```

`Scheme` is a type alias listing the four supported schemes. `fp16` is a
no-op (pure passthrough) — useful for sanity-checking that the layer's plumbing
is correct independent of the quantization logic.

```python
class AdaptiveMXFP6Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, block_size=32, scheme="adaptive"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.scheme = scheme

        self.register_buffer("weight", torch.empty(out_features, in_features))
        if bias:
            self.bias: Optional[nn.Parameter] = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self.fraction_e3m2: Optional[float] = None
        self.fraction_e2m3: Optional[float] = None
        self.choices: Optional[torch.Tensor] = None
```

Standard PyTorch Module setup. Two interesting design choices:

1. **`register_buffer` instead of `nn.Parameter` for the weight**: a buffer
   moves with the model (e.g., `model.to('cuda')`), gets saved by
   `state_dict()`, but **doesn't get gradients**. Since we're doing
   *weight-only* PTQ — the weight is quantized once at load and never
   updated — there are no gradients to flow into it. Making it a buffer is
   semantically correct.

2. **`bias` stays as a parameter in FP32**: biases are tiny (one number per
   output), keeping them in FP32 costs nothing and matches what real PTQ
   recipes do. We never quantize the bias.

```python
    @classmethod
    def from_linear(cls, linear, block_size=32, scheme="adaptive"):
        new = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bias=(linear.bias is not None),
            block_size=block_size,
            scheme=scheme,
        )
        new.quantize_from(linear.weight.detach(), linear.bias.detach() if linear.bias is not None else None)
        return new
```

A factory method that converts an existing `nn.Linear` into an
`AdaptiveMXFP6Linear` with the same shapes, then copies and quantizes the
weight. This is what `quantize_linear_layers_` calls under the hood.

`.detach()` returns a tensor sharing the same data but without the autograd
graph — we don't want quantization to be a differentiable op for PTQ.

```python
    def quantize_from(self, weight, bias=None):
        if weight.shape != (self.out_features, self.in_features):
            raise ValueError(...)
        if self.in_features % self.block_size != 0:
            raise ValueError(...)

        w = weight.to(torch.float32)
        if self.scheme == "adaptive":
            res = adaptive_quantize_dequantize(w, self.block_size)
            self.weight.data.copy_(res.recon)
            self.fraction_e3m2 = res.fraction_e3m2
            self.fraction_e2m3 = res.fraction_e2m3
            self.choices = res.choices
        elif self.scheme == "fixed_e3m2":
            self.weight.data.copy_(fixed_quantize_dequantize(w, "fp6_e3m2", self.block_size))
            self.fraction_e3m2, self.fraction_e2m3 = 1.0, 0.0
        elif self.scheme == "fixed_e2m3":
            self.weight.data.copy_(fixed_quantize_dequantize(w, "fp6_e2m3", self.block_size))
            self.fraction_e3m2, self.fraction_e2m3 = 0.0, 1.0
        elif self.scheme == "fp16":
            self.weight.data.copy_(w)
```

The actual quantization happens here. The function:

1. Validates the shape and block-size compatibility.
2. Casts to FP32 (microxcaling works in FP32).
3. Dispatches on `self.scheme` to either:
   - Call `adaptive_quantize_dequantize` and store both the reconstruction
     and the diagnostic metadata, OR
   - Call `fixed_quantize_dequantize` for one of the baselines, OR
   - Just copy the weight as-is for the `fp16` passthrough.
4. Copies the bias if present.

The fractions are set to `1.0/0.0` for the fixed schemes so downstream code
(like the summary dict) doesn't have to special-case them.

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)
```

The forward pass is trivial: it's just `nn.Linear.forward`, except the
weight has already been quantized-then-dequantized at load. So this is a
faithful simulator of "what would inference look like if we used these
quantized weights" while keeping the math in FP32.

```python
def quantize_linear_layers_(
    model: nn.Module,
    scheme="adaptive",
    block_size=32,
    exclude_name_substrings=("lm_head", "embed"),
) -> dict:
    summary = {}
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(s in name for s in exclude_name_substrings):
            continue
        if module.in_features % block_size != 0:
            continue
        targets.append(name)

    for name in targets:
        parent, attr = _find_parent(model, name)
        old = getattr(parent, attr)
        new = AdaptiveMXFP6Linear.from_linear(old, block_size=block_size, scheme=scheme)
        setattr(parent, attr, new)
        summary[name] = {
            "fraction_e3m2": new.fraction_e3m2,
            "fraction_e2m3": new.fraction_e2m3,
        }
    return summary
```

The big one-call helper. Walks the model, finds every `nn.Linear`, swaps it
for our quantized version. The trailing underscore in the name (`_`) is a
PyTorch convention for in-place mutation (cf. `tensor.add_()`).

Three filters:

- **Type filter**: only `nn.Linear` (we don't touch Conv, LayerNorm, etc.)
- **Name filter**: skip anything containing `"lm_head"` or `"embed"` by
  default — these are precision-sensitive layers in LLMs that quantization
  papers usually leave in full precision.
- **Shape filter**: silently skip layers whose `in_features` isn't divisible
  by `block_size`. This handles real models gracefully where some
  projections have weird shapes.

The function returns a summary dict so you can print "what fraction of
blocks in `model.encoder.layer.3.attention.q_proj` chose e2m3?"

Two-phase loop (collect names first, then mutate) avoids the "modifying a
dict while iterating" bug.

```python
def _find_parent(model: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]
```

Helper to convert a dotted module name like `"encoder.layers.3.fc1"` into
`(the encoder.layers[3] module, "fc1")` so we can `setattr` to replace.

### 5.5 `adaptive_mxfp6/__init__.py`

```python
from .quantize import (
    adaptive_quantize_dequantize,
    fixed_quantize_dequantize,
    AdaptiveResult,
)
from .layers import AdaptiveMXFP6Linear, quantize_linear_layers_

__all__ = [
    "adaptive_quantize_dequantize",
    "fixed_quantize_dequantize",
    "AdaptiveResult",
    "AdaptiveMXFP6Linear",
    "quantize_linear_layers_",
]
```

This file makes the library importable. After this, users write:

```python
from adaptive_mxfp6 import quantize_linear_layers_  # works
```

instead of:

```python
from adaptive_mxfp6.layers import quantize_linear_layers_  # also works but uglier
```

`__all__` declares the public API — what `from adaptive_mxfp6 import *`
exports. Anything not listed there is "internal," even if it's importable.

### 5.6 `tests/test_adaptive.py`

Six tests verifying that the quantizer behaves as advertised. Read them as
**executable documentation** of the contract:

```python
def test_adaptive_never_worse_than_either_fixed():
    """Per-block guarantee: adaptive MSE <= min(fixed_e3m2, fixed_e2m3) per block."""
    torch.manual_seed(0)
    x = torch.randn(8, 256)

    e3m2 = fixed_quantize_dequantize(x, "fp6_e3m2", block_size=32)
    e2m3 = fixed_quantize_dequantize(x, "fp6_e2m3", block_size=32)
    adapt = adaptive_quantize_dequantize(x, block_size=32)

    mse_e3m2_b = _block_mse(x, e3m2, 32)
    mse_e2m3_b = _block_mse(x, e2m3, 32)
    mse_min_b = torch.minimum(mse_e3m2_b, mse_e2m3_b)

    assert torch.allclose(adapt.per_block_mse, mse_min_b, atol=1e-9)
    ...
```

The most important test in the suite. It proves the central invariant:
**adaptive's per-block MSE equals the per-block min of the two fixed
variants** — i.e., the selector picks the right variant every time. If this
test ever fails, the selector is broken.

```python
def test_tight_cluster_prefers_e2m3():
    x = torch.randn(4, 32 * 16) * 0.05
    adapt = adaptive_quantize_dequantize(x, block_size=32)
    frac_e2m3 = adapt.fraction_e2m3
    assert frac_e2m3 > 0.8
```

Sanity check: tight-clustered data should mostly select e2m3 (finer
precision). We saw 100% of blocks choose e2m3 in our run.

```python
def test_outlier_blocks_prefer_e3m2():
    x = torch.randn(1, 32 * 4)
    x[0, 32 + 5] = 100.0
    adapt = adaptive_quantize_dequantize(x, block_size=32)
    choices = adapt.choices.flatten()
    assert choices[1].item() == 0
```

The dual sanity check: inject an outlier into block 1 only, confirm block 1
selects e3m2 (`choice == 0`). We saw choices `[1, 0, 1, 1]` — exactly as
expected, only the outlier block went e3m2.

The remaining tests check shape preservation, dtype preservation, and that
exact-grid values round-trip without error.

### 5.7 `tests/test_layers.py`

Five tests for the Linear-layer plumbing:

- `test_layer_matches_fp16_when_no_quantization`: `scheme="fp16"` is a true
  passthrough (sanity check on the layer's plumbing).
- `test_adaptive_layer_introduces_small_error`: confirms the layer actually
  quantizes (error > 0) but doesn't catastrophically break things.
- `test_adaptive_beats_or_ties_both_fixed_on_layer_output`: at the
  *post-matmul* output level (not just per-element), adaptive shouldn't be
  worse than either fixed. This is a weaker guarantee than per-block —
  matmul can shuffle errors in unexpected ways — so the threshold is
  generous (≤ 2× the better fixed).
- `test_model_swap_replaces_linears`: builds a `nn.Sequential` with 3
  Linears, runs the swap, confirms all 3 became `AdaptiveMXFP6Linear` and
  the model still runs end-to-end.
- `test_model_swap_excludes_filtered`: a `ModuleDict` with an `lm_head`
  layer — confirm the default exclusion preserves it.

---

## 6. How to Use It Yourself

### Setup (already done)

```powershell
cd C:\Users\mathmara\adaptive-mxfp6
.\.venv\Scripts\Activate.ps1     # activate the venv (optional)
```

Or just call `.\.venv\Scripts\python.exe ...` directly without activating.

### Example 1: quantize a single tensor

```python
import sys
sys.path.insert(0, r"C:\Users\mathmara\adaptive-mxfp6")

import torch
from adaptive_mxfp6 import adaptive_quantize_dequantize

x = torch.randn(128, 256)  # last dim divisible by 32
result = adaptive_quantize_dequantize(x, block_size=32)

print(f"shape:                {result.recon.shape}")
print(f"per-block MSE shape:  {result.per_block_mse.shape}")
print(f"choices shape:        {result.choices.shape}")
print(f"fraction e3m2:        {result.fraction_e3m2:.1%}")
print(f"fraction e2m3:        {result.fraction_e2m3:.1%}")
print(f"global MSE:           {(x - result.recon).pow(2).mean().item():.4e}")
```

### Example 2: quantize an MLP

```python
import torch
import torch.nn as nn
from adaptive_mxfp6 import quantize_linear_layers_

mlp = nn.Sequential(
    nn.Linear(256, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, 10),
)

summary = quantize_linear_layers_(mlp, scheme="adaptive", block_size=32,
                                   exclude_name_substrings=())
for name, info in summary.items():
    print(f"{name}: e3m2={info['fraction_e3m2']:.1%}, e2m3={info['fraction_e2m3']:.1%}")

x = torch.randn(8, 256)
output = mlp(x)  # runs with quantized weights
print(output.shape)
```

### Example 3: A/B comparison

```python
import torch
import torch.nn as nn
import copy
from adaptive_mxfp6 import quantize_linear_layers_

base_model = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))
x = torch.randn(32, 256)
ref_output = base_model(x)

for scheme in ["fixed_e3m2", "fixed_e2m3", "adaptive"]:
    model = copy.deepcopy(base_model)
    quantize_linear_layers_(model, scheme=scheme, exclude_name_substrings=())
    output = model(x)
    mse = (output - ref_output).pow(2).mean().item()
    print(f"{scheme}: output MSE vs FP32 = {mse:.4e}")
```

### Example 4: running the test suites

```powershell
.\.venv\Scripts\python.exe tests\test_adaptive.py
.\.venv\Scripts\python.exe tests\test_layers.py
```

Or to re-run the Phase 0 sanity script:

```powershell
.\.venv\Scripts\python.exe verify_mxfp6.py
```

---

## 7. Design Decisions and Why

| Decision | What we chose | Why |
|---|---|---|
| Backend | microxcaling, used via `sys.path` (not pip) | microxcaling pins torch==2.2 which conflicts with our Python 3.14; sys.path bypasses pip's metadata enforcement |
| Quantization pipeline | Pure-Python via `custom_cuda=False` | No MSVC + CUDA toolkit on this Windows machine; pure-Python is slower but correct and avoids a build dependency |
| Block axis | Last dim of weight (= input features) | Matches OCP MX convention and the contraction axis of `F.linear` |
| Selection rule | Per-block MSE minimization | Deterministic, requires no calibration data, provably optimal under MSE |
| Selection encoding | 1 bit per block (stored as `int8` in the simulator) | Real hardware would pack this into spare scale bits; for a simulator, a byte is fine |
| Bias handling | Kept in FP32 (never quantized) | Standard in PTQ — biases are tiny, no benefit to quantizing them |
| Weight handling | Quantize-once at load, store FP32 reconstruction | Faithful simulator; no on-the-fly dequant overhead during inference |
| Block padding | Not implemented — raise an error | Most LLM Linears have multiples-of-32 input dims; we silently skip exceptions in `quantize_linear_layers_` |
| What to quantize | Weights only, activations stay FP32 | Standard PTQ baseline; activation quantization is a separate concern |
| `lm_head`, `embed` | Excluded by default | Standard PTQ practice — these are precision-sensitive |

---

## 8. What's Missing (Next Steps)

Listed in roughly the order they'd be tackled:

1. **Phase 3 — model demo (the headline experiment)**:
   - Load a real model (GPT-2 small, Qwen2-0.5B, etc.) from HuggingFace
   - Quantize with each scheme
   - Evaluate perplexity on WikiText-2
   - Produce the comparison table
   - This is the next thing we discussed before pausing.

2. **Padding for non-multiple-of-32 input dims**:
   - Many real Linears have shapes like `(2048, 768)` — 768 / 32 = 24, fine
   - But some have `(50257, 768)` (token embedding) — fine
   - Or `(768, 3072)` (FFN up-proj) — 3072 / 32 = 96, fine
   - Edge cases (e.g., heads of size 64 with hidden 320) need padding logic.

3. **HuggingFace integration helper**:
   - The `quantize_linear_layers_` function works on any nn.Module, including HF models
   - But a tiny convenience wrapper that calls `AutoModelForCausalLM.from_pretrained` and runs the swap would polish the demo.

4. **Per-layer scheme override**:
   - Some layers might benefit from staying in fp16 while others quantize aggressively
   - The current API takes one global `scheme` string; could extend to a dict like `{"attention.*": "adaptive", "mlp.*": "fixed_e3m2"}`.

5. **Calibration-based selection rule**:
   - Currently the selector minimizes MSE between original and reconstructed weights
   - A smarter selector would minimize MSE between the *layer output* on calibration data
   - This often gives better end-task accuracy.

6. **QAT (learned selector)** — the research stretch goal:
   - Make the per-block choice differentiable via Gumbel-softmax
   - Train on a small calibration dataset
   - Compare against the deterministic rule.

7. **IF6 (stretch goal #2)**:
   - Same machinery, but the selector picks between INT6 and FP6 instead of e3m2 vs e2m3
   - Direct 6-bit analog of IF4 from the first reference paper.

8. **Library packaging (pyproject.toml, README, version)**:
   - Easy to add once the API stabilizes
   - Would allow `pip install adaptive-mxfp6` and easier distribution.

---

## Appendix: Glossary

- **OCP MX**: Open Compute Project Microscaling — the standard that defines
  block-scaled low-bit formats (MXFP4, MXFP6, MXFP8, MXINT8).
- **Block scaling**: a quantization style where a group of N weights shares one
  scale factor. OCP MX uses block size 32 and an 8-bit shared exponent (E8M0).
- **E8M0**: 8 exponent bits, 0 mantissa bits — represents pure powers of 2.
  Used as the MX block scale.
- **e3m2 / e2m3**: FP6 element formats. Numbers are the count of exponent and
  mantissa bits respectively (sign bit is always 1).
- **PTQ**: Post-Training Quantization — quantize an already-trained model
  without retraining. What this prototype does.
- **QAT**: Quantization-Aware Training — train (or fine-tune) the model with
  quantization in the loop. Stretch goal for this project.
- **MSE**: Mean Squared Error — our reconstruction-quality metric.
- **microxcaling**: Microsoft's open-source PyTorch simulator for OCP MX
  formats. We use it as the backend for the actual FP6 quantization math.
