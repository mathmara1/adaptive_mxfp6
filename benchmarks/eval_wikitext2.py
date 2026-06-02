"""
Evaluate adaptive 6-bit quantization on WikiText-2 perplexity.

Each "run" specifies one quantization configuration to evaluate. A run can be:
  * A preset name: "fp16" (no quantization), "adaptive" (the default 2-grid
    PO2), "fixed_e3m2", "fixed_e2m3".
  * A comma-separated list of strategy names (each from AVAILABLE_STRATEGIES):
    e.g., "fp6_e3m2,fp6_e2m3" or "fp6_e3m2,fp6_e2m3,int6,nf6".
      - Single strategy  → fixed quantization with that grid
      - 2+ strategies    → adaptive selection among them (length must be a power of 2)

Usage examples:
    # Default: fp16 baseline + fixed e3m2 + fixed e2m3 + 2-grid adaptive
    .venv\\Scripts\\python.exe benchmarks\\eval_wikitext2.py

    # Smoke test
    .venv\\Scripts\\python.exe benchmarks\\eval_wikitext2.py --n_sequences 5

    # Explicit custom strategy sets (e.g., comparing 2-grid vs 4-grid)
    .venv\\Scripts\\python.exe benchmarks\\eval_wikitext2.py \\
        --runs fp16 "fp6_e3m2,fp6_e2m3" "fp6_e3m2,fp6_e2m3,int6,nf6"

    # Full WikiText-2 (slow on CPU)
    .venv\\Scripts\\python.exe benchmarks\\eval_wikitext2.py --full
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from adaptive_mxfp6 import (  # noqa: E402
    AVAILABLE_STRATEGIES,
    SCHEME_PRESETS,
    quantize_linear_layers_,
    validate_strategies,
)


# A "run spec" parses to (display label, strategies-or-None).
# strategies=None means fp16 / passthrough; otherwise a tuple of strategy names.
RunSpec = Tuple[str, Optional[Tuple[str, ...]]]


def parse_run_spec(spec: str) -> RunSpec:
    """
    Parse one --runs argument into a (label, strategies) pair.

    Accepted forms:
      "fp16"                          -> ("fp16", None)
      "adaptive"                      -> ("adaptive", ("fp6_e3m2", "fp6_e2m3"))
      "fixed_e3m2"                    -> ("fixed_e3m2", ("fp6_e3m2",))
      "fp6_e3m2"                      -> ("fp6_e3m2", ("fp6_e3m2",))
      "fp6_e3m2,fp6_e2m3"             -> ("adaptive(fp6_e3m2,fp6_e2m3)", (...))
      "fp6_e3m2,fp6_e2m3,int6,nf6"    -> ("adaptive(fp6_e3m2,fp6_e2m3,int6,nf6)", (...))
    """
    spec = spec.strip()
    if spec in SCHEME_PRESETS:
        return spec, SCHEME_PRESETS[spec]

    parts = tuple(p.strip() for p in spec.split(",") if p.strip())
    if not parts:
        raise ValueError(f"empty run spec: {spec!r}")

    # validate_strategies raises ValueError / NotImplementedError with a clear message
    validate_strategies(parts)

    if len(parts) == 1:
        label = parts[0]
    else:
        label = f"adaptive({','.join(parts)})"
    return label, parts


def load_model(name: str, device: str = "cpu"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.to(device)
    model.eval()
    return model, tokenizer


def load_wikitext2_text() -> str:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    return "\n\n".join(ds["text"])


@torch.no_grad()
def eval_perplexity(
    model, tokenizer, n_sequences: int, seq_len: int, stride: int, device: str = "cpu"
) -> float:
    """Sliding-window perplexity over WikiText-2 test split (HF tutorial style)."""
    text = load_wikitext2_text()
    enc = tokenizer(text, return_tensors="pt")
    input_ids_full = enc.input_ids
    total_seq_len = int(input_ids_full.size(1))

    starts = list(range(0, total_seq_len, stride))
    if n_sequences > 0:
        starts = starts[:n_sequences]
    print(f"  Eval: {len(starts)} windows, seq_len={seq_len}, stride={stride} "
          f"(of {total_seq_len} total tokens)")

    nll_sum = 0.0
    n_loss_tokens_total = 0
    prev_end_loc = 0
    t0 = time.time()
    for i, begin_loc in enumerate(starts):
        end_loc = min(begin_loc + seq_len, total_seq_len)
        trg_len = end_loc - prev_end_loc
        if trg_len <= 0:
            break

        input_ids = input_ids_full[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        out = model(input_ids, labels=target_ids)
        num_valid_tokens = int((target_ids != -100).sum().item())
        batch_size = int(target_ids.size(0))
        num_loss_tokens = num_valid_tokens - batch_size

        nll_sum += float(out.loss.item()) * num_loss_tokens
        n_loss_tokens_total += num_loss_tokens
        prev_end_loc = end_loc

        if (i + 1) % 5 == 0 or i == len(starts) - 1:
            elapsed = time.time() - t0
            running = math.exp(nll_sum / n_loss_tokens_total)
            print(f"    window {i+1:>3}/{len(starts)}  running_ppl={running:.3f}  elapsed={elapsed:.0f}s")

        if end_loc == total_seq_len:
            break

    return math.exp(nll_sum / n_loss_tokens_total)


def summarize_layers(
    summary: dict,
    label: str,
    strategies: Optional[Tuple[str, ...]],
) -> dict:
    """
    For multi-strategy runs, print mean fractions per strategy and the most/least
    outlier-heavy layers (sorted by the FIRST strategy's selection fraction).
    """
    if not summary:
        print(f"  [{label}] no layers quantized")
        return {}

    n = len(summary)
    if strategies is None or len(strategies) <= 1:
        print(f"  [{label}] {n} Linears quantized (single strategy or passthrough)")
        return {"n_layers": n}

    # Mean fraction selected per strategy, across all quantized layers
    mean_fractions = {}
    for s in strategies:
        vals = [info["fractions"].get(s, 0.0) for info in summary.values()]
        mean_fractions[s] = sum(vals) / n

    print(f"  [{label}] {n} Linears quantized; mean fractions:")
    for s in strategies:
        print(f"      {s:<24}  {mean_fractions[s]*100:5.1f}%")

    # Top/bottom layers ranked by the first strategy's selection fraction
    first_strategy = strategies[0]
    sorted_layers = sorted(
        summary.items(),
        key=lambda kv: kv[1]["fractions"].get(first_strategy, 0.0),
        reverse=True,
    )
    k = min(5, n)
    print(f"    Most {first_strategy}-heavy layers (top {k}):")
    for name, info in sorted_layers[:k]:
        frac = info["fractions"].get(first_strategy, 0.0)
        print(f"      {frac*100:5.1f}%  {name}")
    print(f"    Least {first_strategy}-heavy layers (bottom {k}):")
    for name, info in sorted_layers[-k:]:
        frac = info["fractions"].get(first_strategy, 0.0)
        print(f"      {frac*100:5.1f}%  {name}")

    return {"n_layers": n, "mean_fractions": mean_fractions}


def inspect_layer_shapes(model, block_size: int) -> Tuple[list, list]:
    """Return (quantizable_linears, skipped_linears) for reporting."""
    quantizable, skipped = [], []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        is_excluded_by_name = any(s in name for s in ("lm_head", "embed"))
        is_divisible = module.in_features % block_size == 0
        if is_excluded_by_name or not is_divisible:
            skipped.append((name, module.in_features, "excluded" if is_excluded_by_name else "non-divisible"))
        else:
            quantizable.append((name, module.in_features, module.out_features))
    return quantizable, skipped


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n")[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Available strategies: "
            + ", ".join(
                f"{name}{'*' if info.implemented else ' (planned)'}"
                for name, info in AVAILABLE_STRATEGIES.items()
            )
            + "\n* = implemented"
        ),
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--n_sequences", type=int, default=50,
                        help="Number of sliding windows to eval on (default 50; use --full to override).")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=None,
                        help="Tokens between window starts. Default = seq_len // 2.")
    parser.add_argument("--block_size", type=int, default=32)
    parser.add_argument("--runs", nargs="+", default=None,
                        help="One or more run specs: a preset name (fp16, adaptive, fixed_e3m2, "
                             "fixed_e2m3) or a comma-separated strategy list "
                             "(e.g. 'fp6_e3m2,fp6_e2m3').")
    parser.add_argument("--schemes", nargs="+", default=None,
                        help="Deprecated alias for --runs (kept for backward compatibility).")
    parser.add_argument("--full", action="store_true",
                        help="Evaluate on all available windows (slow on CPU).")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--results_dir", default="benchmarks/results")
    args = parser.parse_args()

    # Resolve which run specs to use
    if args.runs is not None and args.schemes is not None:
        print("Note: both --runs and --schemes given; --runs wins.")
    raw_specs = args.runs or args.schemes or ["fp16", "fixed_e3m2", "fixed_e2m3", "adaptive"]

    # Parse all run specs upfront so errors are surfaced before we load the model
    runs: List[RunSpec] = []
    for spec in raw_specs:
        try:
            runs.append(parse_run_spec(spec))
        except (ValueError, NotImplementedError) as e:
            print(f"ERROR: invalid run spec {spec!r}: {e}")
            sys.exit(2)
    # Deduplicate labels (keep first occurrence) so the table doesn't have collisions
    seen, deduped = set(), []
    for label, strategies in runs:
        if label in seen:
            print(f"Note: duplicate run {label!r} skipped.")
            continue
        seen.add(label)
        deduped.append((label, strategies))
    runs = deduped

    n_seq = 0 if args.full else args.n_sequences
    stride = args.stride if args.stride is not None else args.seq_len // 2

    # Inspection pass to surface what will/won't be quantized
    print(f"Loading {args.model} (inspection pass) ...")
    model0, _ = load_model(args.model, device=args.device)
    quantizable, skipped = inspect_layer_shapes(model0, args.block_size)
    print(f"  Quantizable Linears: {len(quantizable)}")
    print(f"  Skipped Linears:     {len(skipped)}")
    for name, in_f, reason in skipped:
        print(f"    SKIP [{reason:13s}]  in_features={in_f:>6d}  {name}")
    del model0
    gc.collect()

    # Pretty-print the run plan
    print(f"\nRun plan: {len(runs)} run(s)")
    for label, strategies in runs:
        if strategies is None:
            print(f"  - {label:<30}  (passthrough, no quantization)")
        elif len(strategies) == 1:
            print(f"  - {label:<30}  fixed: {strategies[0]}")
        else:
            print(f"  - {label:<30}  adaptive over: {list(strategies)}")

    results: dict = {}
    for label, strategies in runs:
        print(f"\n{'=' * 70}")
        print(f"Run: {label}")
        print(f"{'=' * 70}")

        t_load = time.time()
        print(f"Loading fresh {args.model} ...")
        model, tokenizer = load_model(args.model, device=args.device)
        load_sec = time.time() - t_load

        layer_stats: dict = {}
        if strategies is None:
            quant_sec = 0.0
        else:
            print(f"Quantizing all Linears with strategies={list(strategies)} ...")
            t_q = time.time()
            summary = quantize_linear_layers_(
                model, strategies=strategies, block_size=args.block_size
            )
            quant_sec = time.time() - t_q
            print(f"  Quantization took {quant_sec:.1f}s")
            layer_stats = summarize_layers(summary, label, strategies)

        t_eval = time.time()
        ppl = eval_perplexity(
            model, tokenizer,
            n_sequences=n_seq, seq_len=args.seq_len, stride=stride, device=args.device,
        )
        eval_sec = time.time() - t_eval

        results[label] = {
            "strategies": list(strategies) if strategies else None,
            "perplexity": ppl,
            "load_sec": load_sec,
            "quant_sec": quant_sec,
            "eval_sec": eval_sec,
            "layer_stats": layer_stats,
        }
        print(f"\n[{label}] PPL = {ppl:.4f}  "
              f"(load {load_sec:.0f}s, quant {quant_sec:.0f}s, eval {eval_sec:.0f}s)")

        del model
        gc.collect()

    # Headline table — auto-size the run-label column
    label_width = max(20, max(len(l) for l, _ in runs))
    print(f"\n{'=' * (label_width + 35)}")
    print(f"HEADLINE: {args.model} - WikiText-2 perplexity")
    print(f"  n_sequences={'all' if n_seq <= 0 else n_seq}, seq_len={args.seq_len}, "
          f"stride={stride}, block_size={args.block_size}")
    print(f"{'=' * (label_width + 35)}")
    fp16_ppl = results.get("fp16", {}).get("perplexity")
    print(f"  {'Run':<{label_width}}  {'PPL':>10}  {'vs fp16':>14}")
    print(f"  {'-' * (label_width + 28)}")
    for label, _ in runs:
        ppl = results[label]["perplexity"]
        if label == "fp16":
            delta = "  (baseline)"
        elif fp16_ppl is not None:
            delta = f"  +{ppl - fp16_ppl:.4f}"
        else:
            delta = "  -"
        print(f"  {label:<{label_width}}  {ppl:>10.4f}{delta}")

    # Legacy invariant check (only fires if all three legacy labels are present)
    needed = ("fixed_e3m2", "fixed_e2m3", "adaptive")
    if all(s in results for s in needed):
        best_fixed = min(results["fixed_e3m2"]["perplexity"], results["fixed_e2m3"]["perplexity"])
        adapt = results["adaptive"]["perplexity"]
        tol = max(0.005, best_fixed * 0.001)
        if adapt <= best_fixed + tol:
            print(f"\n  INVARIANT OK: adaptive PPL ({adapt:.4f}) <= "
                  f"min(fixed_e3m2, fixed_e2m3) PPL ({best_fixed:.4f}) (tol {tol:.4f})")
        else:
            print(f"\n  INVARIANT WARNING: adaptive PPL ({adapt:.4f}) > min(fixed) PPL ({best_fixed:.4f})")
            print(f"  Per-block MSE selection is provably optimal at the *block* level, but")
            print(f"  per-block MSE composes non-linearly through the network.")

    # Save raw results
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.model.replace('/', '_')}_wikitext2.json"
    with open(out_file, "w") as f:
        json.dump({
            "model": args.model,
            "n_sequences": n_seq if n_seq > 0 else "all",
            "seq_len": args.seq_len,
            "stride": stride,
            "block_size": args.block_size,
            "runs": [{"label": l, "strategies": list(s) if s else None} for l, s in runs],
            "results": results,
            "skipped_layers": [{"name": n, "in_features": f, "reason": r} for n, f, r in skipped],
        }, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
