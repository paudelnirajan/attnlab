"""
Stage 0a benchmark driver. For every (model, seq_len) cell in the matrix:

  - cold-load the model, measure load time + RSS delta
  - forward pass with names_filter restricted to attention patterns only
    (the "don't cache everything" optimization) vs. a full run_with_cache,
    to quantify the saving the plan claims
  - encode the resulting attention with patterns.py, at every stage of the
    pipeline, to get REAL byte counts instead of hand-computed estimates

Writes bench_results/results.json (raw) and bench_results/RESULTS.md (the
table that replaces the hand-estimated numbers in FEASIBILITY.md and
docs/01-wire-format.md with measured ones).

Run native (fast, MPS, exploratory):  uv run python -m attnlab.bench
Run in the constrained container (the numbers that matter): see Makefile.
"""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from attnlab.instrument import measure
from attnlab.registry import ModelSpec, load_registry
from attnlab.patterns import size_report

SEQ_LENGTHS = [64, 128, 256, 512]
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "bench_results"


def _make_prompt(seq_len: int, tokenizer) -> str:
    """A deterministic, reasonably natural prompt padded/truncated to
    exactly seq_len tokens (after BOS). Repeats a short passage rather
    than random tokens, so tokenizer behavior stays realistic across
    models with very different vocabularies."""
    base = (
        "The quick brown fox jumps over the lazy dog. "
        "Mechanistic interpretability studies how neural networks compute. "
        "Induction heads let transformers copy patterns they have seen before. "
    )
    text = base
    while True:
        ids = tokenizer(text)["input_ids"]
        if len(ids) >= seq_len:
            break
        text += base
    # Truncate to exactly seq_len tokens by re-decoding a token slice.
    ids = tokenizer(text)["input_ids"][:seq_len]
    return tokenizer.decode(ids)


def _cache_nbytes(cache) -> int:
    """Exact, deterministic size of everything TransformerLens cached —
    the right way to quantify names_filter's saving. A single forward
    pass's RSS delta is NOT reliable for this (see comment at the call
    site): tensor byte counts have no allocator-noise ambiguity."""
    return sum(t.numel() * t.element_size() for t in cache.values())


def _load_model(spec: ModelSpec, device: str, dtype: torch.dtype):
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(spec.tl_name, device=device, dtype=dtype)


def bench_model(spec: ModelSpec, *, device: str, dtype: torch.dtype, is_first: bool) -> dict[str, Any]:
    print(f"\n=== {spec.id} ({spec.tl_name}) ===", file=sys.stderr)
    cell_results: list[dict[str, Any]] = []

    with measure("cold_load", model=spec.id, first_in_run=is_first) as m_load:
        model = _load_model(spec, device=device, dtype=dtype)
    print(f"  load: {m_load.duration_ms:.0f}ms, rss_delta={m_load.rss_delta_mb:.0f}MB", file=sys.stderr)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    max_seq = min(spec.max_seq, max(SEQ_LENGTHS))

    # Analytic cross-check, immune to the allocator-reuse problem that
    # empirical RSS deltas suffer from: sequential in-process model
    # loads can show a misleadingly LOW delta for model N when model
    # N-1's freed pages get reused rather than returned to the OS
    # (gc.collect() does not guarantee the OS sees memory back). This
    # number has no such ambiguity — it is exactly params * bytes/param.
    n_params = sum(p.numel() for p in model.parameters())
    analytic_fp32_mb = n_params * 4 / 1024**2

    for seq_len in SEQ_LENGTHS:
        if seq_len > max_seq:
            continue
        prompt = _make_prompt(seq_len, model.tokenizer)
        tokens = model.to_tokens(prompt)
        actual_seq = tokens.shape[-1]

        # (a) forward pass caching ONLY attention patterns — the
        # optimization the plan claims saves real RAM. names_filter lets
        # TransformerLens skip caching MLP activations, residual stream,
        # etc. entirely rather than caching-then-discarding them.
        pattern_filter = lambda name: name.endswith("hook_pattern")  # noqa: E731
        with measure("forward_pattern_only", model=spec.id, seq=actual_seq) as m_pat:
            with torch.no_grad():
                _, cache_pat = model.run_with_cache(tokens, names_filter=pattern_filter)
        patterns = (
            torch.stack([cache_pat[f"blocks.{i}.attn.hook_pattern"][0] for i in range(n_layers)])
            .to(torch.float32)
            .cpu()
            .numpy()
        )  # (n_layers, n_heads, seq, seq)
        pattern_cache_bytes = _cache_nbytes(cache_pat)
        del cache_pat
        gc.collect()

        # (b) full run_with_cache (everything), same input, to quantify
        # the saving of (a) vs "just cache whatever TransformerLens
        # normally caches." The RIGHT way to measure this saving is the
        # actual byte size of what got cached (deterministic, exact) —
        # NOT the process's RSS delta across a single forward pass, which
        # is dominated by allocator noise at this scale and produced
        # nonsensical numbers (-57%, spurious 100%) in an earlier version
        # of this benchmark.
        with measure("forward_full_cache", model=spec.id, seq=actual_seq) as m_full:
            with torch.no_grad():
                _, cache_full = model.run_with_cache(tokens)
        full_cache_bytes = _cache_nbytes(cache_full)
        del cache_full
        gc.collect()

        report = size_report(patterns)

        cell = {
            "model": spec.id,
            "seq_len": actual_seq,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "forward_pattern_only": m_pat.to_record(),
            "forward_full_cache": m_full.to_record(),
            "pattern_cache_bytes": pattern_cache_bytes,
            "full_cache_bytes": full_cache_bytes,
            "sizes": dataclasses_asdict(report),
        }
        cell_results.append(cell)
        print(
            f"  seq={actual_seq:4d}  pattern_only={m_pat.duration_ms:6.1f}ms  "
            f"full_cache={m_full.duration_ms:6.1f}ms  "
            f"per_layer_gz={report.per_layer_uint8_triangle_gzip_bytes/1024:.1f}KB",
            file=sys.stderr,
        )

    del model
    gc.collect()

    return {
        "model_id": spec.id,
        "tl_name": spec.tl_name,
        "n_layers": n_layers,
        "n_heads": n_heads,
        "n_params": n_params,
        "analytic_fp32_mb": analytic_fp32_mb,
        "load": m_load.to_record(),
        "cells": cell_results,
    }


def dataclasses_asdict(obj) -> dict[str, Any]:
    import dataclasses

    return dataclasses.asdict(obj)


def _framework_baseline_mb(device: str) -> float:
    """RSS cost of just initializing torch + the device backend, before any
    HookedTransformer is loaded. Without subtracting this, the FIRST
    model benchmarked absorbs a one-time cost (MPS/CUDA context init,
    threadpool setup) that has nothing to do with that model's actual
    size — and every 'est_ram_mb' derived from it would be wrong in the
    same direction: too high, and by an amount that looks like signal."""
    from attnlab.instrument import measure

    with measure("framework_init") as m:
        t = torch.zeros(8, 8, device=device)
        _ = (t @ t).sum().item()  # force lazy backend init, not just allocation
    return m.rss_after_mb


def _run_one_model_in_process(spec: ModelSpec) -> dict[str, Any]:
    """Runs in a FRESH process (see run_isolated_matrix): no prior model's
    allocator history to contaminate the RSS reading, so `net_rss_delta_mb`
    here is trustworthy on its own terms, not just relative to whatever
    happened to load before it."""
    from attnlab.settings import SETTINGS

    baseline_mb = _framework_baseline_mb(SETTINGS.device)
    result = bench_model(spec, device=SETTINGS.device, dtype=SETTINGS.dtype, is_first=True)
    result["load"]["net_rss_delta_mb"] = result["load"]["rss_after_mb"] - baseline_mb
    result["framework_baseline_mb"] = baseline_mb
    return result


def run_isolated_matrix(specs: list[ModelSpec]) -> dict[str, Any]:
    """One subprocess per model. This is the trustworthy path for RSS
    numbers (see docs/03-decisions.md D8) — sequential in-process loading
    lets one model's freed-but-not-returned memory contaminate the next
    model's measurement, in BOTH directions (looks too small if it reuses
    freed pages, too large if RSS's high-water mark just never drops).
    A fresh process per model has no such history."""
    from attnlab.settings import SETTINGS

    print(
        f"device={SETTINGS.device} dtype={SETTINGS.dtype} threads={SETTINGS.threads} "
        f"ram_budget_gb={SETTINGS.ram_budget_gb}  (isolated: one subprocess per model)",
        file=sys.stderr,
    )

    model_results: list[dict[str, Any]] = []
    baselines: list[float] = []
    for spec in specs:
        print(f"\n--- subprocess: {spec.id} ---", file=sys.stderr)
        # Write the result to a dedicated temp file rather than parsing
        # stdout as JSON: TransformerLens itself does a bare `print()` on
        # model load (not logging, not stderr — see
        # loading_from_pretrained.py's "Loaded pretrained model ..." line),
        # which lands on stdout ahead of our JSON and breaks a naive parse.
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "result.json"
            proc = subprocess.run(
                [sys.executable, "-m", "attnlab.bench", "--single-json", spec.id, str(out_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            print(proc.stdout, file=sys.stderr, end="")
            print(proc.stderr, file=sys.stderr, end="")
            if proc.returncode != 0:
                # Do NOT abort the whole matrix for one model's failure
                # (e.g. OOM-kill, signal -9) — a partial benchmark run
                # that reports what it could is far more useful than
                # losing every result that came before it. This is
                # exactly what happened with bloom-560m under a
                # misconfigured Docker memory limit; see docs/03-decisions.md.
                print(
                    f"!!! {spec.id} failed (exit {proc.returncode}) — recording as failed, continuing",
                    file=sys.stderr,
                )
                model_results.append(
                    {
                        "model_id": spec.id,
                        "tl_name": spec.tl_name,
                        "failed": True,
                        "exit_code": proc.returncode,
                        "cells": [],
                    }
                )
                continue
            result = json.loads(out_path.read_text())
        baselines.append(result.pop("framework_baseline_mb"))
        model_results.append(result)

    return {
        "settings": {
            "device": SETTINGS.device,
            "dtype": str(SETTINGS.dtype),
            "threads": SETTINGS.threads,
            "ram_budget_gb": SETTINGS.ram_budget_gb,
            "framework_baseline_mb": sum(baselines) / len(baselines) if baselines else 0.0,
            "isolated": True,
        },
        "models": model_results,
    }


def run(model_ids: list[str] | None = None) -> dict[str, Any]:
    specs = load_registry()
    if model_ids:
        specs = [s for s in specs if s.id in model_ids]
    else:
        specs = [s for s in specs if s.tier != "disabled"]

    if len(specs) <= 1:
        # Nothing precedes it in the process either way — subprocess
        # isolation would add overhead for no accuracy benefit.
        from attnlab.settings import SETTINGS

        print(
            f"device={SETTINGS.device} dtype={SETTINGS.dtype} threads={SETTINGS.threads} "
            f"ram_budget_gb={SETTINGS.ram_budget_gb}",
            file=sys.stderr,
        )
        results = [_run_one_model_in_process(s) for s in specs]
        baseline = results[0].pop("framework_baseline_mb") if results else 0.0
        return {
            "settings": {
                "device": SETTINGS.device,
                "dtype": str(SETTINGS.dtype),
                "threads": SETTINGS.threads,
                "ram_budget_gb": SETTINGS.ram_budget_gb,
                "framework_baseline_mb": baseline,
                "isolated": False,
            },
            "models": results,
        }

    return run_isolated_matrix(specs)


def write_markdown(results: dict[str, Any], path: Path) -> None:
    lines = ["# Benchmark results", ""]
    s = results["settings"]
    isolated = s.get("isolated", False)
    lines.append(
        f"device=`{s['device']}` dtype=`{s['dtype']}` threads=`{s['threads']}` "
        f"ram_budget_gb=`{s['ram_budget_gb']}` isolation=`{'one subprocess per model' if isolated else 'single process'}`"
    )
    lines.append(
        f"\nframework baseline RSS (torch + `{s['device']}` backend init, paid once): "
        f"**{s['framework_baseline_mb']:.0f} MB** — subtracted from each model's load "
        f"to get `net_rss_delta_mb`."
    )
    if isolated:
        lines.append(
            "\n**Every row below pays one-time Python import overhead** for `transformers` / "
            "`tokenizers` / `huggingface_hub` (their C extensions, vocab/config parsing machinery), "
            "since each model ran in its own fresh subprocess (see D8 in docs/03-decisions.md) — "
            "that overhead has nothing to do with any individual model's actual size. It is a "
            "closer match to Stage 4's real cold-start number (a fresh worker process loading its "
            "first model) than to the marginal cost of adding a 2nd/3rd model to an already-warm "
            "server, which is what Stage 0b's LRU eviction budget actually needs — use "
            "`analytic fp32 MB` for that, not this column."
        )
    else:
        lines.append(
            "\n**Caveat on the first row below:** the first model loaded in this run also pays "
            "one-time Python import overhead for `transformers` / `tokenizers` / `huggingface_hub`, "
            "which has nothing to do with that model's actual size. This is marked `first_in_run` "
            "in `results.json`."
        )
    lines.append(
        "\n**`net rss delta` vs. `analytic fp32`:** these can diverge significantly for models "
        "loaded after a larger one was freed in the same process — macOS/Python allocators reuse "
        "freed pages rather than returning them to the OS, so the empirical delta can "
        "*understate* a model's true cost. `analytic fp32` (`params * 4 bytes`) has no such "
        "ambiguity. **Stage 0b's LRU budget should account against the analytic number, not "
        "the empirical RSS delta** — see docs/03-decisions.md."
    )
    lines.append("")
    lines.append(
        "| model | load ms | net rss delta MB | analytic fp32 MB | seq | pattern-only ms | "
        "full-cache ms | cache saving | per-layer f32 KB | per-layer u8+tri KB | per-layer u8+tri+gz KB |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model in results["models"]:
        if model.get("failed"):
            reason = " — likely OOM-killed (SIGKILL)" if model["exit_code"] == -9 else ""
            lines.append(
                f"| {model['model_id']} | **FAILED, exit {model['exit_code']}{reason}** | "
                "n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |"
            )
            continue
        load = model["load"]
        for cell in model["cells"]:
            pat = cell["forward_pattern_only"]
            full = cell["forward_full_cache"]
            sizes = cell["sizes"]
            # Exact byte counts of what got cached, not RSS deltas (see
            # _cache_nbytes docstring for why the latter is unreliable
            # at single-forward-pass scale).
            pat_bytes = cell["pattern_cache_bytes"]
            full_bytes = cell["full_cache_bytes"]
            saving = f"{(1 - pat_bytes / full_bytes) * 100:.0f}%" if full_bytes > 0 else "n/a"
            f32_per_layer_kb = (
                sizes["float32_bytes"] / model["n_layers"] / 1024
            )
            name = model["model_id"] + (" \\*" if load.get("first_in_run") else "")
            lines.append(
                f"| {name} | {load['duration_ms']:.0f} | "
                f"{load['net_rss_delta_mb']:.0f} | {model['analytic_fp32_mb']:.0f} | {cell['seq_len']} | "
                f"{pat['duration_ms']:.1f} | {full['duration_ms']:.1f} | "
                f"{saving} | {f32_per_layer_kb:.1f} | "
                f"{sizes['per_layer_uint8_triangle_bytes']/1024:.1f} | "
                f"{sizes['per_layer_uint8_triangle_gzip_bytes']/1024:.1f} |"
            )
    path.write_text("\n".join(lines) + "\n")


def _main_single_json(model_id: str, out_path: str) -> None:
    """Internal entry point used by run_isolated_matrix's subprocesses.
    Writes the JSON result to `out_path` rather than stdout — see the
    comment in run_isolated_matrix for why stdout is not safe to parse
    (TransformerLens prints its own unstructured text there on load)."""
    spec = next(s for s in load_registry() if s.id == model_id)
    result = _run_one_model_in_process(spec)
    Path(out_path).write_text(json.dumps(result))


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--single-json":
        _main_single_json(sys.argv[2], sys.argv[3])
    else:
        RESULTS_DIR.mkdir(exist_ok=True)
        model_ids = sys.argv[1:] or None
        results = run(model_ids)
        (RESULTS_DIR / "results.json").write_text(json.dumps(results, indent=2))
        write_markdown(results, RESULTS_DIR / "RESULTS.md")
        print(f"\nwrote {RESULTS_DIR / 'results.json'}", file=sys.stderr)
        print(f"wrote {RESULTS_DIR / 'RESULTS.md'}", file=sys.stderr)
