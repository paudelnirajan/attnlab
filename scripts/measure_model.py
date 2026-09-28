"""
Measure a model's real memory cost before adding it to models.yaml.

    MallocLargeCache=0 uv run python scripts/measure_model.py Qwen/Qwen3-0.6B-Base

Loads the model the way the server does (CPU, float32, LensReadyTransformer),
in this fresh process, and prints the two numbers the zoo budgets against:

  est_ram_mb     what stays resident once loaded
  load_extra_mb  what is held on top of that only while loading

plus the transient cost of a 512-token attention run and a 256-token logit-lens
run, and their timings. The suggested values add a 10% margin. The
MallocLargeCache=0 prefix is what the server runs with (docs/04-self-hosting.md
§ 2); without it every number here roughly doubles on macOS.

Numbers are for the machine you run this on. Memory transfers between Apple
Silicon Macs; timings don't (an M1 Pro is slower than an M4 Pro).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

os.environ.setdefault("MI_DEVICE", "cpu")

import torch  # noqa: E402

from attnlab import hub, memory  # noqa: E402
from attnlab.inference import run_forward  # noqa: E402
from attnlab.lens import LensReadyTransformer, run_lens  # noqa: E402
from attnlab.settings import SETTINGS  # noqa: E402


def _up(x: float) -> int:
    """+10%, rounded up to the next 50 MB."""
    return int(math.ceil(x * 1.1 / 50) * 50)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tl_name", help="TransformerLens model name, e.g. gpt2-small or Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--seq", type=int, default=512)
    args = ap.parse_args()

    hub.install_cached_listing()
    if not memory.large_cache_disabled():
        print("warning: MallocLargeCache=0 is not set; numbers will be inflated (see --help)\n")

    base = memory.footprint_mb() or 0.0
    t = time.time()
    model = LensReadyTransformer.from_pretrained(args.tl_name, device=SETTINGS.device, dtype=SETTINGS.dtype)
    load_s = time.time() - t
    memory.release()
    after_load = (memory.footprint_mb() or 0.0) - base
    peak_load = (memory.peak_footprint_mb() or 0.0) - base

    seq = min(args.seq, model.cfg.n_ctx)
    tokens = torch.randint(1000, min(20000, model.cfg.d_vocab), (1, seq))
    run_forward(model, tokens)  # warm-up
    t = time.time()
    fwd = run_forward(model, tokens)
    fwd_s = time.time() - t
    t = time.time()
    lens = run_lens(model, tokens[:, : min(256, seq)])
    lens_s = time.time() - t
    peak_all = (memory.peak_footprint_mb() or 0.0) - base

    cfg = model.cfg
    out = {
        "tl_name": args.tl_name,
        "device": SETTINGS.device,
        "threads": SETTINGS.threads,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "d_model": cfg.d_model,
        "d_vocab": cfg.d_vocab,
        "n_params": sum(p.numel() for p in model.parameters()),
        "resident_mb": round(after_load),
        "load_extra_mb": round(max(0.0, peak_load - after_load)),
        "request_transient_mb": round(max(0.0, peak_all - max(peak_load, after_load))),
        "stored_run_mb": round(fwd.nbytes / 2**20, 1),
        "stored_lens_run_mb": round(lens.nbytes / 2**20, 1),
        "load_s": round(load_s, 1),
        f"forward_{seq}_s": round(fwd_s, 2),
        "lens_256_s": round(lens_s, 2),
    }
    print(json.dumps(out, indent=2))
    print(
        "\nmodels.yaml:\n"
        f"  est_ram_mb: {_up(after_load)}\n"
        f"  load_extra_mb: {_up(max(0.0, peak_load - after_load))}\n"
        f"  n_layers: {cfg.n_layers}\n  n_heads: {cfg.n_heads}\n  d_model: {cfg.d_model}\n"
        f"  n_params: {out['n_params']}"
    )


if __name__ == "__main__":
    main()
