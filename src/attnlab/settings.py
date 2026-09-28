"""
Environment-driven configuration — the one place Mode A (native/MPS) vs.
Mode B (constrained container) differ. Nothing else in the codebase should
read `os.environ` directly for these values.

See docs/PLAN.md > "Two-mode development" for why this split exists:
Docker Desktop on macOS runs Linux VMs, so MPS is unavailable inside a
container — you cannot have "fast MPS" and "enforced memory limits" in the
same process. So we develop in two modes and treat only Mode B's numbers
as ground truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import torch


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val else default


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val else default


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name) or default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    return default if not val else val.lower() in ("1", "true", "yes", "on")


def _default_device() -> str:
    """CPU unless MI_DEVICE says otherwise, on every machine (D16). Measured on
    Apple Silicon, the CPU was faster than MPS for every model in the zoo
    (gpt2-small at 512 tokens: 0.13 s vs 0.94 s), and TransformerLens warns
    that MPS can be silently wrong on this PyTorch, which matters when
    learners check our numbers against their Colab output (D3)."""
    return "cpu"


@dataclass(frozen=True)
class Settings:
    device: str = field(default_factory=lambda: os.environ.get("MI_DEVICE", _default_device()))
    # dtype is intentionally NOT env-configurable: fp32 on CPU always.
    # TransformerLens's weight processing (LayerNorm folding, centering) is
    # numerically sensitive, and non-AMX x86 CPUs are *slower* in bf16 than
    # fp32 — there is no upside to changing this outside a GPU path.
    dtype: torch.dtype = torch.float32
    threads: int = field(default_factory=lambda: _env_int("MI_THREADS", os.cpu_count() or 4))
    ram_budget_gb: float = field(default_factory=lambda: _env_float("MI_RAM_BUDGET_GB", 12.0))
    max_seq: int = field(default_factory=lambda: _env_int("MI_MAX_SEQ", 512))
    hf_token: str | None = field(default_factory=lambda: os.environ.get("HF_TOKEN"))
    debug_metrics: bool = field(
        default_factory=lambda: os.environ.get("MI_DEBUG_METRICS", "0") == "1"
    )

    # --- Serving caps (docs/04-self-hosting.md § 2-3). The defaults suit a dev
    # machine; deploy/server.env sets the values for the 16 GB server. ---

    # Stored runs, by encoded size. Oldest are dropped first; the 10 min TTL
    # still applies. A dropped run is a 404 the frontend answers by re-running.
    run_budget_mb: float = field(default_factory=lambda: _env_float("MI_RUN_BUDGET_MB", 1024.0))
    lens_budget_mb: float = field(default_factory=lambda: _env_float("MI_LENS_BUDGET_MB", 768.0))
    # Requests allowed to wait for the model slot. Past this: 503 busy.
    max_queue: int = field(default_factory=lambda: _env_int("MI_MAX_QUEUE", 8))
    request_timeout_s: float = field(default_factory=lambda: _env_float("MI_REQUEST_TIMEOUT_S", 60.0))
    # Whole-process physical memory ceiling (0 = off). Above it, caches are
    # emptied and models evicted; if that isn't enough, model work gets 503.
    memory_limit_gb: float = field(default_factory=lambda: _env_float("MI_MEMORY_LIMIT_GB", 0.0))
    max_text_chars: int = field(default_factory=lambda: _env_int("MI_MAX_TEXT_CHARS", 10_000))
    max_body_bytes: int = field(default_factory=lambda: _env_int("MI_MAX_BODY_BYTES", 256 * 1024))
    # Tokenizer-lab work runs outside the model slot; this bounds it instead.
    tokenizer_concurrency: int = field(default_factory=lambda: _env_int("MI_TOKENIZER_CONCURRENCY", 2))
    # Per-client token bucket over POST /api/* (0 = off): `rate_burst` requests
    # at once, refilled at `rate_per_s`.
    rate_burst: int = field(default_factory=lambda: _env_int("MI_RATE_BURST", 0))
    rate_per_s: float = field(default_factory=lambda: _env_float("MI_RATE_PER_S", 1.0))
    # Behind Cloudflare the client address is in CF-Connecting-IP. Only trust
    # it when the server is reachable through Cloudflare alone (a tunnel, or
    # bound to 127.0.0.1), or anyone could pick their own rate-limit key.
    trust_proxy: bool = field(default_factory=lambda: _env_bool("MI_TRUST_PROXY", False))
    cors_origins: tuple[str, ...] = field(
        default_factory=lambda: tuple(o.strip() for o in _env_str("MI_CORS_ORIGINS", "").split(",") if o.strip())
    )
    # The built frontend (web/dist). Served by the API when present, so one
    # process and one origin serve the whole site.
    static_dir: str = field(default_factory=lambda: _env_str("MI_STATIC_DIR", ""))
    # Load the `baked` models at startup rather than on the first request.
    preload: bool = field(default_factory=lambda: _env_bool("MI_PRELOAD", False))

    @property
    def ram_budget_bytes(self) -> int:
        return int(self.ram_budget_gb * 1024**3)


def _apply_thread_limits(settings: Settings) -> None:
    """Must run before any model load. Without this, `--cpus=2` in Docker
    (a CFS *quota*, not CPU affinity) is invisible to torch — it still sees
    the host's full core count via os.cpu_count() and spawns that many OMP
    threads, which then thrash against 2 cores' worth of scheduler quota.
    The result is *worse* than honest 2-thread performance, not just
    unmeasured."""
    torch.set_num_threads(settings.threads)
    os.environ.setdefault("OMP_NUM_THREADS", str(settings.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(settings.threads))


SETTINGS = Settings()
_apply_thread_limits(SETTINGS)

# What the server logs at startup, so a misconfigured deploy is visible.
SERVING_KEYS = (
    "device", "threads", "ram_budget_gb", "run_budget_mb", "lens_budget_mb", "max_queue",
    "request_timeout_s", "memory_limit_gb", "max_text_chars", "max_body_bytes",
    "tokenizer_concurrency", "rate_burst", "rate_per_s", "trust_proxy", "cors_origins",
    "static_dir", "preload",
)


if __name__ == "__main__":
    import json

    print(
        json.dumps(
            {
                "device": SETTINGS.device,
                "dtype": str(SETTINGS.dtype),
                "threads": SETTINGS.threads,
                "torch_num_threads": torch.get_num_threads(),
                "ram_budget_gb": SETTINGS.ram_budget_gb,
                "max_seq": SETTINGS.max_seq,
                "debug_metrics": SETTINGS.debug_metrics,
                **{k: getattr(SETTINGS, k) for k in SERVING_KEYS},
            },
            indent=2,
        )
    )
