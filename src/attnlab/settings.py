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


def _default_device() -> str:
    """Prefer MPS natively on Apple Silicon; CPU everywhere else (Docker
    included — MPS is never available inside a Linux container)."""
    if torch.backends.mps.is_available():
        return "mps"
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
            },
            indent=2,
        )
    )
