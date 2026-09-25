"""
Per-process application state: the model zoo, the run cache, and the
single global semaphore that serializes all model-touching work
(docs/03-decisions.md D6: one forward pass / one model load at a time,
server-wide — correct at this scale, trivially understood, and avoids
multi-worker's fatal flaw of each worker holding its own copy of every
resident model's weights).

Constructed once in app.py's lifespan and attached to `app.state`, not a
module-level global — keeps it possible to spin up isolated app
instances in tests.
"""

from __future__ import annotations

import asyncio
import dataclasses
import secrets
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import numpy as np

from attnlab.toklab import TokenizerCache
from attnlab.zoo import ModelZoo

RUN_TTL_SECONDS = 600  # docs/02-api.md: "~10 min"


@dataclasses.dataclass
class RunRecord:
    run_id: str
    model_id: str
    patterns: np.ndarray  # (n_layers, n_heads, seq, seq) float32 — see
    # the note in AppState.store_run about this being an intentional
    # Stage-0b simplification, not a Stage-4-ready design.
    n_layers: int
    n_heads: int
    seq: int
    created_at: float
    expires_at: float


class AppState:
    def __init__(self, *, registry_path: str | None = None, budget_mb: float | None = None):
        self.zoo = ModelZoo(registry_path=registry_path, budget_mb=budget_mb)
        # Tokenizers are cached separately from models and are NOT behind the
        # semaphore: see api/toklab_routes.py.
        self.tokenizers = TokenizerCache()
        self.semaphore = asyncio.Semaphore(1)
        self.runs: dict[str, RunRecord] = {}
        self._waiting = 0  # requests currently queued behind the semaphore

    @property
    def queue_depth(self) -> int:
        return self._waiting

    @asynccontextmanager
    async def serialize(self) -> AsyncGenerator[None]:
        """Wraps any model-touching work. Tracks queue_depth (for
        GET /api/health) correctly across cancellation: the increment/
        decrement around `acquire()` is in its own try/finally so a
        cancelled *wait* still decrements, and the semaphore is released
        in its own finally so a failure *inside* the block still frees it
        for the next request."""
        self._waiting += 1
        try:
            await self.semaphore.acquire()
        finally:
            self._waiting -= 1
        try:
            yield
        finally:
            self.semaphore.release()

    def _sweep_expired(self) -> None:
        now = time.time()
        expired = [k for k, v in self.runs.items() if v.expires_at < now]
        for k in expired:
            del self.runs[k]

    def store_run(
        self, *, model_id: str, patterns: np.ndarray, n_layers: int, n_heads: int, seq: int
    ) -> RunRecord:
        # NOTE (intentional Stage 0b simplification, flagged rather than
        # hidden): this stores the FULL float32 patterns array per run,
        # not the wire-format-encoded bytes. For gpt2-small @ seq=512
        # that's ~151MB per active run sitting in memory for up to
        # RUN_TTL_SECONDS. GET /run/{id}/patterns encodes on demand from
        # this raw array, which is correct and simple, but a Stage 4
        # hardening pass should revisit whether to store the
        # already-encoded bytes instead if run volume ever makes this a
        # real memory pressure (see docs/PLAN.md Stage 4).
        self._sweep_expired()
        run_id = "r_" + secrets.token_hex(3)
        now = time.time()
        rec = RunRecord(
            run_id=run_id,
            model_id=model_id,
            patterns=patterns,
            n_layers=n_layers,
            n_heads=n_heads,
            seq=seq,
            created_at=now,
            expires_at=now + RUN_TTL_SECONDS,
        )
        self.runs[run_id] = rec
        return rec

    def get_run(self, run_id: str) -> RunRecord | None:
        self._sweep_expired()
        return self.runs.get(run_id)
