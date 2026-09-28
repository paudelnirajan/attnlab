"""
Per-process application state: the model zoo, the stored runs, and the single
model slot that serializes all model-touching work (docs/03-decisions.md D6:
one forward pass / one model load at a time, server-wide — correct at this
scale, trivially understood, and avoids multi-worker's fatal flaw of each
worker holding its own copy of every resident model's weights).

Everything a request can make the server hold is bounded here
(docs/04-self-hosting.md § 2-3):

  models          the zoo's MI_RAM_BUDGET_GB, which reserves each load's peak
  attention runs  MI_RUN_BUDGET_MB of encoded bytes, oldest dropped first
  lens runs       MI_LENS_BUDGET_MB, the same way
  waiting         MI_MAX_QUEUE requests; the next one gets 503 busy
  one request     MI_REQUEST_TIMEOUT_S
  the process     MI_MEMORY_LIMIT_GB, checked before model work: caches are
                  emptied, then models evicted, then requests refused

Constructed once in app.py's lifespan and attached to `app.state`, not a
module-level global — keeps it possible to spin up isolated app instances in
tests.
"""

from __future__ import annotations

import asyncio
import dataclasses
import functools
import logging
import os
import secrets
import signal
import time
from collections import OrderedDict
from typing import Any, Callable, Generic, TypeVar

from starlette.requests import Request

from attnlab import memory
from attnlab.api.errors import BusyError, ClientGoneError, OverloadedError
from attnlab.lens import LensRun
from attnlab.settings import SETTINGS
from attnlab.toklab import TokenizerCache
from attnlab.zoo import ModelZoo

log = logging.getLogger("attnlab")

RUN_TTL_SECONDS = 600  # docs/02-api.md: "~10 min"
# Also bounded by count: a lens run keeps the whole residual stream, and only
# the most recent few are worth keeping whatever their size.
MAX_LENS_RUNS = 4


@dataclasses.dataclass
class RunRecord:
    run_id: str
    model_id: str
    layers: list[bytes]  # one patterns.encode_layer body per layer
    n_layers: int
    n_heads: int
    seq: int
    created_at: float
    expires_at: float

    @property
    def nbytes(self) -> int:
        return sum(len(b) for b in self.layers)


@dataclasses.dataclass
class LensRunRecord:
    run_id: str
    model_id: str
    run: LensRun
    created_at: float
    expires_at: float

    @property
    def nbytes(self) -> int:
        return self.run.nbytes


R = TypeVar("R", RunRecord, LensRunRecord)


class RunStore(Generic[R]):
    """Records by id, oldest first, bounded by total bytes, count and age.
    Adding one drops the oldest until it fits. A dropped or expired run is a
    404 `run_not_found`, which the frontend answers by running again."""

    def __init__(self, *, budget_mb: float, max_items: int | None = None, ttl_s: float = RUN_TTL_SECONDS):
        self.budget_bytes = int(budget_mb * 2**20)
        self.max_items = max_items
        self.ttl_s = ttl_s
        self._items: OrderedDict[str, R] = OrderedDict()
        self.nbytes = 0

    def __len__(self) -> int:
        return len(self._items)

    def _drop(self, run_id: str) -> None:
        rec = self._items.pop(run_id, None)
        if rec is not None:
            self.nbytes -= rec.nbytes

    def sweep(self) -> None:
        now = time.time()
        for k in [k for k, v in self._items.items() if v.expires_at < now]:
            self._drop(k)

    def drop_oldest(self) -> bool:
        if not self._items:
            return False
        self._drop(next(iter(self._items)))
        return True

    def clear(self) -> None:
        self._items.clear()
        self.nbytes = 0

    def put(self, rec: R) -> R:
        self.sweep()
        while self._items and (
            self.nbytes + rec.nbytes > self.budget_bytes
            or (self.max_items is not None and len(self._items) >= self.max_items)
        ):
            self.drop_oldest()
        # A single run bigger than the whole budget is still kept (alone), so
        # the request that made it can be served; the next put drops it.
        self._items[rec.run_id] = rec
        self.nbytes += rec.nbytes
        return rec

    def get(self, run_id: str) -> R | None:
        rec = self._items.get(run_id)
        if rec is None or rec.expires_at < time.time():
            self._drop(run_id)
            return None
        return rec


class AppState:
    def __init__(self, *, registry_path: str | None = None, budget_mb: float | None = None):
        self.zoo = ModelZoo(registry_path=registry_path, budget_mb=budget_mb)
        # Tokenizers are cached separately from models and are NOT behind the
        # model slot (see api/toklab_routes.py); `tokenizer_slots` bounds them.
        self.tokenizers = TokenizerCache()
        self.tokenizer_slots = asyncio.Semaphore(SETTINGS.tokenizer_concurrency)
        self.semaphore = asyncio.Semaphore(1)
        self.runs: RunStore[RunRecord] = RunStore(budget_mb=SETTINGS.run_budget_mb)
        self.lens_runs: RunStore[LensRunRecord] = RunStore(
            budget_mb=SETTINGS.lens_budget_mb, max_items=MAX_LENS_RUNS
        )
        self._waiting = 0  # requests currently queued behind the semaphore
        self.started_at = time.time()
        self.ready = False  # set once startup preloading has finished
        self.counters = {"busy": 0, "client_gone": 0, "timeouts": 0, "memory_guard": 0, "rate_limited": 0}

    @property
    def queue_depth(self) -> int:
        return self._waiting

    # -- the model slot ----------------------------------------------------

    async def run_serialized(
        self, fn: Callable[..., Any], *args: Any, request: Request | None = None, **kwargs: Any
    ) -> Any:
        """Run blocking model work in a thread, one at a time, server-wide.

        The thread executor keeps the event loop serving other connections
        (e.g. /health) while torch runs. In order:

          1. a full queue is refused at once (503 busy, Retry-After), so a
             burst can't pile up minutes of work;
          2. once the slot is ours, a client that has gone away (the frontend
             aborts a run when the prompt changes) is skipped rather than
             computed for nobody;
          3. the memory guard runs;
          4. on timeout the request fails, but the slot stays taken until the
             thread really finishes. A Python thread can't be killed, and
             releasing early would let the next forward pass run alongside it.
        """
        if self._waiting >= SETTINGS.max_queue:
            self.counters["busy"] += 1
            raise BusyError(
                f"the server is busy ({self._waiting} requests waiting); try again in a few seconds",
                {"queue_depth": self._waiting},
                retry_after=5,
            )
        self._waiting += 1
        try:
            await self.semaphore.acquire()
        finally:
            self._waiting -= 1

        release_now = True
        try:
            if request is not None and await request.is_disconnected():
                self.counters["client_gone"] += 1
                raise ClientGoneError("client disconnected before its turn")
            self.guard_memory()
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))
            try:
                return await asyncio.wait_for(asyncio.shield(fut), timeout=SETTINGS.request_timeout_s)
            except asyncio.TimeoutError:
                self.counters["timeouts"] += 1
                release_now = False
                fut.add_done_callback(lambda _f: self.semaphore.release())
                log.warning("request exceeded %.0fs; holding the model slot until it finishes", SETTINGS.request_timeout_s)
                raise BusyError(
                    f"request exceeded the {SETTINGS.request_timeout_s:.0f}s limit", retry_after=10
                ) from None
        finally:
            if release_now:
                self.semaphore.release()

    # -- the memory guard --------------------------------------------------

    def guard_memory(self) -> None:
        """Runs with the model slot held, so nothing is mid-load. Frees the
        cheapest memory first: stored runs, then models, least recently used
        first (the request reloads the one it needs). If the process is still
        over the limit with nothing left to free, the memory isn't anything we
        can hand back (a leak, fragmentation), so the request is refused and,
        under launchd, the process exits cleanly to be restarted fresh."""
        limit = SETTINGS.memory_limit_gb * 1024
        if limit <= 0:
            return
        used = memory.footprint_mb()
        if used is None or used <= limit:
            return
        self.counters["memory_guard"] += 1
        log.warning("memory guard: %.0f MB > %.0f MB limit, freeing", used, limit)
        self.runs.clear()
        self.lens_runs.clear()
        memory.release()
        while (used := memory.footprint_mb() or 0) > limit and self.zoo.resident_ids():
            log.warning("memory guard: evicting %s (%.0f MB used)", self.zoo.evict_lru(), used)
        if used > limit:
            log.error("memory guard: still %.0f MB with nothing left to free", used)
            if os.environ.get("XPC_SERVICE_NAME", "").startswith("com.attnlab.api"):
                log.error("memory guard: restarting (launchd starts a fresh process)")
                asyncio.get_running_loop().call_later(1.0, os.kill, os.getpid(), signal.SIGTERM)
            raise OverloadedError(
                "the server is low on memory; try again shortly", {"used_mb": round(used)}, retry_after=30
            )

    # -- stored runs ---------------------------------------------------------

    def store_run(
        self, *, model_id: str, layers: list[bytes], n_layers: int, n_heads: int, seq: int
    ) -> RunRecord:
        now = time.time()
        return self.runs.put(
            RunRecord(
                run_id="r_" + secrets.token_hex(4),
                model_id=model_id,
                layers=layers,
                n_layers=n_layers,
                n_heads=n_heads,
                seq=seq,
                created_at=now,
                expires_at=now + RUN_TTL_SECONDS,
            )
        )

    def get_run(self, run_id: str) -> RunRecord | None:
        return self.runs.get(run_id)

    def store_lens_run(self, *, model_id: str, run: LensRun) -> LensRunRecord:
        now = time.time()
        return self.lens_runs.put(
            LensRunRecord(
                run_id="l_" + secrets.token_hex(4),
                model_id=model_id,
                run=run,
                created_at=now,
                expires_at=now + RUN_TTL_SECONDS,
            )
        )

    def get_lens_run(self, run_id: str) -> LensRunRecord | None:
        return self.lens_runs.get(run_id)
