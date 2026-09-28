"""
The serving caps (docs/04-self-hosting.md): bounded run stores, the model
slot's queue cap / disconnect skip / timeout, the memory guard, and the HTTP
guards. No real model is loaded here; each test takes a few milliseconds.
"""

from __future__ import annotations

import asyncio
import dataclasses
import threading
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from attnlab.api import guards
from attnlab.api.app import mount_frontend
from attnlab.api.errors import BusyError, ClientGoneError, OverloadedError
from attnlab.api.state import AppState, RunRecord, RunStore
from attnlab.settings import SETTINGS


def _rec(run_id: str, nbytes: int, *, ttl: float = 600) -> RunRecord:
    now = time.time()
    return RunRecord(run_id, "m", [b"x" * nbytes], 1, 1, 1, now, now + ttl)


def _settings(monkeypatch, **changes):
    monkeypatch.setattr("attnlab.api.state.SETTINGS", dataclasses.replace(SETTINGS, **changes))


class TestRunStore:
    def test_drops_oldest_to_stay_under_budget(self):
        store = RunStore(budget_mb=3 / 2**20)  # 3 bytes
        for rid in "abc":
            store.put(_rec(rid, 1))
        store.put(_rec("d", 2))
        assert store.get("a") is None and store.get("b") is None
        assert store.get("c") and store.get("d")
        assert store.nbytes == 3

    def test_max_items(self):
        store = RunStore(budget_mb=1, max_items=2)
        for rid in "abc":
            store.put(_rec(rid, 1))
        assert [store.get(r) is not None for r in "abc"] == [False, True, True]

    def test_expired_runs_are_gone_and_uncounted(self):
        store = RunStore(budget_mb=1)
        store.put(_rec("old", 10, ttl=-1))
        assert store.get("old") is None
        assert store.nbytes == 0

    def test_a_run_bigger_than_the_budget_is_kept_alone(self):
        store = RunStore(budget_mb=1 / 2**20)
        store.put(_rec("a", 1))
        store.put(_rec("big", 5))
        assert store.get("big") is not None and store.get("a") is None


class _FakeRequest:
    def __init__(self, gone: bool):
        self.gone = gone

    async def is_disconnected(self) -> bool:
        return self.gone


class TestModelSlot:
    def test_full_queue_is_refused_with_retry_after(self, monkeypatch):
        _settings(monkeypatch, max_queue=0)
        state = AppState()
        with pytest.raises(BusyError) as e:
            asyncio.run(state.run_serialized(lambda: 1))
        assert e.value.retry_after

    def test_disconnected_client_is_skipped(self):
        state = AppState()
        ran = []
        with pytest.raises(ClientGoneError):
            asyncio.run(state.run_serialized(lambda: ran.append(1), request=_FakeRequest(gone=True)))
        assert ran == []
        assert state.semaphore._value == 1  # slot released

    def test_timeout_keeps_the_slot_until_the_thread_finishes(self, monkeypatch):
        _settings(monkeypatch, request_timeout_s=0.05)
        state = AppState()
        release = threading.Event()

        async def scenario():
            with pytest.raises(BusyError):
                await state.run_serialized(release.wait)
            held_after_timeout = state.semaphore.locked()
            release.set()
            for _ in range(100):
                if not state.semaphore.locked():
                    break
                await asyncio.sleep(0.01)
            return held_after_timeout, state.semaphore.locked()

        held, still_held = asyncio.run(scenario())
        assert held, "the slot was released while the timed-out work was still running"
        assert not still_held


class TestMemoryGuard:
    def _state(self, monkeypatch, footprints):
        _settings(monkeypatch, memory_limit_gb=1.0)  # 1024 MB
        readings = iter(footprints)
        monkeypatch.setattr("attnlab.memory.footprint_mb", lambda: next(readings))
        state = AppState()
        state.runs.put(_rec("r", 10))
        return state

    def test_under_the_limit_does_nothing(self, monkeypatch):
        state = self._state(monkeypatch, [500])
        state.guard_memory()
        assert len(state.runs) == 1

    def test_over_the_limit_frees_runs_first(self, monkeypatch):
        state = self._state(monkeypatch, [2000, 900])
        state.guard_memory()
        assert len(state.runs) == 0
        assert state.counters["memory_guard"] == 1

    def test_still_over_after_freeing_refuses(self, monkeypatch):
        state = self._state(monkeypatch, [2000, 2000, 2000])
        with pytest.raises(OverloadedError):
            state.guard_memory()

    def test_evicts_every_model_if_that_is_what_it_takes(self, monkeypatch):
        state = self._state(monkeypatch, [2000, 1500, 900])
        evicted = []
        monkeypatch.setattr(state.zoo, "resident_ids", lambda: [] if evicted else ["only"])
        monkeypatch.setattr(state.zoo, "evict_lru", lambda keep=None: evicted.append(1) or "only")
        state.guard_memory()  # no exception: the last model went, and that was enough
        assert evicted == [1]


def _app_with(*middleware) -> TestClient:
    app = FastAPI()

    @app.post("/api/echo")
    async def echo(body: dict) -> dict:
        return body

    @app.get("/api/ping")
    async def ping() -> dict:
        return {"ok": True}

    for cls, kwargs in middleware:
        app.add_middleware(cls, **kwargs)
    return TestClient(app)


class TestHttpGuards:
    def test_rate_limit_bucket(self):
        rl = guards.RateLimit(None, burst=2, per_s=1.0)
        assert rl.take("ip", 0.0) == 0 and rl.take("ip", 0.0) == 0
        assert rl.take("ip", 0.0) == pytest.approx(1.0)
        assert rl.take("ip", 1.0) == 0  # refilled one token
        assert rl.take("other", 0.0) == 0  # buckets are per client

    def test_rate_limit_answers_429_on_posts_only(self):
        c = _app_with((guards.RateLimit, {"burst": 1, "per_s": 0.01}))
        assert c.post("/api/echo", json={}).status_code == 200
        r = c.post("/api/echo", json={})
        assert r.status_code == 429 and r.json()["error"]["code"] == "rate_limited"
        assert int(r.headers["retry-after"]) >= 1
        assert c.get("/api/ping").status_code == 200

    def test_body_limit(self):
        c = _app_with((guards.BodyLimit, {"max_bytes": 100}))
        assert c.post("/api/echo", json={"t": "x" * 10}).status_code == 200
        r = c.post("/api/echo", json={"t": "x" * 200})
        assert r.status_code == 413 and r.json()["error"]["code"] == "payload_too_large"

    def test_security_headers(self):
        c = _app_with((guards.Headers, {}))
        assert c.get("/api/ping").headers["x-content-type-options"] == "nosniff"

    def test_client_key_trusts_cloudflare_header_only_when_told(self, monkeypatch):
        scope = {"client": ("127.0.0.1", 1), "headers": [(b"cf-connecting-ip", b"203.0.113.9")]}
        monkeypatch.setattr("attnlab.api.guards.SETTINGS", dataclasses.replace(SETTINGS, trust_proxy=False))
        assert guards.client_key(scope) == "127.0.0.1"
        monkeypatch.setattr("attnlab.api.guards.SETTINGS", dataclasses.replace(SETTINGS, trust_proxy=True))
        assert guards.client_key(scope) == "203.0.113.9"


class TestFrontend:
    @pytest.fixture
    def client(self, tmp_path):
        (tmp_path / "assets").mkdir()
        (tmp_path / "index.html").write_text("<html>app</html>")
        (tmp_path / "assets" / "app-abc123.js").write_text("js")
        (tmp_path / "favicon.svg").write_text("<svg/>")
        app = FastAPI()
        mount_frontend(app, tmp_path.resolve())
        return TestClient(app)

    def test_hashed_assets_are_immutable(self, client):
        r = client.get("/assets/app-abc123.js")
        assert r.text == "js" and "immutable" in r.headers["cache-control"]

    def test_client_routes_get_index_html_revalidated(self, client):
        for path in ("/", "/tokens", "/lens/whatever"):
            r = client.get(path)
            assert r.text == "<html>app</html>" and r.headers["cache-control"] == "no-cache"

    def test_other_files_are_served_but_revalidated(self, client):
        r = client.get("/favicon.svg")
        assert r.text == "<svg/>" and r.headers["cache-control"] == "no-cache"

    def test_unknown_api_paths_are_json_404s(self, client):
        r = client.get("/api/nope")
        assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"

    def test_no_path_traversal(self, client, tmp_path):
        (tmp_path.parent / "secret.txt").write_text("secret")
        r = client.get("/../secret.txt")
        assert "secret" not in r.text
