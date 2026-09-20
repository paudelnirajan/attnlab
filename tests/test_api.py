"""
End-to-end API tests against the REAL app and a REAL (tiny) model —
attn-only-2l-demo loads in a few seconds, so this proves the whole stack
(routes -> zoo -> inference -> patterns wire format) actually works
together, not just that each piece is internally consistent.

Session-scoped TestClient so the model loads once for the whole file
rather than once per test.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from attnlab.api.app import app
from attnlab.patterns import decode_layers

MODEL = "attn-only-2l-demo"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestModelsAndHealth:
    def test_list_models_shape(self, client):
        r = client.get("/api/models")
        assert r.status_code == 200
        body = r.json()
        assert "budget" in body
        ids = {m["id"] for m in body["models"]}
        assert MODEL in ids
        entry = next(m for m in body["models"] if m["id"] == MODEL)
        for key in ("n_layers", "n_heads", "d_model", "n_params", "max_seq", "status", "tier"):
            assert key in entry

    def test_health_shape(self, client):
        r = client.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert "tl_version" in body
        assert body["queue_depth"] == 0


class TestTokenize:
    def test_tokenize_basic(self, client):
        r = client.post("/api/tokenize", json={"model": MODEL, "text": "The cat sat"})
        assert r.status_code == 200
        body = r.json()
        assert body["n_tokens"] == len(body["tokens"])
        assert body["n_tokens"] >= 4  # BOS + at least 3 word tokens
        assert "_meta" in body
        assert body["_meta"]["tl_version"]
        assert body["_meta"]["device"]
        # First token is BOS with a zero-width span (docs: not literally
        # present in the source text).
        assert body["tokens"][0]["start"] == 0
        assert body["tokens"][0]["end"] == 0

    def test_tokenize_unknown_model_404(self, client):
        r = client.post("/api/tokenize", json={"model": "no-such-model", "text": "hi"})
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "unknown_model"

    def test_tokenize_after_run_agrees_on_token_count(self, client):
        """The exact invariant that makes hover-linking possible: /tokenize
        and /run must agree on the token sequence for the same text."""
        text = "A quick brown fox"
        tok = client.post("/api/tokenize", json={"model": MODEL, "text": text}).json()
        run = client.post("/api/run", json={"model": MODEL, "text": text}).json()
        assert tok["n_tokens"] == len(run["tokens"])
        assert [t["id"] for t in tok["tokens"]] == [t["id"] for t in run["tokens"]]


class TestRun:
    def test_run_with_text(self, client):
        r = client.post("/api/run", json={"model": MODEL, "text": "The cat sat on the mat", "top_k": 3})
        assert r.status_code == 200
        body = r.json()
        assert body["run_id"].startswith("r_")
        assert body["n_layers"] == 2
        assert len(body["loss_per_token"]) == len(body["tokens"]) - 1
        assert len(body["top_logits"]) == len(body["tokens"])
        assert len(body["top_logits"][0]) == 3
        assert body["cost"]["attention_bytes_f32"] > 0
        assert body["cost"]["weights_bytes"] > 0
        assert "_meta" in body

    def test_run_with_repeated(self, client):
        r = client.post(
            "/api/run",
            json={"model": MODEL, "repeated": {"length": 5, "seed": 42, "prepend_bos": True}},
        )
        assert r.status_code == 200
        body = r.json()
        assert len(body["tokens"]) == 11  # bos + 5 + 5

    def test_run_repeated_is_seed_deterministic(self, client):
        r1 = client.post("/api/run", json={"model": MODEL, "repeated": {"length": 5, "seed": 7}})
        r2 = client.post("/api/run", json={"model": MODEL, "repeated": {"length": 5, "seed": 7}})
        ids1 = [t["id"] for t in r1.json()["tokens"]]
        ids2 = [t["id"] for t in r2.json()["tokens"]]
        assert ids1 == ids2

    def test_run_rejects_both_text_and_repeated(self, client):
        r = client.post(
            "/api/run",
            json={"model": MODEL, "text": "hi", "repeated": {"length": 5, "seed": 1}},
        )
        assert r.status_code == 422

    def test_run_rejects_neither_text_nor_repeated(self, client):
        r = client.post("/api/run", json={"model": MODEL})
        assert r.status_code == 422

    def test_run_seq_too_long(self, client):
        # attn-only-2l-demo's max_seq is 512; ask for something absurd.
        r = client.post(
            "/api/run",
            json={"model": MODEL, "repeated": {"length": 300, "seed": 1}},
        )
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "seq_too_long"

    def test_run_unknown_model_404(self, client):
        r = client.post("/api/run", json={"model": "nope", "text": "hi"})
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "unknown_model"


class TestPatterns:
    def test_patterns_round_trip_via_wire_format(self, client):
        run = client.post("/api/run", json={"model": MODEL, "text": "The cat sat"}).json()
        run_id = run["run_id"]
        n_layers = run["n_layers"]

        r = client.get(f"/api/run/{run_id}/patterns", params={"layers": "0"})
        assert r.status_code == 200
        assert r.headers["content-type"] == "application/octet-stream"

        decoded = decode_layers(r.content)
        assert decoded.layer_ids == [0]
        assert decoded.n_heads == run["n_heads"]
        assert decoded.seq == len(run["tokens"])
        # Every row of a real attention matrix sums to ~1.
        row_sums = decoded.arrays[0].sum(axis=-1)
        assert (abs(row_sums - 1.0) < 0.05).all()

        if n_layers > 1:
            r_multi = client.get(f"/api/run/{run_id}/patterns", params={"layers": "0,1"})
            decoded_multi = decode_layers(r_multi.content)
            assert decoded_multi.layer_ids == [0, 1]

    def test_patterns_missing_layers_param_is_error(self, client):
        run = client.post("/api/run", json={"model": MODEL, "text": "hi"}).json()
        r = client.get(f"/api/run/{run['run_id']}/patterns")
        assert r.status_code == 422  # FastAPI's own required-query-param rejection

    def test_patterns_out_of_range_layer(self, client):
        run = client.post("/api/run", json={"model": MODEL, "text": "hi"}).json()
        r = client.get(f"/api/run/{run['run_id']}/patterns", params={"layers": "99"})
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "invalid_request"

    def test_patterns_unknown_run_id_404(self, client):
        r = client.get("/api/run/r_doesnotexist/patterns", params={"layers": "0"})
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "run_not_found"
