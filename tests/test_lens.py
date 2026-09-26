"""Logit lens engine + endpoints.

The engine is checked against the plain PyTorch computation from the ARENA-style
notebook the lab is built from (`from_pretrained_no_processing`, normalize by
hand, multiply by the raw W_U). The app's models are loaded with LayerNorm
folded into W_U, so the plain lens is reconstructed from folding facts captured
at load time (lens.RawUnembed). The first test proves that reconstruction, and
the second proves the loader still produces exactly the model every other lab
uses.
"""

from __future__ import annotations

import pytest
import torch
from fastapi.testclient import TestClient
from transformer_lens import HookedTransformer

from attnlab.api.app import app
from attnlab.lens import (
    LensReadyTransformer,
    attribution,
    dangling_per_position,
    label_inputs,
    label_prediction,
    layer_curves,
    lens_logits,
    output_logits,
    position_detail,
    run_lens,
    summarize,
)

DEVICE = "cpu"  # exact comparisons; MPS is a Mode A convenience, not ground truth
PLASMA = (
    "Sometimes, when people say plasma, they mean a state of matter. "
    "Other times, when people say plasma, they mean"
)


@pytest.fixture(scope="module")
def gpt2():
    torch.set_grad_enabled(False)
    return LensReadyTransformer.from_pretrained("gpt2", device=DEVICE)


@pytest.fixture(scope="module")
def plasma(gpt2):
    tokens = gpt2.to_tokens(PLASMA, prepend_bos=False)
    return tokens, run_lens(gpt2, tokens)


def block_rows(run):
    return [i for i, r in enumerate(run.rows) if r.block_end]


class TestLoader:
    def test_weights_identical_to_from_pretrained(self, gpt2):
        ref = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
        a, b = gpt2.state_dict(), ref.state_dict()
        assert a.keys() == b.keys()
        assert all(torch.equal(a[k], b[k]) for k in b)

    def test_raw_facts(self, gpt2):
        raw = gpt2.raw_unembed
        assert raw.norm == "LN" and raw.folded and raw.tied
        assert raw.fold_error < 1e-4
        # the notebook's "ln_final.b @ W_U" frequent-word prior
        assert [gpt2.tokenizer.decode([t]) for t in raw.bias_prior[:4]] == [",", " the", " and", "."]


@pytest.fixture(scope="module")
def notebook(plasma):
    """The notebook's lenses, computed by hand on the unprocessed weights."""
    tokens, _ = plasma
    raw = HookedTransformer.from_pretrained_no_processing("gpt2", device=DEVICE)
    _, cache = raw.run_with_cache(tokens)
    resid = torch.stack([cache["resid_pre", 0]] + [cache["resid_post", li] for li in range(12)])[:, 0]
    mu = resid.mean(-1, keepdim=True)
    normed = (resid - mu) / ((resid - mu).pow(2).mean(-1, keepdim=True) + 1e-5).sqrt()
    return {"plain": normed @ raw.W_U, "ln_final": raw.ln_final(resid) @ raw.W_U + raw.b_U}


class TestAgainstTheNotebook:
    @pytest.mark.parametrize("lens", ["plain", "ln_final"])
    def test_lens_matches_raw_weights(self, gpt2, plasma, notebook, lens):
        _, run = plasma
        ours = lens_logits(gpt2, run.resid[block_rows(run)], lens)
        ref = notebook[lens]
        assert (ours.log_softmax(-1) - ref.log_softmax(-1)).abs().max() < 1e-3
        assert torch.equal(ours.argmax(-1), ref.argmax(-1))

    def test_headline_numbers(self, gpt2, plasma):
        tokens, run = plasma
        final_top = output_logits(gpt2, run).argmax(-1)
        plain_last = lens_logits(gpt2, run.resid[-1], "plain").argmax(-1)
        # "h11_out and h_out agree on top-1 at 17% of positions"
        assert round(float((plain_last == final_top).float().mean()), 2) == 0.17

        ranks = {}
        for lens in ("ln_final", "plain"):
            rn = summarize(gpt2, run, lens, gpt2.tokenizer)["cells"]["rank_next"]
            ranks[lens] = [rn[i][19] for i in block_rows(run)]
        # ' plasma' at the second ' say': rank 2 after block 8, rank 1 from block 9 on
        assert ranks["ln_final"][9:] == [2, 1, 1, 1]
        assert ranks["plain"][-1] > 1000  # the plain lens never finds it


class TestAttribution:
    @pytest.mark.parametrize("contrast", [None, 262])  # 262 = ' the'
    def test_components_sum_to_the_logit(self, gpt2, plasma, contrast):
        tokens, run = plasma
        a = attribution(gpt2, run, gpt2.tokenizer, 19, int(tokens[0, 20]), contrast)
        assert a["error"] < 1e-3
        assert abs(sum(c["value"] for c in a["components"]) + a["bias"] - a["actual"]) < 1e-2
        n_heads = sum(c["kind"] == "head" for c in a["components"])
        assert n_heads == 144

    def test_checks_all_pass(self, plasma):
        _, run = plasma
        assert {c["id"] for c in run.checks} >= {"lens_reproduces_output", "stream_is_a_sum", "heads_sum_to_attn_out"}
        assert all(c["ok"] for c in run.checks), run.checks


class TestSummaries:
    def test_output_row_is_the_model(self, gpt2, plasma):
        _, run = plasma
        s = summarize(gpt2, run, "ln_final", gpt2.tokenizer)
        assert s["rows"][-1]["kind"] == "output"
        assert len(s["cells"]["top"]) == len(run.rows) + 1
        # KL of the output row against itself is 0; the last position has no next token
        assert max(abs(v) for v in s["cells"]["kl"][-1]) < 1e-4
        assert s["cells"]["p_next"][0][-1] is None and s["next_labels"][-1] is None

    def test_layer_curves_end_at_full_agreement(self, gpt2, plasma):
        _, run = plasma
        c = layer_curves(gpt2, run, gpt2.tokenizer)
        assert set(c["lenses"]) == {"ln_final", "plain"}
        assert c["lenses"]["ln_final"]["agree_final"][-1] == 1.0
        assert c["lenses"]["ln_final"]["agree_final"][-2] == 1.0  # same vector, same function

    def test_tracked_token(self, gpt2, plasma):
        tokens, run = plasma
        d = position_detail(gpt2, run, gpt2.tokenizer, 19, "ln_final", 5, [int(tokens[0, 20])])
        assert d["tracked"][0]["rank"][-1] == 1
        assert len(d["top"]) == len(run.rows) + 1


class TestLabels:
    def test_fragments(self, gpt2):
        ids = gpt2.to_tokens("नेपाल एक", prepend_bos=False)[0].tolist()
        assert label_inputs(gpt2.tokenizer, ids)[:4] == ["न⋯", "⋯न", "े⋯", "⋯े"]
        d = dangling_per_position(gpt2.tokenizer, ids)
        # predicting e0 a4 after a whole character: the letter isn't chosen yet
        assert label_prediction(gpt2.tokenizer, d[1], ids[0]) == "ऄ–ऽ⋯"
        # predicting a8 right after e0 a4 finishes न
        assert label_prediction(gpt2.tokenizer, d[0], ids[1]) == "⋯न"


# --- endpoints, on the fast model ---------------------------------------------

MODEL = "attn-only-2l-demo"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def lens_run(client):
    r = client.post("/api/lens/run", json={"model": MODEL, "text": "The cat sat on the mat. The cat sat on the"})
    assert r.status_code == 200, r.text
    return r.json()


class TestEndpoints:
    def test_run_shape(self, lens_run):
        b = lens_run
        seq = len(b["tokens"])
        assert b["lenses"] == ["ln_final"]  # no output norm, so no plain lens
        assert [r["label"] for r in b["rows"]] == ["embed", "L0 +attn", "L1 +attn", "output"]
        for key in ("top", "top_p", "p_next", "rank_next", "rank_final", "entropy", "kl", "norm"):
            assert len(b["cells"][key]) == len(b["rows"])
            assert len(b["cells"][key][0]) == seq
        assert len(b["input_labels"]) == seq
        assert all(c["ok"] for c in b["checks"])
        assert b["anatomy"]["raw"]["norm"] is None

    def test_plain_lens_refused_without_a_norm(self, client, lens_run):
        r = client.post("/api/lens/view", json={"run_id": lens_run["run_id"], "lens": "plain"})
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "invalid_request"

    def test_layers(self, client, lens_run):
        r = client.post("/api/lens/layers", json={"run_id": lens_run["run_id"]})
        assert r.status_code == 200
        assert len(r.json()["lenses"]["ln_final"]["agree_final"]) == len(lens_run["rows"])

    def test_position_and_tracking(self, client, lens_run):
        last = len(lens_run["tokens"]) - 1
        r = client.post(
            "/api/lens/position",
            json={"run_id": lens_run["run_id"], "pos": last, "track": [" mat", "Kathmandu"], "k": 3},
        )
        assert r.status_code == 200
        b = r.json()
        assert len(b["top"]) == len(lens_run["rows"]) and len(b["top"][0]) == 3
        assert b["resolved"][0]["note"] == ""
        assert "tokens" in b["resolved"][1]["note"]  # multi-token strings say so

    def test_attribution_defaults_to_next_token(self, client, lens_run):
        r = client.post("/api/lens/attribution", json={"run_id": lens_run["run_id"], "pos": 1})
        assert r.status_code == 200
        b = r.json()
        assert b["target"]["id"] == lens_run["tokens"][2]["id"]
        assert b["error"] < 1e-3

    def test_errors(self, client, lens_run):
        r = client.post("/api/lens/position", json={"run_id": "l_missing", "pos": 0})
        assert r.status_code == 404 and r.json()["error"]["code"] == "run_not_found"
        r = client.post("/api/lens/position", json={"run_id": lens_run["run_id"], "pos": 999})
        assert r.status_code == 422
        r = client.post("/api/lens/run", json={"model": MODEL, "text": "word " * 400})
        assert r.status_code == 422 and r.json()["error"]["code"] == "seq_too_long"
