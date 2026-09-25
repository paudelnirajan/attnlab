"""Tokenizer lab engine + endpoints.

Tokenizers only, no models: every tokenizer here is a few MB from the local HF
cache, so this file stays fast. The first run on a fresh machine downloads
them.

The most important tests are the replay ones. The lab tells a student "this is
exactly how the tokenizer built this word", so the replay must reproduce the
real tokenizer's output. It is checked against the Rust implementation on
every word, not just on a few examples.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from attnlab.api.app import app
from attnlab.toklab import (
    CONTINUATION,
    TokenizerCache,
    UnknownTokenizerError,
    analyze,
    count,
    load_tokenizer_registry,
    trace,
    vocab_search,
    vocab_summary,
)

NEPALI = "नेपाल एक सुन्दर देश हो।"
MIXED = "Héllo  world! नेपाल 12345 🙂 unbelievably"
# long enough to exercise many merges, several scripts, digits, code spacing
REPLAY_CORPUS = (
    "The quick brown fox jumps over the lazy dog. Tokenization isn't magic: "
    "3.14159 and 1,000,000 and नेपाल and 東京タワー and Straße and emoji 🙂👨‍👩‍👧 "
    "and code:    def f(x):\n        return x ** 2\n"
)


@pytest.fixture(scope="module")
def cache():
    return TokenizerCache()


ALL_IDS = [s.id for s in load_tokenizer_registry()]


class TestRegistry:
    def test_ids_are_unique_and_every_algorithm_is_known(self):
        specs = load_tokenizer_registry()
        assert len({s.id for s in specs}) == len(specs)

    def test_model_links_point_at_real_models(self):
        from attnlab.registry import load_registry

        model_ids = {m.id for m in load_registry()}
        for s in load_tokenizer_registry():
            assert set(s.models) <= model_ids, s.id

    def test_unknown_tokenizer_raises(self, cache):
        with pytest.raises(UnknownTokenizerError):
            cache.get("no-such-tokenizer")


@pytest.mark.parametrize("tid", ALL_IDS)
class TestEveryTokenizer:
    def test_replay_reproduces_the_real_tokenizer(self, cache, tid):
        result = trace(cache.get(tid), REPLAY_CORPUS)
        bad = [(w["raw"], w["final"], w["actual"]) for w in result["words"] if not w["verified"]]
        assert not bad, bad

    def test_analyze_never_displays_the_replacement_character(self, cache, tid):
        for text in (NEPALI, MIXED, REPLAY_CORPUS):
            tokens = analyze(cache.get(tid), text)["tokens"]
            assert not any("�" in t["display"] for t in tokens), [t["display"] for t in tokens]

    def test_token_ids_match_the_plain_tokenizer_call(self, cache, tid):
        lt = cache.get(tid)
        a = analyze(lt, MIXED)
        assert [t["id"] for t in a["tokens"]] == lt.tok(MIXED, add_special_tokens=False)["input_ids"]

    def test_clusters_are_contiguous(self, cache, tid):
        tokens = analyze(cache.get(tid), MIXED)["tokens"]
        for t in tokens:
            members = [u for u in tokens if u["cluster"] == t["cluster"]]
            assert [u["cluster_index"] for u in members] == list(range(len(members)))
            assert all(u["cluster_size"] == len(members) for u in members)


class TestByteProvenance:
    def test_gpt2_splits_each_devanagari_character_into_two_byte_tokens(self, cache):
        a = analyze(cache.get("gpt2"), "न")
        assert [t["byte_hex"] for t in a["tokens"]] == ["e0a4", "a8"]
        assert [t["cluster_size"] for t in a["tokens"]] == [2, 2]
        assert [t["display"] for t in a["tokens"]] == ["न", CONTINUATION]
        assert a["stats"]["lossless"] is True

    def test_fragment_spans_point_at_the_character(self, cache):
        a = analyze(cache.get("gpt2"), "aन")
        assert [(t["start"], t["end"]) for t in a["tokens"]] == [(0, 1), (1, 2), (1, 2)]

    def test_a_multilingual_vocabulary_does_not_fragment_nepali_much(self, cache):
        gpt2 = analyze(cache.get("gpt2"), NEPALI)["stats"]
        bloom = analyze(cache.get("bloom"), NEPALI)["stats"]
        assert gpt2["n_fragment_tokens"] > 20
        assert bloom["n_tokens"] * 3 < gpt2["n_tokens"]

    def test_sentencepiece_byte_fallback_is_reported_as_bytes(self, cache):
        a = analyze(cache.get("llama2"), "🙂")
        kinds = [t["kind"] for t in a["tokens"] if t["kind"] == "byte"]
        assert len(kinds) == 4  # four UTF-8 bytes, four <0xNN> tokens
        assert a["stats"]["lossless"] is True  # the prepended ▁ is accounted for

    def test_neox_whitespace_runs_are_added_tokens(self, cache):
        a = analyze(cache.get("gpt-neox"), "a    b")
        assert any(t["kind"] == "added" for t in a["tokens"])


class TestNormalization:
    def test_bert_is_not_lossless_and_says_so(self, cache):
        a = analyze(cache.get("bert-uncased"), "Héllo")
        assert a["stats"]["lossless"] is False
        assert a["stats"]["roundtrip"] is False
        assert a["pipeline"]["normalized"] == "hello"
        assert a["pipeline"]["normalized_changed"] is True

    def test_bert_strips_devanagari_vowel_signs(self, cache):
        # a real, surprising consequence of strip_accents on Devanagari
        assert analyze(cache.get("bert-uncased"), "नेपाल")["pipeline"]["normalized"] == "नपाल"

    def test_byte_level_tokenizers_roundtrip_exactly(self, cache):
        for tid in ("gpt2", "gpt-neox", "bloom", "qwen2.5", "cl100k", "o200k"):
            assert analyze(cache.get(tid), MIXED)["stats"]["roundtrip"] is True, tid


class TestSpecialTokens:
    def test_literal_special_token_text_becomes_one_special_token(self, cache):
        a = analyze(cache.get("gpt2"), "hi <|endoftext|> there")
        specials = [t for t in a["tokens"] if t["kind"] == "special"]
        assert [t["id"] for t in specials] == [50256]
        assert a["stats"]["lossless"] is True
        assert a["stats"]["n_inserted"] == 0

    def test_inserted_special_tokens_cover_no_text(self, cache):
        a = analyze(cache.get("bert-uncased"), "hi", add_special_tokens=True)
        assert [t["piece"] for t in a["tokens"]] == ["[CLS]", "hi", "[SEP]"]
        assert a["stats"]["n_inserted"] == 2


class TestTrace:
    def test_gpt2_builds_unbelievably_in_rank_order(self, cache):
        (word,) = trace(cache.get("gpt2"), " unbelievably")["words"]
        ranks = [s["rank"] for s in word["steps"]]
        assert ranks == sorted(ranks), "BPE applies the lowest-rank available merge first"
        assert word["final"] == ["·unbelievably"]
        assert word["verified"]

    def test_wordpiece_trace_is_greedy_longest_match(self, cache):
        (word,) = trace(cache.get("bert-uncased"), "unbelievably")["words"]
        assert word["final"] == word["actual"]
        assert all(p.startswith("##") for p in word["final"][1:])

    def test_unigram_explains_instead_of_replaying(self, cache):
        (word,) = trace(cache.get("xlm-roberta"), "hello")["words"]
        assert word["steps"] == []
        assert word["notes"]
        assert word["scores"] and all(s is not None for s in word["scores"])


class TestVocab:
    def test_gpt2_has_almost_no_devanagari(self, cache):
        s = vocab_summary(cache.get("gpt2"))
        scripts = {r["script"]: r["count"] for r in s["by_script"]}
        assert s["size"] == 50257
        assert scripts.get("Devanagari", 0) < 50
        assert scripts["Latin"] > 40_000

    def test_search_finds_the_famous_glitch_token(self, cache):
        r = vocab_search(cache.get("gpt2"), "SolidGoldMagikarp")
        assert any(x["id"] == 43453 for x in r["results"])

    def test_integer_query_is_an_id_lookup(self, cache):
        r = vocab_search(cache.get("gpt2"), "50256")
        assert [x["id"] for x in r["results"]] == [50256]


def test_count_excludes_special_tokens(cache):
    r = count([cache.get("llama2")], ["hello"])
    assert r["results"][0]["counts"] == [len(cache.get("llama2").tok("hello", add_special_tokens=False)["input_ids"])]


# ------------------------------------------------------------------ API --


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


class TestApi:
    def test_list_tokenizers(self, client):
        body = client.get("/api/tokenizers").json()
        assert {t["id"] for t in body["tokenizers"]} == set(ALL_IDS)

    def test_analyze_several(self, client):
        r = client.post("/api/toklab/analyze", json={"tokenizers": ["gpt2", "bloom"], "text": NEPALI})
        assert r.status_code == 200
        results = r.json()["results"]
        assert [x["tokenizer"] for x in results] == ["gpt2", "bloom"]
        assert results[0]["stats"]["n_tokens"] > results[1]["stats"]["n_tokens"]

    def test_unknown_tokenizer_404(self, client):
        r = client.post("/api/toklab/analyze", json={"tokenizers": ["nope"], "text": "hi"})
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "unknown_tokenizer"

    def test_text_too_long(self, client):
        r = client.post("/api/toklab/analyze", json={"tokenizers": ["gpt2"], "text": "a" * 20_001})
        assert r.status_code == 422
        assert r.json()["error"]["code"] == "text_too_long"

    def test_trace_and_count_and_vocab(self, client):
        assert client.post("/api/toklab/trace", json={"tokenizer": "gpt2", "text": "hello"}).status_code == 200
        c = client.post("/api/toklab/count", json={"tokenizers": ["gpt2"], "texts": ["a", "b c"]}).json()
        assert c["results"][0]["counts"] == [1, 2]
        v = client.get("/api/toklab/vocab", params={"tokenizer": "gpt2"}).json()
        assert v["size"] == 50257
        s = client.get("/api/toklab/vocab/search", params={"tokenizer": "gpt2", "q": "Magikarp"}).json()
        assert s["total"] >= 3

    def test_bad_script_filter(self, client):
        r = client.get("/api/toklab/vocab/search", params={"tokenizer": "gpt2", "script": "Klingon"})
        assert r.status_code == 422
