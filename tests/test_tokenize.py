"""Tokenization and byte-fragment clustering.

These exercise the pure functions in inference.py against a real HF tokenizer
but WITHOUT loading a model — a tokenizer is a few hundred KB and loads in
milliseconds, so this file stays fast enough to run on every save.
"""

from __future__ import annotations

import pytest
from transformers import AutoTokenizer

from attnlab.inference import (
    CONTINUATION,
    _build_records,
    _byte_decoder,
    _hf_offsets,
    _search_offsets,
    _token_byte_hex,
)

# GPT-2 has no merges for Devanagari, so every character falls back to raw
# UTF-8 bytes. This is the case the whole clustering feature exists for.
NEPALI = "म हिमालयी"
ENGLISH = "The quick brown fox"


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("gpt2")


def records_for(tok, text, *, bos=False):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if bos:
        ids = [tok.eos_token_id, *ids]
    spans = _hf_offsets(tok, text, ids)
    assert spans is not None, "gpt2 is a fast tokenizer; offsets must be available"
    return _build_records(tok, text, ids, spans)


class TestByteDecoder:
    def test_covers_every_byte(self):
        assert len(_byte_decoder()) == 256
        assert sorted(_byte_decoder().values()) == list(range(256))

    def test_recovers_the_bytes_of_a_latin_token(self, tok):
        (tid,) = tok(" fox", add_special_tokens=False)["input_ids"]
        assert bytes.fromhex(_token_byte_hex(tok, tid)) == b" fox"

    def test_recovers_the_bytes_of_a_fragment_that_cannot_decode(self, tok):
        ids = tok("म", add_special_tokens=False)["input_ids"]
        assert len(ids) == 2, "GPT-2 spends two tokens on this character"
        joined = b"".join(bytes.fromhex(_token_byte_hex(tok, i)) for i in ids)
        # Neither token is a character; together they are exactly the one char.
        assert joined == "म".encode()
        assert all("�" in tok.decode([i]) for i in ids)

    def test_says_nothing_for_a_special_token(self, tok):
        # its "bytes" would be the bytes of the literal name "<|endoftext|>"
        assert _token_byte_hex(tok, tok.eos_token_id) is None


class TestOffsets:
    def test_refuses_when_the_id_sequences_disagree(self, tok):
        # a guard, so model.to_tokens stays the single source of truth
        assert _hf_offsets(tok, ENGLISH, [1, 2, 3]) is None

    def test_accepts_a_leading_bos_and_gives_it_a_zero_width_span(self, tok):
        ids = [tok.eos_token_id, *tok(ENGLISH, add_special_tokens=False)["input_ids"]]
        spans = _hf_offsets(tok, ENGLISH, ids)
        assert spans is not None
        assert spans[0] == (0, 0)
        assert len(spans) == len(ids)

    def test_fragments_of_one_character_all_report_that_character(self, tok):
        ids = tok("म", add_special_tokens=False)["input_ids"]
        assert _hf_offsets(tok, "म", ids) == [(0, 1), (0, 1)]

    def test_search_fallback_still_locates_plain_text(self, tok):
        ids = tok(ENGLISH, add_special_tokens=False)["input_ids"]
        spans = _search_offsets(tok, ENGLISH, ids)
        assert [ENGLISH[a:b] for a, b in spans] == ["The", " quick", " brown", " fox"]


class TestClustering:
    def test_latin_text_is_one_token_per_cluster(self, tok):
        records = records_for(tok, ENGLISH)
        assert all(r.cluster_size == 1 for r in records)
        assert not any(r.is_byte_fallback for r in records)
        assert [r.display for r in records] == ["The", "·quick", "·brown", "·fox"]

    def test_each_devanagari_character_becomes_one_multi_token_cluster(self, tok):
        records = records_for(tok, "म")
        assert [r.cluster_size for r in records] == [2, 2]
        assert [r.cluster_index for r in records] == [0, 1]
        assert {r.cluster for r in records} == {0}
        assert all(r.cluster_text == "म" for r in records)

    def test_no_record_ever_displays_the_replacement_character(self, tok):
        for text in (NEPALI, ENGLISH, "混合 mixed текст"):
            assert not any("�" in r.display for r in records_for(tok, text)), text

    def test_the_first_fragment_shows_the_character_and_the_rest_a_continuation(self, tok):
        records = records_for(tok, "म")
        assert records[0].display == "म"
        assert records[1].display == CONTINUATION

    def test_a_character_the_tokenizer_has_a_whole_token_for_is_not_clustered(self, tok):
        # U+093E DEVANAGARI VOWEL SIGN AA happens to be a single GPT-2 token
        records = records_for(tok, "ा")
        assert len(records) == 1
        assert records[0].cluster_size == 1
        assert records[0].is_byte_fallback is False

    def test_clusters_are_contiguous_and_cover_every_token_exactly_once(self, tok):
        records = records_for(tok, NEPALI, bos=True)
        seen: dict[int, list[int]] = {}
        for i, r in enumerate(records):
            seen.setdefault(r.cluster, []).append(r.cluster_index)
        for cluster, indices in seen.items():
            assert indices == list(range(len(indices))), f"cluster {cluster} is not contiguous"
            assert all(r.cluster_size == len(indices) for r in records if r.cluster == cluster)
        assert sorted(seen) == list(range(len(seen))), "cluster ids must be dense and ordered"

    def test_fragmentation_is_dramatically_worse_than_latin(self, tok):
        """The Stage 3 finding, pinned as a test: if a future tokenizer change
        makes this ratio comparable to English, the premise has changed."""
        nepali = records_for(tok, NEPALI)
        english = records_for(tok, ENGLISH)
        nepali_ratio = len(nepali) / len(NEPALI)
        english_ratio = len(english) / len(ENGLISH)
        assert nepali_ratio > 1.5, nepali_ratio
        assert english_ratio < 0.3, english_ratio
