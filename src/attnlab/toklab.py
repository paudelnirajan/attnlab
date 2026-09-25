"""
The Tokenizer lab's engine: everything the lab shows about a tokenizer, with no
model and no forward pass anywhere in it.

Four jobs, each a pure function of (tokenizer, text):

  analyze()   tokens + exact byte provenance + stats + the pipeline stages
              (normalizer -> pre-tokenizer -> model) that produced them
  trace()     replays the tokenizer's own algorithm step by step (BPE merges
              in rank order, WordPiece's greedy longest match) and checks the
              replay against the real tokenizer's output
  vocab_*()   what is in the vocabulary, and how much of it each script got
  count()     token counts for many texts x many tokenizers, for the
              cross-language comparison

Tokenizers load through TokenizerCache, never through the model zoo: they are
a few MB and load in about a second, which is what lets the lab offer
tokenizers whose models this app could never host. See tokenizers.yaml.

Byte provenance is exact, not heuristic. For every tokenizer family that
reproduces its input byte-for-byte (byte-level BPE; SentencePiece BPE apart
from its prepended space) each token's bytes are recovered from its vocabulary
string, and the token boundaries are checked against UTF-8 character
boundaries directly. A token that starts or ends inside a character is a
fragment; that is a fact about bytes, not an inference from offsets. The
offset-overlap heuristic that inference.py uses is kept only as the fallback
for tokenizers that rewrite their input first (BERT lowercases and strips
accents, XLM-R collapses whitespace), where there are no original bytes left
to point at. Those are reported as not lossless, which is itself one of the
things the lab teaches.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import re
import threading
import unicodedata
from bisect import bisect_right
from collections import Counter
from importlib import resources
from pathlib import Path
from typing import Any

import regex
import yaml

from attnlab.inference import CONTINUATION, _byte_decoder, visible_whitespace

VALID_ALGORITHMS = {"byte-bpe", "sp-bpe", "wordpiece", "unigram"}

# Guards against a request that would make the response, not the tokenizer,
# the expensive part. Tokenizing 20k characters takes milliseconds.
MAX_TEXT_CHARS = 20_000
MAX_PRETOKENS_IN_PIPELINE = 2_000
MAX_TRACE_PRETOKENS = 48
MAX_TRACE_SYMBOLS = 256
MAX_COUNT_TEXTS = 64
MAX_COUNT_TOKENIZERS = 12
MAX_VOCAB_RESULTS = 200

# Scripts worth naming in the vocabulary breakdown. Order is display order for
# ties only; anything with no letter in one of these counts as "other"
# (digits, punctuation, whitespace, symbols, emoji).
SCRIPTS = (
    "Latin", "Cyrillic", "Greek", "Arabic", "Hebrew", "Devanagari", "Bengali",
    "Tamil", "Telugu", "Sinhala", "Thai", "Myanmar", "Khmer", "Ethiopic",
    "Georgian", "Armenian", "Han", "Hiragana", "Katakana", "Hangul",
)
_SCRIPT_RE = regex.compile("|".join(f"(?P<{s}>\\p{{Script={s}}})" for s in SCRIPTS))
_GRAPHEME_RE = regex.compile(r"\X")
_SP_BYTE_RE = re.compile(r"<0x([0-9A-Fa-f]{2})>")


# --------------------------------------------------------------- registry --


@dataclasses.dataclass(frozen=True)
class TokenizerSpec:
    id: str
    label: str
    hf_name: str
    algorithm: str
    year: int
    source: str
    models: list[str]
    blurb: str

    def __post_init__(self) -> None:
        if self.algorithm not in VALID_ALGORITHMS:
            raise ValueError(f"{self.id}: invalid algorithm {self.algorithm!r}, must be one of {VALID_ALGORITHMS}")
        if self.source not in {"official", "port"}:
            raise ValueError(f"{self.id}: source must be 'official' or 'port'")


def load_tokenizer_registry(path: str | Path | None = None) -> list[TokenizerSpec]:
    p = Path(path) if path is not None else Path(str(resources.files("attnlab").joinpath("tokenizers.yaml")))
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{p}: expected a YAML list of tokenizer specs")
    specs: list[TokenizerSpec] = []
    seen: set[str] = set()
    for entry in raw:
        spec = TokenizerSpec(**entry)
        if spec.id in seen:
            raise ValueError(f"duplicate tokenizer id in registry: {spec.id}")
        seen.add(spec.id)
        specs.append(spec)
    return specs


class UnknownTokenizerError(Exception):
    def __init__(self, tokenizer_id: str):
        self.tokenizer_id = tokenizer_id
        super().__init__(f"unknown tokenizer: {tokenizer_id}")


class TokenizerUnavailableError(Exception):
    """The Hub couldn't be reached (or refused) on first load."""

    def __init__(self, tokenizer_id: str, reason: str):
        self.tokenizer_id = tokenizer_id
        self.reason = reason
        super().__init__(f"could not load tokenizer {tokenizer_id}: {reason}")


# ------------------------------------------------------------------ cache --


class LoadedTokenizer:
    """A loaded HF tokenizer plus the derived tables the lab needs, each built
    lazily on first use and kept for the life of the process."""

    def __init__(self, spec: TokenizerSpec, tok: Any):
        self.spec = spec
        self.tok = tok
        self.backend = tok.backend_tokenizer
        self.json = json.loads(self.backend.to_str())
        self.special_ids: frozenset[int] = frozenset(tok.all_special_ids or [])
        # Added tokens are matched on the raw text before normalization and
        # pre-tokenization, and their vocabulary string is literal text, not
        # the byte-level alphabet. NeoX's runs-of-spaces tokens are the
        # non-special example.
        self.added: dict[int, str] = {i: t.content for i, t in tok.added_tokens_decoder.items()}
        self.unk_id: int | None = tok.unk_token_id
        self._lock = threading.Lock()

    # -- derived tables ----------------------------------------------------

    @functools.cached_property
    def model_json(self) -> dict:
        return self.json["model"]

    @functools.cached_property
    def merge_ranks(self) -> dict[tuple[str, str], int]:
        merges = self.model_json.get("merges") or []
        ranks: dict[tuple[str, str], int] = {}
        for rank, m in enumerate(merges):
            pair = tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m)
            ranks.setdefault(pair, rank)  # type: ignore[arg-type]
        return ranks

    @functools.cached_property
    def merge_result_rank(self) -> dict[str, int]:
        """vocabulary string -> the rank of the merge that first produced it.
        BPE learns merges most-frequent-first, so this is roughly 'how common
        was this piece in the tokenizer's training data'."""
        out: dict[str, int] = {}
        for (a, b), rank in self.merge_ranks.items():
            out.setdefault(a + b, rank)
        return out

    @functools.cached_property
    def vocab(self) -> dict[str, int]:
        return self.tok.get_vocab()

    @functools.cached_property
    def unigram_scores(self) -> dict[str, float]:
        if self.model_json.get("type") != "Unigram":
            return {}
        return {piece: float(score) for piece, score in self.model_json.get("vocab", [])}

    @functools.cached_property
    def vocab_table(self) -> list[dict]:
        """One row per id. Built once (about 1 s for a 250k vocabulary)."""
        with self._lock:
            rows: list[dict] = []
            for tid in range(len(self.tok)):
                piece = self.tok.convert_ids_to_tokens(tid)
                if piece is None:  # holes in some converted vocabularies
                    continue
                b = self.piece_bytes(tid, piece)
                text = _utf8_or_none(b) if b is not None else None
                rows.append(
                    {
                        "id": tid,
                        "piece": piece,
                        "text": text,
                        "byte_hex": b.hex() if b is not None else None,
                        "n_bytes": len(b) if b is not None else None,
                        "kind": self.kind(tid, piece, b),
                        "script": _script_of(text) if text is not None else "partial",
                    }
                )
            return rows

    # -- per-token facts ---------------------------------------------------

    def piece_bytes(self, tid: int, piece: str | None = None) -> bytes | None:
        """The exact bytes one token contributes to the text, or None when the
        token isn't text (a special token) or the family can't say."""
        if tid in self.special_ids:
            return None
        if tid in self.added:
            return self.added[tid].encode("utf-8")
        if piece is None:
            piece = self.tok.convert_ids_to_tokens(tid)
        if piece is None:
            return None
        algo = self.spec.algorithm
        if algo == "byte-bpe":
            dec = _byte_decoder()
            try:
                return bytes(dec[c] for c in piece)
            except KeyError:
                return piece.encode("utf-8")
        if algo in {"sp-bpe", "unigram"}:
            m = _SP_BYTE_RE.fullmatch(piece)
            if m:
                return bytes([int(m.group(1), 16)])
            return piece.replace("▁", " ").encode("utf-8")
        if algo == "wordpiece":
            return (piece[2:] if piece.startswith("##") else piece).encode("utf-8")
        return None

    def kind(self, tid: int, piece: str, b: bytes | None) -> str:
        if tid in self.special_ids:
            # Byte-level BPE can spell anything, so it never emits <unk>; GPT-2
            # just reuses <|endoftext|> as its nominal unk_token.
            if tid == self.unk_id and self.spec.algorithm != "byte-bpe":
                return "unk"
            return "special"
        if tid in self.added:
            return "added"
        if self.spec.algorithm in {"sp-bpe", "unigram"} and _SP_BYTE_RE.fullmatch(piece):
            return "byte"
        if b is not None and _utf8_or_none(b) is None:
            return "byte"  # not a character on its own: part of one
        return "piece"

    def rank_of(self, piece: str) -> int | None:
        return self.merge_result_rank.get(piece)

    # -- description of the pipeline, for the "how it works" panel ---------

    def describe(self) -> dict:
        j = self.json

        def names(component: dict | None, key: str) -> list[str]:
            if not component:
                return []
            if component.get("type") == "Sequence":
                return [c.get("type", "?") for c in component.get(key, [])]
            return [component.get("type", "?")]

        pre = j.get("pre_tokenizer")
        patterns: list[str] = []

        def collect_patterns(c: dict | None) -> None:
            if not c:
                return
            if c.get("type") == "Sequence":
                for sub in c.get("pretokenizers", []):
                    collect_patterns(sub)
            elif c.get("type") == "Split":
                pat = c.get("pattern", {})
                patterns.append(pat.get("Regex") or pat.get("String") or "")
            elif c.get("type") == "ByteLevel" and c.get("use_regex"):
                patterns.append("GPT-2 regex: 's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+")

        collect_patterns(pre)
        model = self.model_json
        return {
            "normalizers": names(j.get("normalizer"), "normalizers"),
            "pre_tokenizers": names(pre, "pretokenizers"),
            "split_patterns": patterns,
            "model": model.get("type"),
            "byte_fallback": bool(model.get("byte_fallback")),
            "n_merges": len(model.get("merges") or []),
            "decoders": names(j.get("decoder"), "decoders"),
        }


class TokenizerCache:
    """Process-wide: loads each registered tokenizer once. Thread-safe so the
    API can run tokenizer work in a thread pool without the model semaphore —
    tokenizing must not queue behind a forward pass."""

    def __init__(self, registry_path: str | None = None):
        self._specs = {s.id: s for s in load_tokenizer_registry(registry_path)}
        self._loaded: dict[str, LoadedTokenizer] = {}
        self._lock = threading.Lock()

    def list_specs(self) -> list[TokenizerSpec]:
        return list(self._specs.values())

    def spec(self, tokenizer_id: str) -> TokenizerSpec:
        try:
            return self._specs[tokenizer_id]
        except KeyError:
            raise UnknownTokenizerError(tokenizer_id) from None

    def is_loaded(self, tokenizer_id: str) -> bool:
        return tokenizer_id in self._loaded

    def get(self, tokenizer_id: str) -> LoadedTokenizer:
        spec = self.spec(tokenizer_id)
        with self._lock:
            if tokenizer_id not in self._loaded:
                from transformers import AutoTokenizer

                try:
                    tok = AutoTokenizer.from_pretrained(spec.hf_name)
                except Exception as e:  # noqa: BLE001 - network, gated repo, missing files
                    raise TokenizerUnavailableError(tokenizer_id, f"{type(e).__name__}: {e}") from None
                if not getattr(tok, "is_fast", False):
                    raise TokenizerUnavailableError(tokenizer_id, "no fast (Rust) tokenizer available")
                self._loaded[tokenizer_id] = LoadedTokenizer(spec, tok)
            return self._loaded[tokenizer_id]


# ---------------------------------------------------------------- helpers --


def _utf8_or_none(b: bytes) -> str | None:
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _script_of(text: str) -> str:
    m = _SCRIPT_RE.search(text)
    return m.lastgroup if m and m.lastgroup else "other"


def visible(s: str) -> str:
    """visible_whitespace, plus the characters that are otherwise impossible to
    see on a chip: format/control characters (zero-width space, the joiners
    inside emoji sequences) become ⟨U+200B⟩, and a combining mark that starts
    the string gets a dotted circle to sit on instead of the previous chip."""
    out = []
    for i, ch in enumerate(visible_whitespace(s)):
        cat = unicodedata.category(ch)
        if cat in {"Cf", "Cc"}:
            out.append(f"⟨U+{ord(ch):04X}⟩")
        elif i == 0 and cat.startswith("M"):
            out.append("◌" + ch)
        else:
            out.append(ch)
    return "".join(out)


def _is_char_start(byte: int) -> bool:
    return (byte & 0b1100_0000) != 0b1000_0000


def symbol_display(piece: str, algorithm: str) -> str:
    """How to print one vocabulary string (or BPE working symbol) so a reader
    can recognise it: real text where it decodes, hex where it's half a
    character, never U+FFFD."""
    if algorithm == "byte-bpe":
        dec = _byte_decoder()
        try:
            b = bytes(dec[c] for c in piece)
        except KeyError:
            return visible(piece)
        text = _utf8_or_none(b)
        return visible(text) if text is not None else "‹" + b.hex(" ") + "›"
    if algorithm in {"sp-bpe", "unigram"}:
        return visible(piece.replace("▁", " "))
    return piece


def text_stats(text: str) -> dict:
    return {
        "n_chars": len(text),  # code points
        "n_graphemes": len(_GRAPHEME_RE.findall(text)),  # characters as a reader sees them
        "n_bytes": len(text.encode("utf-8")),
        "n_words": len(text.split()),
    }


# ---------------------------------------------------------------- analyze --


def analyze(lt: LoadedTokenizer, text: str, *, add_special_tokens: bool = False) -> dict:
    tok = lt.tok
    enc = tok(text, add_special_tokens=add_special_tokens, return_offsets_mapping=True)
    ids: list[int] = list(enc["input_ids"])
    hf_spans = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    pieces = tok.convert_ids_to_tokens(ids)
    token_bytes = [lt.piece_bytes(tid, p) for tid, p in zip(ids, pieces)]

    text_b = text.encode("utf-8")
    textual = [i for i, tid in enumerate(ids) if tid not in lt.special_ids]
    # A special token that was MATCHED in the text ("<|endoftext|>" typed
    # literally) covers real characters; one that add_special_tokens inserted
    # (BOS/EOS) covers none. Only the first kind is part of the text.
    inserted = {i for i, tid in enumerate(ids) if tid in lt.special_ids and hf_spans[i][1] <= hf_spans[i][0]}
    check_bytes = [
        text[hf_spans[i][0] : hf_spans[i][1]].encode("utf-8") if ids[i] in lt.special_ids else token_bytes[i]
        for i in range(len(ids))
    ]
    covered = [i for i in range(len(ids)) if i not in inserted]
    joined = b"".join(check_bytes[i] or b"" for i in covered)
    if all(check_bytes[i] is not None for i in covered) and joined == text_b:
        prefix = 0
        lossless = True
    elif all(check_bytes[i] is not None for i in covered) and joined == b" " + text_b:
        prefix = 1  # SentencePiece's prepended ▁ is not in the user's text
        lossless = True
    else:
        prefix = 0
        lossless = False

    n = len(ids)
    spans: list[tuple[int, int]] = [(0, 0)] * n
    starts_mid_char = [False] * n
    ends_mid_char = [False] * n

    if lossless:
        # byte offset of the start of every character, for byte->char mapping
        char_starts = [i for i, byte in enumerate(text_b) if _is_char_start(byte)]
        cursor = 0
        last_char_end = 0
        for i in range(n):
            if i in inserted:
                spans[i] = (last_char_end, last_char_end)
                continue
            b = check_bytes[i] or b""
            b0, b1 = cursor, cursor + len(b)
            cursor = b1
            t0, t1 = max(0, b0 - prefix), max(0, b1 - prefix)
            if t1 <= t0:
                spans[i] = (last_char_end, last_char_end)
                continue
            c0 = bisect_right(char_starts, t0) - 1
            c1 = bisect_right(char_starts, t1 - 1)
            spans[i] = (c0, c1)
            last_char_end = c1
            starts_mid_char[i] = t0 < len(text_b) and not _is_char_start(text_b[t0])
            ends_mid_char[i] = t1 < len(text_b) and not _is_char_start(text_b[t1])
    else:
        spans = hf_spans

    # Clusters: consecutive tokens that together cover one indivisible piece of
    # text. Exact in lossless mode (a boundary inside a character joins the two
    # tokens); by overlapping offsets otherwise.
    clusters: list[list[int]] = []
    for i in range(n):
        special = ids[i] in lt.special_ids
        if lossless:
            joins = bool(clusters) and not special and starts_mid_char[i] and ids[clusters[-1][-1]] not in lt.special_ids
        else:
            prev = clusters[-1] if clusters else None
            joins = (
                prev is not None
                and not special
                and ids[prev[-1]] not in lt.special_ids
                and spans[i][0] < max(spans[j][1] for j in prev)
                and spans[i][1] > spans[i][0]
            )
        if joins:
            clusters[-1].append(i)
        else:
            clusters.append([i])

    records: list[dict] = []
    for cid, members in enumerate(clusters):
        lo = min(spans[i][0] for i in members)
        hi = max(spans[i][1] for i in members)
        cluster_text = text[lo:hi]
        for pos, i in enumerate(members):
            tid, piece, b = ids[i], pieces[i], token_bytes[i]
            special = tid in lt.special_ids
            fragment = len(members) > 1
            decoded = _utf8_or_none(b) if b is not None else None
            if special:
                display = piece
            elif fragment:
                display = visible(cluster_text) if pos == 0 else CONTINUATION
            elif lt.spec.algorithm == "wordpiece":
                display = piece  # the ## is the lesson
            elif decoded is not None:
                display = visible(decoded)
            else:
                display = "‹" + (b.hex(" ") if b else "?") + "›"
            records.append(
                {
                    "index": i,
                    "id": tid,
                    "piece": piece,
                    "display": display or "∅",
                    "text": decoded,
                    "byte_hex": b.hex() if b is not None else None,
                    "start": spans[i][0],
                    "end": spans[i][1],
                    "kind": lt.kind(tid, piece, b),
                    "rank": lt.rank_of(piece) if lt.spec.algorithm in {"byte-bpe", "sp-bpe"} else None,
                    "score": lt.unigram_scores.get(piece) if lt.spec.algorithm == "unigram" else None,
                    "cluster": cid,
                    "cluster_size": len(members),
                    "cluster_index": pos,
                    "cluster_text": visible(cluster_text),
                }
            )

    decoded_text = tok.decode([tid for i, tid in enumerate(ids) if i not in inserted])
    n_textual = len(textual)
    stats = {
        **text_stats(text),
        "n_tokens": n,
        "n_special": n - n_textual,
        "n_inserted": len(inserted),
        "n_fragment_tokens": sum(1 for r in records if r["cluster_size"] > 1),
        "n_byte_tokens": sum(1 for r in records if r["kind"] == "byte"),
        "n_unk": sum(1 for r in records if r["kind"] == "unk"),
        "lossless": lossless,
        "roundtrip": decoded_text == text,
    }

    return {
        "tokenizer": lt.spec.id,
        "tokens": records,
        "stats": stats,
        "decoded": decoded_text,
        "pipeline": pipeline(lt, text),
    }


def pipeline(lt: LoadedTokenizer, text: str) -> dict:
    """The stages before the model: what the normalizer did to the text, and
    the chunks the pre-tokenizer cut it into. BPE merges never cross a
    pre-token boundary, which is why this stage decides so much."""
    bt = lt.backend
    normalized = bt.normalizer.normalize_str(text) if bt.normalizer is not None else text
    if bt.pre_tokenizer is not None:
        pre = bt.pre_tokenizer.pre_tokenize_str(normalized)
    else:
        pre = [(normalized, (0, len(normalized)))]
    truncated = len(pre) > MAX_PRETOKENS_IN_PIPELINE
    return {
        **lt.describe(),
        "normalized": normalized,
        "normalized_changed": normalized != text,
        "pretokens": [
            {"raw": raw, "display": symbol_display(raw, lt.spec.algorithm), "start": int(s), "end": int(e)}
            for raw, (s, e) in pre[:MAX_PRETOKENS_IN_PIPELINE]
        ],
        "pretokens_truncated": truncated,
    }


# ------------------------------------------------------------------ trace --


def _model_words(lt: LoadedTokenizer, text: str) -> list[str]:
    """The strings the tokenizer's model actually runs on, in its own alphabet
    (byte-level characters, or ▁ for spaces)."""
    bt = lt.backend
    normalized = bt.normalizer.normalize_str(text) if bt.normalizer is not None else text
    if bt.pre_tokenizer is not None:
        words = [raw for raw, _ in bt.pre_tokenizer.pre_tokenize_str(normalized)]
    else:
        words = [normalized]
    if lt.spec.algorithm == "sp-bpe" and bt.pre_tokenizer is None:
        # Llama-style: no pre-tokenizer, so the whole normalized string is one
        # BPE word. SentencePiece never merges across a ▁ boundary, so tracing
        # each ▁-word separately is the same computation made readable —
        # and the replay is still checked against the real output below.
        words = [w for word in words for w in re.findall(r"▁[^▁]*|[^▁]+", word)]
    return [w for w in words if w]


def _bpe_trace_word(lt: LoadedTokenizer, word: str) -> dict:
    algo = lt.spec.algorithm
    ranks = lt.merge_ranks
    vocab = lt.vocab
    notes: list[str] = []

    if lt.model_json.get("ignore_merges") and word in vocab:
        symbols = [word]
        notes.append("whole word is already in the vocabulary, so no merges run")
        steps: list[dict] = []
    else:
        symbols = []
        for ch in word:
            if algo == "sp-bpe" and ch not in vocab and lt.model_json.get("byte_fallback"):
                symbols.extend(f"<0x{b:02X}>" for b in ch.encode("utf-8"))
            else:
                symbols.append(ch)
        if algo == "sp-bpe" and any(_SP_BYTE_RE.fullmatch(s) for s in symbols):
            notes.append("characters missing from the vocabulary fell back to their raw UTF-8 bytes")
        steps = []
        while len(symbols) > 1 and len(steps) < MAX_TRACE_SYMBOLS:
            best: tuple[int, str, str] | None = None
            for a, b in zip(symbols, symbols[1:]):
                r = ranks.get((a, b))
                if r is not None and (best is None or r < best[0]):
                    best = (r, a, b)
            if best is None:
                break
            rank, a, b = best
            merged: list[str] = []
            at: list[int] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    at.append(len(merged))
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
            steps.append(
                {
                    "rank": rank,
                    "left": symbol_display(a, algo),
                    "right": symbol_display(b, algo),
                    "merged": symbol_display(a + b, algo),
                    "at": at,
                    "symbols": [symbol_display(s, algo) for s in symbols],
                }
            )
    return {"steps": steps, "final_pieces": symbols, "notes": notes}


def _wordpiece_trace_word(lt: LoadedTokenizer, word: str) -> dict:
    """BERT's rule: repeatedly take the LONGEST vocabulary entry that matches
    at the cursor (continuations spelled with ##). If any position has no
    match at all, the whole word becomes [UNK]."""
    vocab = lt.vocab
    prefix = lt.model_json.get("continuing_subword_prefix", "##")
    unk = lt.model_json.get("unk_token", "[UNK]")
    max_chars = int(lt.model_json.get("max_input_chars_per_word", 100))
    steps: list[dict] = []
    if len(word) > max_chars:
        return {"steps": [], "final_pieces": [unk], "notes": [f"longer than {max_chars} characters: [UNK]"]}
    pieces: list[str] = []
    start = 0
    while start < len(word):
        end = len(word)
        tried = 0
        match = None
        while start < end:
            cand = word[start:end]
            if start > 0:
                cand = prefix + cand
            if cand in vocab:
                match = cand
                break
            tried += 1
            end -= 1
        if match is None:
            steps.append({"rank": None, "left": word[start:], "right": "", "merged": unk, "at": [], "symbols": [unk], "tried": tried})
            return {"steps": steps, "final_pieces": [unk], "notes": ["no vocabulary entry matched here, so the WHOLE word becomes [UNK]"]}
        pieces.append(match)
        steps.append(
            {"rank": None, "left": word[start:], "right": "", "merged": match, "at": [len(pieces) - 1], "symbols": list(pieces), "tried": tried}
        )
        start = end
    return {"steps": steps, "final_pieces": pieces, "notes": []}


def trace(lt: LoadedTokenizer, text: str) -> dict:
    algo = lt.spec.algorithm
    words = _model_words(lt, text)
    truncated = len(words) > MAX_TRACE_PRETOKENS
    out: list[dict] = []
    for word in words[:MAX_TRACE_PRETOKENS]:
        actual = [t.value for t in lt.backend.model.tokenize(word)]
        if algo in {"byte-bpe", "sp-bpe"}:
            if len(word) > MAX_TRACE_SYMBOLS:
                result = {"steps": [], "final_pieces": actual, "notes": ["too long to replay step by step"]}
            else:
                result = _bpe_trace_word(lt, word)
        elif algo == "wordpiece":
            result = _wordpiece_trace_word(lt, word)
        else:
            result = {
                "steps": [],
                "final_pieces": actual,
                "notes": ["Unigram has no merge sequence to replay: it scores every possible segmentation and keeps the most probable one"],
            }
        initial = [symbol_display(c, algo) for c in word] if algo != "wordpiece" else [word]
        out.append(
            {
                "raw": word,
                "display": symbol_display(word, algo),
                "initial": initial,
                "steps": result["steps"],
                "final": [symbol_display(p, algo) for p in result["final_pieces"]],
                "final_ids": [lt.vocab.get(p) for p in result["final_pieces"]],
                "actual": [symbol_display(p, algo) for p in actual],
                "scores": [lt.unigram_scores.get(p) for p in actual] if algo == "unigram" else None,
                "verified": result["final_pieces"] == actual,
                "notes": result["notes"],
            }
        )
    return {
        "tokenizer": lt.spec.id,
        "algorithm": algo,
        "n_merges": len(lt.merge_ranks),
        "words": out,
        "truncated": truncated,
    }


# ------------------------------------------------------------------ vocab --


def vocab_summary(lt: LoadedTokenizer) -> dict:
    rows = lt.vocab_table
    by_script = Counter(r["script"] for r in rows if r["kind"] not in {"special", "unk"})
    kinds = Counter(r["kind"] for r in rows)
    # Tokens with letters in them: the no-letter longest tokens are runs of
    # "=" and "-" from web page layout, true but not what anyone came to see.
    textual = [
        r for r in rows if r["text"] is not None and r["kind"] in {"piece", "added"} and r["script"] not in {"other", "partial"}
    ]
    longest = sorted(textual, key=lambda r: (-len(r["text"]), r["id"]))[:12]
    return {
        "tokenizer": lt.spec.id,
        "size": len(lt.tok),
        "n_rows": len(rows),
        "n_merges": len(lt.merge_ranks),
        "kinds": dict(kinds),
        "by_script": [{"script": s, "count": c} for s, c in by_script.most_common()],
        "longest": [_vocab_row(lt, r) for r in longest],
        "special": [_vocab_row(lt, r) for r in rows if r["kind"] in {"special", "unk"}][:40],
    }


def _vocab_row(lt: LoadedTokenizer, r: dict) -> dict:
    return {
        **r,
        "display": symbol_display(r["piece"], lt.spec.algorithm) if r["kind"] not in {"special", "unk"} else r["piece"],
        "rank": lt.rank_of(r["piece"]),
    }


def vocab_search(lt: LoadedTokenizer, query: str, *, script: str | None = None, limit: int = 100) -> dict:
    """Case-insensitive substring search over what each token SPELLS (not its
    vocabulary string), so 'hello' finds ' Hello' as well. An integer query
    is an id lookup."""
    rows = lt.vocab_table
    limit = max(1, min(limit, MAX_VOCAB_RESULTS))
    q = query.strip()
    if q.lstrip("-").isdigit():
        tid = int(q)
        hits = [r for r in rows if r["id"] == tid]
        total = len(hits)
    else:
        ql = query.casefold()
        hits_iter = (
            r
            for r in rows
            if (script is None or r["script"] == script)
            and (not ql or (r["text"] is not None and ql in r["text"].casefold()))
        )
        hits = []
        total = 0
        for r in hits_iter:
            total += 1
            if len(hits) < limit:
                hits.append(r)
    return {"tokenizer": lt.spec.id, "query": query, "total": total, "results": [_vocab_row(lt, r) for r in hits]}


# ------------------------------------------------------------------ count --


def count(lts: list[LoadedTokenizer], texts: list[str]) -> dict:
    """Tokens per text for each tokenizer, special tokens excluded so the
    numbers compare the TEXT's cost, not each model's BOS convention."""
    out = []
    for lt in lts:
        enc = lt.tok(texts, add_special_tokens=False)["input_ids"]
        out.append({"tokenizer": lt.spec.id, "counts": [len(x) for x in enc]})
    return {"texts": [text_stats(t) for t in texts], "results": out}
