"""
Shared tokenization + forward-pass logic used by the API layer (and
reusable later by bench.py / a CLI). Kept separate from api/routes.py so
it has no FastAPI dependency and is easy to unit test directly.
"""

from __future__ import annotations

import dataclasses
import functools
import re

import numpy as np
import torch
from transformer_lens import HookedTransformer

_VISIBLE_WS = {" ": "·", "\n": "⏎", "\t": "→"}

# Continuation marker for a token that is only part of a character. Shown
# instead of U+FFFD, which is what decoding half a UTF-8 sequence produces and
# which tells the reader nothing except "something broke".
CONTINUATION = "⋯"


def visible_whitespace(s: str) -> str:
    return "".join(_VISIBLE_WS.get(c, c) for c in s)


@functools.lru_cache(maxsize=1)
def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's `bytes_to_unicode`: the fixed map every byte-level
    BPE tokenizer (GPT-2, RoBERTa, NeoX, ...) uses to make raw bytes printable.
    Reimplemented here because the *fast* tokenizers don't expose
    `byte_decoder`, only the slow Python ones do."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def _token_byte_hex(tokenizer, tid: int) -> str | None:
    """The actual UTF-8 bytes this single token contributes, as hex — the one
    fully truthful thing that can be said about a fragment that doesn't decode
    to a character on its own. Returns None when the tokenizer family isn't
    one we can do this for, rather than guessing."""
    try:
        piece = tokenizer.convert_ids_to_tokens(tid)
    except Exception:  # noqa: BLE001 - tokenizer families vary; None is a fine answer
        return None
    if not isinstance(piece, str):
        return None
    # A special token's "bytes" would be the bytes of its *name* ("<|endoftext|>"),
    # which is not what it represents in the stream. Say nothing instead.
    if tid in set(getattr(tokenizer, "all_special_ids", []) or []):
        return None
    # SentencePiece byte-fallback spells it out literally.
    m = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", piece)
    if m:
        return m.group(1).lower()
    dec = _byte_decoder()
    try:
        return bytes(dec[c] for c in piece).hex()
    except KeyError:
        return None


def _hf_offsets(tokenizer, text: str, ids: list[int]) -> list[tuple[int, int]] | None:
    """Character spans into `text` for each of `ids`, from the fast tokenizer's
    offset mapping.

    This is exact where the old cursor-search heuristic could not be: for a
    byte-fragment token, HF reports the span of the character the fragment
    belongs to, so two tokens of one Devanagari character both report that
    character's span. That is precisely the signal needed to show the user
    what a fragment is *part of*.

    Returns None — and the caller falls back to the heuristic — unless the fast
    tokenizer reproduces TL's exact id sequence. `model.to_tokens` stays the
    single source of truth for what the model sees; this only borrows offsets,
    and only when the two paths provably agree.
    """
    if not getattr(tokenizer, "is_fast", False):
        return None
    try:
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    except Exception:  # noqa: BLE001
        return None
    hf_ids = list(enc["input_ids"])
    spans = [(int(a), int(b)) for a, b in enc["offset_mapping"]]

    if ids == hf_ids:
        return spans
    if len(ids) == len(hf_ids) + 1 and ids[1:] == hf_ids:
        return [(0, 0), *spans]  # TL prepended BOS
    return None


def _search_offsets(tokenizer, text: str, ids: list[int]) -> list[tuple[int, int]]:
    """Fallback for tokenizers with no offset mapping: decode each token and
    search for it from a moving cursor. A token whose decoded form doesn't
    literally appear in `text` (BOS, or a byte fragment) gets a zero-width span
    rather than failing."""
    spans: list[tuple[int, int]] = []
    cursor = 0
    for tid in ids:
        tok_str = tokenizer.decode([tid])
        idx = text.find(tok_str, cursor) if tok_str else -1
        if idx == -1:
            spans.append((cursor, cursor))
        else:
            spans.append((idx, idx + len(tok_str)))
            cursor = idx + len(tok_str)
    return spans


@dataclasses.dataclass
class TokenRecord:
    id: int
    str: str
    display: str
    start: int
    end: int
    is_byte_fallback: bool
    # --- fragmentation (see docs/03-decisions.md D12) ---
    #
    # A "cluster" is a run of consecutive tokens that together cover one
    # indivisible piece of source text. For English it is almost always a
    # single token. For a script the tokenizer has no merges for, one character
    # routinely spans 2-3 tokens, and a cluster is how the UI can say so
    # instead of printing a row of replacement characters.
    cluster: int = 0
    cluster_size: int = 1
    cluster_index: int = 0
    cluster_text: str = ""
    byte_hex: str | None = None

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def _build_records(tokenizer, text: str, ids: list[int], spans: list[tuple[int, int]]) -> list[TokenRecord]:
    # Group consecutive tokens whose spans overlap. Two tokens reporting the
    # same (or an overlapping) character span are fragments of that character.
    clusters: list[list[int]] = []
    current: list[int] = []
    current_end = -1
    for i, (start, end) in enumerate(spans):
        if current and start < current_end:
            current.append(i)
            current_end = max(current_end, end)
        else:
            if current:
                clusters.append(current)
            current = [i]
            current_end = end
    if current:
        clusters.append(current)

    records: list[TokenRecord] = []
    for cid, members in enumerate(clusters):
        lo = min(spans[i][0] for i in members)
        hi = max(spans[i][1] for i in members)
        cluster_text = visible_whitespace(text[lo:hi])
        for pos, i in enumerate(members):
            tid = ids[i]
            tok_str = tokenizer.decode([tid])
            fragmented = len(members) > 1
            # A fragment shows what it is part of (first one) or a
            # continuation mark, never U+FFFD.
            if fragmented:
                display = cluster_text if pos == 0 else CONTINUATION
            else:
                display = visible_whitespace(tok_str)
            records.append(
                TokenRecord(
                    id=tid,
                    str=tok_str,
                    display=display,
                    start=spans[i][0],
                    end=spans[i][1],
                    # U+FFFD from decode() means this token is an incomplete
                    # multi-byte sequence; being in a multi-token cluster means
                    # the same thing observed from the offsets side. Either is
                    # enough to call it a fragment.
                    is_byte_fallback=("\ufffd" in tok_str) or fragmented,
                    cluster=cid,
                    cluster_size=len(members),
                    cluster_index=pos,
                    cluster_text=cluster_text,
                    byte_hex=_token_byte_hex(tokenizer, tid),
                )
            )
    return records


def tokenize_with_offsets(
    model: HookedTransformer, text: str, *, prepend_bos: bool | None = None
) -> tuple[list[TokenRecord], torch.Tensor]:
    """Tokenizes via model.to_tokens — the SAME call /run uses to build its
    input tensor — so the token list returned here (for the UI's hover-linking)
    is always in exact 1:1 correspondence with the sequence the model actually
    sees.

    Character offsets come from the fast tokenizer's offset mapping when it
    reproduces TL's id sequence exactly, and from a cursor search otherwise.
    See _hf_offsets for why that guard matters.
    """
    tokens = model.to_tokens(text, prepend_bos=prepend_bos)
    ids = tokens[0].tolist()
    spans = _hf_offsets(model.tokenizer, text, ids) or _search_offsets(model.tokenizer, text, ids)
    return _build_records(model.tokenizer, text, ids, spans), tokens


def token_records_from_ids(model: HookedTransformer, ids: list[int]) -> list[TokenRecord]:
    """For synthetic sequences with no source text to compute offsets against
    (Stage 2's repeated-token induction probe) — every record gets a zero-width
    span, since there is no original string to point a hover-highlight into."""
    records = []
    for i, tid in enumerate(ids):
        tok_str = model.tokenizer.decode([tid])
        records.append(
            TokenRecord(
                id=tid,
                str=tok_str,
                display=visible_whitespace(tok_str),
                start=0,
                end=0,
                is_byte_fallback="\ufffd" in tok_str,
                cluster=i,
                cluster_size=1,
                cluster_index=0,
                cluster_text=visible_whitespace(tok_str),
                byte_hex=_token_byte_hex(model.tokenizer, tid),
            )
        )
    return records


@dataclasses.dataclass
class ForwardResult:
    patterns: np.ndarray  # (n_layers, n_heads, seq, seq) float32, in [0,1]
    loss_per_token: list[float]  # length seq-1 (no target for the last position)
    top_logits: list[list[dict]]  # length seq, each a list of top_k {id,str,logit,prob}


def run_forward(model: HookedTransformer, tokens: torch.Tensor, *, top_k: int = 5) -> ForwardResult:
    """One forward pass, extracting exactly what /run needs: attention
    patterns (via names_filter, NOT the full cache — this is the
    memory-saving path Stage 0a measured at 67-95%), per-token loss, and
    top-k next-token predictions at every position."""
    pattern_filter = lambda name: name.endswith("hook_pattern")  # noqa: E731
    with torch.no_grad():
        output, cache = model.run_with_cache(
            tokens, names_filter=pattern_filter, return_type="both", loss_per_token=True
        )
    logits, loss = output

    n_layers = model.cfg.n_layers
    patterns = (
        torch.stack([cache[f"blocks.{i}.attn.hook_pattern"][0] for i in range(n_layers)])
        .to(torch.float32)
        .cpu()
        .numpy()
    )
    loss_per_token = loss[0].to(torch.float32).cpu().tolist()

    seq = tokens.shape[1]
    probs = torch.softmax(logits[0], dim=-1)  # (seq, d_vocab)
    k = min(top_k, probs.shape[-1])
    topk_probs, topk_ids = probs.topk(k, dim=-1)
    top_logits: list[list[dict]] = []
    for pos in range(seq):
        row = []
        for j in range(k):
            tid = int(topk_ids[pos, j].item())
            row.append(
                {
                    "id": tid,
                    "str": model.tokenizer.decode([tid]),
                    "logit": float(logits[0, pos, tid].item()),
                    "prob": float(topk_probs[pos, j].item()),
                }
            )
        top_logits.append(row)

    return ForwardResult(patterns=patterns, loss_per_token=loss_per_token, top_logits=top_logits)


def make_repeated_tokens(
    model: HookedTransformer, *, length: int, seed: int, prepend_bos: bool = True
) -> torch.Tensor:
    """The canonical induction-head probe (ARENA 1.2): a random sequence
    followed by an exact repeat of itself. Seeded so a permalink
    reproduces the identical sequence — see docs/PLAN.md Stage 2."""
    rng = np.random.default_rng(seed)
    # Avoid the tokenizer's special tokens (typically at the low or high
    # end of the vocab) to keep every sampled id an ordinary content
    # token; a comfortable interior slice of the vocab is enough.
    d_vocab = model.cfg.d_vocab
    lo, hi = int(d_vocab * 0.05), int(d_vocab * 0.95)
    half = rng.integers(lo, hi, size=length).tolist()
    ids = half + half
    if prepend_bos:
        bos_id = model.tokenizer.bos_token_id
        if bos_id is None:
            bos_id = model.tokenizer.eos_token_id
        ids = [bos_id] + ids
    return torch.tensor([ids], dtype=torch.long)
