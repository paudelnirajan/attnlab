"""
Logit lens engine (docs/03-decisions.md D15, docs/02-api.md § Logit lens).

The residual stream is a running sum. Every block reads it and adds its output
back, and only the last value is ever turned into a prediction:

    x = embed + pos                       resid_pre 0
    x = x + attn_0(x)                     resid_mid 0
    x = x + mlp_0(x)                      resid_post 0
    ...
    logits = W_U · ln_final(x) + b_U      the model's prediction

The logit lens applies that last step to every intermediate x, so each layer's
running sum can be read as a prediction. Everything here is computed from one
forward pass whose residual stream is kept for ~10 minutes, so the views that
follow (a position's trajectory, direct logit attribution) are only matrix
products against it.

Two lenses, because the choice of normalization is the lesson:

  ln_final  the model's own final LayerNorm, learned weights and all. This is
            exactly the path the last layer takes, applied to earlier layers.
  plain     mean 0 / std 1 with no learned weights, then the raw unembedding.
            It assumes nothing about earlier layers, and a few huge residual
            dimensions (which ln_final has learned to mute) drown it out.

The zoo loads models with TransformerLens's default processing (LayerNorm
folded into W_U, W_U centred), which removes the learned ln_final weights the
plain lens needs. `LensReadyTransformer` records the few raw quantities it
needs while the weights are being processed, so no second copy of the model is
kept. See `RawUnembed`.
"""

from __future__ import annotations

import dataclasses
import functools
import unicodedata
from typing import Literal

import torch
from transformer_lens import HookedTransformer

from attnlab.inference import _token_byte_hex, visible_whitespace

LensKind = Literal["ln_final", "plain"]
LENSES: tuple[LensKind, ...] = ("ln_final", "plain")

MAX_LENS_SEQ = 256  # the grid is one column per token; past this it stops being readable
TOP_K = 5
MAX_POSITION_K = 25
MAX_TRACKED = 6
# Per-head top tokens cost one [d_model, d_vocab] product per head; decode every
# head for models up to this many, else only the heads that matter most.
MAX_DECODED_HEADS = 48

FRAG = "⋯"  # where a character is cut, the same mark the other labs use (D12)


# ---------------------------------------------------------------------------
# Capturing the raw final LayerNorm while the weights are processed
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class RawUnembed:
    """What the plain lens needs from the unprocessed weights, and facts about
    the output path the "Under the hood" view reports.

    With LayerNorm folded, TransformerLens stores W_U' = w ⊙ W_U - m - k, where
    w is ln_final's learned scale, m is a per-vocab constant (centring over
    d_model) and k a per-dimension constant (centring over vocab). So the raw
    unembedding is recoverable from W_U' plus m alone:

        n · W_U = (n / w) · W_U' + (Σ n/w) · m + (a constant per position)

    and a constant per position changes no probability and no rank. `m` is one
    number per vocab entry, so this costs 200 KB instead of a second W_U.
    """

    norm: str | None  # "LN", "RMS" or None, as the model was trained
    folded: bool  # were w and b folded into W_U?
    w: torch.Tensor | None  # [d_model] raw ln_final.w
    b: torch.Tensor | None  # [d_model] raw ln_final.b
    b_U: torch.Tensor  # [d_vocab] raw unembedding bias
    m: torch.Tensor | None  # [d_vocab] the centring term above; None if not folded
    fold_error: float | None  # max |w ⊙ W_U - W_U' - (m + k)|: 0 up to float error if the algebra holds
    tied: bool | None  # W_U == W_E.T in the released weights
    bias_prior: list[int]  # token ids ln_final.b alone pushes up the most


class LensReadyTransformer(HookedTransformer):
    """A HookedTransformer that keeps `RawUnembed` from its unprocessed state
    dict. `from_pretrained` builds the model through `cls(...)` and then calls
    `load_and_process_state_dict` with the raw weights, so overriding that one
    method sees them without changing how the model is loaded or processed."""

    raw_unembed: RawUnembed | None = None

    def load_and_process_state_dict(self, state_dict, fold_ln=True, *args, **kwargs):  # type: ignore[override]
        raw = _raw_facts(state_dict, fold_ln)
        super().load_and_process_state_dict(state_dict, fold_ln, *args, **kwargs)
        if raw is not None and raw["folded"]:
            # C[d, v] = w_d W_U[d, v] - W_U'[d, v] should be m_v + k_d exactly.
            W_raw = raw.pop("W_U")
            C = W_raw * raw["w"][:, None] - self.W_U.detach().to(W_raw.device, W_raw.dtype)
            m = C.mean(dim=0)
            k = (C - m).mean(dim=1, keepdim=True)
            raw["m"] = m
            raw["fold_error"] = float((C - m - k).abs().max())
            del W_raw, C
        elif raw is not None:
            raw.pop("W_U")
        self.raw_unembed = RawUnembed(**raw) if raw is not None else None


def _raw_facts(state_dict: dict, fold_ln: bool) -> dict | None:
    if "unembed.W_U" not in state_dict:
        return None
    W_U = state_dict["unembed.W_U"].detach().float().clone()
    b_U = state_dict.get("unembed.b_U")
    b_U = b_U.detach().float().clone() if b_U is not None else torch.zeros(W_U.shape[1])
    w = state_dict.get("ln_final.w")
    b = state_dict.get("ln_final.b")
    norm = None if w is None else ("LN" if b is not None else "RMS")
    w = w.detach().float().clone() if w is not None else None
    b = b.detach().float().clone() if b is not None else None
    W_E = state_dict.get("embed.W_E")
    tied = bool(torch.equal(W_U, W_E.detach().float().T)) if W_E is not None else None
    prior = (b @ W_U + b_U) if b is not None else b_U
    return {
        "norm": norm,
        "folded": bool(fold_ln and w is not None),
        "w": w,
        "b": b,
        "b_U": b_U,
        "m": None,
        "fold_error": None,
        "tied": tied,
        "bias_prior": prior.topk(8).indices.tolist(),
        "W_U": W_U,
    }


# ---------------------------------------------------------------------------
# Honest labels for byte fragments (the notebook's label_inputs/label_prediction)
# ---------------------------------------------------------------------------


def _utf8_len(lead: int) -> int:
    return 1 if lead < 0x80 else 2 if lead < 0xE0 else 3 if lead < 0xF0 else 4


@functools.lru_cache(maxsize=4096)
def start_range(frag: bytes) -> str:
    """First–last readable character whose UTF-8 encoding starts with `frag`.

    A predicted `e0 a4` has not chosen a letter yet: it means "some character
    from ऄ to ऽ". Show the range, never one letter. Combining marks are skipped
    as endpoints because on their own they render as a dotted circle."""
    n = _utf8_len(frag[0])
    lo, hi = {2: (0x80, 0x800), 3: (0x800, 0x10000), 4: (0x10000, 0x110000)}.get(n, (0, 0))
    first = last = None
    for cp in range(lo, hi):
        if 0xD800 <= cp <= 0xDFFF:
            continue
        ch = chr(cp)
        if unicodedata.category(ch)[0] in "LNPS" and ch.encode("utf-8").startswith(frag):
            first = first or ch
            last = ch
    return f"{first}–{last}" if first else frag.hex(" ")


def token_bytes(tokenizer, tid: int) -> bytes | None:
    """The raw bytes a token stands for, or None for a special token or a
    tokenizer family whose pieces aren't byte-level."""
    hx = _token_byte_hex(tokenizer, tid)
    return bytes.fromhex(hx) if hx is not None else None


def _dangling(b: bytes) -> bytes:
    """The bytes at the end of `b` that don't yet form a complete character."""
    for k in range(min(4, len(b) + 1)):
        try:
            b[: len(b) - k].decode("utf-8")
            return b[len(b) - k :]
        except UnicodeDecodeError:
            pass
    return b""


def _decoded(tokenizer, tid: int) -> str:
    if tid >= len(tokenizer):
        return f"⟨unused {tid}⟩"  # Pythia pads d_vocab past the tokenizer's vocabulary
    return visible_whitespace(tokenizer.decode([tid]))


def label_inputs(tokenizer, ids: list[int]) -> list[str]:
    """One label per input token. ⋯ marks where a character is cut: न⋯ then ⋯न."""
    pieces = [token_bytes(tokenizer, i) for i in ids]
    labels: list[str] = []
    stream = b"".join(p for p in pieces if p is not None)
    text = stream.decode("utf-8", errors="replace")
    spans, pos = [], 0
    for ch in text:
        n = len(ch.encode("utf-8")) if ch != "�" else 1
        spans.append((pos, pos + n, ch))
        pos += n
    start = 0
    for tid, piece in zip(ids, pieces):
        if piece is None:
            labels.append(_decoded(tokenizer, tid))
            continue
        end = start + len(piece)
        label = ""
        for s, e, ch in spans:
            if s < end and e > start:
                label += (FRAG if s < start else "") + ch + (FRAG if e > end else "")
        labels.append(visible_whitespace(label) if label else _decoded(tokenizer, tid))
        start = end
    return labels


def dangling_per_position(tokenizer, ids: list[int]) -> list[bytes]:
    """For each position, the unfinished character the input so far ends in.
    The same predicted token reads differently after `e0 a4` than after a space."""
    out, stream = [], b""
    for tid in ids:
        b = token_bytes(tokenizer, tid)
        stream = (stream + b)[-8:] if b is not None else b""
        out.append(_dangling(stream))
    return out


def label_prediction(tokenizer, dangling: bytes, tid: int) -> str:
    b = token_bytes(tokenizer, tid)
    if b is None or tid >= len(tokenizer):
        return _decoded(tokenizer, tid)
    try:
        return visible_whitespace(b.decode("utf-8"))  # complete by itself
    except UnicodeDecodeError:
        pass
    if dangling:
        try:
            return FRAG + visible_whitespace((dangling + b).decode("utf-8"))  # finishes the input's character
        except UnicodeDecodeError:
            pass
    starts = [k for k, x in enumerate(b) if x >= 0xC0]
    if starts:
        i = starts[-1]
        try:
            return visible_whitespace(b[:i].decode("utf-8")) + start_range(b[i:]) + FRAG  # starts a new one
        except UnicodeDecodeError:
            pass
    return b.hex(" ")  # an orphan continuation byte: no character to name


class LabelTable:
    """Deduplicated (token id, label) pairs. Cells refer to entries by index,
    so a 50-row grid doesn't repeat ' the' a thousand times."""

    def __init__(self, tokenizer, dangling: list[bytes]):
        self.tokenizer = tokenizer
        self.dangling = dangling
        self.entries: list[dict] = []
        self._index: dict[tuple[int, bytes], int] = {}

    def ref(self, pos: int, tid: int) -> int:
        d = self.dangling[pos]
        key = (tid, d)
        if key not in self._index:
            self._index[key] = len(self.entries)
            self.entries.append({"id": tid, "label": label_prediction(self.tokenizer, d, tid)})
        return self._index[key]

    def label(self, pos: int, tid: int) -> str:
        return self.entries[self.ref(pos, tid)]["label"]


# ---------------------------------------------------------------------------
# One forward pass, kept
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class Row:
    """One readable point in the residual stream."""

    id: str  # the TransformerLens hook it comes from, e.g. "blocks.3.hook_resid_mid"
    label: str
    layer: int  # -1 for the embedding
    kind: Literal["embed", "attn", "mlp"]  # what was added last
    block_end: bool  # the stream between two blocks (what the notebook plots)
    real: bool  # False for a parallel block's "after attention", which the model never forms

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class LensRun:
    ids: list[int]
    rows: list[Row]
    resid: torch.Tensor  # [n_rows, seq, d_model], CPU
    embed: torch.Tensor  # [seq, d_model]
    pos_embed: torch.Tensor | None  # [seq, d_model], only when it is added to the stream
    attn_out: torch.Tensor  # [n_layers, seq, d_model]
    mlp_out: torch.Tensor | None  # [n_layers, seq, d_model]
    z: torch.Tensor  # [n_layers, seq, n_heads, d_head]
    checks: list[dict]

    @property
    def seq(self) -> int:
        return len(self.ids)

    @property
    def nbytes(self) -> int:
        ts = [self.resid, self.embed, self.attn_out, self.z, self.pos_embed, self.mlp_out]
        return sum(t.numel() * t.element_size() for t in ts if t is not None)


def _check(id: str, label: str, ok: bool, value: float | None = None, detail: str = "") -> dict:
    return {"id": id, "label": label, "ok": bool(ok), "value": value, "detail": detail}


def run_lens(model: HookedTransformer, tokens: torch.Tensor) -> LensRun:
    cfg = model.cfg
    L = cfg.n_layers
    has_mlp = not cfg.attn_only
    parallel = bool(getattr(cfg, "parallel_attn_mlp", False))

    def keep(name: str) -> bool:
        return name in ("hook_embed", "hook_pos_embed", "blocks.0.hook_resid_pre") or name.endswith(
            ("hook_resid_mid", "hook_resid_post", "hook_attn_out", "hook_mlp_out", "attn.hook_z")
        )

    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens, names_filter=keep)

    def get(name: str) -> torch.Tensor:
        return cache[name][0].detach().float().cpu()

    rows: list[Row] = [Row("blocks.0.hook_resid_pre", "embed", -1, "embed", True, True)]
    vecs = [get("blocks.0.hook_resid_pre")]
    attn_out = torch.stack([get(f"blocks.{li}.hook_attn_out") for li in range(L)])
    mlp_out = torch.stack([get(f"blocks.{li}.hook_mlp_out") for li in range(L)]) if has_mlp else None
    for li in range(L):
        post = get(f"blocks.{li}.hook_resid_post")
        if has_mlp:
            if parallel:
                vecs.append(_pre(vecs, rows) + attn_out[li])
                rows.append(Row(f"blocks.{li}.hook_resid_pre + attn_out", f"L{li} +attn", li, "attn", False, False))
            else:
                vecs.append(get(f"blocks.{li}.hook_resid_mid"))
                rows.append(Row(f"blocks.{li}.hook_resid_mid", f"L{li} +attn", li, "attn", False, True))
            vecs.append(post)
            rows.append(Row(f"blocks.{li}.hook_resid_post", f"L{li} +mlp", li, "mlp", True, True))
        else:
            vecs.append(post)
            rows.append(Row(f"blocks.{li}.hook_resid_post", f"L{li} +attn", li, "attn", True, True))

    embed = get("hook_embed")
    pos_embed = get("hook_pos_embed") if cfg.positional_embedding_type == "standard" else None
    z = torch.stack([get(f"blocks.{li}.attn.hook_z") for li in range(L)])
    resid = torch.stack(vecs)

    run = LensRun(
        ids=tokens[0].tolist(),
        rows=rows,
        resid=resid,
        embed=embed,
        pos_embed=pos_embed,
        attn_out=attn_out,
        mlp_out=mlp_out,
        z=z,
        checks=[],
    )
    # The real logits are only kept long enough to prove output_logits()
    # reproduces them; storing them would cost seq × d_vocab floats per run
    # (51 MB at 256 tokens), more than the whole residual stream.
    run.checks = _anchor_checks(model, run, logits[0].detach().float().cpu())
    return run


def output_logits(model: HookedTransformer, run: LensRun) -> torch.Tensor:
    """The model's output, [seq, d_vocab] on the model's device: its own
    unembedding of the last residual row. The first anchor check measured
    this against the forward pass's logits for this run."""
    with torch.no_grad():
        return lens_logits(model, run.resid[-1].to(model.W_U.device), "ln_final")


def _pre(vecs: list[torch.Tensor], rows: list[Row]) -> torch.Tensor:
    """The stream entering the current block: the last block-end row."""
    for v, r in zip(reversed(vecs), reversed(rows)):
        if r.block_end:
            return v
    raise AssertionError("no block-end row")


def _anchor_checks(model: HookedTransformer, run: LensRun, real_logits: torch.Tensor) -> list[dict]:
    """Never trust a cache you haven't checked. Each of these is a claim the
    views depend on, recomputed for this run."""
    checks: list[dict] = []
    dev = model.W_U.device

    # 1. The ln_final lens on the last row is the model's own output.
    with torch.no_grad():
        lens_last = lens_logits(model, run.resid[-1].to(dev), "ln_final").cpu()
    err = float((lens_last - real_logits).abs().max())
    checks.append(
        _check(
            "lens_reproduces_output",
            "ln_final lens on the last row = the model's logits",
            err < 1e-2,
            err,
            "max |W_U·ln_final(resid_post[-1]) + b_U − logits| over every position and vocab entry",
        )
    )

    # 2. The stream is a sum: embeddings + every attention and MLP output.
    total = run.embed.clone()
    if run.pos_embed is not None:
        total += run.pos_embed
    total += run.attn_out.sum(0)
    if run.mlp_out is not None:
        total += run.mlp_out.sum(0)
    last = run.resid[-1]
    err = float((total - last).abs().max() / last.abs().max())
    checks.append(
        _check(
            "stream_is_a_sum",
            "embed + pos + Σ attn_out + Σ mlp_out = final residual",
            err < 1e-4,
            err,
            "max error relative to the largest residual entry — this is what makes direct logit attribution exact",
        )
    )

    # 3. Heads add up to the attention output (plus its bias).
    L = model.cfg.n_layers
    with torch.no_grad():
        heads = torch.einsum("lphd,lhdm->lpm", run.z.to(dev), model.W_O) + model.b_O[:, None, :]
    err = float((heads.cpu() - run.attn_out).abs().max() / run.attn_out.abs().max().clamp(min=1e-6))
    checks.append(
        _check(
            "heads_sum_to_attn_out",
            "Σ_h z_h · W_O[h] + b_O = attn_out, in all layers",
            err < 1e-4,
            err,
            f"per-head outputs across {L} layers, relative error",
        )
    )

    # 4. Layer 0's input is the embedding (plus positions, when they are added).
    x0 = run.resid[0]
    manual = run.embed + (run.pos_embed if run.pos_embed is not None else 0)
    err = float((x0 - manual).abs().max())
    label = "resid_pre 0 = W_E[tokens] + W_pos" if run.pos_embed is not None else "resid_pre 0 = W_E[tokens]"
    checks.append(_check("embed_is_row_0", label, err < 1e-4, err, "the first lens row reads the embedding alone"))

    raw = getattr(model, "raw_unembed", None)
    if raw is not None and raw.folded and raw.fold_error is not None:
        scale = float(model.W_U.detach().abs().max())
        checks.append(
            _check(
                "plain_lens_recoverable",
                "raw W_U recovered from the folded one (w ⊙ W_U − W_U′ = m + k)",
                raw.fold_error / scale < 1e-4,
                raw.fold_error / scale,
                "relative residual of the folding algebra, measured once when the model loaded",
            )
        )
    return checks


# ---------------------------------------------------------------------------
# Lenses
# ---------------------------------------------------------------------------


def available_lenses(model: HookedTransformer) -> list[LensKind]:
    raw = getattr(model, "raw_unembed", None)
    if raw is None or raw.norm is None:
        return ["ln_final"]  # no LayerNorm at the output: the two lenses would be the same thing
    return ["ln_final", "plain"]


def _plain_norm(x: torch.Tensor, norm: str, eps: float) -> torch.Tensor:
    if norm == "RMS":
        return x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()
    x = x - x.mean(-1, keepdim=True)
    return x / (x.pow(2).mean(-1, keepdim=True) + eps).sqrt()


def lens_logits(model: HookedTransformer, x: torch.Tensor, lens: LensKind) -> torch.Tensor:
    """[..., d_model] → [..., d_vocab]."""
    if lens == "ln_final":
        # A model with no output norm (attn-only-2l-demo) reads the stream directly.
        return model.unembed(model.ln_final(x) if model.cfg.normalization_type is not None else x)
    raw: RawUnembed | None = getattr(model, "raw_unembed", None)
    if raw is None or raw.norm is None:
        raise ValueError("this model has no output normalization, so there is no plain lens")
    dev = x.device
    n = _plain_norm(x, raw.norm, model.cfg.eps)
    b_U = raw.b_U.to(dev)
    if not raw.folded:
        return n @ model.W_U + b_U
    assert raw.w is not None and raw.m is not None  # folded implies both were captured
    nw = n / raw.w.to(dev)
    return nw @ model.W_U + nw.sum(-1, keepdim=True) * raw.m.to(dev) + b_U


# ---------------------------------------------------------------------------
# The grid: one summary per (row, position)
# ---------------------------------------------------------------------------


def _sig(xs: list, digits: int = 4) -> list:
    """Round for the wire. Four significant digits is past what any view shows."""
    return [None if x is None else float(f"{x:.{digits}g}") for x in xs]


def _row_stats(
    logits: torch.Tensor, final_logp: torch.Tensor, final_top: torch.Tensor, next_ids: torch.Tensor, k: int
) -> dict[str, torch.Tensor]:
    logp = logits.log_softmax(-1)
    p = logp.exp()
    top_p, top_i = p.topk(k, dim=-1)
    ent = -(p * logp).sum(-1)
    kl = (final_logp.exp() * (final_logp - logp)).sum(-1)
    idx = next_ids.clamp(min=0)[:, None]
    t_logit = logits.gather(-1, idx)
    f_logit = logits.gather(-1, final_top[:, None])
    return {
        "top_p": top_p,
        "top_i": top_i,
        "entropy": ent,
        "kl": kl,
        "p_next": p.gather(-1, idx)[:, 0],
        "rank_next": (logits > t_logit).sum(-1) + 1,
        "p_final": p.gather(-1, final_top[:, None])[:, 0],
        "rank_final": (logits > f_logit).sum(-1) + 1,
    }


def summarize(model: HookedTransformer, run: LensRun, lens: LensKind, tokenizer) -> dict:
    """Everything the grid shows, for every row plus the model's output."""
    dev = model.W_U.device
    seq = run.seq
    next_ids = torch.tensor(run.ids[1:] + [-1], device=dev)
    has_next = [i + 1 < seq for i in range(seq)]
    final = output_logits(model, run)
    final_logp = final.log_softmax(-1)
    final_top = final.argmax(-1)
    table = LabelTable(tokenizer, dangling_per_position(tokenizer, run.ids))

    out: dict[str, list] = {k: [] for k in ("top", "top_p", "p_next", "rank_next", "p_final", "rank_final", "entropy", "kl", "norm")}
    with torch.no_grad():
        for r in range(len(run.rows) + 1):
            if r < len(run.rows):
                x = run.resid[r].to(dev)
                lg = lens_logits(model, x, lens)
                norm = x.norm(dim=-1).tolist()
            else:
                lg = final  # the output row is the model itself, not a lens
                norm = [None] * seq
            s = _row_stats(lg, final_logp, final_top, next_ids, TOP_K)
            top_i = s["top_i"].tolist()
            out["top"].append([[table.ref(pos, t) for t in top_i[pos]] for pos in range(seq)])
            out["top_p"].append([_sig(row) for row in s["top_p"].tolist()])
            for key in ("p_next", "p_final", "entropy", "kl"):
                vals = s[key].tolist()
                if key == "p_next":
                    vals = [v if has_next[i] else None for i, v in enumerate(vals)]
                out[key].append(_sig(vals))
            rank_next = s["rank_next"].tolist()
            out["rank_next"].append([v if has_next[i] else None for i, v in enumerate(rank_next)])
            out["rank_final"].append(s["rank_final"].tolist())
            out["norm"].append(_sig(norm))

    return {
        "rows": [r.to_dict() for r in run.rows]
        + [{"id": "logits", "label": "output", "layer": model.cfg.n_layers, "kind": "output", "block_end": True, "real": True}],
        "input_labels": label_inputs(tokenizer, run.ids),
        "next_labels": [table.entries[table.ref(i, run.ids[i + 1])]["label"] if i + 1 < seq else None for i in range(seq)],
        "final_top": [table.ref(i, t) for i, t in enumerate(final_top.tolist())],
        "strings": table.entries,
        "cells": out,
    }


def layer_curves(model: HookedTransformer, run: LensRun, tokenizer) -> dict:
    """Per-row averages over positions, for every lens: how close each layer is
    to the final answer, and to the truth."""
    dev = model.W_U.device
    seq = run.seq
    final = output_logits(model, run)
    final_logp = final.log_softmax(-1)
    final_top = final.argmax(-1)
    next_ids = torch.tensor(run.ids[1:] + [-1], device=dev)
    known = torch.arange(seq, device=dev) < seq - 1
    curves: dict[str, dict[str, list]] = {}
    with torch.no_grad():
        for lens in available_lenses(model):
            c = {k: [] for k in ("agree_final", "agree_next", "ce_next", "kl_final", "entropy", "p_next")}
            for r in range(len(run.rows) + 1):
                lg = lens_logits(model, run.resid[r].to(dev), lens) if r < len(run.rows) else final
                s = _row_stats(lg, final_logp, final_top, next_ids, 1)
                top1 = s["top_i"][:, 0]
                c["agree_final"].append(float((top1 == final_top).float().mean()))
                c["kl_final"].append(float(s["kl"].mean()))
                c["entropy"].append(float(s["entropy"].mean()))
                if known.any():
                    c["agree_next"].append(float((top1 == next_ids)[known].float().mean()))
                    c["ce_next"].append(float((-s["p_next"][known].clamp(min=1e-30).log()).mean()))
                    c["p_next"].append(float(s["p_next"][known].mean()))
                else:
                    for key in ("agree_next", "ce_next", "p_next"):
                        c[key].append(None)
            curves[lens] = {k: _sig(v) for k, v in c.items()}
    norms = run.resid.norm(dim=-1)  # [rows, seq]
    return {
        "lenses": curves,
        "norm_mean": _sig(norms.mean(-1).tolist()),
        "norm_max": _sig(norms.max(-1).values.tolist()),
        # position 0 (often BOS) is an attention sink with an outsized norm; reported apart
        "norm_mean_excl_first": _sig(norms[:, 1:].mean(-1).tolist()) if seq > 1 else None,
    }


# ---------------------------------------------------------------------------
# One position: full top-k per row, and tokens the learner chose to follow
# ---------------------------------------------------------------------------


def resolve_track(model: HookedTransformer, s: str) -> dict:
    """A string to track becomes its first token. Say so when it was more than
    one: ' Kathmandu' is several tokens and only the first is ever predicted next."""
    ids = model.to_tokens(s, prepend_bos=False)[0].tolist()
    if not ids:
        raise ValueError("empty string")
    note = ""
    if len(ids) > 1:
        note = f"{s!r} is {len(ids)} tokens; tracking the first, {model.tokenizer.decode([ids[0]])!r}"
    return {"id": ids[0], "note": note}


def position_detail(
    model: HookedTransformer, run: LensRun, tokenizer, pos: int, lens: LensKind, k: int, track_ids: list[int]
) -> dict:
    dev = model.W_U.device
    table = LabelTable(tokenizer, dangling_per_position(tokenizer, run.ids))
    with torch.no_grad():
        x = run.resid[:, pos].to(dev)  # [rows, d_model]
        lg = torch.cat([lens_logits(model, x, lens), output_logits(model, run)[pos][None]])  # [rows+1, vocab]
        logp = lg.log_softmax(-1)
        top_lp, top_i = logp.topk(k, dim=-1)
        tracked = []
        for tid in track_ids:
            t_logit = lg[:, tid]
            tracked.append(
                {
                    "id": tid,
                    "label": table.label(pos, tid),
                    "logit": _sig(t_logit.tolist()),
                    "prob": _sig(logp[:, tid].exp().tolist()),
                    "rank": ((lg > t_logit[:, None]).sum(-1) + 1).tolist(),
                }
            )
    top_i_l, top_p_l = top_i.tolist(), top_lp.exp().tolist()
    return {
        "pos": pos,
        "lens": lens,
        "top": [
            [{"id": t, "label": table.label(pos, t), "prob": float(f"{p:.4g}")} for t, p in zip(ids_, ps)]
            for ids_, ps in zip(top_i_l, top_p_l)
        ],
        "tracked": tracked,
    }


# ---------------------------------------------------------------------------
# Direct logit attribution
# ---------------------------------------------------------------------------


def attribution(
    model: HookedTransformer, run: LensRun, tokenizer, pos: int, target: int, contrast: int | None
) -> dict:
    """Split one output logit (or a logit difference) into the part each
    component wrote.

    The final logit is linear in the residual once ln_final's scale is fixed at
    its actual value for this position:

        logit_t = Σ_c ((c − mean c) / scale) · w · W_U[:, t]  +  (b · W_U[:, t] + b_U[t])

    so this decomposition is exact, not an approximation. What it cannot say is
    what would happen if a component were removed: the scale would change, and
    later components read earlier ones. That is ablation's job (step 5).
    """
    cfg = model.cfg
    dev = model.W_U.device
    table = LabelTable(tokenizer, dangling_per_position(tokenizer, run.ids))
    L, H = cfg.n_layers, cfg.n_heads

    with torch.no_grad():
        final_x = run.resid[-1, pos].to(dev)
        normed = model.cfg.normalization_type is not None
        if normed:
            centred = cfg.normalization_type in ("LN", "LNPre")
            xf = final_x - final_x.mean() if centred else final_x
            scale = (xf.pow(2).mean() + cfg.eps).sqrt()
        else:
            centred, scale = False, torch.tensor(1.0, device=dev)
        # LN not folded (shortformer models): the learned w, b are still in ln_final
        ln_w = getattr(model.ln_final, "w", None) if normed else None
        ln_b = getattr(model.ln_final, "b", None) if normed else None

        direction = model.W_U[:, target] - (model.W_U[:, contrast] if contrast is not None else 0)
        bias = model.b_U[target] - (model.b_U[contrast] if contrast is not None else 0)
        if ln_b is not None:
            bias = bias + ln_b @ direction

        def project(v: torch.Tensor) -> torch.Tensor:
            """[n, d_model] → what each vector contributes along W_U, after the final LN."""
            if centred:
                v = v - v.mean(-1, keepdim=True)
            v = v / scale
            if ln_w is not None:
                v = v * ln_w
            return v

        comps: list[dict] = []
        vecs: list[torch.Tensor] = []

        def add(id_, label, kind, layer, head, v):
            comps.append({"id": id_, "label": label, "kind": kind, "layer": layer, "head": head})
            vecs.append(v)

        add("embed", "token embedding", "embed", -1, None, run.embed[pos].to(dev))
        if run.pos_embed is not None:
            add("pos", "position embedding", "pos", -1, None, run.pos_embed[pos].to(dev))
        head_out = torch.einsum("lhd,lhdm->lhm", run.z[:, pos].to(dev), model.W_O)  # [L, H, d_model]
        for li in range(L):
            for h in range(H):
                add(f"L{li}H{h}", f"L{li}H{h}", "head", li, h, head_out[li, h])
            add(f"L{li}.b_O", f"L{li} attn bias", "attn_bias", li, None, model.b_O[li])
            if run.mlp_out is not None:
                add(f"L{li}.mlp", f"L{li} MLP", "mlp", li, None, run.mlp_out[li, pos].to(dev))

        P = project(torch.stack(vecs))  # [n, d_model]
        values = (P @ direction).tolist()
        total = sum(values) + float(bias)
        out = output_logits(model, run)[pos]
        actual = float(out[target] - (out[contrast] if contrast is not None else 0))

        # What each component pushes up and down, over the whole vocabulary.
        head_ix = [i for i, c in enumerate(comps) if c["kind"] == "head"]
        decode = [i for i, c in enumerate(comps) if c["kind"] != "head"]
        if len(head_ix) <= MAX_DECODED_HEADS:
            decode += head_ix
        else:
            decode += sorted(head_ix, key=lambda i: -abs(values[i]))[:MAX_DECODED_HEADS]
        vocab_logits = P[decode] @ model.W_U  # [n_decode, d_vocab]
        up_v, up_i = vocab_logits.topk(5, dim=-1)
        dn_v, dn_i = (-vocab_logits).topk(5, dim=-1)

    for j, i in enumerate(decode):
        comps[i]["top_up"] = [{"label": table.label(pos, t), "value": float(f"{v:.4g}")} for t, v in zip(up_i[j].tolist(), up_v[j].tolist())]
        comps[i]["top_down"] = [
            {"label": table.label(pos, t), "value": float(f"{-v:.4g}")} for t, v in zip(dn_i[j].tolist(), dn_v[j].tolist())
        ]
    for c, v in zip(comps, values):
        c["value"] = float(f"{v:.5g}")

    return {
        "pos": pos,
        "target": {"id": target, "label": table.label(pos, target)},
        "contrast": {"id": contrast, "label": table.label(pos, contrast)} if contrast is not None else None,
        "components": comps,
        "bias": float(f"{float(bias):.5g}"),
        "total": float(f"{total:.6g}"),
        "actual": float(f"{actual:.6g}"),
        "error": abs(total - actual),
        "scale": float(scale),
        "decoded_heads": "all" if len(head_ix) <= MAX_DECODED_HEADS else MAX_DECODED_HEADS,
    }


# ---------------------------------------------------------------------------
# The output path, for "Under the hood"
# ---------------------------------------------------------------------------


def anatomy(model: HookedTransformer, run: LensRun, tokenizer) -> dict:
    cfg = model.cfg
    raw: RawUnembed | None = getattr(model, "raw_unembed", None)
    info: dict = {
        "d_model": cfg.d_model,
        "d_vocab": cfg.d_vocab,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "d_head": cfg.d_head,
        "n_rows": len(run.rows),
        "seq": run.seq,
        "attn_only": bool(cfg.attn_only),
        "parallel_attn_mlp": bool(getattr(cfg, "parallel_attn_mlp", False)),
        "positional": cfg.positional_embedding_type,
        "normalization": cfg.normalization_type,
        "raw": None,
        "outlier_dims": [],
    }
    if raw is None:
        return info
    info["raw"] = {
        "norm": raw.norm,
        "folded": raw.folded,
        "tied": raw.tied,
        "w_min": float(raw.w.min()) if raw.w is not None else None,
        "w_max": float(raw.w.max()) if raw.w is not None else None,
        "w_mean": float(raw.w.mean()) if raw.w is not None else None,
        "smallest_w": (
            [{"dim": int(d), "w": float(raw.w[d])} for d in raw.w.abs().argsort()[:6].tolist()] if raw.w is not None else []
        ),
        "bias_prior": [_decoded(tokenizer, t) for t in raw.bias_prior],
    }
    if raw.w is not None:
        # The dimensions that dominate the last residual row, and what ln_final
        # does to them. Plain normalization keeps them loud; ln_final's w mutes them.
        x = run.resid[-1]
        x = x - x.mean(-1, keepdim=True)
        share = x.pow(2).mean(0) / x.pow(2).mean(0).sum()  # share of the squared norm, averaged over positions
        top = share.argsort(descending=True)[:8].tolist()
        info["outlier_dims"] = [{"dim": d, "share": float(share[d]), "w": float(raw.w[d])} for d in top]
    return info
