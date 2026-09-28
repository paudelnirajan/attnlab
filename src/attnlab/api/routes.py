"""
The endpoints from docs/02-api.md, Stage 0b subset:
  GET  /models
  POST /tokenize
  POST /run
  GET  /run/{run_id}/patterns
  GET  /health
  GET  /version

Concurrency: every route that touches the zoo or runs a forward pass
goes through the model slot, `AppState.run_serialized` (docs/03-decisions.md D6 — one
model-touching operation at a time, server-wide) and runs the actual
blocking torch call in a thread executor so the event loop keeps serving
other connections (e.g. /health) while it runs.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import APIRouter, Query, Request, Response

from attnlab import memory
from attnlab.api.errors import InvalidRequestError, RunNotFoundError, SeqTooLongError, TextTooLongError
from attnlab.api.meta import REVISION, TL_VERSION, VERSION, build_meta
from attnlab.api.schemas import RunRequest, TokenizeRequest
from attnlab.api.state import AppState
from attnlab.inference import make_repeated_tokens, run_forward, token_records_from_ids, tokenize_with_offsets
from attnlab.instrument import measure
from attnlab.patterns import frame_layers
from attnlab.settings import SETTINGS

router = APIRouter()

def _state(request: Request) -> AppState:
    return request.app.state.attnlab


async def _serialized(request: Request, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Model work, through the single model slot (AppState.run_serialized)."""
    return await _state(request).run_serialized(fn, *args, request=request, **kwargs)


def check_text(text: str) -> None:
    """Bound the text before tokenizing: tokenizing is O(len) work done while
    holding the model slot, and max_seq only rejects it afterwards."""
    if len(text) > SETTINGS.max_text_chars:
        raise TextTooLongError(
            f"text is {len(text)} characters; the limit is {SETTINGS.max_text_chars}",
            {"max_chars": SETTINGS.max_text_chars, "got": len(text)},
        )


@router.get("/models")
async def list_models(request: Request) -> dict:
    zoo = _state(request).zoo
    models = []
    for spec in zoo.list_specs():
        entry = {
            "id": spec.id,
            "label": spec.label,
            "n_layers": spec.n_layers,
            "n_heads": spec.n_heads,
            "d_model": spec.d_model,
            "n_params": spec.n_params,
            "max_seq": spec.max_seq,
            "languages": spec.languages,
            "tier": spec.tier,
            "status": zoo.status(spec.id),
            "est_ram_mb": spec.est_ram_mb,
            "blurb": spec.blurb,
        }
        reason = zoo.disabled_reason(spec.id)
        if reason is not None:
            entry["reason"] = reason
        models.append(entry)
    return {"models": models, "budget": {"limit_mb": zoo.budget_mb, "used_mb": zoo.used_mb}}


@router.get("/version")
async def version() -> dict:
    return {"version": VERSION, "revision": REVISION}


@router.get("/health")
async def health(request: Request) -> dict:
    """For the uptime monitor and deploy/deploy.sh. `ready` turns true once
    startup preloading is done; the deploy waits for it and for `revision`."""
    state = _state(request)
    zoo = state.zoo
    footprint = memory.footprint_mb()
    return {
        "ok": True,
        "ready": state.ready,
        "version": VERSION,
        "revision": REVISION,
        "uptime_s": round(time.time() - state.started_at),
        "tl_version": TL_VERSION,
        "device": SETTINGS.device,
        "budget": {"limit_mb": zoo.budget_mb, "used_mb": zoo.used_mb},
        "resident_models": zoo.resident_ids(),
        "queue_depth": state.queue_depth,
        "memory": {
            "footprint_mb": round(footprint) if footprint is not None else None,
            "limit_mb": round(SETTINGS.memory_limit_gb * 1024) or None,
            "runs": len(state.runs),
            "runs_mb": round(state.runs.nbytes / 2**20, 1),
            "lens_runs": len(state.lens_runs),
            "lens_runs_mb": round(state.lens_runs.nbytes / 2**20, 1),
        },
        "counters": state.counters,
    }


def _do_tokenize(state: AppState, model_id: str, text: str):
    # Note: this DOES trigger a full model load if `model_id` isn't
    # already resident (HookedTransformer bundles the tokenizer with the
    # weights — there's no lighter-weight path in TransformerLens without
    # bypassing it entirely). docs/02-api.md's "<50ms" target describes
    # the STEADY STATE this endpoint is actually built for: a model the
    # user has already selected and that's already loaded, tokenizing on
    # every keystroke. The one-time cold-load cost is identical to, and
    # no worse than, any other endpoint's first touch of that model.
    model = state.zoo.get_or_load(model_id)
    records, tokens = tokenize_with_offsets(model, text)
    return records, tokens


@router.post("/tokenize")
async def tokenize(request: Request, body: TokenizeRequest) -> dict:
    state = _state(request)
    check_text(body.text)
    with measure("tokenize", model=body.model) as m:
        records, tokens = await _serialized(request, _do_tokenize, state, body.model, body.text)
    spec = state.zoo.spec(body.model)
    return {
        "tokens": [r.to_dict() for r in records],
        "n_tokens": tokens.shape[1],
        "max_seq": spec.max_seq,
        "_meta": build_meta(m),
    }


def _do_run(state: AppState, body: RunRequest):
    model = state.zoo.get_or_load(body.model)
    spec = state.zoo.spec(body.model)

    if body.text is not None:
        token_records, tokens = tokenize_with_offsets(model, body.text)
    else:
        r = body.repeated
        assert r is not None  # guaranteed by RunRequest's validator
        tokens = make_repeated_tokens(
            model, length=r.length, seed=r.seed, prepend_bos=r.prepend_bos
        )
        token_records = token_records_from_ids(model, tokens[0].tolist())

    seq = tokens.shape[1]
    if seq > spec.max_seq:
        raise SeqTooLongError(
            f"{body.model}: sequence length {seq} exceeds max_seq {spec.max_seq}",
            {"max_seq": spec.max_seq, "got": seq},
        )

    result = run_forward(model, tokens, top_k=body.top_k)
    return spec, model, seq, token_records, result


@router.post("/run")
async def run(request: Request, body: RunRequest) -> dict:
    state = _state(request)
    if body.text is not None:
        check_text(body.text)
    with measure("run", model=body.model) as m:
        spec, model, seq, token_records, result = await _serialized(request, _do_run, state, body)

        rec = state.store_run(
            model_id=body.model,
            layers=result.layers,
            n_layers=model.cfg.n_layers,
            n_heads=model.cfg.n_heads,
            seq=seq,
        )

    n_layers, n_heads, d_model = model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_model
    n_params = spec.n_params
    # Deterministic, computed from shapes — NOT measured — so it is exact
    # and identical on every machine (docs/02-api.md, docs/PLAN.md Stage
    # 1.5's public cost panel renders exactly this block).
    cost = {
        "attention_bytes_f32": n_layers * n_heads * seq * seq * 4,
        "kv_cache_bytes": 2 * n_layers * d_model * seq * 4,
        "weights_bytes": n_params * 4,
        "forward_flops": 2 * n_params * seq + 4 * n_layers * seq * seq * d_model,
    }

    return {
        "run_id": rec.run_id,
        "tokens": [r.to_dict() for r in token_records],
        "n_layers": n_layers,
        "n_heads": n_heads,
        "loss_per_token": result.loss_per_token,
        "top_logits": result.top_logits,
        "cost": cost,
        "expires_at": datetime.fromtimestamp(rec.expires_at, tz=timezone.utc).isoformat(),
        "_meta": build_meta(m),
    }


@router.get("/run/{run_id}/patterns")
async def get_patterns(
    request: Request, run_id: str, layers: str = Query(..., description="comma-separated layer indices")
) -> Response:
    # Deliberately NOT wrapped in the model slot: this reads bytes already
    # encoded by a prior /run call and
    # never touches the zoo or the model, so it doesn't compete with
    # forward-pass work for the single global slot (docs/02-api.md: a
    # per-layer fetch is supposed to be cheap and fast).
    state = _state(request)
    rec = state.get_run(run_id)
    if rec is None:
        raise RunNotFoundError(f"run {run_id!r} not found or expired", {"run_id": run_id})

    try:
        layer_ids = sorted({int(x) for x in layers.split(",") if x.strip() != ""})
    except ValueError:
        raise InvalidRequestError(f"invalid `layers` value: {layers!r}", {"layers": layers}) from None

    if not layer_ids:
        raise InvalidRequestError("`layers` is required and must name at least one layer")

    out_of_range = [l for l in layer_ids if l < 0 or l >= rec.n_layers]
    if out_of_range:
        raise InvalidRequestError(
            f"layer(s) out of range for this run: {out_of_range}",
            {"n_layers": rec.n_layers, "requested": layer_ids},
        )

    with measure("encode_patterns", model=rec.model_id, seq=rec.seq):
        payload = frame_layers({l: rec.layers[l] for l in layer_ids}, n_heads=rec.n_heads, seq=rec.seq)

    return Response(content=payload, media_type="application/octet-stream")
