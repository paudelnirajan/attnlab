"""
The endpoints from docs/02-api.md, Stage 0b subset:
  GET  /models
  POST /tokenize
  POST /run
  GET  /run/{run_id}/patterns
  GET  /health

Concurrency: every route that touches the zoo or runs a forward pass
goes through `state.serialize()` (docs/03-decisions.md D6 — one
model-touching operation at a time, server-wide) and runs the actual
blocking torch call in a thread executor so the event loop keeps serving
other connections (e.g. /health) while it runs.
"""

from __future__ import annotations

import asyncio
import functools
from datetime import datetime, timezone
from typing import Any, Callable

from fastapi import APIRouter, Query, Request, Response

from attnlab.api.errors import BusyError, InvalidRequestError, RunNotFoundError, SeqTooLongError
from attnlab.api.meta import TL_VERSION, build_meta
from attnlab.api.schemas import RunRequest, TokenizeRequest
from attnlab.api.state import AppState
from attnlab.inference import make_repeated_tokens, run_forward, token_records_from_ids, tokenize_with_offsets
from attnlab.instrument import measure
from attnlab.patterns import encode_layers
from attnlab.settings import SETTINGS

router = APIRouter()

REQUEST_TIMEOUT_SECONDS = 60.0  # generous vs. the ~2.4s worst case measured in Stage 0a


def _state(request: Request) -> AppState:
    return request.app.state.attnlab


async def _run_blocking(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Runs a blocking (torch) call in a thread executor so the event
    loop isn't blocked while it runs — the semaphore already guarantees
    only one such call proceeds at a time; this just keeps OTHER routes
    (like /health) responsive while it does."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))


async def _serialized(state: AppState, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        async with state.serialize():
            return await asyncio.wait_for(
                _run_blocking(fn, *args, **kwargs), timeout=REQUEST_TIMEOUT_SECONDS
            )
    except asyncio.TimeoutError:
        raise BusyError(f"request exceeded {REQUEST_TIMEOUT_SECONDS:.0f}s timeout") from None


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
        if spec.tier == "disabled":
            entry["reason"] = spec.reason
        models.append(entry)
    return {"models": models, "budget": {"limit_mb": zoo.budget_mb, "used_mb": zoo.used_mb}}


@router.get("/health")
async def health(request: Request) -> dict:
    zoo = _state(request).zoo
    return {
        "ok": True,
        "tl_version": TL_VERSION,
        "device": SETTINGS.device,
        "budget": {"limit_mb": zoo.budget_mb, "used_mb": zoo.used_mb},
        "resident_models": zoo.resident_ids(),
        "queue_depth": _state(request).queue_depth,
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
    with measure("tokenize", model=body.model) as m:
        records, tokens = await _serialized(state, _do_tokenize, state, body.model, body.text)
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
    with measure("run", model=body.model) as m:
        spec, model, seq, token_records, result = await _serialized(state, _do_run, state, body)

        rec = state.store_run(
            model_id=body.model,
            patterns=result.patterns,
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
    # Deliberately NOT wrapped in state.serialize(): this reads an
    # already-computed numpy array cached from a prior /run call and
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
        payload = encode_layers({l: rec.patterns[l] for l in layer_ids})

    return Response(content=payload, media_type="application/octet-stream")
