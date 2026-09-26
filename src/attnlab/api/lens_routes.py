"""
Logit lens lab endpoints (docs/02-api.md § Logit lens):

  POST /lens/run          one forward pass, kept; the grid for one lens
  POST /lens/view         the same run's grid under another lens
  POST /lens/layers       per-layer averages for every lens, and residual norms
  POST /lens/position     one position: top-k at every row, tracked tokens
  POST /lens/attribution  direct logit attribution at one position

Only /lens/run runs the model. The rest are matrix products against the stored
residual stream, but they still read the model's weights (W_U, W_O), so they go
through the same semaphore as every other model-touching route (D6): the zoo
may be evicting or loading that model at the same moment.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from attnlab.api.errors import InvalidRequestError, RunNotFoundError, SeqTooLongError
from attnlab.api.meta import build_meta
from attnlab.api.routes import _serialized, _state
from attnlab.api.schemas import (
    LensAttributionRequest,
    LensPositionRequest,
    LensRunRequest,
    LensViewRequest,
)
from attnlab.api.state import AppState, LensRunRecord
from attnlab.inference import tokenize_with_offsets
from attnlab.instrument import measure
from attnlab.lens import (
    MAX_LENS_SEQ,
    MAX_TRACKED,
    anatomy,
    attribution,
    available_lenses,
    layer_curves,
    output_logits,
    position_detail,
    resolve_track,
    run_lens,
    summarize,
)

router = APIRouter()


def _record(state: AppState, run_id: str) -> LensRunRecord:
    rec = state.get_lens_run(run_id)
    if rec is None:
        raise RunNotFoundError(f"lens run {run_id!r} not found or expired", {"run_id": run_id})
    return rec


def _check_lens(model, lens: str) -> None:
    if lens not in available_lenses(model):
        raise InvalidRequestError(
            f"lens {lens!r} is not available for this model", {"available": available_lenses(model)}
        )


def _check_pos(rec: LensRunRecord, pos: int) -> None:
    if pos >= rec.run.seq:
        raise InvalidRequestError(f"pos {pos} is past the end (seq {rec.run.seq})", {"seq": rec.run.seq})


def _do_run(state: AppState, body: LensRunRequest):
    model = state.zoo.get_or_load(body.model)
    spec = state.zoo.spec(body.model)
    _check_lens(model, body.lens)
    records, tokens = tokenize_with_offsets(model, body.text, prepend_bos=body.prepend_bos)
    limit = min(MAX_LENS_SEQ, spec.max_seq)
    if tokens.shape[1] > limit:
        raise SeqTooLongError(
            f"{tokens.shape[1]} tokens; the logit lens reads up to {limit} (one grid column per token)",
            {"max_seq": limit, "got": tokens.shape[1]},
        )
    if tokens.shape[1] == 0:
        raise InvalidRequestError("nothing to read: the text produced no tokens")
    run = run_lens(model, tokens)
    summary = summarize(model, run, body.lens, model.tokenizer)
    return model, records, run, summary, anatomy(model, run, model.tokenizer)


@router.post("/lens/run")
async def lens_run(request: Request, body: LensRunRequest) -> dict:
    state = _state(request)
    with measure("lens_run", model=body.model) as m:
        model, records, run, summary, anat = await _serialized(state, _do_run, state, body)
        rec = state.store_lens_run(model_id=body.model, run=run)
    return {
        "run_id": rec.run_id,
        "model": body.model,
        "lens": body.lens,
        "lenses": available_lenses(model),
        "prepend_bos": body.prepend_bos,
        "tokens": [r.to_dict() for r in records],
        **summary,
        "checks": run.checks,
        "anatomy": anat,
        "stored_mb": round(run.nbytes / 2**20, 1),
        "_meta": build_meta(m),
    }


@router.post("/lens/view")
async def lens_view(request: Request, body: LensViewRequest) -> dict:
    state = _state(request)
    rec = _record(state, body.run_id)

    def work():
        model = state.zoo.get_or_load(rec.model_id)
        _check_lens(model, body.lens)
        return summarize(model, rec.run, body.lens, model.tokenizer)

    with measure("lens_view", model=rec.model_id) as m:
        summary = await _serialized(state, work)
    return {"run_id": rec.run_id, "lens": body.lens, **summary, "_meta": build_meta(m)}


@router.post("/lens/layers")
async def lens_layers(request: Request, body: LensViewRequest) -> dict:
    state = _state(request)
    rec = _record(state, body.run_id)

    def work():
        model = state.zoo.get_or_load(rec.model_id)
        return layer_curves(model, rec.run, model.tokenizer)

    with measure("lens_layers", model=rec.model_id) as m:
        curves = await _serialized(state, work)
    return {"run_id": rec.run_id, **curves, "_meta": build_meta(m)}


@router.post("/lens/position")
async def lens_position(request: Request, body: LensPositionRequest) -> dict:
    state = _state(request)
    rec = _record(state, body.run_id)
    _check_pos(rec, body.pos)
    if len(body.track) + len(body.track_ids) > MAX_TRACKED:
        raise InvalidRequestError(f"track at most {MAX_TRACKED} tokens")

    def work():
        model = state.zoo.get_or_load(rec.model_id)
        _check_lens(model, body.lens)
        d_vocab = model.cfg.d_vocab
        bad = [t for t in body.track_ids if not 0 <= t < d_vocab]
        if bad:
            raise InvalidRequestError(f"token ids out of range: {bad}", {"d_vocab": d_vocab})
        resolved = []
        for s in body.track:
            try:
                resolved.append({"query": s, **resolve_track(model, s)})
            except ValueError:
                raise InvalidRequestError("cannot track an empty string") from None
        ids = list(dict.fromkeys([*body.track_ids, *(r["id"] for r in resolved)]))
        detail = position_detail(model, rec.run, model.tokenizer, body.pos, body.lens, body.k, ids)
        return detail, resolved

    with measure("lens_position", model=rec.model_id) as m:
        detail, resolved = await _serialized(state, work)
    return {"run_id": rec.run_id, **detail, "resolved": resolved, "_meta": build_meta(m)}


@router.post("/lens/attribution")
async def lens_attribution(request: Request, body: LensAttributionRequest) -> dict:
    state = _state(request)
    rec = _record(state, body.run_id)
    _check_pos(rec, body.pos)

    def work():
        model = state.zoo.get_or_load(rec.model_id)
        run = rec.run
        notes = []

        def pick(tid: int | None, s: str | None) -> int | None:
            if tid is not None:
                if not 0 <= tid < model.cfg.d_vocab:
                    raise InvalidRequestError(f"token id {tid} out of range", {"d_vocab": model.cfg.d_vocab})
                return tid
            if s:
                r = resolve_track(model, s)
                if r["note"]:
                    notes.append(r["note"])
                return r["id"]
            return None

        target = pick(body.target, body.target_str)
        if target is None:
            # the actual next token when there is one, otherwise what the model predicts
            target = run.ids[body.pos + 1] if body.pos + 1 < run.seq else int(output_logits(model, run)[body.pos].argmax())
        contrast = pick(body.contrast, body.contrast_str)
        if contrast == target:
            raise InvalidRequestError("target and contrast are the same token")
        out = attribution(model, run, model.tokenizer, body.pos, target, contrast)
        out["notes"] = notes
        return out

    with measure("lens_attribution", model=rec.model_id) as m:
        out = await _serialized(state, work)
    return {"run_id": rec.run_id, **out, "_meta": build_meta(m)}
