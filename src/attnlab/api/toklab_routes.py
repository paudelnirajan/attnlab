"""
Tokenizer lab endpoints (docs/02-api.md § Tokenizer lab):

  GET  /tokenizers
  POST /toklab/analyze     one text, one or more tokenizers
  POST /toklab/trace       step-by-step replay of the tokenizer's algorithm
  POST /toklab/count       many texts x many tokenizers -> token counts
  GET  /toklab/vocab       summary of one tokenizer's vocabulary
  GET  /toklab/vocab/search

None of these touch the model zoo, so none of them take the forward-pass
semaphore: tokenizing is milliseconds of Rust and must never queue behind a
multi-second forward pass. Work still runs in a thread (first use of a
tokenizer downloads it; building a vocabulary table takes up to a second) so
the event loop stays free.
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Query, Request

from attnlab.api.errors import InvalidRequestError, TextTooLongError
from attnlab.api.schemas import AnalyzeRequest, CountRequest, TraceRequest
from attnlab.toklab import (
    MAX_COUNT_TEXTS,
    MAX_COUNT_TOKENIZERS,
    MAX_TEXT_CHARS,
    SCRIPTS,
    TokenizerCache,
    analyze,
    count,
    trace,
    vocab_search,
    vocab_summary,
)

router = APIRouter()

MAX_ANALYZE_TOKENIZERS = 6


def _cache(request: Request) -> TokenizerCache:
    return request.app.state.attnlab.tokenizers


def _check_text(text: str) -> None:
    if len(text) > MAX_TEXT_CHARS:
        raise TextTooLongError(
            f"text is {len(text)} characters; the tokenizer lab accepts up to {MAX_TEXT_CHARS}",
            {"max_chars": MAX_TEXT_CHARS, "got": len(text)},
        )


@router.get("/tokenizers")
async def list_tokenizers(request: Request) -> dict:
    cache = _cache(request)
    return {
        "tokenizers": [
            {
                "id": s.id,
                "label": s.label,
                "hf_name": s.hf_name,
                "algorithm": s.algorithm,
                "year": s.year,
                "source": s.source,
                "models": s.models,
                "blurb": s.blurb,
                "loaded": cache.is_loaded(s.id),
            }
            for s in cache.list_specs()
        ]
    }


@router.post("/toklab/analyze")
async def analyze_route(request: Request, body: AnalyzeRequest) -> dict:
    _check_text(body.text)
    ids = list(dict.fromkeys(body.tokenizers))  # dedupe, keep order
    if not ids or len(ids) > MAX_ANALYZE_TOKENIZERS:
        raise InvalidRequestError(f"name between 1 and {MAX_ANALYZE_TOKENIZERS} tokenizers", {"got": len(ids)})
    cache = _cache(request)
    for tid in ids:
        cache.spec(tid)  # 404 before doing any work

    def work() -> list[dict]:
        return [analyze(cache.get(tid), body.text, add_special_tokens=body.add_special_tokens) for tid in ids]

    return {"results": await asyncio.to_thread(work)}


@router.post("/toklab/trace")
async def trace_route(request: Request, body: TraceRequest) -> dict:
    if len(body.text) > 2_000:
        raise TextTooLongError("trace accepts up to 2000 characters", {"max_chars": 2_000, "got": len(body.text)})
    cache = _cache(request)
    cache.spec(body.tokenizer)
    return await asyncio.to_thread(lambda: trace(cache.get(body.tokenizer), body.text))


@router.post("/toklab/count")
async def count_route(request: Request, body: CountRequest) -> dict:
    if len(body.texts) > MAX_COUNT_TEXTS or len(body.tokenizers) > MAX_COUNT_TOKENIZERS:
        raise InvalidRequestError(
            f"at most {MAX_COUNT_TEXTS} texts and {MAX_COUNT_TOKENIZERS} tokenizers per request",
            {"texts": len(body.texts), "tokenizers": len(body.tokenizers)},
        )
    for t in body.texts:
        _check_text(t)
    cache = _cache(request)
    ids = list(dict.fromkeys(body.tokenizers))
    for tid in ids:
        cache.spec(tid)
    return await asyncio.to_thread(lambda: count([cache.get(t) for t in ids], body.texts))


@router.get("/toklab/vocab")
async def vocab_route(request: Request, tokenizer: str = Query(...)) -> dict:
    cache = _cache(request)
    cache.spec(tokenizer)
    return await asyncio.to_thread(lambda: vocab_summary(cache.get(tokenizer)))


@router.get("/toklab/vocab/search")
async def vocab_search_route(
    request: Request,
    tokenizer: str = Query(...),
    q: str = Query("", max_length=200),
    script: str | None = Query(None),
    limit: int = Query(100, ge=1, le=200),
) -> dict:
    if script is not None and script not in (*SCRIPTS, "other", "partial"):
        raise InvalidRequestError(f"unknown script {script!r}", {"scripts": [*SCRIPTS, "other", "partial"]})
    cache = _cache(request)
    cache.spec(tokenizer)
    return await asyncio.to_thread(lambda: vocab_search(cache.get(tokenizer), q, script=script, limit=limit))
