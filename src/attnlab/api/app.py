"""
FastAPI app: lifespan (builds AppState once, preloads the baked models, evicts
all models on shutdown), the HTTP guards, the built frontend, and exception
handlers translating both this package's ApiError family and zoo.py's
exceptions into the JSON error shape from docs/02-api.md.

One process serves the whole site: /api/* from the routers, everything else
from the built frontend (web/dist) when it exists (D19). The frontend and the
API it talks to therefore always come from the same release.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.middleware.gzip import GZipMiddleware

from attnlab import memory
from attnlab.api import guards, lens_routes, routes, toklab_routes
from attnlab.api.errors import ApiError
from attnlab.api.meta import REVISION, VERSION
from attnlab.api.state import AppState
from attnlab.settings import SERVING_KEYS, SETTINGS
from attnlab.toklab import TokenizerUnavailableError, UnknownTokenizerError
from attnlab.zoo import BudgetExceededError, ModelDisabledError, UnknownModelError

log = logging.getLogger("attnlab")
if not log.handlers and not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log.setLevel(logging.INFO)


async def _preload(state: AppState) -> None:
    """Load the baked models one at a time, through the model slot like any
    request, so a request arriving mid-preload simply waits its turn."""
    for spec in state.zoo.list_specs():
        if spec.tier != "baked" or state.zoo.status(spec.id) == "disabled":
            continue
        try:
            await state.run_serialized(state.zoo.get_or_load, spec.id)
            log.info("preloaded %s (footprint %.0f MB)", spec.id, memory.footprint_mb() or -1)
        except Exception:  # noqa: BLE001 - one bad model must not stop the server
            log.exception("preloading %s failed", spec.id)
    state.ready = True


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    state = AppState()
    app.state.attnlab = state
    log.info("attnlab %s (%s) starting: %s", VERSION, REVISION, {k: getattr(SETTINGS, k) for k in SERVING_KEYS})
    if not memory.large_cache_disabled():
        log.warning(
            "MallocLargeCache=0 is not set: each model load will leave about a model's worth of "
            "freed memory counted against this process (docs/04-self-hosting.md § 2)"
        )
    task = None
    if SETTINGS.preload:
        task = asyncio.create_task(_preload(state))
    else:
        state.ready = True
    yield
    if task is not None:
        task.cancel()
    state.zoo.evict_all()


app = FastAPI(title="attnlab API", version=VERSION, lifespan=lifespan)
app.include_router(routes.router, prefix="/api")
app.include_router(toklab_routes.router, prefix="/api")
app.include_router(lens_routes.router, prefix="/api")


def _count_rate_limited() -> None:
    state = getattr(app.state, "attnlab", None)
    if state is not None:
        state.counters["rate_limited"] += 1


# Added innermost first: a request passes Headers, then BodyLimit, then
# RateLimit, then gzip, then CORS on its way in.
if SETTINGS.cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(SETTINGS.cors_origins),
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        max_age=3600,
    )
# Attention patterns are ~1.5 MB per layer at 512 tokens and compress about
# 3x; a home upload link is the narrowest pipe on the path. Level 6 is most of
# the saving for a fraction of level 9's CPU.
app.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)
app.add_middleware(guards.RateLimit, burst=SETTINGS.rate_burst, per_s=SETTINGS.rate_per_s, on_limited=_count_rate_limited)
app.add_middleware(guards.BodyLimit, max_bytes=SETTINGS.max_body_bytes)
app.add_middleware(guards.Headers)


# --- errors --------------------------------------------------------------------


def _error_response(
    status_code: int, code: str, message: str, detail: dict[str, Any] | None = None, retry_after: int | None = None
) -> JSONResponse:
    headers = {"Retry-After": str(retry_after)} if retry_after else None
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "detail": detail or {}}},
        headers=headers,
    )


@app.exception_handler(ApiError)
async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
    return _error_response(exc.status_code, exc.code, exc.message, exc.detail, exc.retry_after)


@app.exception_handler(UnknownModelError)
async def handle_unknown_model(request: Request, exc: UnknownModelError) -> JSONResponse:
    return _error_response(404, "unknown_model", str(exc), {"model": exc.model_id})


@app.exception_handler(ModelDisabledError)
async def handle_model_disabled(request: Request, exc: ModelDisabledError) -> JSONResponse:
    return _error_response(403, "model_disabled", str(exc), {"model": exc.model_id, "reason": exc.reason})


@app.exception_handler(BudgetExceededError)
async def handle_budget_exceeded(request: Request, exc: BudgetExceededError) -> JSONResponse:
    return _error_response(
        503,
        "budget_exceeded",
        str(exc),
        {"model": exc.model_id, "needed_mb": exc.needed_mb, "budget_mb": exc.budget_mb},
    )


@app.exception_handler(UnknownTokenizerError)
async def handle_unknown_tokenizer(request: Request, exc: UnknownTokenizerError) -> JSONResponse:
    return _error_response(404, "unknown_tokenizer", str(exc), {"tokenizer": exc.tokenizer_id})


@app.exception_handler(TokenizerUnavailableError)
async def handle_tokenizer_unavailable(request: Request, exc: TokenizerUnavailableError) -> JSONResponse:
    return _error_response(503, "tokenizer_unavailable", str(exc), {"tokenizer": exc.tokenizer_id})


# --- the built frontend ----------------------------------------------------------


def _static_root() -> Path | None:
    if SETTINGS.static_dir:
        p = Path(SETTINGS.static_dir)
    else:
        p = Path(__file__).resolve().parents[3] / "web" / "dist"
    return p.resolve() if (p / "index.html").is_file() else None


def mount_frontend(target: FastAPI, root: Path) -> None:
    """Serve the built frontend from `root` for every non-/api path."""
    # Vite puts a content hash in every file name under assets/, so those can
    # be cached forever (by browsers and by Cloudflare's edge); index.html must
    # always be revalidated, or a release would never reach returning users.
    immutable = {"Cache-Control": "public, max-age=31536000, immutable"}
    revalidate = {"Cache-Control": "no-cache"}

    @target.get("/{full_path:path}", include_in_schema=False)
    async def frontend(full_path: str):
        if full_path == "api" or full_path.startswith("api/"):
            return _error_response(404, "not_found", f"no such endpoint: /{full_path}")
        candidate = (root / full_path).resolve()
        if full_path and candidate.is_file() and candidate.is_relative_to(root):
            cached = candidate.is_relative_to(root / "assets")
            return FileResponse(candidate, headers=immutable if cached else revalidate)
        # Everything else is a client-side route (/tokens, /lens, ...).
        return FileResponse(root / "index.html", headers=revalidate)


STATIC_ROOT = _static_root()
if STATIC_ROOT is not None:
    mount_frontend(app, STATIC_ROOT)
elif "pytest" not in sys.modules:
    log.info("no built frontend found (web/dist); serving the API only")
