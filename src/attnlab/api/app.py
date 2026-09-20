"""
FastAPI app: lifespan (builds AppState once, evicts all models on
shutdown), and exception handlers translating both this package's
ApiError family and zoo.py's exceptions into the JSON error shape from
docs/02-api.md.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from attnlab.api import routes
from attnlab.api.errors import ApiError
from attnlab.api.state import AppState
from attnlab.zoo import BudgetExceededError, ModelDisabledError, UnknownModelError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    app.state.attnlab = AppState()
    yield
    app.state.attnlab.zoo.evict_all()


app = FastAPI(title="attnlab API", lifespan=lifespan)
app.include_router(routes.router, prefix="/api")


def _error_response(status_code: int, code: str, message: str, detail: dict[str, Any] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "detail": detail or {}}},
    )


@app.exception_handler(ApiError)
async def handle_api_error(request: Request, exc: ApiError) -> JSONResponse:
    return _error_response(exc.status_code, exc.code, exc.message, exc.detail)


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
