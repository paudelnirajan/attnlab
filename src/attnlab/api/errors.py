"""
Typed API errors mapped to the exact JSON shape from docs/02-api.md:

    {"error": {"code": ..., "message": ..., "detail": {...}}}

registered as FastAPI exception handlers in app.py. zoo.py's own
exceptions (UnknownModelError, ModelDisabledError, BudgetExceededError)
get their own handlers there too, rather than being caught and
re-raised as one of these — so routes.py doesn't need try/except
boilerplate around every zoo call.
"""

from __future__ import annotations


class ApiError(Exception):
    code: str = "internal_error"
    status_code: int = 500

    def __init__(self, message: str, detail: dict | None = None, *, retry_after: int | None = None):
        self.message = message
        self.detail = detail or {}
        self.retry_after = retry_after  # seconds; sent as the Retry-After header
        super().__init__(message)


class InvalidRequestError(ApiError):
    code = "invalid_request"
    status_code = 422


class SeqTooLongError(ApiError):
    code = "seq_too_long"
    status_code = 422


class RunNotFoundError(ApiError):
    code = "run_not_found"
    status_code = 404


class BusyError(ApiError):
    code = "busy"
    status_code = 503


class TextTooLongError(ApiError):
    code = "text_too_long"
    status_code = 422


class OverloadedError(ApiError):
    """The memory guard (api/state.py) couldn't get under MI_MEMORY_LIMIT_GB."""

    code = "overloaded"
    status_code = 503


class RateLimitedError(ApiError):
    code = "rate_limited"
    status_code = 429


class ClientGoneError(ApiError):
    """The client disconnected while its request was queued. Nobody reads
    the response; 499 is nginx's convention for it, which keeps these
    visible in the access log without looking like server errors."""

    code = "client_gone"
    status_code = 499


class PayloadTooLargeError(ApiError):
    code = "payload_too_large"
    status_code = 413
