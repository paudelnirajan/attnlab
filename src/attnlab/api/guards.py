"""
HTTP-level protection, applied to every request before it reaches a route
(docs/04-self-hosting.md § 3):

  BodyLimit   refuses request bodies over MI_MAX_BODY_BYTES (413). Every
              legitimate body here is a prompt plus a few options.
  RateLimit   a token bucket per client over POST /api/*: MI_RATE_BURST
              requests at once, refilled at MI_RATE_PER_S (429 + Retry-After).
              The attention lab re-runs after every pause in typing, so the
              burst has to cover a normal editing session, not one click.
  Headers     a few standard security headers on every response.

Plain ASGI rather than Starlette's BaseHTTPMiddleware on purpose: that one
wraps the receive channel, and the model slot relies on
`request.is_disconnected()` to skip work for clients that have left.
"""

from __future__ import annotations

import json
import math
import time
from typing import Any

from attnlab.settings import SETTINGS

Scope = dict[str, Any]


async def _send_error(send, status: int, code: str, message: str, headers: tuple[tuple[bytes, bytes], ...] = ()) -> None:
    body = json.dumps({"error": {"code": code, "message": message, "detail": {}}}).encode()
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), *headers],
        }
    )
    await send({"type": "http.response.body", "body": body})


def client_key(scope: Scope) -> str:
    """Who a request counts against. Behind Cloudflare every request arrives
    from cloudflared on 127.0.0.1, and the real address is in
    CF-Connecting-IP; that header is only trusted when MI_TRUST_PROXY is set,
    since anyone reaching the server directly could send their own."""
    if SETTINGS.trust_proxy:
        for name, value in scope.get("headers", ()):
            if name == b"cf-connecting-ip":
                return value.decode("latin-1")
    client = scope.get("client")
    return client[0] if client else "unknown"


class BodyLimit:
    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive, send) -> None:
        if scope["type"] != "http" or self.max_bytes <= 0:
            return await self.app(scope, receive, send)
        for name, value in scope.get("headers", ()):
            if name == b"content-length" and value.isdigit() and int(value) > self.max_bytes:
                return await _send_error(send, 413, "payload_too_large", f"request body over {self.max_bytes} bytes")

        seen = 0

        async def limited_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise _TooLarge()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _TooLarge:
            await _send_error(send, 413, "payload_too_large", f"request body over {self.max_bytes} bytes")


class _TooLarge(Exception):
    pass


class RateLimit:
    MAX_CLIENTS = 10_000  # forget the least recently seen past this

    def __init__(self, app, *, burst: int, per_s: float, on_limited=None):
        self.app = app
        self.burst = burst
        self.per_s = per_s
        self.on_limited = on_limited
        self._buckets: dict[str, tuple[float, float]] = {}  # key -> (tokens, last seen)

    def take(self, key: str, now: float) -> float:
        """Spend one token. Returns 0 if allowed, else seconds until one is."""
        tokens, last = self._buckets.pop(key, (float(self.burst), now))
        tokens = min(float(self.burst), tokens + (now - last) * self.per_s)
        allowed = tokens >= 1.0
        if allowed:
            tokens -= 1.0
        self._buckets[key] = (tokens, now)  # re-inserted: dict order = recency
        if len(self._buckets) > self.MAX_CLIENTS:
            self._buckets.pop(next(iter(self._buckets)))
        return 0.0 if allowed else (1.0 - tokens) / self.per_s

    async def __call__(self, scope: Scope, receive, send) -> None:
        if (
            scope["type"] != "http"
            or self.burst <= 0
            or scope["method"] != "POST"
            or not scope["path"].startswith("/api/")
        ):
            return await self.app(scope, receive, send)
        wait = self.take(client_key(scope), time.monotonic())
        if wait > 0:
            if self.on_limited:
                self.on_limited()
            retry = str(max(1, math.ceil(wait))).encode()
            return await _send_error(
                send, 429, "rate_limited", "too many requests; slow down a little", ((b"retry-after", retry),)
            )
        await self.app(scope, receive, send)


SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
    (b"x-frame-options", b"DENY"),
    (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
]


class Headers:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope: Scope, receive, send) -> None:
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def with_headers(message):
            if message["type"] == "http.response.start":
                message.setdefault("headers", [])
                message["headers"] = [*message["headers"], *SECURITY_HEADERS]
            await send(message)

        await self.app(scope, receive, with_headers)
