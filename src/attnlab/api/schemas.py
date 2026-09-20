"""Pydantic request models. Responses are hand-built dicts in routes.py
(matching docs/02-api.md's exact shapes, including the literal `_meta`
key) rather than pydantic response_model — pydantic v2 disallows
leading-underscore field names as ordinary public fields, and working
around that with aliases would add more complexity than it removes."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class TokenizeRequest(BaseModel):
    model: str
    text: str


class RepeatedSpec(BaseModel):
    # `le=2000` here is a sanity ceiling against a malformed/malicious
    # request, deliberately set well above any registry model's max_seq
    # (currently 512). The REAL "too long" check is per-model and lives
    # in routes.py as SeqTooLongError — keeping this bound generous means
    # every realistic "too long" case funnels through that ONE documented
    # error shape ({"error": {"code": "seq_too_long", ...}}) instead of
    # sometimes hitting this field constraint's own differently-shaped
    # FastAPI validation error and sometimes not, depending on which
    # threshold happens to trip first.
    length: int = Field(gt=0, le=2000)
    seed: int
    prepend_bos: bool = True


class RunRequest(BaseModel):
    model: str
    text: str | None = None
    repeated: RepeatedSpec | None = None
    top_k: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def _exactly_one_input(self) -> "RunRequest":
        if (self.text is None) == (self.repeated is None):
            raise ValueError("provide exactly one of `text` or `repeated`, not both/neither")
        return self
