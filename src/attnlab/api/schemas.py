"""Pydantic request models. Responses are hand-built dicts in routes.py
(matching docs/02-api.md's exact shapes, including the literal `_meta`
key) rather than pydantic response_model — pydantic v2 disallows
leading-underscore field names as ordinary public fields, and working
around that with aliases would add more complexity than it removes."""

from __future__ import annotations

from typing import Literal

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


# --- Tokenizer lab (api/toklab_routes.py) ---------------------------------
# Length limits live in the routes, not here, for the same reason as
# RepeatedSpec.length above: one documented error shape for "too long".


class AnalyzeRequest(BaseModel):
    tokenizers: list[str] = Field(min_length=1)
    text: str
    add_special_tokens: bool = False


class TraceRequest(BaseModel):
    tokenizer: str
    text: str


class CountRequest(BaseModel):
    tokenizers: list[str] = Field(min_length=1)
    texts: list[str] = Field(min_length=1)


# --- Logit lens lab (api/lens_routes.py) ------------------------------------

LensName = Literal["ln_final", "plain"]


class LensRunRequest(BaseModel):
    model: str
    text: str
    lens: LensName = "ln_final"
    prepend_bos: bool = True


class LensViewRequest(BaseModel):
    """Re-read a stored run under a different lens, without a new forward pass."""

    run_id: str
    lens: LensName = "ln_final"


class LensPositionRequest(BaseModel):
    run_id: str
    pos: int = Field(ge=0)
    lens: LensName = "ln_final"
    k: int = Field(default=10, ge=1, le=25)
    track: list[str] = Field(default_factory=list, max_length=6)
    track_ids: list[int] = Field(default_factory=list, max_length=6)


class LensAttributionRequest(BaseModel):
    run_id: str
    pos: int = Field(ge=0)
    # a token id, or a string whose first token is used; neither means "the
    # actual next token", or the model's own top prediction at the last position
    target: int | None = None
    target_str: str | None = None
    contrast: int | None = None
    contrast_str: str | None = None
