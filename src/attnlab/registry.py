"""
Loads models.yaml into typed records. This is the ONLY place that reads
the registry file — bench.py, zoo.py (Stage 0b), and the API all go
through this module, so adding a model is purely a YAML edit.
"""

from __future__ import annotations

import dataclasses
from importlib import resources
from pathlib import Path

import yaml

VALID_TIERS = {"baked", "lazy", "disabled"}


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    id: str
    label: str
    tl_name: str
    tier: str
    max_seq: int
    languages: list[str]
    est_ram_mb: float
    blurb: str
    # Static architecture metadata for GET /api/models — baked in from
    # Stage 0a's benchmark data (bench_results/results.json) plus a
    # lightweight get_pretrained_model_config() lookup for d_model, rather
    # than the API loading each model just to report its shape. Means
    # /api/models works instantly even for `lazy`-tier models that have
    # never been loaded.
    n_layers: int
    n_heads: int
    d_model: int
    n_params: int
    reason: str | None = None  # populated when tier == "disabled"

    def __post_init__(self) -> None:
        if self.tier not in VALID_TIERS:
            raise ValueError(f"{self.id}: invalid tier {self.tier!r}, must be one of {VALID_TIERS}")
        if self.tier == "disabled" and not self.reason:
            raise ValueError(f"{self.id}: tier=disabled requires a 'reason' field")


def _default_registry_path() -> Path:
    return resources.files("attnlab").joinpath("models.yaml")  # type: ignore[return-value]


def load_registry(path: str | Path | None = None) -> list[ModelSpec]:
    p = Path(path) if path is not None else _default_registry_path()
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"{p}: expected a YAML list of model specs")

    specs: list[ModelSpec] = []
    seen_ids: set[str] = set()
    for entry in raw:
        spec = ModelSpec(**entry)
        if spec.id in seen_ids:
            raise ValueError(f"duplicate model id in registry: {spec.id}")
        seen_ids.add(spec.id)
        specs.append(spec)
    return specs


def get_spec(model_id: str, *, path: str | Path | None = None) -> ModelSpec:
    for spec in load_registry(path):
        if spec.id == model_id:
            return spec
    raise KeyError(f"unknown model id: {model_id!r}")


if __name__ == "__main__":
    specs = load_registry()
    total_baked_mb = sum(s.est_ram_mb for s in specs if s.tier == "baked")
    print(f"loaded {len(specs)} models from registry")
    for s in specs:
        print(f"  {s.id:20s} tier={s.tier:8s} est_ram_mb={s.est_ram_mb:>7.0f}  {s.blurb}")
    print(f"\nestimated RAM for all 'baked' models: {total_baked_mb:.0f} MB")
