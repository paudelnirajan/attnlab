"""
LRU model cache with an ENFORCED byte budget (docs/PLAN.md Stage 0b,
docs/03-decisions.md D6 + D8).

Budgets against `est_ram_mb` (resident) and `load_extra_mb` (the extra held
only while loading) from models.yaml — a value MEASURED in
Stage 0a (see the comment at the top of models.yaml), not a live RSS
delta. D8 established why the latter is unreliable: RSS deltas across
sequential in-process loads are allocator-history-dependent, both
under-counting (freed pages reused) and over-counting (freed pages never
returned to the OS). The registry's static, pre-measured number has no
such ambiguity.

Refuses admission outright when a single model's cost exceeds the WHOLE
budget (D6): no amount of eviction helps that case, and pretending
otherwise is exactly how the Stage 0a `bloom-560m` OOM-kill happened
(docs/03-decisions.md D1 amendment) — a visible `BudgetExceededError` beats
a silent kernel SIGKILL every time.

NOT thread-safe on its own. The API layer serializes ALL access through a
single asyncio.Semaphore(1) (docs/PLAN.md D6: "one forward pass at a time,
server-wide") — that serialization is what makes the plain OrderedDict
here safe to use without its own locking.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field

from transformer_lens import HookedTransformer

from attnlab import hub, memory
from attnlab.lens import LensReadyTransformer
from attnlab.registry import ModelSpec, load_registry
from attnlab.settings import SETTINGS

hub.install_cached_listing()  # loads must work with HF_HUB_OFFLINE=1 (hub.py)


class UnknownModelError(Exception):
    def __init__(self, model_id: str):
        self.model_id = model_id
        super().__init__(f"unknown model: {model_id}")


class ModelDisabledError(Exception):
    def __init__(self, model_id: str, reason: str | None):
        self.model_id = model_id
        self.reason = reason or "disabled"
        super().__init__(f"{model_id} is disabled: {self.reason}")


class BudgetExceededError(Exception):
    def __init__(self, model_id: str, needed_mb: float, budget_mb: float):
        self.model_id = model_id
        self.needed_mb = needed_mb
        self.budget_mb = budget_mb
        super().__init__(
            f"{model_id} needs {needed_mb:.0f}MB, budget is {budget_mb:.0f}MB total — "
            "refusing rather than evicting-and-still-not-fitting"
        )


@dataclass
class _Entry:
    model: HookedTransformer
    spec: ModelSpec
    loaded_at: float = field(default_factory=time.time)


class ModelZoo:
    def __init__(
        self,
        registry_path: str | None = None,
        budget_mb: float | None = None,
    ) -> None:
        specs = load_registry(registry_path)
        self._specs: dict[str, ModelSpec] = {s.id: s for s in specs}
        self._resident: OrderedDict[str, _Entry] = OrderedDict()  # LRU order: oldest first
        self.budget_mb = budget_mb if budget_mb is not None else SETTINGS.ram_budget_gb * 1024

    # -- read-only queries, safe to call anytime -------------------------

    def spec(self, model_id: str) -> ModelSpec:
        try:
            return self._specs[model_id]
        except KeyError:
            raise UnknownModelError(model_id) from None

    def list_specs(self) -> list[ModelSpec]:
        return list(self._specs.values())

    def fits(self, spec: ModelSpec) -> bool:
        """Whether this model can be loaded at all under this budget, with
        nothing else resident. One that can't is shown as disabled, so the
        same registry works on a 16 GB server and a bigger one."""
        return spec.load_peak_mb <= self.budget_mb

    def disabled_reason(self, model_id: str) -> str | None:
        spec = self.spec(model_id)
        if spec.tier == "disabled":
            return spec.reason
        if not self.fits(spec):
            return (
                f"needs {spec.load_peak_mb / 1024:.1f} GB while loading; "
                f"this server's model budget is {self.budget_mb / 1024:.1f} GB"
            )
        return None

    def status(self, model_id: str) -> str:
        if self.disabled_reason(model_id) is not None:
            return "disabled"
        return "resident" if model_id in self._resident else "available"

    def resident_ids(self) -> list[str]:
        return list(self._resident.keys())

    @property
    def used_mb(self) -> float:
        return sum(e.spec.est_ram_mb for e in self._resident.values())

    # -- the one mutating operation ---------------------------------------

    def get_or_load(self, model_id: str) -> HookedTransformer:
        spec = self.spec(model_id)
        if spec.tier == "disabled":
            raise ModelDisabledError(model_id, spec.reason)

        if model_id in self._resident:
            # Move to MRU (end of the OrderedDict) without reloading.
            entry = self._resident.pop(model_id)
            self._resident[model_id] = entry
            return entry.model

        if not self.fits(spec):
            raise BudgetExceededError(model_id, spec.load_peak_mb, self.budget_mb)

        # Room for the load's peak, not just what stays resident afterwards.
        while self._resident and self.used_mb + spec.load_peak_mb > self.budget_mb:
            lru_id, _ = next(iter(self._resident.items()))
            self.evict(lru_id)

        if self.used_mb + spec.load_peak_mb > self.budget_mb:
            # Defensive: unreachable given the single-model check above
            # (an empty cache plus a model already confirmed to fit alone
            # cannot fail this), but a loud error here is far preferable
            # to a silent OOM if that invariant is ever broken.
            raise BudgetExceededError(model_id, spec.load_peak_mb, self.budget_mb)

        # Identical to HookedTransformer.from_pretrained (same processing, same
        # weights), but it also records the raw ln_final weights the logit
        # lens lab's plain lens needs before folding discards them (lens.py).
        model = LensReadyTransformer.from_pretrained(
            spec.tl_name, device=SETTINGS.device, dtype=SETTINGS.dtype
        )
        memory.release()  # the raw checkpoint copy is garbage now
        self._resident[model_id] = _Entry(model=model, spec=spec)
        return model

    def evict(self, model_id: str) -> None:
        entry = self._resident.pop(model_id, None)
        if entry is not None:
            del entry.model
            memory.release()

    def evict_lru(self, keep: str | None = None) -> str | None:
        """Evict the least recently used model other than `keep`. Returns its
        id, or None when there was nothing to evict."""
        for model_id in self._resident:
            if model_id != keep:
                self.evict(model_id)
                return model_id
        return None

    def evict_all(self) -> None:
        for model_id in list(self._resident.keys()):
            self.evict(model_id)


if __name__ == "__main__":
    zoo = ModelZoo()
    print(f"budget: {zoo.budget_mb:.0f}MB")
    for spec in zoo.list_specs():
        print(
            f"  {spec.id:20s} tier={spec.tier:8s} status={zoo.status(spec.id):10s} "
            f"est_ram_mb={spec.est_ram_mb:.0f} load_peak_mb={spec.load_peak_mb:.0f}"
        )
