"""
LRU/budget policy tests, mocking HookedTransformer.from_pretrained so
eviction logic runs instantly without loading real weights or touching
the network. One real integration test at the bottom proves the actual
wiring works end to end.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from attnlab.zoo import BudgetExceededError, ModelDisabledError, ModelZoo, UnknownModelError


def _write_registry(tmp_path: Path, *entries: str) -> Path:
    p = tmp_path / "models.yaml"
    p.write_text(textwrap.dedent("\n".join(entries)))
    return p


def _entry(id_: str, ram_mb: float, tier: str = "baked", reason: str = "") -> str:
    # Built as a flat list of unindented lines rather than a dedented
    # f-string: an interpolated reason_line with different leading
    # whitespace than its siblings breaks textwrap.dedent's "common
    # leading whitespace" calculation, silently producing invalid YAML
    # (caught the hard way — this comment is here so it isn't
    # reintroduced).
    lines = [
        f"- id: {id_}",
        f'  label: "{id_}"',
        f"  tl_name: {id_}",
        f"  tier: {tier}",
        "  max_seq: 512",
        "  languages: [en]",
        f"  est_ram_mb: {ram_mb}",
        "  n_layers: 1",
        "  n_heads: 1",
        "  d_model: 8",
        "  n_params: 100",
        '  blurb: "test model"',
    ]
    if tier == "disabled":
        lines.append(f'  reason: "{reason}"')
    return "\n".join(lines)


@pytest.fixture
def mock_load(monkeypatch):
    """Each call returns a distinct sentinel object standing in for a
    loaded HookedTransformer, so identity (`is`) checks confirm whether a
    real reload happened vs. a cache hit."""
    calls: list[str] = []

    def fake_from_pretrained(tl_name, device=None, dtype=None):
        calls.append(tl_name)
        return object()  # unique sentinel per call

    monkeypatch.setattr("attnlab.zoo.HookedTransformer.from_pretrained", fake_from_pretrained)
    return calls


class TestBasicLoadAndCache:
    def test_load_then_cache_hit_does_not_reload(self, tmp_path, mock_load):
        p = _write_registry(tmp_path, _entry("a", 100))
        zoo = ModelZoo(registry_path=p, budget_mb=1000)

        m1 = zoo.get_or_load("a")
        m2 = zoo.get_or_load("a")

        assert m1 is m2
        assert mock_load == ["a"]  # loaded exactly once
        assert zoo.status("a") == "resident"
        assert zoo.used_mb == 100

    def test_unknown_model_raises(self, tmp_path, mock_load):
        p = _write_registry(tmp_path, _entry("a", 100))
        zoo = ModelZoo(registry_path=p, budget_mb=1000)
        with pytest.raises(UnknownModelError):
            zoo.get_or_load("does-not-exist")

    def test_disabled_model_raises_with_reason(self, tmp_path, mock_load):
        p = _write_registry(tmp_path, _entry("a", 100, tier="disabled", reason="too big"))
        zoo = ModelZoo(registry_path=p, budget_mb=1000)
        with pytest.raises(ModelDisabledError, match="too big"):
            zoo.get_or_load("a")
        assert mock_load == []  # never attempted a load


class TestBudgetEnforcement:
    def test_single_model_over_budget_is_refused_not_evicted(self, tmp_path, mock_load):
        """D6: a model that alone exceeds the budget must be refused
        outright — there is no eviction that could ever make it fit."""
        p = _write_registry(tmp_path, _entry("huge", 2000))
        zoo = ModelZoo(registry_path=p, budget_mb=1000)
        with pytest.raises(BudgetExceededError) as exc_info:
            zoo.get_or_load("huge")
        assert exc_info.value.needed_mb == 2000
        assert exc_info.value.budget_mb == 1000
        assert mock_load == []  # refused before ever attempting to load

    def test_eviction_makes_room_for_a_new_model(self, tmp_path, mock_load):
        p = _write_registry(tmp_path, _entry("a", 600), _entry("b", 600))
        zoo = ModelZoo(registry_path=p, budget_mb=1000)

        zoo.get_or_load("a")
        assert zoo.resident_ids() == ["a"]

        zoo.get_or_load("b")  # a (600) + b (600) = 1200 > 1000 -> a must evict
        assert zoo.resident_ids() == ["b"]
        assert zoo.used_mb == 600

    def test_lru_order_evicts_least_recently_used_first(self, tmp_path, mock_load):
        p = _write_registry(tmp_path, _entry("a", 300), _entry("b", 300), _entry("c", 500))
        zoo = ModelZoo(registry_path=p, budget_mb=700)

        zoo.get_or_load("a")
        zoo.get_or_load("b")
        assert zoo.resident_ids() == ["a", "b"]  # both fit: 300+300=600 <= 700

        zoo.get_or_load("a")  # touch a -> a becomes MRU, b is now LRU
        assert zoo.resident_ids() == ["b", "a"]

        zoo.get_or_load("c")  # needs 500; must evict b (LRU) first, then a
        # 300(a) + 500(c) = 800 > 700, so a ALSO gets evicted
        assert zoo.resident_ids() == ["c"]

    def test_refused_admission_does_not_evict_anything(self, tmp_path, mock_load):
        """A model that alone exceeds the budget is refused BEFORE the
        eviction loop runs at all (D6's single-model check) — so an
        already-resident model must survive someone else's failed
        request untouched, not get evicted for nothing."""
        p = _write_registry(tmp_path, _entry("a", 100), _entry("huge", 1200))
        zoo = ModelZoo(registry_path=p, budget_mb=1000)
        zoo.get_or_load("a")
        with pytest.raises(BudgetExceededError):
            zoo.get_or_load("huge")
        assert zoo.resident_ids() == ["a"]
        assert zoo.used_mb == 100

    def test_evict_and_evict_all(self, tmp_path, mock_load):
        p = _write_registry(tmp_path, _entry("a", 100), _entry("b", 100))
        zoo = ModelZoo(registry_path=p, budget_mb=1000)
        zoo.get_or_load("a")
        zoo.get_or_load("b")
        assert zoo.used_mb == 200

        zoo.evict("a")
        assert zoo.resident_ids() == ["b"]
        assert zoo.used_mb == 100

        zoo.evict_all()
        assert zoo.resident_ids() == []
        assert zoo.used_mb == 0

    def test_evict_unknown_model_is_a_noop(self, tmp_path, mock_load):
        p = _write_registry(tmp_path, _entry("a", 100))
        zoo = ModelZoo(registry_path=p, budget_mb=1000)
        zoo.evict("never-loaded")  # must not raise


class TestRealIntegration:
    """One test with NO mocking, against the real shipped registry and a
    real (tiny) model — proves get_or_load actually works end to end, not
    just that the policy logic is internally consistent."""

    def test_loads_real_tiny_model(self):
        zoo = ModelZoo()  # real models.yaml, real budget from SETTINGS
        model = zoo.get_or_load("attn-only-2l-demo")
        assert model.cfg.n_layers == 2
        assert zoo.status("attn-only-2l-demo") == "resident"
        zoo.evict_all()  # clean up so this process doesn't hold it resident
