"""
Registry loading and validation. The registry is meant to be a pure-data
change point (docs/PLAN.md: "adding a model must be a data change, never
a code change") — these tests are what make that promise safe to rely on:
a malformed models.yaml entry should fail loudly at load time, not
surface later as a confusing runtime error deep in zoo.py or the API.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from attnlab.registry import ModelSpec, load_registry


def _write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "models.yaml"
    p.write_text(textwrap.dedent(content))
    return p


class TestLoadRealRegistry:
    """Sanity checks against the actual shipped models.yaml."""

    def test_loads_without_error(self):
        specs = load_registry()
        assert len(specs) > 0

    def test_all_ids_unique(self):
        specs = load_registry()
        ids = [s.id for s in specs]
        assert len(ids) == len(set(ids))

    def test_expected_baked_models_present(self):
        """These three are the Stage 0a benchmark matrix's 'baked' tier —
        if this list drifts, bench.py's isolated-mode default (all
        non-disabled models) silently changes scope."""
        specs = {s.id: s for s in load_registry()}
        for expected_id in ("attn-only-2l-demo", "gpt2-small", "pythia-160m"):
            assert expected_id in specs
            assert specs[expected_id].tier == "baked"

    def test_every_spec_has_positive_max_seq_and_ram(self):
        for s in load_registry():
            assert s.max_seq > 0
            assert s.est_ram_mb > 0


class TestValidation:
    def test_rejects_invalid_tier(self, tmp_path):
        p = _write_yaml(
            tmp_path,
            """
            - id: bad
              label: Bad
              tl_name: bad
              tier: not-a-real-tier
              max_seq: 128
              languages: [en]
              est_ram_mb: 10
              blurb: x
              n_layers: 1
              n_heads: 1
              d_model: 8
              n_params: 100
            """,
        )
        with pytest.raises(ValueError, match="invalid tier"):
            load_registry(p)

    def test_disabled_tier_requires_reason(self, tmp_path):
        p = _write_yaml(
            tmp_path,
            """
            - id: bad
              label: Bad
              tl_name: bad
              tier: disabled
              max_seq: 128
              languages: [en]
              est_ram_mb: 10
              blurb: x
              n_layers: 1
              n_heads: 1
              d_model: 8
              n_params: 100
            """,
        )
        with pytest.raises(ValueError, match="reason"):
            load_registry(p)

    def test_disabled_tier_with_reason_is_fine(self, tmp_path):
        p = _write_yaml(
            tmp_path,
            """
            - id: bad
              label: Bad
              tl_name: bad
              tier: disabled
              max_seq: 128
              languages: [en]
              est_ram_mb: 10
              blurb: x
              n_layers: 1
              n_heads: 1
              d_model: 8
              n_params: 100
              reason: "too large for the free tier"
            """,
        )
        specs = load_registry(p)
        assert specs[0].reason == "too large for the free tier"

    def test_rejects_duplicate_ids(self, tmp_path):
        p = _write_yaml(
            tmp_path,
            """
            - id: dup
              label: A
              tl_name: dup
              tier: baked
              max_seq: 128
              languages: [en]
              est_ram_mb: 10
              blurb: x
              n_layers: 1
              n_heads: 1
              d_model: 8
              n_params: 100
            - id: dup
              label: B
              tl_name: dup
              tier: baked
              max_seq: 128
              languages: [en]
              est_ram_mb: 10
              blurb: y
              n_layers: 1
              n_heads: 1
              d_model: 8
              n_params: 100
            """,
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_registry(p)

    def test_rejects_non_list_yaml(self, tmp_path):
        p = _write_yaml(tmp_path, "not_a_list: true")
        with pytest.raises(ValueError, match="expected a YAML list"):
            load_registry(p)


def test_model_spec_is_frozen():
    spec = ModelSpec(
        id="x", label="X", tl_name="x", tier="baked", max_seq=1, languages=[], est_ram_mb=1,
        blurb="", n_layers=1, n_heads=1, d_model=8, n_params=100,
    )
    with pytest.raises(Exception):
        spec.id = "y"  # type: ignore[misc]
