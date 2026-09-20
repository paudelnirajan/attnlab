"""
Enforces the acceptance criteria from docs/01-wire-format.md exactly.
These are the numbers that make the whole project's data-volume story
true — if these regress, the "sub-megabyte per layer" claim in
FEASIBILITY.md and PLAN.md stops being true.
"""

from __future__ import annotations

import numpy as np
import pytest

from attnlab.patterns import (
    DEFAULT_FLAGS,
    FLAG_SQRT_COMPANDED,
    FLAG_TRIANGLE,
    FLAG_UINT8,
    compand_decode,
    compand_encode,
    decode_layers,
    encode_layers,
    triangle_pack,
    triangle_unpack,
)


def _causal_attention(seq: int, n_heads: int = 4, seed: int = 0) -> np.ndarray:
    """A plausible attention-like tensor: mostly small values, a sharp
    diagonal, causally masked, rows summing to 1 — the shape real
    HookedTransformer output has, without needing a model loaded."""
    rng = np.random.default_rng(seed)
    raw = rng.exponential(scale=0.05, size=(n_heads, seq, seq)).astype(np.float32)
    raw[:, np.arange(seq), np.arange(seq)] += 2.0  # bias toward the diagonal
    causal_mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
    raw[:, causal_mask] = 0.0
    return raw / raw.sum(axis=-1, keepdims=True).clip(min=1e-9)


class TestCompanding:
    def test_round_trip_zero(self):
        z = np.zeros((3, 5), dtype=np.float32)
        assert np.array_equal(compand_decode(compand_encode(z)), z)

    def test_round_trip_one(self):
        o = np.ones((3, 5), dtype=np.float32)
        decoded = compand_decode(compand_encode(o))
        np.testing.assert_allclose(decoded, o, atol=1e-6)

    def test_smallest_nonzero_representable(self):
        """docs/01-wire-format.md claims sqrt companding's smallest
        representable value is ~1.5e-5, vs. linear uint8's 3.9e-3."""
        smallest_u8 = 1
        p = compand_decode(np.array([smallest_u8], dtype=np.uint8))[0]
        assert p < 2e-5, f"expected ~1.5e-5, got {p}"
        assert p > 0

    @pytest.mark.parametrize("p_value", [0.01, 0.05, 0.1, 0.3, 0.5, 0.9, 1.0])
    def test_relative_error_bound_above_1pct(self, p_value):
        """Spec bound: relative error <= 4% for p >= 0.01."""
        p = np.full(1000, p_value, dtype=np.float32)
        decoded = compand_decode(compand_encode(p))
        rel_err = np.abs(decoded - p) / p
        assert rel_err.max() <= 0.04 + 1e-9, f"p={p_value}: rel_err={rel_err.max()}"

    def test_absolute_error_bound_everywhere(self):
        """Spec bound: absolute error <= 0.004 for all p, including near 0
        and near 1 where sqrt companding is weakest in absolute terms."""
        p = np.linspace(0.0, 1.0, 100_000, dtype=np.float32)
        decoded = compand_decode(compand_encode(p))
        assert np.abs(decoded - p).max() <= 0.004 + 1e-9


class TestTrianglePacking:
    def test_pack_unpack_round_trip(self):
        seq = 17
        square = np.arange(seq * seq, dtype=np.uint8).reshape(seq, seq)
        packed = triangle_pack(square)
        assert packed.shape == (seq * (seq + 1) // 2,)
        unpacked = triangle_unpack(packed, seq)
        lower = np.tril(square)
        assert np.array_equal(unpacked, lower)

    def test_above_diagonal_is_zero_after_unpack(self):
        seq = 8
        square = np.full((seq, seq), 255, dtype=np.uint8)
        unpacked = triangle_unpack(triangle_pack(square), seq)
        upper = np.triu(unpacked, k=1)
        assert np.all(upper == 0)


class TestEncodeDecodeLayers:
    def test_round_trip_shape_and_ids(self):
        seq, n_heads = 32, 4
        raw = {0: _causal_attention(seq, n_heads, seed=1), 3: _causal_attention(seq, n_heads, seed=2)}
        encoded = encode_layers(raw)
        decoded = decode_layers(encoded)
        assert decoded.layer_ids == [0, 3]
        assert decoded.n_heads == n_heads
        assert decoded.seq == seq
        assert decoded.packed is True

    def test_round_trip_error_bounds_on_realistic_data(self):
        """The end-to-end pipeline, on data shaped like real model output,
        must satisfy BOTH spec bounds simultaneously — this is the test
        that would have caught the linear-vs-sqrt bound confusion found
        during Stage 0a (see docs/03-decisions.md changelog)."""
        seq, n_heads, n_layers = 64, 8, 3
        raw = {i: _causal_attention(seq, n_heads, seed=i) for i in range(n_layers)}
        decoded = decode_layers(encode_layers(raw))

        for i in range(n_layers):
            orig, arr = raw[i], decoded.arrays[i]
            abs_err = np.abs(arr - orig)
            assert abs_err.max() <= 0.004 + 1e-9

            mask = orig >= 0.01
            rel_err = (abs_err[mask] / orig[mask]).max()
            assert rel_err <= 0.04 + 1e-9

    def test_causal_zeros_stay_exactly_zero(self):
        """Values above the diagonal must round-trip to EXACTLY 0.0, not
        just approximately — a heatmap showing a faint upper triangle
        would look like a bug to anyone who knows these are causal models."""
        seq = 20
        raw = {0: _causal_attention(seq, n_heads=2, seed=5)}
        decoded = decode_layers(encode_layers(raw))
        upper = np.triu(decoded.arrays[0], k=1)
        assert np.all(upper == 0.0)

    def test_rejects_unknown_version(self):
        raw = {0: _causal_attention(8, n_heads=2)}
        encoded = bytearray(encode_layers(raw))
        encoded[4] = 99  # version byte
        with pytest.raises(ValueError, match="version"):
            decode_layers(bytes(encoded))

    def test_rejects_bad_magic(self):
        raw = {0: _causal_attention(8, n_heads=2)}
        encoded = bytearray(encode_layers(raw))
        encoded[0:4] = b"XXXX"
        with pytest.raises(ValueError, match="magic"):
            decode_layers(bytes(encoded))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            encode_layers({})

    def test_default_flags_are_triangle_u8_sqrt(self):
        assert DEFAULT_FLAGS == (FLAG_TRIANGLE | FLAG_UINT8 | FLAG_SQRT_COMPANDED)


class TestSizeMath:
    """Validates the byte-count claims in docs/01-wire-format.md's worked
    example against the real encoder, not hand arithmetic."""

    def test_per_layer_gpt2_small_seq256_is_sub_megabyte(self):
        # gpt2-small: 12 heads. One layer, causally packed, seq=256.
        raw = {0: _causal_attention(256, n_heads=12, seed=0)}
        encoded = encode_layers(raw)
        # spec table: 394,752 bytes for one layer at seq=256, 12 heads
        expected = 16 + 2 + 12 * (256 * 257 // 2)  # header + 1 layer id + triangle values
        assert len(encoded) == expected
        assert len(encoded) < 1_000_000, "per-layer payload must stay sub-megabyte through seq=256"
