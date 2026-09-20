"""
The attention wire format: extract -> sqrt-compand -> quantize -> pack ->
encode. Implements docs/01-wire-format.md exactly; that document is the
source of truth for *why* each step exists — read it before changing this
file. In short:

  - Naive float32 attention for gpt2-small at seq=512 is 151 MB. We ship
    one layer at a time, quantized, causally packed: under 2 MB.
  - Linear uint8 quantizes every probability below ~0.002 to zero, which
    is invisible on a linear heatmap but destroys the log-scale view
    researchers actually use. sqrt companding (`u = round(255*sqrt(p))`)
    fixes this for the same one byte: smallest representable value goes
    from 3.9e-3 to 1.5e-5.
  - Decoder-only attention is causally masked (source > dest is always
    exactly 0), so we store only the lower triangle: seq*(seq+1)/2 values
    instead of seq*seq — just under half the bytes, for free.
"""

from __future__ import annotations

import dataclasses
import gzip
import struct

import numpy as np

MAGIC = b"ATNP"
VERSION = 1

FLAG_TRIANGLE = 1 << 0
FLAG_UINT8 = 1 << 1
FLAG_SQRT_COMPANDED = 1 << 2

DEFAULT_FLAGS = FLAG_TRIANGLE | FLAG_UINT8 | FLAG_SQRT_COMPANDED

_HEADER_FMT = "<4sBBHHHI"  # magic, version, flags, reserved(u16), n_layers, n_heads, seq
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
assert _HEADER_SIZE == 16, _HEADER_SIZE


def compand_encode(p: np.ndarray) -> np.ndarray:
    """p in [0,1] float -> uint8, via u = round(255 * sqrt(p))."""
    return np.round(255.0 * np.sqrt(np.clip(p, 0.0, 1.0))).astype(np.uint8)


def compand_decode(u: np.ndarray) -> np.ndarray:
    """uint8 -> p in [0,1] float, via p = (u/255)^2."""
    v = u.astype(np.float32) / 255.0
    return v * v


def triangle_pack(square: np.ndarray) -> np.ndarray:
    """(seq, seq) -> 1D array of seq*(seq+1)/2 values: row i contributes
    columns 0..i inclusive (the causal lower triangle, diagonal included).
    """
    seq = square.shape[-1]
    rows, cols = np.tril_indices(seq)
    return square[..., rows, cols]


def triangle_unpack(packed: np.ndarray, seq: int) -> np.ndarray:
    """Inverse of triangle_pack: fills a (seq, seq) array, zeros above the
    diagonal."""
    square = np.zeros((seq, seq), dtype=packed.dtype)
    rows, cols = np.tril_indices(seq)
    square[rows, cols] = packed
    return square


@dataclasses.dataclass
class EncodedLayer:
    layer_id: int
    payload: bytes  # concatenated per-head packed+quantized values


def encode_layers(
    pattern_by_layer: dict[int, np.ndarray],
    *,
    flags: int = DEFAULT_FLAGS,
) -> bytes:
    """
    pattern_by_layer: {layer_id: array of shape (n_heads, seq, seq)}, values
    in [0, 1]. All layers must share n_heads and seq (true within one model
    / one run).

    Returns the full ATNP-framed byte string, uncompressed. Gzip is applied
    by the HTTP layer (Stage 0b), not here — keeping this function pure
    bytes-in/bytes-out makes it trivial to unit test and to reuse for the
    benchmark's size report.
    """
    if not pattern_by_layer:
        raise ValueError("encode_layers: no layers given")

    layer_ids = sorted(pattern_by_layer)
    first = pattern_by_layer[layer_ids[0]]
    n_heads, seq, seq2 = first.shape
    if seq != seq2:
        raise ValueError(f"expected square attention, got {first.shape}")

    packed = bool(flags & FLAG_TRIANGLE)
    is_u8 = bool(flags & FLAG_UINT8)
    sqrt_comp = bool(flags & FLAG_SQRT_COMPANDED)

    header = struct.pack(
        _HEADER_FMT, MAGIC, VERSION, flags, 0, len(layer_ids), n_heads, seq
    )
    layer_id_bytes = b"".join(struct.pack("<H", lid) for lid in layer_ids)

    chunks = [header, layer_id_bytes]
    for lid in layer_ids:
        arr = pattern_by_layer[lid]
        if arr.shape != (n_heads, seq, seq):
            raise ValueError(f"layer {lid}: shape {arr.shape} != {(n_heads, seq, seq)}")
        for h in range(n_heads):
            head = arr[h]
            if sqrt_comp:
                head = compand_encode(head) if is_u8 else _compand_encode_u16(head)
            elif is_u8:
                head = np.round(np.clip(head, 0, 1) * 255).astype(np.uint8)
            else:
                head = np.round(np.clip(head, 0, 1) * 65535).astype(np.uint16)
            if packed:
                head = triangle_pack(head)
            chunks.append(head.tobytes())

    return b"".join(chunks)


def _compand_encode_u16(p: np.ndarray) -> np.ndarray:
    return np.round(65535.0 * np.sqrt(np.clip(p, 0.0, 1.0))).astype(np.uint16)


def _compand_decode_u16(u: np.ndarray) -> np.ndarray:
    v = u.astype(np.float32) / 65535.0
    return v * v


@dataclasses.dataclass
class DecodedPatterns:
    layer_ids: list[int]
    n_heads: int
    seq: int
    packed: bool
    arrays: dict[int, np.ndarray]  # layer_id -> (n_heads, seq, seq) float32 in [0,1]


def decode_layers(buf: bytes) -> DecodedPatterns:
    """Inverse of encode_layers. Used for round-trip tests and for a pure
    -Python reference the TypeScript decoder can be checked against."""
    magic, version, flags, _reserved, n_layers, n_heads, seq = struct.unpack(
        _HEADER_FMT, buf[:_HEADER_SIZE]
    )
    if magic != MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported version: {version}")

    packed = bool(flags & FLAG_TRIANGLE)
    is_u8 = bool(flags & FLAG_UINT8)
    sqrt_comp = bool(flags & FLAG_SQRT_COMPANDED)

    offset = _HEADER_SIZE
    layer_ids = []
    for _ in range(n_layers):
        (lid,) = struct.unpack_from("<H", buf, offset)
        layer_ids.append(lid)
        offset += 2

    dtype = np.uint8 if is_u8 else np.uint16
    itemsize = 1 if is_u8 else 2
    per_head_count = seq * (seq + 1) // 2 if packed else seq * seq

    arrays: dict[int, np.ndarray] = {}
    for lid in layer_ids:
        heads = []
        for _h in range(n_heads):
            n_bytes = per_head_count * itemsize
            raw = np.frombuffer(buf, dtype=dtype, count=per_head_count, offset=offset)
            offset += n_bytes
            square = triangle_unpack(raw, seq) if packed else raw.reshape(seq, seq)
            decoded = (
                (compand_decode(square) if is_u8 else _compand_decode_u16(square))
                if sqrt_comp
                else square.astype(np.float32) / (255.0 if is_u8 else 65535.0)
            )
            heads.append(decoded)
        arrays[lid] = np.stack(heads, axis=0)

    return DecodedPatterns(
        layer_ids=layer_ids, n_heads=n_heads, seq=seq, packed=packed, arrays=arrays
    )


# --------------------------------------------------------------------------
# Size reporting for the Stage 0a benchmark: how many bytes does the SAME
# tensor cost at each stage of the pipeline, and after gzip?
# --------------------------------------------------------------------------


@dataclasses.dataclass
class SizeReport:
    n_layers: int
    n_heads: int
    seq: int
    float32_bytes: int
    float16_bytes: int
    uint8_square_bytes: int
    uint8_triangle_bytes: int
    uint8_triangle_gzip_bytes: int
    per_layer_uint8_triangle_bytes: int
    per_layer_uint8_triangle_gzip_bytes: int


def size_report(full_pattern: np.ndarray) -> SizeReport:
    """full_pattern: (n_layers, n_heads, seq, seq) float32 in [0,1], the
    complete cached attention for one run. Reports what the SAME data
    costs at every stage of the wire-format pipeline, so the benchmark can
    validate (or correct) the arithmetic in docs/01-wire-format.md against
    real model output rather than hand-computed estimates."""
    n_layers, n_heads, seq, _ = full_pattern.shape
    n_values = n_layers * n_heads * seq * seq
    n_triangle_values = n_layers * n_heads * seq * (seq + 1) // 2

    full_bytes = encode_layers(
        {i: full_pattern[i] for i in range(n_layers)}, flags=DEFAULT_FLAGS
    )
    one_layer_bytes = encode_layers({0: full_pattern[0]}, flags=DEFAULT_FLAGS)

    return SizeReport(
        n_layers=n_layers,
        n_heads=n_heads,
        seq=seq,
        float32_bytes=n_values * 4,
        float16_bytes=n_values * 2,
        uint8_square_bytes=n_values * 1,
        uint8_triangle_bytes=n_triangle_values * 1,
        uint8_triangle_gzip_bytes=len(gzip.compress(full_bytes, compresslevel=6)),
        per_layer_uint8_triangle_bytes=len(one_layer_bytes),
        per_layer_uint8_triangle_gzip_bytes=len(gzip.compress(one_layer_bytes, compresslevel=6)),
    )


if __name__ == "__main__":
    # Smoke test with synthetic (non-model) data: a plausible attention-like
    # tensor — mostly small values with a sharp diagonal, causally masked.
    rng = np.random.default_rng(0)
    seq = 64
    n_layers, n_heads = 2, 4
    raw = rng.exponential(scale=0.05, size=(n_layers, n_heads, seq, seq)).astype(np.float32)
    causal_mask = np.triu(np.ones((seq, seq), dtype=bool), k=1)
    raw[:, :, causal_mask] = 0.0
    raw = raw / raw.sum(axis=-1, keepdims=True).clip(min=1e-9)

    encoded = encode_layers({i: raw[i] for i in range(n_layers)})
    decoded = decode_layers(encoded)

    max_abs_err = 0.0
    max_rel_err_above_1pct = 0.0
    for lid, arr in decoded.arrays.items():
        orig = raw[lid]
        abs_err = np.abs(arr - orig)
        max_abs_err = max(max_abs_err, float(abs_err.max()))
        mask = orig >= 0.01
        if mask.any():
            rel_err = (abs_err[mask] / orig[mask]).max()
            max_rel_err_above_1pct = max(max_rel_err_above_1pct, float(rel_err))

    report = size_report(raw)
    print(f"n_layers={n_layers} n_heads={n_heads} seq={seq}")
    print(f"round-trip max abs error:              {max_abs_err:.5f}  (spec: <= 0.004)")
    print(f"round-trip max rel error (p>=0.01):     {max_rel_err_above_1pct:.4f}  (spec: <= 0.04)")
    print(f"encoded size (all layers):              {len(encoded):,} bytes")
    print(f"float32 equivalent:                     {report.float32_bytes:,} bytes")
    print(f"per-layer (u8+triangle):                {report.per_layer_uint8_triangle_bytes:,} bytes")
    print(f"per-layer (u8+triangle+gzip):            {report.per_layer_uint8_triangle_gzip_bytes:,} bytes")

    assert max_abs_err <= 0.004 + 1e-9, "FAILED spec bound: absolute error"
    assert max_rel_err_above_1pct <= 0.04 + 1e-9, "FAILED spec bound: relative error"
    print("OK — within docs/01-wire-format.md acceptance bounds")
