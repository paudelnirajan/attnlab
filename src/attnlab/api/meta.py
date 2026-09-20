"""Builds the `_meta` block documented in docs/02-api.md — one shared
function so every endpoint's `_meta` has the same shape, sourced from the
same Stage 0a instrumentation record (attnlab.instrument.Measurement)."""

from __future__ import annotations

import importlib.metadata
from typing import Any

from attnlab.instrument import Measurement
from attnlab.settings import SETTINGS

TL_VERSION = importlib.metadata.version("transformer-lens")

_DEBUG_FIELDS = (
    "rss_delta_mb",
    "peak_rss_mb",
    "cgroup_current_mb",
    "cgroup_peak_mb",
    "cpu_throttled_us",
    "threads",
)


def build_meta(measurement: Measurement) -> dict[str, Any]:
    rec = measurement.to_record()
    meta: dict[str, Any] = {
        "op": rec["op"],
        "duration_ms": round(rec["duration_ms"], 2),
        "tl_version": TL_VERSION,
        "device": SETTINGS.device,
        "dtype": "float32",
    }
    for key in ("model", "seq", "bytes_out"):
        if rec.get(key) is not None:
            meta[key] = rec[key]

    # Process metrics (RSS, cgroup, CPU throttling, thread count) are
    # opt-in: they're useful for the ?debug=1 dev overlay (docs/PLAN.md
    # Stage 1.5) but meaningless to an ordinary user and a small amount
    # of capacity information leaked to the public otherwise.
    if SETTINGS.debug_metrics:
        for key in _DEBUG_FIELDS:
            if key in rec:
                meta[key] = rec[key]

    return meta
