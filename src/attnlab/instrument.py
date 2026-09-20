"""
Measurement context manager. One record shape, three consumers:
  1. the Stage 0a benchmark table (bench.py)
  2. the `_meta` block on every API response (Stage 0b+)
  3. the `?debug=1` dev overlay (Stage 1.5)

Two correctness traps this module exists to avoid — both would silently
corrupt every number downstream if missed:

  - `resource.getrusage().ru_maxrss` is **bytes on macOS, kilobytes on
    Linux**. Get this wrong and every native (Mac) measurement is 1024x
    off, in a direction that looks plausible (too large) rather than
    obviously broken.

  - `psutil.virtual_memory()` / `/proc/meminfo` report the Docker VM's
    memory, not the cgroup limit the container is actually held to. It
    will say "plenty of headroom" right up until the OOM killer fires.
    Real containers use cgroup v2; we read `/sys/fs/cgroup/*` directly and
    return None for these fields when not running under cgroup v2 (e.g.
    natively on macOS) rather than fabricate a number.
"""

from __future__ import annotations

import contextlib
import dataclasses
import platform
import resource
import threading
import time
from pathlib import Path
from typing import Any, Generator

_CGROUP = Path("/sys/fs/cgroup")
_IS_LINUX = platform.system() == "Linux"


def _read_cgroup_int(name: str) -> int | None:
    """Read a cgroup v2 file that holds a single integer, or 'max'."""
    path = _CGROUP / name
    try:
        raw = path.read_text().strip()
    except (FileNotFoundError, PermissionError):
        return None
    if raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _read_cpu_stat() -> dict[str, int]:
    """cpu.stat is a key-value file: 'usage_usec 123\\nthrottled_usec 0\\n...'"""
    path = _CGROUP / "cpu.stat"
    try:
        lines = path.read_text().splitlines()
    except (FileNotFoundError, PermissionError):
        return {}
    out: dict[str, int] = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            out[parts[0]] = int(parts[1])
    return out


def _ru_maxrss_mb() -> float:
    """Normalize the platform-dependent unit of ru_maxrss to MB."""
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 if _IS_LINUX else 1024.0 * 1024.0  # KB on Linux, bytes on macOS
    return raw / divisor


class _PeakSampler:
    """Background thread sampling ru_maxrss (monotonic high-water mark) and
    cgroup memory.current at a fixed interval, so we catch a peak that
    occurs and releases *within* a single measured operation — a single
    before/after snapshot would miss that entirely."""

    def __init__(self, interval_s: float = 0.02) -> None:
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_rss_mb = 0.0
        self.peak_cgroup_mb: float | None = None

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak_rss_mb = max(self.peak_rss_mb, _ru_maxrss_mb())
            cur = _read_cgroup_int("memory.current")
            if cur is not None:
                cur_mb = cur / 1024**2
                self.peak_cgroup_mb = max(self.peak_cgroup_mb or 0.0, cur_mb)
            self._stop.wait(self.interval_s)

    def __enter__(self) -> "_PeakSampler":
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)


@dataclasses.dataclass
class Measurement:
    op: str
    tags: dict[str, Any]
    duration_ms: float = 0.0
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    rss_delta_mb: float = 0.0
    peak_rss_mb: float = 0.0
    cgroup_current_mb: float | None = None
    cgroup_max_mb: float | None = None
    cgroup_peak_mb: float | None = None
    cpu_throttled_us: int | None = None
    threads: int = 0
    bytes_out: int | None = None

    def to_record(self) -> dict[str, Any]:
        """The one shape shared by the benchmark table, API `_meta`, and
        the debug overlay."""
        d = dataclasses.asdict(self)
        d.update(d.pop("tags"))
        return d


@contextlib.contextmanager
def measure(op: str, *, bytes_out: int | None = None, **tags: Any) -> Generator[Measurement]:
    """
    with measure("forward+cache", model="gpt2-small", seq=256) as m:
        ...
    # m.duration_ms, m.rss_delta_mb, m.peak_rss_mb, m.cgroup_peak_mb,
    # m.cpu_throttled_us are populated on exit.
    """
    import torch

    m = Measurement(op=op, tags=tags, threads=torch.get_num_threads())
    m.rss_before_mb = _ru_maxrss_mb()
    m.cgroup_current_mb = (
        cur / 1024**2 if (cur := _read_cgroup_int("memory.current")) is not None else None
    )
    m.cgroup_max_mb = (
        mx / 1024**2 if (mx := _read_cgroup_int("memory.max")) is not None else None
    )
    cpu_before = _read_cpu_stat()
    t0 = time.perf_counter()

    with _PeakSampler() as sampler:
        try:
            yield m
        finally:
            m.duration_ms = (time.perf_counter() - t0) * 1000.0
            m.rss_after_mb = _ru_maxrss_mb()
            m.rss_delta_mb = m.rss_after_mb - m.rss_before_mb
            m.peak_rss_mb = max(sampler.peak_rss_mb, m.rss_after_mb)
            m.cgroup_peak_mb = sampler.peak_cgroup_mb
            cpu_after = _read_cpu_stat()
            if "throttled_usec" in cpu_before and "throttled_usec" in cpu_after:
                m.cpu_throttled_us = cpu_after["throttled_usec"] - cpu_before["throttled_usec"]
            if bytes_out is not None:
                m.bytes_out = bytes_out


if __name__ == "__main__":
    import json

    from attnlab.settings import SETTINGS  # noqa: F401  (applies thread limits on import)

    with measure("smoke_test", note="just sleeping and allocating") as m:
        _ = bytearray(50 * 1024 * 1024)  # 50 MB, to see rss_delta move
        time.sleep(0.05)
    print(json.dumps(m.to_record(), indent=2))
