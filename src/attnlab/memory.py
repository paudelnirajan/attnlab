"""
How much physical memory this process is using right now, and handing freed
memory back to the OS. The server's memory guard (api/state.py) acts on these.

Why "physical footprint" and not RSS. On macOS the footprint is what Activity
Monitor's Memory column shows and what the OS weighs under memory pressure. It
counts compressed and swapped-out pages the process still owns, and Metal (MPS)
buffers, none of which RSS includes. On Linux, RSS from /proc is the usual
equivalent.

Why `MallocLargeCache=0` matters (D17). macOS's allocator keeps large freed
blocks mapped for reuse, and they still count toward the footprint. Loading a
model briefly holds the raw weights and the processed copy side by side, so the
cache ends up holding a whole model's worth of memory the process no longer
uses: measured on Qwen3-0.6B, 6.4 GB with the cache vs. 3.4 GB without. The
variable is read once when the process starts, so it has to be in the
environment before Python starts (deploy/run-api.sh sets it).
"""

from __future__ import annotations

import ctypes
import gc
import os
import sys
from functools import cache

_RUSAGE_INFO_V4 = 4


class _RUsageInfo(ctypes.Structure):
    # rusage_info_v4 from <sys/resource.h>, up to the fields read here. The
    # kernel writes the whole struct, so the tail is padding big enough for it.
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins",
            "wired_size", "resident_size", "phys_footprint", "proc_start_abstime",
            "proc_exit_abstime", "child_user_time", "child_system_time",
            "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins",
            "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten",
            "cpu_time_qos_default", "cpu_time_qos_maintenance", "cpu_time_qos_background",
            "cpu_time_qos_utility", "cpu_time_qos_legacy", "cpu_time_qos_user_initiated",
            "cpu_time_qos_user_interactive", "billed_system_time", "serviced_system_time",
            "logical_writes", "lifetime_max_phys_footprint",
        )
    ] + [("_pad", ctypes.c_uint64 * 64)]


@cache
def _libproc():
    lib = ctypes.CDLL("/usr/lib/libproc.dylib")
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    return lib


def _darwin_rusage() -> _RUsageInfo | None:
    info = _RUsageInfo()
    try:
        if _libproc().proc_pid_rusage(os.getpid(), _RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
            return None
    except OSError:
        return None
    return info


def _linux_rss_mb() -> float | None:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        pass
    return None


def footprint_mb() -> float | None:
    """Physical memory this process is using now, in MB, or None if unknown."""
    if sys.platform == "darwin":
        info = _darwin_rusage()
        return info.phys_footprint / 2**20 if info else None
    return _linux_rss_mb()


def peak_footprint_mb() -> float | None:
    """The most this process has ever used (macOS only)."""
    if sys.platform == "darwin":
        info = _darwin_rusage()
        return info.lifetime_max_phys_footprint / 2**20 if info else None
    return None


def large_cache_disabled() -> bool:
    """Whether the allocator setting above is in effect (always True off macOS,
    where it doesn't apply)."""
    return sys.platform != "darwin" or os.environ.get("MallocLargeCache") == "0"


def release() -> None:
    """Collect garbage and ask the allocator to return free pages to the OS.
    Cheap, so it runs after every eviction."""
    gc.collect()
    if sys.platform == "darwin":
        try:
            libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib")
            libc.malloc_zone_pressure_relief.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            libc.malloc_zone_pressure_relief(None, 0)
        except (OSError, AttributeError):
            pass
    else:
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass
