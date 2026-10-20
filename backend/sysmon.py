"""Lightweight realtime system monitoring for the Z-Image-Turbo web UI.

Deliberately cheap: the endpoint is polled ~1/s, so every call must be fast and
must not spawn a subprocess if it can be avoided.

GPU  -> NVML (nvidia-ml-py) for utilisation, memory, temperature, power, clocks.
CPU  -> /proc/stat deltas for the busy percentage (no psutil dependency).
RAM  -> /proc/meminfo (MemAvailable is the honest number on a zram box).
Disk -> statvfs on the output directory.
"""

import os
import time
from pathlib import Path
from typing import Any

try:
    import pynvml

    pynvml.nvmlInit()
    _GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
    _GPU_NAME = pynvml.nvmlDeviceGetName(_GPU_HANDLE)
    _GPU_OK = True
except Exception:  # pragma: no cover - NVML missing or no GPU
    pynvml = None  # type: ignore[assignment]
    _GPU_HANDLE = None
    _GPU_NAME = "unknown"
    _GPU_OK = False


def _read_proc_stat() -> tuple[int, int] | None:
    """Return (busy_ticks, total_ticks) from /proc/stat's cpu line."""
    try:
        with open("/proc/stat") as f:
            for line in f:
                if line.startswith("cpu "):
                    parts = [int(x) for x in line.split()[1:]]
                    idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
                    return sum(parts), idle
    except OSError:
        pass
    return None


def _read_per_cpu() -> list[tuple[int, int]]:
    """Return [(busy_ticks, total_ticks)] for each logical CPU in /proc/stat."""
    out: list[tuple[int, int]] = []
    try:
        with open("/proc/stat") as f:
            for line in f:
                if not line.startswith("cpu") or line.startswith("cpu "):
                    continue
                head = line.split()[0]
                if not head[3:].isdigit():
                    continue
                parts = [int(x) for x in line.split()[1:]]
                idle = parts[3] + (parts[4] if len(parts) > 4 else 0)
                out.append((sum(parts), idle))
    except OSError:
        pass
    return out


def _busy_pct(prev: tuple[int, int] | None, cur: tuple[int, int] | None) -> float:
    """Busy percentage between two (busy, total) tick snapshots."""
    if not cur or not prev:
        return 0.0
    d_total = cur[0] - prev[0]
    d_idle = cur[1] - prev[1]
    if d_total <= 0:
        return 0.0
    return max(0.0, min(100.0, (1.0 - d_idle / d_total) * 100.0))


class _CpuSampler:
    """Tracks CPU busy% between successive samples."""

    def __init__(self) -> None:
        self._prev = _read_proc_stat()
        self._last = time.time()

    def sample(self) -> float:
        cur = _read_proc_stat()
        now = time.time()
        prev, self._prev = self._prev, cur
        self._last = now
        if not cur or not prev:
            return 0.0
        d_total = cur[0] - prev[0]
        d_idle = cur[1] - prev[1]
        if d_total <= 0:
            return 0.0
        return max(0.0, min(100.0, (1.0 - d_idle / d_total) * 100.0))


_CPU = _CpuSampler()


class _PerCpuSampler:
    """Tracks per-logical-CPU busy% between successive samples."""

    def __init__(self) -> None:
        self._prev = _read_per_cpu()

    def sample(self) -> list[float]:
        cur = _read_per_cpu()
        prev, self._prev = self._prev, cur
        if not cur:
            return []
        if not prev or len(prev) != len(cur):
            return [0.0] * len(cur)
        return [round(_busy_pct(p, c), 1) for p, c in zip(prev, cur)]


_PER_CPU = _PerCpuSampler()


def _meminfo() -> dict[str, float]:
    """RAM / swap usage in MB. MemAvailable accounts for reclaimable page cache."""
    out: dict[str, float] = {}
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                key, _, rest = line.partition(":")
                info[key] = int(rest.split()[0])  # kB
    except OSError:
        return out
    total = info.get("MemTotal", 0) / 1024
    avail = info.get("MemAvailable", 0) / 1024
    used = total - avail
    swap_total = info.get("SwapTotal", 0) / 1024
    swap_free = info.get("SwapFree", 0) / 1024
    out["ram_total_mb"] = round(total)
    out["ram_used_mb"] = round(used)
    out["ram_available_mb"] = round(avail)
    out["ram_percent"] = round(used / total * 100, 1) if total else 0.0
    out["swap_total_mb"] = round(swap_total)
    out["swap_used_mb"] = round(swap_total - swap_free)
    out["swap_percent"] = (
        round((swap_total - swap_free) / swap_total * 100, 1) if swap_total else 0.0
    )
    return out


def _disk_usage(path: Path) -> dict[str, Any]:
    try:
        st = os.statvfs(path)
        # Linux statvfs has f_bfree (free to any user) and f_bavail (free to
        # this user, respecting the reserved-blocks quota). There is no f_free.
        total = st.f_blocks * st.f_frsize
        free = st.f_bavail * st.f_frsize
        used = total - st.f_bfree * st.f_frsize
        return {
            "total_gb": round(total / 1024**3, 1),
            "used_gb": round(used / 1024**3, 1),
            "free_gb": round(free / 1024**3, 1),
            "percent": round(used / total * 100, 1) if total else 0.0,
        }
    except OSError:
        return {}


def _gpu_stats() -> dict[str, Any]:
    if not _GPU_OK or pynvml is None:
        return {"available": False}

    out: dict[str, Any] = {"available": True, "name": _GPU_NAME}
    try:
        util = pynvml.nvmlDeviceGetUtilizationRates(_GPU_HANDLE)
        out["util_percent"] = util.gpu
        # NOTE: util.memory is memory-CONTROLLER activity (busy time), not
        # how much VRAM is allocated. The VRAM bar needs occupancy, which is
        # computed from the memory info below. Keep this as the controller
        # metric under its own name.
        out["memory_controller_percent"] = util.memory
    except Exception:
        out["util_percent"] = None
        out["memory_controller_percent"] = None

    try:
        mem = pynvml.nvmlDeviceGetMemoryInfo(_GPU_HANDLE)
        out["vram_total_gb"] = round(mem.total / 1024**3, 2)
        out["vram_used_gb"] = round(mem.used / 1024**3, 2)
        out["vram_free_gb"] = round(mem.free / 1024**3, 2)
        # True VRAM occupancy: the fraction of the card that is allocated.
        # This is what the VRAM bar should fill.
        out["memory_percent"] = (
            round(mem.used / mem.total * 100, 1) if mem.total else 0.0
        )
    except Exception:
        out["vram_total_gb"] = None
        out["vram_used_gb"] = None
        out["vram_free_gb"] = None
        out["memory_percent"] = None

    for key, fn in (
        ("temp_c", lambda: pynvml.nvmlDeviceGetTemperature(
            _GPU_HANDLE, pynvml.NVML_TEMPERATURE_GPU)),
        ("power_w", lambda: pynvml.nvmlDeviceGetPowerUsage(_GPU_HANDLE) / 1000.0),
        ("clock_mhz", lambda: pynvml.nvmlDeviceGetClockInfo(
            _GPU_HANDLE, pynvml.NVML_CLOCK_GRAPHICS)),
    ):
        try:
            out[key] = round(fn(), 1)
        except Exception:
            out[key] = None

    return out


def snapshot(output_dir: Path) -> dict[str, Any]:
    """Full system snapshot. Cheap enough to poll at 1 Hz."""
    data: dict[str, Any] = {
        "ts": time.time(),
        "cpu_percent": round(_CPU.sample(), 1),
        "cpu_cores": os.cpu_count(),
        "cpu_per_core": _PER_CPU.sample(),
    }
    data.update(_meminfo())
    data.update(_disk_usage(output_dir))
    data["gpu"] = _gpu_stats()
    return data