"""CPU / memory sampling during analysis (psutil, local only)."""
from __future__ import annotations

import os
import threading
import time


def _psutil():
    try:
        import psutil

        return psutil
    except ImportError:  # pragma: no cover
        return None


def is_available() -> bool:
    return _psutil() is not None


def system_snapshot() -> dict:
    """Current machine-wide CPU / memory and this process's memory."""
    ps = _psutil()
    if ps is None:
        return {"available": False}
    vm = ps.virtual_memory()
    proc = ps.Process(os.getpid())
    return {
        "available": True,
        "cpu_percent": ps.cpu_percent(interval=0.2),
        "cpu_count": ps.cpu_count(logical=True),
        "memory_total_gb": vm.total / 1024 ** 3,
        "memory_available_gb": vm.available / 1024 ** 3,
        "memory_percent": vm.percent,
        "process_rss_mb": proc.memory_info().rss / 1024 ** 2,
    }


class ResourceMonitor:
    """Samples this process's RSS and CPU% every ``interval`` seconds."""

    def __init__(self, interval: float = 0.5):
        self.interval = interval
        self.samples: list[tuple[float, float, float]] = []  # (t, rss_mb, cpu_percent_of_machine)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        ps = _psutil()
        self._proc = ps.Process(os.getpid()) if ps else None
        self._cores = (ps.cpu_count(logical=True) or 1) if ps else 1

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def start(self) -> None:
        if self._proc is None:
            return
        self._t0 = time.perf_counter()
        self._proc.cpu_percent(None)  # prime the counter
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                rss = self._proc.memory_info().rss / 1024 ** 2
                cpu = self._proc.cpu_percent(None) / self._cores  # 0-100 % of the whole machine
                self.samples.append((time.perf_counter() - self._t0, rss, cpu))
            except Exception:  # process info can transiently fail on Windows
                pass

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def summary(self) -> dict:
        if self._proc is None:
            return {"available": False}
        if not self.samples:
            rss = self._proc.memory_info().rss / 1024 ** 2
            return {"available": True, "peak_rss_mb": rss, "mean_rss_mb": rss, "mean_cpu_percent": None,
                    "peak_cpu_percent": None, "samples": 0, "timeline": []}
        rss = [s[1] for s in self.samples]
        cpu = [s[2] for s in self.samples]
        step = max(1, len(self.samples) // 120)  # keep stored timeline small
        return {
            "available": True,
            "peak_rss_mb": max(rss),
            "mean_rss_mb": sum(rss) / len(rss),
            "mean_cpu_percent": sum(cpu) / len(cpu),
            "peak_cpu_percent": max(cpu),
            "cpu_cores": self._cores,
            "samples": len(self.samples),
            "timeline": [[round(t, 1), round(r, 1), round(c, 1)] for t, r, c in self.samples[::step]],
        }
