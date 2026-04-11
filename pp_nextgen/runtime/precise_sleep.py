"""Higher-resolution delays on Windows (timeBeginPeriod + QPC busy-wait tail).

Async entrypoints avoid blocking the event loop for long sleeps.
"""

from __future__ import annotations

import asyncio
import atexit
import ctypes
import sys
import time
from ctypes import wintypes
from typing import Optional

_is_windows = sys.platform == "win32"
_winmm: Optional[ctypes.WinDLL] = None
_kernel32: Optional[ctypes.WinDLL] = None
_timer_period_started = False


def _ensure_windows_timer_resolution() -> None:
    global _winmm, _kernel32, _timer_period_started
    if not _is_windows or _timer_period_started:
        return
    _winmm = ctypes.WinDLL("winmm")
    _kernel32 = ctypes.windll.kernel32
    _winmm.timeBeginPeriod(1)
    _timer_period_started = True

    def _restore() -> None:
        if _winmm is not None:
            _winmm.timeEndPeriod(1)

    atexit.register(_restore)


class PreciseSleep:
    """Windows: 1 ms timer period + optional QPC spin for sub-tick accuracy."""

    def __init__(self) -> None:
        self.is_windows = _is_windows
        if self.is_windows:
            _ensure_windows_timer_resolution()
            self._k = ctypes.windll.kernel32

    def sleep_ms(self, milliseconds: float) -> None:
        ms = max(0, int(round(milliseconds)))
        if self.is_windows:
            self._k.Sleep(ms)
        else:
            time.sleep(ms / 1000.0)

    def precise_sleep_ms(self, milliseconds: float) -> None:
        """Hybrid coarse Sleep + QPC busy-wait for short intervals."""
        if milliseconds <= 0:
            return
        if not self.is_windows:
            time.sleep(milliseconds / 1000.0)
            return

        _ensure_windows_timer_resolution()
        freq = wintypes.LARGE_INTEGER()
        self._k.QueryPerformanceFrequency(ctypes.byref(freq))
        start = wintypes.LARGE_INTEGER()
        self._k.QueryPerformanceCounter(ctypes.byref(start))
        target_counts = int((milliseconds / 1000.0) * freq.value)

        if milliseconds > 2.0:
            self._k.Sleep(int(milliseconds - 2.0))

        while True:
            cur = wintypes.LARGE_INTEGER()
            self._k.QueryPerformanceCounter(ctypes.byref(cur))
            if cur.value - start.value >= target_counts:
                break


_default: Optional[PreciseSleep] = None

# Decode-style hops are often ~20–40ms; asyncio.sleep on Windows aligns to ~15.6ms ticks.
# Keep coarse asyncio only for long sleeps (e.g. large prefill) to avoid thread-pool churn.
_WIN_ASYNCIO_SLEEP_THRESHOLD_MS = 100.0


def _singleton() -> PreciseSleep:
    global _default
    if _default is None:
        _default = PreciseSleep()
    return _default


def _precise_sleep_measure_ms(milliseconds: float) -> float:
    """Run hybrid QPC sleep on current thread and return elapsed ms (avoids event-loop tick bias)."""
    if milliseconds <= 0:
        return 0.0
    # perf_counter keeps QPC-grade resolution on Windows worker threads; monotonic can tick ~15.6ms.
    t0 = time.perf_counter()
    _singleton().precise_sleep_ms(milliseconds)
    return (time.perf_counter() - t0) * 1000.0


async def sleep_seconds_async(seconds: float) -> float:
    """Sleep for ``seconds`` wall time; returns elapsed ms (intended for metrics).

    On Windows, delays under ~100ms use ``timeBeginPeriod`` + QPC hybrid sleep on a
    worker thread and **measure inside that thread**, so ``actual_compute_ms`` is not
    inflated by asyncio's ~15.6ms wakeup quantization after ``run_in_executor`` completes.
    """
    if seconds <= 0:
        return 0.0
    s = _singleton()
    if not s.is_windows:
        t0 = time.perf_counter()
        await asyncio.sleep(seconds)
        return (time.perf_counter() - t0) * 1000.0

    _ensure_windows_timer_resolution()
    ms = seconds * 1000.0
    if ms >= _WIN_ASYNCIO_SLEEP_THRESHOLD_MS:
        t0 = time.perf_counter()
        await asyncio.sleep(seconds)
        return (time.perf_counter() - t0) * 1000.0
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _precise_sleep_measure_ms, ms)
