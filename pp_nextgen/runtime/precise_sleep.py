"""Higher-resolution delays: Windows uses timeBeginPeriod + QPC tail; POSIX uses coarse
``time.sleep`` plus ``perf_counter`` spin to the deadline (same hybrid idea as Windows).

Async entrypoints avoid blocking the event loop for short sleeps by using a thread pool.
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


_HYBRID_TAIL_MARGIN_MS = 2.0


class PreciseSleep:
    """Windows: 1 ms timer period + QPC spin. POSIX: ``sleep`` most of interval + perf_counter spin."""

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
        """Hybrid coarse delay + high-resolution busy-wait to target (matches Windows strategy)."""
        if milliseconds <= 0:
            return
        if not self.is_windows:
            deadline = time.perf_counter() + milliseconds / 1000.0
            coarse_s = max(0.0, (milliseconds - _HYBRID_TAIL_MARGIN_MS) / 1000.0)
            if coarse_s > 0:
                time.sleep(coarse_s)
            while time.perf_counter() < deadline:
                pass
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

# Decode-style hops are often ~20–40ms; asyncio.sleep aligns to coarse ticks on many platforms.
# Keep asyncio only for long sleeps (e.g. large prefill) to avoid thread-pool churn.
_LONG_ASYNCIO_SLEEP_THRESHOLD_MS = 100.0


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

    Short sleeps use hybrid precise timing on a worker thread (Windows: timeBeginPeriod + QPC;
    POSIX: ``time.sleep`` + perf_counter spin) so ``actual_compute_ms`` is not inflated by
    asyncio's wakeup quantization after ``run_in_executor`` completes.
    """
    if seconds <= 0:
        return 0.0
    ms = seconds * 1000.0
    if ms >= _LONG_ASYNCIO_SLEEP_THRESHOLD_MS:
        t0 = time.perf_counter()
        await asyncio.sleep(seconds)
        return (time.perf_counter() - t0) * 1000.0

    if _singleton().is_windows:
        _ensure_windows_timer_resolution()
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _precise_sleep_measure_ms, ms)
