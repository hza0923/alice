"""Fit timing models (ms) from legacy-style samples using numpy polyfit."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from pp_nextgen.profiling.constants import KV_MODULES


def _sorted_xy_from_map(d: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
    items = sorted(((int(k), float(v)) for k, v in d.items()), key=lambda t: t[0])
    x = np.array([t[0] for t in items], dtype=np.float64)
    y = np.array([t[1] for t in items], dtype=np.float64)
    return x, y


def _linear_c0_c1(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if len(x) == 0:
        return 0.0, 0.0
    if len(x) < 2:
        return float(y[0]), 0.0
    c1, c0 = np.polyfit(x, y, 1)
    return float(c0), float(c1)


def _quadratic_c0_c1_c2(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    if len(x) < 3:
        c0, c1 = _linear_c0_c1(x, y)
        return c0, c1, 0.0
    c2, c1, c0 = np.polyfit(x, y, 2)
    return float(c0), float(c1), float(c2)


def fit_prefill_time(module_name: str, prefill_times: Dict[str, float]) -> Dict[str, Any]:
    x, y = _sorted_xy_from_map(prefill_times)
    if module_name in KV_MODULES:
        c0, c1, c2 = _quadratic_c0_c1_c2(x, y)
        return {
            "form": "quadratic",
            "c0": c0,
            "c1": c1,
            "c2": c2,
            "x": "seq_len",
            "unit": "ms",
        }
    c0, c1 = _linear_c0_c1(x, y)
    return {
        "form": "linear",
        "c0": c0,
        "c1": c1,
        "c2": 0.0,
        "x": "seq_len",
        "unit": "ms",
    }


def fit_decode_time(module_name: str, decode_times: Dict[str, Any]) -> Dict[str, Any]:
    if module_name in KV_MODULES:
        d = {k: float(v) for k, v in decode_times.items() if k != "average"}
        x, y = _sorted_xy_from_map(d)
        c0, c1 = _linear_c0_c1(x, y)
        return {
            "form": "linear",
            "c0": c0,
            "c1": c1,
            "c2": 0.0,
            "x": "context_len",
            "unit": "ms",
        }
    avg = float(decode_times.get("average", 0.0))
    return {
        "form": "constant",
        "c0": avg,
        "c1": 0.0,
        "c2": 0.0,
        "x": "context_len",
        "unit": "ms",
    }
