"""Convert device_registry (v2 legacy or v3) into scheduler runtime dicts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional


def load_registry(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _pick_best_bs_bucket(by_bs: Dict[str, Any], prefer_bs: int) -> Optional[str]:
    if not by_bs:
        return None
    prefer_key = str(int(prefer_bs))
    if prefer_key in by_bs:
        return prefer_key
    keys = [int(k) for k in by_bs.keys()]
    nearest = min(keys, key=lambda x: abs(x - prefer_bs))
    return str(nearest)


def _decode_time_to_base_increase(bucket: Dict[str, Any]) -> tuple[float, float]:
    """Map v3 time bucket (c0,c1,c2) or legacy v2 (base,inc) to decode base+increase*seq."""
    if "c0" in bucket or bucket.get("form") in ("linear", "quadratic", "constant"):
        c0 = float(bucket.get("c0", 0.0))
        c1 = float(bucket.get("c1", 0.0))
        return c0, c1
    base = float(bucket.get("base", 0.0))
    inc = float(bucket.get("inc", 0.0))
    return base, inc


def _memory_to_base_inc(bucket: Dict[str, Any]) -> tuple[float, float]:
    if "c0" in bucket or bucket.get("form") == "linear":
        return float(bucket.get("c0", 0.0)), float(bucket.get("c1", 0.0))
    base = float(bucket.get("base", 0.0))
    inc = float(bucket.get("inc", 0.0))
    return base, inc


def build_device_performance_from_registry(
    registry: Dict[str, Any],
    prefer_bs: int,
) -> Dict[str, Any]:
    devices = registry.get("devices", {})
    perf: Dict[str, Any] = {}
    schema = str(registry.get("schema_version", ""))

    for dev_name, dev_entry in devices.items():
        modules = dev_entry.get("modules", {})
        out_modules: Dict[str, Any] = {}

        for module_name, module_entry in modules.items():
            if schema == "device_registry.v3" or "time_models" in module_entry:
                time_info = module_entry.get("time_models", {})
                decode = time_info.get("decode", {})
                by_bs = decode.get("by_bs", {})
            else:
                time_info = module_entry.get("time", {})
                decode = time_info.get("decode", {})
                by_bs = decode.get("by_bs", {})

            bs_key = _pick_best_bs_bucket(by_bs, prefer_bs)
            if bs_key is None:
                continue
            dec_bucket = by_bs[bs_key]
            base, increase = _decode_time_to_base_increase(dec_bucket)
            out_modules[module_name] = {"base": base, "increase": increase}

        memory_gb = float(dev_entry.get("memory_gb", 0.0))
        perf[dev_name] = {"memory_gb": memory_gb, "modules": out_modules}

    return perf


def build_module_memory_from_registry(
    registry: Dict[str, Any],
    prefer_bs: int,
) -> Dict[str, Any]:
    devices = registry.get("devices", {})
    any_dev = next(iter(devices.values()), None)
    if not any_dev:
        return {}
    modules = any_dev.get("modules", {})
    out: Dict[str, Any] = {}
    schema = str(registry.get("schema_version", ""))

    for module_name, module_entry in modules.items():
        if schema == "device_registry.v3" or "memory_models" in module_entry:
            mem = module_entry.get("memory_models", {})
            decode = mem.get("decode", {})
            by_bs = decode.get("by_bs", {})
        else:
            mem = module_entry.get("memory", {})
            decode = mem.get("decode", {})
            by_bs = decode.get("by_bs", {})

        bs_key = _pick_best_bs_bucket(by_bs, prefer_bs)
        if bs_key is None:
            continue
        bucket = by_bs[bs_key]
        base, inc = _memory_to_base_inc(bucket)
        out[module_name] = {"base": base, "inc": inc}

    return out
