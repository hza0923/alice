"""Convert device_registry.v3 into scheduler runtime dicts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from pp_nextgen.profiling.constants import KV_MODULES

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


def _bucket_to_poly(bucket: Dict[str, Any]) -> Dict[str, Any]:
    """Map a registry model bucket to unified polynomial coefficients (c0 + c1*x + c2*x^2)."""
    form = str(bucket.get("form", "")).lower()
    return {
        "form": form or "linear",
        "c0": float(bucket.get("c0", 0.0)),
        "c1": float(bucket.get("c1", 0.0)),
        "c2": float(bucket.get("c2", 0.0)),
        "x": str(bucket.get("x", "seq_len")),
        "unit": str(bucket.get("unit", "ms")),
    }


def _memory_to_base_inc(bucket: Dict[str, Any]) -> tuple[float, float]:
    return float(bucket.get("c0", 0.0)), float(bucket.get("c1", 0.0))


def build_device_performance_from_registry(
    registry: Dict[str, Any],
    prefer_bs: int,
) -> Dict[str, Any]:
    devices = registry.get("devices", {})
    perf: Dict[str, Any] = {}
    schema = str(registry.get("schema_version", ""))
    if schema != "device_registry.v3":
        raise ValueError(f"unsupported registry schema: {schema!r}; expected 'device_registry.v3'")

    for dev_name, dev_entry in devices.items():
        modules = dev_entry.get("modules", {})
        out_modules: Dict[str, Any] = {}

        for module_name, module_entry in modules.items():
            time_info = module_entry.get("time_models", {})
            prefill = time_info.get("prefill", {})
            decode = time_info.get("decode", {})
            p_by_bs = prefill.get("by_bs", {})
            by_bs = decode.get("by_bs", {})

            bs_key = _pick_best_bs_bucket(by_bs, prefer_bs)
            p_bs_key = _pick_best_bs_bucket(p_by_bs, prefer_bs)
            if bs_key is None and p_bs_key is None:
                continue
            dec_bucket = by_bs.get(bs_key, {}) if bs_key is not None else {}
            pre_bucket = p_by_bs.get(p_bs_key, {}) if p_bs_key is not None else {}
            out_modules[module_name] = {
                "decode": _bucket_to_poly(dec_bucket),
                "prefill": _bucket_to_poly(pre_bucket),
            }

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
    if schema != "device_registry.v3":
        raise ValueError(f"unsupported registry schema: {schema!r}; expected 'device_registry.v3'")

    for module_name, module_entry in modules.items():
        mem = module_entry.get("memory_models", {})
        decode = mem.get("decode", {})
        by_bs = decode.get("by_bs", {})

        bs_key = _pick_best_bs_bucket(by_bs, prefer_bs)
        if bs_key is None:
            continue
        bucket = by_bs[bs_key]
        base, inc = _memory_to_base_inc(bucket)
        bs_bucket = int(bs_key)
        if module_name in KV_MODULES and bs_bucket > 0:
            kv_per_token_gb = float(inc) / float(bs_bucket)
            out[module_name] = {
                "base": float(base),
                "inc": float(inc),
                "kv_per_token_gb": kv_per_token_gb,
                "batch_size_bucket": bs_bucket,
            }
        else:
            out[module_name] = {
                "base": float(base),
                "inc": float(inc),
                "kv_per_token_gb": 0.0,
                "batch_size_bucket": bs_bucket,
            }

    return out
