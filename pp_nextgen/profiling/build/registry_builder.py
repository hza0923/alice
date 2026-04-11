"""Assemble device_registry.v3 from legacy profiles + model memory contract."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from pp_nextgen.profiling.constants import KV_MODULES
from pp_nextgen.profiling.fit.fitter import fit_decode_time, fit_prefill_time
from pp_nextgen.profiling.legacy_ingest import iter_legacy_module_samples, load_legacy_all_results


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _gb_per_token_for_kv(kv_per_token_bytes: int, batch_size: int) -> float:
    return float(kv_per_token_bytes) * float(batch_size) / (1024.0**3)


def _memory_decode_model(
    module_name: str,
    weight_size_gb: float,
    kv_per_token_bytes: int,
    batch_size: int,
) -> Dict[str, Any]:
    if module_name in KV_MODULES:
        c1 = _gb_per_token_for_kv(kv_per_token_bytes, batch_size)
        return {
            "form": "linear",
            "c0": 0.0,
            "c1": c1,
            "c2": 0.0,
            "x": "seq_len",
            "unit": "GB",
            "note": "KV decode memory from model kv_per_token_bytes * batch_size per token (no profile kvcache).",
        }
    return {
        "form": "linear",
        "c0": float(weight_size_gb),
        "c1": 0.0,
        "c2": 0.0,
        "x": "seq_len",
        "unit": "GB",
    }


def device_entry_from_legacy_path(
    path: str | Path,
    kv_per_token_bytes: int,
) -> Dict[str, Any]:
    bundle = load_legacy_all_results(path)
    device_entry: Dict[str, Any] = {
        "device_type": bundle.device_id,
        "model": bundle.model,
        "memory_gb": float(bundle.device_memory_gb),
        "source": str(Path(path).resolve()),
        "modules": {},
    }
    for bs, mod_name, comp in iter_legacy_module_samples(bundle):
        weight_size_gb = float(comp.get("weight_size_gb", 0.0))
        prefill_times = {str(k): float(v) for k, v in comp.get("prefill_times", {}).items()}
        decode_times = comp.get("decode_times", {})
        if not isinstance(decode_times, dict):
            decode_times = {}

        dev_mod = device_entry["modules"].setdefault(
            mod_name,
            {
                "weight_size_gb": weight_size_gb,
                "time_models": {"prefill": {"by_bs": {}}, "decode": {"by_bs": {}}},
                "memory_models": {"decode": {"by_bs": {}}},
            },
        )
        if float(dev_mod.get("weight_size_gb", 0.0)) == 0.0 and weight_size_gb > 0:
            dev_mod["weight_size_gb"] = weight_size_gb

        dev_mod["time_models"]["prefill"]["by_bs"][str(bs)] = fit_prefill_time(mod_name, prefill_times)
        dev_mod["time_models"]["decode"]["by_bs"][str(bs)] = fit_decode_time(mod_name, decode_times)
        dev_mod["memory_models"]["decode"]["by_bs"][str(bs)] = _memory_decode_model(
            mod_name, weight_size_gb, kv_per_token_bytes, bs
        )
    return device_entry


def merge_device_entries(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    if not dst:
        out = dict(src)
        out["sources"] = [out.pop("source")] if "source" in out else []
        return out
    if dst.get("model") != src.get("model"):
        raise ValueError(f"model mismatch merging device: {dst.get('model')} vs {src.get('model')}")
    out = dst
    out.setdefault("sources", [])
    if "source" in src:
        out["sources"].append(src["source"])
    if float(out.get("memory_gb", 0.0)) == 0.0 and float(src.get("memory_gb", 0.0)) != 0.0:
        out["memory_gb"] = float(src["memory_gb"])
    for mod_name, mod_entry in src.get("modules", {}).items():
        dst_mod = out.setdefault("modules", {}).setdefault(mod_name, mod_entry)
        if dst_mod is mod_entry:
            continue
        if float(dst_mod.get("weight_size_gb", 0.0)) == 0.0:
            dst_mod["weight_size_gb"] = float(mod_entry.get("weight_size_gb", 0.0))
        for section in ("time_models", "memory_models"):
            if section not in mod_entry:
                continue
            dst_mod.setdefault(section, {})
            for phase, phase_entry in mod_entry[section].items():
                dst_mod[section].setdefault(phase, {})
                src_by_bs = phase_entry.get("by_bs", {})
                dst_mod[section][phase].setdefault("by_bs", {})
                dst_mod[section][phase]["by_bs"].update(src_by_bs)
    return out


def build_device_registry_v3(
    input_paths: List[str | Path],
    model_name: str,
    kv_per_token_bytes: int,
) -> Dict[str, Any]:
    devices: Dict[str, Any] = {}
    resolved_model = model_name
    for p in input_paths:
        bundle = load_legacy_all_results(p)
        resolved_model = bundle.model or model_name
        dev_id = bundle.device_id
        entry = device_entry_from_legacy_path(p, kv_per_token_bytes=kv_per_token_bytes)
        entry["model"] = resolved_model
        devices[dev_id] = merge_device_entries(devices.get(dev_id, {}), entry)
    return {
        "schema_version": "device_registry.v3",
        "generated_at": _iso_now(),
        "model": resolved_model,
        "devices": devices,
    }


def write_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
