"""Read legacy split_module *_all_results.json into normalized internal rows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from pp_nextgen.profiling.constants import PROFILE_TO_MODULE


@dataclass(frozen=True)
class LegacyProfileBundle:
    """Raw legacy file plus optional normalized v1 header fields."""

    raw: dict[str, Any]
    model: str
    device_id: str
    device_memory_gb: float


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_legacy_all_results(path: str | Path) -> LegacyProfileBundle:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    model = str(raw.get("model", "unknown"))
    device_id = str(raw.get("device", raw.get("device_id", "unknown")))
    mem = float(raw.get("device_memory_gb", 0.0))
    return LegacyProfileBundle(raw=raw, model=model, device_id=device_id, device_memory_gb=mem)


def legacy_to_all_results_v1(bundle: LegacyProfileBundle, source_path: str | None = None) -> dict[str, Any]:
    """Convert legacy JSON to all_results.v1-shaped document (for validation/export)."""
    cfgs = bundle.raw.get("test_configurations", [])
    out_cfgs: list[dict[str, Any]] = []
    for cfg in cfgs:
        bs = int(cfg["batch_size"])
        mods: dict[str, Any] = {}
        components = cfg.get("components", {})
        for prof_name, mod in PROFILE_TO_MODULE.items():
            if prof_name not in components:
                continue
            c = components[prof_name]
            if c.get("status") != "completed":
                continue
            pre = {str(k): float(v) for k, v in c.get("prefill_times", {}).items()}
            dec = c.get("decode_times", {})
            dec_out: dict[str, float] = {}
            if isinstance(dec, dict):
                for k, v in dec.items():
                    if k == "average":
                        dec_out["average"] = float(v)
                    else:
                        dec_out[str(k)] = float(v)
            mods[mod] = {
                "status": "completed",
                "prefill_samples_ms": pre,
                "decode_samples_ms": dec_out,
                "weight_size_gb": float(c.get("weight_size_gb", 0.0)),
            }
        prefill_keys: set[int] = set()
        for m in mods.values():
            for k in m.get("prefill_samples_ms", {}):
                prefill_keys.add(int(k))
        prefill_lens = sorted(prefill_keys)
        decode_keys: set[int] = set()
        aq = mods.get("attn_qk", {}).get("decode_samples_ms", {})
        for k in aq:
            if k == "average":
                continue
            decode_keys.add(int(k))
        decode_lens = sorted(decode_keys)
        out_cfgs.append(
            {
                "batch_size": bs,
                "prefill_seq_lens": prefill_lens,
                "decode_context_lens": decode_lens,
                "legacy_meta": {
                    "p_max_len": cfg.get("p_max_len"),
                    "d_max_len": cfg.get("d_max_len"),
                    "step": cfg.get("step"),
                },
                "modules": mods,
            }
        )
    return {
        "schema_version": "all_results.v1",
        "generated_at": _iso_now(),
        "model": bundle.model,
        "device_id": bundle.device_id,
        "device_type": bundle.device_id,
        "device_memory_gb": bundle.device_memory_gb,
        "source_path": source_path,
        "test_configurations": out_cfgs,
    }


def iter_legacy_module_samples(bundle: LegacyProfileBundle) -> Iterator[tuple[int, str, dict[str, Any]]]:
    """Yield (batch_size, module_name, component_dict) for completed legacy components."""
    for cfg in bundle.raw.get("test_configurations", []):
        bs = int(cfg["batch_size"])
        comps = cfg.get("components", {})
        for prof, mod in PROFILE_TO_MODULE.items():
            if prof not in comps:
                continue
            c = comps[prof]
            if c.get("status") != "completed":
                continue
            yield bs, mod, c
