"""Lightweight JSON contract checks (minimal, no external deps)."""

from __future__ import annotations

from typing import Any, Dict


def validate_device_registry_minimal(reg: Dict[str, Any]) -> None:
    if str(reg.get("schema_version", "")) != "device_registry.v3":
        raise ValueError("registry schema_version must be device_registry.v3")
    if "devices" not in reg or not isinstance(reg["devices"], dict):
        raise ValueError("registry must have devices object")
    for dev_id, dev in reg["devices"].items():
        if "modules" not in dev or not isinstance(dev["modules"], dict):
            raise ValueError(f"device {dev_id} missing modules")
        for mod_name, mod in dev["modules"].items():
            tm = mod.get("time_models", {})
            mm = mod.get("memory_models", {})
            if not isinstance(tm, dict) or not isinstance(mm, dict):
                raise ValueError(f"device {dev_id} module {mod_name} missing time_models/memory_models")
            if "prefill" not in tm or "decode" not in tm:
                raise ValueError(f"device {dev_id} module {mod_name} missing prefill/decode time models")
            if "decode" not in mm:
                raise ValueError(f"device {dev_id} module {mod_name} missing decode memory model")


def validate_all_results_minimal(doc: Dict[str, Any]) -> None:
    if doc.get("schema_version") != "all_results.v1":
        return
    if "test_configurations" not in doc:
        raise ValueError("all_results.v1 missing test_configurations")
