"""Lightweight JSON contract checks (minimal, no external deps)."""

from __future__ import annotations

from typing import Any, Dict


def validate_device_registry_minimal(reg: Dict[str, Any]) -> None:
    if "devices" not in reg or not isinstance(reg["devices"], dict):
        raise ValueError("registry must have devices object")
    for dev_id, dev in reg["devices"].items():
        if "modules" not in dev or not isinstance(dev["modules"], dict):
            raise ValueError(f"device {dev_id} missing modules")


def validate_all_results_minimal(doc: Dict[str, Any]) -> None:
    if doc.get("schema_version") != "all_results.v1":
        return
    if "test_configurations" not in doc:
        raise ValueError("all_results.v1 missing test_configurations")
