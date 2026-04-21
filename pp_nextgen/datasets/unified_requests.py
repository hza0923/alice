"""Load/save unified dataset request JSON used by runtime submit and DES simulator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

UNIFIED_SCHEMA_VERSION = "unified_dataset_requests.v1"


def _positive_int(name: str, value: Any) -> int:
    v = int(value)
    if v <= 0:
        raise ValueError(f"{name} must be > 0, got {v}")
    return v


def _non_negative_int(name: str, value: Any) -> int:
    v = int(value)
    if v < 0:
        raise ValueError(f"{name} must be >= 0, got {v}")
    return v


def load_unified_requests_doc(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate a unified_dataset_requests.v1 document and return the `requests` rows."""
    if not isinstance(data, dict):
        raise ValueError("unified request document must be a JSON object")
    ver = data.get("schema_version")
    if ver != UNIFIED_SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version: {ver!r}, expected {UNIFIED_SCHEMA_VERSION!r}")
    rows = data.get("requests")
    if not isinstance(rows, list) or not rows:
        raise ValueError("unified request document needs a non-empty 'requests' array")
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(rows):
        if not isinstance(item, dict):
            raise ValueError(f"requests[{i}] must be an object")
        c = _positive_int(f"requests[{i}].context_len", item["context_len"])
        t = _non_negative_int(f"requests[{i}].target_len", item["target_len"])
        if t < c:
            raise ValueError(f"requests[{i}]: target_len must be >= context_len")
        row = dict(item)
        row["context_len"] = c
        row["target_len"] = max(c, t)
        out.append(row)
    return out


def load_unified_requests_path(path: str | Path) -> List[Dict[str, Any]]:
    """Load unified_dataset_requests.v1 from a JSON file."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return load_unified_requests_doc(data)


def load_simulation_specs_from_json_path(path: str | Path) -> List[Dict[str, Any]]:
    """Load request shapes for DES: unified doc or legacy submit JSON list."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = normalize_json_payload_to_submit_specs(data)
    return [
        {
            "req_id": r.get("req_id"),
            "context_len": int(r["context_len"]),
            "target_len": int(r["target_len"]),
        }
        for r in rows
    ]


def normalize_json_payload_to_submit_specs(payload: Union[List[Any], Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Accept either a bare JSON list (legacy submit list) or a unified document object."""
    if isinstance(payload, dict):
        if payload.get("schema_version") == UNIFIED_SCHEMA_VERSION:
            return load_unified_requests_doc(payload)
        raise ValueError("JSON object must use unified_dataset_requests.v1 schema")
    if isinstance(payload, list):
        specs: List[Dict[str, Any]] = []
        for i, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"entry[{i}] must be an object")
            b = _positive_int(f"entry[{i}].batch_size", item["batch_size"])
            c = _non_negative_int(f"entry[{i}].context_len", item["context_len"])
            t = _non_negative_int(f"entry[{i}].target_len", item["target_len"])
            specs.append(
                {
                    "req_id": item.get("req_id"),
                    "batch_size": b,
                    "context_len": c,
                    "target_len": max(c, t),
                }
            )
        if not specs:
            raise ValueError("request spec list is empty")
        return specs
    raise ValueError("request spec JSON must be a list or unified document object")


def write_unified_requests_json(
    *,
    path: str | Path,
    rows: List[Dict[str, Any]],
) -> None:
    """Write unified_dataset_requests.v1 JSON."""
    doc = {"schema_version": UNIFIED_SCHEMA_VERSION, "requests": rows}
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
