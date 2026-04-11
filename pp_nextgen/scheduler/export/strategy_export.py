"""Write per-worker strategy JSON slices from global pipeline_strategy."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from pp_nextgen.runtime.strategy import head_tail_modules_for_worker


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _canonical_modules_from_layer_names(layer_names: List[str]) -> List[str]:
    out: List[str] = []
    for ln in layer_names:
        if ln.startswith("input_embed_"):
            out.append(ln.replace("input_embed_", "", 1))
        elif ln.startswith("output_embed_"):
            out.append(ln.replace("output_embed_", "", 1))
        elif ln.startswith("layer"):
            parts = ln.split("_", 1)
            out.append(parts[1] if len(parts) > 1 else ln)
        else:
            out.append(ln)
    return out


def write_worker_strategies(global_strategy: Dict[str, Any], out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    ref = "pipeline_strategy.json"
    si = global_strategy.get("schedule_input") or {}
    designated = str(si.get("designated_device") or "")
    tail_n = int(si.get("designated_tail_n") or 0)
    for stage in global_strategy.get("pipeline_stages", []):
        worker = stage["worker_name"]
        mods = _canonical_modules_from_layer_names(stage.get("modules_to_execute", []))
        head_m, tail_m = head_tail_modules_for_worker(worker, mods, designated, tail_n)
        doc = {
            "schema_version": "worker_strategy.v1",
            "generated_at": _iso_now(),
            "worker_name": worker,
            "global_strategy_ref": ref,
            "execution_plan": {
                "head_ordered_modules": head_m,
                "tail_ordered_modules": tail_m,
                "phase_rules": {"prefill": {"enabled": True}, "decode": {"enabled": True}},
                "transfer_policy": {
                    "compute_payload_bytes_locally": True,
                    "formula_ref": "model_config.comm_contract.v1",
                },
            },
        }
        safe = worker.replace("/", "_").replace("\\", "_")
        with (out_path / f"{safe}.strategy.json").open("w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)
