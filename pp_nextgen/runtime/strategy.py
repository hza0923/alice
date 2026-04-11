"""Load pipeline / worker strategy JSON and evaluate timing / comm models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pp_nextgen.config_loader import load_yaml


def load_pipeline_strategy(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_worker_strategy(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def worker_strategy_filename(worker_name: str) -> str:
    """Export file name convention, e.g. 3060_0 -> 3060_0.strategy.json (same-type replicas differ by suffix)."""
    return f"{worker_name}.strategy.json"


def worker_strategy_path(export_workers_dir: str | Path, worker_name: str) -> Path:
    return Path(export_workers_dir) / worker_strategy_filename(worker_name)


def pipeline_stage_order(doc: Dict[str, Any]) -> List[str]:
    return [s["worker_name"] for s in doc.get("pipeline_stages", [])]


def linear_next_worker(order: List[str], worker_name: str) -> Optional[str]:
    """Linear pipeline order for NotifyPipelineEnd (not the data-plane ring next_worker)."""
    try:
        i = order.index(worker_name)
    except ValueError:
        return None
    if i + 1 < len(order):
        return str(order[i + 1])
    return None


def find_stage_for_worker(doc: Dict[str, Any], worker_name: str) -> Optional[Dict[str, Any]]:
    for st in doc.get("pipeline_stages", []):
        if st.get("worker_name") == worker_name:
            return st
    return None


def next_worker_name(doc: Dict[str, Any], worker_name: str) -> Optional[str]:
    st = find_stage_for_worker(doc, worker_name)
    if st is None:
        return None
    n = st.get("next_worker")
    return str(n) if n is not None else None


def is_last_worker(doc: Dict[str, Any], worker_name: str) -> bool:
    st = find_stage_for_worker(doc, worker_name)
    if st is None:
        return False
    return bool(st.get("is_last_worker", False))


def model_block_from_pipeline(doc: Dict[str, Any]) -> Dict[str, Any]:
    m = doc.get("model")
    if isinstance(m, dict):
        return m
    return {}


def _linear_eval(model: Optional[Dict[str, Any]], x: float) -> Optional[float]:
    if not model or model.get("form") != "linear":
        return None
    base = float(model.get("base", 0.0))
    inc = float(model.get("inc", 0.0))
    return base + inc * x


def expected_compute_ms(
    stage: Dict[str, Any],
    phase_name: str,
    context_len: int,
    batch_size: int,
) -> float:
    sm = stage.get("stage_models") or {}
    phase = sm.get(phase_name) or {}
    tm = phase.get("time_ms") if isinstance(phase, dict) else None
    x = float(context_len)
    v = _linear_eval(tm, x)
    if v is not None:
        return float(v)
    sp = stage.get("stage_params") or {}
    if phase_name == "prefill":
        return float(sp.get("comp_time_ms", 0.0))
    # Legacy-style fallback (see grpc_heterogeneous_pipeline/3060/role/worker.py)
    base_time = float(sp.get("base_time", 0.0))
    inc_time = float(sp.get("increase_time", 0.0))
    return base_time + inc_time * float(batch_size) * x


def expected_comm_bytes(
    stage: Dict[str, Any],
    phase_name: str,
    context_len: int,
    batch_size: int,
) -> int:
    sm = stage.get("stage_models") or {}
    phase = sm.get(phase_name) or {}
    cb = phase.get("comm_bytes") if isinstance(phase, dict) else None
    x = float(context_len)
    v = _linear_eval(cb, x)
    if v is not None:
        return max(0, int(round(v)))
    sp = stage.get("stage_params") or {}
    if phase_name == "prefill":
        raw = stage.get("comm_bytes_to_next")
        if raw is not None:
            return max(0, int(round(float(raw))))
    base_size = float(sp.get("base_size", 0.0))
    inc_size = float(sp.get("inc_size", 0.0))
    return max(0, int(round(base_size + inc_size * float(batch_size) * x)))


def expected_comm_ms(stage: Dict[str, Any]) -> float:
    v = stage.get("comm_time_ms")
    if v is not None:
        return float(v)
    return 0.0


def load_model_yaml(path: str | Path) -> Dict[str, Any]:
    return load_yaml(path)


def merge_model_for_runtime(pipeline_doc: Dict[str, Any], model_yaml: Dict[str, Any]) -> Dict[str, Any]:
    """Prefer scheduler export model dict; fill gaps from configs/model/*.yaml."""
    out: Dict[str, Any] = dict(model_yaml.get("model") or {})
    pb = model_block_from_pipeline(pipeline_doc)
    for k, v in pb.items():
        out.setdefault(k, v)
    out.setdefault("name", out.get("name", "llama2-7b"))
    return out


def worker_matches_designated_device(worker_name: str, designated_device: str) -> bool:
    """True if this worker stage belongs to the designated device (e.g. 3060_0 / 3060_1 for 3060)."""
    d = (designated_device or "").strip()
    if not d:
        return False
    w = (worker_name or "").strip()
    if w == d:
        return True
    return w.startswith(d + "_")


def head_tail_modules_for_worker(
    worker_name: str,
    modules: List[str],
    designated_device: str,
    designated_tail_n: int,
) -> Tuple[List[str], List[str]]:
    """
    Put the last ``designated_tail_n`` canonical module names into tail on the designated device only;
    all other workers get (modules, []).
    """
    mods = list(modules)
    n = max(0, int(designated_tail_n))
    if n <= 0 or not worker_matches_designated_device(worker_name, designated_device):
        return mods, []
    take = min(n, len(mods))
    if take <= 0:
        return mods, []
    return mods[:-take], mods[-take:]


def split_head_tail_modules_from_execution_plan(
    exec_plan: Dict[str, Any],
    *,
    worker_name: str,
    is_first_worker: bool,
    pipeline_doc: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Resolve head/tail lists from worker JSON.

    New export uses head_ordered_modules + tail_ordered_modules only.
    Legacy files may still have ordered_modules; for the first pipeline worker only,
    apply the same designated_device / designated_tail_n rule as the exporter.
    """
    head = list(exec_plan.get("head_ordered_modules") or [])
    tail = list(exec_plan.get("tail_ordered_modules") or [])
    if head or tail:
        return head, tail

    legacy = list(exec_plan.get("ordered_modules") or [])
    if not legacy:
        return [], []
    if not is_first_worker:
        return legacy, []

    si = (pipeline_doc or {}).get("schedule_input") or {}
    return head_tail_modules_for_worker(
        worker_name,
        legacy,
        str(si.get("designated_device") or ""),
        int(si.get("designated_tail_n") or 0),
    )
