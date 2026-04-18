"""Load pipeline / worker strategy JSON and evaluate timing / comm models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pp_nextgen.config_loader import load_yaml


def load_pipeline_strategy(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        doc = json.load(f)
    if str(doc.get("schema_version", "")) != "pipeline_strategy.v2":
        raise ValueError("pipeline strategy must be pipeline_strategy.v2")
    return doc


def load_worker_strategy(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        doc = json.load(f)
    if str(doc.get("schema_version", "")) != "worker_strategy.v1":
        raise ValueError("worker strategy must be worker_strategy.v1")
    return doc


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
    if not model:
        return None
    form = str(model.get("form", "")).lower()
    if form == "constant":
        return float(model.get("c0", 0.0))
    if form == "quadratic":
        c0 = float(model.get("c0", 0.0))
        c1 = float(model.get("c1", 0.0))
        c2 = float(model.get("c2", 0.0))
        return c0 + c1 * x + c2 * x * x
    if form == "linear":
        c0 = float(model.get("c0", 0.0))
        c1 = float(model.get("c1", 0.0))
        return c0 + c1 * x
    return None


def _select_branch_model(stage: Dict[str, Any], phase_name: str, model_key: str, branch: str) -> Optional[Dict[str, Any]]:
    sm = stage.get("stage_models") or {}
    phase = sm.get(phase_name) or {}
    if not isinstance(phase, dict):
        return None
    raw = phase.get(model_key)
    if isinstance(raw, dict):
        b = (branch or "single").strip().lower()
        return raw.get(b) or raw.get("single")
    return raw if isinstance(raw, dict) else None


def stage_has_worker0_head_tail(stage: Dict[str, Any]) -> bool:
    decode_tm = _select_branch_model(stage, "decode", "time_ms", "head")
    decode_tail = _select_branch_model(stage, "decode", "time_ms", "tail")
    pre_tm = _select_branch_model(stage, "prefill", "time_ms", "head")
    pre_tail = _select_branch_model(stage, "prefill", "time_ms", "tail")
    return all(isinstance(x, dict) for x in (decode_tm, decode_tail, pre_tm, pre_tail))


def expected_compute_ms(
    stage: Dict[str, Any],
    phase_name: str,
    context_len: int,
    batch_size: int,
    branch: str = "single",
) -> float:
    sm = stage.get("stage_models") or {}
    phase = sm.get(phase_name) or {}
    tm = None
    if isinstance(phase, dict):
        tms = phase.get("time_ms")
        if isinstance(tms, dict):
            b = (branch or "single").strip().lower()
            tm = tms.get(b) or tms.get("single")
    x = float(context_len)
    v = _linear_eval(tm, x)
    if v is not None:
        return max(0.0, float(v))
    raise ValueError(f"missing stage_models.{phase_name}.time_ms for branch={branch}")


def expected_comm_bytes(
    stage: Dict[str, Any],
    phase_name: str,
    context_len: int,
    batch_size: int,
    branch: str = "single",
) -> int:
    sm = stage.get("stage_models") or {}
    phase = sm.get(phase_name) or {}
    cb = None
    if isinstance(phase, dict):
        cbs = phase.get("comm_bytes")
        if isinstance(cbs, dict):
            b = (branch or "single").strip().lower()
            cb = cbs.get(b) or cbs.get("single")
    x = float(context_len)
    v = _linear_eval(cb, x)
    if v is not None:
        return max(0, int(round(v)))
    raise ValueError(f"missing stage_models.{phase_name}.comm_bytes for branch={branch}")


def expected_comm_ms(
    stage: Dict[str, Any],
    phase_name: str = "decode",
    context_len: int = 0,
    branch: str = "single",
) -> float:
    model = _select_branch_model(stage, phase_name, "comm_time_ms", branch)
    if model:
        x = float(context_len)
        v = _linear_eval(model, x)
        if v is not None:
            return float(v)
    # Backward-compat fallback for legacy exported strategies:
    # comm_time_ms may be a stage-level scalar instead of phase/branch model.
    legacy = stage.get("comm_time_ms")
    if isinstance(legacy, (int, float)):
        return float(legacy)
    raise ValueError(f"missing stage_models.{phase_name}.comm_time_ms for branch={branch}")


def expected_decode_memory_gb(
    stage: Dict[str, Any],
    context_len: int,
    batch_size: int,
    branch: str = "single",
) -> float:
    model = _select_branch_model(stage, "decode", "memory_gb", branch)
    if isinstance(model, dict):
        c0 = float(model.get("c0", 0.0))
        c1 = float(model.get("c1", 0.0))
        # decode-only memory contract: c0 + c1 * seq_len * batch_size.
        return c0 + c1 * float(context_len) * float(batch_size)

    raise ValueError(f"missing stage_models.decode.memory_gb for branch={branch}")


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
    """
    head = list(exec_plan.get("head_ordered_modules") or [])
    tail = list(exec_plan.get("tail_ordered_modules") or [])
    if not head and not tail:
        raise ValueError("worker strategy must provide head_ordered_modules/tail_ordered_modules")
    return head, tail


# Canonical decoder step order (matches configs/model/*.yaml scheduler.component_config.decoder_layer).
DEFAULT_DECODER_SUBMODULES: List[str] = [
    "qkv_projection",
    "attn_qk",
    "attn_av",
    "o_projection",
    "up_projection",
    "down_projection",
]


def decoder_submodules_from_model_yaml(model_yaml: Dict[str, Any]) -> List[str]:
    """Ordered submodule names used when expanding coarse ``decoder_layer`` placeholders."""
    sch = model_yaml.get("scheduler") or {}
    cc = sch.get("component_config") or {}
    dl = cc.get("decoder_layer")
    if isinstance(dl, list) and dl:
        return [str(x) for x in dl]
    return list(DEFAULT_DECODER_SUBMODULES)


def expand_decoder_layer_placeholders(
    module_names: List[str], decoder_submodules: Optional[List[str]] = None
) -> List[str]:
    """Replace each ``decoder_layer`` token with the configured decoder submodule list (fine-grained exports unchanged)."""
    sub = list(decoder_submodules) if decoder_submodules else list(DEFAULT_DECODER_SUBMODULES)
    if not sub:
        sub = list(DEFAULT_DECODER_SUBMODULES)
    out: List[str] = []
    for n in module_names:
        if str(n).lower() == "decoder_layer":
            out.extend(sub)
        else:
            out.append(str(n))
    return out
