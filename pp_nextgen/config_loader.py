"""Load YAML configs for model + cluster solve."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be mapping: {path}")
    return data


def build_scheduler_model_cfg(model_doc: Dict[str, Any]) -> Dict[str, Any]:
    m = model_doc.get("model", model_doc)
    sched = model_doc.get("scheduler", {})
    if not sched:
        raise ValueError("model YAML must contain a top-level 'scheduler:' block for the solver")
    mm = dict(sched.get("module_memory_gb") or {})
    return {
        "name": m.get("name", "llama2-7b"),
        "num_layers": int(sched.get("num_layers", m.get("num_layers", 32))),
        "hidden_size": int(sched.get("hidden_size", m.get("hidden_size", 4096))),
        "n_head": int(sched.get("n_head", m.get("num_attention_heads", 32))),
        "n_kv_head": int(sched.get("n_kv_head", m.get("num_key_value_heads", 32))),
        "ffn_dim": int(sched.get("ffn_dim", m.get("ffn_dim", 11008))),
        "vocab_size": int(sched.get("vocab_size", m.get("vocab_size", 32000))),
        "dtype_bytes": int(sched.get("dtype_bytes", m.get("dtype_bytes", 2))),
        "component_config": sched["component_config"],
        "module_memory_gb": mm,
    }


def kv_per_token_bytes(model_doc: Dict[str, Any]) -> int:
    mc = model_doc.get("memory_contract", {})
    return int(mc.get("kv_per_token_bytes", 0))
