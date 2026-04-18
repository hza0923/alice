"""Load runtime YAML (configs/runtime/*.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from pp_nextgen.config_loader import load_yaml


@dataclass
class RuntimeConfig:
    task_queue_maxsize: int
    send_queue_maxsize: int
    strict_ordering: bool
    max_in_flight_requests: int
    rpc_timeout_ms: int
    registration_wait_timeout_ms: int
    execution_mode: str
    shape_enable_sdpa: bool
    shape_random_seed: int
    shape_max_batch_size: int
    shape_max_seq_len: int
    shape_device: str


def load_runtime_config(path: str | Path) -> RuntimeConfig:
    doc = load_yaml(path)
    r = doc.get("runtime") or doc
    q = r.get("queues") or {}
    grpc = r.get("grpc") or {}
    sched = r.get("scheduling") or {}
    ex = r.get("execution") or {}
    shape = ex.get("shape_executor") or {}
    return RuntimeConfig(
        task_queue_maxsize=int(q.get("task_queue_maxsize", 2048)),
        send_queue_maxsize=int(q.get("send_queue_maxsize", 2048)),
        strict_ordering=bool(sched.get("strict_ordering", False)),
        max_in_flight_requests=int(sched.get("max_in_flight_requests", 512)),
        rpc_timeout_ms=int(grpc.get("rpc_timeout_ms", 300000)),
        registration_wait_timeout_ms=int(grpc.get("registration_wait_timeout_ms", 600_000)),
        execution_mode=str(ex.get("mode", "sleep_executor")),
        shape_enable_sdpa=bool(shape.get("enable_sdpa", False)),
        shape_random_seed=int(shape.get("random_seed", 42)),
        shape_max_batch_size=int(shape.get("max_batch_size", 32)),
        shape_max_seq_len=int(shape.get("max_seq_len", 4096)),
        shape_device=str(shape.get("device", "cpu")),
    )


def runtime_section(doc: Dict[str, Any]) -> Dict[str, Any]:
    return doc.get("runtime") or doc
