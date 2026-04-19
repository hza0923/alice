"""Load runtime YAML (configs/runtime/*.yaml)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

from pp_nextgen.config_loader import load_yaml


@dataclass
class RuntimeConfig:
    task_queue_maxsize: int
    send_queue_maxsize: int
    strict_ordering: bool
    max_in_flight_requests: int
    rpc_timeout_ms: int
    registration_wait_timeout_ms: int
    payload_size_divisor: float
    execution_mode: str
    sleep_compute_offset_ms_default: float
    sleep_compute_offset_by_worker: Dict[str, float]
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
    sleep_ex = ex.get("sleep_executor") or {}
    shape = ex.get("shape_executor") or {}
    psd = float(grpc.get("payload_size_divisor", 1.0))
    if psd <= 0:
        psd = 1.0

    offsets_raw = sleep_ex.get("worker_compute_sleep_offset_ms")
    if offsets_raw is None:
        offsets_raw = {}
    if not isinstance(offsets_raw, Mapping):
        raise TypeError("execution.sleep_executor.worker_compute_sleep_offset_ms must be a mapping")
    sleep_compute_offset_by_worker = {str(k): float(v) for k, v in offsets_raw.items()}

    default_ms = sleep_ex.get("default_compute_sleep_offset_ms")
    if default_ms is None:
        # Legacy single field applied to every worker when no per-worker map was used.
        default_ms = sleep_ex.get("compute_sleep_offset_ms", 0.0)
    sleep_compute_offset_ms_default = float(default_ms)

    return RuntimeConfig(
        task_queue_maxsize=int(q.get("task_queue_maxsize", 2048)),
        send_queue_maxsize=int(q.get("send_queue_maxsize", 2048)),
        strict_ordering=bool(sched.get("strict_ordering", False)),
        max_in_flight_requests=int(sched.get("max_in_flight_requests", 512)),
        rpc_timeout_ms=int(grpc.get("rpc_timeout_ms", 300000)),
        registration_wait_timeout_ms=int(grpc.get("registration_wait_timeout_ms", 600_000)),
        payload_size_divisor=psd,
        execution_mode=str(ex.get("mode", "sleep_executor")),
        sleep_compute_offset_ms_default=sleep_compute_offset_ms_default,
        sleep_compute_offset_by_worker=sleep_compute_offset_by_worker,
        shape_enable_sdpa=bool(shape.get("enable_sdpa", False)),
        shape_random_seed=int(shape.get("random_seed", 42)),
        shape_max_batch_size=int(shape.get("max_batch_size", 32)),
        shape_max_seq_len=int(shape.get("max_seq_len", 4096)),
        shape_device=str(shape.get("device", "cpu")),
    )


def runtime_section(doc: Dict[str, Any]) -> Dict[str, Any]:
    return doc.get("runtime") or doc


def sleep_compute_offset_ms_for_worker(worker_name: str, rt: RuntimeConfig) -> float:
    """Sleep timing correction for ``sleep_executor`` (ms added to profiled compute before sleeping)."""
    m = rt.sleep_compute_offset_by_worker
    if worker_name in m:
        return float(m[worker_name])
    return float(rt.sleep_compute_offset_ms_default)
