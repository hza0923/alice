"""Per-req observability: compute, comm, and timing (per worker)."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class HopRecord:
    step_id: int
    phase: str
    expected_compute_ms: float
    actual_compute_ms: float
    expected_comm_bytes: int
    expected_comm_ms: float
    actual_comm_ms: float
    payload_bytes_sent: int


@dataclass
class RequestTrace:
    req_id: str
    worker_name: str
    created_monotonic: float
    hops: List[HopRecord] = field(default_factory=list)
    finished_monotonic: Optional[float] = None


class RequestJournal:
    """Per-worker journal keyed by req_id (decode / prefill hops on this node)."""

    def __init__(self, worker_name: str) -> None:
        self.worker_name = worker_name
        self._traces: Dict[str, RequestTrace] = {}

    def ensure_req(self, req_id: str) -> RequestTrace:
        if req_id not in self._traces:
            self._traces[req_id] = RequestTrace(
                req_id=req_id,
                worker_name=self.worker_name,
                created_monotonic=time.monotonic(),
            )
        return self._traces[req_id]

    def record_hop(self, req_id: str, hop: HopRecord) -> None:
        tr = self.ensure_req(req_id)
        tr.hops.append(hop)

    def mark_finished(self, req_id: str) -> None:
        tr = self.ensure_req(req_id)
        tr.finished_monotonic = time.monotonic()

    def to_serializable(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "worker_name": self.worker_name,
            "schema_version": "request_journal.v1",
            "requests": {},
        }
        for rid, tr in self._traces.items():
            out["requests"][rid] = {
                "req_id": tr.req_id,
                "worker_name": tr.worker_name,
                "created_monotonic": tr.created_monotonic,
                "finished_monotonic": tr.finished_monotonic,
                "hops": [asdict(h) for h in tr.hops],
            }
        return out

    def export_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_serializable(), f, indent=2, ensure_ascii=False)


class MasterLatencyTracker:
    """Master-side e2e latency from SubmitTask to ReportRequestFinished."""

    def __init__(self) -> None:
        self._start: Dict[str, float] = {}
        self._records: Dict[str, Dict[str, float]] = {}

    def mark_submit(self, req_id: str) -> None:
        self._start[req_id] = time.time()

    def mark_finished(self, req_id: str) -> None:
        st = self._start.pop(req_id, None)
        if st is None:
            return
        end = time.time()
        self._records[req_id] = {"start": st, "end": end, "latency_s": end - st}

    def to_serializable(self) -> Dict[str, Any]:
        return {"schema_version": "master_latency.v1", "requests": dict(self._records)}

    def export_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_serializable(), f, indent=2, ensure_ascii=False)
