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
    ingress_monotonic: Optional[float] = None
    first_token_monotonic: Optional[float] = None
    finished_monotonic: Optional[float] = None
    batch_size: int = 1
    context_len: int = 0
    target_len: int = 0

    @property
    def generated_tokens(self) -> int:
        return int(self.batch_size) * max(0, int(self.target_len) - int(self.context_len))


def _percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    vals = sorted(values)
    idx = q * (len(vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(vals) - 1)
    frac = idx - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


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

    def mark_ingress(
        self,
        req_id: str,
        *,
        batch_size: Optional[int] = None,
        context_len: Optional[int] = None,
        target_len: Optional[int] = None,
    ) -> None:
        tr = self.ensure_req(req_id)
        if tr.ingress_monotonic is None:
            tr.ingress_monotonic = time.monotonic()
        if batch_size is not None:
            tr.batch_size = int(batch_size)
        if context_len is not None:
            tr.context_len = int(context_len)
        if target_len is not None:
            tr.target_len = int(target_len)

    def mark_first_token(self, req_id: str) -> None:
        tr = self.ensure_req(req_id)
        if tr.first_token_monotonic is None:
            tr.first_token_monotonic = time.monotonic()

    def mark_finished(self, req_id: str) -> None:
        tr = self.ensure_req(req_id)
        tr.finished_monotonic = time.monotonic()

    def to_serializable(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "worker_name": self.worker_name,
            "schema_version": "request_journal.v1",
            "summary": self.summary(),
            "requests": {},
        }
        for rid, tr in self._traces.items():
            out["requests"][rid] = {
                "req_id": tr.req_id,
                "worker_name": tr.worker_name,
                "created_monotonic": tr.created_monotonic,
                "ingress_monotonic": tr.ingress_monotonic,
                "first_token_monotonic": tr.first_token_monotonic,
                "finished_monotonic": tr.finished_monotonic,
                "batch_size": tr.batch_size,
                "context_len": tr.context_len,
                "target_len": tr.target_len,
                "generated_tokens": tr.generated_tokens,
                "ttft_ms": _delta_ms(tr.ingress_monotonic, tr.first_token_monotonic),
                "e2e_ms": _delta_ms(tr.ingress_monotonic, tr.finished_monotonic),
                "hops": [asdict(h) for h in tr.hops],
            }
        return out

    def summary(self) -> Dict[str, Any]:
        reqs = list(self._traces.values())
        if not reqs:
            return {
                "request_count": 0,
                "throughput_req_s": 0.0,
                "throughput_token_s": 0.0,
                "avg_ttft_s": None,
                "p95_ttft_s": None,
                "avg_e2e_s": None,
                "p95_e2e_s": None,
            }
        ingress_ts = [x.ingress_monotonic for x in reqs if x.ingress_monotonic is not None]
        finish_ts = [x.finished_monotonic for x in reqs if x.finished_monotonic is not None]
        request_count = len(reqs)
        elapsed_s = None
        if ingress_ts and finish_ts:
            elapsed_s = max(0.0, max(finish_ts) - min(ingress_ts))
        ttfts = [
            _delta_s(x.ingress_monotonic, x.first_token_monotonic)
            for x in reqs
            if x.ingress_monotonic is not None and x.first_token_monotonic is not None
        ]
        e2es = [
            _delta_s(x.ingress_monotonic, x.finished_monotonic)
            for x in reqs
            if x.ingress_monotonic is not None and x.finished_monotonic is not None
        ]
        ttfts = [x for x in ttfts if x is not None]
        e2es = [x for x in e2es if x is not None]
        total_tokens = sum(
            x.generated_tokens
            for x in reqs
            if x.ingress_monotonic is not None and x.finished_monotonic is not None
        )
        throughput_req_s = 0.0
        throughput_token_s = 0.0
        if elapsed_s is not None and elapsed_s > 0:
            throughput_req_s = float(len(finish_ts)) / elapsed_s
            throughput_token_s = float(total_tokens) / elapsed_s
        return {
            "request_count": request_count,
            "throughput_req_s": throughput_req_s,
            "throughput_token_s": throughput_token_s,
            "avg_ttft_s": (sum(ttfts) / len(ttfts)) if ttfts else None,
            "p95_ttft_s": _percentile(ttfts, 0.95),
            "avg_e2e_s": (sum(e2es) / len(e2es)) if e2es else None,
            "p95_e2e_s": _percentile(e2es, 0.95),
        }

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
        self._first_submit_ts: Optional[float] = None
        self._pipeline_closed_ts: Optional[float] = None
        self._total_tokens: int = 0

    def mark_submit(self, req_id: str) -> None:
        ts = time.time()
        self._start[req_id] = ts
        if self._first_submit_ts is None:
            self._first_submit_ts = ts

    def mark_finished(self, req_id: str) -> None:
        st = self._start.pop(req_id, None)
        if st is None:
            return
        end = time.time()
        self._records[req_id] = {"start": st, "end": end, "latency_s": end - st}

    def mark_pipeline_closed(self, total_tokens: int) -> None:
        self._pipeline_closed_ts = time.time()
        self._total_tokens = int(total_tokens)

    def to_serializable(self) -> Dict[str, Any]:
        total_s = _delta_s(self._first_submit_ts, self._pipeline_closed_ts)
        throughput = None if total_s is None or total_s <= 0 else float(self._total_tokens) / total_s
        return {
            "schema_version": "master_latency.v2",
            "pipeline": {
                "first_submit_ts": self._first_submit_ts,
                "closed_ts": self._pipeline_closed_ts,
                "working_time_s": total_s,
                "total_tokens": self._total_tokens,
                "throughput_tokens_per_s": throughput,
            },
            "requests": dict(self._records),
        }

    def export_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_serializable(), f, indent=2, ensure_ascii=False)


def _delta_ms(start: Optional[float], end: Optional[float]) -> Optional[float]:
    if start is None or end is None:
        return None
    return (end - start) * 1000.0


def _delta_s(start: Optional[float], end: Optional[float]) -> Optional[float]:
    if start is None or end is None:
        return None
    return end - start
