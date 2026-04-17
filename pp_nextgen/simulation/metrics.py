"""Metrics collection and report generation for DES."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


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


@dataclass
class RequestMetrics:
    req_id: str
    arrival_ts: float
    service_start_ts: Optional[float] = None
    first_token_ts: Optional[float] = None
    finish_ts: Optional[float] = None
    batch_size: int = 1
    context_len: int = 0
    target_len: int = 0
    generated_tokens: int = 0

    def ttft_s(self) -> Optional[float]:
        if self.first_token_ts is None:
            return None
        return self.first_token_ts - self.arrival_ts

    def e2e_latency_s(self) -> Optional[float]:
        if self.finish_ts is None:
            return None
        return self.finish_ts - self.arrival_ts

    def queue_wait_s(self) -> Optional[float]:
        if self.service_start_ts is None:
            return None
        return self.service_start_ts - self.arrival_ts

    def to_dict(self) -> Dict[str, Any]:
        return {
            "req_id": self.req_id,
            "arrival_ts": self.arrival_ts,
            "service_start_ts": self.service_start_ts,
            "first_token_ts": self.first_token_ts,
            "finish_ts": self.finish_ts,
            "batch_size": self.batch_size,
            "context_len": self.context_len,
            "target_len": self.target_len,
            "generated_tokens": self.generated_tokens,
            "ttft_s": self.ttft_s(),
            "e2e_latency_s": self.e2e_latency_s(),
            "queue_wait_s": self.queue_wait_s(),
        }


@dataclass
class SimulationSummary:
    request_count: int
    completed_count: int
    start_ts: float
    end_ts: float
    elapsed_s: float
    throughput_req_s: float
    throughput_token_s: float
    avg_ttft_s: Optional[float]
    p95_ttft_s: Optional[float]
    avg_e2e_s: Optional[float]
    p95_e2e_s: Optional[float]
    avg_queue_wait_s: Optional[float]
    p95_queue_wait_s: Optional[float]


@dataclass
class SimulationReport:
    config: Dict[str, Any]
    summary: SimulationSummary
    requests: List[RequestMetrics]
    stage_utilization: Dict[str, float]
    link_utilization: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": "pipeline_des_metrics.v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": self.config,
            "summary": asdict(self.summary),
            "stage_utilization": self.stage_utilization,
            "link_utilization": self.link_utilization,
            "requests": [r.to_dict() for r in self.requests],
        }

    def export_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)


class MetricsCollector:
    def __init__(self) -> None:
        self._requests: Dict[str, RequestMetrics] = {}
        self._stage_busy: Dict[str, float] = {}
        self._link_busy: Dict[str, float] = {}

    def ensure_request(self, req_id: str, arrival_ts: float) -> RequestMetrics:
        if req_id not in self._requests:
            self._requests[req_id] = RequestMetrics(req_id=req_id, arrival_ts=arrival_ts)
        return self._requests[req_id]

    def mark_arrival(self, *, req_id: str, arrival_ts: float, batch_size: int, context_len: int, target_len: int) -> None:
        item = self.ensure_request(req_id, arrival_ts=arrival_ts)
        item.batch_size = batch_size
        item.context_len = context_len
        item.target_len = target_len
        item.generated_tokens = int(batch_size) * max(0, int(target_len) - int(context_len))

    def mark_service_start(self, req_id: str, service_start_ts: float) -> None:
        item = self.ensure_request(req_id, arrival_ts=service_start_ts)
        if item.service_start_ts is None:
            item.service_start_ts = service_start_ts

    def mark_first_token(self, req_id: str, first_token_ts: float) -> None:
        item = self.ensure_request(req_id, arrival_ts=first_token_ts)
        if item.first_token_ts is None:
            item.first_token_ts = first_token_ts

    def mark_finished(self, req_id: str, finish_ts: float) -> None:
        item = self.ensure_request(req_id, arrival_ts=finish_ts)
        item.finish_ts = finish_ts

    def add_stage_busy(self, stage_name: str, duration_s: float) -> None:
        self._stage_busy[stage_name] = self._stage_busy.get(stage_name, 0.0) + max(0.0, duration_s)

    def add_link_busy(self, link_name: str, duration_s: float) -> None:
        self._link_busy[link_name] = self._link_busy.get(link_name, 0.0) + max(0.0, duration_s)

    def build_report(self, *, config: Dict[str, Any], start_ts: float, end_ts: float) -> SimulationReport:
        reqs = sorted(self._requests.values(), key=lambda x: x.arrival_ts)
        elapsed = max(0.0, end_ts - start_ts)
        completed = [r for r in reqs if r.finish_ts is not None]
        ttfts = [x.ttft_s() for x in completed if x.ttft_s() is not None]
        e2es = [x.e2e_latency_s() for x in completed if x.e2e_latency_s() is not None]
        waits = [x.queue_wait_s() for x in completed if x.queue_wait_s() is not None]
        total_tokens = sum(r.generated_tokens for r in completed)

        stage_util = {
            k: (v / elapsed if elapsed > 0 else 0.0)
            for k, v in sorted(self._stage_busy.items(), key=lambda it: it[0])
        }
        link_util = {
            k: (v / elapsed if elapsed > 0 else 0.0)
            for k, v in sorted(self._link_busy.items(), key=lambda it: it[0])
        }

        summary = SimulationSummary(
            request_count=len(reqs),
            completed_count=len(completed),
            start_ts=start_ts,
            end_ts=end_ts,
            elapsed_s=elapsed,
            throughput_req_s=(len(completed) / elapsed if elapsed > 0 else 0.0),
            throughput_token_s=(total_tokens / elapsed if elapsed > 0 else 0.0),
            avg_ttft_s=(sum(ttfts) / len(ttfts) if ttfts else None),
            p95_ttft_s=_percentile(ttfts, 0.95),
            avg_e2e_s=(sum(e2es) / len(e2es) if e2es else None),
            p95_e2e_s=_percentile(e2es, 0.95),
            avg_queue_wait_s=(sum(waits) / len(waits) if waits else None),
            p95_queue_wait_s=_percentile(waits, 0.95),
        )
        return SimulationReport(
            config=config,
            summary=summary,
            requests=reqs,
            stage_utilization=stage_util,
            link_utilization=link_util,
        )
