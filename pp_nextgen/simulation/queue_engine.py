"""Queue-driven simulator that schedules from worker0 queues."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from pp_nextgen.runtime.strategy import (
    expected_comm_bytes,
    expected_compute_ms,
    pipeline_stage_order,
)

from .metrics import MetricsCollector, SimulationReport
from .request_model import SimRequest


@dataclass(frozen=True)
class QueueSimConfig:
    max_batch_size: int
    default_link_bandwidth_gbps: float
    link_bandwidth_overrides: Dict[str, float]


@dataclass(frozen=True)
class _WorkItem:
    req: SimRequest
    phase: str  # "prefill" | "decode"
    decode_step: int
    ready_ts: float


class Worker0QueueSimulator:
    """Pipeline simulator driven by worker0 prefill/decode queues."""

    def __init__(self, *, strategy_doc: Dict[str, Any], config: QueueSimConfig) -> None:
        stages = strategy_doc.get("pipeline_stages") or []
        if not stages:
            raise ValueError("strategy has no pipeline_stages")
        if config.max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if config.default_link_bandwidth_gbps <= 0:
            raise ValueError("default_link_bandwidth_gbps must be > 0")

        self.strategy_doc = strategy_doc
        self.config = config
        self.stages: List[Dict[str, Any]] = stages
        self.stage_order: List[str] = pipeline_stage_order(strategy_doc)
        if len(self.stage_order) != len(self.stages):
            raise ValueError("pipeline stage order mismatch")

        self.metrics = MetricsCollector()
        self._stage_available: Dict[str, float] = {w: 0.0 for w in self.stage_order}
        self._link_available: Dict[Tuple[str, str], float] = {}
        self._link_bw_bytes_per_s = self._build_link_bandwidth_map()
        self._sim_start_ts = 0.0
        self._sim_end_ts = 0.0

        self._req_trace: Dict[str, Dict[str, Any]] = {}
        self._stage_trace: Dict[str, List[Dict[str, Any]]] = {}
        self._next_decode_step: Dict[str, int] = {}
        self._decode_steps_after_prefill: Dict[str, int] = {}

    def _ensure_req_trace(self, req_id: str) -> Dict[str, Any]:
        item = self._req_trace.get(req_id)
        if item is None:
            item = {"req_id": req_id, "events": []}
            self._req_trace[req_id] = item
        return item

    def _log_req_event(self, req_id: str, ts: float, event: str, **kwargs: Any) -> None:
        item = self._ensure_req_trace(req_id)
        payload = {"ts": float(ts), "event": event}
        payload.update(kwargs)
        item["events"].append(payload)

    def _log_stage_event(self, stage_key: str, ts: float, event: str, **kwargs: Any) -> None:
        arr = self._stage_trace.setdefault(stage_key, [])
        payload = {"ts": float(ts), "event": event}
        payload.update(kwargs)
        arr.append(payload)

    def _build_link_bandwidth_map(self) -> Dict[Tuple[str, str], float]:
        out: Dict[Tuple[str, str], float] = {}
        default_bytes_per_s = float(self.config.default_link_bandwidth_gbps) * 1_000_000_000.0
        n = len(self.stages)
        for idx, stage in enumerate(self.stages):
            src = str(stage.get("worker_name"))
            dst = str(self.stages[(idx + 1) % n].get("worker_name"))
            bw = default_bytes_per_s
            ref_bytes = float(stage.get("comm_bytes_to_next", 0.0))
            ref_ms = float(stage.get("comm_time_ms", 0.0))
            if ref_bytes > 0 and ref_ms > 0:
                bw = ref_bytes / (ref_ms / 1000.0)
            ov_key = f"{src}->{dst}"
            if ov_key in self.config.link_bandwidth_overrides:
                bw = float(self.config.link_bandwidth_overrides[ov_key]) * 1_000_000_000.0
            out[(src, dst)] = max(1.0, bw)
            self._link_available[(src, dst)] = 0.0
        return out

    @staticmethod
    def _ctx_len_for(item: _WorkItem) -> int:
        if item.phase == "prefill":
            return int(item.req.context_len)
        return int(item.req.context_len) + max(0, int(item.decode_step))

    @staticmethod
    def _expected_compute_ms_single_merged(
        stage: Dict[str, Any], phase: str, ctx_len: int, batch_size: int
    ) -> float:
        try:
            return expected_compute_ms(stage, phase, ctx_len, batch_size, branch="single")
        except ValueError:
            # For designated worker0 exports that only have head/tail branches,
            # approximate merged-single by summing head and tail times.
            head = expected_compute_ms(stage, phase, ctx_len, batch_size, branch="head")
            tail = expected_compute_ms(stage, phase, ctx_len, batch_size, branch="tail")
            return head + tail

    @staticmethod
    def _expected_comm_bytes_single_merged(
        stage: Dict[str, Any], phase: str, ctx_len: int, batch_size: int
    ) -> int:
        try:
            return expected_comm_bytes(stage, phase, ctx_len, batch_size, branch="single")
        except ValueError:
            # Head branch carries the outgoing tensor for worker0 split exports.
            return expected_comm_bytes(stage, phase, ctx_len, batch_size, branch="head")

    def _simulate_item(self, item: _WorkItem) -> float:
        req = item.req
        req_ids = [req.req_id]
        ctx_len = self._ctx_len_for(item)
        arrival_ts = float(item.ready_ts)

        next_input_ready = arrival_ts
        for idx, stage in enumerate(self.stages):
            stage_name = str(stage.get("worker_name"))
            comp_ms = self._expected_compute_ms_single_merged(stage, item.phase, ctx_len, int(req.batch_size))
            start_ts = max(next_input_ready, self._stage_available[stage_name])
            end_ts = start_ts + comp_ms / 1000.0
            self._stage_available[stage_name] = end_ts
            self.metrics.add_stage_busy(stage_name, max(0.0, end_ts - start_ts))

            self._log_stage_event(
                stage_name,
                next_input_ready,
                "stage_received",
                req_ids=req_ids,
                phase=item.phase,
                decode_step=item.decode_step,
            )
            self._log_stage_event(
                stage_name,
                start_ts,
                "stage_start",
                req_ids=req_ids,
                phase=item.phase,
                decode_step=item.decode_step,
            )
            self._log_stage_event(
                stage_name,
                end_ts,
                "stage_end",
                req_ids=req_ids,
                phase=item.phase,
                decode_step=item.decode_step,
            )
            self._log_req_event(
                req.req_id,
                next_input_ready,
                "stage_received",
                stage=stage_name,
                phase=item.phase,
                decode_step=item.decode_step,
            )
            self._log_req_event(
                req.req_id,
                start_ts,
                "stage_start",
                stage=stage_name,
                phase=item.phase,
                decode_step=item.decode_step,
            )
            self._log_req_event(
                req.req_id,
                end_ts,
                "stage_end",
                stage=stage_name,
                phase=item.phase,
                decode_step=item.decode_step,
            )

            if item.phase == "prefill" and idx == 0:
                self.metrics.mark_service_start(req.req_id, start_ts)
                self._log_req_event(req.req_id, start_ts, "service_start")

            is_last = idx == (len(self.stages) - 1)
            if is_last:
                next_input_ready = end_ts
                continue

            next_stage_name = str(self.stages[idx + 1].get("worker_name"))
            link_key = (stage_name, next_stage_name)
            bytes_to_send = self._expected_comm_bytes_single_merged(stage, item.phase, ctx_len, int(req.batch_size))
            bw = self._link_bw_bytes_per_s.get(link_key, 1.0)
            comm_s = max(0.0, float(bytes_to_send) / bw)
            tx_start = max(end_ts, self._link_available[link_key])
            tx_end = tx_start + comm_s
            self._link_available[link_key] = tx_end
            link_name = f"{stage_name}->{next_stage_name}"
            self.metrics.add_link_busy(link_name, max(0.0, tx_end - tx_start))

            self._log_stage_event(
                stage_name,
                tx_start,
                "transfer_start",
                req_ids=req_ids,
                phase=item.phase,
                decode_step=item.decode_step,
                link=link_name,
                bytes=bytes_to_send,
            )
            self._log_req_event(
                req.req_id,
                tx_start,
                "transfer_start",
                src_stage=stage_name,
                dst_stage=next_stage_name,
                phase=item.phase,
                decode_step=item.decode_step,
                bytes=bytes_to_send,
            )
            next_input_ready = tx_end

        return next_input_ready

    def run(self, requests: List[SimRequest]) -> SimulationReport:
        if not requests:
            raise ValueError("requests cannot be empty")

        reqs = sorted(requests, key=lambda r: (r.arrival_ts, r.req_id))
        self._sim_start_ts = min(r.arrival_ts for r in reqs)

        pending_prefill: Deque[_WorkItem] = deque()
        pending_decode: Deque[_WorkItem] = deque()

        arrival_idx = 0
        n = len(reqs)
        completed = 0

        for req in reqs:
            if int(req.batch_size) > int(self.config.max_batch_size):
                raise ValueError(
                    f"single request batch_size={req.batch_size} exceeds max_batch_size={self.config.max_batch_size}"
                )
            self.metrics.mark_arrival(
                req_id=req.req_id,
                arrival_ts=req.arrival_ts,
                batch_size=req.batch_size,
                context_len=req.context_len,
                target_len=req.target_len,
            )
            self._log_req_event(req.req_id, req.arrival_ts, "request_arrival")
            self._decode_steps_after_prefill[req.req_id] = max(0, int(req.new_tokens) - 1)
            self._next_decode_step[req.req_id] = 1

        worker0 = self.stage_order[0]

        def load_arrivals(until_ts: float) -> None:
            nonlocal arrival_idx
            while arrival_idx < n and reqs[arrival_idx].arrival_ts <= until_ts:
                req = reqs[arrival_idx]
                pending_prefill.append(_WorkItem(req=req, phase="prefill", decode_step=0, ready_ts=req.arrival_ts))
                arrival_idx += 1

        while completed < n:
            worker0_available = self._stage_available[worker0]
            load_arrivals(worker0_available)

            ready_prefill = bool(pending_prefill and pending_prefill[0].ready_ts <= worker0_available)
            ready_decode = bool(pending_decode and pending_decode[0].ready_ts <= worker0_available)

            item: Optional[_WorkItem] = None
            if ready_prefill:
                item = pending_prefill.popleft()
            elif ready_decode:
                item = pending_decode.popleft()
            else:
                next_arrival_ts = reqs[arrival_idx].arrival_ts if arrival_idx < n else None
                next_decode_ts = pending_decode[0].ready_ts if pending_decode else None
                cands = [x for x in [next_arrival_ts, next_decode_ts] if x is not None]
                if not cands:
                    raise RuntimeError("no remaining work but simulation is not complete")
                target_ts = min(cands)
                load_arrivals(target_ts)
                if pending_prefill and pending_prefill[0].ready_ts <= target_ts:
                    item = pending_prefill.popleft()
                elif pending_decode and pending_decode[0].ready_ts <= target_ts:
                    item = pending_decode.popleft()
                else:
                    continue

            done_ts = self._simulate_item(item)
            self._sim_end_ts = max(self._sim_end_ts, done_ts)

            req_id = item.req.req_id
            if item.phase == "prefill":
                self.metrics.mark_first_token(req_id, done_ts)
                self._log_req_event(req_id, done_ts, "first_token")
                if self._decode_steps_after_prefill[req_id] > 0:
                    pending_decode.append(
                        _WorkItem(
                            req=item.req,
                            phase="decode",
                            decode_step=self._next_decode_step[req_id],
                            ready_ts=done_ts,
                        )
                    )
                else:
                    self.metrics.mark_finished(req_id, done_ts)
                    self._log_req_event(req_id, done_ts, "request_finished")
                    completed += 1
            else:
                next_step = self._next_decode_step[req_id] + 1
                self._next_decode_step[req_id] = next_step
                if item.decode_step < self._decode_steps_after_prefill[req_id]:
                    pending_decode.append(
                        _WorkItem(
                            req=item.req,
                            phase="decode",
                            decode_step=next_step,
                            ready_ts=done_ts,
                        )
                    )
                else:
                    self.metrics.mark_finished(req_id, done_ts)
                    self._log_req_event(req_id, done_ts, "request_finished")
                    completed += 1

        cfg: Dict[str, Any] = {
            "schedule_input": dict(self.strategy_doc.get("schedule_input") or {}),
            "max_batch_size": self.config.max_batch_size,
            "default_link_bandwidth_gbps": self.config.default_link_bandwidth_gbps,
            "link_bandwidth_overrides_gbps": dict(self.config.link_bandwidth_overrides),
            "stage_order": list(self.stage_order),
            "simulator_type": "worker0_queue",
            "priority_policy": "prefill_first_then_fcfs",
        }
        return self.metrics.build_report(config=cfg, start_ts=self._sim_start_ts, end_ts=self._sim_end_ts)

    def export_traces(self, *, req_trace_path: str | Path, stage_trace_path: str | Path) -> None:
        req_p = Path(req_trace_path)
        req_p.parent.mkdir(parents=True, exist_ok=True)
        req_doc = {
            "schema_version": "pipeline_queue_req_trace.v1",
            "requests": sorted(self._req_trace.values(), key=lambda x: x["req_id"]),
        }
        with req_p.open("w", encoding="utf-8") as f:
            json.dump(req_doc, f, indent=2, ensure_ascii=False)

        stage_p = Path(stage_trace_path)
        stage_p.parent.mkdir(parents=True, exist_ok=True)
        stage_doc = {
            "schema_version": "pipeline_queue_stage_trace.v1",
            "stages": {k: v for k, v in sorted(self._stage_trace.items(), key=lambda it: it[0])},
        }
        with stage_p.open("w", encoding="utf-8") as f:
            json.dump(stage_doc, f, indent=2, ensure_ascii=False)
