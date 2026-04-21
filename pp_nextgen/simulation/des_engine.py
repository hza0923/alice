"""Discrete-event simulator for pipeline strategy execution."""

from __future__ import annotations

import heapq
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pp_nextgen.runtime.strategy import (
    expected_comm_bytes,
    expected_compute_ms,
    pipeline_stage_order,
    stage_has_worker0_head_tail,
    worker_matches_designated_device,
)

from .batching import FCFSContiguousBatchScheduler, PackedBatch
from .metrics import MetricsCollector, SimulationReport
from .request_model import SimRequest

_K_ARRIVAL = "request_arrival"
_K_COMPUTE_DONE = "stage_compute_done"
_K_LINK_DONE = "link_transfer_done"
_K_WORKER0_TAIL_DONE = "worker0_tail_done"


@dataclass(frozen=True)
class DESConfig:
    max_batch_size: int
    packing_window_ms: float
    default_link_bandwidth_gbps: float
    link_bandwidth_overrides: Dict[str, float]
    max_in_flight_requests: int = 512
    max_events: int = 2_000_000


@dataclass
class _BatchContext:
    batch_id: str
    packed: PackedBatch
    decode_steps_after_prefill: int


@dataclass(order=True)
class _Event:
    ts: float
    seq: int
    kind: str
    payload: Dict[str, Any]


class PipelineDESSimulator:
    def __init__(self, *, strategy_doc: Dict[str, Any], config: DESConfig) -> None:
        stages = strategy_doc.get("pipeline_stages") or []
        if not stages:
            raise ValueError("strategy has no pipeline_stages")
        if config.default_link_bandwidth_gbps <= 0:
            raise ValueError("default_link_bandwidth_gbps must be > 0")
        if int(config.max_in_flight_requests) <= 0:
            raise ValueError("max_in_flight_requests must be > 0")

        self.strategy_doc = strategy_doc
        self.config = config
        self.stages: List[Dict[str, Any]] = stages
        self.stage_order: List[str] = pipeline_stage_order(strategy_doc)
        if len(self.stage_order) != len(self.stages):
            raise ValueError("pipeline stage order mismatch")

        self.scheduler = FCFSContiguousBatchScheduler(
            max_batch_size=config.max_batch_size,
            packing_window_ms=config.packing_window_ms,
        )
        self.metrics = MetricsCollector()
        self._event_q: List[_Event] = []
        self._event_seq = 0
        self._batch_counter = 0
        self._stage_available: Dict[str, float] = {w: 0.0 for w in self.stage_order}
        self._link_available: Dict[Tuple[str, str], float] = {}
        self._link_bw_bytes_per_s = self._build_link_bandwidth_map()
        self._active_batches: Dict[str, _BatchContext] = {}
        self._now = 0.0
        self._sim_start_ts = 0.0
        self._running_req_count = 0
        self._worker0_stage = self.stages[0]
        designated = str((strategy_doc.get("schedule_input") or {}).get("designated_device") or "")
        first_worker = str(self._worker0_stage.get("worker_name"))
        self._worker0_head_tail = worker_matches_designated_device(first_worker, designated) and (
            stage_has_worker0_head_tail(self._worker0_stage)
        )
        if self._worker0_head_tail:
            self._stage_available[f"{first_worker}|head"] = 0.0
            self._stage_available[f"{first_worker}|tail"] = 0.0
        self._req_trace: Dict[str, Dict[str, Any]] = {}
        self._stage_trace: Dict[str, List[Dict[str, Any]]] = {}

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

    def _push_event(self, ts: float, kind: str, payload: Dict[str, Any]) -> None:
        self._event_seq += 1
        heapq.heappush(self._event_q, _Event(ts=ts, seq=self._event_seq, kind=kind, payload=payload))

    def _pop_event(self) -> _Event:
        return heapq.heappop(self._event_q)

    def _phase_context_len(self, batch: _BatchContext, phase: str, decode_step: int) -> int:
        if phase == "prefill":
            return batch.packed.context_len
        # decode_step starts from 1 after prefill.
        return batch.packed.context_len + max(0, decode_step)

    def _stage_resource_key(self, stage_idx: int, branch: str) -> str:
        worker = str(self.stages[stage_idx].get("worker_name"))
        if stage_idx == 0 and self._worker0_head_tail and branch in ("head", "tail"):
            return f"{worker}|{branch}"
        return worker

    def _schedule_stage_compute(
        self,
        *,
        batch: _BatchContext,
        stage_idx: int,
        phase: str,
        decode_step: int,
        ready_ts: float,
        branch: str = "single",
        event_kind: str = _K_COMPUTE_DONE,
    ) -> None:
        stage = self.stages[stage_idx]
        res_key = self._stage_resource_key(stage_idx, branch)
        ctx_len = self._phase_context_len(batch, phase, decode_step)
        comp_ms = expected_compute_ms(
            stage,
            phase,
            ctx_len,
            batch.packed.packed_batch_size,
            branch=branch,
        )

        start_ts = max(ready_ts, self._stage_available[res_key])
        end_ts = start_ts + comp_ms / 1000.0
        self._stage_available[res_key] = end_ts
        self.metrics.add_stage_busy(res_key, max(0.0, end_ts - start_ts))
        req_ids = [r.req_id for r in batch.packed.requests]
        self._log_stage_event(
            res_key,
            ready_ts,
            "stage_received",
            req_ids=req_ids,
            batch_id=batch.batch_id,
            phase=phase,
            decode_step=decode_step,
            branch=branch,
        )
        self._log_stage_event(
            res_key,
            start_ts,
            "stage_start",
            req_ids=req_ids,
            batch_id=batch.batch_id,
            phase=phase,
            decode_step=decode_step,
            branch=branch,
        )
        self._log_stage_event(
            res_key,
            end_ts,
            "stage_end",
            req_ids=req_ids,
            batch_id=batch.batch_id,
            phase=phase,
            decode_step=decode_step,
            branch=branch,
        )
        for req in batch.packed.requests:
            self._log_req_event(
                req.req_id,
                ready_ts,
                "stage_received",
                stage=res_key,
                batch_id=batch.batch_id,
                phase=phase,
                decode_step=decode_step,
                branch=branch,
            )
            self._log_req_event(
                req.req_id,
                start_ts,
                "stage_start",
                stage=res_key,
                batch_id=batch.batch_id,
                phase=phase,
                decode_step=decode_step,
                branch=branch,
            )
            self._log_req_event(
                req.req_id,
                end_ts,
                "stage_end",
                stage=res_key,
                batch_id=batch.batch_id,
                phase=phase,
                decode_step=decode_step,
                branch=branch,
            )

        self._push_event(
            end_ts,
            event_kind,
            {
                "batch_id": batch.batch_id,
                "stage_idx": stage_idx,
                "phase": phase,
                "decode_step": decode_step,
                "ctx_len": ctx_len,
                "branch": branch,
            },
        )

    def _schedule_transfer(
        self,
        *,
        batch: _BatchContext,
        stage_idx: int,
        phase: str,
        decode_step: int,
        ctx_len: int,
        ready_ts: float,
        branch: str = "single",
        next_stage_idx: Optional[int] = None,
        next_branch: str = "single",
    ) -> None:
        src = str(self.stages[stage_idx].get("worker_name"))
        if next_stage_idx is None:
            next_stage_idx = stage_idx + 1
        dst = str(self.stages[next_stage_idx].get("worker_name"))
        link_key = (src, dst)
        bytes_to_send = expected_comm_bytes(
            self.stages[stage_idx],
            phase,
            ctx_len,
            batch.packed.packed_batch_size,
            branch=branch,
        )
        bw = self._link_bw_bytes_per_s.get(link_key, 1.0)
        comm_s = max(0.0, float(bytes_to_send) / bw)
        start_ts = max(ready_ts, self._link_available[link_key])
        end_ts = start_ts + comm_s
        self._link_available[link_key] = end_ts
        link_name = f"{src}->{dst}"
        self.metrics.add_link_busy(link_name, max(0.0, end_ts - start_ts))
        req_ids = [r.req_id for r in batch.packed.requests]
        self._log_stage_event(
            src,
            start_ts,
            "transfer_start",
            req_ids=req_ids,
            batch_id=batch.batch_id,
            phase=phase,
            decode_step=decode_step,
            branch=branch,
            link=link_name,
            bytes=bytes_to_send,
        )
        for req in batch.packed.requests:
            self._log_req_event(
                req.req_id,
                start_ts,
                "transfer_start",
                src_stage=src,
                dst_stage=dst,
                batch_id=batch.batch_id,
                phase=phase,
                decode_step=decode_step,
                branch=branch,
                bytes=bytes_to_send,
            )
        self._push_event(
            end_ts,
            _K_LINK_DONE,
            {
                "batch_id": batch.batch_id,
                "next_stage_idx": next_stage_idx,
                "phase": phase,
                "decode_step": decode_step,
                "next_branch": next_branch,
            },
        )

    def _try_dispatch_prefill(self, now_ts: float) -> bool:
        dispatched = False
        first_worker = self.stage_order[0]
        first_head_key = (
            f"{first_worker}|head"
            if self._worker0_head_tail and stage_has_worker0_head_tail(self._worker0_stage)
            else first_worker
        )
        max_run = int(self.config.max_in_flight_requests)
        while (
            now_ts >= self._stage_available[first_head_key]
            and self.scheduler.has_pending()
            and self._running_req_count < max_run
        ):
            packed = self.scheduler.pop_next_batch(now_ts)
            if packed is None:
                break
            self._batch_counter += 1
            batch = _BatchContext(
                batch_id=f"batch-{self._batch_counter}",
                packed=packed,
                decode_steps_after_prefill=max(0, max(r.new_tokens for r in packed.requests) - 1),
            )
            self._active_batches[batch.batch_id] = batch
            for req in packed.requests:
                self.metrics.mark_running_enter(req.req_id, now_ts)
                self._log_req_event(req.req_id, now_ts, "running_enter")
            self._running_req_count += len(packed.requests)
            self._schedule_stage_compute(
                batch=batch,
                stage_idx=0,
                phase="prefill",
                decode_step=0,
                ready_ts=now_ts,
                branch="head" if self._worker0_head_tail else "single",
            )
            dispatched = True
        return dispatched

    def _finish_batch(self, batch: _BatchContext, ts: float) -> None:
        for req in batch.packed.requests:
            self.metrics.mark_finished(req.req_id, ts)
            self._log_req_event(req.req_id, ts, "request_finished")
        self._running_req_count = max(0, self._running_req_count - len(batch.packed.requests))
        self._active_batches.pop(batch.batch_id, None)

    def run(self, requests: List[SimRequest]) -> SimulationReport:
        if not requests:
            raise ValueError("requests cannot be empty")

        self._sim_start_ts = min(r.arrival_ts for r in requests)
        for req in requests:
            self.metrics.mark_arrival(
                req_id=req.req_id,
                arrival_ts=req.arrival_ts,
                batch_size=req.batch_size,
                context_len=req.context_len,
                target_len=req.target_len,
            )
            self._log_req_event(req.req_id, req.arrival_ts, "request_arrival")
            self._push_event(req.arrival_ts, _K_ARRIVAL, {"req": req})

        processed_events = 0
        while self._event_q:
            if processed_events >= self.config.max_events:
                raise RuntimeError("event limit exceeded; aborting DES to avoid infinite loop")
            evt = self._pop_event()
            processed_events += 1
            self._now = evt.ts

            if evt.kind == _K_ARRIVAL:
                req = evt.payload["req"]
                self.scheduler.enqueue(req)
                self._try_dispatch_prefill(self._now)
                continue

            batch_id = str(evt.payload.get("batch_id"))
            batch = self._active_batches.get(batch_id)
            if batch is None:
                continue

            if evt.kind == _K_LINK_DONE:
                nb = str(evt.payload.get("next_branch") or "single")
                self._schedule_stage_compute(
                    batch=batch,
                    stage_idx=int(evt.payload["next_stage_idx"]),
                    phase=str(evt.payload["phase"]),
                    decode_step=int(evt.payload["decode_step"]),
                    ready_ts=self._now,
                    branch=nb,
                    event_kind=_K_WORKER0_TAIL_DONE if nb == "tail" else _K_COMPUTE_DONE,
                )
                continue

            if evt.kind not in (_K_COMPUTE_DONE, _K_WORKER0_TAIL_DONE):
                continue

            stage_idx = int(evt.payload["stage_idx"])
            phase = str(evt.payload["phase"])
            decode_step = int(evt.payload["decode_step"])
            ctx_len = int(evt.payload["ctx_len"])
            branch = str(evt.payload.get("branch") or "single")

            if evt.kind == _K_WORKER0_TAIL_DONE:
                if phase == "prefill":
                    for req in batch.packed.requests:
                        self.metrics.mark_first_token(req.req_id, self._now)
                        self._log_req_event(req.req_id, self._now, "first_token")
                    if batch.decode_steps_after_prefill > 0:
                        self._schedule_stage_compute(
                            batch=batch,
                            stage_idx=0,
                            phase="decode",
                            decode_step=1,
                            ready_ts=self._now,
                            branch="head" if self._worker0_head_tail else "single",
                        )
                    else:
                        self._finish_batch(batch, self._now)
                        self._try_dispatch_prefill(self._now)
                    continue

                if decode_step < batch.decode_steps_after_prefill:
                    self._schedule_stage_compute(
                        batch=batch,
                        stage_idx=0,
                        phase="decode",
                        decode_step=decode_step + 1,
                        ready_ts=self._now,
                        branch="head" if self._worker0_head_tail else "single",
                    )
                else:
                    self._finish_batch(batch, self._now)
                    self._try_dispatch_prefill(self._now)
                continue

            is_last_stage = stage_idx == (len(self.stages) - 1)
            if not is_last_stage:
                self._schedule_transfer(
                    batch=batch,
                    stage_idx=stage_idx,
                    phase=phase,
                    decode_step=decode_step,
                    ctx_len=ctx_len,
                    ready_ts=self._now,
                    branch=branch,
                )
                continue

            if self._worker0_head_tail:
                self._schedule_transfer(
                    batch=batch,
                    stage_idx=stage_idx,
                    phase=phase,
                    decode_step=decode_step,
                    ctx_len=ctx_len,
                    ready_ts=self._now,
                    branch=branch,
                    next_stage_idx=0,
                    next_branch="tail",
                )
                continue

            if phase == "prefill":
                for req in batch.packed.requests:
                    self.metrics.mark_first_token(req.req_id, self._now)
                    self._log_req_event(req.req_id, self._now, "first_token")
                if batch.decode_steps_after_prefill > 0:
                    self._schedule_stage_compute(
                        batch=batch,
                        stage_idx=0,
                        phase="decode",
                        decode_step=1,
                        ready_ts=self._now,
                    )
                else:
                    self._finish_batch(batch, self._now)
                    self._try_dispatch_prefill(self._now)
                continue

            if decode_step < batch.decode_steps_after_prefill:
                self._schedule_stage_compute(
                    batch=batch,
                    stage_idx=0,
                    phase="decode",
                    decode_step=decode_step + 1,
                    ready_ts=self._now,
                )
            else:
                self._finish_batch(batch, self._now)
                self._try_dispatch_prefill(self._now)

        cfg: Dict[str, Any] = {
            "schedule_input": dict(self.strategy_doc.get("schedule_input") or {}),
            "max_batch_size": self.config.max_batch_size,
            "packing_window_ms": self.config.packing_window_ms,
            "max_in_flight_requests": int(self.config.max_in_flight_requests),
            "default_link_bandwidth_gbps": self.config.default_link_bandwidth_gbps,
            "link_bandwidth_overrides_gbps": dict(self.config.link_bandwidth_overrides),
            "stage_order": list(self.stage_order),
        }
        return self.metrics.build_report(
            config=cfg,
            start_ts=self._sim_start_ts,
            end_ts=self._now,
        )

    def export_traces(self, *, req_trace_path: str | Path, stage_trace_path: str | Path) -> None:
        req_p = Path(req_trace_path)
        req_p.parent.mkdir(parents=True, exist_ok=True)
        req_doc = {
            "schema_version": "pipeline_des_req_trace.v1",
            "requests": sorted(self._req_trace.values(), key=lambda x: x["req_id"]),
        }
        with req_p.open("w", encoding="utf-8") as f:
            json.dump(req_doc, f, indent=2, ensure_ascii=False)

        stage_p = Path(stage_trace_path)
        stage_p.parent.mkdir(parents=True, exist_ok=True)
        stage_doc = {
            "schema_version": "pipeline_des_stage_trace.v1",
            "stages": {k: v for k, v in sorted(self._stage_trace.items(), key=lambda it: it[0])},
        }
        with stage_p.open("w", encoding="utf-8") as f:
            json.dump(stage_doc, f, indent=2, ensure_ascii=False)
