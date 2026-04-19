"""gRPC worker: strategy-driven sleep/shape execution, ring forwarding, metrics."""

from __future__ import annotations

import asyncio
import inspect
import logging
import time
from typing import Any, Dict, Optional, Set, Tuple

import grpc

from pp_nextgen.runtime.config import RuntimeConfig, sleep_compute_offset_ms_for_worker
from pp_nextgen.runtime.executors.shape import ShapeExecutor
from pp_nextgen.runtime.executors.sleep import SleepExecutor
from pp_nextgen.runtime.grpc_gen import pipeline_v2_pb2 as pv2
from pp_nextgen.runtime.grpc_gen import pipeline_v2_pb2_grpc as pv2_grpc
from pp_nextgen.runtime.metrics import HopRecord, RequestJournal, avg_bandwidth_mbps
from pp_nextgen.runtime.model import SleepPipelineModel
from pp_nextgen.runtime.strategy import (
    decoder_submodules_from_model_yaml,
    expand_decoder_layer_placeholders,
    expected_comm_bytes,
    expected_comm_ms,
    find_stage_for_worker,
    linear_next_worker,
    load_model_yaml,
    load_pipeline_strategy,
    load_worker_strategy,
    merge_model_for_runtime,
    next_worker_name,
    pipeline_stage_order,
    split_head_tail_modules_from_execution_plan,
)

LOG = logging.getLogger("pp_nextgen.runtime.worker")


def _phase_name(frame: pv2.Frame) -> str:
    if frame.phase == pv2.PHASE_PREFILL:
        return "prefill"
    return "decode"


def _copy_frame(src: pv2.Frame, **kwargs: Any) -> pv2.Frame:
    out = pv2.Frame()
    out.CopyFrom(src)
    for k, v in kwargs.items():
        setattr(out, k, v)
    return out


class DataPlaneServicer(pv2_grpc.DataPlaneServicer):
    def __init__(
        self,
        worker_name: str,
        master_addr: str,
        pipeline_path: str,
        worker_strategy_path: str,
        model_yaml_path: str,
        rt: RuntimeConfig,
        metrics_out: Optional[str] = None,
    ) -> None:
        self.worker_name = worker_name
        self.master_addr = master_addr
        self._rt = rt
        self._metrics_out = metrics_out
        self._pipeline_doc = load_pipeline_strategy(pipeline_path)
        self._worker_doc = load_worker_strategy(worker_strategy_path)
        self._stage = find_stage_for_worker(self._pipeline_doc, worker_name)
        if self._stage is None:
            raise ValueError(f"worker {worker_name} not found in pipeline strategy")
        self._order = pipeline_stage_order(self._pipeline_doc)
        self._first_worker = self._order[0] if self._order else ""
        self._is_first_worker = worker_name == self._first_worker
        model_yaml_doc = load_model_yaml(model_yaml_path)
        self._merged_model = merge_model_for_runtime(self._pipeline_doc, model_yaml_doc)
        exec_plan = (self._worker_doc.get("execution_plan") or {}) if self._worker_doc else {}
        head_mods = list(exec_plan.get("head_ordered_modules") or [])
        tail_mods = list(exec_plan.get("tail_ordered_modules") or [])

        use_shape = rt.execution_mode == "shape_executor"
        dec_sub = decoder_submodules_from_model_yaml(model_yaml_doc)
        if use_shape:
            from pp_nextgen.runtime.shape_pipeline_model import ShapeTorchPipelineModel

        if self._is_first_worker:
            if not head_mods and not tail_mods:
                head_mods, tail_mods = split_head_tail_modules_from_execution_plan(
                    exec_plan,
                    worker_name=self.worker_name,
                    is_first_worker=True,
                    pipeline_doc=self._pipeline_doc,
                )
            if use_shape:
                hm = expand_decoder_layer_placeholders(head_mods, dec_sub)
                tm = expand_decoder_layer_placeholders(tail_mods, dec_sub)
                self._model = ShapeTorchPipelineModel.from_configs(
                    self._merged_model,
                    head_ordered_modules=hm,
                    tail_ordered_modules=tm if tm else None,
                    rt=rt,
                )
            else:
                self._model = SleepPipelineModel(has_tail=bool(tail_mods))
        else:
            if not head_mods and not tail_mods:
                combined, _ = split_head_tail_modules_from_execution_plan(
                    exec_plan,
                    worker_name=self.worker_name,
                    is_first_worker=False,
                    pipeline_doc=self._pipeline_doc,
                )
            else:
                combined = head_mods + tail_mods
            if use_shape:
                combined = expand_decoder_layer_placeholders(combined, dec_sub)
                self._model = ShapeTorchPipelineModel.from_configs(
                    self._merged_model,
                    ordered_modules=combined,
                    rt=rt,
                )
            else:
                self._model = SleepPipelineModel(has_tail=False)

        if use_shape:
            self._executor: Any = ShapeExecutor()
        else:
            self._executor = SleepExecutor(
                compute_sleep_offset_ms=sleep_compute_offset_ms_for_worker(worker_name, rt),
            )

        self._journal = RequestJournal(worker_name)
        maxsz = rt.task_queue_maxsize
        if self._is_first_worker:
            self._head_queue: asyncio.Queue[pv2.Frame] = asyncio.Queue(maxsize=maxsz)
            self._tail_queue: asyncio.Queue[pv2.Frame] = asyncio.Queue(maxsize=maxsz)
        else:
            self._task_queue: asyncio.Queue[pv2.Frame] = asyncio.Queue(maxsize=maxsz)

        self._send_queue: asyncio.Queue[Tuple[pv2.Frame, Optional[HopRecord]]] = asyncio.Queue(
            maxsize=rt.send_queue_maxsize
        )
        self._done = asyncio.Event()
        self._master: Optional[pv2_grpc.MasterControlStub] = None
        self._next_stub: Optional[pv2_grpc.DataPlaneStub] = None
        self._active_reqs: Set[str] = set()
        self._open_requests: Set[str] = set()
        self._total_tokens = 0
        self._pending_pipeline_stop: Optional[pv2.Frame] = None
        self._outbound_xfer_bytes: int = 0
        self._outbound_xfer_ms: float = 0.0

    def _scaled_payload_byte_len(self, nominal_bytes: int) -> int:
        div = float(self._rt.payload_size_divisor)
        if div <= 0:
            div = 1.0
        return max(0, int(float(nominal_bytes) / div))

    def _transfer_summary_dict(self) -> Dict[str, Any]:
        total_s = self._outbound_xfer_ms / 1000.0
        mbps = avg_bandwidth_mbps(int(self._outbound_xfer_bytes), total_s)
        return {
            "total_payload_bytes_sent": int(self._outbound_xfer_bytes),
            "total_transfer_time_ms": float(self._outbound_xfer_ms),
            "total_transfer_time_s": float(total_s),
            "avg_bandwidth_mbps": mbps,
        }

    async def initialize(self) -> bool:
        ch = grpc.aio.insecure_channel(self.master_addr)
        self._master = pv2_grpc.MasterControlStub(ch)
        reg = await self._master.RegisterWorker(
            pv2.RegisterWorkerRequest(worker_name=self.worker_name, worker_address=self._public_addr),
            timeout=float(self._rt.rpc_timeout_ms) / 1000.0,
        )
        if not reg.ok:
            LOG.error("register failed: %s", reg.message)
            return False
        reg_wait_s = float(self._rt.registration_wait_timeout_ms) / 1000.0
        while True:
            nxt = await self._master.GetNextWorker(
                pv2.NextWorkerRequest(worker_name=self.worker_name),
                timeout=reg_wait_s,
            )
            if nxt.ok and nxt.all_registered and nxt.has_next and nxt.next_worker_address:
                nch = grpc.aio.insecure_channel(nxt.next_worker_address)
                self._next_stub = pv2_grpc.DataPlaneStub(nch)
                LOG.info("next hop %s", nxt.next_worker_address)
                break
            await asyncio.sleep(0.2)
        await self._initialize_executor()
        self._prewarm_model_layers()
        if self._is_first_worker:
            asyncio.create_task(self._task_processor_worker0())
        else:
            asyncio.create_task(self._task_processor())
        asyncio.create_task(self._send_processor())
        return True

    async def _initialize_executor(self) -> None:
        init_fn = getattr(self._executor, "initialize", None)
        if init_fn is None:
            LOG.info("executor %s initialization skipped (no-op)", self._executor.__class__.__name__)
            return
        ret = init_fn()
        if inspect.isawaitable(ret):
            await ret
        LOG.info("executor %s initialization done", self._executor.__class__.__name__)

    def _prewarm_model_layers(self) -> None:
        """Build decode module chain during initialization to avoid first-request lazy cost."""
        if isinstance(self._model, SleepPipelineModel):
            LOG.info("sleep executor: no compute pipeline to prewarm")
            return
        self._model.init_layers()
        LOG.info("compute pipeline prewarmed (%s)", type(self._model).__name__)

    def set_public_address(self, addr: str) -> None:
        self._public_addr = addr

    async def SendFrame(self, request: pv2.Frame, context: grpc.aio.ServicerContext) -> pv2.Ack:
        if self._is_first_worker:
            # Without tail modules, all ingress (master + ring return) uses one head queue + head decode.
            if not self._model.has_tail:
                await self._head_queue.put(request)
            elif request.ring_return:
                await self._tail_queue.put(request)
            else:
                await self._head_queue.put(request)
        else:
            await self._task_queue.put(request)
        return pv2.Ack(ok=True, message="")

    async def NotifyPipelineEnd(
        self, request: pv2.PipelineEndNotification, context: grpc.aio.ServicerContext
    ) -> pv2.Ack:
        LOG.info("NotifyPipelineEnd: %s ring_remaining=%s", request.message, request.ring_remaining)
        r = int(request.ring_remaining or 0)
        if r > 1 and self._master:
            nxt = linear_next_worker(self._order, self.worker_name)
            if nxt:
                try:
                    peer = await self._master.GetPeerAddress(
                        pv2.PeerAddressRequest(worker_name=nxt),
                        timeout=float(self._rt.rpc_timeout_ms) / 1000.0,
                    )
                    if peer.ok and peer.address:
                        stub = pv2_grpc.DataPlaneStub(grpc.aio.insecure_channel(peer.address))
                        await stub.NotifyPipelineEnd(
                            pv2.PipelineEndNotification(
                                message=request.message,
                                ring_remaining=r - 1,
                            ),
                            timeout=float(self._rt.rpc_timeout_ms) / 1000.0,
                        )
                except Exception as e:
                    LOG.error("forward NotifyPipelineEnd failed: %s", e)
        self._done.set()
        self._model.close_all_kv_sessions()
        if self._metrics_out:
            self._journal.export_json(self._metrics_out, transfer_summary=self._transfer_summary_dict())
        return pv2.Ack(ok=True, message="")

    async def _send_processor(self) -> None:
        timeout = float(self._rt.rpc_timeout_ms) / 1000.0
        while not self._done.is_set():
            frame, hop = await self._send_queue.get()
            if self._next_stub:
                try:
                    t0 = time.perf_counter()
                    await self._next_stub.SendFrame(frame, timeout=timeout)
                    dt = (time.perf_counter() - t0) * 1000.0
                    self._outbound_xfer_bytes += len(frame.payload)
                    self._outbound_xfer_ms += dt
                    if hop is not None:
                        hop.actual_comm_ms = dt
                except Exception as e:
                    LOG.error("SendFrame failed: %s", e)
            self._send_queue.task_done()

    async def _task_processor(self) -> None:
        assert self._master is not None
        while not self._done.is_set():
            frame = await self._task_queue.get()
            try:
                await self._handle_middle_frame(frame)
            finally:
                self._task_queue.task_done()

    async def _task_processor_worker0(self) -> None:
        assert self._master is not None
        if not self._model.has_tail: # 如果没有尾模块，则只处理头模块
            while not self._done.is_set():
                await self._maybe_flush_pending_stop()
                frame = await self._head_queue.get()
                try:
                    await self._handle_worker0_head_frame(frame)
                finally:
                    self._head_queue.task_done()
            return
        while not self._done.is_set():
            await self._maybe_flush_pending_stop()
            processed = False
            if not self._head_queue.empty():
                frame = await self._head_queue.get()
                try:
                    await self._handle_worker0_head_frame(frame)
                finally:
                    self._head_queue.task_done()
                processed = True
            elif not self._tail_queue.empty():
                frame = await self._tail_queue.get()
                try:
                    await self._handle_worker0_tail_frame(frame)
                finally:
                    self._tail_queue.task_done()
                processed = True
            if not processed:
                await asyncio.sleep(0)

    async def _maybe_flush_pending_stop(self) -> None:
        if self._pending_pipeline_stop is None:
            return
        # Do not flush until every request has reported finished (tail may not have run yet).
        if self._open_requests:
            return
        if not self._head_queue.empty():
            return
        if self._model.has_tail and not self._tail_queue.empty():
            return
        if self._send_queue.qsize() > 0:
            return
        f = self._pending_pipeline_stop
        self._pending_pipeline_stop = None
        if len(self._order) == 1:
            to = float(self._rt.rpc_timeout_ms) / 1000.0
            await self._master.ReportPipelineComplete(
                pv2.PipelineCompleteReport(total_tokens=self._total_tokens),
                timeout=to,
            )
            return
        await self._send_queue.put((f, None))

    def _next_frame_towards_peer(self, frame: pv2.Frame) -> pv2.Frame:
        nxt = next_worker_name(self._pipeline_doc, self.worker_name)
        rr = bool(frame.ring_return)
        if nxt == self._first_worker:
            rr = True
        return _copy_frame(frame, ring_return=rr)

    async def _handle_middle_frame(self, frame: pv2.Frame) -> None:
        assert self._master is not None
        if frame.pipeline_stop:
            await self._forward_pipeline_stop_ring(frame)
            return

        self._model.ensure_kv_session(
            frame.req_id,
            int(frame.batch_size or 1),
            int(frame.target_len),
        )
        ph = _phase_name(frame)
        exp_ms, act_ms = await self._executor.run(
            frame, self._stage, self._merged_model, self._model, branch="single"
        )
        exp_bytes = expected_comm_bytes(
            self._stage, ph, int(frame.context_len), int(frame.batch_size or 1), branch="single"
        )
        exp_cm = expected_comm_ms(
            self._stage,
            phase_name=ph,
            context_len=int(frame.context_len),
            branch="single",
        )
        payload = b"x" * self._scaled_payload_byte_len(exp_bytes)
        out = self._next_frame_towards_peer(
            _copy_frame(
                frame,
                step_id=frame.step_id + 1,
                payload=payload,
                pipeline_stop=False,
            )
        )
        hop = HopRecord(
            step_id=int(frame.step_id),
            phase=ph,
            expected_compute_ms=exp_ms,
            actual_compute_ms=act_ms,
            expected_comm_bytes=exp_bytes,
            expected_comm_ms=exp_cm,
            actual_comm_ms=0.0,
            payload_bytes_sent=len(payload),
        )
        self._journal.record_hop(frame.req_id, hop)
        await self._send_queue.put((out, hop))

    async def _forward_pipeline_stop_ring(self, frame: pv2.Frame) -> None:
        nxt = next_worker_name(self._pipeline_doc, self.worker_name)
        r = int(frame.ring_remaining)
        if r > 1:
            out = _copy_frame(frame, ring_remaining=r - 1)
            out = self._next_frame_towards_peer(out)
            await self._send_queue.put((out, None))
            return
        if r == 1:
            if nxt == self.worker_name:
                to = float(self._rt.rpc_timeout_ms) / 1000.0
                await self._master.ReportPipelineComplete(
                    pv2.PipelineCompleteReport(total_tokens=self._total_tokens),
                    timeout=to,
                )
                return
            out = _copy_frame(
                frame,
                ring_remaining=0,
                pipeline_total_tokens=int(self._total_tokens),
            )
            out = self._next_frame_towards_peer(out)
            await self._send_queue.put((out, None))

    async def _handle_worker0_head_frame(self, frame: pv2.Frame) -> None:
        assert self._master is not None
        if frame.pipeline_stop:
            self._pending_pipeline_stop = _copy_frame(frame)
            LOG.info("recorded pending pipeline_stop (waiting for active requests to finish)")
            return

        if not frame.ring_return and int(frame.step_id) == 0:
            self._open_requests.add(frame.req_id)
            self._journal.mark_ingress(
                frame.req_id,
                batch_size=int(frame.batch_size or 1),
                context_len=int(frame.context_len),
                target_len=int(frame.target_len),
            )

        self._model.ensure_kv_session(
            frame.req_id,
            int(frame.batch_size or 1),
            int(frame.target_len),
        )
        ph = _phase_name(frame)
        branch = "head" if self._model.has_tail else "single"
        exp_ms, act_ms = await self._executor.run(
            frame, self._stage, self._merged_model, self._model, branch=branch
        )
        exp_bytes = expected_comm_bytes(
            self._stage, ph, int(frame.context_len), int(frame.batch_size or 1), branch=branch
        )
        exp_cm = expected_comm_ms(
            self._stage,
            phase_name=ph,
            context_len=int(frame.context_len),
            branch=branch,
        )
        payload = b"x" * self._scaled_payload_byte_len(exp_bytes)

        if not self._model.has_tail:
            self._active_reqs.add(frame.req_id)
            new_ctx = int(frame.context_len) + 1
            self._total_tokens += int(frame.batch_size or 1)
            self._journal.mark_first_token(frame.req_id)
            done = new_ctx >= int(frame.target_len)
            out = _copy_frame(
                frame,
                step_id=frame.step_id + 1,
                context_len=new_ctx,
                payload=payload,
                end_of_request=done,
                pipeline_stop=False,
                ring_return=False,
            )
            hop = HopRecord(
                step_id=int(frame.step_id),
                phase=ph,
                expected_compute_ms=exp_ms,
                actual_compute_ms=act_ms,
                expected_comm_bytes=exp_bytes,
                expected_comm_ms=exp_cm,
                actual_comm_ms=0.0,
                payload_bytes_sent=len(payload),
            )
            self._journal.record_hop(frame.req_id, hop)
            if done:
                await self._master.ReportRequestFinished(
                    pv2.RequestFinishedReport(req_id=frame.req_id),
                    timeout=float(self._rt.rpc_timeout_ms) / 1000.0,
                )
                self._journal.mark_finished(frame.req_id)
                self._active_reqs.discard(frame.req_id)
                self._open_requests.discard(frame.req_id)
                self._model.close_kv_session(frame.req_id)
                return
            await self._send_queue.put((out, hop))
            return

        out = _copy_frame(
            frame,
            step_id=frame.step_id + 1,
            payload=payload,
            pipeline_stop=False,
            ring_return=False,
        )
        hop = HopRecord(
            step_id=int(frame.step_id),
            phase=ph,
            expected_compute_ms=exp_ms,
            actual_compute_ms=act_ms,
            expected_comm_bytes=exp_bytes,
            expected_comm_ms=exp_cm,
            actual_comm_ms=0.0,
            payload_bytes_sent=len(payload),
        )
        self._journal.record_hop(frame.req_id, hop)
        await self._send_queue.put((out, hop))

    async def _handle_worker0_tail_frame(self, frame: pv2.Frame) -> None:
        assert self._master is not None
        if frame.pipeline_stop:
            await self._handle_worker0_tail_pipeline_stop(frame)
            return

        self._model.ensure_kv_session(
            frame.req_id,
            int(frame.batch_size or 1),
            int(frame.target_len),
        )
        ph = _phase_name(frame)
        self._active_reqs.add(frame.req_id)
        exp_ms, act_ms = await self._executor.run(
            frame, self._stage, self._merged_model, self._model, branch="tail"
        )
        exp_bytes = expected_comm_bytes(
            self._stage, ph, int(frame.context_len), int(frame.batch_size or 1), branch="tail"
        )
        exp_cm = expected_comm_ms(
            self._stage,
            phase_name=ph,
            context_len=int(frame.context_len),
            branch="tail",
        )
        payload = b"x" * self._scaled_payload_byte_len(exp_bytes)
        new_ctx = int(frame.context_len) + 1
        self._total_tokens += int(frame.batch_size or 1)
        self._journal.mark_first_token(frame.req_id)
        done = new_ctx >= int(frame.target_len)
        hop = HopRecord(
            step_id=int(frame.step_id),
            phase=ph,
            expected_compute_ms=exp_ms,
            actual_compute_ms=act_ms,
            expected_comm_bytes=exp_bytes,
            expected_comm_ms=exp_cm,
            actual_comm_ms=0.0,
            payload_bytes_sent=len(payload),
        )
        self._journal.record_hop(frame.req_id, hop)
        if done:
            await self._master.ReportRequestFinished(
                pv2.RequestFinishedReport(req_id=frame.req_id),
                timeout=float(self._rt.rpc_timeout_ms) / 1000.0,
            )
            self._journal.mark_finished(frame.req_id)
            self._active_reqs.discard(frame.req_id)
            self._open_requests.discard(frame.req_id)
            self._model.close_kv_session(frame.req_id)
            return
        # After tail compute on the prefill lap, switch ring to decode for subsequent head hops.
        next_phase = pv2.PHASE_DECODE if frame.phase == pv2.PHASE_PREFILL else frame.phase
        out = _copy_frame(
            frame,
            step_id=frame.step_id + 1,
            context_len=new_ctx,
            payload=payload,
            end_of_request=False,
            pipeline_stop=False,
            ring_return=False,
            phase=next_phase,
        )
        await self._head_queue.put(out)

    async def _handle_worker0_tail_pipeline_stop(self, frame: pv2.Frame) -> None:
        """Stop frame returning on ring_return path; typically r==0 with token count from last hop."""
        assert self._master is not None
        r = int(frame.ring_remaining)
        to = float(self._rt.rpc_timeout_ms) / 1000.0

        if r > 1:
            out = _copy_frame(frame, ring_remaining=r - 1)
            out = self._next_frame_towards_peer(out)
            await self._send_queue.put((out, None))
            return

        # Token count is accumulated on worker0 tail; middle nodes forward 0 in the stop frame.
        tokens = int(frame.pipeline_total_tokens) if frame.pipeline_total_tokens else int(
            self._total_tokens
        )
        await self._master.ReportPipelineComplete(
            pv2.PipelineCompleteReport(total_tokens=tokens),
            timeout=to,
        )


class WorkerRuntime:
    def __init__(
        self,
        worker_name: str,
        master_addr: str,
        bind_addr: str,
        public_addr: str,
        pipeline_path: str,
        worker_strategy_path: str,
        model_yaml_path: str,
        rt: RuntimeConfig,
        metrics_out: Optional[str] = None,
    ) -> None:
        self._bind = bind_addr
        self._svc = DataPlaneServicer(
            worker_name,
            master_addr,
            pipeline_path,
            worker_strategy_path,
            model_yaml_path,
            rt,
            metrics_out=metrics_out,
        )
        self._svc.set_public_address(public_addr)
        self._server: Optional[grpc.aio.Server] = None

    async def start(self) -> None:
        self._server = grpc.aio.server()
        pv2_grpc.add_DataPlaneServicer_to_server(self._svc, self._server)
        self._server.add_insecure_port(self._bind)
        await self._server.start()
        LOG.info("DataPlane worker listening on %s", self._bind)

    async def connect_master(self) -> None:
        if not await self._svc.initialize():
            raise RuntimeError("worker init failed")

    async def wait_done(self) -> None:
        await self._svc._done.wait()

    async def stop(self) -> None:
        if self._server:
            await self._server.stop(grace=1.0)
