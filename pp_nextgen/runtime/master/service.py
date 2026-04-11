"""gRPC master: registration, SubmitTask, ring topology from pipeline_strategy.json."""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional

import grpc

from pp_nextgen.runtime.config import RuntimeConfig
from pp_nextgen.runtime.grpc_gen import pipeline_v2_pb2 as pv2
from pp_nextgen.runtime.grpc_gen import pipeline_v2_pb2_grpc as pv2_grpc
from pp_nextgen.runtime.metrics import MasterLatencyTracker
from pp_nextgen.runtime.strategy import load_pipeline_strategy, pipeline_stage_order

LOG = logging.getLogger("pp_nextgen.runtime.master")


class MasterControlServicer(pv2_grpc.MasterControlServicer):
    def __init__(self, pipeline_path: str, rt: RuntimeConfig) -> None:
        self._rt = rt
        self._doc = load_pipeline_strategy(pipeline_path)
        self._order = pipeline_stage_order(self._doc)
        self._expected = set(self._order)
        self._addresses: Dict[str, str] = {}
        self._all_registered = asyncio.Event()
        self._ready = asyncio.Event()
        self._first_worker = self._order[0] if self._order else ""
        self._send_queue: asyncio.Queue[pv2.Frame] = asyncio.Queue(maxsize=rt.send_queue_maxsize)
        self._done = asyncio.Event()
        self._latency = MasterLatencyTracker()
        self._first_stub: Optional[pv2_grpc.DataPlaneStub] = None
        self._total_tokens_reported = 0

    @property
    def latency_tracker(self) -> MasterLatencyTracker:
        return self._latency

    async def start_send_loop(self) -> None:
        asyncio.create_task(self._send_loop())

    async def _send_loop(self) -> None:
        timeout = float(self._rt.rpc_timeout_ms) / 1000.0
        while not self._done.is_set():
            frame = await self._send_queue.get()
            if self._first_stub is None:
                LOG.error("first worker stub missing; drop frame")
                self._send_queue.task_done()
                continue
            try:
                await self._first_stub.SendFrame(frame, timeout=timeout)
            except Exception as e:
                LOG.error("SendFrame to first worker failed: %s", e)
            self._send_queue.task_done()

    async def RegisterWorker(
        self, request: pv2.RegisterWorkerRequest, context: grpc.aio.ServicerContext
    ) -> pv2.RegisterWorkerResponse:
        name = request.worker_name
        addr = request.worker_address
        if name not in self._expected:
            return pv2.RegisterWorkerResponse(ok=False, message=f"unknown worker_name {name}")
        if name in self._addresses:
            return pv2.RegisterWorkerResponse(ok=False, message=f"duplicate register {name}")
        self._addresses[name] = addr
        LOG.info("registered %s -> %s (%d/%d)", name, addr, len(self._addresses), len(self._expected))
        if self._addresses.keys() == self._expected:
            self._all_registered.set()
            ch = grpc.aio.insecure_channel(self._addresses[self._first_worker])
            self._first_stub = pv2_grpc.DataPlaneStub(ch)
            self._ready.set()
            await self.start_send_loop()
        return pv2.RegisterWorkerResponse(ok=True, message="")

    async def GetNextWorker(
        self, request: pv2.NextWorkerRequest, context: grpc.aio.ServicerContext
    ) -> pv2.NextWorkerResponse:
        await self._all_registered.wait()
        me = request.worker_name
        from pp_nextgen.runtime.strategy import next_worker_name

        nxt = next_worker_name(self._doc, me)
        if not nxt:
            return pv2.NextWorkerResponse(
                ok=False,
                message="unknown worker",
                all_registered=True,
                has_next=False,
                next_worker_address="",
            )
        addr = self._addresses.get(nxt, "")
        return pv2.NextWorkerResponse(
            ok=True,
            message="",
            all_registered=True,
            has_next=True,
            next_worker_address=addr,
        )

    async def GetPeerAddress(
        self, request: pv2.PeerAddressRequest, context: grpc.aio.ServicerContext
    ) -> pv2.PeerAddressResponse:
        await self._all_registered.wait()
        name = request.worker_name
        addr = self._addresses.get(name)
        if not addr:
            return pv2.PeerAddressResponse(ok=False, message=f"unknown worker {name}", address="")
        return pv2.PeerAddressResponse(ok=True, message="", address=addr)

    async def SubmitTask(
        self, request: pv2.TaskSubmitRequest, context: grpc.aio.ServicerContext
    ) -> pv2.TaskSubmitResponse:
        await self._ready.wait()
        if request.is_end:
            n = max(1, len(self._order))
            frame = pv2.Frame(
                req_id="__pipeline_stop__",
                step_id=0,
                phase=pv2.PHASE_DECODE,
                context_len=0,
                target_len=0,
                batch_size=0,
                pipeline_stop=True,
                ring_remaining=n,
            )
            await self._send_queue.put(frame)
            return pv2.TaskSubmitResponse(ok=True, message="")
        self._latency.mark_submit(request.req_id)
        frame = pv2.Frame(
            req_id=request.req_id,
            step_id=0,
            phase=pv2.PHASE_DECODE,
            context_len=request.context_len,
            target_len=request.target_len,
            batch_size=request.batch_size or 1,
            end_of_request=False,
            pipeline_stop=False,
        )
        await self._send_queue.put(frame)
        return pv2.TaskSubmitResponse(ok=True, message="")

    async def ReportRequestFinished(
        self, request: pv2.RequestFinishedReport, context: grpc.aio.ServicerContext
    ) -> pv2.Ack:
        self._latency.mark_finished(request.req_id)
        LOG.info("request finished %s", request.req_id)
        return pv2.Ack(ok=True, message="")

    async def ReportPipelineComplete(
        self, request: pv2.PipelineCompleteReport, context: grpc.aio.ServicerContext
    ) -> pv2.Ack:
        self._total_tokens_reported = int(request.total_tokens)
        LOG.info("pipeline complete total_tokens=%s", request.total_tokens)
        await self._notify_shutdown()
        self._done.set()
        return pv2.Ack(ok=True, message="")

    async def _notify_shutdown(self) -> None:
        """Linear NotifyPipelineEnd chain (not the data-plane ring)."""
        if not self._first_worker or not self._order:
            return
        addr = self._addresses.get(self._first_worker)
        if not addr:
            return
        timeout = float(self._rt.rpc_timeout_ms) / 1000.0
        ch = grpc.aio.insecure_channel(addr)
        stub = pv2_grpc.DataPlaneStub(ch)
        try:
            await stub.NotifyPipelineEnd(
                pv2.PipelineEndNotification(
                    message="shutdown",
                    ring_remaining=len(self._order),
                ),
                timeout=timeout,
            )
        except Exception as e:
            LOG.error("NotifyPipelineEnd to first worker failed: %s", e)


class MasterRuntime:
    def __init__(self, bind_addr: str, pipeline_path: str, rt: RuntimeConfig) -> None:
        self.bind_addr = bind_addr
        self.pipeline_path = pipeline_path
        self.rt = rt
        self._server: Optional[grpc.aio.Server] = None
        self._servicer: Optional[MasterControlServicer] = None

    async def start(self) -> None:
        self._servicer = MasterControlServicer(self.pipeline_path, self.rt)
        self._server = grpc.aio.server()
        pv2_grpc.add_MasterControlServicer_to_server(self._servicer, self._server)
        self._server.add_insecure_port(self.bind_addr)
        await self._server.start()
        LOG.info("MasterControl listening on %s", self.bind_addr)

    async def wait_done(self) -> None:
        assert self._servicer is not None
        await self._servicer._done.wait()

    async def stop(self) -> None:
        if self._server:
            await self._server.stop(grace=1.0)

    @property
    def servicer(self) -> MasterControlServicer:
        assert self._servicer is not None
        return self._servicer
