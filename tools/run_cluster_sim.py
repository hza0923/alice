#!/usr/bin/env python3
"""Local multi-process style cluster sim: master + N workers + SubmitTask client.

Requires: pip install grpcio grpcio-tools (see pyproject optional deps).

Example (from repo root, adjust paths):

  python tools/run_cluster_sim.py ^
    --pipeline-strategy scheduler/export/pipeline_strategy.json ^
    --model-config configs/model/llama2_7b.yaml ^
    --workers-export-dir scheduler/export/workers
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import grpc

from pp_nextgen.runtime.config import load_runtime_config
from pp_nextgen.runtime.grpc_gen import pipeline_v2_pb2 as pv2
from pp_nextgen.runtime.grpc_gen import pipeline_v2_pb2_grpc as pv2_grpc
from pp_nextgen.runtime.master.service import MasterRuntime
from pp_nextgen.runtime.strategy import load_pipeline_strategy, pipeline_stage_order
from pp_nextgen.runtime.worker.service import WorkerRuntime


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pipeline-strategy",
        type=str,
        default=str(root / "scheduler" / "export" / "pipeline_strategy.json"),
    )
    p.add_argument(
        "--workers-export-dir",
        type=str,
        default=str(root / "scheduler" / "export" / "workers"),
    )
    p.add_argument(
        "--runtime-logs-dir",
        type=str,
        default=str(root / "runtime" / "logs"),
        help="Where to write per-worker *.runtime_metrics.json (default: repo runtime/logs)",
    )
    p.add_argument(
        "--model-config",
        type=str,
        default=str(root / "configs" / "model" / "llama2_7b.yaml"),
    )
    p.add_argument(
        "--runtime-config",
        type=str,
        default=str(root / "configs" / "runtime" / "runtime.example.yaml"),
    )
    p.add_argument("--master-bind", default="127.0.0.1:50050")
    p.add_argument("--base-port", type=int, default=50051)
    p.add_argument("--context-len", type=int, default=10)
    p.add_argument("--target-len", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=1)
    return p.parse_args()


async def _client_submit(master_addr: str, context_len: int, target_len: int, batch_size: int) -> None:
    await asyncio.sleep(0.8)
    ch = grpc.aio.insecure_channel(master_addr)
    stub = pv2_grpc.MasterControlStub(ch)
    await stub.SubmitTask(
        pv2.TaskSubmitRequest(
            req_id="sim-req-1",
            batch_size=batch_size,
            context_len=context_len,
            target_len=target_len,
            is_end=False,
        ),
        timeout=5.0,
    )
    # Allow the ring to finish decode steps before injecting pipeline_stop.
    await asyncio.sleep(1.0)
    await stub.SubmitTask(
        pv2.TaskSubmitRequest(
            req_id="__end__",
            batch_size=1,
            context_len=0,
            target_len=0,
            is_end=True,
        ),
        timeout=5.0,
    )
    await ch.close()


async def main_async() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rt = load_runtime_config(args.runtime_config)
    doc = load_pipeline_strategy(args.pipeline_strategy)
    order = pipeline_stage_order(doc)
    if not order:
        raise SystemExit("pipeline_strategy has no pipeline_stages")

    master = MasterRuntime(args.master_bind, args.pipeline_strategy, rt)
    await master.start()

    workers: list[WorkerRuntime] = []
    export_dir = Path(args.workers_export_dir)
    log_dir = Path(args.runtime_logs_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(order):
        port = args.base_port + i
        bind = f"127.0.0.1:{port}"
        ws = export_dir / f"{name}.strategy.json"
        if not ws.is_file():
            raise SystemExit(f"missing worker strategy file: {ws}")
        wr = WorkerRuntime(
            name,
            args.master_bind,
            bind,
            bind,
            args.pipeline_strategy,
            str(ws),
            args.model_config,
            rt,
            metrics_out=str(log_dir / f"{name}.runtime_metrics.json"),
        )
        await wr.start()
        workers.append(wr)

    await asyncio.gather(*(wr.connect_master() for wr in workers))

    asyncio.create_task(
        _client_submit(
            args.master_bind,
            args.context_len,
            args.target_len,
            args.batch_size,
        )
    )

    await asyncio.gather(master.wait_done(), *[w.wait_done() for w in workers])
    for w in workers:
        await w.stop()
    await master.stop()


def main() -> int:
    asyncio.run(main_async())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
