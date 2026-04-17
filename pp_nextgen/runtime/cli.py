"""CLI entrypoints for master / worker processes."""

from __future__ import annotations

import argparse
import asyncio
import logging
import socket

from pp_nextgen.runtime.config import load_runtime_config
from pp_nextgen.runtime.master.service import MasterRuntime
from pp_nextgen.runtime.worker.service import WorkerRuntime


def get_local_ip() -> str:
    """Best-effort LAN IP (same approach as legacy `3060/role/worker.py`)."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _host_port_from_bind(bind: str) -> tuple[str, str]:
    if ":" not in bind:
        raise ValueError(f"invalid --bind (expected host:port): {bind!r}")
    host, port = bind.rsplit(":", 1)
    if not host or not port:
        raise ValueError(f"invalid --bind (expected host:port): {bind!r}")
    return host, port


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Nextgen pipeline runtime (gRPC)")
    sub = p.add_subparsers(dest="cmd", required=True)

    m = sub.add_parser("master", help="Run MasterControl (reads pipeline_strategy.json)")
    m.add_argument("--bind", default="0.0.0.0:50050", help="Listen address")
    m.add_argument(
        "--pipeline-strategy",
        required=True,
        help="Path to scheduler/export/pipeline_strategy.json",
    )
    m.add_argument(
        "--runtime-config",
        default=None,
        help="configs/runtime/runtime.example.yaml",
    )
    m.add_argument("--metrics-out", default=None, help="Optional master latency JSON path")

    w = sub.add_parser("worker", help="Run DataPlane worker")
    w.add_argument("--worker-name", required=True, help="e.g. 3060_0")
    w.add_argument("--bind", default="0.0.0.0:50051", help="Listen address")
    w.add_argument(
        "--public-address",
        default=None,
        help=(
            "Address master uses to reach this worker. "
            "If omitted, defaults to <detected-LAN-IP>:<port from --bind> (legacy worker.py get_local_ip)."
        ),
    )
    w.add_argument("--master", required=True, help="Master gRPC address")
    w.add_argument("--pipeline-strategy", required=True, help="Global pipeline_strategy.json")
    w.add_argument(
        "--worker-strategy",
        required=True,
        help="Per-worker strategy, e.g. scheduler/export/workers/3060_0.strategy.json",
    )
    w.add_argument("--model-config", required=True, help="configs/model/llama2_7b.yaml")
    w.add_argument("--runtime-config", default=None)
    w.add_argument("--metrics-out", default=None, help="Per-worker request_journal JSON path")

    ns = p.parse_args()
    if ns.cmd == "worker" and ns.public_address is None:
        _, port = _host_port_from_bind(ns.bind)
        ns.public_address = f"{get_local_ip()}:{port}"
    return ns


async def _run_master(args: argparse.Namespace) -> None:
    from pathlib import Path

    rt_path = args.runtime_config or str(
        Path(__file__).resolve().parents[2] / "configs" / "runtime" / "runtime.example.yaml"
    )
    rt = load_runtime_config(rt_path)
    mr = MasterRuntime(args.bind, args.pipeline_strategy, rt)
    await mr.start()
    await mr.wait_done()
    if args.metrics_out:
        mr.servicer.latency_tracker.export_json(args.metrics_out)
    await mr.stop()


async def _run_worker(args: argparse.Namespace) -> None:
    from pathlib import Path

    rt_path = args.runtime_config or str(
        Path(__file__).resolve().parents[2] / "configs" / "runtime" / "runtime.example.yaml"
    )
    rt = load_runtime_config(rt_path)
    wr = WorkerRuntime(
        args.worker_name,
        args.master,
        args.bind,
        args.public_address,
        args.pipeline_strategy,
        args.worker_strategy,
        args.model_config,
        rt,
        metrics_out=args.metrics_out,
    )
    await wr.start()
    try:
        await wr.connect_master()
        await wr.wait_done()
    finally:
        await wr.stop()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    args = _parse()
    if args.cmd == "master":
        asyncio.run(_run_master(args))
    elif args.cmd == "worker":
        asyncio.run(_run_worker(args))
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
