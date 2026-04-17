#!/usr/bin/env python3
"""Send TaskSubmitRequest to a running pp_nextgen master (MasterControl.SubmitTask).

Prerequisites: workers have registered and master is ready (otherwise SubmitTask blocks).

Examples:
  python tools/runtime/submit_task_to_master.py --master 127.0.0.1:50050 \\
    --req-id demo-1 --batch-size 1 --context-len 128 --target-len 32

  # Signal pipeline drain / shutdown (same as TaskSubmitRequest.is_end in proto)
  python tools/runtime/submit_task_to_master.py --master 127.0.0.1:50050 --pipeline-stop
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running without editable install when executed from repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import grpc

from pp_nextgen.runtime.grpc_gen import pipeline_v2_pb2 as pv2
from pp_nextgen.runtime.grpc_gen import pipeline_v2_pb2_grpc as pv2_grpc


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit tasks to pp_nextgen MasterControl gRPC")
    p.add_argument("--master", default="127.0.0.1:50050", help="Master gRPC host:port")
    p.add_argument("--timeout", type=float, default=60.0, help="RPC timeout seconds")
    p.add_argument(
        "--pipeline-stop",
        action="store_true",
        help="Send is_end=True (drain ring and shut down pipeline per master logic)",
    )
    p.add_argument("--req-id", default="cli-req", help="Request id (ignored when --pipeline-stop)")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--context-len", type=int, default=1)
    p.add_argument("--target-len", type=int, default=3)
    return p.parse_args()


def main() -> None:
    args = _parse()
    channel = grpc.insecure_channel(args.master)
    stub = pv2_grpc.MasterControlStub(channel)
    if args.pipeline_stop:
        req = pv2.TaskSubmitRequest(
            req_id="",
            batch_size=0,
            context_len=0,
            target_len=0,
            is_end=True,
        )
    else:
        req = pv2.TaskSubmitRequest(
            req_id=args.req_id,
            batch_size=max(1, args.batch_size),
            context_len=max(0, args.context_len),
            target_len=max(0, args.target_len),
            is_end=False,
        )
    try:
        resp = stub.SubmitTask(req, timeout=args.timeout)
    except grpc.RpcError as e:
        print(f"gRPC error: {e.code()} {e.details()}", file=sys.stderr)
        raise SystemExit(1) from e
    finally:
        channel.close()
    if not resp.ok:
        print(f"master rejected: {resp.message}", file=sys.stderr)
        raise SystemExit(2)
    print(f"ok: {resp.message!r}")


if __name__ == "__main__":
    main()
