#!/usr/bin/env python3
"""Send Poisson-arrival task stream to pp_nextgen master.

Supports three parameter sources:
- fixed: same (batch/context/target) for all requests
- random: sample each field from an integer range
- file: read request specs from JSON list for dataset-driven replay
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import grpc

# Allow running without editable install when executed from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pp_nextgen.runtime.grpc_gen import pipeline_v2_pb2 as pv2
from pp_nextgen.runtime.grpc_gen import pipeline_v2_pb2_grpc as pv2_grpc


def _positive_int(name: str, value: int) -> int:
    if int(value) <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return int(value)


def _non_negative_int(name: str, value: int) -> int:
    if int(value) < 0:
        raise ValueError(f"{name} must be >= 0, got {value}")
    return int(value)


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Poisson arrival sender for MasterControl.SubmitTask")
    p.add_argument("--master", default="127.0.0.1:50050", help="Master gRPC host:port")
    p.add_argument("--timeout", type=float, default=30.0, help="RPC timeout seconds")
    p.add_argument("--num-requests", type=int, default=10, help="Number of requests to submit")
    p.add_argument("--arrival-rate", type=float, default=1.0, help="Poisson rate lambda (req/s)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for Poisson/random mode")
    p.add_argument("--req-prefix", default="poisson", help="Req id prefix")
    p.add_argument("--send-pipeline-stop", action="store_true", help="Send pipeline-stop at end")
    p.add_argument(
        "--mode",
        choices=("fixed", "random", "file"),
        default="fixed",
        help="How to generate request parameters",
    )

    # fixed mode
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--context-len", type=int, default=1)
    p.add_argument("--target-len", type=int, default=3)

    # random mode
    p.add_argument("--batch-size-min", type=int, default=1)
    p.add_argument("--batch-size-max", type=int, default=8)
    p.add_argument("--context-len-min", type=int, default=1)
    p.add_argument("--context-len-max", type=int, default=512)
    p.add_argument("--target-len-min", type=int, default=2)
    p.add_argument("--target-len-max", type=int, default=1024)

    # file mode
    p.add_argument(
        "--request-spec-file",
        default=None,
        help=(
            "Path to JSON: either a list [{req_id?, batch_size, context_len, target_len}, ...] "
            "or unified_dataset_requests.v1 {\"schema_version\", \"requests\": [...]}"
        ),
    )

    ns = p.parse_args()
    if ns.arrival_rate <= 0:
        raise SystemExit("--arrival-rate must be > 0")
    if ns.num_requests <= 0:
        raise SystemExit("--num-requests must be > 0")
    if ns.mode == "file" and not ns.request_spec_file:
        raise SystemExit("--request-spec-file is required when --mode file")
    return ns


def _fixed_spec(args: argparse.Namespace) -> Dict[str, int]:
    b = _positive_int("batch_size", args.batch_size)
    c = _non_negative_int("context_len", args.context_len)
    t = _non_negative_int("target_len", args.target_len)
    return {"batch_size": b, "context_len": c, "target_len": max(c, t)}


def _random_spec(args: argparse.Namespace, rng: random.Random) -> Dict[str, int]:
    bmin = _positive_int("batch_size_min", args.batch_size_min)
    bmax = _positive_int("batch_size_max", args.batch_size_max)
    cmin = _non_negative_int("context_len_min", args.context_len_min)
    cmax = _non_negative_int("context_len_max", args.context_len_max)
    tmin = _non_negative_int("target_len_min", args.target_len_min)
    tmax = _non_negative_int("target_len_max", args.target_len_max)
    if bmin > bmax or cmin > cmax or tmin > tmax:
        raise ValueError("invalid min/max range")
    b = rng.randint(bmin, bmax)
    c = rng.randint(cmin, cmax)
    t = rng.randint(tmin, tmax)
    return {"batch_size": b, "context_len": c, "target_len": max(c, t)}


def _load_file_specs(path: str) -> List[Dict[str, Any]]:
    from pp_nextgen.datasets.unified_requests import normalize_json_payload_to_submit_specs

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return normalize_json_payload_to_submit_specs(data)


def main() -> None:
    args = _parse()
    rng = random.Random(args.seed)
    file_specs: List[Dict[str, Any]] = []
    if args.mode == "file":
        file_specs = _load_file_specs(args.request_spec_file)

    channel = grpc.insecure_channel(args.master)
    stub = pv2_grpc.MasterControlStub(channel)
    sent = 0
    try:
        for idx in range(args.num_requests):
            if args.mode == "fixed":
                spec = _fixed_spec(args)
            elif args.mode == "random":
                spec = _random_spec(args, rng)
            else:
                row = file_specs[idx % len(file_specs)]
                bs = row.get("batch_size", args.batch_size)
                spec = {
                    "req_id": row.get("req_id"),
                    "batch_size": _positive_int("batch_size", int(bs)),
                    "context_len": int(row["context_len"]),
                    "target_len": int(row["target_len"]),
                }

            req_id = str(spec.get("req_id") or f"{args.req_prefix}-{idx + 1}")
            req = pv2.TaskSubmitRequest(
                req_id=req_id,
                batch_size=int(spec["batch_size"]),
                context_len=int(spec["context_len"]),
                target_len=int(spec["target_len"]),
                is_end=False,
            )
            resp = stub.SubmitTask(req, timeout=args.timeout)
            if not resp.ok:
                raise RuntimeError(f"master rejected {req_id}: {resp.message}")
            sent += 1
            print(
                f"sent {req_id} batch={req.batch_size} context={req.context_len} target={req.target_len}"
            )
            if idx < args.num_requests - 1:
                time.sleep(rng.expovariate(args.arrival_rate))

        if args.send_pipeline_stop:
            stop_req = pv2.TaskSubmitRequest(
                req_id="",
                batch_size=0,
                context_len=0,
                target_len=0,
                is_end=True,
            )
            stop_resp = stub.SubmitTask(stop_req, timeout=args.timeout)
            if not stop_resp.ok:
                raise RuntimeError(f"pipeline-stop rejected: {stop_resp.message}")
            print("sent pipeline-stop")
    except grpc.RpcError as e:
        print(f"gRPC error: {e.code()} {e.details()}", file=sys.stderr)
        raise SystemExit(1) from e
    finally:
        channel.close()
    print(f"done: sent={sent}, mode={args.mode}, arrival_rate={args.arrival_rate} req/s")


if __name__ == "__main__":
    main()

