#!/usr/bin/env python3
"""Run worker0-queue pipeline simulation and export metrics JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pp_nextgen.runtime.strategy import load_pipeline_strategy
from pp_nextgen.simulation import (
    QueueSimConfig,
    Worker0QueueSimulator,
    generate_poisson_requests,
    generate_poisson_requests_from_specs,
)


def _parse_link_overrides(raw: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    text = raw.strip()
    if not text:
        return out
    for item in text.split(","):
        pair = item.strip()
        if not pair:
            continue
        if ":" not in pair:
            raise ValueError(f"invalid --link-bandwidth item: {pair}")
        k, v = pair.split(":", 1)
        key = k.strip()
        try:
            gbps = float(v.strip())
        except ValueError as exc:
            raise ValueError(f"invalid bandwidth for {key}: {v}") from exc
        if gbps <= 0:
            raise ValueError(f"bandwidth must be > 0, got {gbps} for {key}")
        out[key] = gbps
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", required=True, help="Path to pipeline_strategy.json")
    p.add_argument("--arrival-rate", type=float, default=4.0, help="Poisson arrival rate (req/s)")
    p.add_argument("--duration-s", type=float, default=10.0, help="Simulation arrival horizon in seconds")
    p.add_argument("--num-requests", type=int, default=0, help="Optional hard cap; 0 means unlimited by count")
    p.add_argument("--batch-size", type=int, default=1, help="Per-request batch size")
    p.add_argument(
        "--request-file",
        default="",
        help=(
            "Optional JSON: unified_dataset_requests.v1 or legacy submit list; "
            "Poisson spacing via --arrival-rate; batch via --batch-size."
        ),
    )
    p.add_argument("--context-len", type=int, default=128, help="Per-request context length (synthetic)")
    p.add_argument("--target-len", type=int, default=256, help="Per-request target length (synthetic)")
    p.add_argument("--max-batch-size", type=int, default=32, help="Single-request batch upper bound")
    p.add_argument(
        "--max-in-flight",
        type=int,
        default=512,
        help="worker0-head running queue capacity (matches runtime scheduling.max_in_flight_requests)",
    )
    p.add_argument("--default-link-bandwidth-gbps", type=float, default=0.1, help="Fallback link bandwidth")
    p.add_argument(
        "--link-bandwidth",
        default="",
        help='Optional overrides: "workerA->workerB:24,workerB->workerC:16"',
    )
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--out", default="outputs/runtime/simulation/pipeline_queue.sim_metrics.json", help="Output JSON path")
    p.add_argument(
        "--master-latency-sidecar-out",
        default="",
        help="Optional path to export master_latency.v1-like JSON for easier runtime comparison",
    )
    p.add_argument(
        "--req-trace-out",
        default="",
        help="Optional path to export per-request queue trace JSON",
    )
    p.add_argument(
        "--stage-trace-out",
        default="",
        help="Optional path to export per-stage queue trace JSON",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    strategy_path = Path(args.strategy)
    if not strategy_path.is_absolute():
        strategy_path = (_REPO / strategy_path).resolve()
    if not strategy_path.is_file():
        print(f"strategy file not found: {strategy_path}", file=sys.stderr)
        return 2

    num_requests = int(args.num_requests) if int(args.num_requests) > 0 else None
    duration_s = float(args.duration_s) if float(args.duration_s) > 0 else None
    if num_requests is None and duration_s is None:
        print("either --duration-s > 0 or --num-requests > 0 is required", file=sys.stderr)
        return 2

    try:
        link_overrides = _parse_link_overrides(args.link_bandwidth)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    strategy = load_pipeline_strategy(strategy_path)
    trace_path = (args.request_file or "").strip()
    if trace_path:
        from pp_nextgen.datasets.unified_requests import load_simulation_specs_from_json_path

        rp = Path(trace_path)
        if not rp.is_absolute():
            rp = (_REPO / rp).resolve()
        if not rp.is_file():
            print(f"request file not found: {rp}", file=sys.stderr)
            return 2
        specs = load_simulation_specs_from_json_path(rp)
        if not specs:
            print("request file produced no specs", file=sys.stderr)
            return 2
        requests = generate_poisson_requests_from_specs(
            specs=specs,
            rate_per_sec=float(args.arrival_rate),
            duration_s=duration_s,
            num_requests=num_requests,
            batch_size=int(args.batch_size),
            seed=int(args.seed),
        )
    else:
        requests = generate_poisson_requests(
            rate_per_sec=float(args.arrival_rate),
            duration_s=duration_s,
            num_requests=num_requests,
            batch_size=int(args.batch_size),
            context_len=int(args.context_len),
            target_len=int(args.target_len),
            seed=int(args.seed),
        )
    if not requests:
        print("generated no requests; adjust --duration-s / --num-requests / --arrival-rate", file=sys.stderr)
        return 2

    sim = Worker0QueueSimulator(
        strategy_doc=strategy,
        config=QueueSimConfig(
            max_batch_size=int(args.max_batch_size),
            default_link_bandwidth_gbps=float(args.default_link_bandwidth_gbps),
            link_bandwidth_overrides=link_overrides,
            max_in_flight_requests=int(args.max_in_flight),
        ),
    )
    report = sim.run(requests)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (_REPO / out_path).resolve()
    report.export_json(out_path)
    if args.req_trace_out and args.stage_trace_out:
        req_trace_path = Path(args.req_trace_out)
        if not req_trace_path.is_absolute():
            req_trace_path = (_REPO / req_trace_path).resolve()
        stage_trace_path = Path(args.stage_trace_out)
        if not stage_trace_path.is_absolute():
            stage_trace_path = (_REPO / stage_trace_path).resolve()
        sim.export_traces(req_trace_path=req_trace_path, stage_trace_path=stage_trace_path)
        print(f"Wrote req trace: {req_trace_path}")
        print(f"Wrote stage trace: {stage_trace_path}")
    elif args.req_trace_out or args.stage_trace_out:
        print("both --req-trace-out and --stage-trace-out are required together", file=sys.stderr)
        return 2
    if args.master_latency_sidecar_out:
        sidecar_path = Path(args.master_latency_sidecar_out)
        if not sidecar_path.is_absolute():
            sidecar_path = (_REPO / sidecar_path).resolve()
        sidecar = {"schema_version": "master_latency.v1", "requests": {}}
        for req in report.requests:
            if req.finish_ts is None:
                continue
            sidecar["requests"][req.req_id] = {
                "start": req.running_enter_ts if req.running_enter_ts is not None else req.arrival_ts,
                "end": req.finish_ts,
                "latency_s": req.e2e_latency_s(),
            }
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        with sidecar_path.open("w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2, ensure_ascii=False)
        print(f"Wrote sidecar latency: {sidecar_path}")

    summary = report.summary
    print(f"Wrote queue metrics: {out_path}")
    print(
        json.dumps(
            {
                "request_count": summary.request_count,
                "completed_count": summary.completed_count,
                "throughput_req_s": summary.throughput_req_s,
                "throughput_token_s": summary.throughput_token_s,
                "avg_ttft_s": summary.avg_ttft_s,
                "avg_e2e_s": summary.avg_e2e_s,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
