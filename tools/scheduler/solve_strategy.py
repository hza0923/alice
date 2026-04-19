#!/usr/bin/env python3
"""Solve pipeline strategy with designated_tail_n sweep; export global + per-worker JSON."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pp_nextgen.config_loader import build_scheduler_model_cfg, load_yaml
from pp_nextgen.scheduler.adapters.registry_adapter import (
    build_device_performance_from_registry,
    build_module_memory_from_registry,
    load_registry,
)
from pp_nextgen.scheduler.export.strategy_export import write_worker_strategies
from pp_nextgen.scheduler.tail_sweep import solve_with_tail_sweep


def _repo_root() -> Path:
    return _REPO


def _resolve(path_str: str, bases: list[Path]) -> Path:
    p = Path(path_str)
    if p.is_file():
        return p.resolve()
    for b in bases:
        c = (b / path_str).resolve()
        if c.is_file():
            return c
    return p.resolve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cluster", required=True, help="Cluster YAML (device_group, solve, network)")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = _repo_root()
    cluster_path = _resolve(args.cluster, [Path.cwd(), root])
    if not cluster_path.is_file():
        print(f"cluster file not found: {args.cluster}", file=sys.stderr)
        return 2

    cluster = load_yaml(cluster_path)
    solve = cluster.get("solve") or {}
    if not solve:
        print("cluster YAML missing solve: block", file=sys.stderr)
        return 2

    device_group = cluster.get("device_group")
    designated_device = cluster.get("designated_device")
    network_bw = cluster.get("network_bandwidth_mbps") or {"default": 10000.0}
    if not isinstance(device_group, dict) or not device_group:
        print("cluster YAML must define device_group mapping", file=sys.stderr)
        return 2
    if not designated_device:
        print("cluster YAML must define designated_device", file=sys.stderr)
        return 2

    model_cfg_path = _resolve(str(solve.get("model_config", "configs/model/llama2_7b.yaml")), [cluster_path.parent, Path.cwd(), root])
    registry_path = _resolve(str(solve.get("registry_path", "outputs/profiling/registry/device_registry.v3.json")), [cluster_path.parent, Path.cwd(), root])

    if not model_cfg_path.is_file() or not registry_path.is_file():
        print(f"model_config or registry_path missing:\n  {model_cfg_path}\n  {registry_path}", file=sys.stderr)
        return 2

    model_doc = load_yaml(model_cfg_path)
    model_cfg = build_scheduler_model_cfg(model_doc)
    reg = load_registry(registry_path)
    prefer_bs = int(solve.get("prefer_bs", 32))
    device_perf = build_device_performance_from_registry(reg, prefer_bs=prefer_bs)
    mem = build_module_memory_from_registry(reg, prefer_bs=prefer_bs)
    model_cfg["module_memory_gb"].update(mem)

    missing = [d for d in device_group if d not in device_perf]
    if missing:
        print(f"registry missing devices for device_group keys: {missing}", file=sys.stderr)
        return 2

    bs = int(solve.get("batch_size", 1))
    target_seq_len = int(solve.get("target_seq_len", 2048))
    max_seq_len_raw = solve.get("max_seq_len")
    max_seq_len = int(max_seq_len_raw) if max_seq_len_raw is not None else None
    use_fg = bool(solve.get("use_fine_grained", True))
    tail = list(solve.get("tail_candidates", [1, 2, 3, 4, 5, 6, 7]))

    out_dir = Path(solve.get("out_dir", "outputs/scheduler/export"))
    if not out_dir.is_absolute():
        out_dir = (root / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    strategy_path = out_dir / "pipeline_strategy.json"

    strat, report = solve_with_tail_sweep(
        model_cfg,
        device_perf,
        network_bw,
        {str(k): int(v) for k, v in device_group.items()},
        str(designated_device),
        bs=bs,
        target_seq_len=target_seq_len,
        use_fine_grained=use_fg,
        tail_candidates=tail,
        strategy_output_path=str(strategy_path),
        max_seq_len=max_seq_len,
    )

    report_path = out_dir / "tail_sweep_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if strat is None:
        print("No feasible schedule for any tail_n.", file=sys.stderr)
        return 3

    write_worker_strategies(strat, out_dir / "workers")
    print(f"Wrote {strategy_path}")
    print(f"Wrote {report_path}")
    print(f"Wrote per-worker strategies under {out_dir / 'workers'}")
    print(
        f"Best tail_n={report.get('best_designated_tail_n')}  "
        f"tbt_ms={report.get('best_tbt_ms')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
