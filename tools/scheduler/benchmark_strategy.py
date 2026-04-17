#!/usr/bin/env python3
"""Benchmark one fixed strategy over bs/context grid and generate comparison figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pp_nextgen.evaluation import Scenario, evaluate_strategy_grid, plot_benchmark_report
from pp_nextgen.evaluation.strategy_benchmark import load_strategy, save_report


def _parse_int_list(raw: str, *, name: str) -> list[int]:
    vals = []
    for item in raw.split(","):
        text = item.strip()
        if not text:
            continue
        try:
            v = int(text)
        except ValueError as exc:
            raise ValueError(f"invalid integer in --{name}: {text}") from exc
        if v <= 0:
            raise ValueError(f"--{name} values must be > 0, got {v}")
        vals.append(v)
    if not vals:
        raise ValueError(f"--{name} cannot be empty")
    return vals


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--strategy", required=True, help="Path to pipeline_strategy.json")
    p.add_argument("--batch-sizes", required=True, help="Comma-separated batch sizes, e.g. 1,2,4,8,16")
    p.add_argument("--contexts", required=True, help="Comma-separated context lens, e.g. 128,256,512,1024")
    p.add_argument("--out-dir", default="outputs/scheduler/benchmark", help="Output directory for report + plots")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    strategy_path = Path(args.strategy).resolve()
    if not strategy_path.is_file():
        print(f"strategy file not found: {strategy_path}", file=sys.stderr)
        return 2

    try:
        batch_sizes = _parse_int_list(args.batch_sizes, name="batch-sizes")
        contexts = _parse_int_list(args.contexts, name="contexts")
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 2

    scenarios = [Scenario(bs=bs, context_len=ctx) for bs in batch_sizes for ctx in contexts]
    strategy = load_strategy(strategy_path)
    report = evaluate_strategy_grid(strategy, scenarios, strategy_path=str(strategy_path))

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = (_REPO / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    report_path = out_dir / "benchmark_report.json"
    save_report(report, report_path)
    fig_paths = plot_benchmark_report(report, out_dir)

    print(f"Wrote report: {report_path}")
    print(f"Wrote runtime figure: {fig_paths['runtime_png']}")
    print(f"Wrote bubble figure: {fig_paths['bubble_png']}")
    print(f"Wrote tbt figure: {fig_paths['tbt_png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
