"""Benchmark a fixed pipeline strategy across bs/context scenarios."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class Scenario:
    bs: int
    context_len: int

    @property
    def key(self) -> str:
        return f"bs={self.bs},ctx={self.context_len}"


@dataclass
class ScenarioMetrics:
    scenario: Scenario
    pipeline_tbt_ms: float
    per_device_runtime_ms: Dict[str, float]
    per_device_bubble_ms: Dict[str, float]


@dataclass
class BenchmarkReport:
    strategy_path: str
    reference_bs: int
    scenarios: List[ScenarioMetrics]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "strategy_path": self.strategy_path,
            "reference_bs": self.reference_bs,
            "scenarios": [
                {
                    "scenario": asdict(s.scenario),
                    "pipeline_tbt_ms": s.pipeline_tbt_ms,
                    "per_device_runtime_ms": s.per_device_runtime_ms,
                    "per_device_bubble_ms": s.per_device_bubble_ms,
                }
                for s in self.scenarios
            ],
        }


def _device_of_worker(worker_name: str) -> str:
    if "_" not in worker_name:
        return worker_name
    return worker_name.rsplit("_", 1)[0]


def _stage_time_ms(stage: Dict[str, Any], *, bs: int, context_len: int, ref_bs: int) -> float:
    params = stage.get("stage_params") or {}
    comp = float(params.get("base_time", 0.0)) + float(params.get("increase_time", 0.0)) * float(context_len)

    base_size_ref_bs = float(params.get("base_size", 0.0))
    inc_size_ref_bs = float(params.get("inc_size", 0.0))
    if ref_bs <= 0:
        raise ValueError("reference bs in strategy must be > 0")
    bs_scale = float(bs) / float(ref_bs)
    comm_bytes = (base_size_ref_bs + inc_size_ref_bs * float(context_len)) * bs_scale

    comm_time_ref = float(stage.get("comm_time_ms", 0.0))
    ref_comm_bytes = float(stage.get("comm_bytes_to_next", 0.0))
    # Communication latency is scaled by effective bytes when bs/context changes.
    comm = (comm_time_ref * comm_bytes / ref_comm_bytes) if ref_comm_bytes > 0 else 0.0
    return max(comp, comm)


def evaluate_strategy_grid(strategy: Dict[str, Any], scenarios: List[Scenario], *, strategy_path: str = "") -> BenchmarkReport:
    stages = strategy.get("pipeline_stages") or []
    if not stages:
        raise ValueError("strategy has no pipeline_stages")

    ref_bs = int((strategy.get("schedule_input") or {}).get("bs", 0))
    if ref_bs <= 0:
        raise ValueError("strategy.schedule_input.bs is required and must be > 0")

    outputs: List[ScenarioMetrics] = []
    for sc in scenarios:
        stage_times = [_stage_time_ms(stage, bs=sc.bs, context_len=sc.context_len, ref_bs=ref_bs) for stage in stages]
        tbt = max(stage_times)

        runtime_by_device: Dict[str, float] = {}
        bubble_by_device: Dict[str, float] = {}
        for stage, st in zip(stages, stage_times):
            dev = _device_of_worker(str(stage.get("worker_name", "unknown")))
            runtime_by_device[dev] = runtime_by_device.get(dev, 0.0) + st
            bubble_by_device[dev] = bubble_by_device.get(dev, 0.0) + max(0.0, tbt - st)

        outputs.append(
            ScenarioMetrics(
                scenario=sc,
                pipeline_tbt_ms=tbt,
                per_device_runtime_ms=runtime_by_device,
                per_device_bubble_ms=bubble_by_device,
            )
        )
    return BenchmarkReport(strategy_path=strategy_path, reference_bs=ref_bs, scenarios=outputs)


def _collect_devices(report: BenchmarkReport) -> List[str]:
    all_devices = set()
    for item in report.scenarios:
        all_devices.update(item.per_device_runtime_ms.keys())
        all_devices.update(item.per_device_bubble_ms.keys())
    return sorted(all_devices)


def plot_benchmark_report(report: BenchmarkReport, out_dir: str | Path) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = [s.scenario.key for s in report.scenarios]
    x = list(range(len(labels)))
    devices = _collect_devices(report)

    runtime_path = out / "device_runtime_comparison.png"
    bubble_path = out / "device_bubble_comparison.png"
    tbt_path = out / "pipeline_tbt_comparison.png"

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.6), 5))
    bar_w = 0.8 / max(1, len(devices))
    for i, dev in enumerate(devices):
        vals = [s.per_device_runtime_ms.get(dev, 0.0) for s in report.scenarios]
        ax.bar([k + i * bar_w for k in x], vals, width=bar_w, label=dev)
    ax.set_xticks([k + bar_w * (len(devices) - 1) / 2 for k in x])
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Runtime (ms)")
    ax.set_title("Per-device Runtime Comparison")
    ax.legend(loc="upper left", ncol=min(4, max(1, len(devices))))
    fig.tight_layout()
    fig.savefig(runtime_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.6), 5))
    for i, dev in enumerate(devices):
        vals = [s.per_device_bubble_ms.get(dev, 0.0) for s in report.scenarios]
        ax.bar([k + i * bar_w for k in x], vals, width=bar_w, label=dev)
    ax.set_xticks([k + bar_w * (len(devices) - 1) / 2 for k in x])
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Bubble Time (ms)")
    ax.set_title("Per-device Bubble Comparison")
    ax.legend(loc="upper left", ncol=min(4, max(1, len(devices))))
    fig.tight_layout()
    fig.savefig(bubble_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.4), 4.5))
    tbt_vals = [s.pipeline_tbt_ms for s in report.scenarios]
    ax.plot(labels, tbt_vals, marker="o", linewidth=1.8)
    ax.set_ylabel("TBT (ms)")
    ax.set_title("Pipeline TBT Comparison")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(tbt_path, dpi=150)
    plt.close(fig)

    return {"runtime_png": str(runtime_path), "bubble_png": str(bubble_path), "tbt_png": str(tbt_path)}


def load_strategy(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def save_report(report: BenchmarkReport, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
