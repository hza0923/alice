"""Outer loop over designated_tail_n; pick minimum TBT."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pp_nextgen.scheduler.core.flexible_pipeline_scheduler import FlexiblePipelineScheduler


def solve_with_tail_sweep(
    model_config: Dict[str, Any],
    device_perf: Dict[str, Any],
    network_bw: Dict[str, Any],
    device_group: Dict[str, int],
    designated_device: str,
    *,
    bs: int,
    target_seq_len: int,
    use_fine_grained: bool,
    tail_candidates: List[int],
    strategy_output_path: Optional[str] = None,
) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_n: Optional[int] = None
    best_tbt = float("inf")
    report: Dict[str, Any] = {"candidates": []}

    for n in tail_candidates:
        if n < 1:
            continue
        sch = FlexiblePipelineScheduler(
            model_config,
            device_perf,
            network_bw,
            device_group,
            designated_device,
            use_fine_grained=use_fine_grained,
            designated_tail_n=n,
            strategy_output_path=None,
        )
        strat = sch.schedule(bs, target_seq_len, quiet=False)
        row = {"designated_tail_n": n, "feasible": strat is not None}
        if strat is not None:
            row["tbt_ms"] = float(strat.get("tbt_ms", strat.get("objective", {}).get("value", 0.0)))
            if row["tbt_ms"] < best_tbt:
                best_tbt = row["tbt_ms"]
                best = strat
                best_n = n
        report["candidates"].append(row)
        if not use_fine_grained:
            break

    report["best_designated_tail_n"] = best_n
    report["best_tbt_ms"] = best_tbt if best is not None else None

    if best is not None and strategy_output_path:
        outp = Path(strategy_output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", encoding="utf-8") as f:
            json.dump(best, f, indent=2, ensure_ascii=False)

    return best, report
