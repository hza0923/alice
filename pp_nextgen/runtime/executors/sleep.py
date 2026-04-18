"""Sleep-based executor using profiled timing models.

No real model or KV cache: ``SleepPipelineModel`` is a routing placeholder; timing is sleep-only.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from pp_nextgen.runtime.precise_sleep import sleep_seconds_async
from pp_nextgen.runtime.strategy import expected_compute_ms


def _phase_name(frame: Any) -> str:
    from pp_nextgen.runtime.grpc_gen import pipeline_v2_pb2

    if frame.phase == pipeline_v2_pb2.PHASE_PREFILL:
        return "prefill"
    return "decode"


class SleepExecutor:
    async def initialize(self) -> None:
        return

    async def run(
        self,
        frame: Any,
        stage: Dict[str, Any],
        merged_model: Dict[str, Any],
        model: Any,
        branch: str = "single",
    ) -> Tuple[float, float]:
        _ = merged_model
        _ = model
        ph = _phase_name(frame)
        exp_ms = expected_compute_ms(
            stage, ph, int(frame.context_len), int(frame.batch_size or 1), branch=branch
        )
        # Wall time follows profiled delay only; shape_executor runs the torch microbench path.
        actual_ms = await sleep_seconds_async(exp_ms / 1000.0)
        return exp_ms, actual_ms
