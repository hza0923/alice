"""Sleep-based executor using profiled timing models."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from pp_nextgen.runtime.model import PipelineModel
from pp_nextgen.runtime.precise_sleep import sleep_seconds_async
from pp_nextgen.runtime.strategy import expected_compute_ms


def _phase_name(frame: Any) -> str:
    from pp_nextgen.runtime.grpc_gen import pipeline_v2_pb2

    if frame.phase == pipeline_v2_pb2.PHASE_PREFILL:
        return "prefill"
    return "decode"


class SleepExecutor:
    async def run(
        self,
        frame: Any,
        stage: Dict[str, Any],
        merged_model: Dict[str, Any],
        model: PipelineModel,
        branch: str = "single",
    ) -> Tuple[float, float]:
        _ = merged_model
        _ = model
        ph = _phase_name(frame)
        exp_ms = expected_compute_ms(
            stage, ph, int(frame.context_len), int(frame.batch_size or 1), branch=branch
        )
        # Sleep path keeps wall time = profiled delay; full numpy module chain is in shape_executor.
        actual_ms = await sleep_seconds_async(exp_ms / 1000.0)
        return exp_ms, actual_ms
