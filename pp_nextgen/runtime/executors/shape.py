"""Shape / no-weight execution path (placeholder for torch kernels)."""

from __future__ import annotations

import time
from typing import Any, Dict, Tuple

from pp_nextgen.runtime.model import PipelineModel
from pp_nextgen.runtime.strategy import expected_compute_ms


def _phase_name(frame: Any) -> str:
    from pp_nextgen.runtime.grpc_gen import pipeline_v2_pb2

    if frame.phase == pipeline_v2_pb2.PHASE_PREFILL:
        return "prefill"
    return "decode"


class ShapeExecutor:
    """Runs `PipelineModel` forward then attributes wall time to compute."""

    async def run(
        self,
        frame: Any,
        stage: Dict[str, Any],
        merged_model: Dict[str, Any],
        model: PipelineModel,
        branch: str = "single",
    ) -> Tuple[float, float]:
        _ = merged_model
        ph = _phase_name(frame)
        exp_ms = expected_compute_ms(
            stage, ph, int(frame.context_len), int(frame.batch_size or 1), branch=branch
        )
        t0 = time.perf_counter()
        rid = frame.req_id
        ctx = int(frame.context_len)
        bs = int(frame.batch_size or 1)
        if branch == "tail" and model.has_tail:
            model.forward_decode_step_tail(rid, ctx, bs)
        elif branch == "head" and model.has_tail:
            model.forward_decode_step_head(rid, ctx, bs)
        else:
            model.forward_decode_step(rid, ctx, bs)
        actual_ms = (time.perf_counter() - t0) * 1000.0
        return exp_ms, actual_ms
