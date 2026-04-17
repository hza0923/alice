"""Request models and arrival generators for DES."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class SimRequest:
    req_id: str
    arrival_ts: float
    batch_size: int
    context_len: int
    target_len: int

    @property
    def new_tokens(self) -> int:
        return max(0, int(self.target_len) - int(self.context_len))


def generate_poisson_requests(
    *,
    rate_per_sec: float,
    duration_s: Optional[float],
    num_requests: Optional[int],
    batch_size: int,
    context_len: int,
    target_len: int,
    seed: int,
    req_id_prefix: str = "req",
) -> List[SimRequest]:
    """Generate requests with exponential inter-arrival gaps."""
    if rate_per_sec <= 0:
        raise ValueError("rate_per_sec must be > 0")
    if duration_s is None and num_requests is None:
        raise ValueError("either duration_s or num_requests must be provided")
    if duration_s is not None and duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    if num_requests is not None and num_requests <= 0:
        raise ValueError("num_requests must be > 0")
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if context_len <= 0:
        raise ValueError("context_len must be > 0")
    if target_len < context_len:
        raise ValueError("target_len must be >= context_len")

    rng = random.Random(seed)
    out: List[SimRequest] = []
    now = 0.0
    idx = 0
    while True:
        if num_requests is not None and idx >= num_requests:
            break
        if duration_s is not None and now > duration_s:
            break
        out.append(
            SimRequest(
                req_id=f"{req_id_prefix}-{idx+1}",
                arrival_ts=now,
                batch_size=batch_size,
                context_len=context_len,
                target_len=target_len,
            )
        )
        idx += 1
        now += rng.expovariate(rate_per_sec)
    return out
