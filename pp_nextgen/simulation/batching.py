"""Contiguous-batch schedulers for request admission."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

from .request_model import SimRequest


@dataclass(frozen=True)
class PackedBatch:
    requests: List[SimRequest]
    packed_batch_size: int
    context_len: int
    target_len: int


class FCFSContiguousBatchScheduler:
    """
    FCFS single-request scheduler.

    Requests are admitted strictly one-by-one in arrival order.
    No cross-request packing is performed.
    """

    def __init__(self, *, max_batch_size: int, packing_window_ms: float = 0.0) -> None:
        if max_batch_size <= 0:
            raise ValueError("max_batch_size must be > 0")
        if packing_window_ms < 0:
            raise ValueError("packing_window_ms must be >= 0")
        self.max_batch_size = int(max_batch_size)
        self.packing_window_s = float(packing_window_ms) / 1000.0
        self._queue: Deque[SimRequest] = deque()

    def enqueue(self, req: SimRequest) -> None:
        self._queue.append(req)

    def has_pending(self) -> bool:
        return bool(self._queue)

    def pending_count(self) -> int:
        return len(self._queue)

    def pop_next_batch(self, now_ts: float) -> Optional[PackedBatch]:
        if not self._queue:
            return None
        head = self._queue[0]
        if self.packing_window_s > 0 and now_ts < head.arrival_ts + self.packing_window_s:
            return None

        req = self._queue.popleft()
        total_bs = int(req.batch_size)
        if total_bs > self.max_batch_size:
            raise ValueError(
                f"single request batch_size={total_bs} exceeds max_batch_size={self.max_batch_size}"
            )

        return PackedBatch(
            requests=[req],
            packed_batch_size=total_bs,
            context_len=int(req.context_len),
            target_len=int(req.target_len),
        )
