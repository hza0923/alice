"""Discrete-event simulation helpers for pipeline scheduling."""

from .batching import FCFSContiguousBatchScheduler, PackedBatch
from .des_engine import DESConfig, PipelineDESSimulator
from .metrics import SimulationReport, SimulationSummary
from .queue_engine import QueueSimConfig, Worker0QueueSimulator
from .request_model import SimRequest, generate_poisson_requests, generate_poisson_requests_from_specs

__all__ = [
    "DESConfig",
    "FCFSContiguousBatchScheduler",
    "PackedBatch",
    "PipelineDESSimulator",
    "QueueSimConfig",
    "SimRequest",
    "SimulationReport",
    "SimulationSummary",
    "Worker0QueueSimulator",
    "generate_poisson_requests",
    "generate_poisson_requests_from_specs",
]
