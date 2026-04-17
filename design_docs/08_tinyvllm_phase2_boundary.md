# tinyvllm Phase-2 Integration Boundary

## Positioning

tinyvllm integration is explicitly deferred to phase-2. This iteration defines
only stable boundaries to avoid architecture lock-in.

## Why Deferred

- current priority is stable cross-node PP runtime
- scheduler/runtime contracts need to be finalized first
- tinyvllm internals assume full-model engine semantics that differ from
  module-segment execution

## Integration Principles

- do not embed scheduler in every worker
- keep request ordering policy global and deterministic
- avoid cross-device phase divergence due to partial preemption

## Hard Boundary Contracts

- `EngineAdapter` interface only:
  - receives `WorkerExecutionPlan`
  - executes module-segment workload for given frame
  - returns updated frame state
- scheduler output remains engine-agnostic JSON contract
- runtime transport remains engine-agnostic frame protocol

## Non-Goals (Phase-2 Boundary)

- no deep reuse of tinyvllm block manager in phase-1
- no request preemption in distributed PP mode
- no mixed prefill/decode state divergence across workers

## Preemption Policy

Distributed PP mode defaults to `preemption_disabled=true`.

Rationale:

- prevents worker-local memory asymmetry from changing request phase at only
  subset of workers
- keeps pipeline semantics aligned across all nodes

Future optional mode:

- coordinated preemption only if global control plane can force synchronized
  state transitions for the same `req_id` across all workers

## Proposed Adapter Interface

```python
from dataclasses import dataclass
from typing import Protocol

@dataclass
class WorkerExecutionPlan:
    worker_name: str
    ordered_modules: list[str]
    preemption_disabled: bool

@dataclass
class RuntimeFrame:
    req_id: str
    step_id: int
    phase: str
    context_len: int
    target_len: int

class EngineAdapter(Protocol):
    async def initialize(self, plan: WorkerExecutionPlan) -> None: ...
    async def execute(self, frame: RuntimeFrame) -> RuntimeFrame: ...
    async def shutdown(self) -> None: ...
```

## Readiness Checklist for Phase-2 Start

- scheduler and runtime contracts unchanged for one full release cycle
- no unresolved queue/backpressure correctness issues
- shape executor benchmark baseline archived
- memory policy for heterogeneous devices validated by integration tests
