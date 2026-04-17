# Real Execution Mode (No Weights)

## Purpose

Provide a real compute path that does not load model weights, but still executes
shape-correct operations and KV cache lifecycle to validate runtime behavior.

## Scope

- Keep request inputs:
  - `req_id`
  - `context_len`
  - `target_len`
  - phase and module segment
- Keep deterministic per-worker module execution ordering from strategy.
- Keep per-request KV cache state in worker-local memory.

## Executor Types

- `sleep_executor`:
  - uses profiled formula for time simulation only
- `shape_executor`:
  - executes PyTorch kernels with synthetic tensors
  - no model weight loading
  - validates memory and lifecycle behavior

## KV Cache Lifecycle

Per worker:

- `kv_store: Dict[str, KvSession]` keyed by `req_id`
- each `KvSession` tracks:
  - `num_tokens`
  - layer/module cache tensors
  - last `step_id`
  - state (`prefill` or `decode`)

Lifecycle:

1. `SessionOpen(req_id)` on first frame
2. `PrefillStep` appends initial KV
3. `DecodeStep` appends one token (or configured chunk)
4. `SessionClose(req_id)` on end-of-request or failure cleanup

## Compute Model by Module

- `qkv_projection`, `o_projection`, `gate_up_projection`, `down_projection`:
  - synthetic dense matmul with configured dimensions
- `attn_qk`, `attn_av`:
  - baseline: explicit `matmul` path
  - optional capability path: `torch.nn.functional.scaled_dot_product_attention`

## Kernel Selection Policy

`shape_executor` chooses kernel by runtime capabilities:

- default:
  - always use matmul baseline for compatibility
- optional:
  - enable SDPA when explicit flag is true and capability check passes

Capability checks:

- torch version
- CUDA availability
- dtype support
- fallback verification pass

## Performance/Correctness Focus

Not a throughput benchmark for absolute peak.

Primary goals:

- request ordering correctness
- overlap correctness
- KV cache memory growth trend
- resource cleanup correctness

## API Skeleton

```python
from dataclasses import dataclass
from typing import Dict, Protocol

@dataclass
class ExecContext:
    req_id: str
    step_id: int
    phase: str
    context_len: int
    target_len: int
    batch_size: int

@dataclass
class KvSession:
    req_id: str
    num_tokens: int
    last_step_id: int
    state: str
    tensors: Dict[str, object]

class ModuleExecutor(Protocol):
    async def run_modules(self, ctx: ExecContext, module_ids: list[str]) -> None: ...

class KvManager(Protocol):
    def open_if_needed(self, req_id: str) -> KvSession: ...
    def append(self, req_id: str, module_id: str, tensor: object) -> None: ...
    def close(self, req_id: str) -> None: ...
```

## Failure Handling

- stale `step_id` for existing `req_id`:
  - reject or ignore based on strict mode
- missing `req_id` on decode:
  - recover by explicit prefill-required response
- OOM:
  - emit terminal error and force session close

## Deliverables in Code Phase

- `runtime/executors/shape_executor.py`
- `runtime/common/kv_manager.py`
- integration with worker service frame processor
