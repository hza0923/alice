# Scheduler Design

## Design Principle

Preserve legacy DP optimization objective and state semantics, while moving all
I/O and model-specific normalization to adapter layers.

## Layered Structure

- `scheduler/adapters/input_adapter.py`
  - Load config + registry + runtime solve options.
  - Normalize module names and batch-size bucket selection.
  - Provide pure-cost query interfaces to core solver.
- `scheduler/core/solver_legacy_compatible.py`
  - Keeps equivalent DP state and transition semantics.
  - No direct JSON parsing and no CLI concerns.
- `scheduler/core/tail_sweep.py`
  - Outer loop over `designated_tail_n in [1..7]`.
  - Runs solver for each candidate and picks min `tbt_ms`.
- `scheduler/export/strategy_writer.py`
  - Writes global and per-worker strategy files.

## Core-Preserved Constraints

- DP objective remains min-max TBT over compute/comm stage bottlenecks.
- Device-state counting policy remains compatible with designated device reserve.
- Memory feasibility checks remain hard constraints per candidate split.
- Communication cost model remains consumed by solver, but model source is
  adapter-supplied.

## Tail Sweep Algorithm

Input:

- `module_graph` with ordered effective modules per logical layer
- `designated_device`
- candidate set `N = {1,2,3,4,5,6,7}`

Output:

- best strategy and full candidate report

Pseudo-flow:

```text
best = None
for tail_n in N:
    constrained_problem = apply_designated_tail_constraint(problem, tail_n)
    result = solve_dp_legacy_compatible(constrained_problem)
    if result.feasible and (best is None or result.tbt_ms < best.tbt_ms):
        best = result
return best
```

## DP Boundaries that Must Track `tail_n`

- Initial designated prefix+suffix reservation.
- Middle free-range DP upper index bound.
- Final hop to designated stage boundary index.
- Final allocation append/extend for designated tail modules.

## Solver Interfaces (Markdown Skeleton)

```python
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class SolveRequest:
    model_name: str
    batch_size: int
    target_seq_len: int
    designated_device: str
    designated_tail_n: int
    use_fine_grained: bool

@dataclass
class SolveResult:
    feasible: bool
    tbt_ms: float
    allocation: Dict[str, List[int]]
    debug: dict

class CostProvider:
    def layer_latency_ms(self, device: str, layer_idx: int, seq_len: int) -> float: ...
    def layer_memory_gb(self, device: str, layer_idx: int, seq_len: int) -> float: ...
    def comm_time_ms(self, src: str, dst: str, layer_idx: int, seq_len: int) -> float: ...

def solve_dp_legacy_compatible(req: SolveRequest, costs: CostProvider) -> SolveResult: ...
def solve_with_tail_sweep(req: SolveRequest, costs: CostProvider, n_values: List[int]) -> SolveResult: ...
```

## Strategy Export Contract

- Global strategy:
  - `designated_tail_n` chosen
  - objective score + solve metadata
- Per-worker strategy:
  - **`head_ordered_modules` + `tail_ordered_modules`**（取代单一 `ordered_modules` 列表；仅 **designated** 对应 worker 的 **tail** 为全局策略最后 `designated_tail_n` 个模块，其余 worker 的 **tail** 为空；见 **`docs/02_data_contracts.md`**）
  - phase execution toggles
  - local transfer-size computation policy

## Compatibility Checklist

- Keep legacy module-order interpretation unchanged by default.
- Keep communication estimator inputs unchanged at core level.
- Keep memory hard constraints at same decision points.
- Allow configurable strict mode to assert parity with legacy outputs on fixed
  test cases.
