# pp_nextgen Module

## Purpose
Main implementation package for profiling, scheduling, runtime serving, and simulation.

## Core Execution Logic
- `profiling`: collects module timings and builds registry models.
- `scheduler`: solves stage partitioning using time/comm/memory constraints.
- `runtime`: executes strategy over gRPC workers.
- `simulation`: runs DES with the same strategy formulas.

## Math Formulas
- Time: `c0 + c1*x + c2*x^2` (`x=seq_len` for prefill, `x=context_len` for decode)
- Communication: `c0 + c1*x + c2*x^2` from tensor-shape analytics
- Decode memory: `c0 + c1*seq_len*batch_size`

## Input / Output and Legacy
- Input: model/cluster YAML and profiling JSON.
- Output: registry + pipeline/worker strategy + runtime metrics.
- Runtime keeps fallback support for legacy stage params.

## How to Invoke
- Build registry: `python tools/profiling/build_registry.py ...`
- Solve strategy: `python tools/scheduler/solve_strategy.py ...`
- Run runtime/sim: see `tools/` scripts.
