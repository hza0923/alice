# Migration and Validation

> **当前实现进度**（runtime 用法、worker0、已完成/未完成清单）以 **`docs/09_runtime_implementation_status.md`** 为准；本节保留分阶段目标与校验思路。

## Migration Strategy

Incremental migration with parity gates:

1. build docs/contracts and skeleton modules
2. implement profiling pipeline refactor
3. implement scheduler adapter + tail sweep
4. implement runtime refactor with sleep executor（**已完成**：`pp_nextgen/runtime` + worker0 + `pipeline_v2`）
5. enable shape executor and KV lifecycle（**部分完成**：shape + numpy 占位 KV，无真实权重）
6. run transport A/B and decide default（**未完成**）

Legacy system remains runnable during all phases.

## Milestone Gates

### Gate 1: Contracts Stable

- schemas validated for:
  - `all_results`
  - `device_registry`
  - `pipeline_strategy`
  - `worker_strategy`
- config templates load successfully

### Gate 2: Scheduler Parity

- for fixed sample inputs, `tail_n=1` output is behavior-compatible with legacy
  objective semantics
- `tail_n=1..7` sweep returns feasible best candidate report

### Gate 3: Runtime Correctness

- request lifecycle reaches terminal state under:
  - single request（**本机模拟已通过**：`tools/run_cluster_sim.py`）
  - mixed burst requests（**未系统化验证**）
- bounded queue behavior verified during synthetic downstream stalls（**未系统化验证**）

### Gate 4: Real Execution Mode

- per-`req_id` KV creation/append/cleanup verified（**占位 KV**：无权重，会话随请求结束关闭）
- no stale sessions after completion（**模拟路径已做**；真实推理未接）
- shape executor path is deterministic under fixed seed（**model 内 RNG 固定种子**）

### Gate 5: Transport Decision

- benchmark report generated（**待定**；见 `docs/05_transport_ab_plan.md`）
- default transport and fallback are documented

## Validation Checklist

- Unit tests:
  - schema validators
  - cost model evaluators
  - strategy serializer/deserializer
- Integration tests:
  - multi-worker ring with sleep executor
  - mixed request lengths with deterministic order mode
- Failure tests:
  - timeout
  - duplicate frame
  - forced shutdown and restart

## Rollback Plan

- keep feature flags for:
  - `transport_mode`
  - `executor_mode`
  - `strict_ordering`
- fallback path:
  - `grpc + sleep_executor + tail_n=1`

## Deliverables

- `docs/*` finalized
- `tools/` command entrypoints wired
- smoke test scripts and benchmark reports archived under artifacts
