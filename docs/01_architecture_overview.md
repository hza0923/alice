# Architecture Overview

## Context

This project rebuilds the legacy fine-grained PP pipeline in a separate
workspace. Legacy reference source:

- `grpc_heterogeneous_pipeline/3060/profile`
- `grpc_heterogeneous_pipeline/3060/src`
- `grpc_heterogeneous_pipeline/3060/role`

The redesign keeps solver intent and deployment constraints while reducing
coupling and improving maintainability.

## High-Level Boundaries

- `profiling`: runs per-device benchmarks and produces normalized timing/memory
  parameters.
- `scheduler`: consumes normalized registry + model/cluster config and produces
  strategy artifacts.
- `runtime`: executes strategy over gRPC with overlap and backpressure safety.
- `tools`: small CLIs for profile build, strategy solve, and simulation.

## End-to-End Data Flow

```mermaid
flowchart LR
  profileCapture[ProfileCapturePerDevice] --> fitParams[FitParamsByBsAndPhase]
  fitParams --> registryBuild[DeviceRegistryBuild]
  registryBuild --> schedulerInput[SchedulerInputAdapter]
  schedulerInput --> solverCore[SolverCorePreserved]
  solverCore --> tailSweep[TailN1To7Sweep]
  tailSweep --> strategyExport[StrategyExport]
  strategyExport --> masterNode[RuntimeMaster]
  masterNode --> workerNodes[RuntimeWorkers]
  workerNodes --> masterNode
```

## Legacy to Nextgen Mapping

| Legacy File | Nextgen Module | Responsibility Shift |
|---|---|---|
| `3060/profile/split_module_test3.py` | `profiling/capture/runner.py` + `profiling/capture/kernels.py` | Separate benchmark orchestration from kernel-level profiling |
| `3060/profile/build_device_registry.py` | `profiling/fit/fitter.py` + `profiling/build/registry_builder.py` | Split fitting and registry assembly |
| `3060/profile/llama2-7b_all_results.json` | `profiling/artifacts/all_results/*.json` | One-file-per-device canonical raw result |
| `3060/profile/device_registry.json` | `profiling/artifacts/registry/device_registry.v3.json` | Versioned normalized registry |
| `3060/src/llama2_7b_config.py` | `configs/model/*.yaml` + `scheduler/adapters/input_adapter.py` | Move constants to config, keep adapter logic in scheduler layer |
| `3060/src/cluster_fine_grained_pp_scheduler.py` | `scheduler/core/solver_legacy_compatible.py` + `scheduler/core/tail_sweep.py` | Preserve DP core semantics, add outer tail sweep |
| `3060/role/master.py` | `pp_nextgen/runtime/master/service.py` | Control plane (registration, SubmitTask, lifecycle reports) |
| `3060/role/worker.py` | `pp_nextgen/runtime/worker/service.py` + `pp_nextgen/runtime/executors/*` | Data plane, worker0 head/tail queues, executors |
| `3060/role/pipeline.proto` | `runtime/proto/pipeline_v2.proto`（生成代码在 `pp_nextgen/runtime/grpc_gen/`） | `pipeline_stop`、`ring_return`、停止环与 worker0 语义 |

## Directory Layout（与当前仓库一致）

```text
grpc_heterogeneous_pipeline_nextgen/
  docs/
  configs/
    model/
    cluster/
    runtime/
  pp_nextgen/                 # 可安装 Python 包（pip install -e .）
    runtime/                  # Master/Worker 服务、executors、metrics、model、precise_sleep
    scheduler/
    profiling/
    ...
  runtime/
    proto/                    # pipeline_v2.proto（生成物在 pp_nextgen/runtime/grpc_gen/）
    README.md                 # 子模块说明（实现代码在 pp_nextgen）
  runtime/logs/               # 默认 runtime 指标输出（run_cluster_sim）
  scheduler/
    export/
      pipeline_strategy.json
      workers/*.strategy.json
  profiling/
  tools/
```

实现状态与调用说明见 **`docs/09_runtime_implementation_status.md`**。

## Non-Goals in This Iteration

- No modification to legacy `3060` code.
- No immediate tinyvllm deep merge (kept as phase-2 boundary).
- No mandatory transport replacement for first production path (gRPC remains
  baseline).

## Runtime 进展（摘要）

- **已实现**：`pp_nextgen` 下 gRPC Master/Worker、worker0 head+tail 双队列、按 designated 设备切分 `head_ordered_modules`/`tail_ordered_modules`、延迟 `pipeline_stop`、Windows 侧可配置的高精度 sleep 路径、指标默认写入 `runtime/logs/`。
- **未实现**：真实权重推理、流式传输、设计稿中部分未落地的 Master RPC、完整契约校验与测试矩阵。详见 **`docs/09_runtime_implementation_status.md`**。
