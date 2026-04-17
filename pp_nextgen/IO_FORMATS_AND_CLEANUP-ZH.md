# 当前输入输出格式确认与精简清单

## 已明确并作为唯一生效的格式

## 1) Profiling 输入
- 采集输入（build 阶段）：
  - `*_all_results.json`（来自 `pp_nextgen/profiling/capture/split_module_bench.py`）
- Registry 输出：
  - `device_registry.v3`

## 2) Scheduler 输入输出
- 输入：
  - `device_registry.v3`
  - 模型/集群 YAML
- 输出：
  - `pipeline_strategy.v2`
  - `worker_strategy.v1`

## 3) Runtime / Simulator 输入
- Runtime：
  - 仅接受 `pipeline_strategy.v2` + `worker_strategy.v1`
- Simulator：
  - 仅接受 `pipeline_strategy.v2`

## 4) 关键公式契约（当前标准）
- 时间：`c0 + c1*x + c2*x^2`
- 通信：`c0 + c1*x + c2*x^2`
- decode 空间：`c0 + c1*seq_len*batch_size`

## 已执行的代码精简
- `pp_nextgen/scheduler/adapters/registry_adapter.py`
  - 删除对旧 registry schema 的分支回退，改为严格 `device_registry.v3`
- `pp_nextgen/runtime/strategy.py`
  - 删除 `stage_params` 与旧字段回退计算路径
  - 强制 `pipeline_strategy.v2` / `worker_strategy.v1`
  - 强制从 `stage_models` 读取 time/comm/memory
- `pp_nextgen/schemas/validate.py`
  - 增强 `device_registry.v3` 最小校验（schema + 模型字段完整性）

## 仍保留但标注为“可继续收敛”的路径
- `profiling/legacy_ingest.py` 仍承担采集 JSON 的标准化入口（当前 capture 输出命名仍沿用历史约定）。
- `pipeline_strategy.v2` 顶层 `comm_time_ms` / `comm_bytes_to_next` 仍保留（便于旧分析脚本读取），但运行核心已使用 `stage_models`。

## 下一步建议（可继续做）
- 将 capture 输出名从 `*_all_results.json` 迁移到明确版本名（例如 `all_results.v1.json`），并同步 `tools/build_registry.py` 入参约束。
- 在 `pp_nextgen/schemas` 中新增 `validate_pipeline_strategy_v2` 与 `validate_worker_strategy_v1` 的严格字段校验并在 runtime/sim 入口调用。
