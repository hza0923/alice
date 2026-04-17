# Scheduler（核心）

## 核心输入
- 模型配置（`build_scheduler_model_cfg`）
- Registry（`device_registry.v3`）
- 集群配置（`device_group`、`designated_device`、`network_bandwidth_mbps`）
- 求解参数（`prefer_bs`、`batch_size`、`target_seq_len`、`tail_candidates`）

## 主要功能
1) 从 registry 选择最优 `bs` 桶参数。
2) 构建 stage 计算模型（decode/prefill 的多项式系数和）。
3) 根据张量 shape 规则构建通信模型（decode/prefill）。
4) 使用 decode 空间模型做可行性约束：
   - `memory_gb = c0 + c1 * seq_len * batch_size`
5) 执行 DP 分配与 `designated_tail_n` sweep 搜索。
6) 导出：
   - 全局 `pipeline_strategy.v2`
   - 每 worker 的 `worker_strategy.v1`

## 核心算法
- 目标：最小化包含计算/通信耦合瓶颈的 TBT。
- 状态：层切分位置 + 剩余设备实例可用状态。
- 转移：若内存可行，将下一个层段分配给某设备实例。
- 代价：`stage_time = max(stage_compute_ms, stage_comm_ms)`。

## 输出契约
- `pipeline_strategy.json`：
  - `stage_models.decode/prefill.time_ms`
  - `stage_models.decode/prefill.comm_bytes`
  - `stage_models.decode/prefill.comm_time_ms`
  - `stage_models.decode.memory_gb`（仅 decode）
- `workers/*.strategy.json`：
  - `head_ordered_modules`
  - `tail_ordered_modules`

## 数据对齐规则
- 时间/通信均使用 phase-aware 多项式。
- 时间参数来自 registry 的 `bs` 桶。
- 通信参数来自固定 shape 规则，结果可复算。
- decode 空间来源在 stage model 中可追踪。

## 调用方式
- `python tools/scheduler/solve_strategy.py --cluster configs/cluster/all_devices.yaml`
