# Runtime gRPC Design（与当前实现对齐）

> 早期草稿中的部分 RPC 名（如 `PushStrategy`、`Heartbeat`）与标志名（如 `SHUTDOWN`）**未在 `pipeline_v2.proto` 中实现**。以 **`runtime/proto/pipeline_v2.proto`** 与 **`pp_nextgen/runtime/*`** 为准。  
> **完整操作说明、worker0 职责核对、未完成项**见 **`docs/09_runtime_implementation_status.md`**。

## Objectives

- 跨节点通信以 **gRPC unary `SendFrame`** 为基线。
- Master 控制面与 Worker 数据面分离；**worker0** 同时承担 **ingress（head）** 与 **环返回（tail）** 两段计算（见下节）。
- 显式 **停止环**（`pipeline_stop` + `ring_remaining`）与 **Master 线性 `NotifyPipelineEnd`**，避免提前关断与在途帧竞态。

## Runtime Roles

### Master（`MasterControl`）

- 登记所有 worker；收齐后为 **首个 pipeline stage** 建立 **DataPlane** 连接以注入首帧。
- **`SubmitTask`**：业务请求或 **`is_end`** → **`pipeline_stop`** 控制帧。
- **`ReportRequestFinished` / `ReportPipelineComplete`**：记录延迟、触发 **`NotifyPipelineEnd`** 链（沿 **线性 worker 顺序**，非数据面环顺序）。

### Worker（`DataPlane`）

- **`SendFrame`**：接收上一跳或 Master 的 **Frame**。
- **`NotifyPipelineEnd`**：控制面收尾通知；转发给线性下一节点后本地 `_done`、关 KV、可导出指标。

### Worker0（首个 `pipeline_stages[].worker_name`）

- **双队列**（存在 **tail 模块** 时）：**`ring_return=false`** → **head 队列**（Master 首注与 tail 未完成时的再入队）；**`ring_return=true`** → **tail 队列**（环上最后一跳回到本机）。
- **无 tail 模块**时：所有帧进 **单一 head 队列**，在本机完成原「末节点」decode 语义。
- **`pipeline_stop`**：仅在 **head** 上 **记录 pending**，待 **`_open_requests` 清空** 后再送入环；**`ReportPipelineComplete`** 在 **tail** 收到 **`ring_remaining==0`** 的停止帧后由 **worker0** 上报。

### 其他 Worker

- **单队列**；按 **`head_ordered_modules`** 执行（**`tail_ordered_modules` 导出为空**）。
- 转发时若 **下一跳为 worker0**，置 **`ring_return=true`**。
- **不再**负责每 token 的 **`context_len` 推进** 与 **`ReportRequestFinished`**（已迁至 **worker0**）。

## Protocol（`pipeline_v2.proto` 摘要）

**Frame**（数据面）主要字段：

- `req_id`, `step_id`, `phase`（`PREFILL` / `DECODE`）
- `context_len`, `target_len`, `batch_size`
- `payload`（当前模拟为占位字节）
- `end_of_request`
- **`pipeline_stop`**：停止环控制
- **`ring_remaining`**：每 hop 递减
- **`pipeline_total_tokens`**：停止环最后一跳携带（中间节点可为 0，**worker0 以本地累计为准**）
- **`ring_return`**：**true** 表示环回到 worker0，走 **tail** 路径

**MasterControl** RPC：

- `RegisterWorker`, `GetNextWorker`, `GetPeerAddress`
- `SubmitTask`, `ReportRequestFinished`, `ReportPipelineComplete`

**DataPlane** RPC：

- `SendFrame`, `NotifyPipelineEnd`

> 设计讨论过的 **`PushStrategy` / `Heartbeat` / 流式 `PipeFrames`** 等：**未实现**；见 **`docs/09_runtime_implementation_status.md`** 未完成表。

## Concurrency Model（实现侧）

- 每 Worker：**bounded `task_queue` 或 worker0 的 head/tail 队列** + **bounded `send_queue`**。
- **`_task_processor`**（及 worker0 变体）与 **`_send_processor`** 异步任务分离 send 与 compute。
- Master：**`_send_loop`** 将任务帧 **SendFrame** 到首节点。

## Observability

- Per-worker **`RequestJournal`** 可导出 JSON（默认目录 **`runtime/logs/`**，由 `run_cluster_sim` 的 `--runtime-logs-dir` 配置）。
- Master **`MasterLatencyTracker`** 可选导出。
- 指标字段含义与 **Windows 上 `actual_compute_ms` 量化**见 **`docs/09_runtime_implementation_status.md` 第一节**。

## Migration Notes from Legacy Role Code

- **结束条件**：停止帧 **pending**，待请求全部 **`ReportRequestFinished`** 后再入环（worker0 **`_open_requests`**）。
- **末节点 decode**：迁至 **worker0 tail**（或 head-only 模式）。
- **不要在未Drain 环上提前 `NotifyPipelineEnd`**：`ReportPipelineComplete` 在 **worker0** 收到 **`ring_remaining==0`** 后再由 Master 通知。
