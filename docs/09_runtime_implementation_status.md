# Runtime 实现状态与操作说明

本文档基于当前仓库实现整理：**已完成能力**、**调用关系**、**Master / Worker（含 worker0）工作流**、**原「末节点」职责迁移核对**，以及**未完成项**（模块归属、作用、接口）。

---

## 一、指标里 `actual_compute_ms` 为何曾常落在约 16ms / 31ms？

**不是** `time.monotonic()` 粗粒度：在 Windows 上 CPython 的 `time.monotonic()` 通常基于 **QueryPerformanceCounter**，分辨率远高于毫秒。

**历史原因**在 **sleep 路径与事件循环定时**：

1. 若短延时走 **`asyncio.sleep`**：在 **Windows** 上，asyncio 的底层等待往往与 **系统默认定时器刻度**（常见约 **15.625ms**）对齐，实际休眠容易落在 **约 16ms / 31ms** 附近，与 **`expected_compute_ms`（例如 ~25ms）** 不一致。
2. 若短延时走 **`run_in_executor` + 在主协程上用 `monotonic()` 包一层 `await`**：线程内睡眠已较准，但 **协程恢复** 仍可能落在上述刻度上，**多计 0～16ms**，也会出现 **16 / 31ms** 这类聚集。

**当前实现（对齐 legacy `new_sleep` 思路）**：`precise_sleep.py` 在 Windows 上对 **低于 ~100ms** 的延时使用 **`timeBeginPeriod(1)` + `Sleep(ms-2)` + QPC 忙等尾**（与旧仓库 `new_sleep.py` 的混合策略一致）；短延时在 **线程池线程内** 睡眠并用 **`time.perf_counter()`** 计量（在部分 Windows/Python 组合上 **`time.monotonic()` 对短间隔会呈 ~15.6ms 量化**，与线程无关，会导致指标上出现虚假的 16/31ms 台阶）。

仅 **较长** 休眠（默认阈值 **≥ 100ms**）仍走 **`asyncio.sleep`**，以减少线程池占用；其返回的耗时同样用 **`perf_counter()`** 计量。

**`actual_comm_ms`**：在 `SendFrame` 前后用 **`perf_counter()`** 相减；**首包/建连**可能出现极大值（如数万 ms），属于 **gRPC 冷启动**，不是稳定传输 RTT。部分 hop 上为 **0** 与异步发送队列、记录时机有关，需结合实现阅读 `worker/service.py` 中 `_send_processor`。

---

## 二、已完成部分

### 2.1 使用方法

| 能力 | 命令 / 入口 |
|------|----------------|
| 安装包 | 仓库根目录：`pip install -e .` |
| 生成 gRPC 桩 | `python tools/gen_runtime_proto.py`（需 `grpcio-tools`） |
| 求解策略并导出 worker JSON | `python tools/solve_strategy.py --cluster <cluster.yaml>` |
| 本机 Master + 多 Worker 模拟 | `python tools/run_cluster_sim.py`（可选 `--runtime-logs-dir`，默认 `runtime/logs`） |
| 单独进程 | `python -m pp_nextgen.runtime.cli master ...` / `worker ...` |

**Worker 策略文件**（`scheduler/export/workers/*.strategy.json`）使用 **`head_ordered_modules` + `tail_ordered_modules`**；仅 **`designated_device`** 对应 worker 的 **`tail_ordered_modules` 为全局策略里最后 `designated_tail_n` 个模块**，其余机器 **`tail_ordered_modules` 为空**。兼容旧字段 **`ordered_modules`** 时由 `split_head_tail_modules_from_execution_plan` 解析。

### 2.2 执行与调用逻辑（简要）

1. **客户端 / 模拟脚本** → **Master** `SubmitTask`（或 `is_end` 注入 `pipeline_stop`）。
2. **Master** 将任务打成 **Frame**，经发送队列对 **首个 stage 的 DataPlane** 做 **`SendFrame`**（`ring_return=false`）。
3. **Worker（含 worker0）**：
   - **非首节点**：单队列收帧 → 执行（sleep/shape）→ **`SendFrame` 下一跳**；若下一跳是 **worker0**，则 **`ring_return=true`**。
   - **worker0**：**Master 与环上返回** 分流到 **`_head_queue` / `_tail_queue`**（无 tail 模块时全部走 head 队列）。
4. **Decode 推进**：有 **tail 模块** 时在 **worker0 的 tail 路径**上 **`context_len++`**、累加 **`_total_tokens`**、判断是否 **`ReportRequestFinished`**；无 tail 时在 **worker0 head** 上完成同等逻辑。
5. **`pipeline_stop`**：在 **worker0 head** 上**仅记录 pending**，待 **`_open_requests` 清空**（请求已 `ReportRequestFinished`）后再送入环；**`ReportPipelineComplete`** 在 **worker0 tail** 收到停止环最后一跳后上报；Master 再 **`NotifyPipelineEnd`** 链式通知各节点。
6. **指标**：`run_cluster_sim` 默认将 **`*.runtime_metrics.json`** 写到 **`runtime/logs/`**。

### 2.3 Master 节点工作流

- **`RegisterWorker`**：登记 `worker_name → address`，收齐后对首节点建 **`DataPlaneStub`**。
- **`GetNextWorker` / `GetPeerAddress`**：按 **`pipeline_strategy.json`** 提供环拓扑与线性通知链地址。
- **`SubmitTask`**：业务帧或 **`pipeline_stop`** 帧入队，由后台循环 **`SendFrame`** 到首 worker。
- **`ReportRequestFinished` / `ReportPipelineComplete`**：收 worker 上报；**`ReportPipelineComplete`** 后触发 **`NotifyPipelineEnd`**（从首节点沿 **线性顺序** 转发，与数据面环不同）。

### 2.4 Worker 节点工作流（含 worker0）

| 角色 | 队列 / 路径 | 职责 |
|------|-------------|------|
| **worker0** | `ring_return=false` → **head**；`ring_return=true` → **tail**（有 tail 模块时） | **head**：ingress 计算并送入环；**tail**：decode 步进、KV/会话收尾、`ReportRequestFinished`；**pending stop** 与 **`_open_requests`** 门控 |
| **中间节点** | 单 `task_queue` | 按本机 **head 列表**（全量在 head，tail 为空）计算并转发；**停止帧**仅转发；**不向 Master 报 decode 完成** |
| **原语义「环上最后一台物理机」** | 仍执行其 stage 模块，但 **不再承担**「每 token 推进 context、报请求结束」——已迁到 **worker0** |

### 2.5「末节点」职责是否已迁到 worker0（核对）

| 原「is_last_worker / 末节点」职责 | 当前承担位置 |
|-----------------------------------|----------------|
| 每 decode 步 **`context_len` 递增** | **worker0 tail**（有 tail）；否则 **worker0 head** |
| **`ReportRequestFinished`** | 同上 |
| **`_total_tokens` 累加**（业务语义） | **worker0** |
| **`pipeline_stop` 与全部业务请求完成顺序** | **worker0**：pending stop + **`_open_requests`** |
| **`ReportPipelineComplete`（带 token 统计）** | **worker0 tail** 收到 **`ring_remaining==0`** 的 stop 帧；中间节点转发的 **`pipeline_total_tokens` 可为 0**，以 **worker0 本地 `_total_tokens`** 为准 |
| 中间节点仅 **转发 stop、递减 `ring_remaining`** | **各非首 worker**（与旧环传播一致） |

---

## 三、未完成部分（按模块）

以下项在规划或骨架中存在，**尚未作为生产路径完成**或与文档初稿不一致。

| 模块 / 区域 | 作用 | 接口 / 约定（当前或目标） |
|-------------|------|---------------------------|
| **真实推理执行** | 用权重与真实算子替换 sleep/shape | Worker 内 Executor 对接 `PipelineModel` 或外部引擎；Frame 载荷与 KV 生命周期需与 scheduler 内存模型一致 |
| **`runtime/proto` 中未实现的 RPC** | 控制面扩展 | 如设计稿中的 `PushStrategy`、`Heartbeat`、流式 `PipeFrames` 等：**未**在 `pipeline_v2.proto` / Master 中实现 |
| **传输 A/B 与流式** | 降延迟、可替换传输 | `docs/05_transport_ab_plan.md`；当前仅 unary **`SendFrame`** |
| **Master 侧队列深度 / 节流** | 背压与可观测性 | 设计见 `04_runtime_grpc_design.md`；实现未接 heartbeat 上报 |
| **Schema 校验器** | 合同强校验 | `docs/02_data_contracts.md` 中的 `validate_*` 骨架；**未**形成完整 CI 校验 |
| **自动化测试** | 回归与故障注入 | `docs/07_migration_and_validation.md` 清单；仓库内**未**系统化落地 |
| **严格「同队列 stop 与 decode 顺序」** | 避免 stop 插队在未完成帧之前 | 依赖客户端顺序；**未**实现显式优先级队列或重排 |
| **文档与实现细节对齐** | `END_OF_REQUEST`/`SHUTDOWN` 等标志 | `04_runtime_grpc_design.md` 部分早期字段名与 **`pipeline_v2.proto`** 中 **`pipeline_stop` / `ring_return`** 不一致，以 **proto + `pp_nextgen/runtime` 代码** 为准 |

---

## 四、相关文档修订说明

- **`01_architecture_overview.md`**：目录与 legacy 映射已与 **`pp_nextgen/`** 包布局对齐。
- **`02_data_contracts.md`**：worker 策略示例改为 **`head_ordered_modules` / `tail_ordered_modules`**，并注明 **`schedule_input`** 字段名。
- **`03_scheduler_design.md`**：导出契约中 **per-worker** 已写明 **`head_ordered_modules` / `tail_ordered_modules`**，与 runtime 消费一致。
- **`04_runtime_grpc_design.md`**：与当前 **MasterControl / DataPlane**、**worker0**、**Frame 扩展字段** 对齐；过时 RPC/标志已标注或删除。
- **`07_migration_and_validation.md`**：里程碑与当前进度对齐。

详细协议字段以 **`runtime/proto/pipeline_v2.proto`** 与 **`pp_nextgen/runtime/grpc_gen/*`** 为准。
