# pp_nextgen 模块

## 功能定位
这是项目核心实现包，覆盖 profiling、scheduler、runtime、simulator 全流程。

## 执行根本逻辑
- `profiling`：采集模块耗时并构建 registry。
- `scheduler`：在时间/通信/空间约束下求解 stage 切分。
- `runtime`：按策略在 gRPC worker 上执行。
- `simulation`：用同一策略公式做离散事件仿真。

## 相关公式
- 时间：`c0 + c1*x + c2*x^2`（prefill 用 `seq_len`，decode 用 `context_len`）
- 通信：按张量形状解析得到 `c0 + c1*x + c2*x^2`
- decode 空间：`c0 + c1*seq_len*batch_size`

## 输入输出与 Legacy
- 输入：模型/集群 YAML 与 profiling JSON。
- 输出：registry、pipeline/worker strategy、运行指标。
- runtime 保留 legacy stage 参数回退读取。

## 如何调用
- 构建 registry：`python tools/profiling/build_registry.py ...`
- 生成策略：`python tools/scheduler/solve_strategy.py ...`
- 运行 runtime/sim：参见 `tools/` 下脚本。
