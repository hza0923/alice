# Simulation（核心）

## 输入
- `pipeline_strategy.v2`（必需）
- 请求流配置（到达率、时长/数量、batch/context/target 长度）
- 可选链路带宽覆盖

## 宏观流程
1) 读取策略并初始化各 stage 资源。
2) 构建链路带宽映射。
3) 生成请求到达事件。
4) 运行离散事件循环：
   - 请求到达
   - stage 计算完成
   - stage 间通信完成
   - worker0 tail 完成
5) 汇总指标并可选导出 trace。

## 模块依赖图（文本版）
```
tools/simulation/run_pipeline_des_sim.py
  -> pp_nextgen.runtime.strategy.load_pipeline_strategy
  -> pp_nextgen.simulation.generate_poisson_requests
  -> pp_nextgen.simulation.PipelineDESSimulator
       -> pp_nextgen.simulation.batching.FCFSContiguousBatchScheduler
       -> pp_nextgen.simulation.metrics.MetricsCollector
       -> pp_nextgen.runtime.strategy.expected_compute_ms
       -> pp_nextgen.runtime.strategy.expected_comm_bytes
       -> pp_nextgen.runtime.strategy.pipeline_stage_order
       -> pp_nextgen.runtime.strategy.stage_has_worker0_head_tail
       -> pp_nextgen.runtime.strategy.worker_matches_designated_device
  -> report.export_json(...)
  -> sim.export_traces(...) [optional]
```

## 单请求事件流图（文本版）
```
[request_arrival]
  -> 入 FCFS 等待队列
  -> 主控尝试 dispatch prefill(仅从等待队列取 1 个请求，不拼包)
  -> stage0 compute_done(prefill)
  -> link_transfer_done -> stage1 compute_done -> ... -> stageN compute_done
  -> (若 worker0 head/tail 开启)
       stageN -> link_transfer_done(to worker0 tail) -> worker0_tail_done
     (否则直接在末 stage 进入下个状态)
  -> 标记 first_token
  -> 若仍有 decode step:
       stage0 decode(step=1) -> ... -> stageN decode -> (tail/single) -> 回到 stage0
       重复直到 decode 结束
  -> request_finished
```

## 核心流水线建模
- stage 计算时长：
  - 通过 `expected_compute_ms(stage, phase, context_len, batch_size, branch)`
- 边传输字节数：
  - 通过 `expected_comm_bytes(stage, phase, context_len, batch_size, branch)`
- 分支建模：
  - worker0 可拆分为 `head/tail` 两类资源
- decode 上下文增长：
  - `context_len + decode_step`
- 调度边界：
  - 调度器只在请求到达后参与“prefill 入场”（决定何时把等待队列里的请求交给 stage0）
  - decode 阶段不再经过调度器，而由主控根据事件循环持续推进
- 当前调度策略：
  - FCFS，一次只放行 1 个请求（不做跨请求拼包）

## 输出
- DES 指标 JSON
- 可选输出：
  - 请求级 trace JSON
  - stage 级 trace JSON
  - 与 master latency 对齐的 sidecar

## 数据对齐标准
- 与 runtime 使用同一套策略公式计算期望计算/通信。
- 对策略 schema 执行严格约束（`pipeline_strategy.v2`）。

## 调用方式
- `python tools/simulation/run_pipeline_des_sim.py --strategy outputs/scheduler/export/pipeline_strategy.json --arrival-rate 5 --duration-s 10 --batch-size 1 --context-len 32 --target-len 64 --out outputs/runtime/simulation/metrics/run.json`
