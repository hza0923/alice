# Runtime（核心）

## 角色与输入
- Master 角色：
  - 读取流水线拓扑并负责 worker 注册/收敛控制
- Worker 角色：
  - 读取 `pipeline_strategy.v2` 与 `worker_strategy.v1`
  - 读取模型 YAML 构建执行图

## 宏观工作流
1) 启动 master，worker 完成注册。
2) worker 从 master 获取下一跳地址。
3) 首 worker 接收请求并启动 prefill/decode 流程。
4) 各 worker 按模块计划（`head/tail/single`）执行并向下一阶段转发 payload。
5) 末阶段结果回流到 worker0 tail 或进入最终完成路径。
6) master 汇总完成状态并导出延迟/指标。

## 通信逻辑
- 数据面：worker 间 gRPC `SendFrame`。
- 控制面：注册、下一跳查询、完成上报。
- 期望通信大小/耗时来自策略 `stage_models.<phase>.comm_*`。

## 执行逻辑
- 计算执行：
  - `sleep_executor`：按期望模型时延执行（可控）
  - `shape_executor`：执行 shape-only 路径
- 期望计算/通信都从策略多项式模型计算。

## 输出数据
- worker 级 metrics/journal JSON（runtime logs）。
- master 延迟 sidecar（兼容 `master_latency.v1`）。

## 数据对齐标准
- runtime 仅接受：
  - `pipeline_strategy.v2`
  - `worker_strategy.v1`
- 期望计算/通信模型只从 `stage_models` 读取。

## 调用方式
- 通过 runtime CLI（`pp_nextgen/runtime/cli.py`）启动 master/worker，传入：
  - pipeline strategy 路径
  - worker strategy 目录
  - model config 路径
