# 配置模块

## 功能定位
存放 profiling、scheduler、runtime、simulator 共同使用的模型、集群与运行参数 YAML。

## 执行根本逻辑
各 CLI 从该目录读取 YAML，完成字段标准化后传给下游模块。

## 相关公式
- 本目录不直接做拟合。
- 但它提供其他模块计算所需参数：
  - 时间/通信：`c0 + c1*x + c2*x^2`
  - decode 空间：`c0 + c1*seq_len*batch_size`

## 输入输出与 Legacy
- 输入：人工维护的 YAML。
- 输出：进程内配置字典。
- 通过可选字段与默认值保持 legacy 配置兼容。

## 如何调用
- 构建 registry：`python tools/profiling/build_registry.py --model-config configs/model/llama2_7b.yaml ...`
- 生成策略：`python tools/scheduler/solve_strategy.py --cluster configs/cluster/all_devices.yaml`
