# Profiling Build

## 核心功能
收集多设备 profile 结果，完成分模块/分阶段拟合，并组装输出 `device_registry.v3`。

## 输入
- 采集文件：`*_all_results.json`（可传文件或目录）
- 模型 YAML：
  - 必须包含 `memory_contract.kv_per_token_bytes`

## 内部流程
1) 读取并标准化采集文件（`legacy_ingest.py`）。
2) 遍历每个 `(batch_size, module)` 样本集合。
3) 执行时间拟合：
   - `fit_prefill_time()`
   - `fit_decode_time()`
4) 构建 decode 空间模型：
   - KV 模块：`c0=0, c1=kv_per_token_bytes*bs/1024^3`
   - 非 KV 模块：`c0=weight_size_gb, c1=0`
5) 合并多设备结果，输出单一 `device_registry.v3`。

## 输出
- `outputs/profiling/registry/device_registry.v3.json`
- 关键字段：
  - `devices.<dev>.modules.<module>.time_models.prefill.by_bs.<bs>`
  - `devices.<dev>.modules.<module>.time_models.decode.by_bs.<bs>`
  - `devices.<dev>.modules.<module>.memory_models.decode.by_bs.<bs>`

## 数据对齐标准
- 参数按 batch-size 分桶存储（运行时不再额外乘 bs）。
- `prefill.x=seq_len`，`decode.x=context_len`。
- 统一多项式字段：`form/c0/c1/c2/x/unit`。

## 调用方式
- `python tools/profiling/build_registry.py --inputs <file-or-dir...> --model-config <yaml> --out <registry-path>`
- 可选：
  - `--emit-all-results <dir>` 输出规范化 `all_results.v1` 映射文件
