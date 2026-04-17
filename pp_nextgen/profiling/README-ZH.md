# Profiling（核心）

## 功能定位
该模块在真实设备上测量分组件时延，并将原始 benchmark 结果转换成 scheduler 可直接使用的 `device_registry.v3` 标准模型。

## 模块结构
- `capture/`：运行 split-component 基准测试，输出 `<model>_all_results.json`
- `fit/`：把采样曲线拟合成多项式系数
- `build/`：组装多设备 registry JSON
- `legacy_ingest.py`：统一采集文件结构
- `constants.py`：模块名映射与 KV 模块定义

## 数据对齐规则
- 时间统一表达：`c0 + c1*x + c2*x^2`，按 batch-size 分桶保存。
- `prefill`：
  - `attn_qk`、`attn_av`：关于 `seq_len` 的二次项
  - 其余模块：关于 `seq_len` 的线性项（除显式常数路径）
- `decode`：
  - `attn_qk`、`attn_av`：关于 `context_len` 的线性项
  - 其余模块：使用 decode 平均值常数
- 系数已按 bs 分桶，不应在调度/运行时再乘 bs。

## 输入输出
- 输入：
  - 采集产物 `*_all_results.json`
  - 含 `kv_per_token_bytes` 的模型 YAML
- 输出：
  - `outputs/profiling/registry/device_registry.v3.json`
  - 可选 `*_all_results.v1.json` 转换文件

## 调用方式
1) 采集：
- `python pp_nextgen/profiling/capture/split_module_bench.py --device-id 3060 --output-dir my_profiles --components embed_tokens qkv_rope attn_qk attn_av attn_wo up_proj down_proj lm_head`
2) 构建 registry：
- `python tools/profiling/build_registry.py --inputs my_profiles --model-config configs/model/llama2_7b.yaml --out outputs/profiling/registry/device_registry.v3.json`
