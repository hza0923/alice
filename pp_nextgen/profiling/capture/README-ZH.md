# Profiling Capture

## 测量目标
`split_module_bench.py` 在指定设备上测量 Llama 风格解码器各模块的计算时延。

### 模型定义
默认模型配置：
- `llama2-7b`, `vocab=32000`, `hidden=4096`, `num_heads=32`, `num_kv_heads=32`, `ffn_dim=11008`

### 设备定义
- 通过以下方式确定执行设备：
  - `--cpu` 强制 CPU
  - `--cuda-device` 指定 GPU
- 输出文件中的 `device` 来自 `--device-id`，必须与 cluster/scheduler 中设备命名对齐。

### 模块与 Shape（按 p/d 区分）
- `embed_tokens`: `[bs, seq_or_1, hidden]`
- `qkv_rope`: 等价于 q/k/v 投影 + RoPE 的输出
- `attn_qk`:
  - prefill: `[bs, n_head, seq, seq]`
  - decode: `[bs, n_head, 1, context_len]`
- `attn_av`:
  - prefill: `[bs, n_head, seq, head_dim]`
  - decode: `[bs, n_head, 1, head_dim]`
- `attn_wo`: `[bs, seq_or_1, hidden]`
- `up_proj`: `[bs, seq_or_1, ffn_dim]`
- `down_proj`: `[bs, seq_or_1, hidden]`
- `lm_head`: `[bs, seq_or_1, vocab]`

## Prefill/Decode 采样规则
- Prefill：遍历 `seq_len = 1..p_max_len`，步长 `step`。
- Decode：遍历上下文增长 `1..d_max_len`。
- KV 模块（`attn_qk`/`attn_av`）记录每个 `context_len` 的 decode 时延。
- 其他模块记录 decode 平均值（下游拟合为常数）。

## 输出文件
每个模型输出：
- `<model>_all_results.json`
- 核心字段：
  - `model`, `device`, `device_memory_gb`
  - `test_configurations[]`
  - 每项含 `batch_size`, `p_max_len`, `d_max_len`, `step`
  - `components.<module>.prefill_times`
  - `components.<module>.decode_times`
  - `weight_size_gb`, `kvcache_size_gb`, `status`, `error_info`

## CLI 参数
- `--device-id`：用于与 scheduler 对齐的设备标识
- `--cuda-device`：GPU 索引
- `--cpu`：强制 CPU
- `--n-repeats`：计时重复次数
- `--warmup`：预热次数
- `--output-dir`：输出目录
- `--components`：指定模块子集
- `--quick`：快速烟测配置
