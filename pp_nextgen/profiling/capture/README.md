# Profiling Capture

## What It Measures
`split_module_bench.py` measures module-level compute latency for a Llama-style decoder stack on a specific device.

### Model Definition
Default model tuple:
- `llama2-7b`, `vocab=32000`, `hidden=4096`, `num_heads=32`, `num_kv_heads=32`, `ffn_dim=11008`

### Device
- Runtime target is selected by:
  - `--cpu` (force CPU), or
  - CUDA device via `--cuda-device`
- Device identity written to output as `device` comes from `--device-id` and must match cluster/scheduler naming.

### Modules and Shape Contracts
- `embed_tokens`: output `[bs, seq_or_1, hidden]`
- `qkv_rope`: output equivalent to q/k/v projection tensors with rotary embedding
- `attn_qk`:
  - prefill: `[bs, n_head, seq, seq]`
  - decode: `[bs, n_head, 1, context_len]`
- `attn_av`:
  - prefill: `[bs, n_head, seq, head_dim]`
  - decode: `[bs, n_head, 1, head_dim]`
- `attn_wo`: output `[bs, seq_or_1, hidden]`
- `up_proj`: output `[bs, seq_or_1, ffn_dim]`
- `down_proj`: output `[bs, seq_or_1, hidden]`
- `lm_head`: output `[bs, seq_or_1, vocab]`

## Prefill / Decode Measurement Rules
- Prefill loops `seq_len = 1..p_max_len` with `step`.
- Decode loops context growth `1..d_max_len`.
- KV modules (`attn_qk`, `attn_av`) record decode timings per `context_len`.
- Other modules record decode average (constant form downstream).

## Output File
Per model:
- `<model>_all_results.json`
- key fields:
  - `model`, `device`, `device_memory_gb`
  - `test_configurations[]`
  - each config has `batch_size`, `p_max_len`, `d_max_len`, `step`
  - `components.<module>.prefill_times`
  - `components.<module>.decode_times`
  - `weight_size_gb`, `kvcache_size_gb`, `status`, `error_info`

## CLI
- `--device-id`: required semantic identity for scheduler alignment
- `--cuda-device`: CUDA index
- `--cpu`: force CPU mode
- `--n-repeats`: timing repeats
- `--warmup`: warmup repeats
- `--output-dir`: where `<model>_all_results.json` is written
- `--components`: optional subset
- `--quick`: tiny test grid for smoke runs
