# Profiling (Core)

## Purpose
This module measures per-component latency on real devices and converts raw benchmarks into normalized `device_registry.v3` models used by scheduler.

## Module Layout
- `capture/`: run split-component benchmarks and emit `<model>_all_results.json`
- `fit/`: convert sampled curves to polynomial coefficients
- `build/`: assemble multi-device registry JSON
- `legacy_ingest.py`: normalize capture JSON into builder-friendly structure
- `constants.py`: canonical module mapping and KV-module declarations

## Alignment Rules
- Time model form: `c0 + c1*x + c2*x^2`, bucketed by batch-size.
- `prefill`:
  - `attn_qk`, `attn_av`: quadratic in `seq_len`
  - others: linear in `seq_len` (except any explicit constant sampled path)
- `decode`:
  - `attn_qk`, `attn_av`: linear in `context_len`
  - others: constant from decode average
- No extra `*batch_size` at runtime for time/comm model coefficients because coefficients are already bucket-specific.

## Input / Output
- Input:
  - capture output(s): `*_all_results.json`
  - model YAML for memory contract (`kv_per_token_bytes`)
- Output:
  - `outputs/profiling/registry/device_registry.v3.json`
  - optional converted `*_all_results.v1.json`

## How To Run
1) Capture:
- `python pp_nextgen/profiling/capture/split_module_bench.py --device-id 3060 --output-dir my_profiles --components embed_tokens qkv_rope attn_qk attn_av attn_wo up_proj down_proj lm_head`
2) Build registry:
- `python tools/profiling/build_registry.py --inputs my_profiles --model-config configs/model/llama2_7b.yaml --out outputs/profiling/registry/device_registry.v3.json`
