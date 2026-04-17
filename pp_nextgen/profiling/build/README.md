# Profiling Build

## Core Function
Builds `device_registry.v3` by collecting multi-device capture outputs, fitting per-module/per-phase coefficients, and attaching decode memory models.

## Inputs
- Legacy capture files: `*_all_results.json` (single file or directory)
- Model YAML:
  - `memory_contract.kv_per_token_bytes` is required

## Internal Pipeline
1) Load and normalize capture files (`legacy_ingest.py`).
2) Iterate each `(batch_size, module)` sample set.
3) Fit time models:
   - `fit_prefill_time()`
   - `fit_decode_time()`
4) Build decode memory model:
   - KV modules: `c0=0, c1=kv_per_token_bytes*bs/1024^3`
   - non-KV modules: `c0=weight_size_gb, c1=0`
5) Merge all devices into one `device_registry.v3`.

## Output
- `outputs/profiling/registry/device_registry.v3.json`
- schema highlights:
  - `devices.<dev>.modules.<module>.time_models.prefill.by_bs.<bs>`
  - `devices.<dev>.modules.<module>.time_models.decode.by_bs.<bs>`
  - `devices.<dev>.modules.<module>.memory_models.decode.by_bs.<bs>`

## Data Alignment Standard
- Per-batch-size bucketed coefficients (no runtime bs multiplication).
- `prefill.x=seq_len`, `decode.x=context_len`.
- Time/comm use polynomial fields (`form/c0/c1/c2/x/unit`).

## CLI
- `python tools/profiling/build_registry.py --inputs <file-or-dir...> --model-config <yaml> --out <registry-path>`
- optional:
  - `--emit-all-results <dir>` to emit standardized all_results.v1 mirrors
