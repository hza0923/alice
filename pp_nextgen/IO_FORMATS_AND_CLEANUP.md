# Active I/O Formats and Cleanup Notes

## Canonical Formats (Current)

## 1) Profiling
- Capture/build input:
  - `*_all_results.json` produced by `pp_nextgen/profiling/capture/split_module_bench.py`
- Registry output:
  - `device_registry.v3`

## 2) Scheduler
- Inputs:
  - `device_registry.v3`
  - model/cluster YAML
- Outputs:
  - `pipeline_strategy.v2`
  - `worker_strategy.v1`

## 3) Runtime / Simulator
- Runtime accepts only:
  - `pipeline_strategy.v2`
  - `worker_strategy.v1`
- Simulator accepts only:
  - `pipeline_strategy.v2`

## 4) Formula Contract
- time: `c0 + c1*x + c2*x^2`
- communication: `c0 + c1*x + c2*x^2`
- decode memory: `c0 + c1*seq_len*batch_size`

## Cleanup Applied
- `pp_nextgen/scheduler/adapters/registry_adapter.py`
  - removed old registry schema fallback paths
  - strict `device_registry.v3` enforcement
- `pp_nextgen/runtime/strategy.py`
  - removed legacy `stage_params` fallback compute/comm/memory paths
  - strict schema checks for `pipeline_strategy.v2` and `worker_strategy.v1`
  - stage model reads are now mandatory
- `pp_nextgen/schemas/validate.py`
  - strengthened `device_registry.v3` minimal validation

## Remaining Legacy-Labeled Areas
- `profiling/legacy_ingest.py` is still used as normalization entry for capture JSON naming conventions.
- Top-level `comm_time_ms` / `comm_bytes_to_next` in `pipeline_strategy.v2` is still exported for compatibility with existing analysis scripts, while runtime core already reads `stage_models`.
