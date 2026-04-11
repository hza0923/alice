# Tools

## Planned CLI Entrypoints

- `build_registry.py`
  - input: `all_results` files
  - output: normalized `device_registry`
- `solve_strategy.py`
  - input: model config + cluster config + registry
  - output: global strategy + per-worker strategies
- `run_cluster_sim.py`
  - starts local/synthetic runtime for functional validation
- `run_transport_benchmark.py`
  - transport A/B benchmark automation

## Shared CLI Rules

- all commands support `--config` and `--out-dir`
- all commands emit JSON summary report
- non-zero exit code on validation failure
