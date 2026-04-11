# Scheduler Module

## Submodules

- `core/`: legacy-compatible DP solve logic and tail sweep driver
- `adapters/`: config/registry to solver input normalization
- `export/`: strategy serialization for global and per-worker outputs

## Responsibilities

- load normalized cost models
- solve split allocation under memory/network constraints
- export runtime-consumable strategy artifacts
