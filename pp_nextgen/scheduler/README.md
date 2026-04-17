# Scheduler (Core)

## Core Inputs
- Model config (`build_scheduler_model_cfg`)
- Registry (`device_registry.v3`)
- Cluster config (`device_group`, `designated_device`, `network_bandwidth_mbps`)
- Solve options (`prefer_bs`, `batch_size`, `target_seq_len`, `tail_candidates`)

## What Scheduler Does
1) Select the best per-module `bs` bucket from registry.
2) Build stage compute models (decode/prefill polynomial sums).
3) Build communication models from tensor-shape contracts (decode/prefill).
4) Apply memory feasibility with decode memory model:
   - `memory_gb = c0 + c1 * seq_len * batch_size`
5) Run DP allocation + designated tail sweep (`designated_tail_n` candidates).
6) Export:
   - global `pipeline_strategy.v2`
   - per-worker `worker_strategy.v1`

## Core Algorithm
- Objective: minimize TBT bottleneck with compute/communication coupling.
- State: layer split position + remaining device-instance availability.
- Transition: assign next layer segment to a device instance if memory-feasible.
- Cost: `stage_time = max(stage_compute_ms, stage_comm_ms)`.

## Output Contract
- `pipeline_strategy.json`:
  - `stage_models.decode/prefill.time_ms`
  - `stage_models.decode/prefill.comm_bytes`
  - `stage_models.decode/prefill.comm_time_ms`
  - `stage_models.decode.memory_gb` (decode-only)
- `workers/*.strategy.json`:
  - `head_ordered_modules`
  - `tail_ordered_modules`

## Data Alignment Rules
- Time/comm coefficients are all polynomial and phase-specific.
- Time coefficients come from registry by `bs` bucket.
- Communication coefficients are deterministic from shape contracts.
- Decode memory source is explicit and traceable in stage model.

## CLI
- `python tools/scheduler/solve_strategy.py --cluster configs/cluster/all_devices.yaml`
