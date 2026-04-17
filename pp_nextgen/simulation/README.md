# Simulation (Core)

## Inputs
- `pipeline_strategy.v2` (required)
- request stream configuration (arrival rate, duration/count, batch/context/target length)
- optional link-bandwidth overrides

## Macro Flow
1) Load strategy and initialize per-stage resources.
2) Build link bandwidth map.
3) Generate request arrivals.
4) Run one of two simulators:
   - DES loop:
   - request arrival
   - stage compute completion
   - inter-stage transfer completion
   - worker0 tail completion
   - worker0 queue loop:
   - worker0 picks next item by priority (`prefill` first, then FCFS)
   - selected item is propagated through all stages/links immediately
   - decode next-step item is re-enqueued at worker0 after current step completes
5) Collect metrics and optional traces.

## Pipeline Modeling
- Compute duration per stage:
  - from `expected_compute_ms(stage, phase, context_len, batch_size, branch)`
- Transfer bytes per edge:
  - from `expected_comm_bytes(stage, phase, context_len, batch_size, branch)`
- Branching:
  - worker0 may split to `head/tail` branch resources
- Decode context growth:
  - `context_len + decode_step`

## Outputs
- DES metrics JSON
- optional:
  - per-request trace JSON
  - per-stage trace JSON
  - master-latency-compatible sidecar

## Data Alignment Rules
- Same strategy formulas as runtime are used for expected compute/communication.
- Strategy schema is treated as strict (`pipeline_strategy.v2`).

## CLI
- `python tools/simulation/run_pipeline_des_sim.py --strategy outputs/scheduler/export/pipeline_strategy.json --arrival-rate 5 --duration-s 10 --batch-size 1 --context-len 32 --target-len 64 --out outputs/runtime/simulation/metrics/run.json`
- `python tools/simulation/run_pipeline_queue_sim.py --strategy outputs/scheduler/export/pipeline_strategy.json --arrival-rate 5 --duration-s 10 --batch-size 1 --context-len 32 --target-len 64 --out outputs/runtime/simulation/metrics/run_queue.json`
