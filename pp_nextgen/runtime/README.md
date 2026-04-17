# Runtime (Core)

## Roles and Inputs
- Master role:
  - reads pipeline topology and controls worker registration/completion
- Worker role:
  - reads `pipeline_strategy.v2` and `worker_strategy.v1`
  - reads model YAML for execution model construction

## Macro Workflow
1) Master starts, workers register.
2) Workers discover next-hop worker from master.
3) First worker receives requests and drives prefill/decode pipeline.
4) Each worker executes its module plan (`head/tail/single`) and forwards frame payload to next stage.
5) Last-stage output returns to worker0 tail or final completion path.
6) Master records completion and exports latency/metrics.

## Communication Model
- Data-plane: gRPC `SendFrame` between workers.
- Control-plane: registration, next-peer lookup, completion reports.
- Expected transfer size/time derives from strategy `stage_models.<phase>.comm_*`.

## Execution Model
- Compute:
  - `sleep_executor` mode uses expected model time for deterministic timing.
  - `shape_executor` mode executes shape-only graph.
- Timing and communication expectation are computed from strategy polynomial models.

## Output Files
- Per-worker metrics/journal JSON (runtime logs directory).
- Master latency sidecar compatible with `master_latency.v1`.

## Data Alignment Rules
- Runtime only accepts:
  - `pipeline_strategy.v2`
  - `worker_strategy.v1`
- Expected compute/comm models are read from `stage_models` only.

## How To Run
- Start master/worker via runtime CLI (`pp_nextgen/runtime/cli.py`) with:
  - pipeline strategy path
  - worker strategy directory
  - model config path
