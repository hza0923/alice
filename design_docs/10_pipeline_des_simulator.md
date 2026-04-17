# Pipeline DES Simulator

This document describes the offline discrete-event simulator for heterogeneous
pipeline-parallel inference in this repository.

## 1. Goals and scope

The simulator is designed to answer:

- How request queueing behaves under continuous arrivals.
- How per-stage compute serialization and per-link transfer serialization affect
  TTFT and E2E latency.
- How throughput changes as offered load increases.

It intentionally does **not** replace runtime gRPC execution. It is a
virtual-time approximation that reuses strategy timing/communication models.

## 2. Where the implementation lives

- `pp_nextgen/simulation/request_model.py`
  - `SimRequest` request data model.
  - `generate_poisson_requests(...)` arrival generator.
- `pp_nextgen/simulation/batching.py`
  - `FCFSContiguousBatchScheduler` (contiguous FCFS packing).
  - `PackedBatch` packed batch unit.
- `pp_nextgen/simulation/des_engine.py`
  - `DESConfig` simulator config.
  - `PipelineDESSimulator` event loop and resource scheduling.
- `pp_nextgen/simulation/metrics.py`
  - Per-request and aggregate metrics.
  - JSON export schema: `pipeline_des_metrics.v1`.
- `tools/simulation/run_pipeline_des_sim.py`
  - CLI entrypoint.

## 3. Inputs and outputs

## 3.1 Inputs

1. `pipeline_strategy.v2` JSON (for stage order and performance models), e.g.:
   - `outputs/scheduler/export/pipeline_strategy.json`
2. Traffic parameters:
   - Poisson rate (`--arrival-rate`)
   - request shape (`--batch-size`, `--context-len`, `--target-len`)
   - stop condition (`--duration-s` and/or `--num-requests`)
3. Batching + network parameters:
   - `--max-batch-size`
   - `--packing-window-ms`
   - `--default-link-bandwidth-gbps`
   - optional per-link overrides.

## 3.2 Outputs

Default output path:

- `outputs/runtime/simulation/pipeline_des.sim_metrics.json`

Optional runtime-compatible sidecar:

- `--master-latency-sidecar-out outputs/runtime/simulation/pipeline_des.master_latency.json`
- schema aligns with `master_latency.v1` style (`start`, `end`, `latency_s`).

## 3.3 Required strategy fields for runtime-aligned DES

The simulator now reads phase and branch specific models from strategy:

- `stage_params.decode`
- `stage_params.prefill`

For non-designated workers, use `single_model`.

For designated `worker0`, use:

- `head_model`: first-node work on ingress path
- `tail_model`: last-node work on ring-return path

Both `decode` and `prefill` must be distinguishable in strategy models.

## 4. Module collaboration and flow

1. Request generation:
   - `generate_poisson_requests(...)` creates a request timeline.
2. Batch admission:
   - arrivals are enqueued into `FCFSContiguousBatchScheduler`.
   - scheduler pops the next contiguous pack when first stage can accept work.
3. Event simulation:
   - simulator pushes and pops events from a min-heap.
   - each event advances virtual clock `now`.
4. Resource constraints:
   - each stage has `stage_available_at` (compute serialization).
   - each link has `link_available_at` (transfer serialization).
   - compute and comm are independent resources and can overlap in time.
5. Metrics:
   - collector stamps per-request lifecycle timestamps and computes aggregate KPIs.

## 5. Event model

Implemented event types:

- `request_arrival`
- `stage_compute_done`
- `link_transfer_done`
- `worker0_tail_done`

Internal request lifecycle:

1. `request_arrival` -> queue.
2. packed batch enters stage-0 prefill when stage-0 compute resource is free.
3. compute completion at stage-i:
   - if not last stage: schedule link transfer i->i+1
   - if last stage: schedule `worker0` tail compute (when worker0 has head/tail split)
4. `worker0_tail_done`:
   - if phase is prefill: mark first token time (TTFT anchor)
   - if decode unfinished: schedule next decode step at worker0 head
   - else: mark request finished.

## 6. Mathematical model

Let:

- `s` be stage index.
- `k` be decode step index (`k >= 1`).
- `B` be packed batch size.
- `L` be context length.
- `bw_(s,s+1)` be bandwidth (bytes/s) on link s->s+1.

For decode phase, effective sequence length is:

- `L_k = L + k`

For prefill phase, effective sequence length:

- `L_prefill = L`

Compute time per stage (from strategy model):

- `T_comp(s, phase, L_eff, B) = expected_compute_ms(...) / 1000`

Communication size per stage (except last stage):

- `C_bytes(s, phase, L_eff, B) = expected_comm_bytes(...)`

Communication time:

- `T_comm(s, phase, L_eff, B) = C_bytes / bw_(s,s+1)`

Resource serialization:

- compute start at stage `s`:
  - `t_start_comp = max(t_ready, stage_available_at[s])`
- compute finish:
  - `t_end_comp = t_start_comp + T_comp`
- then update:
  - `stage_available_at[s] = t_end_comp`

When designated `worker0` is split into head/tail, compute resources are:

- `stage_available_at[worker0|head]`
- `stage_available_at[worker0|tail]`

This mirrors runtime behavior where worker0 has separate head/tail queues.

For communication:

- `t_start_comm = max(t_ready, link_available_at[s,s+1])`
- `t_end_comm = t_start_comm + T_comm`
- `link_available_at[s,s+1] = t_end_comm`

Because compute and communication use separate clocks, they can overlap, but
same-type operations are strictly serialized by `available_at`.

## 7. Metric definitions

Per request:

- `arrival_ts`: generated arrival timestamp.
- `service_start_ts`: first time stage-0 starts prefill for that request.
- `first_token_ts`: prefill completion time on `worker0 tail` (final token generation point).
- `finish_ts`: final decode completion on `worker0 tail` (or prefill completion if no decode).
- `ttft_s = first_token_ts - arrival_ts`
- `e2e_latency_s = finish_ts - arrival_ts`
- `queue_wait_s = service_start_ts - arrival_ts`

Aggregate:

- `throughput_req_s = completed_count / elapsed_s`
- `throughput_token_s = sum(generated_tokens) / elapsed_s`
- average and p95 for TTFT/E2E/queue wait.
- stage/link utilization:
  - `util = busy_time / elapsed_s`

## 8. Relationship with other tools

- `tools/simulation/run_pipeline_des_sim.py`:
  - virtual-time DES (queueing-aware).
- `tools/simulation/run_cluster_sim.py`:
  - process-level runtime simulation over asyncio + gRPC.
- `tools/scheduler/benchmark_strategy.py`:
  - analytic grid benchmarking (no request timeline queueing).

Use DES when you need queueing/latency distribution under load.

## 9. Quick start (current repo layout)

From repository root:

```powershell
python tools/simulation/run_pipeline_des_sim.py `
  --strategy outputs/scheduler/export/pipeline_strategy.json `
  --arrival-rate 6 `
  --duration-s 10 `
  --batch-size 1 `
  --context-len 128 `
  --target-len 256 `
  --max-batch-size 32 `
  --packing-window-ms 0 `
  --default-link-bandwidth-gbps 12.5 `
  --out outputs/runtime/simulation/pipeline_des.sim_metrics.json `
  --master-latency-sidecar-out outputs/runtime/simulation/pipeline_des.master_latency.json
```

Expected artifacts:

- `outputs/runtime/simulation/pipeline_des.sim_metrics.json`
- `outputs/runtime/simulation/pipeline_des.master_latency.json` (if enabled)

## 10. Example: custom per-link bandwidth

```powershell
python tools/simulation/run_pipeline_des_sim.py `
  --strategy outputs/scheduler/export/pipeline_strategy.json `
  --arrival-rate 8 `
  --duration-s 15 `
  --batch-size 1 `
  --context-len 64 `
  --target-len 128 `
  --link-bandwidth "3060_0->2080super_0:24,2080super_0->3060_0:16" `
  --out outputs/runtime/simulation/pipeline_des.custom_bw.sim_metrics.json
```

## 11. Validation checklist

1. Single-request check:
   - run with `--num-requests 1` and confirm finite TTFT/E2E.
2. Load trend check:
   - compare low/high `--arrival-rate` runs.
   - verify high load increases queue wait and latency.
3. Optional consistency:
   - compare low-load trend with `tools/scheduler/benchmark_strategy.py` TBT scale.
