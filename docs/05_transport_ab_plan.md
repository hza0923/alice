# Transport A/B Plan

## Goal

Evaluate whether any transport outperforms gRPC for this cross-node
heterogeneous PP runtime without violating correctness and operability.

## Baseline and Candidates

- Baseline: gRPC (`unary` and optional `streaming` mode)
- Candidate A: ZeroMQ (custom framing)
- Candidate B: UCX host-buffer mode (if environment supports)

## Constraints

- Heterogeneous GPUs and mixed CUDA compatibility.
- Avoid assumptions about GPU direct communication availability.
- Keep request semantics identical across all transports.

## Test Matrix

Dimensions:

- payload mode:
  - metadata-only
  - small payload (`<= 64KB`)
  - medium payload (`64KB~1MB`)
- ring size:
  - 2 / 4 / 8 workers
- request pattern:
  - single long request
  - mixed short/long requests
  - burst arrivals

Metrics:

- p50/p95/p99 end-to-end latency
- per-hop send latency
- throughput (req/s, token-step/s)
- peak RSS and queue growth
- error rate and recovery time

## Pass/Fail Criteria

A candidate is considered viable only if:

- correctness:
  - no request loss/duplication under fault injection
  - ordering guarantees satisfy configured mode
- performance:
  - p99 latency not worse than gRPC by more than 10%
  - throughput at least 1.1x of gRPC, or equivalent with lower variance
- operability:
  - bounded memory behavior under downstream stall
  - clear timeout/retry hooks and observability support

## Fault Injection Scenarios

- downstream worker artificial stall (5s / 15s)
- intermittent network timeout
- sender retry storms
- late duplicate frame delivery

## Deliverables

- benchmark runner:
  - `tools/run_transport_benchmark.py`
- result summary:
  - `profiling/artifacts/benchmarks/transport_ab_report.md`
- recommendation:
  - default transport decision + rollback condition
