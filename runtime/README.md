# Runtime Module

## Submodules

- `proto/`: transport protocol definitions
- `master/`: control-plane service
- `worker/`: execution-plane service
- `common/`: queueing, frame model, tracing helpers
- `executors/`: sleep and shape execution backends

## Responsibilities

- execute per-worker strategy
- preserve request lifecycle consistency
- support compute/send overlap with bounded queues
