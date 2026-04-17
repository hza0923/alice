# Scheduler Interfaces

## Data Sources

- Time models come from `outputs/profiling/registry/device_registry.v3.json` by `bs` bucket.
- Communication models are analytic tensor-shape polynomials emitted by scheduler.
- Decode memory model is exported as:
  - `memory_gb = c0 + c1 * seq_len * batch_size`
  - `c0`: stage weight footprint (GB)
  - `c1`: stage KV per-token footprint (GB/token)

## Strategy Contract

`pipeline_strategy.json` (`pipeline_strategy.v2`) exposes:

- `schedule_input` (`bs`, `target_seq_len`, `designated_device`, `designated_tail_n`)
- `pipeline_stages[]` in execution order
- stage routing fields:
  - `worker_name`, `next_worker`, `is_last_worker`

Each stage includes two phase blocks (`decode`, `prefill`) under:

- `stage_models.<phase>.time_ms.<branch>`
- `stage_models.<phase>.comm_bytes.<branch>`
- `stage_models.<phase>.comm_time_ms.<branch>`

Decode-only memory lives at:

- `stage_models.decode.memory_gb.<branch>`

where `<branch>` is:

- regular workers: `single`
- designated worker0 split: `head` and `tail`

Model object schema:

```json
{
  "form": "constant|linear|quadratic",
  "c0": 0.0,
  "c1": 0.0,
  "c2": 0.0,
  "x": "seq_len|context_len",
  "unit": "ms|bytes|GB",
  "expr": "c0 + c1 * x + c2 * x^2"
}
```

Decode memory model uses:

```json
{
  "form": "linear",
  "c0": 0.0,
  "c1": 0.0,
  "c2": 0.0,
  "x": "seq_len",
  "batch_term": "batch_size",
  "unit": "GB",
  "expr": "c0 + c1 * seq_len * batch_size"
}
```

## Legacy Compatibility

- `stage_params.decode/prefill` remains exported for old runtime paths.
- Flat fallback fields (`base_time`, `increase_time`, `comp_time_ms`, `base_size`, `inc_size`) remain.
- Runtime should prefer `stage_models` and fall back to `stage_params`/flat fields.
