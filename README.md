# grpc_heterogeneous_pipeline_nextgen

This repository is a clean-slate redesign for fine-grained pipeline-parallel
inference across heterogeneous multi-node GPUs.

## Goals

- Separate profiling, fitting, registry build, scheduling, and runtime execution.
- Keep scheduler core solving semantics compatible with legacy implementation.
- Support `designated_tail_n` sweep (`1..7`) for designated device tail modules.
- Preserve gRPC as the primary runtime transport while enabling transport A/B.
- Add a no-weight real execution mode with req-id KV cache lifecycle.

## Repository Layout

- `docs/`: architecture and implementation specifications
- `configs/`: model/cluster/runtime configuration templates
- `profiling/`: profile capture, fitting, registry generation
- `scheduler/`: solver wrappers, adapters, strategy export
- `runtime/`: communication protocol and node runtime architecture
- `tools/`: command-line entry scripts

## Scope in This Iteration

This iteration delivers docs and Markdown interface skeletons to guide
implementation. Legacy code under `grpc_heterogeneous_pipeline/3060` is used as
reference only and is not modified here.

## Quickstart (runnable)

```bash
cd grpc_heterogeneous_pipeline_nextgen
pip install -e .

# 1) Build device_registry.v3 from legacy profile JSON (one file per machine)
python tools/build_registry.py ^
  --inputs path/to/llama2-7b_all_results.json ^
  --model-config configs/model/llama2_7b.yaml ^
  --out profiling/artifacts/registry/device_registry.v3.json

# 2) Solve strategy (edit configs/cluster/single_3060.example.yaml paths first)
python tools/solve_strategy.py --cluster configs/cluster/single_3060.example.yaml
```

Outputs:

- `profiling/artifacts/registry/device_registry.v3.json`
- `scheduler/export/pipeline_strategy.json`
- `scheduler/export/tail_sweep_report.json`
- `scheduler/export/workers/*.strategy.json`

**Feasibility note:** KV memory scales with `memory_contract.kv_per_token_bytes * batch_size * seq_len`
per `attn_qk` / `attn_av` module. Long `target_seq_len` or an oversized `kv_per_token_bytes`
can make the DP infeasible on small `memory_gb`. Reduce `target_seq_len` or fix the KV
bytes model to match your deployment.
