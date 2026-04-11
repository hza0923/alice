# grpc_heterogeneous_pipeline_nextgen

Clean-slate redesign for fine-grained pipeline-parallel inference across heterogeneous multi-node GPUs: profiling, registry build, scheduling, and a gRPC runtime.

## Repository layout

| Path | Role |
|------|------|
| `docs/` | Architecture and design specs |
| `configs/` | Model, cluster, and runtime YAML |
| `profiling/` | Capture benchmarks, fitted registry (`artifacts/`) |
| `scheduler/` | Solver, exported `pipeline_strategy.json` and per-worker slices |
| `runtime/` | Extra runtime notes; code lives under `pp_nextgen/runtime/` |
| `tools/` | CLI entry scripts |
| `pp_nextgen/` | Importable Python package (runtime, profiling, scheduler) |

## Install

From the repository root:

```bash
pip install -e .
```

Profiling benchmarks need PyTorch (not installed by default):

```bash
pip install -e ".[capture]"
```

Use a CUDA-enabled PyTorch build on each GPU machine. The runtime also needs `grpcio` (already in the base dependencies).

## Naming rules (read this before running anything)

These strings must stay consistent end-to-end:

1. **Profile JSON `device` field** (written by capture) becomes **`device_type`** inside `device_registry.v3.json`. It must match **`device_group`** keys in your cluster YAML (examples: `3060`, `2080super`, `tx2`, `agx`). Use the same spelling on every command line (`--device-id`) and in the cluster file.
2. **`designated_device`** in the cluster YAML is a **device type string**, not a worker name. It selects which device family owns the tail modules (see `docs/02_data_contracts.md`).
3. **Worker names** in `pipeline_strategy.json` and in filenames `scheduler/export/workers/<worker_name>.strategy.json` follow **`{device_type}_{instance}`** with a zero-based instance id (`3060_0`, `3060_1`, `tx2_0`, …). The **`--worker-name`** argument to the runtime worker must match **`worker_name`** in the strategy exactly.
4. **Model id** in profile JSON (`model`, e.g. `llama2-7b`) should match what you pass through `build_registry` and the solver; keep filenames optional but conventional: `<model>_all_results.json`.

## Quick start (single machine, smoke)

All paths below are relative to the repo root. PowerShell uses backtick continuation; Bash uses `\`.

**1. Capture one legacy profile (CPU smoke; add CUDA on real hardware):**

```powershell
python tools/capture_split_module_profiles.py --cpu --quick --n-repeats 1 --warmup 0 --device-id smoke_cpu --output-dir profiling/artifacts/tmp
```

**2. Build registry** (for a real run, pass one JSON per physical device type, each with a distinct `--device-id` from step 1):

```powershell
python tools/build_registry.py `
  --inputs profiling/artifacts/all_results/llama2-7b_3060_all_results.v1.json `
  --model-config configs/model/llama2_7b.yaml `
  --out profiling/artifacts/registry/device_registry.v3.json
```

**3. Solve** (edit `configs/cluster/*.yaml` so `solve.registry_path` and `device_group` match your registry):

```powershell
python tools/solve_strategy.py --cluster configs/cluster/single_3060.example.yaml
```

**4. Runtime (shape/sleep backend)** — start **master**, then each **worker** (replace host addresses and worker names with your export):

```powershell
python -m pp_nextgen.runtime.cli master --bind 0.0.0.0:50050 --pipeline-strategy scheduler/export/pipeline_strategy.json
```

```powershell
python -m pp_nextgen.runtime.cli worker --worker-name 3060_0 --bind 0.0.0.0:50051 --public-address 192.168.1.10:50051 --master 192.168.1.1:50050 --pipeline-strategy scheduler/export/pipeline_strategy.json --worker-strategy scheduler/export/workers/3060_0.strategy.json --model-config configs/model/llama2_7b.yaml
```

`--public-address` defaults to detected LAN IP plus the listen port; set it explicitly if the master cannot reach the auto-detected address.

## Real multi-device workflow

### A. On each machine (install + profile)

1. Clone this repo and `pip install -e ".[capture]"` with the correct Torch wheel for that machine (CUDA version / JetPack, etc.).
2. On machine *M*, run capture once per GPU (or once per device class you represent), choosing `--device-id` equal to the key you will use in `device_group` (same string for all GPUs of that type, unless you intentionally split types):

```powershell
python tools/capture_split_module_profiles.py --cuda-device 0 --device-id 3060 --output-dir ./my_profiles
```

3. Repeat on every other box (`--device-id tx2`, `agx`, …). Full sweeps are the default grid; use `--quick` only for sanity checks.

Outputs look like `./my_profiles/llama2-7b_all_results.json` (one combined file per model per run).

### B. Collect profiles on the node that builds the registry

Copy all `*_all_results.json` files onto one host (scp, shared NFS, etc.). Paths are arbitrary; `build_registry` only needs the file list.

### C. Build `device_registry.v3.json`

```powershell
python tools/build_registry.py `
  --inputs ./profiles/llama2-7b_3060.json ./profiles/llama2-7b_tx2.json ./profiles/llama2-7b_agx.json `
  --model-config configs/model/llama2_7b.yaml `
  --out profiling/artifacts/registry/device_registry.v3.json
```

Optional: `--emit-all-results profiling/artifacts/all_results` to also write normalized `all_results.v1` copies.

### D. Master generates strategy

1. Edit a cluster YAML (start from `configs/cluster/all_devices.yaml` or `single_3060.example.yaml`):

   - `device_group`: counts of workers per **device type string** (must exist in the registry).
   - `designated_device`: which device type carries the designated tail.
   - `solve.registry_path` / `solve.model_config` / `solve.out_dir` (defaults under `scheduler/export`).

2. Run:

```powershell
python tools/solve_strategy.py --cluster configs/cluster/all_devices.yaml
```

Artifacts:

- `scheduler/export/pipeline_strategy.json`
- `scheduler/export/tail_sweep_report.json`
- `scheduler/export/workers/*.strategy.json`

### E. Deploy exports to every runtime host

Replicate the **same relative layout** (recommended) or any layout you reference in CLI flags:

- Every host needs **`scheduler/export/pipeline_strategy.json`** (same file everywhere).
- Each worker host needs **its own** `scheduler/export/workers/<worker_name>.strategy.json` (only the files for workers that run on that machine).

Example tree on a worker that runs `3060_0`:

```text
scheduler/export/pipeline_strategy.json
scheduler/export/workers/3060_0.strategy.json
configs/model/llama2_7b.yaml
configs/runtime/runtime.example.yaml   # optional override
```

### F. Start runtime processes

Use the same `pipeline_strategy.json` path on master and workers. Each worker passes **`--worker-name`** matching both the stage in `pipeline_strategy.json` and the basename of its strategy file.

Full one-liner templates (Bash; adjust paths and IPs):

```bash
python -m pp_nextgen.runtime.cli master --bind 0.0.0.0:50050 --pipeline-strategy scheduler/export/pipeline_strategy.json
```

```bash
python -m pp_nextgen.runtime.cli worker --worker-name 3060_0 --bind 0.0.0.0:50051 --master 192.168.1.1:50050 --pipeline-strategy scheduler/export/pipeline_strategy.json --worker-strategy scheduler/export/workers/3060_0.strategy.json --model-config configs/model/llama2_7b.yaml --metrics-out runtime/logs/3060_0.metrics.json
```

Start the **first** `pipeline_stages` worker before or after others as your process manager allows; all workers must register before the master can drive a full pipeline.

## Feasibility note

KV memory in the scheduler scales with `memory_contract.kv_per_token_bytes * batch_size * seq_len` for attention modules. If solving fails, lower `target_seq_len` / `batch_size` in cluster `solve:` or verify `kv_per_token_bytes` in `configs/model/llama2_7b.yaml`.

## Legacy reference project

The older tree `grpc_heterogeneous_pipeline/` (sibling checkout) remains a reference; this repo vendors the split-module benchmark as `pp_nextgen/profiling/capture/split_module_bench.py` plus `tools/capture_split_module_profiles.py`.
