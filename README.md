# grpc_heterogeneous_pipeline_nextgen

Heterogeneous pipeline-parallel inference workflow:
capture profiles -> build registry -> solve strategy -> run runtime/simulation.

## Repository layout

| Path | Role |
|------|------|
| `pp_nextgen/` | Core package (`profiling`, `scheduler`, `runtime`, `simulation`) |
| `tools/` | CLI scripts grouped by stage |
| `configs/` | Model / cluster / runtime YAML |
| `outputs/` | Generated profiling/scheduler/runtime artifacts |
| `design_docs/` | Architecture and design docs |

## Install

```bash
pip install -e .
pip install -e ".[capture]"   # profiling capture with torch
```

## Standard workflow

### 1) Capture

```powershell
python tools/profiling/capture_split_module_profiles.py --cuda-device 0 --device-id 3060 --output-dir ./my_profiles
```

### 2) Build registry

```powershell
python tools/profiling/build_registry.py --inputs ./my_profiles/example --model-config configs/model/llama2_7b.yaml --out outputs/profiling/registry/device_registry.v3.json --emit-all-results outputs/profiling/all_results
```

### 3) Solve strategy

```powershell
python tools/scheduler/solve_strategy.py --cluster configs/cluster/all_devices.yaml
```

Outputs:

- `outputs/scheduler/export/pipeline_strategy.json`
- `outputs/scheduler/export/tail_sweep_report.json`
- `outputs/scheduler/export/workers/*.strategy.json`

### 4) Runtime

```powershell
# master-3060 
python -m pp_nextgen.runtime.cli master --pipeline-strategy outputs/scheduler/export/coarse-grained-bs32-len32/pipeline_strategy.json --bind 0.0.0.0:50050 
```

```powershell
# worker-3060_0
python -m pp_nextgen.runtime.cli worker --worker-name 3060_0 --master 127.0.0.1:50050 --pipeline-strategy outputs/scheduler/export/coarse-grained-bs32-len32/pipeline_strategy.json --worker-strategy outputs/scheduler/export/coarse-grained-bs32-len32/3060_0.strategy.json --model-config configs/model/llama2_7b.yaml --metrics-out outputs/runtime/logs/coarse-grained-bs32-len32/3060_0.metrics.json --runtime-config configs/runtime/runtime.example.yaml

# worker-2080super_0
python -m pp_nextgen.runtime.cli worker --worker-name 2080super_0 --master 192.168.31.237:50050 --pipeline-strategy outputs/scheduler/export/coarse-grained-bs32-len32/pipeline_strategy.json --worker-strategy outputs/scheduler/export/coarse-grained-bs32-len32/2080super_0.strategy.json --model-config configs/model/llama2_7b.yaml --metrics-out outputs/runtime/logs/coarse-grained-bs32-len32/2080super_0.metrics.json --runtime-config configs/runtime/runtime.example.yaml

# worker-agx_0
python -m pp_nextgen.runtime.cli worker --worker-name agx_0 --master 192.168.31.237:50050 --pipeline-strategy outputs/scheduler/export/coarse-grained-bs32-len32/pipeline_strategy.json --worker-strategy outputs/scheduler/export/coarse-grained-bs32-len32/agx_0.strategy.json --model-config configs/model/llama2_7b.yaml --metrics-out outputs/runtime/logs/coarse-grained-bs32-len32/agx_0.metrics.json --runtime-config configs/runtime/runtime.example.yaml

# worker-tx2_0
python -m pp_nextgen.runtime.cli worker --worker-name tx2_0 --master 192.168.31.237:50050 --pipeline-strategy outputs/scheduler/export/coarse-grained-bs32-len32/pipeline_strategy.json --worker-strategy outputs/scheduler/export/coarse-grained-bs32-len32/tx2_0.strategy.json --model-config configs/model/llama2_7b.yaml --metrics-out outputs/runtime/logs/coarse-grained-bs32-len32/tx2_0.metrics.json --runtime-config configs/runtime/runtime.example.yaml
```

Submit one request:

```powershell
python tools/runtime/submit_task_to_master.py --master 192.168.31.237:50050 --batch-size 32 --context-len 32 --target-len 64 --req-id demo-1
```

### 5) Simulation

```powershell
python tools/simulation/run_pipeline_des_sim.py `
  --strategy outputs/scheduler/export/pipeline_strategy.json `
  --arrival-rate 6 `
  --duration-s 10 `
  --batch-size 1 `
  --context-len 128 `
  --target-len 256 `
  --out outputs/runtime/simulation/pipeline_des.sim_metrics.json `
  --master-latency-sidecar-out outputs/runtime/simulation/pipeline_des.master_latency.json
```

## Tool groups

- `tools/profiling/`: capture and registry build
- `tools/scheduler/`: strategy solve and benchmark
- `tools/runtime/`: proto generation and submit task
- `tools/simulation/`: DES and local runtime cluster simulation
- `tools/transport/`: transport A/B benchmark placeholder