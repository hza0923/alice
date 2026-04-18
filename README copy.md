
### 4) Runtime
==============================================================================
coarse-grained-bs32-len32
```powershell
# master-3060 
python -m pp_nextgen.runtime.cli master --pipeline-strategy outputs/scheduler/export/coarse-grained-bs32-len32/pipeline_strategy.json --bind 0.0.0.0:50050 --metrics-out outputs/runtime/logs/coarse-grained-bs32-len32/master.metrics.json
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
==================================================================================
fine-grained-bs32-len32
```powershell
# master-3060 
python -m pp_nextgen.runtime.cli master --pipeline-strategy outputs/scheduler/export/fine-grained-bs32-len32/pipeline_strategy.json --bind 0.0.0.0:50050 --metrics-out outputs/runtime/logs/fine-grained-bs32-len32/master.metrics.json
```

```powershell
# worker-3060_0
python -m pp_nextgen.runtime.cli worker --worker-name 3060_0 --master 127.0.0.1:50050 --pipeline-strategy outputs/scheduler/export/fine-grained-bs32-len32/pipeline_strategy.json --worker-strategy outputs/scheduler/export/fine-grained-bs32-len32/3060_0.strategy.json --model-config configs/model/llama2_7b.yaml --metrics-out outputs/runtime/logs/fine-grained-bs32-len32/3060_0.metrics.json --runtime-config configs/runtime/runtime.example.yaml

# worker-2080super_0
python -m pp_nextgen.runtime.cli worker --worker-name 2080super_0 --master 192.168.31.237:50050 --pipeline-strategy outputs/scheduler/export/fine-grained-bs32-len32/pipeline_strategy.json --worker-strategy outputs/scheduler/export/fine-grained-bs32-len32/2080super_0.strategy.json --model-config configs/model/llama2_7b.yaml --metrics-out outputs/runtime/logs/fine-grained-bs32-len32/2080super_0.metrics.json --runtime-config configs/runtime/runtime.example.yaml

# worker-agx_0
python -m pp_nextgen.runtime.cli worker --worker-name agx_0 --master 192.168.31.237:50050 --pipeline-strategy outputs/scheduler/export/fine-grained-bs32-len32/pipeline_strategy.json --worker-strategy outputs/scheduler/export/fine-grained-bs32-len32/agx_0.strategy.json --model-config configs/model/llama2_7b.yaml --metrics-out outputs/runtime/logs/fine-grained-bs32-len32/agx_0.metrics.json --runtime-config configs/runtime/runtime.example.yaml

# worker-tx2_0
python -m pp_nextgen.runtime.cli worker --worker-name tx2_0 --master 192.168.31.237:50050 --pipeline-strategy outputs/scheduler/export/fine-grained-bs32-len32/pipeline_strategy.json --worker-strategy outputs/scheduler/export/fine-grained-bs32-len32/tx2_0.strategy.json --model-config configs/model/llama2_7b.yaml --metrics-out outputs/runtime/logs/fine-grained-bs32-len32/tx2_0.metrics.json --runtime-config configs/runtime/runtime.example.yaml
```


Submit one request:

```powershell
python tools/runtime/submit_task_to_master.py --master 192.168.31.237:50050 --batch-size 32 --context-len 32 --target-len 64 --req-id demo-1
```

Submit poisson requests:
```powershell
python tools/runtime/send_poisson_tasks.py --master 192.168.31.237:50050 --mode fixed --num-requests 20 --arrival-rate 40 --batch-size 32 --context-len 1 --target-len 63 --send-pipeline-stop
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