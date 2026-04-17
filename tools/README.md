# Tools

`tools/` is now organized by functional stage:

- `tools/profiling/`: capture and registry build
- `tools/scheduler/`: strategy solve and analytic benchmark
- `tools/runtime/`: proto generation and runtime task submit
- `tools/simulation/`: DES and local multi-process cluster simulation
- `tools/transport/`: transport A/B benchmark placeholder

## Recommended command order

1. Capture profiles  
   `python tools/profiling/capture_split_module_profiles.py --help`
2. Build registry  
   `python tools/profiling/build_registry.py --help`
3. Solve strategy  
   `python tools/scheduler/solve_strategy.py --help`
4. Runtime and simulators  
   `python tools/runtime/submit_task_to_master.py --help`  
   `python tools/simulation/run_pipeline_des_sim.py --help`  
   `python tools/simulation/run_cluster_sim.py --help`

## Compatibility

Legacy flat entries (`tools/build_registry.py`, `tools/solve_strategy.py`, etc.) are still kept for compatibility, while new docs use the grouped paths above.
