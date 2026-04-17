# Configs Module

## Purpose
Holds model, cluster, and runtime configuration files consumed by profiling, scheduler, runtime, and simulator CLIs.

## Core Execution Logic
Tools load YAML from this directory, normalize values, then pass typed dictionaries to downstream modules.

## Math Formulas
- No direct fitting happens here.
- Parameters in this directory drive formulas elsewhere:
  - time/comm: `c0 + c1*x + c2*x^2`
  - decode memory: `c0 + c1*seq_len*batch_size`

## Input / Output and Legacy
- Input: hand-authored YAML.
- Output: in-memory config dicts.
- Legacy compatibility is preserved through optional fields and fallback defaults.

## How to Invoke
- Registry build: `python tools/profiling/build_registry.py --model-config configs/model/llama2_7b.yaml ...`
- Strategy solve: `python tools/scheduler/solve_strategy.py --cluster configs/cluster/all_devices.yaml`
