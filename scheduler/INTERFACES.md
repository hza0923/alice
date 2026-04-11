# Scheduler Interfaces

## Adapter Layer

```python
def load_solve_context(
    model_cfg_path: str,
    cluster_cfg_path: str,
    registry_path: str,
    runtime_options: dict
) -> dict: ...
```

## Core Solver Layer

```python
def solve_once(context: dict, designated_tail_n: int) -> dict: ...
def solve_tail_sweep(context: dict, tail_candidates: list[int]) -> dict: ...
```

## Export Layer

```python
def write_pipeline_strategy(result: dict, out_path: str) -> None: ...
def write_worker_strategies(result: dict, out_dir: str) -> None: ...
```
