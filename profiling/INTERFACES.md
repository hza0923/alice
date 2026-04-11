# Profiling Interfaces

## capture

```python
from dataclasses import dataclass

@dataclass
class CaptureRequest:
    model_name: str
    device_id: str
    batch_sizes: list[int]
    prefill_seq_lens: list[int]
    decode_context_lens: list[int]
    modules: list[str]

def run_capture(req: CaptureRequest) -> dict: ...
```

## fit

```python
def fit_time_models(all_results: dict) -> dict: ...
def fit_memory_models(all_results: dict, model_cfg: dict) -> dict: ...
```

## build

```python
def build_device_registry(fitted_by_device: list[dict], model_name: str) -> dict: ...
def merge_registries(registries: list[dict]) -> dict: ...
```
