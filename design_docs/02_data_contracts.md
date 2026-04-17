# Data Contracts

## Goals

- Define strict, versioned contracts for:
  - `all_results` (per-device raw profiling output)
  - `device_registry` (normalized scheduler input)
  - `pipeline_strategy` (global solve result)
  - `worker_strategy` (per-node execution plan)
- Keep equations free from runtime `*bs` factors by storing params per batch-size.

## Common Rules

- Every artifact has `schema_version` and `generated_at` (ISO-8601 UTC).
- Numeric units:
  - Time: `ms`
  - Memory: `GB` for persisted model terms, `bytes` for transfer payload
- All module IDs use canonical names:
  - `input_embed`
  - `qkv_projection`
  - `attn_qk`
  - `attn_av`
  - `o_projection`
  - `gate_up_projection`
  - `down_projection`
  - `output_embed`

## all_results Schema (per device)

Path pattern:

- `outputs/profiling/all_results/{model}_{device}.json`

Shape:

```json
{
  "schema_version": "all_results.v1",
  "generated_at": "2026-04-10T10:00:00Z",
  "model": "llama2-7b",
  "device_id": "node-3060-0",
  "device_type": "RTX3060",
  "device_memory_gb": 12.0,
  "test_configurations": [
    {
      "batch_size": 1,
      "prefill_seq_lens": [128, 256, 512],
      "decode_context_lens": [128, 256, 512, 1024],
      "modules": {
        "attn_qk": {
          "status": "completed",
          "prefill_samples_ms": {"128": 0.81, "256": 2.91},
          "decode_samples_ms": {"128": 0.13, "256": 0.26},
          "weight_size_gb": 0.0
        }
      }
    }
  ]
}
```

Notes:

- `attn_qk` and `attn_av` must include sequence-indexed samples for both phases.
- Other modules can still store indexed samples; fitter chooses model form.
- `kvcache_size_gb` is intentionally removed from this artifact.

## device_registry Schema (normalized)

Path:

- `outputs/profiling/registry/device_registry.v3.json`

Shape:

```json
{
  "schema_version": "device_registry.v3",
  "generated_at": "2026-04-10T10:30:00Z",
  "model": "llama2-7b",
  "devices": {
    "node-3060-0": {
      "device_type": "RTX3060",
      "memory_gb": 12.0,
      "modules": {
        "attn_qk": {
          "time_models": {
            "prefill": {
              "by_bs": {
                "1": {"form": "quadratic", "c0": 0.01, "c1": 0.002, "c2": 0.00001, "x": "seq_len", "unit": "ms"}
              }
            },
            "decode": {
              "by_bs": {
                "1": {"form": "linear", "c0": 0.03, "c1": 0.0008, "x": "context_len", "unit": "ms"}
              }
            }
          },
          "memory_models": {
            "decode": {
              "by_bs": {
                "1": {"form": "linear", "c0": 0.0, "c1": 0.000002, "x": "seq_len", "unit": "GB"}
              }
            }
          },
          "weight_size_gb": 0.0
        }
      }
    }
  }
}
```

Model forms:

- `constant`: `t = c0`
- `linear`: `t = c0 + c1 * x`
- `quadratic`: `t = c0 + c1 * x + c2 * x^2`

Memory modeling rules:

- `attn_qk`, `attn_av`:
  - no weight term
  - decode memory from config constant:
    - `memory_gb = kv_per_token_bytes * batch_size * seq_len / 1024^3`
- other modules:
  - `memory_gb = weight_size_gb`
  - no activation-dependent term in scheduler decision

## pipeline_strategy Schema (global)

Path:

- `outputs/scheduler/export/pipeline_strategy.json`

Shape:

```json
{
  "schema_version": "pipeline_strategy.v2",
  "generated_at": "2026-04-10T11:00:00Z",
  "model": "llama2-7b",
  "schedule_input": {
    "bs": 1,
    "target_seq_len": 2048,
    "designated_device": "3060",
    "designated_tail_n": 3,
    "use_fine_grained": true
  },
  "objective": {"name": "min_tbt_ms", "value": 6.42},
  "pipeline_stages": [
    {
      "worker_name": "node-3060-0",
      "modules_to_execute": ["input_embed", "qkv_projection"],
      "stage_models": {
        "decode": {
          "time_ms": {"single": {"form": "linear", "c0": 1.1, "c1": 0.0025, "x": "context_len"}},
          "comm_bytes": {"single": {"form": "linear", "c0": 4096, "c1": 8192, "x": "context_len"}},
          "comm_time_ms": {"single": {"form": "linear", "c0": 0.03, "c1": 0.00008, "x": "context_len"}},
          "memory_gb": {
            "single": {
              "form": "linear",
              "c0": 1.6,
              "c1": 0.000002,
              "x": "seq_len",
              "batch_term": "batch_size",
              "expr": "c0 + c1 * seq_len * batch_size"
            }
          }
        },
        "prefill": {
          "time_ms": {"single": {"form": "quadratic", "c0": 0.2, "c1": 0.001, "c2": 0.000001, "x": "seq_len"}},
          "comm_bytes": {"single": {"form": "quadratic", "c0": 0, "c1": 8192, "c2": 32, "x": "seq_len"}},
          "comm_time_ms": {"single": {"form": "quadratic", "c0": 0, "c1": 0.00008, "c2": 0.0000003, "x": "seq_len"}}
        }
      },
      "next_worker": "node-2080s-0"
    }
  ]
}
```

Rules:

- Time and communication forms are both persisted as `c0 + c1*x + c2*x^2`.
- Time coefficients are selected from registry `by_bs` buckets, no extra `*bs` at runtime.
- Communication coefficients are shape-analytic and phase-specific:
  - decode uses `x=context_len`
  - prefill uses `x=seq_len`
- Decode memory model is:
  - `memory_gb = c0 + c1 * seq_len * batch_size`
  - `c0` is weight-only sum; `c1` is KV-per-token sum.

## worker_strategy Schema (per worker)

Path pattern:

- `outputs/scheduler/export/workers/{worker_name}.strategy.json`

Shape（**当前导出格式**；已取代单独的 `ordered_modules`）:

```json
{
  "schema_version": "worker_strategy.v1",
  "generated_at": "2026-04-10T11:00:00Z",
  "worker_name": "3060_0",
  "global_strategy_ref": "pipeline_strategy.json",
  "execution_plan": {
    "head_ordered_modules": ["input_embed", "qkv_projection", "attn_qk"],
    "tail_ordered_modules": ["lm_head"],
    "phase_rules": {
      "prefill": {"enabled": true},
      "decode": {"enabled": true}
    },
    "transfer_policy": {
      "compute_payload_bytes_locally": true,
      "formula_ref": "model_config.comm_contract.v1"
    }
  }
}
```

规则（与 `pp_nextgen/scheduler/export/strategy_export.py` 一致）:

- **`head_ordered_modules` + `tail_ordered_modules`** 覆盖该 worker 的全部 `modules_to_execute` 顺序。
- 仅当 **`worker_name`** 匹配 **`pipeline_strategy.schedule_input.designated_device`**（如 `3060` 匹配 `3060_0` / `3060_1`）时，把该 worker 模块列表的 **最后 `designated_tail_n` 个** 放入 **`tail_ordered_modules`**；**其余 worker 的 `tail_ordered_modules` 为空数组**。
- 遗留文件若仍含 **`ordered_modules`**，runtime 通过 `split_head_tail_modules_from_execution_plan` 按同一 designated 规则在 **首 worker** 上解析（非首 worker 视为整表在 head）。

运行时语义见 **`docs/09_runtime_implementation_status.md`**。

## Validation Rules

- Registry must contain all devices listed in cluster config.
- Every strategy stage module must exist in model `module_order`.
- `designated_tail_n` must satisfy `1 <= n <= 7`.
- For selected `batch_size`, all required module models must exist in registry.

## Python Interface Skeleton (Contracts)

```python
from dataclasses import dataclass
from typing import Dict, Literal

Form = Literal["constant", "linear", "quadratic"]

@dataclass
class ParamModel:
    form: Form
    c0: float
    c1: float = 0.0
    c2: float = 0.0
    x: str = "seq_len"
    unit: str = "ms"

def validate_all_results(payload: dict) -> None: ...
def validate_device_registry(payload: dict) -> None: ...
def validate_pipeline_strategy(payload: dict) -> None: ...
def validate_worker_strategy(payload: dict) -> None: ...
```
