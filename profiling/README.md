# Profiling Module

## Submodules

- `capture/`: per-device split-module benchmarks (Torch)
- `fit/`: phase/module model fitting by batch-size
- `build/`: registry assembly and merge
- `artifacts/`: generated JSON artifacts

## Responsibilities

- collect module-level timing samples
- fit formulas according to phase/module rules
- emit normalized device registry for scheduler consumption

## Capture (split-module bench)

Entry point:

```bash
python tools/capture_split_module_profiles.py --help
```

Important flags:

- `--device-id`: written to the legacy JSON `device` field; must match cluster `device_group` / registry `device_type` strings.
- `--output-dir`: where `<model>_all_results.json` is written.
- `--quick`: tiny len grid for smoke tests.
- `--components`: optional subset; default set matches `pp_nextgen/profiling/constants.py` (`PROFILE_TO_MODULE`).

Programmatic API:

```python
from pp_nextgen.profiling.capture import configure_runtime, run_all_benchmarks

configure_runtime(force_cpu=False, cuda_device=0, device_label="3060", n_repeats=20, warmup_repeats=1)
run_all_benchmarks(output_dir=Path("out"))
```

Next step: `python tools/build_registry.py --inputs ...` (see repository `README.md`).
