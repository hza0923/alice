# Profiling Module

## Submodules

- `capture/`: per-device benchmark execution
- `fit/`: phase/module model fitting by batch-size
- `build/`: registry assembly and merge
- `schemas/`: data models and validators
- `artifacts/`: generated JSON artifacts

## Responsibilities

- collect module-level timing samples
- fit formulas according to phase/module rules
- emit normalized device registry for scheduler consumption
