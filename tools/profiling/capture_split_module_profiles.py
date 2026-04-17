#!/usr/bin/env python3
"""Run split-module benchmarks and emit legacy ``*_all_results.json`` (see ``profiling/README.md``)."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pp_nextgen.profiling.capture.split_module_bench import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
