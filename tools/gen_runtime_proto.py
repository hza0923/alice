#!/usr/bin/env python3
"""Regenerate pp_nextgen/runtime/grpc_gen from runtime/proto/pipeline_v2.proto."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    proto = root / "runtime" / "proto" / "pipeline_v2.proto"
    out = root / "pp_nextgen" / "runtime" / "grpc_gen"
    out.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        "-I",
        str(root / "runtime" / "proto"),
        f"--python_out={out}",
        f"--grpc_python_out={out}",
        str(proto),
    ]
    subprocess.check_call(cmd)
    grpc_py = out / "pipeline_v2_pb2_grpc.py"
    text = grpc_py.read_text(encoding="utf-8")
    text = text.replace(
        "import pipeline_v2_pb2 as pipeline__v2__pb2",
        "from . import pipeline_v2_pb2 as pipeline__v2__pb2",
    )
    grpc_py.write_text(text, encoding="utf-8")
    print(f"Wrote generated stubs under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
