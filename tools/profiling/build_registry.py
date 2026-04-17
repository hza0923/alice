#!/usr/bin/env python3
"""Build device_registry.v3 from legacy *_all_results.json files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pp_nextgen.config_loader import kv_per_token_bytes, load_yaml
from pp_nextgen.profiling.build.registry_builder import build_device_registry_v3, write_json
from pp_nextgen.profiling.legacy_ingest import legacy_to_all_results_v1, load_legacy_all_results
from pp_nextgen.schemas.validate import validate_device_registry_minimal


def _repo_root() -> Path:
    return _REPO


def _resolve_path(raw: str, root: Path) -> Path:
    """Resolve a user path against cwd first, then repo root."""
    p = Path(raw)
    if p.is_file() or p.is_dir():
        return p.resolve()
    alt = (root / raw).resolve()
    if alt.is_file() or alt.is_dir():
        return alt
    return p.resolve()


def _legacy_profile_files_in_dir(d: Path) -> list[Path]:
    """Non-recursive: legacy capture files named *_all_results.json (excludes *_all_results.v1.json)."""
    files = sorted(
        p
        for p in d.glob("*_all_results.json")
        if p.is_file() and not p.name.endswith("_all_results.v1.json")
    )
    return files


def _expand_inputs(raw_inputs: list[str], root: Path) -> tuple[list[Path] | None, str | None]:
    """Each entry is a legacy JSON file or a directory of such files."""
    out: list[Path] = []
    for raw in raw_inputs:
        ip = _resolve_path(raw, root)
        if ip.is_dir():
            found = _legacy_profile_files_in_dir(ip)
            if not found:
                return None, f"no *_all_results.json under directory: {raw}"
            out.extend(found)
            continue
        if not ip.is_file():
            return None, f"input not found: {raw}"
        out.append(ip)
    if not out:
        return None, "--inputs expanded to no files"
    return out, None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Legacy *_all_results.json paths, or directories containing them",
    )
    p.add_argument(
        "--model-config",
        required=True,
        help="Model YAML (uses memory_contract.kv_per_token_bytes)",
    )
    p.add_argument(
        "--out",
        default="outputs/profiling/registry/device_registry.v3.json",
        help="Output registry JSON path",
    )
    p.add_argument(
        "--emit-all-results",
        default=None,
        help="If set, directory to write all_results.v1 JSON per input",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = _repo_root()
    model_path = Path(args.model_config)
    if not model_path.is_file():
        alt = root / args.model_config
        if alt.is_file():
            model_path = alt
        else:
            print(f"model config not found: {args.model_config}", file=sys.stderr)
            return 2

    model_doc = load_yaml(model_path)
    kv = kv_per_token_bytes(model_doc)
    if kv <= 0:
        print("memory_contract.kv_per_token_bytes must be > 0", file=sys.stderr)
        return 2

    inputs, err = _expand_inputs(list(args.inputs), root)
    if err:
        print(err, file=sys.stderr)
        return 2

    model_name = load_legacy_all_results(inputs[0]).model
    reg = build_device_registry_v3([str(p) for p in inputs], model_name=model_name, kv_per_token_bytes=kv)
    validate_device_registry_minimal(reg)

    out = Path(args.out)
    if not out.is_absolute():
        out = (root / out).resolve()
    write_json(out, reg)
    print(f"Wrote registry: {out}")

    if args.emit_all_results:
        eroot = Path(args.emit_all_results)
        eroot.mkdir(parents=True, exist_ok=True)
        for ip in inputs:
            b = load_legacy_all_results(ip)
            doc = legacy_to_all_results_v1(b, source_path=str(ip))
            safe = f"{b.model}_{b.device_id}".replace("/", "_")
            with (eroot / f"{safe}_all_results.v1.json").open("w", encoding="utf-8") as f:
                json.dump(doc, f, ensure_ascii=False, indent=2)
        print(f"Wrote all_results.v1 under: {eroot.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
