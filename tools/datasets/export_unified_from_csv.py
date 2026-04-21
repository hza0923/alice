#!/usr/bin/env python3
"""Export unified_dataset_requests.v1 JSON from *_e_rows.csv token statistics.

Filters: min_total_tokens < total_tokens < max_total_tokens, prompt_tokens < max_prompt_tokens.
Takes the first ``limit`` matching rows per CSV (in file order).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pp_nextgen.datasets.unified_requests import UNIFIED_SCHEMA_VERSION, write_unified_requests_json


def _rel_to_repo(p: Path) -> str:
    try:
        return str(p.relative_to(_REPO_ROOT))
    except ValueError:
        return str(p)


def _slug_from_csv(path: Path) -> str:
    name = path.name
    if name.endswith("_e_rows.csv"):
        return name[: -len("_e_rows.csv")]
    return path.stem


def _iter_filtered_rows(
    csv_path: Path,
    *,
    min_total_excl: int,
    max_total_excl: int,
    max_prompt_excl: int,
    limit: int,
) -> Tuple[List[Dict[str, Any]], int]:
    """Return up to ``limit`` row dicts and total scan count of matching rows in file."""
    out: List[Dict[str, Any]] = []
    scanned_matches = 0
    slug = _slug_from_csv(csv_path)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            pt = int(row["prompt_tokens"])
            tt = int(row["total_tokens"])
            at = int(row["answers_tokens"]) if "answers_tokens" in row else tt - pt
            if not (min_total_excl < tt < max_total_excl and pt < max_prompt_excl):
                continue
            scanned_matches += 1
            idx = len(out) + 1
            out.append(
                {
                    "req_id": f"{slug}-{idx}",
                    "context_len": pt,
                    "target_len": tt,
                    "prompt_tokens": pt,
                    "answers_tokens": at,
                    "total_tokens": tt,
                }
            )
            if len(out) >= limit:
                break
    return out, scanned_matches


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--datasets-dir",
        default="outputs/datasets",
        help="Directory containing *_e_rows.csv files",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/datasets/samples_1k4k_promptlt500_100",
        help="Directory for per-dataset unified JSON + manifest",
    )
    p.add_argument("--limit", type=int, default=100, help="Max rows per dataset")
    p.add_argument("--min-total", type=int, default=1024, help="Exclusive lower bound on total_tokens")
    p.add_argument("--max-total", type=int, default=4096, help="Exclusive upper bound on total_tokens")
    p.add_argument("--max-prompt", type=int, default=500, help="Exclusive upper bound on prompt_tokens")
    p.add_argument(
        "--csv",
        action="append",
        default=None,
        help="Optional explicit CSV path(s); default: all *_e_rows.csv under --datasets-dir",
    )
    return p.parse_args()


def main() -> int:
    args = _parse()
    limit = int(args.limit)
    if limit <= 0:
        print("--limit must be > 0", file=sys.stderr)
        return 2
    min_tt = int(args.min_total)
    max_tt = int(args.max_total)
    max_pt = int(args.max_prompt)
    if not (min_tt < max_tt):
        print("--min-total must be < --max-total", file=sys.stderr)
        return 2

    ds_dir = (_REPO_ROOT / args.datasets_dir).resolve()
    out_dir = (_REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.csv:
        csv_paths = [(_REPO_ROOT / c).resolve() for c in args.csv]
    else:
        csv_paths = sorted(ds_dir.glob("*_e_rows.csv"))

    if not csv_paths:
        print("no CSV files found", file=sys.stderr)
        return 2

    manifest: Dict[str, Any] = {
        "min_total_tokens_exclusive": min_tt,
        "max_total_tokens_exclusive": max_tt,
        "max_prompt_tokens_exclusive": max_pt,
        "limit_per_csv": limit,
        "schema_version": UNIFIED_SCHEMA_VERSION,
        "datasets": [],
    }

    for csv_path in csv_paths:
        if not csv_path.is_file():
            print(f"missing: {csv_path}", file=sys.stderr)
            return 2
        rows, _ = _iter_filtered_rows(
            csv_path,
            min_total_excl=min_tt,
            max_total_excl=max_tt,
            max_prompt_excl=max_pt,
            limit=limit,
        )
        slug = _slug_from_csv(csv_path)
        out_name = f"{slug}_unified_requests.json"
        out_path = out_dir / out_name
        write_unified_requests_json(path=out_path, rows=rows)
        manifest["datasets"].append(
            {
                "csv": _rel_to_repo(csv_path),
                "unified_json": _rel_to_repo(out_path),
                "exported_rows": len(rows),
            }
        )
        print(f"{csv_path.name}: exported {len(rows)} rows -> {_rel_to_repo(out_path)}")

    man_path = out_dir / "manifest.json"
    with man_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"wrote {_rel_to_repo(man_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
