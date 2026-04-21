#!/usr/bin/env python3
"""Calculate equivalent target-len from a unified request statistics JSON.

Algorithm:
For each request, count every integer length in (prompt_len, target_len),
i.e. prompt_len + 1 ... target_len - 1. Let freq[length] be the aggregated
counts. Total events N = sum(freq). The empirical probability is
P(length) = freq[length] / N. Then:

    equivalent_target_len = sum(length * P(length)) over all observed lengths.

This equals the weighted mean of length under the multiset of counted positions.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _pick_int(row: Dict[str, Any], keys: Iterable[str], field_label: str) -> int:
    for key in keys:
        if key in row:
            return int(row[key])
    wanted = ", ".join(keys)
    raise KeyError(f"missing {field_label}; expected one of: {wanted}")


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict):
        rows = payload.get("requests")
        if not isinstance(rows, list):
            raise ValueError("JSON object must contain a 'requests' list")
        return rows
    if isinstance(payload, list):
        return payload
    raise ValueError("JSON payload must be either a list or an object with 'requests'")


def _compute_equivalent_target_len(
    rows: List[Dict[str, Any]],
) -> Tuple[float, Dict[int, int], int, int]:
    freq: Dict[int, int] = defaultdict(int)
    valid_rows = 0

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"rows[{i}] must be an object")

        prompt_len = _pick_int(
            row,
            keys=("prompt_len", "context_len", "prompt_tokens"),
            field_label="prompt/context length",
        )
        target_len = _pick_int(
            row,
            keys=("target_len", "total_tokens"),
            field_label="target/total length",
        )

        if target_len <= prompt_len:
            continue

        valid_rows += 1
        for length in range(prompt_len + 1, target_len):
            freq[length] += 1

    freq_sorted = dict(sorted(freq.items()))
    total_events = sum(freq_sorted.values())
    if total_events <= 0:
        equivalent_target_len = 0.0
    else:
        equivalent_target_len = sum(
            length * (count / total_events) for length, count in freq_sorted.items()
        )
    return equivalent_target_len, freq_sorted, valid_rows, total_events


def _build_result_doc(
    *,
    stats_file: Path,
    equivalent_target_len: float,
    freq: Dict[int, int],
    valid_rows: int,
    total_events: int,
    rows_total: int,
) -> Dict[str, Any]:
    distribution: List[Dict[str, Any]] = []
    for length, count in freq.items():
        p = (count / total_events) if total_events > 0 else 0.0
        distribution.append({"length": length, "frequency": count, "probability": p})

    return {
        "stats_file": str(stats_file),
        "requests_total": rows_total,
        "requests_counted": valid_rows,
        "total_interval_events": total_events,
        "distinct_lengths": len(freq),
        "equivalent_target_len": equivalent_target_len,
        "distribution": distribution,
        "note": "probability = frequency / total_interval_events; "
        "equivalent_target_len = sum(length * probability).",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stats_file",
        help="Path to unified statistics JSON (object with requests[] or plain list)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Write full results (JSON) to this path",
    )
    parser.add_argument(
        "--show-top",
        type=int,
        default=20,
        help="Show first N length-frequency entries (default: 20, use <=0 to hide)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    stats_file = Path(args.stats_file).resolve()
    if not stats_file.is_file():
        raise FileNotFoundError(f"file not found: {stats_file}")

    rows = _load_rows(stats_file)
    eq_target_len, freq, valid_rows, total_events = _compute_equivalent_target_len(rows)

    out_path = Path(args.output).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = _build_result_doc(
        stats_file=stats_file,
        equivalent_target_len=eq_target_len,
        freq=freq,
        valid_rows=valid_rows,
        total_events=total_events,
        rows_total=len(rows),
    )
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)

    print(f"file: {stats_file}")
    print(f"requests_total: {len(rows)}")
    print(f"requests_counted: {valid_rows}")
    print(f"total_interval_events: {total_events}")
    print(f"distinct_lengths: {len(freq)}")
    print(f"equivalent_target_len: {eq_target_len}")
    print(f"wrote: {out_path}")

    show_top = int(args.show_top)
    if show_top > 0:
        print("length_frequency_preview:")
        shown = 0
        for length, count in freq.items():
            print(f"  {length}: {count}")
            shown += 1
            if shown >= show_top:
                break
        if len(freq) > show_top:
            print(f"  ... ({len(freq) - show_top} more)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
