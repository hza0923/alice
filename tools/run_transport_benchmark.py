#!/usr/bin/env python3
"""Run transport A/B benchmarks."""

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Benchmark config")
    parser.add_argument("--out-dir", required=True, help="Benchmark output directory")
    return parser.parse_args()


def main() -> int:
    _ = parse_args()
    raise SystemExit("Not implemented. See docs/05_transport_ab_plan.md.")


if __name__ == "__main__":
    main()
