#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.edge_library import EdgeLibrary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List ETH edge library records.")
    parser.add_argument("--registry", default="edge_library/registry.json")
    parser.add_argument("--status", default="", help="Optional status filter.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = EdgeLibrary(args.registry).load()
    if args.status:
        records = [r for r in records if r.status == args.status]
    if not records:
        print("No edge records found.")
        return 0
    print(f"{'id':<34} {'status':<18} {'family':<24} {'name'}")
    print("-" * 110)
    for record in records:
        print(f"{record.id:<34} {record.status:<18} {record.family:<24} {record.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
