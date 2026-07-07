#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment import ExperimentRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List experiment registry records.")
    parser.add_argument("--registry", default="experiments/registry.json")
    parser.add_argument("--status", default="", help="Optional status filter.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = ExperimentRegistry(args.registry).load()
    if args.status:
        records = [r for r in records if r.status == args.status]
    if not records:
        print("No experiment records found.")
        return 0
    print(f"{'id':<34} {'stage':<12} {'status':<20} {'family':<24} {'title'}")
    print("-" * 120)
    for record in records:
        print(f"{record.id:<34} {record.stage:<12} {record.status:<20} {record.family:<24} {record.title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
