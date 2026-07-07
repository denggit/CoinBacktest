#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment import ExperimentRecord, ExperimentRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create or update an experiment record.")
    parser.add_argument("--id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--stage", default="idea")
    parser.add_argument("--status", default="idea")
    parser.add_argument("--symbol", default="ETH-USDT-SWAP")
    parser.add_argument("--family", default="unclassified")
    parser.add_argument("--hypothesis", default="")
    parser.add_argument("--data-required", nargs="*", default=[])
    parser.add_argument("--notes", default="")
    parser.add_argument("--registry", default="experiments/registry.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    registry = ExperimentRegistry(args.registry)
    current = registry.get(args.id)
    if current is None:
        record = ExperimentRecord(
            id=args.id,
            title=args.title,
            stage=args.stage,
            status=args.status,
            symbol=args.symbol,
            family=args.family,
            hypothesis=args.hypothesis,
            data_required=tuple(args.data_required),
            notes=args.notes,
        )
    else:
        record = current.with_update(
            title=args.title,
            stage=args.stage,
            status=args.status,
            symbol=args.symbol,
            family=args.family,
            hypothesis=args.hypothesis or current.hypothesis,
            data_required=tuple(args.data_required),
            notes=args.notes or current.notes,
        )
    registry.upsert(record)
    print(f"experiment registered | id={record.id} stage={record.stage} status={record.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
