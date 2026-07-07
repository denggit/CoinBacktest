#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.experiment import write_decision


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a standard 09_decision.json artifact.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--next-action", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    path = write_decision(
        args.out_dir,
        experiment_id=args.experiment_id,
        stage=args.stage,
        status=args.status,
        reason=args.reason,
        next_action=args.next_action,
    )
    print(f"decision written | path={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
