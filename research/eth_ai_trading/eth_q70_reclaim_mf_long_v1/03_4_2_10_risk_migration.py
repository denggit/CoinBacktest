#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Thin entrypoint for R03.4.2.10 risk migration research."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_research.long_tail_risk_migration.pipeline import run_risk_migration_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_risk_migration_audit(data_dir=args.data_dir, progress=not args.no_progress)
    print(f"[decision] {result.decision}")
    print(f"[report] {result.report_dir}")
    return 0 if result.decision not in {"FAIL_RUNTIME", "BLOCKED_SOURCE_REPORT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
