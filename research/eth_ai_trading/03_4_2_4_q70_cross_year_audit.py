#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI for R03.4.2.4 q70 versus q90 cross-year audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_research.long_tail_q70_audit.pipeline import run_q70_cross_year_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETH AI R03.4.2.4 q70 cross-year opening-pool audit")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--force-rebuild-outcomes", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_q70_cross_year_audit(
        data_dir=args.data_dir,
        force_rebuild_outcomes=args.force_rebuild_outcomes,
        progress=not args.no_progress,
    )
    print(f"[decision] {result.decision}")
    print(f"[report] {result.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
