#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI for R03.4.2.8B account-level dual risk-slot research."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_research.long_tail_tranche_account.pipeline import run_tranche_account_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETH AI R03.4.2.8B dual risk-slot account audit")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_tranche_account_audit(
        data_dir=args.data_dir,
        progress=not args.no_progress,
    )
    print(f"[decision] {result.decision}")
    print(f"[report] {result.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
