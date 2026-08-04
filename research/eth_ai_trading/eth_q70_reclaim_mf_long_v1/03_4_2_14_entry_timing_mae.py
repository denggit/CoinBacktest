#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Thin entry point for R03.4.2.14."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT=Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0,str(PROJECT_ROOT))

from src.ai_research.long_tail_entry_timing_mae import run_entry_timing_audit


def main() -> int:
    parser=argparse.ArgumentParser(description="R03.4.2.14 entry timing and MAE attribution")
    parser.add_argument("--data-dir",default=None); parser.add_argument("--no-progress",action="store_true")
    args=parser.parse_args(); result=run_entry_timing_audit(data_dir=args.data_dir,progress=not args.no_progress)
    print(f"[decision] {result.decision}"); print(f"[report] {result.report_dir}")
    return 0 if result.decision not in {"FAIL_RUNTIME","BLOCKED_SOURCE_REPORT"} else 1


if __name__=="__main__": raise SystemExit(main())
