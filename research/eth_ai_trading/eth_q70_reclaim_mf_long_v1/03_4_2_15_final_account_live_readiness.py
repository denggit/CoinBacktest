#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Thin entry point for R03.4.2.15."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_research.long_tail_final_account_audit import run_final_account_audit


def main() -> int:
    parser = argparse.ArgumentParser(description="R03.4.2.15 frozen C2 final account and live-readiness audit")
    parser.add_argument("--source-report-dir", default=None)
    args = parser.parse_args()
    result = run_final_account_audit(source_report_dir=args.source_report_dir)
    print(f"[decision] {result.decision}")
    print(f"[report] {result.report_dir}")
    return 0 if result.decision not in {"FAIL_RUNTIME", "BLOCKED_SOURCE_REPORT"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
