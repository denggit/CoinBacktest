#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run R03.3.1 independent-alert and remaining-opportunity audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_research.future_process_forecast.alert_audit_pipeline import run_alert_audit_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--force-rebuild-events", action="store_true")
    parser.add_argument("--force-rebuild-micro", action="store_true")
    parser.add_argument("--force-rebuild-long-context", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_alert_audit_pipeline(
        data_dir=args.data_dir,
        force_rebuild_events=args.force_rebuild_events,
        force_rebuild_micro=args.force_rebuild_micro,
        force_rebuild_long_context=args.force_rebuild_long_context,
        progress=not args.no_progress,
    )
    print(f"[R03.3.1] decision={result.decision}")
    print(f"[R03.3.1] report={result.report_dir}")
    return 2 if result.decision.startswith("BLOCKED_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
