#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compatibility entrypoint for R03.3.3.1 market-state continuity audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_research.market_state_continuity.pipeline import run_market_state_continuity_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--force-rebuild-state-cache", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_market_state_continuity_pipeline(
        data_dir=args.data_dir,
        force_rebuild_state_cache=args.force_rebuild_state_cache,
        progress=not args.no_progress,
    )
    print(f"[R03.3.3.1] decision={result.decision}")
    print(f"[R03.3.3.1] report={result.report_dir}")
    return 2 if result.decision.startswith("BLOCKED_") else 0


if __name__ == "__main__":
    raise SystemExit(main())
