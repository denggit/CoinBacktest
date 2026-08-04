#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R03: daily/4H direction plus lower-timeframe swing entry research."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_research.swing_baseline.config import DEFAULT_SWING_BASELINE_CONFIG  # noqa: E402
from src.ai_research.swing_baseline.modeling import ARCHITECTURES  # noqa: E402
from src.ai_research.swing_baseline.pipeline import run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="R03 ETH medium-horizon swing supervised baseline",
    )
    parser.add_argument("--symbol", default=DEFAULT_SWING_BASELINE_CONFIG.symbol)
    parser.add_argument("--start-date", default=DEFAULT_SWING_BASELINE_CONFIG.research_start)
    parser.add_argument("--end-date", default=DEFAULT_SWING_BASELINE_CONFIG.research_end)
    parser.add_argument(
        "--architectures",
        default=",".join(ARCHITECTURES),
        help="comma-separated: high_logistic,high_lightgbm,full_lightgbm,hierarchical_lightgbm",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    architectures = tuple(item.strip() for item in args.architectures.split(",") if item.strip())
    config = replace(
        DEFAULT_SWING_BASELINE_CONFIG,
        symbol=args.symbol,
        research_start=args.start_date,
        research_end=args.end_date,
    )
    print("[R03] ETH Swing supervised baseline")
    print(f"[window] {config.research_start} -> {config.research_end}")
    print(f"[architectures] {architectures}")
    print("[contract] 1D/4H/1H direction + 30m/15m/5m/1m entry; structural exits; no fixed-time primary exit")
    result = run_pipeline(
        config=config,
        architectures=architectures,
        force_rebuild_cache=args.force_rebuild_cache,
        data_dir=args.data_dir,
        progress=not args.no_progress,
    )
    print(f"[R03 decision] {result.decision}")
    print(f"[report] {result.report_dir}")
    return 0 if result.decision != "BLOCKED_PUBLIC_LOADER" else 2


if __name__ == "__main__":
    raise SystemExit(main())
