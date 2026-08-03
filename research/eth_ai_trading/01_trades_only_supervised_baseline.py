#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run R01 as a complete trades-only research gate.

This script deliberately does not perform a separate forensic data project.
It uses the existing ``src.data_feed`` 1s interface, runs a small smoke check,
then immediately builds causal samples, trains Ridge/LightGBM baselines, and
executes market-order cost/latency stress.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_research.trades_baseline.config import DEFAULT_TRADES_BASELINE_CONFIG
from src.ai_research.trades_baseline.pipeline import run_pipeline


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETH AI R01 trades-only supervised baseline")
    parser.add_argument("--start-date", default=DEFAULT_TRADES_BASELINE_CONFIG.research_start)
    parser.add_argument("--end-date", default=DEFAULT_TRADES_BASELINE_CONFIG.research_end)
    parser.add_argument("--data-dir", default=None, help="Optional CoinBacktest data directory override")
    parser.add_argument("--cache-dir", default=DEFAULT_TRADES_BASELINE_CONFIG.cache_dir)
    parser.add_argument("--report-dir", default=DEFAULT_TRADES_BASELINE_CONFIG.report_dir)
    parser.add_argument("--force-rebuild-cache", action="store_true")
    parser.add_argument("--max-train-rows", type=int, default=DEFAULT_TRADES_BASELINE_CONFIG.train_sample_cap)
    parser.add_argument("--models", nargs="+", choices=("ridge", "lightgbm"), default=("ridge", "lightgbm"))
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--fail-on-not-pass", action="store_true", help="Compatibility flag; R01 always writes an explicit PASS/FAIL decision")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = replace(
        DEFAULT_TRADES_BASELINE_CONFIG,
        research_start=args.start_date,
        research_end=args.end_date,
        train_sample_cap=args.max_train_rows,
        cache_dir=args.cache_dir,
        report_dir=args.report_dir,
    )
    result = run_pipeline(
        config,
        data_dir=args.data_dir,
        force_rebuild_cache=args.force_rebuild_cache,
        models=tuple(args.models),
        progress=not args.no_progress,
    )
    print(f"[R01] decision={result.decision}")
    print(f"[R01] report_dir={result.report_dir}")
    return 0 if result.decision.startswith("PASS") or result.decision.startswith("FAIL") else 2


if __name__ == "__main__":
    raise SystemExit(main())
