#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R01: train pre-OOS opportunity models and immediately backtest a strategy."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ai_research.rl_market_agent.r01_config import DEFAULT_R01_CONFIG
from src.ai_research.rl_market_agent.r01_pipeline import config_with_overrides, run_r01


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETH RL Market Agent V1 - R01 opportunity strategy backtest")
    parser.add_argument("--data-dir", default=None, help="Optional data root passed only to src.data_feed loaders.")
    parser.add_argument("--cache-dir", default=None, help="R00.4 cache directory.")
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--no-review-pack", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_with_overrides(
        DEFAULT_R01_CONFIG,
        r00_cache_dir=args.cache_dir,
        report_dir=args.report_dir,
    )
    print("[run] ETH RL Market Agent V1 - R01 opportunity model -> executable strategy", flush=True)
    print(f"[seal] 2026 holdout remains closed from {config.sealed_holdout_start}", flush=True)
    print(f"[cost] base={config.round_trip_cost:.3%} stress={config.cost_stress_multipliers}", flush=True)
    print(f"[templates] {[x.name for x in config.trade_templates]}", flush=True)
    result = run_r01(config, data_dir=args.data_dir, finalize_report=not args.no_review_pack)
    print(f"[decision] {result['decision']}", flush=True)
    print(f"[report] {result['report_dir']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
