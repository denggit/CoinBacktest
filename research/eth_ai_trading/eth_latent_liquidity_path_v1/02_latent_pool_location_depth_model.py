#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai_research.latent_liquidity_pool_forecast import DEFAULT_CONFIG, run_latent_liquidity_pool_forecast


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R02 pre-event latent liquidity-pool location and sweep-depth forecast")
    p.add_argument("--start-date", default=DEFAULT_CONFIG.research_start)
    p.add_argument("--end-date", default=DEFAULT_CONFIG.research_end)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--snapshot-minutes", type=int, default=DEFAULT_CONFIG.snapshot_interval_minutes)
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = replace(DEFAULT_CONFIG, research_start=args.start_date, research_end=args.end_date, snapshot_interval_minutes=args.snapshot_minutes)
    run_latent_liquidity_pool_forecast(data_dir=args.data_dir, db_name=args.db_name, skip_review_pack=args.skip_review_pack, use_cache=not args.no_cache, config=config)


if __name__ == "__main__":
    main()
