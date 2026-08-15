#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R01.1 liquidity-first latent-pool path atlas."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_research.latent_liquidity_path_atlas import DEFAULT_CONFIG, run_latent_liquidity_path_atlas  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETH Latent Liquidity Pool Path Learning V1 - R01.1")
    parser.add_argument("--symbol", default=DEFAULT_CONFIG.symbol)
    parser.add_argument("--warmup-start-date", default=DEFAULT_CONFIG.warmup_start)
    parser.add_argument("--start-date", default=DEFAULT_CONFIG.research_start)
    parser.add_argument("--end-date", default=DEFAULT_CONFIG.research_end)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--db-name", default="okx_trade_bars.db")
    parser.add_argument("--chunk-days", type=int, default=DEFAULT_CONFIG.chunk_days)
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--build-missing", action="store_true")
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--skip-review-pack", action="store_true")
    parser.add_argument("--no-chunk-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = replace(
        DEFAULT_CONFIG,
        symbol=args.symbol,
        warmup_start=args.warmup_start_date,
        research_start=args.start_date,
        research_end=args.end_date,
        chunk_days=int(args.chunk_days),
        candidate_cap=int(args.max_events),
    )
    run_latent_liquidity_path_atlas(
        data_dir=args.data_dir,
        db_name=args.db_name,
        build_missing=bool(args.build_missing),
        force_rebuild=bool(args.force_rebuild),
        progress=not bool(args.no_progress),
        skip_review_pack=bool(args.skip_review_pack),
        use_chunk_cache=not bool(args.no_chunk_cache),
        config=config,
    )


if __name__ == "__main__":
    main()
