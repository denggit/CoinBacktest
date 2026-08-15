#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compatibility entry for the superseding R01.1 liquidity-first atlas."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_research.latent_liquidity_path_atlas import (  # noqa: E402
    DEFAULT_CONFIG,
    run_latent_liquidity_path_atlas,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETH Latent Liquidity Pool Path Learning V1 - R01.1")
    p.add_argument("--symbol", default=DEFAULT_CONFIG.symbol)
    p.add_argument("--warmup-start-date", default=DEFAULT_CONFIG.warmup_start)
    p.add_argument("--start-date", default=DEFAULT_CONFIG.research_start)
    p.add_argument("--end-date", default=DEFAULT_CONFIG.research_end)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--chunk-days", type=int, default=DEFAULT_CONFIG.chunk_days)
    p.add_argument("--max-events", type=int, default=0)
    p.add_argument("--build-missing", action="store_true")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--no-chunk-cache", action="store_true")
    return p.parse_args(argv)


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
