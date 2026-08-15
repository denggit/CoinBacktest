#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI for R02.3 distance-normalized excess-liquidity ranking."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai_research.latent_liquidity_excess_ranking.config import DEFAULT_CONFIG
from src.ai_research.latent_liquidity_excess_ranking.pipeline import run_excess_liquidity_ranking


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETH latent liquidity path V1 R02.3 excess-liquidity ranking")
    parser.add_argument("--no-cache", action="store_true", help="Rebuild only the fast R02.3 transformed-label cache")
    parser.add_argument("--skip-review-pack", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_excess_liquidity_ranking(
        config=DEFAULT_CONFIG,
        use_cache=not args.no_cache,
        skip_review_pack=args.skip_review_pack,
        progress=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
