#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI for R02.2 first-touch relative liquidity ranking."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _bootstrap() -> None:
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_bootstrap()

from src.ai_research.latent_liquidity_first_touch_ranking.pipeline import run_first_touch_liquidity_ranking


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETH latent liquidity R02.2 first-touch relative ranking")
    parser.add_argument("--skip-review-pack", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_first_touch_liquidity_ranking(
        skip_review_pack=args.skip_review_pack,
        use_cache=not args.no_cache,
        progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
