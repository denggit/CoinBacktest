#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai_research.latent_liquidity_pool_strength import DEFAULT_CONFIG, run_latent_liquidity_pool_strength


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="R02.1 arrival-independent latent liquidity-pool strength / density model")
    p.add_argument("--skip-review-pack", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_latent_liquidity_pool_strength(skip_review_pack=args.skip_review_pack, use_cache=not args.no_cache, config=DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
