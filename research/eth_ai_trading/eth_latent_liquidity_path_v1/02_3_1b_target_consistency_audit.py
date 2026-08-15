#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI for R02.3.1b hurdle target-consistency and residual-distance audit."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai_research.latent_liquidity_target_consistency.config import DEFAULT_CONFIG
from src.ai_research.latent_liquidity_target_consistency.pipeline import run_target_consistency_audit


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETH latent liquidity path V1 R02.3.1b target consistency audit")
    parser.add_argument("--no-cache", action="store_true", help="Rebuild R02.3.1b nuisance predictions and target-consistency cache")
    parser.add_argument("--skip-review-pack", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_target_consistency_audit(
        config=DEFAULT_CONFIG,
        use_cache=not args.no_cache,
        skip_review_pack=args.skip_review_pack,
        progress=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
