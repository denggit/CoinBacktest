#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R00: build/audit the causal state + forward opportunity dataset.

No trading model is trained in this stage.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.ai_research.rl_market_agent.config import DEFAULT_CONFIG
from src.ai_research.rl_market_agent.pipeline import config_with_overrides, run_r00


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ETH RL Market Agent V1 - R00 causal dataset audit")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--data-dir", default=None, help="Optional data root passed only to src.data_feed loaders.")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    parser.add_argument("--audit-only", action="store_true", help="Run all construction/audits but do not persist NumPy shards.")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild completed monthly shards instead of resuming.")
    parser.add_argument("--max-shards", type=int, default=None, help="Development smoke-test limiter. Do not use for full R00.")
    parser.add_argument("--require-micro", action="store_true")
    parser.add_argument("--require-range", action="store_true")
    parser.add_argument("--require-footprint", action="store_true")
    parser.add_argument("--no-review-pack", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = config_with_overrides(
        DEFAULT_CONFIG,
        research_start=args.start_date,
        research_end=args.end_date,
        cache_dir=args.cache_dir,
        report_dir=args.report_dir,
        require_micro_trade_bars=True if args.require_micro else None,
        require_range_bars=True if args.require_range else None,
        require_footprint=True if args.require_footprint else None,
    )
    print("[run] ETH RL Market Agent V1 - R00 causal state dataset", flush=True)
    print(f"[window] warmup={config.warmup_start} research_data={config.research_start} -> {config.research_end}", flush=True)
    print(f"[decision-window] {config.research_start} -> {config.decision_end} (tail reserved for forward labels)", flush=True)
    print(f"[holdout] sealed from {config.sealed_holdout_start}", flush=True)
    print(f"[decision] interval={config.decision_interval} horizons={config.label_horizons_minutes}", flush=True)
    result = run_r00(
        config,
        data_dir=args.data_dir,
        overwrite=args.overwrite,
        audit_only=args.audit_only,
        max_shards=args.max_shards,
        finalize_report=not args.no_review_pack,
    )
    print(f"[done] shards={len(result['records'])}", flush=True)
    print(f"[report] {result['report_dir']}", flush=True)
    if not args.audit_only:
        print(f"[cache] {result['cache_dir']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
