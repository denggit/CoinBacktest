#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run R01.2 stable-path explanation and executable-confirmation audit."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_research.latent_liquidity_execution_audit import (  # noqa: E402
    DEFAULT_CONFIG,
    run_stable_path_execution_audit,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report-dir", default=DEFAULT_CONFIG.source_report_dir)
    parser.add_argument("--report-dir", default=DEFAULT_CONFIG.report_dir)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--db-name", default="okx_trade_bars.db")
    parser.add_argument("--read-chunk-rows", type=int, default=DEFAULT_CONFIG.csv_read_chunk_rows)
    parser.add_argument("--profile-sample-per-stratum", type=int, default=DEFAULT_CONFIG.profile_sample_per_stratum)
    parser.add_argument("--replay-sample-per-stratum", type=int, default=DEFAULT_CONFIG.replay_sample_per_stratum)
    parser.add_argument("--bootstrap-repetitions", type=int, default=DEFAULT_CONFIG.bootstrap_repetitions)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--skip-review-pack", action="store_true")
    parser.add_argument("--skip-micro-replay", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = replace(
        DEFAULT_CONFIG,
        source_report_dir=args.source_report_dir,
        report_dir=args.report_dir,
        csv_read_chunk_rows=int(args.read_chunk_rows),
        profile_sample_per_stratum=int(args.profile_sample_per_stratum),
        replay_sample_per_stratum=int(args.replay_sample_per_stratum),
        bootstrap_repetitions=int(args.bootstrap_repetitions),
    )
    result = run_stable_path_execution_audit(
        data_dir=args.data_dir,
        db_name=args.db_name,
        progress=not bool(args.no_progress),
        skip_review_pack=bool(args.skip_review_pack),
        skip_micro_replay=bool(args.skip_micro_replay),
        use_cache=not bool(args.no_cache),
        config=config,
    )
    return 1 if result.decision.startswith("BLOCKED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
