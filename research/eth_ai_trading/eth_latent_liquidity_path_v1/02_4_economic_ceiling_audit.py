#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run R02.4 latent-liquidity economic ceiling audit."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_research.latent_liquidity_economic_ceiling import DEFAULT_CONFIG, run_economic_ceiling_audit  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report-dir", default=DEFAULT_CONFIG.source_report_dir)
    parser.add_argument("--report-dir", default=DEFAULT_CONFIG.report_dir)
    parser.add_argument("--read-chunk-rows", type=int, default=DEFAULT_CONFIG.csv_read_chunk_rows)
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--skip-review-pack", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = replace(
        DEFAULT_CONFIG,
        source_report_dir=args.source_report_dir,
        report_dir=args.report_dir,
        csv_read_chunk_rows=int(args.read_chunk_rows),
    )
    result = run_economic_ceiling_audit(
        progress=not bool(args.no_progress),
        skip_review_pack=bool(args.skip_review_pack),
        config=config,
    )
    return 1 if result.decision.startswith("BLOCKED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
