#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""RDP V1 Stage 01: continuous ETH directional return-distribution baseline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_research.return_distribution_portfolio.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--force-rebuild-shards", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_pipeline(
        data_dir=args.data_dir,
        force_rebuild_shards=args.force_rebuild_shards,
        progress=not args.no_progress,
    )
    print(f"[RDP V1] report={result.report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
