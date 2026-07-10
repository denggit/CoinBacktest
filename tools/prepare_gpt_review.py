#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Prepare a GPT review pack for an existing CoinBacktest report directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.research_common.review_pack import finalize_research_report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build gpt_review_pack.zip for a research report.")
    p.add_argument("--report-dir", required=True, help="Existing report directory to package.")
    p.add_argument("--experiment-id", default=None, help="Optional experiment id override.")
    p.add_argument("--edge-id", default=None, help="Optional edge id override.")
    p.add_argument("--title", default=None, help="Optional title override.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    result = finalize_research_report(
        Path(args.report_dir),
        experiment_id=args.experiment_id,
        edge_id=args.edge_id,
        title=args.title,
    )
    print(f"GPT review pack: {result.zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
