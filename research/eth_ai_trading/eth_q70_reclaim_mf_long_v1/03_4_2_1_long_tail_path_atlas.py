#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI for R03.4.2.1 q90 complete event path atlas."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_research.long_tail_path_atlas import run_long_tail_path_atlas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R03.4.2.1 frozen q90 complete 48h path atlas")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--force-rebuild-outcomes", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_long_tail_path_atlas(
        data_dir=args.data_dir,
        force_rebuild_outcomes=args.force_rebuild_outcomes,
        progress=not args.no_progress,
    )
    print(f"[decision] {result.decision}", flush=True)
    print(f"[report] {result.report_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
