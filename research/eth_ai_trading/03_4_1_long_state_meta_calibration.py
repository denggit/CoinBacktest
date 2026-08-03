#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""CLI for R03.4.1 long-opportunity soft-state meta calibration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai_research.long_state_calibration import run_long_state_calibration


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--force-rebuild-state-cache", action="store_true")
    parser.add_argument("--force-rebuild-outcomes", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_long_state_calibration(
        data_dir=args.data_dir,
        force_rebuild_state_cache=args.force_rebuild_state_cache,
        force_rebuild_outcomes=args.force_rebuild_outcomes,
        progress=not args.no_progress,
    )
    print(f"[decision] {result.decision}")
    print(f"[report] {result.report_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
