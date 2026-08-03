#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run R03.2 long-context 3%-5% swing opportunity research."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_research.swing_long_context.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--force-rebuild-long-context-cache", action="store_true")
    parser.add_argument("--force-rebuild-exact-labels", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_pipeline(
        force_rebuild_exact_labels=args.force_rebuild_exact_labels,
        force_rebuild_long_context_cache=args.force_rebuild_long_context_cache,
        data_dir=args.data_dir,
        progress=not args.no_progress,
    )
    print(f"[R03.2] decision={result.decision}")
    print(f"[R03.2] report={result.report_dir}")
    return 0 if result.decision != "BLOCKED_PUBLIC_LOADER" else 2


if __name__ == "__main__":
    raise SystemExit(main())
