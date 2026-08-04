#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run R03.1 exact-path 3%-5% swing entry MVP research."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ai_research.swing_entry_mvp import DEFAULT_SWING_ENTRY_MVP_CONFIG, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-rebuild-exact-labels", action="store_true")
    parser.add_argument("--force-rebuild-base-cache", action="store_true")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_pipeline(
        config=DEFAULT_SWING_ENTRY_MVP_CONFIG,
        force_rebuild_exact_labels=args.force_rebuild_exact_labels,
        force_rebuild_base_cache=args.force_rebuild_base_cache,
        data_dir=args.data_dir,
        progress=not args.no_progress,
    )
    print(f"[R03.1] decision={result.decision}")
    print(f"[R03.1] report_dir={result.report_dir}")
    return 0 if result.decision != "BLOCKED_PUBLIC_LOADER" else 2


if __name__ == "__main__":
    raise SystemExit(main())
