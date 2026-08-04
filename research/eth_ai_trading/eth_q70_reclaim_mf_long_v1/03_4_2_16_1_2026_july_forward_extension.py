#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run R03.4.2.16.1 July-2026 forward extension of frozen C2."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_research.long_tail_forward_extension.pipeline import run_forward_extension


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--force-rebuild-base", action="store_true")
    parser.add_argument("--force-rebuild-outcomes", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_forward_extension(
        data_dir=args.data_dir,
        force_rebuild_base=args.force_rebuild_base,
        force_rebuild_outcomes=args.force_rebuild_outcomes,
        progress=not args.no_progress,
    )
    print(f"[decision] {result.decision}")
    print(f"[report] {result.report_dir}")
    return 0 if result.decision not in {"FAIL_RUNTIME", "BLOCKED_SOURCE_OR_SEAL"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
