#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build full OOS activity-persistence predictions for Analyze Tool visualization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyze_tool.ai_market_state_artifacts import (  # noqa: E402
    DEFAULT_ARTIFACT_DIR,
    build_activity_prediction_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_ARTIFACT_DIR),
        help="Analyze Tool derivative artifact directory",
    )
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    outputs = build_activity_prediction_artifacts(
        artifact_dir=args.output_dir,
        force_rebuild=bool(args.force_rebuild),
    )
    print(f"[done] AI market-state Analyze Tool artifacts={len(outputs)}")
    for output in outputs:
        print(f"  {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
