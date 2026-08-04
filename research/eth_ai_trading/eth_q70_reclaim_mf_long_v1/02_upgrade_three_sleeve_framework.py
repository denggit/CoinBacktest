#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Write and validate the R02 three-sleeve framework snapshot."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ai_research.config import DEFAULT_REPORT_ROOT  # noqa: E402
from src.ai_research.sleeves.artifacts import write_sleeve_framework_artifacts  # noqa: E402
from src.ai_research.sleeves.registry import SLEEVE_SPECS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="R02 ETH AI three-sleeve framework upgrade")
    parser.add_argument(
        "--report-dir",
        default=str(DEFAULT_REPORT_ROOT / "02_three_sleeve_framework"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if set(SLEEVE_SPECS) != {"short_horizon", "intraday_trend", "swing"}:
        raise RuntimeError("R02 sleeve registry is incomplete")
    paths = write_sleeve_framework_artifacts(args.report_dir)
    print("[R02] three-sleeve framework READY")
    for name, path in paths.items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
