#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validate and materialise the ETH AI research framework artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ai_research.artifacts import write_framework_artifacts  # noqa: E402
from src.ai_research.config import DEFAULT_REPORT_ROOT  # noqa: E402
from src.ai_research.plan import DEFAULT_RESEARCH_PLAN, validate_research_plan  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_ROOT / "00_framework",
        help="framework artifact directory",
    )
    parser.add_argument(
        "--overwrite-status",
        action="store_true",
        help="reset the stage status CSV; do not use after research progress exists",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_research_plan(DEFAULT_RESEARCH_PLAN)
    paths = write_framework_artifacts(
        args.report_dir,
        plan=DEFAULT_RESEARCH_PLAN,
        overwrite_status=bool(args.overwrite_status),
    )

    config = DEFAULT_RESEARCH_PLAN.config
    print(f"[framework] plan={DEFAULT_RESEARCH_PLAN.plan_id} version={DEFAULT_RESEARCH_PLAN.version}")
    print(
        f"[framework] input={config.input_bar_seconds}s decision={config.decision_interval_seconds}s "
        f"symbol={config.symbol}"
    )
    for stage in DEFAULT_RESEARCH_PLAN.stages:
        dependencies = ",".join(stage.depends_on) or "-"
        print(f"[stage] {stage.stage_id} owner={stage.owner} depends_on={dependencies} name={stage.name}")
    for label, path in paths.items():
        print(f"[report] {label}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
