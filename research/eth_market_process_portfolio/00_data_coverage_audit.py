#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit local data coverage before ETH market-process research starts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_RESEARCH_ROOT = _PROJECT_ROOT / "research"
for _path in (_PROJECT_ROOT, _RESEARCH_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from eth_market_process_portfolio.common.config import DEFAULT_REPORT_ROOT  # noqa: E402
from eth_market_process_portfolio.common.coverage import (  # noqa: E402
    audit_local_coverage,
    coverage_frame,
    overall_gate,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=_PROJECT_ROOT / "data")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_ROOT / "00_data_coverage_audit")
    parser.add_argument(
        "--fail-on-incomplete-core",
        action="store_true",
        help="return exit code 2 when mandatory full-history inputs are blocked or partial",
    )
    return parser.parse_args(argv)


def _markdown(frame, gate: str) -> str:
    lines = [
        "# ETH Market Process Portfolio — Local Data Coverage Audit",
        "",
        f"Overall mandatory-data gate: **{gate}**",
        "",
        "This audit is local-only. `WINDOW_ONLY` means the source may be used only inside its actual overlapping coverage; it does not qualify as full-history evidence.",
        "",
        "| Module | Dataset | Table | Rows | Start | End | Status | Reason |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            f"| {row.module} | {row.dataset} | {row.table or '-'} | {int(row.rows):,} | "
            f"{row.start or '-'} | {row.end or '-'} | {row.status} | {row.reason} |"
        )
    lines.extend(
        [
            "",
            "## Gate semantics",
            "",
            "- `READY`: required full-history coverage is present.",
            "- `PARTIAL`: mandatory input exists but does not cover the frozen research window.",
            "- `BLOCKED`: mandatory database/table is absent or empty.",
            "- `WINDOW_ONLY`: optional/short-history source is usable only on its real overlap window.",
            "- `MISSING_OPTIONAL`: optional source is unavailable and dependent studies must not run.",
        ]
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = audit_local_coverage(data_dir=args.data_dir)
    frame = coverage_frame(records)
    gate = overall_gate(records)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.report_dir / "coverage.csv"
    md_path = args.report_dir / "coverage.md"
    frame.to_csv(csv_path, index=False)
    md_path.write_text(_markdown(frame, gate), encoding="utf-8")

    print(f"[coverage] mandatory gate={gate}")
    for row in frame.itertuples(index=False):
        print(
            f"[{row.module}] {row.dataset} table={row.table or '-'} rows={int(row.rows):,} "
            f"start={row.start or '-'} end={row.end or '-'} status={row.status}"
        )
    print(f"[report] {md_path}")
    print(f"[report] {csv_path}")

    if args.fail_on_incomplete_core and gate != "READY":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
