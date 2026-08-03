#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic framework snapshot writers for ETH AI research."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.experiment.models import utc_now_iso

from .models import ResearchPlan
from .plan import DEFAULT_RESEARCH_PLAN, validate_research_plan


STATUS_FIELDS = (
    "stage_id",
    "name",
    "owner",
    "status",
    "depends_on",
    "decision",
    "updated_at",
)


def initial_stage_status(plan: ResearchPlan = DEFAULT_RESEARCH_PLAN) -> list[dict[str, str]]:
    """Build the first status board without claiming unexecuted research passed."""
    validate_research_plan(plan)
    now = utc_now_iso()
    rows: list[dict[str, str]] = []
    for index, stage in enumerate(plan.stages):
        rows.append(
            {
                "stage_id": stage.stage_id,
                "name": stage.name,
                "owner": stage.owner,
                "status": "READY" if index == 0 else "BLOCKED",
                "depends_on": ",".join(stage.depends_on),
                "decision": "",
                "updated_at": now,
            }
        )
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_status_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _summary_markdown(plan: ResearchPlan) -> str:
    config = plan.config
    lines = [
        f"# {plan.title} — Framework Snapshot",
        "",
        f"- Plan ID: `{plan.plan_id}`",
        f"- Version: `{plan.version}`",
        f"- Symbol: `{config.symbol}`",
        f"- Input: `{config.input_bar_seconds}s` trade bars",
        f"- First decision cadence: `{config.decision_interval_seconds}s`",
        f"- Research window: `{config.research_start}` → `{config.research_end}`",
        f"- Sealed holdout starts: `{config.sealed_holdout_start}`",
        f"- Default round-trip fee: `{config.round_trip_fee_rate:.4%}`",
        f"- Authoritative plan: `{plan.plan_doc}`",
        "",
        "| Stage | Owner | Name | Depends on |",
        "|---|---|---|---|",
    ]
    for stage in plan.stages:
        lines.append(
            f"| {stage.stage_id} | {stage.owner} | {stage.name} | {', '.join(stage.depends_on) or '-'} |"
        )
    lines.extend(
        [
            "",
            "## Operating rule",
            "",
            "Only one stage should be treated as the active research gate. A later stage must not be used to rescue a failed earlier stage. Failed stages require an explicit stop, redesign, or rejection decision.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_framework_artifacts(
    report_dir: str | Path,
    *,
    plan: ResearchPlan = DEFAULT_RESEARCH_PLAN,
    overwrite_status: bool = False,
) -> dict[str, Path]:
    """Write plan snapshot, status board, and readable summary.

    The status CSV is preserved by default so rerunning the initializer cannot
    erase research progress. Use ``overwrite_status=True`` only for an explicit
    reset of a newly created programme.
    """
    validate_research_plan(plan)
    target = Path(report_dir)
    target.mkdir(parents=True, exist_ok=True)

    plan_path = target / "00_plan_snapshot.json"
    status_path = target / "01_stage_status.csv"
    summary_path = target / "02_framework_summary.md"
    _write_json(plan_path, plan.to_dict())
    if overwrite_status or not status_path.exists():
        _write_status_csv(status_path, initial_stage_status(plan))
    summary_path.write_text(_summary_markdown(plan), encoding="utf-8")
    return {"plan": plan_path, "status": status_path, "summary": summary_path}
