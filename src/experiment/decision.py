#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Standard decision artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .models import utc_now_iso


def build_decision(
    *,
    experiment_id: str,
    stage: str,
    status: str,
    reason: str,
    next_action: str,
    metrics: Mapping[str, Any] | None = None,
    reports: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "stage": stage,
        "status": status,
        "reason": reason,
        "next_action": next_action,
        "metrics": dict(metrics or {}),
        "reports": dict(reports or {}),
        "created_at": utc_now_iso(),
    }


def write_decision(
    out_dir: str | Path,
    *,
    experiment_id: str,
    stage: str,
    status: str,
    reason: str,
    next_action: str,
    metrics: Mapping[str, Any] | None = None,
    reports: Mapping[str, str] | None = None,
    filename: str = "09_decision.json",
) -> Path:
    target_dir = Path(out_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename
    payload = build_decision(
        experiment_id=experiment_id,
        stage=stage,
        status=status,
        reason=reason,
        next_action=next_action,
        metrics=metrics,
        reports=reports,
    )
    target.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    return target
