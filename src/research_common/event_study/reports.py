#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report writers for standard event-study outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import EventStudyResult


def write_event_study_report(result: "EventStudyResult", out_dir: str | Path) -> None:
    """Write standard CSV/JSON outputs for one event-study run."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    result.events.to_csv(path / "01_events.csv", index=False)
    result.overview.to_csv(path / "02_overview.csv", index=False)
    result.yearly.to_csv(path / "03_yearly.csv", index=False)
    result.side_stats.to_csv(path / "04_side_stats.csv", index=False)
    result.horizon_stats.to_csv(path / "05_horizon_stats.csv", index=False)
    result.causal_audit.to_csv(path / "08_causal_audit.csv", index=False)
    with (path / "10_meta.json").open("w", encoding="utf-8") as f:
        json.dump(result.meta, f, ensure_ascii=False, indent=2, default=str)
