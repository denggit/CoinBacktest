#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R03.2 wrapper over the exact-path engine with a frozen long-context feature profile."""

from __future__ import annotations

from pathlib import Path

from src.ai_research.swing_entry_mvp.pipeline import SwingEntryMvpResult, run_pipeline as run_entry_pipeline

from .config import (
    DEFAULT_SWING_LONG_CONTEXT_CONFIG,
    FEATURE_PROFILE,
    STAGE_ID,
    STAGE_NAME,
    validate_long_context_contract,
)
from .reports import write_reports


def run_pipeline(
    *,
    force_rebuild_exact_labels: bool = False,
    force_rebuild_long_context_cache: bool = False,
    data_dir: str | Path | None = None,
    progress: bool = True,
) -> SwingEntryMvpResult:
    validate_long_context_contract(DEFAULT_SWING_LONG_CONTEXT_CONFIG)
    return run_entry_pipeline(
        config=DEFAULT_SWING_LONG_CONTEXT_CONFIG,
        force_rebuild_exact_labels=force_rebuild_exact_labels,
        force_rebuild_base_cache=force_rebuild_long_context_cache,
        data_dir=data_dir,
        progress=progress,
        feature_profile=FEATURE_PROFILE,
        stage_id=STAGE_ID,
        stage_name=STAGE_NAME,
        report_writer=write_reports,
        pass_decision="PASS_SWING_LONG_CONTEXT_MVP",
    )
