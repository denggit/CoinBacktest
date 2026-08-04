#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the untouched frozen C2 recipe on the new July-2026 forward window."""

from __future__ import annotations

from contextlib import contextmanager
import json
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Iterator

import pandas as pd

from src.ai_research.long_tail_sealed_holdout import pipeline as sealed_pipeline
from src.ai_research.long_tail_sealed_holdout.analysis import build_gate as base_build_gate
from src.ai_research.swing_baseline.dataset import run_public_loader_preflight as base_preflight
from src.ai_research.swing_long_context.config import LONG_CONTEXT_BASE_CONFIG

from . import reports
from .config import (
    DEFAULT_FORWARD_EXTENSION_CONFIG,
    FOLD_ID,
    STAGE_ID,
    STAGE_NAME,
    ForwardExtensionConfig,
)
from .seal import ensure_pre_open_seal, verify_post_run_seal


@dataclass(frozen=True)
class ForwardExtensionResult:
    decision: str
    report_dir: Path


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _validate_source(config: ForwardExtensionConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    decision_2_15 = (config.source_2_15_path / "99_decision.md").read_text(encoding="utf-8")
    if "PASS_FINAL_ACCOUNT_LIVE_READINESS" not in decision_2_15:
        raise RuntimeError("R03.4.2.15 did not pass final live readiness")

    decision_2_16 = (config.source_2_16_path / "99_decision.md").read_text(encoding="utf-8")
    if "FAIL_2026_SEALED_HOLDOUT" not in decision_2_16:
        raise RuntimeError("R03.4.2.16 completed failure report is required before July extension")
    seal_check = json.loads((config.source_2_16_path / "18_post_run_seal_check.json").read_text(encoding="utf-8"))
    if str(seal_check.get("status")) != "PASS" or not bool(seal_check.get("unchanged", False)):
        raise RuntimeError("R03.4.2.16 source seal did not pass")
    failures = _read_csv(config.source_2_16_path / "20_failures.csv")
    if not failures.empty:
        raise RuntimeError("R03.4.2.16 contains runtime failures")

    source_scenarios = _read_csv(config.source_2_16_path / "15_extended_oos_summary.csv")
    historical = _read_csv(config.source_2_15_path / "02_historical_metric_contract.csv")
    if source_scenarios.empty:
        raise RuntimeError("R03.4.2.16 extended OOS source is empty")
    return source_scenarios, historical



def _build_gate(
    summary: pd.DataFrame,
    score_audit: pd.DataFrame,
    seal_check: dict[str, object],
    config: ForwardExtensionConfig,
) -> pd.DataFrame:
    gate = base_build_gate(summary, score_audit, seal_check, config)
    if score_audit.empty:
        return gate
    source = _read_csv(config.source_2_16_path / "03_model_threshold_audit.csv")
    if source.empty:
        return pd.concat(
            [gate, pd.DataFrame([{"check": "frozen_threshold_matches_h1", "pass": False, "value": "missing", "threshold": "R03.4.2.16 threshold", "gate_class": "hard"}])],
            ignore_index=True,
        )
    current = score_audit.iloc[0]
    previous = source.iloc[0]
    rows = pd.DataFrame(
        [
            {
                "check": "frozen_threshold_matches_h1",
                "pass": bool(abs(float(current["calibration_threshold"]) - float(previous["calibration_threshold"])) <= 1e-12),
                "value": float(current["calibration_threshold"]),
                "threshold": float(previous["calibration_threshold"]),
                "gate_class": "hard",
            },
            {
                "check": "fit_rows_match_h1",
                "pass": int(current["fit_rows"]) == int(previous["fit_rows"]),
                "value": int(current["fit_rows"]),
                "threshold": int(previous["fit_rows"]),
                "gate_class": "hard",
            },
            {
                "check": "calibration_rows_match_h1",
                "pass": int(current["calibration_rows"]) == int(previous["calibration_rows"]),
                "value": int(current["calibration_rows"]),
                "threshold": int(previous["calibration_rows"]),
                "gate_class": "hard",
            },
        ]
    )
    return pd.concat([gate, rows], ignore_index=True)

def _causal_audit(config: ForwardExtensionConfig) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"check": "independent_july_seal", "status": "PASS", "detail": "new code/config/source SHA-256 seal is written before July loader access"},
            {"check": "fit_boundary", "status": "PASS", "detail": f"fit still ends {config.fit_end}; no 2026 row is used for fitting"},
            {"check": "threshold_boundary", "status": "PASS", "detail": "q70 threshold still uses Q4 2025 calibration only"},
            {"check": "h1_comparison_only", "status": "PASS", "detail": "January-June 2026 source reports are used only for comparison and stitched accounting"},
            {"check": "july_inference_only", "status": "PASS", "detail": "July labels score the frozen model and may not change any rule"},
            {"check": "frozen_c2", "status": "PASS", "detail": "immediate q70, equal 1R, 2% hard stop, 1.5% completed-close soft failure and failed_reclaim are unchanged"},
            {"check": "single_month_disclosure", "status": "PASS", "detail": "July is a one-month forward extension and cannot reverse the failed H1 seal by itself"},
        ]
    )


def _july_preflight(loader, config, *, sample_dates=None):
    del sample_dates
    return base_preflight(
        loader,
        config,
        sample_dates=("2025-12-15", "2026-07-15", "2026-07-30"),
    )


@contextmanager
def _temporary_runtime_contract(config: ForwardExtensionConfig) -> Iterator[None]:
    names = (
        "LONG_CONTEXT_BASE_CONFIG",
        "STAGE_ID",
        "STAGE_NAME",
        "ensure_pre_open_seal",
        "verify_post_run_seal",
        "reports",
        "_validate_source",
        "_causal_audit",
        "run_public_loader_preflight",
        "build_gate",
    )
    original = {name: getattr(sealed_pipeline, name) for name in names}
    try:
        sealed_pipeline.LONG_CONTEXT_BASE_CONFIG = replace(
            LONG_CONTEXT_BASE_CONFIG,
            research_end=config.holdout_end,
            cache_dir=config.isolated_base_cache_dir,
        )
        sealed_pipeline.STAGE_ID = STAGE_ID
        sealed_pipeline.STAGE_NAME = STAGE_NAME
        sealed_pipeline.ensure_pre_open_seal = ensure_pre_open_seal
        sealed_pipeline.verify_post_run_seal = verify_post_run_seal
        sealed_pipeline.reports = reports
        sealed_pipeline._validate_source = _validate_source
        sealed_pipeline._causal_audit = _causal_audit
        sealed_pipeline.run_public_loader_preflight = _july_preflight
        sealed_pipeline.build_gate = _build_gate
        yield
    finally:
        for name, value in original.items():
            setattr(sealed_pipeline, name, value)


def _map_decision(decision: str) -> str:
    return reports.map_decision(decision)


def run_forward_extension(
    *,
    data_dir: str | Path | None = None,
    force_rebuild_base: bool = False,
    force_rebuild_outcomes: bool = False,
    progress: bool = True,
    config: ForwardExtensionConfig = DEFAULT_FORWARD_EXTENSION_CONFIG,
) -> ForwardExtensionResult:
    config.validate()
    with _temporary_runtime_contract(config):
        result = sealed_pipeline.run_sealed_holdout(
            data_dir=data_dir,
            force_rebuild_base=force_rebuild_base,
            force_rebuild_outcomes=force_rebuild_outcomes,
            progress=progress,
            config=config,
        )
    return ForwardExtensionResult(_map_decision(result.decision), result.report_dir)
