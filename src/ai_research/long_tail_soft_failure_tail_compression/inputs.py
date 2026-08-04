#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validated frozen inputs for R03.4.2.12."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from src.ai_research.long_tail_staged_execution.config import StagedExecutionConfig
from src.ai_research.long_tail_staged_execution.inputs import load_staged_execution_inputs

from .config import TailCompressionConfig


@dataclass(frozen=True)
class TailCompressionInputs:
    manifest_2_11: dict[str, object]
    folds: pd.DataFrame
    selected_events: pd.DataFrame
    structure_timeline: pd.DataFrame
    source_cycles: pd.DataFrame
    source_legs: pd.DataFrame
    source_summary: pd.DataFrame
    source_p0_summary: pd.DataFrame
    source_f1_summary: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except EmptyDataError:
        return pd.DataFrame()


def _normalize_times(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column])
    return result


def load_tail_compression_inputs(config: TailCompressionConfig) -> TailCompressionInputs:
    config.validate()
    root = config.source_2_11_path
    required = (
        "00_run_manifest.json",
        "04_account_cycles.csv",
        "05_account_legs.csv",
        "08_policy_summary.csv",
        "10_causal_audit.csv",
        "12_failures.csv",
        "99_decision.md",
    )
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"R03.4.2.11 source report missing files: {missing}")

    manifest = json.loads((root / "00_run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("stage") != "R03.4.2.11":
        raise RuntimeError(f"unexpected 2.11 source stage: {manifest.get('stage')}")
    source_config = manifest.get("config", {})
    if not isinstance(source_config, dict):
        raise RuntimeError("R03.4.2.11 manifest config is invalid")
    if pd.Timestamp(str(source_config.get("research_end"))) >= pd.Timestamp(config.sealed_holdout_start):
        raise RuntimeError("R03.4.2.11 opened the sealed 2026 period")

    failures = _read_csv(root / "12_failures.csv")
    if not failures.empty:
        raise RuntimeError(f"R03.4.2.11 source contains failures: {failures.to_dict('records')[:3]}")
    causal = _read_csv(root / "10_causal_audit.csv")
    if causal.empty or not causal["status"].astype(str).eq("PASS").all():
        raise RuntimeError("R03.4.2.11 causal audit is incomplete or red")
    decision_text = (root / "99_decision.md").read_text(encoding="utf-8")
    if "FAIL_NO_ROBUST_STAGED_EXECUTION" not in decision_text:
        raise RuntimeError("R03.4.2.12 expects the frozen 2.11 staged-execution failure")

    prior_config = StagedExecutionConfig(
        source_2_8a_report_dir=config.source_2_8a_report_dir,
        source_2_8b_report_dir=config.source_2_8b_report_dir,
        source_2_9_report_dir=config.source_2_9_report_dir,
        source_2_10_report_dir=config.source_2_10_report_dir,
        report_dir=config.source_2_11_report_dir,
    )
    prior = load_staged_execution_inputs(prior_config)

    cycles = _read_csv(root / "04_account_cycles.csv")
    cycles = _normalize_times(
        cycles,
        ("decision_time", "entry_time", "source_exit_time", "final_exit_time"),
    )
    legs = _read_csv(root / "05_account_legs.csv")
    legs = _normalize_times(legs, ("entry_time", "exit_time"))
    if "cycle_event_id" in legs.columns and "event_id" not in legs.columns:
        legs = legs.rename(columns={"cycle_event_id": "event_id"})
    summary = _read_csv(root / "08_policy_summary.csv")
    if cycles.empty or legs.empty or summary.empty:
        raise RuntimeError("R03.4.2.11 source cycles/legs/summary are empty")

    required_policies = {"P0_single_1R", "F1_soft_failure_1p5"}
    if not required_policies.issubset(set(cycles["policy"].astype(str).unique())):
        raise RuntimeError("R03.4.2.11 source is missing P0 or F1")

    p0 = summary.loc[summary["policy"].astype(str).eq("P0_single_1R")].copy()
    f1 = summary.loc[summary["policy"].astype(str).eq("F1_soft_failure_1p5")].copy()
    return TailCompressionInputs(
        manifest_2_11=manifest,
        folds=prior.folds.copy(),
        selected_events=prior.selected_events.copy(),
        structure_timeline=prior.structure_timeline.copy(),
        source_cycles=cycles.reset_index(drop=True),
        source_legs=legs.reset_index(drop=True),
        source_summary=summary.reset_index(drop=True),
        source_p0_summary=p0.reset_index(drop=True),
        source_f1_summary=f1.reset_index(drop=True),
    )
