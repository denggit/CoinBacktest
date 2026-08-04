#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validated frozen inputs for R03.4.2.11."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from src.ai_research.long_tail_risk_migration.config import RiskMigrationConfig
from src.ai_research.long_tail_risk_migration.inputs import load_risk_migration_inputs

from .config import StagedExecutionConfig


@dataclass(frozen=True)
class StagedExecutionInputs:
    manifest_2_10: dict[str, object]
    folds: pd.DataFrame
    selected_events: pd.DataFrame
    structure_timeline: pd.DataFrame
    source_p0_summary: pd.DataFrame
    source_p0_trades: pd.DataFrame
    source_gate: pd.DataFrame


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


def load_staged_execution_inputs(config: StagedExecutionConfig) -> StagedExecutionInputs:
    config.validate()
    root = config.source_2_10_path
    required = (
        "00_run_manifest.json",
        "03_soft_structure_timeline.csv",
        "08_account_trades.csv",
        "10_account_policy_summary.csv",
        "11_policy_gate.csv",
        "12_causal_audit.csv",
        "14_failures.csv",
        "99_decision.md",
    )
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"R03.4.2.10 source report missing files: {missing}")

    manifest = json.loads((root / "00_run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("stage") != "R03.4.2.10":
        raise RuntimeError(f"unexpected 2.10 source stage: {manifest.get('stage')}")
    source_config = manifest.get("config", {})
    if not isinstance(source_config, dict):
        raise RuntimeError("R03.4.2.10 manifest config is invalid")
    if pd.Timestamp(str(source_config.get("research_end"))) >= pd.Timestamp(config.sealed_holdout_start):
        raise RuntimeError("R03.4.2.10 opened the sealed 2026 period")
    if abs(float(source_config.get("disaster_stop_distance", -1.0)) - 0.03) > 1e-12:
        raise RuntimeError("R03.4.2.10 did not preserve the 3% disaster floor")

    failures = _read_csv(root / "14_failures.csv")
    if not failures.empty:
        raise RuntimeError(f"R03.4.2.10 source contains failures: {failures.to_dict('records')[:3]}")
    causal = _read_csv(root / "12_causal_audit.csv")
    if causal.empty or not causal["status"].astype(str).eq("PASS").all():
        raise RuntimeError("R03.4.2.10 causal audit is incomplete or red")
    decision_text = (root / "99_decision.md").read_text(encoding="utf-8")
    if "FAIL_NO_ROBUST_PARTIAL_OR_MIGRATION" not in decision_text:
        raise RuntimeError("R03.4.2.11 expects the frozen 2.10 migration failure")

    # Reuse only src-level validated loaders. Research scripts never import one another.
    prior_config = RiskMigrationConfig(
        source_2_8a_report_dir=config.source_2_8a_report_dir,
        source_2_8b_report_dir=config.source_2_8b_report_dir,
        source_2_9_report_dir=config.source_2_9_report_dir,
        report_dir=config.source_2_10_report_dir,
    )
    prior = load_risk_migration_inputs(prior_config)
    structural = _normalize_times(prior.structural, ("decision_time", "entry_time", "exit_time"))
    structural["event_id"] = structural["event_id"].astype(str)
    structural["delay_minutes"] = structural["delay_minutes"].astype(int)

    source_trades = _read_csv(root / "08_account_trades.csv")
    source_trades = _normalize_times(source_trades, ("decision_time", "entry_time", "source_exit_time", "exit_time"))
    source_p0_trades = source_trades.loc[source_trades["policy"].astype(str).eq("P0_single_1R")].copy()
    if source_p0_trades.empty:
        raise RuntimeError("R03.4.2.10 P0 source trades are empty")
    source_p0_trades["event_id"] = source_p0_trades["event_id"].astype(str)
    source_p0_trades["delay_minutes"] = source_p0_trades["delay_minutes"].astype(int)

    selected_keys = source_p0_trades[["fold_id", "delay_minutes", "event_id"]].drop_duplicates()
    selected_events = structural.merge(
        selected_keys,
        on=["fold_id", "delay_minutes", "event_id"],
        how="inner",
        validate="many_to_one",
    )
    selected_events = selected_events.sort_values(
        ["fold_id", "delay_minutes", "entry_time", "decision_time", "score"],
        ascending=[True, True, True, True, False],
    ).drop_duplicates(["fold_id", "delay_minutes", "event_id"], keep="first")
    if len(selected_events) != len(selected_keys):
        missing_count = len(selected_keys) - len(selected_events)
        raise RuntimeError(f"failed to reconstruct {missing_count} frozen P0 events")

    timeline = _read_csv(root / "03_soft_structure_timeline.csv")
    timeline = _normalize_times(
        timeline,
        ("decision_time", "entry_time", "source_exit_time", "structure_close_time", "effective_time"),
    )
    timeline["event_id"] = timeline["event_id"].astype(str)
    timeline["delay_minutes"] = timeline["delay_minutes"].astype(int)
    timeline = timeline.merge(selected_keys, on=["fold_id", "delay_minutes", "event_id"], how="inner")

    source_summary = _read_csv(root / "10_account_policy_summary.csv")
    source_p0_summary = source_summary.loc[source_summary["policy"].astype(str).eq("P0_single_1R")].copy()
    source_gate = _read_csv(root / "11_policy_gate.csv")

    return StagedExecutionInputs(
        manifest_2_10=manifest,
        folds=prior.folds.copy(),
        selected_events=selected_events.reset_index(drop=True),
        structure_timeline=timeline.reset_index(drop=True),
        source_p0_summary=source_p0_summary.reset_index(drop=True),
        source_p0_trades=source_p0_trades.reset_index(drop=True),
        source_gate=source_gate,
    )
