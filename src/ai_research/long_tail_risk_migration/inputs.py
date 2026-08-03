#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validated frozen inputs for R03.4.2.10."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from src.ai_research.long_tail_dynamic_risk_release.config import DynamicRiskReleaseConfig
from src.ai_research.long_tail_dynamic_risk_release.inputs import load_dynamic_risk_release_inputs

from .config import RiskMigrationConfig


@dataclass(frozen=True)
class RiskMigrationInputs:
    manifest_2_8a: dict[str, object]
    manifest_2_8b: dict[str, object]
    manifest_2_9: dict[str, object]
    folds: pd.DataFrame
    structural: pd.DataFrame
    fixed_6h: pd.DataFrame
    p0_summary: pd.DataFrame
    p0_trades: pd.DataFrame
    source_2_9_gate: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except EmptyDataError:
        return pd.DataFrame()


def load_risk_migration_inputs(config: RiskMigrationConfig) -> RiskMigrationInputs:
    config.validate()

    prior_config = DynamicRiskReleaseConfig(
        source_2_8a_report_dir=config.source_2_8a_report_dir,
        source_2_8b_report_dir=config.source_2_8b_report_dir,
        report_dir=config.source_2_9_report_dir,
    )
    prior = load_dynamic_risk_release_inputs(prior_config)

    root = config.source_2_9_path
    required = (
        "00_run_manifest.json",
        "03_protection_summary.csv",
        "08_account_policy_summary.csv",
        "11_policy_gate.csv",
        "13_causal_audit.csv",
        "14_failures.csv",
        "99_decision.md",
    )
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"R03.4.2.9 source report missing files: {missing}")

    manifest_2_9 = json.loads((root / "00_run_manifest.json").read_text(encoding="utf-8"))
    if manifest_2_9.get("stage") != "R03.4.2.9":
        raise RuntimeError(f"unexpected 2.9 source stage: {manifest_2_9.get('stage')}")
    source_config = manifest_2_9.get("config", {})
    if not isinstance(source_config, dict):
        raise RuntimeError("R03.4.2.9 manifest config is invalid")
    if pd.Timestamp(str(source_config.get("research_end"))) >= pd.Timestamp(config.sealed_holdout_start):
        raise RuntimeError("R03.4.2.9 opened the sealed 2026 period")
    if abs(float(source_config.get("disaster_stop_distance", -1.0)) - 0.03) > 1e-12:
        raise RuntimeError("R03.4.2.9 did not preserve the frozen 3% disaster stop")

    failures = _read_csv(root / "14_failures.csv")
    if not failures.empty:
        raise RuntimeError(f"R03.4.2.9 source contains failures: {failures.to_dict('records')[:3]}")
    causal = _read_csv(root / "13_causal_audit.csv")
    if causal.empty or not causal["status"].astype(str).eq("PASS").all():
        raise RuntimeError("R03.4.2.9 causal audit is incomplete or red")
    decision_text = (root / "99_decision.md").read_text(encoding="utf-8")
    if "FAIL_NO_ROBUST_STRUCTURE_PROTECTION" not in decision_text:
        raise RuntimeError("R03.4.2.10 expects the frozen 2.9 structure-stop failure")

    structural = prior.structural.copy()
    for column in ("decision_time", "entry_time", "exit_time"):
        structural[column] = pd.to_datetime(structural[column])
    structural["event_id"] = structural["event_id"].astype(str)
    structural["delay_minutes"] = structural["delay_minutes"].astype(int)

    p0_summary = prior.p0_summary.copy()
    p0_trades = prior.p0_trades.copy()
    gate = _read_csv(root / "11_policy_gate.csv")

    return RiskMigrationInputs(
        manifest_2_8a=prior.manifest_2_8a,
        manifest_2_8b=prior.manifest_2_8b,
        manifest_2_9=manifest_2_9,
        folds=prior.folds.copy(),
        structural=structural,
        fixed_6h=prior.fixed_6h.copy(),
        p0_summary=p0_summary,
        p0_trades=p0_trades,
        source_2_9_gate=gate,
    )
