#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validated frozen inputs for R03.4.2.9."""

from __future__ import annotations

from dataclasses import dataclass
import json

import pandas as pd
from pandas.errors import EmptyDataError

from src.ai_research.long_tail_tranche_account.config import TrancheAccountConfig
from src.ai_research.long_tail_tranche_account.inputs import load_tranche_account_inputs

from .config import DynamicRiskReleaseConfig


@dataclass(frozen=True)
class DynamicRiskReleaseInputs:
    manifest_2_8a: dict[str, object]
    manifest_2_8b: dict[str, object]
    folds: pd.DataFrame
    structural: pd.DataFrame
    fixed_6h: pd.DataFrame
    p0_summary: pd.DataFrame
    p0_trades: pd.DataFrame
    source_2_8b_gate: pd.DataFrame


def _read_csv(path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except EmptyDataError:
        return pd.DataFrame()


def load_dynamic_risk_release_inputs(config: DynamicRiskReleaseConfig) -> DynamicRiskReleaseInputs:
    config.validate()

    # Reuse the strict 2.8A loader instead of duplicating its causal and schema
    # validation. The default paths are the same frozen source artifacts.
    tranche_config = TrancheAccountConfig(
        source_report_dir=config.source_2_8a_report_dir,
        report_dir=config.source_2_8b_report_dir,
    )
    source_2_8a = load_tranche_account_inputs(tranche_config)

    root = config.source_2_8b_path
    required = (
        "00_run_manifest.json",
        "05_account_policy_summary.csv",
        "06_account_trades.csv",
        "09_policy_gate.csv",
        "11_causal_audit.csv",
        "12_failures.csv",
        "99_decision.md",
    )
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"R03.4.2.8B source report missing files: {missing}")

    manifest_2_8b = json.loads((root / "00_run_manifest.json").read_text(encoding="utf-8"))
    if manifest_2_8b.get("stage") != "R03.4.2.8B":
        raise RuntimeError(f"unexpected 2.8B source stage: {manifest_2_8b.get('stage')}")
    source_config = manifest_2_8b.get("config", {})
    if not isinstance(source_config, dict):
        raise RuntimeError("R03.4.2.8B manifest config is invalid")
    if pd.Timestamp(str(source_config.get("research_end"))) >= pd.Timestamp(config.sealed_holdout_start):
        raise RuntimeError("R03.4.2.8B opened the sealed 2026 period")
    if float(source_config.get("disaster_stop_distance", -1.0)) != 0.03:
        raise RuntimeError("R03.4.2.8B did not use the frozen 3% disaster distance")

    failures = _read_csv(root / "12_failures.csv")
    if not failures.empty:
        raise RuntimeError(f"R03.4.2.8B source contains failures: {failures.to_dict('records')[:3]}")
    causal = _read_csv(root / "11_causal_audit.csv")
    if causal.empty or not causal["status"].astype(str).eq("PASS").all():
        raise RuntimeError("R03.4.2.8B causal audit is incomplete or red")

    p0_summary = _read_csv(root / "05_account_policy_summary.csv")
    p0_summary = p0_summary.loc[p0_summary["policy"] == "P0_single_1R"].copy()
    if p0_summary.empty:
        raise RuntimeError("R03.4.2.8B P0 baseline summary is missing")
    p0_trades = _read_csv(root / "06_account_trades.csv")
    p0_trades = p0_trades.loc[p0_trades["policy"] == "P0_single_1R"].copy()
    for column in ("decision_time", "entry_time", "exit_time"):
        if column in p0_trades.columns:
            p0_trades[column] = pd.to_datetime(p0_trades[column])
    gate = _read_csv(root / "09_policy_gate.csv")

    structural = source_2_8a.structural.copy()
    for column in ("decision_time", "entry_time", "exit_time"):
        structural[column] = pd.to_datetime(structural[column])
    structural["event_id"] = structural["event_id"].astype(str)
    structural["delay_minutes"] = structural["delay_minutes"].astype(int)

    return DynamicRiskReleaseInputs(
        manifest_2_8a=source_2_8a.manifest,
        manifest_2_8b=manifest_2_8b,
        folds=source_2_8a.folds.copy(),
        structural=structural,
        fixed_6h=source_2_8a.fixed_6h.copy(),
        p0_summary=p0_summary,
        p0_trades=p0_trades,
        source_2_8b_gate=gate,
    )
