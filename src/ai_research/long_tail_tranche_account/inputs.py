#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validated R03.4.2.8A artifact loading for R03.4.2.8B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd
from pandas.errors import EmptyDataError

from .config import TrancheAccountConfig


@dataclass(frozen=True)
class TrancheAccountInputs:
    manifest: dict[str, object]
    folds: pd.DataFrame
    structural: pd.DataFrame
    fixed_6h: pd.DataFrame
    p0_baseline: pd.DataFrame
    source_atlas: pd.DataFrame
    source_gate: pd.DataFrame
    source_causal_audit: pd.DataFrame


_REQUIRED_FILES = (
    "00_run_manifest.json",
    "06_occupied_signal_atlas.csv",
    "12_tranche_eligibility_gate.csv",
    "13_causal_audit.csv",
    "14_failures.csv",
    "15_p0_failed_reclaim_trades.csv",
    "16_standalone_signal_outcomes.csv",
    "99_decision.md",
)


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except EmptyDataError:
        return pd.DataFrame()


def load_tranche_account_inputs(config: TrancheAccountConfig) -> TrancheAccountInputs:
    """Load the frozen 2.8A outputs and reject incomplete or drifted artifacts."""

    config.validate()
    root = config.source_report_path
    missing = [name for name in _REQUIRED_FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"R03.4.2.8A source report missing files: {missing}")

    manifest = json.loads((root / "00_run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("stage") != "R03.4.2.8A":
        raise RuntimeError(f"unexpected source stage: {manifest.get('stage')}")
    manifest_config = manifest.get("config", {})
    if not isinstance(manifest_config, dict):
        raise RuntimeError("R03.4.2.8A manifest config is invalid")
    if float(manifest_config.get("evaluation_quantile", -1.0)) != 0.70:
        raise RuntimeError("source report is not the frozen q70 opening pool")
    if pd.Timestamp(str(manifest_config.get("research_end"))) >= pd.Timestamp(config.sealed_holdout_start):
        raise RuntimeError("source report opened the sealed 2026 period")

    failures = _read_csv(root / "14_failures.csv")
    if not failures.empty:
        raise RuntimeError(f"R03.4.2.8A source contains failures: {failures.to_dict('records')[:3]}")

    causal = _read_csv(root / "13_causal_audit.csv")
    if causal.empty or not causal["status"].astype(str).eq("PASS").all():
        raise RuntimeError("R03.4.2.8A causal audit is incomplete or red")

    outcomes = _read_csv(root / "16_standalone_signal_outcomes.csv")
    required_columns = {
        "event_id",
        "fold_id",
        "decision_time",
        "entry_time",
        "exit_time",
        "delay_minutes",
        "signal_quantile",
        "score",
        "score_percentile",
        "score_tier",
        "entry_price",
        "exit_price",
        "gross_return",
        "mfe",
        "mae",
        "holding_minutes",
        "exit_reason",
        "is_censored",
        "standalone_outcome",
    }
    missing_columns = sorted(required_columns - set(outcomes.columns))
    if missing_columns:
        raise RuntimeError(f"standalone outcome columns missing: {missing_columns}")

    for column in ("decision_time", "entry_time", "exit_time"):
        outcomes[column] = pd.to_datetime(outcomes[column])
    outcomes["event_id"] = outcomes["event_id"].astype(str)
    outcomes["fold_id"] = outcomes["fold_id"].astype(str)
    outcomes["delay_minutes"] = outcomes["delay_minutes"].astype(int)

    structural = outcomes.loc[outcomes["standalone_outcome"] == "failed_reclaim"].copy()
    fixed = outcomes.loc[outcomes["standalone_outcome"] == "fixed_6h"].copy()
    if structural.empty or fixed.empty:
        raise RuntimeError("source report does not contain both structural and fixed-6h outcomes")
    duplicates = structural.duplicated(["fold_id", "delay_minutes", "event_id"], keep=False)
    if duplicates.any():
        raise RuntimeError("structural outcomes contain duplicate event rows")

    folds = pd.DataFrame(manifest.get("folds", []))
    required_fold_columns = {"fold_id", "test_start", "test_end"}
    if folds.empty or not required_fold_columns.issubset(folds.columns):
        raise RuntimeError("source manifest fold definitions are incomplete")
    folds["test_start"] = pd.to_datetime(folds["test_start"])
    folds["test_end"] = pd.to_datetime(folds["test_end"])
    if set(folds["fold_id"]) != {"WF_2024", "WF_2025"}:
        raise RuntimeError("R03.4.2.8B requires the same WF_2024 and WF_2025 folds")

    p0_baseline = _read_csv(root / "15_p0_failed_reclaim_trades.csv")
    for column in ("decision_time", "entry_time", "exit_time"):
        if column in p0_baseline.columns:
            p0_baseline[column] = pd.to_datetime(p0_baseline[column])
    atlas = _read_csv(root / "06_occupied_signal_atlas.csv")
    for column in ("decision_time", "new_entry_time", "root_entry_time", "root_exit_time"):
        if column in atlas.columns:
            atlas[column] = pd.to_datetime(atlas[column])
    gate = _read_csv(root / "12_tranche_eligibility_gate.csv")

    return TrancheAccountInputs(
        manifest=manifest,
        folds=folds,
        structural=structural.sort_values(["fold_id", "delay_minutes", "entry_time", "event_id"]).reset_index(drop=True),
        fixed_6h=fixed.sort_values(["fold_id", "delay_minutes", "entry_time", "event_id"]).reset_index(drop=True),
        p0_baseline=p0_baseline,
        source_atlas=atlas,
        source_gate=gate,
        source_causal_audit=causal,
    )
