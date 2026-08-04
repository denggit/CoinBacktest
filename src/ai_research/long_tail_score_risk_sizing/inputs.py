#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validated R03.4.2.12 report inputs for score-risk sizing."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from .config import ScoreRiskConfig


@dataclass(frozen=True)
class ScoreRiskInputs:
    manifest: dict[str, object]
    selected_events: pd.DataFrame
    source_cycles: pd.DataFrame
    source_legs: pd.DataFrame
    source_summary: pd.DataFrame
    source_c2_summary: pd.DataFrame


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except EmptyDataError:
        return pd.DataFrame()


def _times(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = pd.to_datetime(result[column])
    return result


def load_score_risk_inputs(config: ScoreRiskConfig) -> ScoreRiskInputs:
    config.validate()
    root = config.source_path
    required = (
        "00_run_manifest.json",
        "04_selected_p0_cycles.csv",
        "07_account_cycles.csv",
        "08_account_legs.csv",
        "11_policy_summary.csv",
        "12_policy_gate.csv",
        "13_causal_audit.csv",
        "15_failures.csv",
        "99_decision.md",
    )
    missing = [name for name in required if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"R03.4.2.12 source missing files: {missing}")
    manifest = json.loads((root / "00_run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("stage") != "R03.4.2.12":
        raise RuntimeError(f"unexpected source stage: {manifest.get('stage')}")
    decision = (root / "99_decision.md").read_text(encoding="utf-8")
    if "PASS_REAL_1R_TAIL_COMPRESSION_CANDIDATE" not in decision:
        raise RuntimeError("R03.4.2.13 requires the passed C2 source decision")
    failures = _read_csv(root / "15_failures.csv")
    if not failures.empty:
        raise RuntimeError(f"R03.4.2.12 source contains failures: {failures.head(3).to_dict('records')}")
    causal = _read_csv(root / "13_causal_audit.csv")
    if causal.empty or not causal["status"].astype(str).eq("PASS").all():
        raise RuntimeError("R03.4.2.12 causal audit is incomplete")

    selected = _times(_read_csv(root / "04_selected_p0_cycles.csv"), ("decision_time", "entry_time", "exit_time"))
    cycles = _times(_read_csv(root / "07_account_cycles.csv"), ("decision_time", "entry_time", "source_exit_time", "exit_time"))
    legs = _times(_read_csv(root / "08_account_legs.csv"), ("entry_time", "exit_time"))
    summary = _read_csv(root / "11_policy_summary.csv")
    if selected.empty or cycles.empty or legs.empty or summary.empty:
        raise RuntimeError("R03.4.2.12 source tables are empty")
    required_tiers = {"q70_to_q80", "q80_to_q90", "q90_plus"}
    if not required_tiers.issubset(set(selected["score_tier"].astype(str).unique())):
        raise RuntimeError("source score tiers are incomplete")
    source_cycles = cycles.loc[cycles["policy"].astype(str).eq(config.source_policy)].copy()
    source_legs = legs.loc[legs["policy"].astype(str).eq(config.source_policy)].copy()
    source_summary = summary.loc[summary["policy"].astype(str).eq(config.source_policy)].copy()
    if source_cycles.empty or source_legs.empty or source_summary.empty:
        raise RuntimeError("passed C2 source policy is missing")
    keys = ["event_id", "fold_id", "delay_minutes"]
    tier_map = selected[keys + ["score", "score_percentile", "score_tier"]].drop_duplicates(keys)
    source_cycles = source_cycles.drop(columns=[c for c in ("score",) if c in source_cycles.columns]).merge(
        tier_map, on=keys, how="left", validate="many_to_one"
    )
    if source_cycles["score_tier"].isna().any():
        raise RuntimeError("some C2 cycles could not be mapped to frozen score tiers")
    return ScoreRiskInputs(
        manifest=manifest,
        selected_events=selected.reset_index(drop=True),
        source_cycles=source_cycles.reset_index(drop=True),
        source_legs=source_legs.reset_index(drop=True),
        source_summary=summary.reset_index(drop=True),
        source_c2_summary=source_summary.reset_index(drop=True),
    )
