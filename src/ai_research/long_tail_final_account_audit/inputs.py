#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Load the passed R03.4.2.14 report chain."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from .config import FinalAccountAuditConfig


@dataclass(frozen=True)
class FinalAuditInputs:
    historical_contract: pd.DataFrame
    cycles: pd.DataFrame
    legs: pd.DataFrame
    daily_equity: pd.DataFrame
    source_summary: pd.DataFrame
    source_causal_audit: pd.DataFrame


def _read_csv(root: Path, name: str, *, dates: tuple[str, ...] = ()) -> pd.DataFrame:
    path = root / name
    if not path.exists():
        raise FileNotFoundError(path)
    try:
        frame = pd.read_csv(path)
    except EmptyDataError:
        return pd.DataFrame()
    for column in dates:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def load_final_audit_inputs(config: FinalAccountAuditConfig) -> FinalAuditInputs:
    root = config.source_2_14_path
    decision_path = root / "99_decision.md"
    if not decision_path.exists():
        raise FileNotFoundError(decision_path)
    decision_text = decision_path.read_text(encoding="utf-8")
    if "PASS_C2_FROZEN_NO_ENTRY_UPLIFT" not in decision_text:
        raise RuntimeError("R03.4.2.14 must pass with immediate C2 frozen before final audit")

    failures = _read_csv(root, "14_failures.csv")
    if not failures.empty:
        raise RuntimeError("R03.4.2.14 contains runtime failures")

    cycles = _read_csv(
        root,
        "07_account_cycles.csv",
        dates=("decision_time", "entry_time", "source_exit_time", "exit_time"),
    )
    legs = _read_csv(root, "08_account_legs.csv", dates=("entry_time", "exit_time"))
    daily = _read_csv(root, "09_daily_equity.csv", dates=("date",))
    summary = _read_csv(root, "10_policy_summary.csv")
    historical = _read_csv(root, "02_historical_metric_contract.csv")
    causal = _read_csv(root, "12_causal_audit.csv")

    required_folds = {"WF_2024", "WF_2025"}
    for name, frame in (("cycles", cycles), ("legs", legs), ("daily", daily), ("summary", summary)):
        if frame.empty:
            raise RuntimeError(f"R03.4.2.14 {name} is empty")
        folds = set(frame.loc[frame["policy"].astype(str).eq(config.source_policy), "fold_id"].astype(str))
        if not required_folds.issubset(folds):
            raise RuntimeError(f"R03.4.2.14 {name} missing frozen folds")

    return FinalAuditInputs(
        historical_contract=historical,
        cycles=cycles,
        legs=legs,
        daily_equity=daily,
        source_summary=summary,
        source_causal_audit=causal,
    )
