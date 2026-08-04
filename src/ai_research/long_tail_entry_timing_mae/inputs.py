#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Validated frozen report inputs for R03.4.2.14."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pandas as pd
from pandas.errors import EmptyDataError

from .config import EntryTimingConfig


@dataclass(frozen=True)
class EntryTimingInputs:
    folds: pd.DataFrame
    selected_events: pd.DataFrame
    all_q70_signals: pd.DataFrame
    source_c2_cycles: pd.DataFrame
    source_c2_legs: pd.DataFrame
    source_c2_summary: pd.DataFrame
    historical_contract: pd.DataFrame


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


def _require(root: Path, names: tuple[str, ...], label: str) -> None:
    missing = [name for name in names if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"{label} missing files: {missing}")


def load_entry_timing_inputs(config: EntryTimingConfig) -> EntryTimingInputs:
    config.validate()
    r8 = config.source_2_8a_path
    r12 = config.source_2_12_path
    r13 = config.source_2_13_path
    _require(r8, ("00_run_manifest.json", "04_frozen_baseline_summary.csv", "16_standalone_signal_outcomes.csv", "99_decision.md"), "R03.4.2.8A")
    _require(r12, ("00_run_manifest.json", "04_selected_p0_cycles.csv", "99_decision.md"), "R03.4.2.12")
    _require(r13, ("00_run_manifest.json", "05_account_cycles.csv", "06_account_legs.csv", "08_policy_summary.csv", "10_causal_audit.csv", "12_failures.csv", "99_decision.md"), "R03.4.2.13")

    m8 = json.loads((r8 / "00_run_manifest.json").read_text(encoding="utf-8"))
    m12 = json.loads((r12 / "00_run_manifest.json").read_text(encoding="utf-8"))
    m13 = json.loads((r13 / "00_run_manifest.json").read_text(encoding="utf-8"))
    if m8.get("stage") != "R03.4.2.8A" or m12.get("stage") != "R03.4.2.12" or m13.get("stage") != "R03.4.2.13":
        raise RuntimeError("unexpected source stage in R03.4.2.14 chain")
    if "PASS_REAL_1R_TAIL_COMPRESSION_CANDIDATE" not in (r12 / "99_decision.md").read_text(encoding="utf-8"):
        raise RuntimeError("R03.4.2.14 requires passed C2 source")
    if "PASS_EQUAL_RISK_RETAINED" not in (r13 / "99_decision.md").read_text(encoding="utf-8"):
        raise RuntimeError("R03.4.2.14 requires frozen equal-risk result")
    failures = _read_csv(r13 / "12_failures.csv")
    causal = _read_csv(r13 / "10_causal_audit.csv")
    if not failures.empty or causal.empty or not causal["status"].astype(str).eq("PASS").all():
        raise RuntimeError("R03.4.2.13 source is not clean")

    folds = pd.DataFrame(m8.get("folds", []))
    if folds.empty:
        raise RuntimeError("R03.4.2.8A folds missing")
    for column in ("fit_start", "fit_end", "calibration_start", "calibration_end", "test_start", "test_end"):
        if column in folds.columns:
            folds[column] = pd.to_datetime(folds[column])
    if pd.Timestamp(folds["test_end"].max()) >= pd.Timestamp(config.sealed_holdout_start):
        raise RuntimeError("source folds opened 2026")

    selected = _times(_read_csv(r12 / "04_selected_p0_cycles.csv"), ("decision_time", "entry_time", "exit_time"))
    signals = _times(_read_csv(r8 / "16_standalone_signal_outcomes.csv"), ("decision_time", "entry_time", "exit_time"))
    signals = signals.loc[signals["standalone_outcome"].astype(str).eq("fixed_6h")].copy()
    signals = signals.sort_values(["fold_id", "delay_minutes", "decision_time", "score"], ascending=[True, True, True, False]).drop_duplicates(["fold_id", "delay_minutes", "event_id"])

    cycles = _times(_read_csv(r13 / "05_account_cycles.csv"), ("decision_time", "entry_time", "exit_time"))
    legs = _times(_read_csv(r13 / "06_account_legs.csv"), ("entry_time", "exit_time"))
    cycles = cycles.loc[cycles["policy"].astype(str).eq(config.source_policy)].copy()
    legs = legs.loc[legs["policy"].astype(str).eq(config.source_policy)].copy()
    summary = _read_csv(r13 / "08_policy_summary.csv")
    summary = summary.loc[summary["policy"].astype(str).eq(config.source_policy)].copy()
    if selected.empty or signals.empty or cycles.empty or legs.empty or summary.empty:
        raise RuntimeError("R03.4.2.14 source frames are empty")

    fixed = _read_csv(r8 / "04_frozen_baseline_summary.csv")
    hist_rows: list[dict[str, object]] = []
    for fold in ("WF_2024", "WF_2025"):
        for baseline, label in (("q70_fixed_6h_diagnostic", "fixed_6h_all_signals"), ("P0_failed_reclaim_single_position", "P0_failed_reclaim")):
            row = fixed.loc[(fixed["baseline"].astype(str).eq(baseline)) & fixed["fold_id"].astype(str).eq(fold) & fixed["delay_minutes"].astype(int).eq(1) & fixed["cost_multiplier"].astype(float).eq(2.0)]
            if not row.empty:
                item = row.iloc[0]
                hist_rows.append({"fold_id": fold, "metric_scope": label, "trades": int(item["signals"]), "win_rate": float(item["win_rate"]), "profit_factor": float(item["profit_factor"]), "total_return": float(item["total_compounded_return"]), "max_drawdown": float(item["max_drawdown_diagnostic"]), "positive_months": pd.NA})
        row = summary.loc[summary["fold_id"].astype(str).eq(fold) & summary["delay_minutes"].astype(int).eq(1) & summary["cost_multiplier"].astype(float).eq(2.0)]
        if not row.empty:
            item = row.iloc[0]
            hist_rows.append({"fold_id": fold, "metric_scope": "C2_equal_1R_account", "trades": int(item["executed_cycles"]), "win_rate": float(item["win_rate"]), "profit_factor": float(item["profit_factor"]), "total_return": float(item["total_net_return"]), "max_drawdown": float(item["max_drawdown"]), "positive_months": int(item["positive_months"])})

    return EntryTimingInputs(
        folds=folds.reset_index(drop=True),
        selected_events=selected.reset_index(drop=True),
        all_q70_signals=signals.reset_index(drop=True),
        source_c2_cycles=cycles.reset_index(drop=True),
        source_c2_legs=legs.reset_index(drop=True),
        source_c2_summary=summary.reset_index(drop=True),
        historical_contract=pd.DataFrame(hist_rows),
    )
