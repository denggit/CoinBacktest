#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""MAE attribution and qualification gates for R03.4.2.14."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import MinutePathData

from .config import EntryTimingConfig


def build_anchor_mae_attribution(source_cycles: pd.DataFrame, selected_events: pd.DataFrame, *, path: MinutePathData, fold_id: str) -> pd.DataFrame:
    cycles = source_cycles.loc[source_cycles["fold_id"].astype(str).eq(fold_id) & source_cycles["delay_minutes"].astype(int).eq(1) & np.isclose(source_cycles["cost_multiplier"].astype(float), 2.0)].copy()
    selected = selected_events.loc[selected_events["fold_id"].astype(str).eq(fold_id) & selected_events["delay_minutes"].astype(int).eq(1), ["event_id", "exit_time"]].rename(columns={"exit_time": "p0_exit_time"})
    work = cycles.merge(selected, on="event_id", how="left")
    rows: list[dict[str, object]] = []
    for row in work.to_dict("records"):
        entry = path.locate_exact(pd.Timestamp(row["entry_time"])); exit_pos = path.locate_exact(pd.Timestamp(row["exit_time"])); p0_exit = path.locate_exact(pd.Timestamp(row["p0_exit_time"])) if pd.notna(row.get("p0_exit_time")) else exit_pos
        if entry is None or exit_pos is None:
            continue
        price = float(path.open[entry])
        metrics: dict[str, float] = {}
        for horizon in (15, 30, 60, 120):
            right = min(exit_pos, entry+horizon-1)
            metrics[f"mae_{horizon}m"] = float(np.min(path.low[entry:right+1])/price-1)
            metrics[f"mfe_{horizon}m"] = float(np.max(path.high[entry:right+1])/price-1)
        recovered_after_exit = False
        if p0_exit is not None and p0_exit > exit_pos:
            recovered_after_exit = bool(np.max(path.high[exit_pos:min(p0_exit+1, len(path.high))]) >= price)
        cycle_return = float(row["cycle_return"])
        if cycle_return > 0 and metrics["mae_60m"] <= -0.01:
            klass = "DEEP_MAE_RECOVERED_WIN"
        elif cycle_return > 0:
            klass = "LOW_MAE_WIN"
        elif bool(row.get("hard_stop_exit")) or bool(row.get("soft_failure_exit")):
            klass = "EARLY_STOP_THEN_RECOVER" if recovered_after_exit else "TRUE_EARLY_FAILURE"
        else:
            klass = "STRUCTURAL_LOSS"
        rows.append({"event_id": row["event_id"], "fold_id": fold_id, "cycle_return": cycle_return, "hard_stop_exit": bool(row.get("hard_stop_exit")), "soft_failure_exit": bool(row.get("soft_failure_exit")), "recovered_after_c2_exit_before_p0_exit": recovered_after_exit, "mae_class": klass, **metrics})
    return pd.DataFrame(rows)


def summarize_mae_attribution(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (fold, klass), group in frame.groupby(["fold_id", "mae_class"], sort=True):
        rows.append({"fold_id": fold, "mae_class": klass, "events": int(len(group)), "share": float(len(group)/len(frame.loc[frame["fold_id"].eq(fold)])), "mean_cycle_return": float(group["cycle_return"].mean()), "mean_mae_60m": float(group["mae_60m"].mean()), "recovery_share": float(group["recovered_after_c2_exit_before_p0_exit"].mean())})
    return pd.DataFrame(rows)


def policy_gate(summary: pd.DataFrame, config: EntryTimingConfig) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    base = summary.loc[summary["policy"].astype(str).eq("E0_immediate_C2") & summary["delay_minutes"].astype(int).eq(1) & np.isclose(summary["cost_multiplier"].astype(float), 2.0)].set_index("fold_id")
    rows: list[dict[str, object]] = []
    for policy in config.policies:
        primary = summary.loc[summary["policy"].astype(str).eq(policy.name) & summary["delay_minutes"].astype(int).eq(1) & np.isclose(summary["cost_multiplier"].astype(float), 2.0)]
        fold_checks: list[bool] = []; retentions=[]; coverage=[]; mae_improvements=[]; win_deltas=[]; stop_reductions=[]; mdd_ratios=[]
        for row in primary.to_dict("records"):
            fold = str(row["fold_id"])
            if fold not in base.index: continue
            b = base.loc[fold]
            retention = float(row["total_net_return"])/float(b["total_net_return"]) if float(b["total_net_return"])>0 else np.nan
            mdd_ratio = abs(float(row["max_drawdown"]))/abs(float(b["max_drawdown"])) if abs(float(b["max_drawdown"]))>0 else np.inf
            mae_improvement = (abs(float(b["mean_mae_60m"]))-abs(float(row["mean_mae_60m"])))/abs(float(b["mean_mae_60m"])) if abs(float(b["mean_mae_60m"]))>0 else 0.0
            win_delta = float(row["win_rate"])-float(b["win_rate"])
            base_stop = float(b["hard_stop_share"])+float(b["soft_failure_share"]); candidate_stop=float(row["hard_stop_share"])+float(row["soft_failure_share"])
            stop_reduction=(base_stop-candidate_stop)/base_stop if base_stop>0 else 0.0
            retentions.append(retention); coverage.append(float(row["coverage_ratio"])); mae_improvements.append(mae_improvement); win_deltas.append(win_delta); stop_reductions.append(stop_reduction); mdd_ratios.append(mdd_ratio)
            quality_uplift = mae_improvement >= config.minimum_mae60_improvement or win_delta >= config.minimum_win_rate_improvement or stop_reduction >= config.minimum_stop_share_reduction
            fold_checks.append(bool(float(row["total_net_return"])>0 and retention>=config.minimum_return_retention_each_year and float(row["coverage_ratio"])>=config.minimum_coverage_ratio and mdd_ratio<=config.maximum_mdd_multiple and abs(float(row["max_drawdown"]))<=config.maximum_absolute_mdd and int(row["positive_quarters"])>=config.minimum_positive_quarters_per_year and float(row["total_return_without_top10"])>0 and float(row["top10_profit_share"])-float(b["top10_profit_share"])<=config.maximum_top10_profit_share_increase and quality_uplift))
        stress_checks=[]
        stress=summary.loc[summary["policy"].astype(str).eq(policy.name)]
        for fold in ("WF_2024","WF_2025"):
            required=stress.loc[stress["fold_id"].astype(str).eq(fold) & stress["delay_minutes"].astype(int).isin(config.entry_delay_minutes) & stress["cost_multiplier"].astype(float).isin(config.cost_multipliers)]
            stress_checks.append(bool(len(required)==6 and (required["total_net_return"].astype(float)>0).all()))
        candidate_total=float(primary["total_net_return"].sum()); base_total=float(base["total_net_return"].sum()); combined=candidate_total/base_total if base_total>0 else np.nan
        rows.append({"policy":policy.name,"qualifying_candidate":policy.qualifying_candidate,"minimum_return_retention":min(retentions) if retentions else np.nan,"minimum_coverage_ratio":min(coverage) if coverage else np.nan,"minimum_mae60_improvement":min(mae_improvements) if mae_improvements else np.nan,"minimum_win_rate_delta":min(win_deltas) if win_deltas else np.nan,"minimum_stop_share_reduction":min(stop_reductions) if stop_reductions else np.nan,"maximum_mdd_ratio":max(mdd_ratios) if mdd_ratios else np.nan,"combined_return_ratio":combined,"fold_gate_pass":bool(len(fold_checks)==2 and all(fold_checks)),"stress_gate_pass":bool(len(stress_checks)==2 and all(stress_checks)),"pass_to_next_stage":bool(policy.qualifying_candidate and len(fold_checks)==2 and all(fold_checks) and len(stress_checks)==2 and all(stress_checks) and combined>=config.minimum_combined_return_ratio)})
    return pd.DataFrame(rows)
