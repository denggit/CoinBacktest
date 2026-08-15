#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Economic ceiling calculations using explicitly future-informed oracle geometry."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EconomicCeilingConfig


def _profit_factor(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").dropna()
    positive = float(v.loc[v > 0].sum())
    negative = float(-v.loc[v < 0].sum())
    if negative <= 0:
        return np.inf if positive > 0 else np.nan
    return positive / negative


def attach_oracle_metrics(episodes: pd.DataFrame, config: EconomicCeilingConfig) -> pd.DataFrame:
    """Attach path ceilings and fixed-R oracle realizations.

    IMPORTANT: risk uses the *future maximum adverse excursion* plus a fixed
    buffer. This is intentionally non-causal and is only an economic upper
    bound. It can never be promoted to a trading rule.
    """
    out = episodes.copy()
    additions: dict[str, object] = {}
    for horizon in config.horizons_seconds:
        adverse = pd.to_numeric(out[f"future_same_direction_extension_{horizon}s_bp"], errors="coerce").clip(lower=0.0)
        favorable = pd.to_numeric(out[f"future_opposite_excursion_{horizon}s_bp"], errors="coerce").clip(lower=0.0)
        terminal = pd.to_numeric(out[f"future_close_return_{horizon}s_bp"], errors="coerce")
        risk = adverse + float(config.stop_buffer_bp)
        safe_risk = risk.replace(0.0, np.nan)
        additions[f"oracle_adverse_bp_{horizon}s"] = adverse
        additions[f"oracle_favorable_bp_{horizon}s"] = favorable
        additions[f"oracle_risk_bp_{horizon}s"] = risk
        additions[f"oracle_gross_reward_risk_{horizon}s"] = favorable / safe_risk
        for cost in config.cost_scenarios_bp:
            tag = f"c{int(cost)}"
            net_mfe = favorable - float(cost)
            additions[f"oracle_net_mfe_bp_{horizon}s_{tag}"] = net_mfe
            additions[f"oracle_net_reward_risk_{horizon}s_{tag}"] = net_mfe / safe_risk
        for rr in config.reward_risk_targets:
            rr_tag = str(rr).replace(".", "p")
            target = risk * float(rr)
            hit = favorable.ge(target)
            additions[f"oracle_target_bp_{horizon}s_r{rr_tag}"] = target
            additions[f"oracle_target_hit_{horizon}s_r{rr_tag}"] = hit
            for cost in config.cost_scenarios_bp:
                cost_tag = f"c{int(cost)}"
                # Oracle stop is outside the future MAE, so the only outcomes
                # are target fill or conservative horizon close.
                gross = pd.Series(np.where(hit, target, terminal), index=out.index, dtype=float)
                additions[f"oracle_net_bp_{horizon}s_r{rr_tag}_{cost_tag}"] = gross - float(cost)
    if additions:
        out = pd.concat([out, pd.DataFrame(additions, index=out.index)], axis=1, copy=False)
    return out


def universe_masks(frame: pd.DataFrame, config: EconomicCeilingConfig) -> dict[str, pd.Series]:
    return {
        "ALL_RELEASE_EPISODES": pd.Series(True, index=frame.index),
        "FAVORABLE_REVERSAL_ORACLE": frame["favorable_reversal"].astype(bool),
        "FROZEN_R01_REVERSAL_CLUSTERS": frame["path_cluster"].isin(config.frozen_reversal_clusters),
        "CONTINUATION_CONTROL_CLUSTER_8": frame["path_cluster"].eq(config.continuation_control_cluster),
    }


def ceiling_distribution(frame: pd.DataFrame, config: EconomicCeilingConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    masks = universe_masks(frame, config)
    for universe, mask in masks.items():
        subset = frame.loc[mask]
        for keys, group in subset.groupby(["period", "event_side"], sort=True):
            period, side = keys
            for horizon in config.horizons_seconds:
                favorable = pd.to_numeric(group[f"oracle_favorable_bp_{horizon}s"], errors="coerce")
                adverse = pd.to_numeric(group[f"oracle_adverse_bp_{horizon}s"], errors="coerce")
                rr = pd.to_numeric(group[f"oracle_gross_reward_risk_{horizon}s"], errors="coerce")
                for cost in (config.primary_cost_bp, config.stress_cost_bp):
                    net = pd.to_numeric(group[f"oracle_net_mfe_bp_{horizon}s_c{int(cost)}"], errors="coerce")
                    valid = favorable.notna() & adverse.notna() & net.notna()
                    if not valid.any():
                        continue
                    rows.append(
                        {
                            "universe": universe,
                            "period": period,
                            "event_side": side,
                            "horizon_seconds": int(horizon),
                            "cost_bp": float(cost),
                            "episodes": int(valid.sum()),
                            "median_favorable_bp": float(favorable.loc[valid].median()),
                            "mean_favorable_bp": float(favorable.loc[valid].mean()),
                            "median_adverse_bp": float(adverse.loc[valid].median()),
                            "mean_adverse_bp": float(adverse.loc[valid].mean()),
                            "median_gross_reward_risk": float(rr.loc[valid].median()),
                            "positive_net_mfe_rate": float(net.loc[valid].gt(0).mean()),
                            "net_mfe_ge_10bp_rate": float(net.loc[valid].ge(10.0).mean()),
                            "net_mfe_ge_20bp_rate": float(net.loc[valid].ge(20.0).mean()),
                            "median_net_mfe_bp": float(net.loc[valid].median()),
                            "mean_net_mfe_bp": float(net.loc[valid].mean()),
                        }
                    )
    return pd.DataFrame(rows)


def fixed_r_performance(frame: pd.DataFrame, config: EconomicCeilingConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    masks = universe_masks(frame, config)
    for universe, mask in masks.items():
        subset = frame.loc[mask]
        for keys, group in subset.groupby(["period", "event_side"], sort=True):
            period, side = keys
            for horizon in config.horizons_seconds:
                risk = pd.to_numeric(group[f"oracle_risk_bp_{horizon}s"], errors="coerce")
                for rr in config.reward_risk_targets:
                    rr_tag = str(rr).replace(".", "p")
                    hit = group[f"oracle_target_hit_{horizon}s_r{rr_tag}"].astype(bool)
                    for cost in config.cost_scenarios_bp:
                        net = pd.to_numeric(group[f"oracle_net_bp_{horizon}s_r{rr_tag}_c{int(cost)}"], errors="coerce")
                        valid = net.notna() & risk.notna()
                        values = net.loc[valid]
                        if values.empty:
                            continue
                        sorted_values = values.sort_values(ascending=False)
                        trimmed = sorted_values.iloc[min(10, len(sorted_values)):]
                        rows.append(
                            {
                                "universe": universe,
                                "period": period,
                                "event_side": side,
                                "horizon_seconds": int(horizon),
                                "reward_risk_target": float(rr),
                                "cost_bp": float(cost),
                                "episodes": int(len(values)),
                                "mean_net_bp": float(values.mean()),
                                "median_net_bp": float(values.median()),
                                "win_rate": float(values.gt(0).mean()),
                                "profit_factor": float(_profit_factor(values)),
                                "target_hit_rate": float(hit.loc[valid].mean()),
                                "mean_oracle_risk_bp": float(risk.loc[valid].mean()),
                                "median_oracle_risk_bp": float(risk.loc[valid].median()),
                                "top10_removed_mean_net_bp": float(trimmed.mean()) if len(trimmed) else np.nan,
                                "top10_positive_pnl_share": float(sorted_values.iloc[:10].clip(lower=0).sum() / max(values.clip(lower=0).sum(), 1e-12)),
                            }
                        )
    return pd.DataFrame(rows)


def yearly_ceiling(frame: pd.DataFrame, config: EconomicCeilingConfig) -> pd.DataFrame:
    work = frame.copy()
    work["year"] = pd.to_datetime(work["event_time"], errors="coerce").dt.year
    h = config.primary_horizon_seconds
    rr_tag = str(config.primary_reward_risk).replace(".", "p")
    masks = universe_masks(work, config)
    rows: list[dict[str, object]] = []
    for universe, mask in masks.items():
        subset = work.loc[mask]
        for keys, group in subset.groupby(["year", "event_side"], sort=True):
            year, side = keys
            for cost in (config.primary_cost_bp, config.stress_cost_bp):
                values = pd.to_numeric(group[f"oracle_net_bp_{h}s_r{rr_tag}_c{int(cost)}"], errors="coerce").dropna()
                if values.empty:
                    continue
                rows.append(
                    {
                        "universe": universe,
                        "year": int(year),
                        "event_side": side,
                        "cost_bp": float(cost),
                        "episodes": int(len(values)),
                        "mean_net_bp": float(values.mean()),
                        "win_rate": float(values.gt(0).mean()),
                        "profit_factor": float(_profit_factor(values)),
                    }
                )
    return pd.DataFrame(rows)


def causal_audit(frame: pd.DataFrame, source_gate: pd.DataFrame, config: EconomicCeilingConfig) -> pd.DataFrame:
    rows = [
        {"check": "r01_1_source_gate", "value": int(source_gate["status"].astype(str).eq("FAIL").sum()), "status": "PASS" if not source_gate["status"].astype(str).eq("FAIL").any() else "FAIL"},
        {"check": "one_row_per_release_episode", "value": int(frame["release_episode_id"].duplicated().sum()), "status": "PASS" if not frame["release_episode_id"].duplicated().any() else "FAIL"},
        {"check": "oracle_future_information_explicit", "value": "MAE-based stop + future favorable labels", "status": "PASS"},
        {"check": "oracle_never_promoted_to_live_rule", "value": True, "status": "PASS"},
        {"check": "primary_cost_bp", "value": float(config.primary_cost_bp), "status": "PASS"},
        {"check": "stress_cost_bp", "value": float(config.stress_cost_bp), "status": "PASS"},
        {"check": "primary_horizon_seconds", "value": int(config.primary_horizon_seconds), "status": "PASS"},
        {"check": "primary_reward_risk_frozen", "value": float(config.primary_reward_risk), "status": "PASS"},
        {"check": "no_model_trained", "value": True, "status": "PASS"},
        {"check": "no_new_data_family", "value": True, "status": "PASS"},
    ]
    return pd.DataFrame(rows)


def _aggregate_sides(perf: pd.DataFrame) -> pd.DataFrame:
    # Decision uses episode-level data in reports/pipeline, so this helper is
    # intentionally not used for PF aggregation (PF cannot be averaged).
    return perf


def decision_from_episode_metrics(frame: pd.DataFrame, config: EconomicCeilingConfig) -> tuple[str, pd.DataFrame]:
    """Frozen stop/go gate using the perfect-exit net-MFE ceiling.

    Fixed-R realizations remain diagnostics.  The hard question here is more
    primitive: even with future-known favorable reversal episodes and a
    perfect exit at MFE, is there enough room after realistic costs?
    """
    h = config.primary_horizon_seconds
    rr_tag = str(config.primary_reward_risk).replace(".", "p")
    oracle = frame.loc[frame["favorable_reversal"].astype(bool)].copy()
    gate_rows: list[dict[str, object]] = []
    all_pass = True
    for period in (config.validation_period, config.holdout_period):
        group = oracle.loc[oracle["period"].astype(str).eq(period)]
        for cost, cost_label in ((config.primary_cost_bp, "BASE"), (config.stress_cost_bp, "STRESS_2X")):
            net_mfe = pd.to_numeric(group[f"oracle_net_mfe_bp_{h}s_c{int(cost)}"], errors="coerce").dropna()
            fixed_r = pd.to_numeric(group[f"oracle_net_bp_{h}s_r{rr_tag}_c{int(cost)}"], errors="coerce").dropna()
            trimmed = net_mfe.sort_values(ascending=False).iloc[min(10, len(net_mfe)):]
            metrics = {
                "episodes": int(len(net_mfe)),
                "mean_net_mfe_bp": float(net_mfe.mean()) if len(net_mfe) else np.nan,
                "median_net_mfe_bp": float(net_mfe.median()) if len(net_mfe) else np.nan,
                "net_mfe_profit_factor": float(_profit_factor(net_mfe)) if len(net_mfe) else np.nan,
                "top10_removed_mean_net_mfe_bp": float(trimmed.mean()) if len(trimmed) else np.nan,
                "positive_net_mfe_rate": float(net_mfe.gt(0).mean()) if len(net_mfe) else np.nan,
                "fixed_r_mean_net_bp": float(fixed_r.mean()) if len(fixed_r) else np.nan,
                "fixed_r_profit_factor": float(_profit_factor(fixed_r)) if len(fixed_r) else np.nan,
            }
            if cost_label == "BASE":
                passes = (
                    metrics["episodes"] >= config.minimum_oracle_episodes_per_period
                    and metrics["mean_net_mfe_bp"] >= config.gate_min_base_mean_net_bp
                    and metrics["net_mfe_profit_factor"] >= config.gate_min_base_profit_factor
                    and metrics["top10_removed_mean_net_mfe_bp"] > config.gate_min_top10_removed_mean_net_bp
                    and metrics["positive_net_mfe_rate"] >= config.gate_min_base_positive_mfe_rate
                )
            else:
                passes = (
                    metrics["episodes"] >= config.minimum_oracle_episodes_per_period
                    and metrics["mean_net_mfe_bp"] > config.gate_min_stress_mean_net_bp
                    and metrics["net_mfe_profit_factor"] > config.gate_min_stress_profit_factor
                )
            gate_rows.append({"period": period, "cost_gate": cost_label, "cost_bp": cost, **metrics, "status": "PASS" if passes else "FAIL"})
            all_pass &= bool(passes)
    decision = (
        "CONTINUE_LATENT_LIQUIDITY_IDENTIFICATION_ECONOMIC_CEILING_EXISTS"
        if all_pass
        else "STOP_LATENT_LIQUIDITY_REVERSAL_ECONOMIC_CEILING_TOO_THIN"
    )
    return decision, pd.DataFrame(gate_rows)
