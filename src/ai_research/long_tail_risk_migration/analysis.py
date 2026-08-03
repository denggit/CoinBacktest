#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summary and qualification logic for R03.4.2.10."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import RiskMigrationConfig


def build_account_summary(rows: list[dict[str, object]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["policy", "fold_id", "delay_minutes", "cost_multiplier"]
    ).reset_index(drop=True)


def policy_gate(account_summary: pd.DataFrame, config: RiskMigrationConfig) -> pd.DataFrame:
    if account_summary.empty:
        return pd.DataFrame()
    baseline = account_summary.loc[account_summary["policy"] == "P0_single_1R"].copy()
    baseline_lookup = {
        (str(row.fold_id), int(row.delay_minutes), float(row.cost_multiplier)): row
        for row in baseline.itertuples()
    }
    base_focus = baseline.loc[
        (baseline["delay_minutes"].astype(int) == 1)
        & (baseline["cost_multiplier"].astype(float) == 2.0)
    ]
    baseline_cross_year = float(base_focus["total_net_return"].sum())

    rows: list[dict[str, object]] = []
    for policy in config.policies:
        group = account_summary.loc[account_summary["policy"] == policy.name].copy()
        focus = group.loc[
            (group["delay_minutes"].astype(int) == 1)
            & (group["cost_multiplier"].astype(float) == 2.0)
        ]
        fold_checks: list[bool] = []
        retentions: list[float] = []
        mdd_multiples: list[float] = []
        for row in focus.itertuples():
            base = baseline_lookup.get((str(row.fold_id), 1, 2.0))
            if base is None or float(base.total_net_return) <= 0:
                fold_checks.append(False)
                continue
            retention = float(row.total_net_return / base.total_net_return)
            mdd_multiple = abs(float(row.max_drawdown)) / max(abs(float(base.max_drawdown)), 1e-12)
            retentions.append(retention)
            mdd_multiples.append(mdd_multiple)
            frequency_ok = True
            migration_safety_ok = True
            if policy.allow_migration:
                frequency_ok = bool(
                    float(row.coverage_ratio) >= config.minimum_coverage_ratio_for_migration
                    and float(row.monthly_tranches) >= config.minimum_monthly_tranches_for_migration
                )
                migration_safety_ok = bool(
                    float(row.losing_migration_share) <= config.maximum_losing_migration_share
                    and float(row.broken_migration_share) <= config.maximum_broken_migration_share
                )
            fold_checks.append(
                bool(
                    float(row.total_net_return) > 0
                    and retention >= config.minimum_return_retention_each_year
                    and mdd_multiple <= config.maximum_mdd_multiple
                    and float(row.total_return_without_top10) > 0
                    and int(row.positive_quarters) >= config.minimum_positive_quarters_per_year
                    and float(row.max_cycle_allocated_r)
                    <= config.maximum_cycle_r + config.maximum_cycle_r_tolerance
                    and frequency_ok
                    and migration_safety_ok
                )
            )

        stress = group.loc[group["cost_multiplier"].astype(float).isin(config.cost_multipliers)]
        stress_pass = bool(
            len(stress) == 12
            and (stress["total_net_return"].astype(float) > 0).all()
            and (
                stress["max_cycle_allocated_r"].astype(float)
                <= config.maximum_cycle_r + config.maximum_cycle_r_tolerance
            ).all()
        )
        cross_year = float(focus["total_net_return"].sum())
        combined_ratio = cross_year / baseline_cross_year if baseline_cross_year > 0 else np.nan
        final_pass = bool(
            policy.name != "P0_single_1R"
            and len(fold_checks) == 2
            and all(fold_checks)
            and stress_pass
            and np.isfinite(combined_ratio)
            and combined_ratio >= config.minimum_combined_return_ratio
        )
        rows.append(
            {
                "policy": policy.name,
                "minimum_return_retention": float(min(retentions)) if retentions else np.nan,
                "maximum_mdd_multiple": float(max(mdd_multiples)) if mdd_multiples else np.nan,
                "cross_year_total_return": cross_year,
                "baseline_cross_year_total_return": baseline_cross_year,
                "combined_return_ratio": combined_ratio,
                "fold_gate_pass": bool(len(fold_checks) == 2 and all(fold_checks)),
                "stress_gate_pass": stress_pass,
                "pass_to_next_stage": final_pass,
            }
        )
    return pd.DataFrame(rows).sort_values("policy").reset_index(drop=True)
