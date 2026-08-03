#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Score-tier failure-overlay execution and robustness metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import FailureOverlayConfig


@dataclass(frozen=True)
class OverlayThresholds:
    global_warning: float
    global_confirm: float
    ultra_confirm: float
    tier_warning: dict[str, float]
    tier_confirm: dict[str, float]


def score_tier(percentile: float) -> str:
    value = float(percentile)
    if value >= 0.90:
        return "q90_plus"
    if value >= 0.80:
        return "q80_to_q90"
    return "q70_to_q80"


def _flag(value: object) -> bool:
    try:
        return bool(float(value) >= 0.5)
    except (TypeError, ValueError):
        return False


def structural_gate_count(row: pd.Series, config: FailureOverlayConfig) -> tuple[int, dict[str, bool]]:
    flags = {
        "below_entry": _flag(row.get("x180_current_below_entry", 0.0)),
        "negative_last60": float(row.get("x180_last60_return", np.nan)) < 0.0,
        "prior_low_break": (
            _flag(row.get("x180_broke_prior_low_60", 0.0))
            or float(row.get("x180_distance_to_prior_low_60", np.nan)) <= 0.0
        ),
        "lower_low_structure": float(row.get("x180_bar15_lower_low_share", np.nan)) >= config.lower_low_share_minimum,
        "weak_recovery": float(row.get("x180_recovery_from_trough", np.nan)) <= config.maximum_recovery_from_trough,
        "mostly_underwater": float(row.get("x180_underwater_fraction", np.nan)) >= config.underwater_fraction_minimum,
    }
    return int(sum(flags.values())), flags


def _threshold_for(policy: str, tier: str, thresholds: OverlayThresholds, config: FailureOverlayConfig) -> tuple[float, float, int]:
    if policy == "global_failure_overlay":
        return thresholds.global_warning, thresholds.global_confirm, config.standard_gate_count
    if policy == "tiered_failure_overlay":
        return thresholds.tier_warning[tier], thresholds.tier_confirm[tier], config.standard_gate_count
    if policy == "ultra_failure_overlay":
        return thresholds.tier_warning[tier], max(thresholds.ultra_confirm, thresholds.tier_confirm[tier]), config.ultra_gate_count
    raise ValueError(policy)


def _execution_fields(row: pd.Series, delay_minutes: int) -> tuple[pd.Timestamp, float, pd.Timestamp, float]:
    entry_time = pd.Timestamp(row[f"entry_time_delay_{delay_minutes}m"])
    entry_price = float(row[f"entry_price_delay_{delay_minutes}m"])
    fixed_exit_time = pd.Timestamp(row[f"fixed_exit_time_delay_{delay_minutes}m"])
    fixed_exit_price = float(row[f"fixed_exit_price_delay_{delay_minutes}m"])
    return entry_time, entry_price, fixed_exit_time, fixed_exit_price


def simulate_overlay_event(
    row: pd.Series,
    *,
    policy: str,
    delay_minutes: int,
    thresholds: OverlayThresholds,
    config: FailureOverlayConfig,
) -> dict[str, object] | None:
    entry_time, entry_price, fixed_exit_time, fixed_exit_price = _execution_fields(row, delay_minutes)
    if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(fixed_exit_price):
        return None

    tier = score_tier(float(row["event_score_percentile"]))
    exit_time = fixed_exit_time
    exit_price = fixed_exit_price
    exit_reason = "fixed_6h_diagnostic"
    overlay_triggered = False
    warning = False
    gate_count, gate_flags = structural_gate_count(row, config)

    disaster_time_value = row.get(f"disaster_exit_time_delay_{delay_minutes}m", pd.NaT)
    disaster_price_value = row.get(f"disaster_exit_price_delay_{delay_minutes}m", np.nan)
    disaster_time = pd.Timestamp(disaster_time_value) if pd.notna(disaster_time_value) else None
    disaster_price = float(disaster_price_value) if pd.notna(disaster_price_value) else np.nan

    if policy != "fixed_6h":
        if policy != "fixed_6h_disaster_stop":
            warning_threshold, confirm_threshold, required_gates = _threshold_for(policy, tier, thresholds, config)
            p60 = float(row.get("p_failure_60", np.nan))
            p180 = float(row.get("p_failure_180", np.nan))
            warning = bool(
                np.isfinite(p60)
                and p60 >= warning_threshold
                and _flag(row.get("x60_current_below_entry", 0.0))
            )
            confirmed = bool(
                warning
                and np.isfinite(p180)
                and p180 >= confirm_threshold
                and gate_count >= required_gates
            )
            if confirmed:
                candidate_time = pd.Timestamp(row[f"overlay_exit_time_delay_{delay_minutes}m"])
                candidate_price = float(row[f"overlay_exit_price_delay_{delay_minutes}m"])
                if candidate_time < exit_time and np.isfinite(candidate_price):
                    exit_time = candidate_time
                    exit_price = candidate_price
                    exit_reason = "confirmed_persistent_failure_t180"
                    overlay_triggered = True
        if disaster_time is not None and disaster_time < exit_time and np.isfinite(disaster_price):
            exit_time = disaster_time
            exit_price = disaster_price
            exit_reason = "disaster_stop_next_open"
            overlay_triggered = False

    gross_return = float(exit_price / entry_price - 1.0)
    fixed_gross_return = float(fixed_exit_price / entry_price - 1.0)
    return {
        "event_id": row["event_id"],
        "fold_id": row["fold_id"],
        "scope": "broad_q70",
        "policy": policy,
        "decision_time": pd.Timestamp(row["decision_time"]),
        "entry_time": entry_time,
        "exit_time": exit_time,
        "delay_minutes": int(delay_minutes),
        "event_score_percentile": float(row["event_score_percentile"]),
        "score_tier": tier,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return": gross_return,
        "fixed6h_gross_return": fixed_gross_return,
        "overlay_uplift_gross": float(gross_return - fixed_gross_return),
        "mfe": float(row.get("mfe_360m", np.nan)),
        "mae": float(row.get("mae_360m", np.nan)),
        "holding_minutes": int(round((exit_time - entry_time) / pd.Timedelta(minutes=1))) + 1,
        "exit_reason": exit_reason,
        "warning_t60": warning,
        "overlay_triggered": overlay_triggered,
        "structural_gate_count": gate_count,
        "p_failure_60": float(row.get("p_failure_60", np.nan)),
        "p_failure_180": float(row.get("p_failure_180", np.nan)),
        "persistent_failure_target": int(row.get("label_persistent_failure", 0)),
        "fixed6h_would_win_1x": bool(fixed_gross_return > config.base_round_trip_cost),
        "score_upgrade_by_180": bool(row.get("score_upgrade_by_180", False)),
        "score_upgrade_by_360": bool(row.get("score_upgrade_by_360", False)),
        **{f"gate_{name}": bool(value) for name, value in gate_flags.items()},
    }


def enforce_non_overlap(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame.copy(), 0
    work = frame.sort_values(["entry_time", "decision_time", "event_score_percentile"], ascending=[True, True, False]).copy()
    keep: list[int] = []
    last_exit: pd.Timestamp | None = None
    skipped = 0
    for index, row in work.iterrows():
        entry = pd.Timestamp(row["entry_time"])
        if last_exit is not None and entry <= last_exit:
            skipped += 1
            continue
        keep.append(index)
        last_exit = pd.Timestamp(row["exit_time"])
    return work.loc[keep].sort_values("entry_time").reset_index(drop=True), skipped


def profit_factor(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    gains = float(array[array > 0].sum())
    losses = float(-array[array < 0].sum())
    return gains / losses if losses > 0 else np.inf if gains > 0 else np.nan


def maximum_drawdown(values: np.ndarray) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return np.nan, np.nan
    equity = np.cumprod(1.0 + array)
    peaks = np.maximum.accumulate(np.concatenate([[1.0], equity]))[1:]
    drawdown = equity / peaks - 1.0
    return float(drawdown.min()), float(equity[-1] - 1.0)


def summarize(frame: pd.DataFrame, *, cost_multiplier: float, config: FailureOverlayConfig) -> dict[str, object]:
    if frame.empty:
        return {"trades": 0, "mean_net_return": np.nan, "profit_factor": np.nan}
    work = frame.sort_values("entry_time").copy()
    work["net_return"] = work["gross_return"].astype(float) - config.base_round_trip_cost * float(cost_multiplier)
    net = work["net_return"].to_numpy(dtype=float)
    winners = net[net > 0]
    losers = net[net < 0]
    mdd, total = maximum_drawdown(net)
    sorted_net = np.sort(net)[::-1]
    top_count = min(10, len(sorted_net))
    gross_profit = float(winners.sum())
    top_share = float(sorted_net[:top_count].sum() / gross_profit) if gross_profit > 0 else np.nan
    without_top = sorted_net[top_count:] if len(sorted_net) > top_count else np.empty(0)
    overlay = work.loc[work["overlay_triggered"].astype(bool)]
    false_exit = overlay.loc[overlay["fixed6h_would_win_1x"].astype(bool)]
    return {
        "trades": int(len(work)),
        "mean_net_return": float(net.mean()),
        "median_net_return": float(np.median(net)),
        "win_rate": float(np.mean(net > 0)),
        "profit_factor": profit_factor(net),
        "mean_winner": float(winners.mean()) if len(winners) else np.nan,
        "mean_loser": float(losers.mean()) if len(losers) else np.nan,
        "total_compounded_return": total,
        "max_drawdown": mdd,
        "mean_mfe": float(work["mfe"].mean()),
        "mean_mae": float(work["mae"].mean()),
        "median_holding_minutes": float(work["holding_minutes"].median()),
        "top10_profit_share": top_share,
        "mean_net_without_top10": float(without_top.mean()) if len(without_top) else np.nan,
        "overlay_exits": int(len(overlay)),
        "overlay_exit_share": float(len(overlay) / len(work)),
        "overlay_mean_uplift_gross": float(overlay["overlay_uplift_gross"].mean()) if len(overlay) else np.nan,
        "overlay_false_exit_share": float(len(false_exit) / len(overlay)) if len(overlay) else np.nan,
        "persistent_failure_capture": float(overlay["persistent_failure_target"].sum() / max(1, work["persistent_failure_target"].sum())),
        "disaster_stops": int((work["exit_reason"] == "disaster_stop_next_open").sum()),
    }


def build_policy_tables(
    trades: pd.DataFrame,
    *,
    config: FailureOverlayConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    summary_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []
    tier_rows: list[dict[str, object]] = []
    exit_rows: list[dict[str, object]] = []
    for keys, group in trades.groupby(["fold_id", "policy", "delay_minutes"], sort=False):
        fold_id, policy, delay = keys
        base = {"fold_id": fold_id, "policy": policy, "delay_minutes": int(delay)}
        for cost in config.cost_multipliers:
            summary_rows.append({**base, "cost_multiplier": float(cost), **summarize(group, cost_multiplier=cost, config=config)})
        quarter = pd.to_datetime(group["entry_time"]).dt.to_period("Q").astype(str)
        for period, part in group.groupby(quarter, sort=True):
            for cost in (1.0, 2.0):
                period_rows.append({**base, "quarter": period, "cost_multiplier": cost, **summarize(part, cost_multiplier=cost, config=config)})
        for tier, part in group.groupby("score_tier", sort=True):
            for cost in (1.0, 2.0, 3.0):
                tier_rows.append({**base, "score_tier": tier, "cost_multiplier": cost, **summarize(part, cost_multiplier=cost, config=config)})
        for reason, part in group.groupby("exit_reason", sort=True):
            exit_rows.append(
                {
                    **base,
                    "exit_reason": reason,
                    "count": int(len(part)),
                    "share": float(len(part) / len(group)),
                    "mean_gross_return": float(part["gross_return"].mean()),
                    "mean_fixed6h_gross_return": float(part["fixed6h_gross_return"].mean()),
                    "mean_uplift_gross": float(part["overlay_uplift_gross"].mean()),
                    "persistent_failure_rate": float(part["persistent_failure_target"].mean()),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(period_rows), pd.DataFrame(tier_rows), pd.DataFrame(exit_rows)


def stable_candidates(summary: pd.DataFrame, periods: pd.DataFrame, config: FailureOverlayConfig) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    primary = summary.loc[(summary["delay_minutes"] == 1) & (summary["cost_multiplier"] == 2.0)].copy()
    baseline = primary.loc[primary["policy"] == "fixed_6h"].set_index("fold_id")
    safety = primary.loc[primary["policy"] == "fixed_6h_disaster_stop"].set_index("fold_id")
    rows: list[dict[str, object]] = []
    for policy, group in primary.groupby("policy", sort=False):
        by_fold = group.set_index("fold_id")
        required_folds = {"WF_2024", "WF_2025"}
        if not required_folds.issubset(by_fold.index) or not required_folds.issubset(baseline.index) or not required_folds.issubset(safety.index):
            continue
        folds = ("WF_2024", "WF_2025")
        values = {column: [float(by_fold.loc[fold, column]) for fold in folds] for column in (
            "mean_net_return", "profit_factor", "total_compounded_return", "max_drawdown",
            "top10_profit_share", "mean_net_without_top10", "overlay_exit_share",
            "overlay_mean_uplift_gross", "overlay_false_exit_share",
        )}
        trades = [int(by_fold.loc[fold, "trades"]) for fold in folds]
        baseline_total = [float(baseline.loc[fold, "total_compounded_return"]) for fold in folds]
        safety_total = [float(safety.loc[fold, "total_compounded_return"]) for fold in folds]
        retention = [
            (1.0 + values["total_compounded_return"][i]) / (1.0 + baseline_total[i])
            if baseline_total[i] > -1 else np.nan
            for i in range(2)
        ]
        quarters = periods.loc[
            (periods["policy"] == policy)
            & (periods["delay_minutes"] == 1)
            & (periods["cost_multiplier"] == 2.0)
        ]
        positive_quarters = int((quarters["mean_net_return"] > 0).sum())
        is_overlay = policy in {"global_failure_overlay", "tiered_failure_overlay", "ultra_failure_overlay"}
        stable = bool(
            min(values["mean_net_return"]) > 0
            and min(values["profit_factor"]) >= config.minimum_pf_2x
            and min(trades) >= config.minimum_trades_per_year
            and max(abs(value) for value in values["max_drawdown"]) <= config.maximum_mdd
            and max(values["top10_profit_share"]) <= config.maximum_top10_profit_share
            and min(values["mean_net_without_top10"]) > 0
            and positive_quarters >= config.minimum_positive_quarters
        )
        overlay_valid = bool(
            is_overlay
            and min(values["overlay_exit_share"]) >= config.minimum_overlay_exit_share
            and max(values["overlay_exit_share"]) <= config.maximum_overlay_exit_share
            and min(values["overlay_mean_uplift_gross"]) > 0
            and min(retention) >= config.minimum_baseline_profit_retention
        )
        beats_baseline = bool(
            is_overlay
            and all(values["total_compounded_return"][i] > baseline_total[i] for i in range(2))
        )
        beats_safety = bool(
            is_overlay
            and all(values["total_compounded_return"][i] > safety_total[i] for i in range(2))
        )
        rows.append(
            {
                "policy": policy,
                "mean_net_2x_2024": values["mean_net_return"][0],
                "mean_net_2x_2025": values["mean_net_return"][1],
                "pf_2x_2024": values["profit_factor"][0],
                "pf_2x_2025": values["profit_factor"][1],
                "trades_2024": trades[0],
                "trades_2025": trades[1],
                "total_return_2x_2024": values["total_compounded_return"][0],
                "total_return_2x_2025": values["total_compounded_return"][1],
                "baseline_total_return_2024": baseline_total[0],
                "baseline_total_return_2025": baseline_total[1],
                "safety_total_return_2024": safety_total[0],
                "safety_total_return_2025": safety_total[1],
                "profit_retention_2024": retention[0],
                "profit_retention_2025": retention[1],
                "overlay_exit_share_2024": values["overlay_exit_share"][0],
                "overlay_exit_share_2025": values["overlay_exit_share"][1],
                "overlay_uplift_2024": values["overlay_mean_uplift_gross"][0],
                "overlay_uplift_2025": values["overlay_mean_uplift_gross"][1],
                "false_exit_share_2024": values["overlay_false_exit_share"][0],
                "false_exit_share_2025": values["overlay_false_exit_share"][1],
                "positive_quarters": positive_quarters,
                "stable_positive_expectancy": stable,
                "valid_high_confidence_overlay": overlay_valid,
                "beats_fixed6h_both_years": beats_baseline,
                "beats_safety_baseline_both_years": beats_safety,
                "stable_overlay_upgrade": bool(stable and overlay_valid and beats_safety),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["stable_overlay_upgrade", "valid_high_confidence_overlay", "stable_positive_expectancy", "mean_net_2x_2025", "mean_net_2x_2024"],
        ascending=[False, False, False, False, False],
    ) if rows else pd.DataFrame()
