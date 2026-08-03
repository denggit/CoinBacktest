#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen multi-stage policy diagnostics for q70/q90 OOS events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import LongTailMultistageConfig


@dataclass(frozen=True)
class PolicyThresholds:
    fail_high_180: float
    fail_safe_180: float
    recovery_low_180: float
    recovery_high_180: float
    fail_high_360: float
    recovery_low_360: float
    continuation_high_360: float
    longhold_high_1440: float

    def to_dict(self) -> dict[str, float]:
        return self.__dict__.copy()


def _probability(row: pd.Series, name: str, default: float = np.nan) -> float:
    value = row.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bad_at(row: pd.Series, minute: int, thresholds: PolicyThresholds) -> bool:
    fail = _probability(row, f"p_failure_{minute}")
    recovery = _probability(row, f"p_recovery_{minute}")
    weak = bool(row.get(f"weak_now_{minute}", False))
    if not np.isfinite(fail):
        return False
    fail_threshold = thresholds.fail_high_180 if minute == 180 else thresholds.fail_high_360
    recovery_threshold = thresholds.recovery_low_180 if minute == 180 else thresholds.recovery_low_360
    return bool(fail >= fail_threshold and weak and np.isfinite(recovery) and recovery <= recovery_threshold)


def _healthy_at_180(row: pd.Series, thresholds: PolicyThresholds) -> bool:
    fail = _probability(row, "p_failure_180")
    recovery = _probability(row, "p_recovery_180")
    weak = bool(row.get("weak_now_180", False))
    if not np.isfinite(fail) or fail > thresholds.fail_safe_180:
        return False
    if not weak:
        return True
    return bool(np.isfinite(recovery) and recovery >= thresholds.recovery_high_180)


def _select_exit(row: pd.Series, thresholds: PolicyThresholds) -> tuple[int, str]:
    if _bad_at(row, 180, thresholds):
        return 180, "confirmed_failure_t180"
    if _bad_at(row, 360, thresholds):
        return 360, "confirmed_failure_t360"
    continuation = _probability(row, "p_continuation_360")
    recovery = _probability(row, "p_recovery_360")
    weak = bool(row.get("weak_now_360", False))
    continue24 = bool(
        (np.isfinite(continuation) and continuation >= thresholds.continuation_high_360)
        or (weak and np.isfinite(recovery) and recovery >= thresholds.recovery_high_180)
    )
    if not continue24:
        return 360, "no_post6_hold_signal"
    longhold = _probability(row, "p_longhold_1440")
    if np.isfinite(longhold) and longhold >= thresholds.longhold_high_1440:
        return 7200, "five_day_longhold"
    return 1440, "one_day_hold"


def _price(row: pd.Series, name: str) -> float:
    value = float(row[name])
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"invalid price {name}={value}")
    return value


def simulate_policy_event(
    row: pd.Series,
    *,
    policy: str,
    delay_minutes: int,
    thresholds: PolicyThresholds,
    config: LongTailMultistageConfig,
) -> dict[str, object] | None:
    exit_minute, exit_reason = _select_exit(row, thresholds)
    exit_price = _price(row, f"close_price_{exit_minute}m")
    canonical_entry = pd.Timestamp(row["entry_time"])
    decision_time = pd.Timestamp(row["decision_time"])
    initial_entry = _price(row, f"entry_price_delay_{delay_minutes}m")
    initial_entry_time = decision_time + pd.Timedelta(minutes=delay_minutes)
    weight = 1.0
    added = False
    skipped_reason: str | None = None

    if policy == "fixed_6h":
        exit_minute, exit_reason = 360, "fixed_6h_close"
        exit_price = _price(row, "close_price_360m")
        gross = exit_price / initial_entry - 1.0
    elif policy == "full_multistage":
        gross = exit_price / initial_entry - 1.0
    elif policy == "half_probe_then_add":
        weight = 0.5
        if _bad_at(row, 180, thresholds):
            exit_minute, exit_reason = 180, "half_probe_failure_exit"
            exit_price = _price(row, "close_price_180m")
            gross = 0.5 * (exit_price / initial_entry - 1.0)
        else:
            gross = 0.5 * (exit_price / initial_entry - 1.0)
            if _healthy_at_180(row, thresholds):
                add_price = _price(row, "open_after_180m")
                gross += 0.5 * (exit_price / add_price - 1.0)
                weight = 1.0
                added = True
    elif policy == "delayed_confirm_180":
        if not _healthy_at_180(row, thresholds):
            skipped_reason = "not_healthy_at_180"
            return {
                "event_id": row["event_id"],
                "fold_id": row["fold_id"],
                "scope": row["scope"],
                "policy": policy,
                "delay_minutes": delay_minutes,
                "executed": False,
                "skipped_reason": skipped_reason,
                "decision_time": decision_time,
            }
        initial_entry = _price(row, "open_after_180m")
        initial_entry_time = canonical_entry + pd.Timedelta(minutes=180)
        if exit_minute <= 180:
            exit_minute, exit_reason = 360, "delayed_confirm_minimum_hold"
            exit_price = _price(row, "close_price_360m")
        gross = exit_price / initial_entry - 1.0
    else:
        raise ValueError(policy)

    exit_time = canonical_entry + pd.Timedelta(minutes=exit_minute - 1)
    horizon_key = 360 if exit_minute <= 360 else 1440 if exit_minute <= 1440 else 7200
    return {
        "event_id": row["event_id"],
        "fold_id": row["fold_id"],
        "scope": row["scope"],
        "policy": policy,
        "delay_minutes": int(delay_minutes),
        "executed": True,
        "skipped_reason": skipped_reason,
        "decision_time": decision_time,
        "entry_time": initial_entry_time,
        "exit_time": exit_time,
        "entry_price": initial_entry,
        "exit_price": exit_price,
        "gross_return": float(gross),
        "cost_weight": float(weight),
        "position_added_at_180": bool(added),
        "exit_minute": int(exit_minute),
        "exit_reason": exit_reason,
        "holding_minutes": int((exit_time - initial_entry_time) / pd.Timedelta(minutes=1) + 1),
        "mfe": float(row[f"mfe_{horizon_key}m"]),
        "mae": float(row[f"mae_{horizon_key}m"]),
        "entry_score_percentile": float(row["event_score_percentile"]),
        "path_class_360": row.get("path_class_360", ""),
    }


def enforce_non_overlap(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame.copy(), 0
    executed = frame.loc[frame["executed"] == True].sort_values(["entry_time", "decision_time"]).copy()  # noqa: E712
    keep: list[int] = []
    last_exit: pd.Timestamp | None = None
    skipped = 0
    for index, row in executed.iterrows():
        entry = pd.Timestamp(row["entry_time"])
        if last_exit is not None and entry <= last_exit:
            skipped += 1
            continue
        keep.append(index)
        last_exit = pd.Timestamp(row["exit_time"])
    return executed.loc[keep].reset_index(drop=True), skipped


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


def summarize_policy(
    frame: pd.DataFrame,
    *,
    cost_multiplier: float,
    config: LongTailMultistageConfig,
) -> dict[str, object]:
    if frame.empty:
        return {"trades": 0, "mean_net_return": np.nan, "profit_factor": np.nan}
    work = frame.copy()
    work["net_return"] = work["gross_return"].astype(float) - (
        config.base_round_trip_cost * cost_multiplier * work["cost_weight"].astype(float)
    )
    net = work["net_return"].to_numpy(dtype=float)
    mdd, total = maximum_drawdown(net)
    winners = net[net > 0]
    losers = net[net < 0]
    sorted_net = np.sort(net)[::-1]
    top_count = min(10, len(sorted_net))
    gross_profit = float(winners.sum())
    top_share = float(sorted_net[:top_count].sum() / gross_profit) if gross_profit > 0 else np.nan
    without_top = sorted_net[top_count:] if len(sorted_net) > top_count else np.empty(0)
    return {
        "trades": int(len(work)),
        "mean_gross_return": float(work["gross_return"].mean()),
        "mean_net_return": float(np.mean(net)),
        "median_net_return": float(np.median(net)),
        "win_rate": float(np.mean(net > 0)),
        "profit_factor": profit_factor(net),
        "mean_winner": float(np.mean(winners)) if len(winners) else np.nan,
        "mean_loser": float(np.mean(losers)) if len(losers) else np.nan,
        "payoff_ratio": float(np.mean(winners) / abs(np.mean(losers))) if len(winners) and len(losers) else np.nan,
        "mean_mfe": float(work["mfe"].mean()),
        "mean_mae": float(work["mae"].mean()),
        "median_holding_minutes": float(work["holding_minutes"].median()),
        "mean_holding_minutes": float(work["holding_minutes"].mean()),
        "total_compounded_return": total,
        "max_drawdown": mdd,
        "top10_profit_share": top_share,
        "mean_net_without_top10": float(np.mean(without_top)) if len(without_top) else np.nan,
        "add_at_180_share": float(work["position_added_at_180"].mean()),
    }


def build_policy_tables(
    trades: pd.DataFrame,
    *,
    overlap_audit: pd.DataFrame,
    config: LongTailMultistageConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    summary_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []
    exit_rows: list[dict[str, object]] = []
    for keys, group in trades.groupby(["fold_id", "scope", "policy", "delay_minutes"], sort=False):
        fold_id, scope, policy, delay = keys
        base = {"fold_id": fold_id, "scope": scope, "policy": policy, "delay_minutes": int(delay)}
        for cost in config.cost_multipliers:
            summary_rows.append({**base, "cost_multiplier": float(cost), **summarize_policy(group, cost_multiplier=cost, config=config)})
        quarter = pd.to_datetime(group["entry_time"]).dt.to_period("Q").astype(str)
        for period, period_group in group.groupby(quarter, sort=True):
            for cost in (1.0, 2.0):
                period_rows.append(
                    {
                        **base,
                        "quarter": period,
                        "cost_multiplier": cost,
                        **summarize_policy(period_group, cost_multiplier=cost, config=config),
                    }
                )
        for reason, count in group["exit_reason"].value_counts().items():
            exit_rows.append(
                {
                    **base,
                    "exit_reason": reason,
                    "count": int(count),
                    "share": float(count / len(group)),
                    "mean_gross_return": float(group.loc[group["exit_reason"] == reason, "gross_return"].mean()),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(period_rows), pd.DataFrame(exit_rows), overlap_audit


def stable_policy_candidates(summary: pd.DataFrame, periods: pd.DataFrame, config: LongTailMultistageConfig) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    primary = summary.loc[(summary["cost_multiplier"] == 2.0) & (summary["delay_minutes"] == 1)].copy()
    q90_fixed = primary.loc[
        (primary["scope"] == "primary_q90") & (primary["policy"] == "fixed_6h")
    ].set_index("fold_id")
    for keys, group in primary.groupby(["scope", "policy"], sort=False):
        scope, policy = keys
        by_fold = group.set_index("fold_id")
        if not {"WF_2024", "WF_2025"}.issubset(by_fold.index):
            continue
        quarters = periods.loc[
            (periods["scope"] == scope)
            & (periods["policy"] == policy)
            & (periods["delay_minutes"] == 1)
            & (periods["cost_multiplier"] == 2.0)
        ]
        positive_quarters = int((quarters["mean_net_return"] > 0).sum())
        folds = ("WF_2024", "WF_2025")
        mean_values = [float(by_fold.loc[fold, "mean_net_return"]) for fold in folds]
        pf_values = [float(by_fold.loc[fold, "profit_factor"]) for fold in folds]
        trade_values = [int(by_fold.loc[fold, "trades"]) for fold in folds]
        mdd_values = [abs(float(by_fold.loc[fold, "max_drawdown"])) for fold in folds]
        top_values = [float(by_fold.loc[fold, "top10_profit_share"]) for fold in folds]
        without_top = [float(by_fold.loc[fold, "mean_net_without_top10"]) for fold in folds]
        total_values = [float(by_fold.loc[fold, "total_compounded_return"]) for fold in folds]
        passed = (
            min(mean_values) > 0
            and min(pf_values) >= config.minimum_pf_2x
            and min(trade_values) >= config.minimum_trades_per_year
            and max(mdd_values) <= config.maximum_mdd
            and max(top_values) <= config.maximum_top10_profit_share
            and min(without_top) > 0
            and positive_quarters >= config.minimum_positive_quarters
        )

        same_scope_fixed = primary.loc[
            (primary["scope"] == scope) & (primary["policy"] == "fixed_6h")
        ].set_index("fold_id")
        same_scope_base = [
            float(same_scope_fixed.loc[fold, "total_compounded_return"])
            if fold in same_scope_fixed.index else np.nan
            for fold in folds
        ]
        same_scope_uplift = [
            total_values[i] - same_scope_base[i]
            if np.isfinite(same_scope_base[i]) else np.nan
            for i in range(2)
        ]
        beats_same_scope_fixed = bool(
            policy != "fixed_6h"
            and all(np.isfinite(value) and value > 0 for value in same_scope_uplift)
        )

        q90_base = [
            float(q90_fixed.loc[fold, "total_compounded_return"])
            if fold in q90_fixed.index else np.nan
            for fold in folds
        ]
        q70_vs_q90_uplift = [
            total_values[i] - q90_base[i]
            if np.isfinite(q90_base[i]) else np.nan
            for i in range(2)
        ]
        q70_expands_total_profit = bool(
            scope == "broad_q70"
            and all(np.isfinite(value) and value > 0 for value in q70_vs_q90_uplift)
        )
        rows.append(
            {
                "scope": scope,
                "policy": policy,
                "mean_net_2x_2024": mean_values[0],
                "mean_net_2x_2025": mean_values[1],
                "pf_2x_2024": pf_values[0],
                "pf_2x_2025": pf_values[1],
                "trades_2024": trade_values[0],
                "trades_2025": trade_values[1],
                "total_compounded_2x_2024": total_values[0],
                "total_compounded_2x_2025": total_values[1],
                "uplift_vs_same_scope_fixed_2024": same_scope_uplift[0],
                "uplift_vs_same_scope_fixed_2025": same_scope_uplift[1],
                "uplift_vs_q90_fixed_2024": q70_vs_q90_uplift[0],
                "uplift_vs_q90_fixed_2025": q70_vs_q90_uplift[1],
                "maximum_mdd": max(mdd_values),
                "maximum_top10_profit_share": max(top_values),
                "positive_quarters_2x": positive_quarters,
                "stable_positive_expectancy": bool(passed),
                "beats_same_scope_fixed_both_years": beats_same_scope_fixed,
                "q70_expands_total_profit_both_years": q70_expands_total_profit,
                "stable_multistage_upgrade": bool(passed and beats_same_scope_fixed),
                "stable_q70_expansion": bool(passed and q70_expands_total_profit),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["stable_multistage_upgrade", "stable_q70_expansion", "stable_positive_expectancy", "mean_net_2x_2025", "mean_net_2x_2024"],
        ascending=[False, False, False, False, False],
    ) if rows else pd.DataFrame()

