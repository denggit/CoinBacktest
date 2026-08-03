#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal six-hour diagnostic execution and q70/q90 robustness metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import MinutePathData
from src.ai_research.long_tail_exit_audit.simulator import EventCandidate

from .config import Q70CrossYearAuditConfig


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float)[np.isfinite(reference)])
    values = np.asarray(values, dtype=float)
    output = np.full(len(values), np.nan, dtype=float)
    valid = np.isfinite(values)
    if len(ref):
        output[valid] = np.searchsorted(ref, values[valid], side="right") / len(ref)
    return output


def simulate_fixed_horizon_event(
    event: EventCandidate,
    *,
    fold_id: str,
    scope: str,
    delay_minutes: int,
    score_percentile: float,
    path: MinutePathData,
    config: Q70CrossYearAuditConfig,
) -> dict[str, object] | None:
    entry_ns = int(event.decision_time_ns + pd.Timedelta(minutes=delay_minutes).value)
    entry_pos = int(np.searchsorted(path.timestamps_ns, entry_ns, side="left"))
    if entry_pos >= len(path.timestamps_ns) or int(path.timestamps_ns[entry_pos]) != entry_ns:
        return None
    exit_pos = entry_pos + config.diagnostic_horizon_hours * 60 - 1
    if exit_pos >= len(path.timestamps_ns):
        return None
    expected_exit_ns = entry_ns + int(pd.Timedelta(minutes=config.diagnostic_horizon_hours * 60 - 1).value)
    if int(path.timestamps_ns[exit_pos]) != expected_exit_ns:
        return None
    entry_price = float(path.open[entry_pos])
    exit_price = float(path.close[exit_pos])
    highs = np.asarray(path.high[entry_pos : exit_pos + 1], dtype=float)
    lows = np.asarray(path.low[entry_pos : exit_pos + 1], dtype=float)
    if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(exit_price):
        return None
    if len(highs) != config.diagnostic_horizon_hours * 60 or not np.isfinite(highs).all() or not np.isfinite(lows).all():
        return None
    entry_time = pd.Timestamp(entry_ns, unit="ns")
    exit_time = pd.Timestamp(int(path.timestamps_ns[exit_pos]), unit="ns")
    return {
        "event_id": event.event_id,
        "fold_id": fold_id,
        "scope": scope,
        "decision_time": pd.Timestamp(event.decision_time_ns, unit="ns"),
        "entry_time": entry_time,
        "exit_time": exit_time,
        "delay_minutes": int(delay_minutes),
        "signal_quantile": float(event.signal_quantile),
        "score": float(event.score),
        "score_percentile": float(score_percentile),
        "score_band": "q90_plus" if score_percentile >= 0.90 else "q70_to_q90",
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return": float(exit_price / entry_price - 1.0),
        "mfe": float(np.max(highs) / entry_price - 1.0),
        "mae": float(np.min(lows) / entry_price - 1.0),
        "holding_minutes": int(config.diagnostic_horizon_hours * 60),
        "year": int(entry_time.year),
        "quarter": str(entry_time.to_period("Q")),
        "month": str(entry_time.to_period("M")),
    }


def enforce_non_overlap(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if frame.empty:
        return frame.copy(), 0
    work = frame.sort_values(["entry_time", "decision_time", "score"], ascending=[True, True, False]).copy()
    keep: list[int] = []
    last_exit: pd.Timestamp | None = None
    skipped = 0
    for index, row in work.iterrows():
        entry_time = pd.Timestamp(row["entry_time"])
        if last_exit is not None and entry_time <= last_exit:
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


def longest_losing_streak(values: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(values, dtype=float):
        if value < 0:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def summarize(frame: pd.DataFrame, *, cost_multiplier: float, config: Q70CrossYearAuditConfig) -> dict[str, object]:
    if frame.empty:
        return {"trades": 0, "mean_net_return": np.nan, "profit_factor": np.nan}
    work = frame.sort_values("entry_time").copy()
    work["net_return"] = work["gross_return"].astype(float) - config.base_round_trip_cost * float(cost_multiplier)
    net = work["net_return"].to_numpy(dtype=float)
    winners = net[net > 0]
    losers = net[net < 0]
    mdd, total = maximum_drawdown(net)
    positive_profit = float(winners.sum())
    sorted_net = np.sort(net)[::-1]
    top_count = min(10, len(sorted_net))
    top_share = float(sorted_net[:top_count].sum() / positive_profit) if positive_profit > 0 else np.nan
    without_top = sorted_net[top_count:] if len(sorted_net) > top_count else np.empty(0)
    return {
        "trades": int(len(work)),
        "mean_gross_return": float(work["gross_return"].mean()),
        "mean_net_return": float(net.mean()),
        "median_net_return": float(np.median(net)),
        "win_rate": float(np.mean(net > 0)),
        "profit_factor": profit_factor(net),
        "mean_winner": float(winners.mean()) if len(winners) else np.nan,
        "mean_loser": float(losers.mean()) if len(losers) else np.nan,
        "payoff_ratio": float(winners.mean() / abs(losers.mean())) if len(winners) and len(losers) else np.nan,
        "mean_mfe": float(work["mfe"].mean()),
        "mean_mae": float(work["mae"].mean()),
        "total_compounded_return": total,
        "max_drawdown": mdd,
        "top10_profit_share": top_share,
        "mean_net_without_top10": float(without_top.mean()) if len(without_top) else np.nan,
        "longest_losing_streak": longest_losing_streak(net),
    }


def build_tables(
    trades: pd.DataFrame,
    *,
    overlap_audit: pd.DataFrame,
    config: Q70CrossYearAuditConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, overlap_audit
    summary_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []
    band_rows: list[dict[str, object]] = []
    score_rows: list[dict[str, object]] = []
    keys = ["fold_id", "scope", "delay_minutes"]
    for values, group in trades.groupby(keys, sort=False):
        common = dict(zip(keys, values, strict=True))
        for cost in config.cost_multipliers:
            summary_rows.append({**common, "cost_multiplier": cost, **summarize(group, cost_multiplier=cost, config=config)})
        for period_kind, column in (("quarter", "quarter"), ("month", "month")):
            for period, part in group.groupby(column, sort=True):
                for cost in (1.0, 2.0):
                    period_rows.append({**common, "period_kind": period_kind, "period": period, "cost_multiplier": cost, **summarize(part, cost_multiplier=cost, config=config)})
        if common["scope"] == "broad_q70":
            for band, part in group.groupby("score_band", sort=True):
                for cost in config.cost_multipliers:
                    band_rows.append({**common, "score_band": band, "cost_multiplier": cost, **summarize(part, cost_multiplier=cost, config=config)})
            deciles = pd.qcut(group["score_percentile"].rank(method="first"), q=min(10, len(group)), labels=False, duplicates="drop")
            for decile, part in group.groupby(deciles, sort=True):
                score_rows.append({**common, "score_decile": int(decile) + 1, **summarize(part, cost_multiplier=2.0, config=config)})
    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(period_rows),
        pd.DataFrame(band_rows),
        pd.DataFrame(score_rows),
        overlap_audit,
    )


def comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    base = summary.loc[(summary["delay_minutes"] == 1) & (summary["cost_multiplier"] == 2.0)].copy()
    rows: list[dict[str, object]] = []
    for fold_id, group in base.groupby("fold_id", sort=True):
        by_scope = group.set_index("scope")
        if not {"broad_q70", "primary_q90"}.issubset(by_scope.index):
            continue
        q70 = by_scope.loc["broad_q70"]
        q90 = by_scope.loc["primary_q90"]
        rows.append(
            {
                "fold_id": fold_id,
                "q70_trades": int(q70.trades),
                "q90_trades": int(q90.trades),
                "added_trades": int(q70.trades - q90.trades),
                "trade_increase": float(q70.trades / q90.trades - 1.0),
                "q70_mean_net": float(q70.mean_net_return),
                "q90_mean_net": float(q90.mean_net_return),
                "expectancy_delta": float(q70.mean_net_return - q90.mean_net_return),
                "q70_pf": float(q70.profit_factor),
                "q90_pf": float(q90.profit_factor),
                "q70_total_return": float(q70.total_compounded_return),
                "q90_total_return": float(q90.total_compounded_return),
                "total_return_delta": float(q70.total_compounded_return - q90.total_compounded_return),
                "q70_mdd": float(q70.max_drawdown),
                "q90_mdd": float(q90.max_drawdown),
            }
        )
    return pd.DataFrame(rows)
