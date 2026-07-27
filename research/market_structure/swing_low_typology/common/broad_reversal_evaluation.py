#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Close-only path labels, deterministic execution, and uncertainty metrics.

All labels start at the next bar open and inspect future *closed-bar closes*.
The primary execution is label-aligned: the first closed bar at or above +1%
exits at the predeclared target; otherwise the trade exits at the horizon close.
No ordinary stop is used in research 16's first pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from src.research_common.progress import ProgressReporter
except Exception:  # pragma: no cover
    ProgressReporter = None  # type: ignore[assignment]

from research.market_structure.swing_low_typology.common.deployable_first_sweep_backtest import (
    enforce_single_position,
    remove_strongest_days,
    summarize_trades,
)

EPS = 1e-12


@dataclass(frozen=True)
class CloseTargetCostSpec:
    entry_fee_rate: float = 0.00055
    exit_fee_rate: float = 0.00055
    entry_slippage_pct: float = 0.00020
    exit_slippage_pct: float = 0.00020
    cost_multiplier: float = 1.0


def _token(value: float) -> str:
    text = f"{float(value):g}"
    return text.replace("-", "m").replace(".", "p")


def build_multi_horizon_close_labels(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizons: Sequence[int] = (30, 60, 180),
    target_levels_pct: Sequence[float] = (0.5, 1.0, 1.5, 2.0),
    adverse_level_pct: float = 0.5,
    vectorized_chunk_size: int = 20_000,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Build one vectorized close-only path atlas for all requested horizons."""

    horizons_ = tuple(sorted(set(int(value) for value in horizons)))
    targets = tuple(sorted(set(float(value) for value in target_levels_pct)))
    if not horizons_ or horizons_[0] < 1:
        raise ValueError("horizons must be positive")
    if not targets or targets[0] <= 0 or float(adverse_level_pct) <= 0:
        raise ValueError("target/adverse levels must be positive")
    if vectorized_chunk_size < 1:
        raise ValueError("vectorized_chunk_size must be >= 1")
    if events.empty:
        return pd.DataFrame()

    index = pd.DatetimeIndex(bars.index)
    open_values = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float, copy=False)
    close_values = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float, copy=False)
    maximum_horizon = max(horizons_)
    windows = np.lib.stride_tricks.sliding_window_view(close_values, maximum_horizon)
    reporter = (
        ProgressReporter("[labels] multi-horizon close paths", total=len(events), every=max(1, int(vectorized_chunk_size)))
        if ProgressReporter is not None and show_progress
        else None
    )
    parts: list[pd.DataFrame] = []
    processed = 0
    for start in range(0, len(events), int(vectorized_chunk_size)):
        source = events.iloc[start : start + int(vectorized_chunk_size)]
        positions = pd.to_numeric(source["extreme_pos"], errors="coerce").to_numpy(dtype=np.int64)
        entry_positions = positions + 1
        valid = (entry_positions >= 0) & (entry_positions < len(windows))
        valid_index = np.flatnonzero(valid)
        if valid_index.size:
            valid_entries = entry_positions[valid_index]
            valid[valid_index] &= np.isfinite(open_values[valid_entries]) & (open_values[valid_entries] > EPS)
        if not valid.any():
            processed += len(source)
            if reporter is not None and processed < len(events):
                reporter.update(processed)
            continue
        chunk = source.iloc[np.flatnonzero(valid)].reset_index(drop=True)
        entry_positions = entry_positions[valid]
        entry = open_values[entry_positions]
        path_full = windows[entry_positions]
        finite = np.isfinite(path_full).all(axis=1)
        if not finite.all():
            chunk = chunk.iloc[np.flatnonzero(finite)].reset_index(drop=True)
            entry_positions = entry_positions[finite]
            entry = entry[finite]
            path_full = path_full[finite]
        if not len(chunk):
            processed += len(source)
            continue

        output: dict[str, object] = {
            "event_id": chunk["event_id"].to_numpy(),
            "entry_time": index[entry_positions],
            "entry_price": entry,
            "label_end_time": index[entry_positions + maximum_horizon - 1],
        }
        for horizon in horizons_:
            path = path_full[:, :horizon]
            returns = path / entry[:, None] - 1.0
            maximum = np.max(returns, axis=1)
            minimum = np.min(returns, axis=1)
            terminal = returns[:, -1]
            output[f"mfe_h{horizon}_pct"] = (np.maximum(maximum, 0.0) * 100.0).astype(np.float32)
            output[f"mae_h{horizon}_pct"] = (np.maximum(-minimum, 0.0) * 100.0).astype(np.float32)
            output[f"terminal_h{horizon}_pct"] = (terminal * 100.0).astype(np.float32)
            output[f"mfe_mae_ratio_h{horizon}"] = np.divide(
                np.maximum(maximum, 0.0),
                np.maximum(-minimum, 0.0),
                out=np.full(len(chunk), np.nan, dtype=float),
                where=np.maximum(-minimum, 0.0) > EPS,
            ).astype(np.float32)

            adverse_mask = returns <= -float(adverse_level_pct) / 100.0
            adverse_hit = adverse_mask.any(axis=1)
            adverse_first = np.argmax(adverse_mask, axis=1)
            tp1_hit: np.ndarray | None = None
            tp1_first: np.ndarray | None = None
            for target_pct in targets:
                token = _token(target_pct)
                target_mask = returns >= float(target_pct) / 100.0
                hit = target_mask.any(axis=1)
                first = np.argmax(target_mask, axis=1)
                output[f"tp_{token}_h{horizon}"] = hit
                output[f"time_to_tp_{token}_h{horizon}"] = np.where(hit, first + 1, np.nan).astype(np.float32)
                before = np.arange(horizon)[None, :] <= first[:, None]
                mae_before = np.where(
                    hit,
                    np.maximum(-np.min(np.where(before, returns, np.nan), axis=1), 0.0) * 100.0,
                    np.nan,
                )
                output[f"mae_before_tp_{token}_h{horizon}_pct"] = mae_before.astype(np.float32)
                if abs(target_pct - 1.0) < 1e-12:
                    tp1_hit = hit
                    tp1_first = first
            if tp1_hit is None or tp1_first is None:
                raise RuntimeError("target_levels_pct must include 1.0 for aligned execution diagnostics")
            output[f"clean_0p5_h{horizon}"] = tp1_hit & (~adverse_hit | (tp1_first < adverse_first))
            output[f"deep_sweep_recovery_h{horizon}"] = tp1_hit & (minimum <= -0.0100)
            output[f"permanent_failure_h{horizon}"] = (~tp1_hit) & (terminal <= 0.0)

        parts.append(pd.DataFrame(output))
        processed += len(source)
        if reporter is not None and processed < len(events):
            reporter.update(processed)
    if reporter is not None:
        reporter.close()
    if not parts:
        return pd.DataFrame()
    result = pd.concat(parts, ignore_index=True)
    if result["event_id"].duplicated().any():
        raise RuntimeError("multi-horizon label builder produced duplicate event_id")
    return result


def replay_close_target_events(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon_bars: int,
    target_pct: float = 0.0100,
    costs: CloseTargetCostSpec | None = None,
    entry_delay_bars: int = 0,
) -> pd.DataFrame:
    """Vectorized label-aligned execution with no ordinary stop."""

    if horizon_bars < 1 or target_pct <= 0:
        raise ValueError("invalid horizon/target")
    if events.empty:
        return pd.DataFrame()
    costs = costs or CloseTargetCostSpec()
    frame = bars.sort_index()
    index = pd.DatetimeIndex(frame.index)
    open_values = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float, copy=False)
    close_values = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float, copy=False)
    windows = np.lib.stride_tricks.sliding_window_view(close_values, int(horizon_bars))

    positions = pd.to_numeric(events["extreme_pos"], errors="coerce").to_numpy(dtype=np.int64)
    entry_positions = positions + 1 + int(entry_delay_bars)
    valid = (entry_positions >= 0) & (entry_positions < len(windows))
    valid_index = np.flatnonzero(valid)
    if valid_index.size:
        valid_entries = entry_positions[valid_index]
        valid[valid_index] &= np.isfinite(open_values[valid_entries]) & (open_values[valid_entries] > EPS)
    selected_indices = np.flatnonzero(valid)
    source = events.iloc[selected_indices].reset_index(drop=True)
    signal_positions = positions[selected_indices]
    entry_positions = entry_positions[selected_indices]
    if source.empty:
        return pd.DataFrame()
    raw_entry = open_values[entry_positions]
    path = windows[entry_positions]
    finite = np.isfinite(path).all(axis=1)
    source = source.iloc[np.flatnonzero(finite)].reset_index(drop=True)
    signal_positions = signal_positions[finite]
    entry_positions = entry_positions[finite]
    raw_entry = raw_entry[finite]
    path = path[finite]
    if source.empty:
        return pd.DataFrame()

    threshold = raw_entry[:, None] * (1.0 + float(target_pct))
    hit_mask = path >= threshold
    hit = hit_mask.any(axis=1)
    first = np.argmax(hit_mask, axis=1)
    exit_offsets = np.where(hit, first, int(horizon_bars) - 1).astype(np.int64)
    exit_positions = entry_positions + exit_offsets
    raw_exit = np.where(hit, raw_entry * (1.0 + float(target_pct)), path[np.arange(len(path)), exit_offsets])

    multiplier = float(costs.cost_multiplier)
    entry_exec = raw_entry * (1.0 + float(costs.entry_slippage_pct) * multiplier)
    exit_exec = raw_exit * (1.0 - float(costs.exit_slippage_pct) * multiplier)
    gross = exit_exec / entry_exec - 1.0
    net = gross - (float(costs.entry_fee_rate) + float(costs.exit_fee_rate)) * multiplier
    close_returns = path / raw_entry[:, None] - 1.0
    mae = np.minimum(np.min(close_returns, axis=1), 0.0)
    mfe = np.maximum(np.max(close_returns, axis=1), 0.0)

    score = pd.to_numeric(source.get("opportunity_score", pd.Series(np.nan, index=source.index)), errors="coerce")
    output = pd.DataFrame(
        {
            "valid": True,
            "event_id": source["event_id"].astype(str).to_numpy(),
            "origin_event_id": source.get("origin_event_id", source["event_id"]).astype(str).to_numpy(),
            "signal_time": index[signal_positions],
            "entry_time": index[entry_positions],
            "exit_time": index[exit_positions],
            "signal_pos": signal_positions,
            "entry_pos": entry_positions,
            "exit_pos": exit_positions,
            "entry_delay_bars": int(entry_delay_bars),
            "bars_held": exit_offsets + 1,
            "exit_spec_id": f"CLOSE_TP1_H{int(horizon_bars)}",
            "horizon_bars": int(horizon_bars),
            "target_pct": float(target_pct),
            "stop_mode": "none",
            "stop_value": 0.0,
            "stop_price": np.nan,
            "tp_price": raw_entry * (1.0 + float(target_pct)),
            "raw_initial_entry_price": raw_entry,
            "avg_entry_price": entry_exec,
            "raw_exit_price": raw_exit,
            "exit_price": exit_exec,
            "exit_reason": np.where(hit, "take_profit_on_closed_bar", f"time_h{int(horizon_bars)}"),
            "stop_hit": False,
            "tp_hit": hit,
            "same_bar_stop_tp_both_hit_flag": False,
            "add_on_enabled": False,
            "add_on_filled": False,
            "filled_weight": 1.0,
            "leg_count": 1,
            "gross_return": gross,
            "net_return": net,
            "gross_return_per_filled": gross,
            "net_return_per_filled": net,
            "mae": mae,
            "mfe": mfe,
            "mae_per_filled": mae,
            "mfe_per_filled": mfe,
            "min_close_mtm": mae,
            "cost_multiplier": multiplier,
            "opportunity_score": score.to_numpy(dtype=float),
            "add_on_opportunity_score": np.nan,
            "sweep_low": pd.to_numeric(source.get("sweep_low", pd.Series(np.nan, index=source.index)), errors="coerce").to_numpy(dtype=float),
            "level_price": pd.to_numeric(source.get("level_price", pd.Series(np.nan, index=source.index)), errors="coerce").to_numpy(dtype=float),
        }
    )
    for column in ("fold", "baseline_layer", "mechanism", "primary_mechanism", "policy_id", "model_group", "replicate"):
        if column in source.columns:
            output[column] = source[column].to_numpy()
    if not (pd.to_datetime(output["entry_time"]) > pd.to_datetime(output["signal_time"])).all():
        raise RuntimeError("execution violated closed-bar signal -> future open entry")
    return output


def executable_trade_set(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon_bars: int,
    costs: CloseTargetCostSpec,
    entry_delay_bars: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    isolated = replay_close_target_events(
        bars,
        events,
        horizon_bars=int(horizon_bars),
        target_pct=0.0100,
        costs=costs,
        entry_delay_bars=int(entry_delay_bars),
    )
    portfolio, skipped = enforce_single_position(isolated)
    return portfolio, {
        "raw_signals": int(len(events)),
        "deduplicated_signals": 0,
        "skipped_overlap": int(skipped),
    }


def path_metrics(events: pd.DataFrame, *, horizon: int, months: int) -> dict[str, object]:
    if events.empty:
        return {"events": 0, "events_per_month": 0.0}
    row: dict[str, object] = {
        "events": int(len(events)),
        "events_per_month": float(len(events) / max(1, int(months))),
    }
    for target in (0.5, 1.0, 1.5, 2.0):
        column = f"tp_{_token(target)}_h{int(horizon)}"
        row[f"tp_{_token(target)}_rate"] = float(events[column].astype(bool).mean())
    for column, output in (
        (f"mfe_h{horizon}_pct", "median_mfe_pct"),
        (f"mae_h{horizon}_pct", "median_mae_pct"),
        (f"mae_before_tp_1_h{horizon}_pct", "median_mae_before_tp_pct"),
        (f"mfe_mae_ratio_h{horizon}", "median_mfe_mae_ratio"),
        (f"time_to_tp_1_h{horizon}", "median_time_to_tp_bars"),
    ):
        values = pd.to_numeric(events[column], errors="coerce")
        row[output] = float(values.median()) if values.notna().any() else np.nan
    row["clean_0p5_rate"] = float(events[f"clean_0p5_h{horizon}"].astype(bool).mean())
    row["deep_sweep_recovery_rate"] = float(events[f"deep_sweep_recovery_h{horizon}"].astype(bool).mean())
    row["permanent_failure_rate"] = float(events[f"permanent_failure_h{horizon}"].astype(bool).mean())
    return row


def _annualized_account_metrics(
    trades: pd.DataFrame,
    *,
    months: int,
    capital_fraction: float,
) -> dict[str, float]:
    if trades.empty:
        return {
            "net_expectancy_bps": np.nan,
            "daily_sharpe": np.nan,
            "cagr": np.nan,
            "calmar": np.nan,
            "account_total_return": np.nan,
            "account_max_drawdown": np.nan,
        }
    ordered = trades.sort_values(["entry_time", "event_id"], kind="mergesort")
    net = pd.to_numeric(ordered["net_return"], errors="coerce").fillna(0.0)
    account_return = net * float(capital_fraction)
    day = pd.to_datetime(ordered["entry_time"]).dt.floor("D")
    daily = (1.0 + account_return).groupby(day).prod() - 1.0
    if not daily.empty:
        full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
        daily = daily.reindex(full_index, fill_value=0.0)
    daily_std = float(daily.std(ddof=0)) if len(daily) else np.nan
    daily_sharpe = (
        float(daily.mean() / daily_std * np.sqrt(365.0))
        if len(daily) and np.isfinite(daily_std) and daily_std > EPS
        else np.nan
    )
    compounded = (1.0 + account_return).cumprod().to_numpy(dtype=float)
    equity = np.r_[1.0, compounded]
    total_return = float(equity[-1] - 1.0)
    years = max(float(months) / 12.0, 1.0 / 365.25)
    cagr = float((1.0 + total_return) ** (1.0 / years) - 1.0) if total_return > -1.0 else -1.0
    peaks = np.maximum.accumulate(equity)
    drawdown = equity / peaks - 1.0
    maximum_drawdown = float(np.min(drawdown))
    calmar = float(cagr / abs(maximum_drawdown)) if maximum_drawdown < -EPS else np.nan
    return {
        "net_expectancy_bps": float(net.mean() * 10_000.0),
        "daily_sharpe": daily_sharpe,
        "cagr": cagr,
        "calmar": calmar,
        "account_total_return": total_return,
        "account_max_drawdown": maximum_drawdown,
    }


def execution_metrics(
    trades: pd.DataFrame,
    *,
    months: int,
    counts: dict[str, int],
    capital_fraction: float = 0.10,
) -> dict[str, object]:
    summary = summarize_trades(
        trades,
        months=max(1, int(months)),
        capital_fraction=float(capital_fraction),
        starting_equity=1.0,
        raw_signals=int(counts["raw_signals"]),
        skipped_overlap=int(counts["skipped_overlap"]),
        deduplicated_signals=int(counts["deduplicated_signals"]),
    )
    summary.update(
        _annualized_account_metrics(
            trades,
            months=max(1, int(months)),
            capital_fraction=float(capital_fraction),
        )
    )
    return summary


def strongest_day_stress(
    trades: pd.DataFrame,
    *,
    remove_days: int,
    months: int,
    raw_signals: int,
    capital_fraction: float,
) -> dict[str, object]:
    stressed = remove_strongest_days(trades, int(remove_days))
    return execution_metrics(
        stressed,
        months=max(1, int(months)),
        capital_fraction=float(capital_fraction),
        counts={
            "raw_signals": int(raw_signals),
            "skipped_overlap": 0,
            "deduplicated_signals": 0,
        },
    )


def bootstrap_day_metrics(
    frame: pd.DataFrame,
    *,
    horizon: int,
    replicates: int = 500,
    random_state: int = 42,
) -> pd.DataFrame:
    """Fast day-block bootstrap for TP, mean net return, and profit factor."""

    if frame.empty or replicates < 1:
        return pd.DataFrame()
    data = frame.copy()
    time_column = "entry_time" if "entry_time" in data.columns else "extreme_time"
    data["_day"] = pd.to_datetime(data[time_column]).dt.floor("D")
    target_column = f"tp_1_h{int(horizon)}"
    if target_column not in data.columns and "tp_hit" in data.columns:
        data[target_column] = data["tp_hit"]
    data["_tp"] = data.get(target_column, pd.Series(np.nan, index=data.index)).astype(float)
    net = pd.to_numeric(data.get("net_return", pd.Series(np.nan, index=data.index)), errors="coerce")
    data["_net"] = net
    data["_win_sum"] = net.where(net > 0.0, 0.0)
    data["_loss_sum"] = (-net.where(net < 0.0, 0.0))
    daily = data.groupby("_day", sort=True).agg(
        event_count=("_tp", "size"),
        tp_count=("_tp", "sum"),
        net_count=("_net", "count"),
        net_sum=("_net", "sum"),
        win_sum=("_win_sum", "sum"),
        loss_sum=("_loss_sum", "sum"),
    )
    if daily.empty:
        return pd.DataFrame()
    values = daily.to_numpy(dtype=float)
    rng = np.random.default_rng(int(random_state))
    sample = rng.integers(0, len(daily), size=(int(replicates), len(daily)))
    totals = values[sample].sum(axis=1)
    event_count = totals[:, 0]
    tp_rate = np.divide(totals[:, 1], event_count, out=np.full(int(replicates), np.nan), where=event_count > 0)
    net_count = totals[:, 2]
    mean_net = np.divide(totals[:, 3], net_count, out=np.full(int(replicates), np.nan), where=net_count > 0)
    pf = np.divide(totals[:, 4], totals[:, 5], out=np.full(int(replicates), np.nan), where=totals[:, 5] > EPS)
    return pd.DataFrame(
        {
            "bootstrap_replicate": np.arange(int(replicates), dtype=np.int32),
            "tp_1_rate": tp_rate,
            "mean_net_return": mean_net,
            "profit_factor": pf,
        }
    )


def bootstrap_interval(distribution: pd.DataFrame) -> dict[str, object]:
    row: dict[str, object] = {"bootstrap_replicates": int(len(distribution))}
    for metric in ("tp_1_rate", "mean_net_return", "profit_factor"):
        values = pd.to_numeric(distribution.get(metric), errors="coerce").dropna()
        row[f"{metric}_bootstrap_mean"] = float(values.mean()) if len(values) else np.nan
        row[f"{metric}_ci_low"] = float(values.quantile(0.025)) if len(values) else np.nan
        row[f"{metric}_ci_high"] = float(values.quantile(0.975)) if len(values) else np.nan
    return row


def matched_random_replicate_metrics(
    frame: pd.DataFrame,
    *,
    horizon: int,
) -> pd.DataFrame:
    if frame.empty or "replicate" not in frame.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for replicate, part in frame.groupby("replicate", sort=True):
        row = path_metrics(part, horizon=int(horizon), months=1)
        row["replicate"] = int(replicate)
        rows.append(row)
    return pd.DataFrame(rows)
