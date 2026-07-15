#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic long-only replay for respected-macro first-sweep signals.

The module is deliberately strategy-agnostic: callers provide already-frozen
signals and a small predeclared execution specification.  Signal generation is
outside this module.  Entry is a future bar open, exits use only bars after the
signal, and an ambiguous same-bar TP/SL touch is resolved stop-first.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

EPS = 1e-12




@dataclass(frozen=True)
class PreparedBars:
    index: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray


def prepare_bars(bars: pd.DataFrame) -> PreparedBars:
    required = {"open", "high", "low", "close"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise RuntimeError(f"backtest bars missing fields: {missing}")
    frame = bars.sort_index()
    index = pd.DatetimeIndex(frame.index)
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise RuntimeError("backtest bars index must be unique and increasing")
    return PreparedBars(
        index=index,
        open=pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float),
        high=pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float),
        low=pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float),
        close=pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float),
    )


@dataclass(frozen=True)
class ExitSpec:
    spec_id: str
    horizon_bars: int
    target_pct: float | None
    stop_mode: str  # structural | fixed_pct | none
    stop_value: float = 0.0  # structural buffer bp or fixed stop pct


@dataclass(frozen=True)
class CostSpec:
    entry_fee_rate: float = 0.00055
    exit_fee_rate: float = 0.00055
    entry_slippage_pct: float = 0.00020
    exit_slippage_pct: float = 0.00020
    cost_multiplier: float = 1.0


@dataclass(frozen=True)
class AddOnSpec:
    enabled: bool = False
    initial_weight: float = 0.5
    add_weight: float = 0.5
    checkpoint_offset: int = 1
    minimum_score: float = 70.0
    maximum_chase_pct: float = 0.0025


def default_exit_specs() -> tuple[ExitSpec, ...]:
    """Small predeclared grid; no optimizer-generated stop sweep."""

    return (
        ExitSpec("TP1_STRUCT_B00_H60", 60, 0.0100, "structural", 0.0),
        ExitSpec("TP1_STRUCT_B10_H60", 60, 0.0100, "structural", 10.0),
        ExitSpec("TP1_STRUCT_B25_H60", 60, 0.0100, "structural", 25.0),
        ExitSpec("TP1_FIXED_050_H60", 60, 0.0100, "fixed_pct", 0.0050),
        ExitSpec("TP1_FIXED_075_H60", 60, 0.0100, "fixed_pct", 0.0075),
        ExitSpec("TP1_FIXED_100_H60", 60, 0.0100, "fixed_pct", 0.0100),
        ExitSpec("TP1_STRUCT_B10_H180", 180, 0.0100, "structural", 10.0),
        ExitSpec("TP1_FIXED_075_H180", 180, 0.0100, "fixed_pct", 0.0075),
        ExitSpec("TIME_ONLY_H60", 60, None, "none", 0.0),
        ExitSpec("TIME_ONLY_H180", 180, None, "none", 0.0),
    )


def _as_float(value: object, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _stop_price(event: pd.Series, raw_entry: float, spec: ExitSpec) -> float | None:
    if spec.stop_mode == "none":
        return None
    if spec.stop_mode == "fixed_pct":
        return raw_entry * (1.0 - float(spec.stop_value))
    if spec.stop_mode == "structural":
        sweep_low = _as_float(event.get("sweep_low", event.get("origin_sweep_low", np.nan)))
        if not np.isfinite(sweep_low) or sweep_low <= 0.0:
            return None
        stop = sweep_low * (1.0 - float(spec.stop_value) / 10_000.0)
        return stop if stop < raw_entry else None
    raise ValueError(f"unsupported stop_mode={spec.stop_mode!r}")


def _execution_entry(raw_price: float, costs: CostSpec) -> float:
    return float(raw_price) * (1.0 + float(costs.entry_slippage_pct) * float(costs.cost_multiplier))


def _execution_exit(raw_price: float, costs: CostSpec) -> float:
    return float(raw_price) * (1.0 - float(costs.exit_slippage_pct) * float(costs.cost_multiplier))


def _weighted_return(
    legs: Sequence[tuple[float, float]],
    raw_exit_price: float,
    costs: CostSpec,
) -> tuple[float, float, float, float]:
    total_weight = float(sum(weight for weight, _ in legs))
    if total_weight <= EPS:
        return np.nan, np.nan, np.nan, np.nan
    exit_exec = _execution_exit(raw_exit_price, costs)
    gross = float(sum(weight * (exit_exec / entry_exec - 1.0) for weight, entry_exec in legs))
    entry_fee = total_weight * float(costs.entry_fee_rate) * float(costs.cost_multiplier)
    exit_fee = total_weight * float(costs.exit_fee_rate) * float(costs.cost_multiplier)
    net = gross - entry_fee - exit_fee
    avg_entry = float(sum(weight * entry_exec for weight, entry_exec in legs) / total_weight)
    return gross, net, avg_entry, exit_exec


def replay_trade(
    bars: pd.DataFrame | PreparedBars,
    event: pd.Series,
    *,
    exit_spec: ExitSpec,
    costs: CostSpec,
    entry_delay_bars: int = 0,
    add_on: AddOnSpec | None = None,
) -> dict[str, object]:
    """Replay one long trade.

    ``extreme_pos`` is the closed signal bar.  Baseline entry is the next open,
    therefore ``entry_delay_bars=0`` maps to ``extreme_pos + 1``.  Delay stress
    adds whole bars beyond that baseline.
    """

    prepared = prepare_bars(bars) if isinstance(bars, pd.DataFrame) else bars
    if len(prepared.index) == 0:
        return {"valid": False, "invalid_reason": "empty_bars"}
    opens, highs, lows, closes, index = prepared.open, prepared.high, prepared.low, prepared.close, prepared.index

    signal_pos = int(event["extreme_pos"])
    entry_pos = signal_pos + 1 + int(entry_delay_bars)
    planned_exit_pos = entry_pos + int(exit_spec.horizon_bars) - 1
    if entry_pos < 0 or planned_exit_pos >= len(index) or entry_pos >= planned_exit_pos + 1:
        return {"valid": False, "invalid_reason": "insufficient_future_bars"}
    raw_entry = float(opens[entry_pos])
    if not np.isfinite(raw_entry) or raw_entry <= 0.0:
        return {"valid": False, "invalid_reason": "invalid_entry_open"}

    add_on = add_on or AddOnSpec(enabled=False, initial_weight=1.0, add_weight=0.0)
    initial_weight = float(add_on.initial_weight if add_on.enabled else 1.0)
    legs: list[tuple[float, float]] = [(initial_weight, _execution_entry(raw_entry, costs))]
    add_pos = signal_pos + int(add_on.checkpoint_offset) + 1
    add_score = _as_float(event.get("add_on_opportunity_score", np.nan))
    add_state_ok = bool(event.get("add_on_eligible", False))
    add_filled = False

    stop = _stop_price(event, raw_entry, exit_spec)
    if exit_spec.stop_mode != "none" and stop is None:
        return {"valid": False, "invalid_reason": "invalid_stop_geometry"}
    target = raw_entry * (1.0 + float(exit_spec.target_pct)) if exit_spec.target_pct is not None else None

    def _mark_to_market(raw_price: float) -> float:
        return float(sum(weight * (float(raw_price) / entry_exec - 1.0) for weight, entry_exec in legs))

    exit_pos = planned_exit_pos
    raw_exit = float(closes[planned_exit_pos])
    exit_reason = f"time_h{int(exit_spec.horizon_bars)}"
    same_bar_both = False
    stop_hit = False
    tp_hit = False

    mtm_low: list[float] = []
    mtm_high: list[float] = []
    mtm_close: list[float] = []

    for pos in range(entry_pos, planned_exit_pos + 1):
        # A delayed add is decided after its checkpoint close and enters at the
        # following open.  The existing position can still stop at that open.
        if add_on.enabled and not add_filled and pos == add_pos and pos > entry_pos:
            chase = float(opens[pos] / raw_entry - 1.0)
            if (
                add_state_ok
                and np.isfinite(add_score)
                and add_score >= float(add_on.minimum_score)
                and chase <= float(add_on.maximum_chase_pct)
                and (stop is None or float(opens[pos]) > stop)
            ):
                legs.append((float(add_on.add_weight), _execution_entry(float(opens[pos]), costs)))
                add_filled = True

        # Gap-through stop is executed at the worse bar open.  A standing TP
        # limit is conservatively credited only at its target.
        if stop is not None and float(opens[pos]) <= stop:
            stop_hit = True
            exit_reason = "stop_gap"
            raw_exit = float(opens[pos])
            exit_pos = pos
            mtm_low.append(_mark_to_market(raw_exit))
            break
        if target is not None and float(opens[pos]) >= target:
            tp_hit = True
            exit_reason = "take_profit_gap"
            raw_exit = float(target)
            exit_pos = pos
            mtm_high.append(_mark_to_market(raw_exit))
            break

        hit_stop = stop is not None and float(lows[pos]) <= stop
        hit_tp = target is not None and float(highs[pos]) >= target
        if hit_stop and hit_tp:
            # Intrabar order is unknowable in 1m OHLC.  Resolve against the
            # long strategy to avoid optimistic path assumptions.
            same_bar_both = True
            stop_hit = True
            exit_reason = "stop_first_same_bar_both"
            raw_exit = float(stop)
            exit_pos = pos
            mtm_low.append(_mark_to_market(raw_exit))
            break
        if hit_stop:
            stop_hit = True
            exit_reason = "stop"
            raw_exit = float(stop)
            exit_pos = pos
            mtm_low.append(_mark_to_market(raw_exit))
            break
        if hit_tp:
            tp_hit = True
            exit_reason = "take_profit"
            raw_exit = float(target)
            exit_pos = pos
            mtm_low.append(_mark_to_market(float(lows[pos])))
            mtm_high.append(_mark_to_market(raw_exit))
            break

        mtm_low.append(_mark_to_market(float(lows[pos])))
        mtm_high.append(_mark_to_market(float(highs[pos])))
        mtm_close.append(_mark_to_market(float(closes[pos])))

    gross, net, avg_entry, exit_exec = _weighted_return(legs, raw_exit, costs)
    total_weight = float(sum(weight for weight, _ in legs))
    return {
        "valid": True,
        "event_id": str(event.get("event_id", "")),
        "origin_event_id": str(event.get("origin_event_id", event.get("event_id", ""))),
        "signal_time": index[signal_pos],
        "entry_time": index[entry_pos],
        "exit_time": index[exit_pos],
        "signal_pos": signal_pos,
        "entry_pos": entry_pos,
        "exit_pos": exit_pos,
        "entry_delay_bars": int(entry_delay_bars),
        "bars_held": int(exit_pos - entry_pos + 1),
        "exit_spec_id": exit_spec.spec_id,
        "horizon_bars": int(exit_spec.horizon_bars),
        "target_pct": float(exit_spec.target_pct) if exit_spec.target_pct is not None else np.nan,
        "stop_mode": exit_spec.stop_mode,
        "stop_value": float(exit_spec.stop_value),
        "stop_price": float(stop) if stop is not None else np.nan,
        "tp_price": float(target) if target is not None else np.nan,
        "raw_initial_entry_price": raw_entry,
        "avg_entry_price": avg_entry,
        "raw_exit_price": raw_exit,
        "exit_price": exit_exec,
        "exit_reason": exit_reason,
        "stop_hit": stop_hit,
        "tp_hit": tp_hit,
        "same_bar_stop_tp_both_hit_flag": same_bar_both,
        "add_on_enabled": bool(add_on.enabled),
        "add_on_filled": bool(add_filled),
        "filled_weight": total_weight,
        "leg_count": len(legs),
        "gross_return": gross,
        "net_return": net,
        "gross_return_per_filled": gross / total_weight if total_weight > EPS else np.nan,
        "net_return_per_filled": net / total_weight if total_weight > EPS else np.nan,
        "mae": float(np.nanmin(mtm_low)) if mtm_low else min(net, 0.0),
        "mfe": float(np.nanmax(mtm_high)) if mtm_high else max(net, 0.0),
        "mae_per_filled": (float(np.nanmin(mtm_low)) / total_weight) if mtm_low and total_weight > EPS else np.nan,
        "mfe_per_filled": (float(np.nanmax(mtm_high)) / total_weight) if mtm_high and total_weight > EPS else np.nan,
        "min_close_mtm": float(np.nanmin(mtm_close)) if mtm_close else min(net, 0.0),
        "cost_multiplier": float(costs.cost_multiplier),
        "opportunity_score": _as_float(event.get("opportunity_score", np.nan)),
        "add_on_opportunity_score": add_score,
        "sweep_low": _as_float(event.get("sweep_low", np.nan)),
        "level_price": _as_float(event.get("level_price", np.nan)),
    }


def deduplicate_signals(events: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep the highest score when multiple levels signal the same next open."""

    if events.empty:
        return events.copy(), 0
    out = events.copy()
    out["_entry_key"] = pd.to_datetime(out["extreme_time"]) + pd.Timedelta(minutes=1)
    out = out.sort_values(["_entry_key", "opportunity_score", "event_id"], ascending=[True, False, True], kind="mergesort")
    before = len(out)
    out = out.drop_duplicates("_entry_key", keep="first").drop(columns="_entry_key").reset_index(drop=True)
    return out, before - len(out)


def enforce_single_position(trades: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if trades.empty:
        return trades.copy(), 0
    ordered = trades.sort_values(["entry_time", "opportunity_score", "event_id"], ascending=[True, False, True], kind="mergesort")
    kept: list[int] = []
    last_exit = pd.Timestamp.min
    skipped = 0
    for idx, row in ordered.iterrows():
        entry = pd.Timestamp(row["entry_time"])
        if entry <= last_exit:
            skipped += 1
            continue
        kept.append(idx)
        last_exit = pd.Timestamp(row["exit_time"])
    return ordered.loc[kept].reset_index(drop=True), skipped


def _profit_factor(values: pd.Series) -> float:
    x = pd.to_numeric(values, errors="coerce").dropna()
    wins = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    if losses <= EPS:
        return np.inf if wins > 0 else np.nan
    return wins / losses


def _max_consecutive_losses(values: Iterable[float]) -> int:
    best = current = 0
    for value in values:
        if float(value) < 0.0:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _top_winner_share(values: pd.Series, count: int = 5) -> float:
    positive = pd.to_numeric(values, errors="coerce").dropna()
    positive = positive[positive > 0].sort_values(ascending=False)
    total = float(positive.sum())
    return float(positive.head(count).sum() / total) if total > EPS else np.nan


def summarize_trades(
    trades: pd.DataFrame,
    *,
    months: int,
    capital_fraction: float = 0.10,
    starting_equity: float = 1.0,
    raw_signals: int | None = None,
    skipped_overlap: int = 0,
    deduplicated_signals: int = 0,
) -> dict[str, object]:
    if trades.empty:
        return {
            "raw_signals": int(raw_signals or 0), "trades": 0, "events_per_month": 0.0,
            "skipped_overlap": int(skipped_overlap), "deduplicated_signals": int(deduplicated_signals),
        }
    x = pd.to_numeric(trades["net_return"], errors="coerce").fillna(0.0)
    account_r = x * float(capital_fraction)
    equity = float(starting_equity) * (1.0 + account_r).cumprod()
    peaks = equity.cummax()
    drawdown = equity / peaks - 1.0
    wins = x[x > 0]
    losses = x[x < 0]
    entry_times = pd.to_datetime(trades["entry_time"]).sort_values()
    day_gaps = entry_times.diff().dt.total_seconds().div(86400.0).dropna()
    by_day = trades.assign(_date=pd.to_datetime(trades["entry_time"]).dt.date).groupby("_date").size()
    return {
        "raw_signals": int(raw_signals if raw_signals is not None else len(trades)),
        "trades": int(len(trades)),
        "events_per_month": float(len(trades) / max(int(months), 1)),
        "skipped_overlap": int(skipped_overlap),
        "deduplicated_signals": int(deduplicated_signals),
        "mean_net_return": float(x.mean()),
        "median_net_return": float(x.median()),
        "win_rate": float((x > 0).mean()),
        "profit_factor": _profit_factor(x),
        "avg_win": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss": float(losses.mean()) if len(losses) else np.nan,
        "payoff_ratio": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) and losses.mean() < 0 else np.nan,
        "position_return_sum": float(x.sum()),
        "account_total_return": float(equity.iloc[-1] / float(starting_equity) - 1.0),
        "account_max_drawdown": float(drawdown.min()),
        "max_consecutive_losses": _max_consecutive_losses(x),
        "top5_winner_share": _top_winner_share(x, 5),
        "return_std": float(x.std(ddof=0)),
        "mean_to_std": float(x.mean() / x.std(ddof=0)) if float(x.std(ddof=0)) > EPS else np.nan,
        "max_days_without_trade": float(day_gaps.max()) if len(day_gaps) else np.nan,
        "worst_trade": float(x.min()),
        "best_trade": float(x.max()),
        "median_mae": float(pd.to_numeric(trades["mae"], errors="coerce").median()),
        "p90_adverse": float((-pd.to_numeric(trades["mae"], errors="coerce")).quantile(0.90)),
        "median_mfe": float(pd.to_numeric(trades["mfe"], errors="coerce").median()),
        "tp_rate": float(pd.to_numeric(trades["tp_hit"], errors="coerce").mean()),
        "stop_rate": float(pd.to_numeric(trades["stop_hit"], errors="coerce").mean()),
        "same_bar_both_rate": float(pd.to_numeric(trades["same_bar_stop_tp_both_hit_flag"], errors="coerce").mean()),
        "add_fill_rate": float(pd.to_numeric(trades["add_on_filled"], errors="coerce").mean()),
        "median_bars_held": float(pd.to_numeric(trades["bars_held"], errors="coerce").median()),
        "max_day_event_share": float(by_day.max() / len(trades)) if len(by_day) else np.nan,
        "top5_day_event_share": float(by_day.nlargest(5).sum() / len(trades)) if len(by_day) else np.nan,
    }


def remove_strongest_days(trades: pd.DataFrame, count: int) -> pd.DataFrame:
    if trades.empty or int(count) <= 0:
        return trades.copy()
    out = trades.copy()
    out["_date"] = pd.to_datetime(out["entry_time"]).dt.date
    daily = out.groupby("_date", sort=False)["net_return"].sum().sort_values(ascending=False)
    remove = set(daily.head(int(count)).index)
    return out.loc[~out["_date"].isin(remove)].drop(columns="_date").reset_index(drop=True)
