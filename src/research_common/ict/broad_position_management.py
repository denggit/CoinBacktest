#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Broad-universe causal position-management primitives for SOXL ICT R20.

The module deliberately preserves the entry universe.  It studies what happens
*after* a broad first-visible MSS market entry instead of adding setup filters.
All protective-stop changes based on intrabar progress become active only from
the next 1m bar.  Resting partial/target orders may fill intrabar, while the
initial/protective stop wins same-minute ambiguity conservatively.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .premarket_mss_fvg import NY_TZ, slice_ny_day
from .trade_management import TradeManagementConfig, build_management_structure_catalog

EPS = 1e-12


@dataclass(frozen=True)
class BroadPositionManagementConfig:
    session_end_hour: int = 16
    session_end_minute: int = 30
    structure_tf: int = 2
    pivot_left: int = 1
    pivot_right: int = 1
    partial_fraction: float = 0.25
    partial_trigger_r: float = 1.0
    protect_trigger_r: float = 1.0
    lock_trigger_r: float = 2.0
    lock_r: float = 0.5


SCENARIOS: tuple[str, ...] = (
    "full_opposite_liquidity",
    "be_after_1r",
    "partial25_1r_be",
    "partial25_1r_be_lock05_after2r",
    "partial25_1r_be_trail2m",
)


def _as_ny_ts(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize(NY_TZ)
    return ts.tz_convert(NY_TZ)


def _side_return(price: float, entry: float, is_long: bool) -> float:
    return float((price / entry - 1.0) if is_long else (entry / price - 1.0))


def _price_at_r(entry: float, risk_abs: float, r_value: float, is_long: bool) -> float:
    return float(entry + risk_abs * r_value if is_long else entry - risk_abs * r_value)


def _profit_factor(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    gains = float(x[x > 0].sum())
    losses = float(-x[x < 0].sum())
    if losses > EPS:
        return gains / losses
    return np.inf if gains > 0 else np.nan


def _max_consecutive_losses(values: Iterable[float]) -> int:
    best = cur = 0
    for value in values:
        if np.isfinite(value) and float(value) < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _latest_known_trail(
    structures: pd.DataFrame,
    *,
    at_time: pd.Timestamp,
    after_time: pd.Timestamp,
    is_long: bool,
    tf: int,
    current_stop: float,
    current_open: float,
) -> float | None:
    if structures.empty:
        return None
    wanted = "low" if is_long else "high"
    g = structures.loc[
        (structures["hierarchy"].astype(str) == "ST")
        & (pd.to_numeric(structures["structure_tf"], errors="coerce") == int(tf))
        & (structures["pivot_side"].astype(str) == wanted)
        & (pd.to_datetime(structures["confirmation_available_time"]) <= at_time)
        & (pd.to_datetime(structures["confirmation_available_time"]) > after_time)
    ].copy()
    if g.empty:
        return None
    g = g.sort_values("confirmation_available_time", kind="mergesort")
    for _, row in g.iloc[::-1].iterrows():
        px = float(row["pivot_price"])
        if is_long and current_stop < px < current_open:
            return px
        if (not is_long) and current_stop > px > current_open:
            return px
    return None


def _delayed_fill(
    day: pd.DataFrame,
    trade: dict[str, object],
    *,
    delay_minutes: int,
) -> tuple[pd.Timestamp, float, str]:
    original = _as_ny_ts(trade["fill_time"])
    wanted = original + pd.Timedelta(minutes=int(delay_minutes))
    starts = pd.DatetimeIndex(day.index)
    pos = int(starts.searchsorted(wanted, side="left"))
    if pos >= len(day):
        return pd.NaT, np.nan, "delay_after_session"
    fill_time = pd.Timestamp(starts[pos])
    if int(delay_minutes) <= 0:
        replay_px = pd.to_numeric(pd.Series([trade.get("entry_price_replay")]), errors="coerce").iloc[0]
        if np.isfinite(replay_px):
            return fill_time, float(replay_px), ""
    return fill_time, float(day["open"].iloc[pos]), ""


def replay_position_scenario(
    day_1m: pd.DataFrame,
    trade: dict[str, object],
    structures: pd.DataFrame,
    *,
    scenario: str,
    round_trip_cost: float,
    delay_minutes: int = 0,
    config: BroadPositionManagementConfig = BroadPositionManagementConfig(),
) -> dict[str, object]:
    """Replay one broad entry under one predeclared management policy."""
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown management scenario: {scenario}")
    result = dict(trade)
    result.update({
        "management_scenario": scenario,
        "entry_delay_minutes": int(delay_minutes),
        "managed": False,
        "invalid_reason": "",
    })
    if not bool(trade.get("filled", False)):
        result["invalid_reason"] = "base_not_filled"
        return result

    fill_time, entry, fill_error = _delayed_fill(day_1m, trade, delay_minutes=delay_minutes)
    if fill_error:
        result["invalid_reason"] = fill_error
        return result
    is_long = str(trade.get("trade_side", "")).upper() == "LONG"
    stop0 = float(trade.get("stop_price", np.nan))
    target = float(trade.get("target_price", np.nan))
    risk_abs = entry - stop0 if is_long else stop0 - entry
    target_ahead = target > entry + EPS if is_long else target < entry - EPS
    if not (np.isfinite(entry) and np.isfinite(stop0) and np.isfinite(target) and risk_abs > EPS):
        result["invalid_reason"] = "invalid_risk_or_price"
        return result
    if not target_ahead:
        result["invalid_reason"] = "opposite_target_not_ahead"
        return result

    # If execution is delayed, do not enter after the original terminal stop was
    # already invalidated.  This is an execution-invalid case, not an alpha filter.
    original_fill = _as_ny_ts(trade["fill_time"])
    if fill_time > original_fill:
        before = day_1m.loc[(day_1m.index >= original_fill) & (day_1m.index < fill_time)]
        if not before.empty:
            invalidated = bool((pd.to_numeric(before["low"], errors="coerce") <= stop0 + EPS).any()) if is_long else bool((pd.to_numeric(before["high"], errors="coerce") >= stop0 - EPS).any())
            if invalidated:
                result["invalid_reason"] = "stop_invalidated_before_delayed_entry"
                return result

    end = pd.Timestamp(
        f"{pd.Timestamp(trade['ny_date']).date()} {config.session_end_hour:02d}:{config.session_end_minute:02d}",
        tz=NY_TZ,
    )
    path = day_1m.loc[(day_1m.index >= fill_time) & (day_1m.index < end)].copy()
    if path.empty:
        result["invalid_reason"] = "no_post_fill_bars"
        return result

    partial_enabled = scenario.startswith("partial25_")
    be_enabled = scenario != "full_opposite_liquidity"
    lock_enabled = "lock05_after2r" in scenario
    trail_enabled = "trail2m" in scenario

    current_stop = stop0
    pending_stop: float | None = None
    partial_done = False
    protect_armed = False
    lock_armed = False
    trail_after = fill_time
    remaining = 1.0
    exits: list[dict[str, object]] = []
    mfe_r = 0.0
    mae_r = 0.0

    partial_price = _price_at_r(entry, risk_abs, config.partial_trigger_r, is_long)
    protect_price = entry  # fees remain charged below; this is deliberately not an optimistic fee-adjusted stop.
    lock_price = _price_at_r(entry, risk_abs, config.lock_r, is_long)
    protect_trigger_price = _price_at_r(entry, risk_abs, config.protect_trigger_r, is_long)
    lock_trigger_price = _price_at_r(entry, risk_abs, config.lock_trigger_r, is_long)

    for bar_start, bar in path.iterrows():
        bar_start = pd.Timestamp(bar_start)
        op = float(bar["open"]); hi = float(bar["high"]); lo = float(bar["low"])

        if pending_stop is not None:
            if (is_long and pending_stop > current_stop) or ((not is_long) and pending_stop < current_stop):
                current_stop = pending_stop
            pending_stop = None

        if trail_enabled and partial_done and remaining > EPS:
            trail = _latest_known_trail(
                structures,
                at_time=bar_start,
                after_time=trail_after,
                is_long=is_long,
                tf=int(config.structure_tf),
                current_stop=current_stop,
                current_open=op,
            )
            if trail is not None:
                current_stop = float(trail)

        stop_touch = lo <= current_stop + EPS if is_long else hi >= current_stop - EPS
        if stop_touch:
            # Stop-market gap handling: if the bar opens through the stop, use
            # the worse opening price rather than an impossible stop-price fill.
            stop_fill = min(current_stop, op) if is_long else max(current_stop, op)
            exits.append({"fraction": remaining, "price": stop_fill, "time": bar_start + pd.Timedelta(minutes=1), "reason": "initial_stop" if abs(current_stop - stop0) <= EPS else "protective_stop"})
            remaining = 0.0
            mae_r = min(mae_r, (lo - entry) / risk_abs if is_long else (entry - hi) / risk_abs)
            break

        favourable_r = (hi - entry) / risk_abs if is_long else (entry - lo) / risk_abs
        adverse_r = (lo - entry) / risk_abs if is_long else (entry - hi) / risk_abs
        mfe_r = max(mfe_r, float(favourable_r))
        mae_r = min(mae_r, float(adverse_r))

        # Resting partial order.  Stop already won same-minute ambiguity above.
        partial_touch = hi >= partial_price - EPS if is_long else lo <= partial_price + EPS
        if partial_enabled and (not partial_done) and partial_touch:
            frac = min(float(config.partial_fraction), remaining)
            exits.append({"fraction": frac, "price": partial_price, "time": bar_start + pd.Timedelta(minutes=1), "reason": "partial_1r"})
            remaining -= frac
            partial_done = True
            trail_after = bar_start + pd.Timedelta(minutes=1)
            # Protection activates next bar, never retroactively inside this OHLC bar.
            pending_stop = protect_price

        target_touch = hi >= target - EPS if is_long else lo <= target + EPS
        if target_touch and remaining > EPS:
            exits.append({"fraction": remaining, "price": target, "time": bar_start + pd.Timedelta(minutes=1), "reason": "opposite_liquidity_tp"})
            remaining = 0.0
            break

        if be_enabled and (not protect_armed):
            trigger = hi >= protect_trigger_price - EPS if is_long else lo <= protect_trigger_price + EPS
            if trigger:
                protect_armed = True
                if pending_stop is None or (is_long and protect_price > pending_stop) or ((not is_long) and protect_price < pending_stop):
                    pending_stop = protect_price

        if lock_enabled and (not lock_armed):
            trigger = hi >= lock_trigger_price - EPS if is_long else lo <= lock_trigger_price + EPS
            if trigger:
                lock_armed = True
                if pending_stop is None or (is_long and lock_price > pending_stop) or ((not is_long) and lock_price < pending_stop):
                    pending_stop = lock_price

    if remaining > EPS:
        last_time = pd.Timestamp(path.index[-1]) + pd.Timedelta(minutes=1)
        last_close = float(path["close"].iloc[-1])
        exits.append({"fraction": remaining, "price": last_close, "time": last_time, "reason": "session_close"})
        remaining = 0.0

    weighted_gross = float(sum(float(x["fraction"]) * _side_return(float(x["price"]), entry, is_long) for x in exits))
    net = weighted_gross - float(round_trip_cost)
    risk_pct = risk_abs / entry
    final = exits[-1]
    result.update({
        "managed": True,
        "managed_fill_time": fill_time,
        "managed_entry_price": entry,
        "managed_initial_stop": stop0,
        "managed_target_price": target,
        "management_exit_time": final["time"],
        "management_exit_price": float(final["price"]),
        "management_exit_reason": str(final["reason"]),
        "management_exit_count": int(len(exits)),
        "management_exit_ledger": "|".join(f"{float(x['fraction']):.4f}@{float(x['price']):.6f}:{x['reason']}" for x in exits),
        "management_gross_return": weighted_gross,
        "management_net_return": net,
        "management_net_r": float(net / risk_pct) if risk_pct > EPS else np.nan,
        "management_mfe_r": float(mfe_r),
        "management_mae_r": float(mae_r),
        "management_hold_minutes": float((pd.Timestamp(final["time"]) - fill_time).total_seconds() / 60.0),
        "partial_taken": bool(partial_done),
        "protect_armed": bool(protect_armed),
        "lock_armed": bool(lock_armed),
        "final_stop_price": float(current_stop),
    })
    return result


def replay_position_scenarios(
    bars_ny: pd.DataFrame,
    lifecycle: pd.DataFrame,
    *,
    round_trip_cost: float,
    cost_multipliers: Sequence[float] = (1.0, 1.5, 2.0),
    delays: Sequence[int] = (0, 1, 2),
    scenarios: Sequence[str] = SCENARIOS,
    config: BroadPositionManagementConfig = BroadPositionManagementConfig(),
    progress=None,
) -> pd.DataFrame:
    """Replay all predeclared management policies without dropping valid base trades."""
    if lifecycle.empty:
        return pd.DataFrame()
    base = lifecycle.loc[lifecycle["filled"].fillna(False).astype(bool)].copy()
    if base.empty:
        return pd.DataFrame()

    day_cache: dict[str, pd.DataFrame] = {}
    struct_cache: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, object]] = []
    done = 0
    structure_cfg = TradeManagementConfig(
        structure_timeframes=(int(config.structure_tf),),
        runner_timeframes=(int(config.structure_tf),),
        pivot_left=int(config.pivot_left),
        pivot_right=int(config.pivot_right),
    )
    for trade in base.to_dict("records"):
        day_key = str(pd.Timestamp(trade["ny_date"]).date())
        if day_key not in day_cache:
            day = slice_ny_day(bars_ny, pd.Timestamp(day_key).date(), pd.Timestamp("08:30").time(), pd.Timestamp("16:30").time())
            day_cache[day_key] = day
            struct_cache[day_key] = build_management_structure_catalog(day, config=structure_cfg)
        day = day_cache[day_key]; structures = struct_cache[day_key]
        for delay in delays:
            for scenario in scenarios:
                # Price path is independent of transaction-cost stress.  Replay it
                # once at 1x cost, then clone gross outcome for higher costs.
                base_row = replay_position_scenario(
                    day,
                    trade,
                    structures,
                    scenario=str(scenario),
                    round_trip_cost=float(round_trip_cost),
                    delay_minutes=int(delay),
                    config=config,
                )
                for cm in cost_multipliers:
                    row = dict(base_row)
                    row["cost_multiple"] = float(cm)
                    if bool(row.get("managed", False)):
                        gross = float(row["management_gross_return"])
                        net = gross - float(round_trip_cost) * float(cm)
                        entry = float(row["managed_entry_price"]); stop = float(row["managed_initial_stop"])
                        risk_pct = abs(entry - stop) / entry
                        row["management_net_return"] = net
                        row["management_net_r"] = net / risk_pct if risk_pct > EPS else np.nan
                    rows.append(row)
                done += 1
                if progress is not None:
                    progress.update(done)
    return pd.DataFrame(rows)


def _longest_no_trade_sessions(valid_sessions: Sequence[object], active_days: set[str]) -> int:
    best = cur = 0
    for value in valid_sessions:
        day = str(pd.Timestamp(value).date())
        if day in active_days:
            cur = 0
        else:
            cur += 1
            best = max(best, cur)
    return int(best)


def account_metrics(
    trades: pd.DataFrame,
    *,
    valid_sessions: Sequence[object],
    risk_fraction: float,
    max_notional_multiple: float,
    initial_capital: float,
) -> dict[str, object]:
    q = trades.loc[trades["managed"].fillna(False).astype(bool)].copy() if not trades.empty else pd.DataFrame()
    if q.empty:
        return {
            "trades": 0,
            "trades_per_session": 0.0,
            "active_day_rate": 0.0,
            "longest_no_trade_sessions": len(valid_sessions),
        }
    q["managed_fill_time"] = pd.to_datetime(q["managed_fill_time"], errors="coerce", utc=True)
    q = q.sort_values("managed_fill_time", kind="mergesort").reset_index(drop=True)
    entry = pd.to_numeric(q["managed_entry_price"], errors="coerce")
    stop = pd.to_numeric(q["managed_initial_stop"], errors="coerce")
    risk_pct = (entry - stop).abs() / entry
    notional = (float(risk_fraction) / risk_pct).clip(upper=float(max_notional_multiple))
    net = pd.to_numeric(q["management_net_return"], errors="coerce").fillna(0.0)
    account_ret = (net * notional).clip(lower=-0.999)
    equity = float(initial_capital) * (1.0 + account_ret).cumprod()
    running_peak = equity.cummax()
    drawdown = equity / running_peak - 1.0
    months = q["managed_fill_time"].dt.tz_convert(NY_TZ).dt.strftime("%Y-%m")
    monthly = pd.DataFrame({"month": months, "ret": account_ret}).groupby("month")["ret"].apply(lambda x: float(np.prod(1.0 + x.to_numpy(float)) - 1.0))
    active = set(q["ny_date"].astype(str))
    valid_n = max(1, len(valid_sessions))
    first = q["managed_fill_time"].iloc[0]
    last = q["managed_fill_time"].iloc[-1]
    years = max((last - first).total_seconds() / (365.2425 * 24 * 3600), 1.0 / 365.2425)
    final_capital = float(equity.iloc[-1])
    cagr = float((final_capital / float(initial_capital)) ** (1.0 / years) - 1.0) if final_capital > 0 else -1.0
    return {
        "trades": int(len(q)),
        "trades_per_session": float(len(q) / valid_n),
        "active_days": int(len(active)),
        "active_day_rate": float(len(active) / valid_n),
        "longest_no_trade_sessions": _longest_no_trade_sessions(valid_sessions, active),
        "win_rate": float((net > 0).mean()),
        "mean_net_return": float(net.mean()),
        "mean_net_r": float(pd.to_numeric(q["management_net_r"], errors="coerce").mean()),
        "profit_factor": _profit_factor(net),
        "max_consecutive_losses": _max_consecutive_losses(net),
        "positive_month_rate": float((monthly > 0).mean()) if len(monthly) else np.nan,
        "median_hold_minutes": float(pd.to_numeric(q["management_hold_minutes"], errors="coerce").median()),
        "final_capital": final_capital,
        "total_return": float(final_capital / float(initial_capital) - 1.0),
        "cagr": cagr,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "median_notional_multiple": float(notional.median()),
    }


def summarize_management(
    managed: pd.DataFrame,
    *,
    valid_sessions: Sequence[object],
    risk_fraction: float,
    max_notional_multiple: float,
    initial_capital: float,
) -> pd.DataFrame:
    if managed.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, g in managed.groupby(["management_scenario", "cost_multiple", "entry_delay_minutes"], sort=True, dropna=False):
        metrics = account_metrics(
            g,
            valid_sessions=valid_sessions,
            risk_fraction=risk_fraction,
            max_notional_multiple=max_notional_multiple,
            initial_capital=initial_capital,
        )
        invalid = int((~g["managed"].fillna(False).astype(bool)).sum())
        rows.append({
            "management_scenario": keys[0],
            "cost_multiple": float(keys[1]),
            "entry_delay_minutes": int(keys[2]),
            "base_rows": int(len(g)),
            "execution_invalid_rows": invalid,
            **metrics,
        })
    return pd.DataFrame(rows)


def period_label(values: pd.Series) -> pd.Series:
    d = pd.to_datetime(values, errors="coerce")
    return pd.Series(
        np.select(
            [d.dt.year <= 2024, d.dt.year == 2025, d.dt.year >= 2026],
            ["discovery_2023H2_2024", "validation_2025", "forward_2026"],
            default="unknown",
        ),
        index=values.index,
        dtype="object",
    )


def summarize_periods(
    managed: pd.DataFrame,
    *,
    valid_sessions: Sequence[object],
    risk_fraction: float,
    max_notional_multiple: float,
    initial_capital: float,
) -> pd.DataFrame:
    if managed.empty:
        return pd.DataFrame()
    q = managed.copy()
    q["period"] = period_label(q["ny_date"])
    valid = pd.Series([str(pd.Timestamp(x).date()) for x in valid_sessions], dtype="object")
    vperiod = period_label(valid)
    rows: list[dict[str, object]] = []
    for keys, g in q.groupby(["management_scenario", "cost_multiple", "entry_delay_minutes", "period"], sort=True, dropna=False):
        p = str(keys[3])
        p_sessions = valid.loc[vperiod.eq(p)].tolist()
        metrics = account_metrics(
            g,
            valid_sessions=p_sessions,
            risk_fraction=risk_fraction,
            max_notional_multiple=max_notional_multiple,
            initial_capital=initial_capital,
        )
        rows.append({
            "management_scenario": keys[0],
            "cost_multiple": float(keys[1]),
            "entry_delay_minutes": int(keys[2]),
            "period": p,
            **metrics,
        })
    return pd.DataFrame(rows)


def select_discovery_policy(
    period_summary: pd.DataFrame,
    *,
    minimum_trades_per_session: float = 0.5,
) -> dict[str, object]:
    """Freeze one policy from Discovery only, using the user's stability priorities."""
    if period_summary.empty:
        return {"selected_policy": "", "selection_status": "NO_DATA"}
    q = period_summary.loc[
        period_summary["period"].astype(str).eq("discovery_2023H2_2024")
        & pd.to_numeric(period_summary["cost_multiple"], errors="coerce").eq(1.0)
        & pd.to_numeric(period_summary["entry_delay_minutes"], errors="coerce").eq(0)
    ].copy()
    if q.empty:
        return {"selected_policy": "", "selection_status": "NO_DISCOVERY_ROWS"}
    q = q.loc[pd.to_numeric(q["trades_per_session"], errors="coerce") >= float(minimum_trades_per_session)].copy()
    if q.empty:
        return {"selected_policy": "", "selection_status": "FREQUENCY_GATE_FAIL"}
    positive = q.loc[
        (pd.to_numeric(q["mean_net_return"], errors="coerce") > 0)
        & (pd.to_numeric(q["profit_factor"], errors="coerce") > 1.0)
    ].copy()
    if positive.empty:
        return {"selected_policy": "", "selection_status": "NO_POSITIVE_DISCOVERY_POLICY"}
    # Lexicographic stability ordering: flat time -> loss streak -> MDD -> CAGR -> return.
    positive["_mdd_abs"] = pd.to_numeric(positive["max_drawdown"], errors="coerce").abs()
    positive = positive.sort_values(
        ["longest_no_trade_sessions", "max_consecutive_losses", "_mdd_abs", "cagr", "total_return"],
        ascending=[True, True, True, False, False],
        kind="mergesort",
    )
    best = positive.iloc[0]
    return {
        "selected_policy": str(best["management_scenario"]),
        "selection_status": "SELECTED_FROM_DISCOVERY_ONLY",
        "discovery_trades_per_session": float(best["trades_per_session"]),
        "discovery_profit_factor": float(best["profit_factor"]),
        "discovery_mean_net_return": float(best["mean_net_return"]),
        "discovery_max_consecutive_losses": int(best["max_consecutive_losses"]),
        "discovery_max_drawdown": float(best["max_drawdown"]),
        "discovery_cagr": float(best["cagr"]),
    }
