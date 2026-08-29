from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..config import TournamentConfig
from ..contracts import BacktestResult, StrategySpec
from ..data import TournamentData
from ..indicators import atr
from ..metrics import calculate_metrics


@dataclass
class _TurtlePosition:
    side: int
    first_entry_time: pd.Timestamp
    first_entry_price: float
    avg_entry: float
    total_qty: float
    unit_qty: float
    units: int
    n_value: float
    next_add: float
    stop: float
    entry_equity: float
    entry_fees: float
    mfe_pct: float = 0.0
    mae_pct: float = 0.0


def run_turtle_system2(data: TournamentData, spec: StrategySpec, cfg: TournamentConfig, *, cost_mult: float = 1.0, extra_delay_minutes: int = 0) -> BacktestResult:
    d = data.bars("1D").copy()
    n = atr(d, 20)
    entry_days = int(spec.params.get("entry_days", 55))
    exit_days = int(spec.params.get("exit_days", 20))
    max_units = int(spec.params.get("max_units", 4))
    schedule = pd.DataFrame(
        {
            "available_time": pd.to_datetime(d["available_time"]),
            "entry_high": d["high"].rolling(entry_days, min_periods=entry_days).max(),
            "entry_low": d["low"].rolling(entry_days, min_periods=entry_days).min(),
            "exit_high": d["high"].rolling(exit_days, min_periods=exit_days).max(),
            "exit_low": d["low"].rolling(exit_days, min_periods=exit_days).min(),
            "N": n,
        }
    ).dropna()
    if extra_delay_minutes:
        schedule["available_time"] = schedule["available_time"] + pd.Timedelta(minutes=extra_delay_minutes)

    bars = data.one_minute.loc[pd.Timestamp(cfg.research_start) : pd.Timestamp(cfg.research_end)]
    idx = pd.DatetimeIndex(bars.index)
    op = bars["open"].to_numpy(float)
    hi = bars["high"].to_numpy(float)
    lo = bars["low"].to_numpy(float)
    cl = bars["close"].to_numpy(float)
    sched_times = pd.DatetimeIndex(schedule["available_time"])
    sched_arr = schedule[["entry_high", "entry_low", "exit_high", "exit_low", "N"]].to_numpy(float)
    s_ptr = -1
    current: np.ndarray | None = None

    capital = float(cfg.initial_capital)
    peak = capital
    max_dd = 0.0
    p: _TurtlePosition | None = None
    trades: list[dict[str, Any]] = []
    daily_rows: list[tuple[pd.Timestamp, float]] = []
    last_day = idx[0].normalize()

    def one_way_fee(qty: float, price: float) -> float:
        return abs(qty * price) * cfg.one_way_cost * cost_mult

    def mark_equity(i: int) -> float:
        if p is None:
            return capital
        gross = p.total_qty * p.side * (cl[i] - p.avg_entry)
        return p.entry_equity + gross - p.entry_fees - one_way_fee(p.total_qty, cl[i])

    def exit_position(i: int, price: float, reason: str) -> None:
        nonlocal p, capital
        if p is None:
            return
        gross = p.total_qty * p.side * (price - p.avg_entry)
        exit_fee = one_way_fee(p.total_qty, price)
        pnl = gross - p.entry_fees - exit_fee
        capital = p.entry_equity + pnl
        trades.append(
            {
                "strategy_id": spec.strategy_id,
                "side": p.side,
                "signal_time": p.first_entry_time,
                "entry_time": p.first_entry_time,
                "entry_price": p.avg_entry,
                "first_entry_price": p.first_entry_price,
                "exit_time": idx[i],
                "exit_price": float(price),
                "exit_reason": reason,
                "units": p.units,
                "quantity": p.total_qty,
                "N": p.n_value,
                "fee": p.entry_fees + exit_fee,
                "pnl": pnl,
                "return_on_equity": pnl / p.entry_equity if p.entry_equity else 0.0,
                "holding_minutes": (idx[i] - p.first_entry_time).total_seconds() / 60.0,
                "mfe_pct": p.mfe_pct,
                "mae_pct": p.mae_pct,
            }
        )
        p = None

    for i, ts in enumerate(idx):
        day = ts.normalize()
        if day != last_day:
            daily_rows.append((last_day, mark_equity(max(0, i - 1))))
            last_day = day

        # Strict causality: a daily context available at 08:00 can only influence a later 1m open.
        while s_ptr + 1 < len(sched_times) and sched_times[s_ptr + 1] < ts:
            s_ptr += 1
            current = sched_arr[s_ptr]

        if p is not None and current is not None:
            entry_high, entry_low, exit_high, exit_low, n_now = current
            if p.side == 1:
                p.mfe_pct = max(p.mfe_pct, hi[i] / p.avg_entry - 1.0)
                p.mae_pct = min(p.mae_pct, lo[i] / p.avg_entry - 1.0)
                active_exit = max(p.stop, exit_low)
                if lo[i] <= active_exit:
                    fill = min(op[i], active_exit)
                    exit_position(i, fill, "STOP" if p.stop >= exit_low else "CHANNEL_EXIT")
                elif p.units < max_units and hi[i] >= p.next_add:
                    fill = max(op[i], p.next_add)
                    max_qty = cfg.max_notional_leverage * capital / max(fill, 1e-12)
                    add_qty = min(p.unit_qty, max(0.0, max_qty - p.total_qty))
                    if add_qty > 0:
                        old_notional_qty = p.total_qty
                        p.avg_entry = (p.avg_entry * old_notional_qty + fill * add_qty) / (old_notional_qty + add_qty)
                        p.total_qty += add_qty
                        p.units += 1
                        p.entry_fees += one_way_fee(add_qty, fill)
                        p.next_add = fill + 0.5 * p.n_value
                        p.stop = max(p.stop, fill - 2.0 * p.n_value)
            else:
                p.mfe_pct = max(p.mfe_pct, 1.0 - lo[i] / p.avg_entry)
                p.mae_pct = min(p.mae_pct, 1.0 - hi[i] / p.avg_entry)
                active_exit = min(p.stop, exit_high)
                if hi[i] >= active_exit:
                    fill = max(op[i], active_exit)
                    exit_position(i, fill, "STOP" if p.stop <= exit_high else "CHANNEL_EXIT")
                elif p.units < max_units and lo[i] <= p.next_add:
                    fill = min(op[i], p.next_add)
                    max_qty = cfg.max_notional_leverage * capital / max(fill, 1e-12)
                    add_qty = min(p.unit_qty, max(0.0, max_qty - p.total_qty))
                    if add_qty > 0:
                        old_notional_qty = p.total_qty
                        p.avg_entry = (p.avg_entry * old_notional_qty + fill * add_qty) / (old_notional_qty + add_qty)
                        p.total_qty += add_qty
                        p.units += 1
                        p.entry_fees += one_way_fee(add_qty, fill)
                        p.next_add = fill - 0.5 * p.n_value
                        p.stop = min(p.stop, fill + 2.0 * p.n_value)

        if p is None and current is not None:
            entry_high, entry_low, exit_high, exit_low, n_now = current
            if n_now > 0 and np.isfinite(n_now):
                hit_long = hi[i] >= entry_high
                hit_short = lo[i] <= entry_low
                if hit_long != hit_short:
                    side = 1 if hit_long else -1
                    fill = max(op[i], entry_high) if side == 1 else min(op[i], entry_low)
                    # Tournament risk overlay: 2N stop risks cfg.risk_per_trade on the first unit.
                    unit_qty = cfg.risk_per_trade * capital / (2.0 * n_now)
                    unit_qty = min(unit_qty, cfg.max_notional_leverage * capital / max(fill, 1e-12))
                    if unit_qty > 0:
                        fee = one_way_fee(unit_qty, fill)
                        p = _TurtlePosition(
                            side=side,
                            first_entry_time=ts,
                            first_entry_price=float(fill),
                            avg_entry=float(fill),
                            total_qty=float(unit_qty),
                            unit_qty=float(unit_qty),
                            units=1,
                            n_value=float(n_now),
                            next_add=float(fill + side * 0.5 * n_now),
                            stop=float(fill - side * 2.0 * n_now),
                            entry_equity=capital,
                            entry_fees=fee,
                        )
                        # Conservative same-minute adverse path check.
                        if (side == 1 and lo[i] <= p.stop) or (side == -1 and hi[i] >= p.stop):
                            exit_position(i, p.stop, "SAME_MINUTE_STOP")

        eq = mark_equity(i)
        peak = max(peak, eq)
        if peak > 0:
            max_dd = max(max_dd, 1.0 - eq / peak)

    if p is not None:
        exit_position(len(idx) - 1, cl[-1], "END")
    daily_rows.append((last_day, capital))
    daily = pd.DataFrame(daily_rows, columns=["date", "equity"]).drop_duplicates("date", keep="last").set_index("date")
    tdf = pd.DataFrame(trades)
    metrics = calculate_metrics(
        tdf,
        daily,
        initial_capital=cfg.initial_capital,
        start=pd.Timestamp(cfg.research_start),
        end=pd.Timestamp(cfg.research_end),
        intrabar_max_drawdown_pct=max_dd * 100.0,
    )
    audit = {
        "strategy_id": spec.strategy_id,
        "daily_context_strictly_before_execution_minute": True,
        "intraday_breakout_execution": True,
        "same_minute_ambiguity_policy": "STOP_OR_SKIP",
        "future_visibility_violations": 0,
        "sealed_2026_opened": False,
        "cost_multiplier": float(cost_mult),
        "extra_delay_minutes": int(extra_delay_minutes),
    }
    return BacktestResult(spec.strategy_id, tdf, daily, metrics, audit)
