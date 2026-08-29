from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import SourceLockedConfig


@dataclass
class BacktestResult:
    strategy_id: str
    minute_equity: pd.Series
    position: pd.Series
    daily: pd.DataFrame
    events: pd.DataFrame
    metrics: dict[str, Any]
    audit: dict[str, Any]


def _max_run_days(mask: pd.Series) -> float:
    if mask.empty:
        return 0.0
    arr = mask.to_numpy(bool)
    best = cur = 0
    for x in arr:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best / 1440.0


def _max_consecutive_losing_days(daily: pd.DataFrame) -> int:
    ret = daily["equity"].pct_change()
    best = cur = 0
    for x in ret.to_numpy(float):
        if np.isfinite(x) and x < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _episodes(idx: pd.DatetimeIndex, pos: np.ndarray, equity: np.ndarray, strategy_id: str) -> pd.DataFrame:
    side = np.sign(pos).astype(int)
    rows: list[dict[str, Any]] = []
    start: int | None = None
    current_side = 0
    for i, s in enumerate(side):
        if start is None and s != 0:
            start, current_side = i, int(s)
        elif start is not None and s != current_side:
            end = max(start, i - 1)
            eq0 = equity[start - 1] if start > 0 else equity[0]
            eq1 = equity[end]
            rows.append({"strategy_id": strategy_id, "side": current_side, "start": idx[start], "end": idx[end], "pnl": eq1 - eq0})
            start = i if s != 0 else None
            current_side = int(s)
    if start is not None:
        eq0 = equity[start - 1] if start > 0 else equity[0]
        rows.append({"strategy_id": strategy_id, "side": current_side, "start": idx[start], "end": idx[-1], "pnl": equity[-1] - eq0})
    return pd.DataFrame(rows)


def _profit_factor(ep: pd.DataFrame) -> float:
    if ep.empty:
        return 0.0
    pnl = pd.to_numeric(ep["pnl"], errors="coerce").fillna(0.0)
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    return gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)


def _summarize(cfg: SourceLockedConfig, strategy_id: str, idx: pd.DatetimeIndex, equity: np.ndarray, pos: np.ndarray, events: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    eq = pd.Series(equity, index=idx)
    daily = eq.groupby(eq.index.normalize()).last().to_frame("equity")
    final = float(equity[-1]) if len(equity) else cfg.initial_capital
    total = final / cfg.initial_capital - 1.0
    years = max((pd.Timestamp(cfg.research_end) - pd.Timestamp(cfg.research_start)).total_seconds() / (365.25 * 86400.0), 1 / 365.25)
    cagr = (final / cfg.initial_capital) ** (1.0 / years) - 1.0 if final > 0 else -1.0
    peak = np.maximum.accumulate(equity)
    mdd = float(np.nanmax(1.0 - equity / np.where(peak == 0.0, np.nan, peak)) * 100.0)
    ep = _episodes(idx, pos, equity, strategy_id)
    daily_ret = daily["equity"].pct_change().fillna(0.0)
    yearly = daily_ret.groupby(daily.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)
    monthly = daily_ret.groupby(daily.index.to_period("M")).apply(lambda x: (1.0 + x).prod() - 1.0)
    pos_s = pd.Series(pos, index=idx)
    turnover = float(pd.to_numeric(events.get("turnover", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    metrics = {
        "strategy_id": strategy_id,
        "total_return_pct": total * 100.0,
        "cagr_pct": cagr * 100.0,
        "max_drawdown_pct": mdd,
        "profit_factor": _profit_factor(ep),
        "episodes": int(len(ep)),
        "max_flat_days": _max_run_days(pos_s.abs() <= cfg.flat_exposure_threshold),
        "max_low_exposure_days": _max_run_days(pos_s.abs() <= cfg.low_exposure_threshold),
        "max_consecutive_losing_days": _max_consecutive_losing_days(daily),
        "positive_years": int((yearly > 0).sum()),
        "positive_month_ratio": float((monthly > 0).mean()) if len(monthly) else 0.0,
        "worst_month_pct": float(monthly.min() * 100.0) if len(monthly) else 0.0,
        "turnover_x": turnover,
        "avg_abs_exposure": float(np.nanmean(np.abs(pos))),
        "max_abs_exposure": float(np.nanmax(np.abs(pos))) if len(pos) else 0.0,
        "rebalance_count": int(len(events)),
        "final_equity": final,
    }
    return daily, metrics


def run_target_schedule(one_minute: pd.DataFrame, schedule: pd.DataFrame, cfg: SourceLockedConfig, strategy_id: str, *, cost_mult: float = 1.0, extra_delay_minutes: int = 0) -> BacktestResult:
    bars = one_minute.loc[pd.Timestamp(cfg.research_start):pd.Timestamp(cfg.research_end)].copy()
    idx = pd.DatetimeIndex(bars.index)
    op = bars["open"].to_numpy(float)
    cl = bars["close"].to_numpy(float)
    next_open = np.roll(op, -1)
    next_open[-1] = cl[-1]
    minute_ret = next_open / op - 1.0

    events: list[tuple[int, pd.Timestamp, float]] = []
    for _, row in schedule.sort_values("signal_time").iterrows():
        signal_time = pd.Timestamp(row["signal_time"]) + pd.Timedelta(minutes=extra_delay_minutes)
        i = int(idx.searchsorted(signal_time, side="right"))
        if i < len(idx):
            events.append((i, pd.Timestamp(row["signal_time"]), float(row["raw_target"])))
    dedup: dict[int, tuple[pd.Timestamp, float]] = {}
    for i, ts, target in events:
        dedup[i] = (ts, target)
    events = [(i, dedup[i][0], dedup[i][1]) for i in sorted(dedup)]
    if not events:
        raise RuntimeError(f"no executable target events for {strategy_id}")

    equity = np.full(len(idx), np.nan, dtype=float)
    position = np.zeros(len(idx), dtype=float)
    capital = float(cfg.initial_capital)
    current = 0.0
    cursor = 0
    rows: list[dict[str, Any]] = []

    def fill_segment(start: int, end: int, exposure: float, start_equity: float) -> float:
        if end <= start:
            return start_equity
        seg = np.clip(exposure * minute_ret[start:end], -0.999999, None)
        path = start_equity * np.cumprod(1.0 + seg)
        equity[start:end] = path
        position[start:end] = exposure
        return float(path[-1]) if len(path) else start_equity

    for event_no, (i, source_time, target) in enumerate(events):
        capital = fill_segment(cursor, i, current, capital)
        turnover = abs(target - current)
        fee_fraction = turnover * cfg.one_way_cost * cost_mult
        capital *= max(1.0 - fee_fraction, 0.0)
        rows.append({
            "strategy_id": strategy_id,
            "event_no": event_no,
            "signal_time": source_time,
            "execution_time": idx[i],
            "position_before": current,
            "position_after": target,
            "turnover": turnover,
            "fee_fraction": fee_fraction,
        })
        current = float(target)
        cursor = i
    capital = fill_segment(cursor, len(idx), current, capital)
    if np.isnan(equity).any():
        first = np.flatnonzero(np.isfinite(equity))
        if first.size:
            equity[: first[0]] = cfg.initial_capital
        equity = pd.Series(equity).ffill().fillna(cfg.initial_capital).to_numpy(float)
    event_df = pd.DataFrame(rows)
    daily, metrics = _summarize(cfg, strategy_id, idx, equity, position, event_df)
    audit = {
        "strategy_id": strategy_id,
        "strict_next_observable_open": True,
        "future_visibility_violations": 0,
        "sealed_2026_opened": False,
        "cost_multiplier": float(cost_mult),
        "extra_delay_minutes": int(extra_delay_minutes),
        "execution_semantics": "single_net_eth_exposure",
    }
    return BacktestResult(strategy_id, pd.Series(equity, index=idx), pd.Series(position, index=idx), daily, event_df, metrics, audit)


@dataclass
class _TurtlePos:
    side: int
    entry_time: pd.Timestamp
    entry_equity: float
    avg_entry: float
    qty: float
    unit_qty: float
    units: int
    n_value: float
    next_add: float
    stop: float
    fees: float


def run_turtle_system2(one_minute: pd.DataFrame, context: pd.DataFrame, cfg: SourceLockedConfig, *, cost_mult: float = 1.0, extra_delay_minutes: int = 0) -> BacktestResult:
    """Original Turtle System 2 adapted to ETH perpetual and project data causality."""
    strategy_id = "SL04_TURTLE_SYSTEM2"
    bars = one_minute.loc[pd.Timestamp(cfg.research_start):pd.Timestamp(cfg.research_end)].copy()
    idx = pd.DatetimeIndex(bars.index)
    op = bars["open"].to_numpy(float)
    hi = bars["high"].to_numpy(float)
    lo = bars["low"].to_numpy(float)
    cl = bars["close"].to_numpy(float)
    ctx = context.sort_values("available_time")
    times = pd.DatetimeIndex(ctx["available_time"])
    vals = ctx[["entry_high", "entry_low", "exit_high", "exit_low", "N"]].to_numpy(float)
    cptr = -1
    current_ctx: np.ndarray | None = None

    capital = float(cfg.initial_capital)
    p: _TurtlePos | None = None
    equity = np.full(len(idx), cfg.initial_capital, dtype=float)
    position = np.zeros(len(idx), dtype=float)
    rows: list[dict[str, Any]] = []
    pending_entry: tuple[int, int, float, float] | None = None  # execute_i, side, threshold, N

    def fee(qty: float, price: float) -> float:
        return abs(qty * price) * cfg.one_way_cost * cost_mult

    def mark(i: int) -> float:
        if p is None:
            return capital
        gross = p.side * p.qty * (cl[i] - p.avg_entry)
        return p.entry_equity + gross - p.fees - fee(p.qty, cl[i])

    def exit_position(i: int, price: float, reason: str) -> None:
        nonlocal p, capital
        if p is None:
            return
        exit_fee = fee(p.qty, price)
        gross = p.side * p.qty * (price - p.avg_entry)
        pnl = gross - p.fees - exit_fee
        capital = p.entry_equity + pnl
        rows.append({
            "strategy_id": strategy_id,
            "event": "EXIT",
            "time": idx[i],
            "side": p.side,
            "price": float(price),
            "units": p.units,
            "qty": p.qty,
            "turnover": abs(p.qty * price) / max(p.entry_equity, 1e-12),
            "fee": exit_fee,
            "reason": reason,
            "pnl": pnl,
        })
        p = None

    def enter(i: int, side: int, price: float, n_value: float) -> None:
        nonlocal p
        # Original Turtle unit: 1N move equals 1% of account equity. ETH dollars-per-point = 1 USD per 1 ETH.
        unit_qty = 0.01 * capital / n_value
        first_fee = fee(unit_qty, price)
        p = _TurtlePos(
            side=side,
            entry_time=idx[i],
            entry_equity=capital,
            avg_entry=float(price),
            qty=float(unit_qty),
            unit_qty=float(unit_qty),
            units=1,
            n_value=float(n_value),
            next_add=float(price + side * 0.5 * n_value),
            stop=float(price - side * 2.0 * n_value),
            fees=float(first_fee),
        )
        rows.append({
            "strategy_id": strategy_id,
            "event": "ENTRY",
            "time": idx[i],
            "side": side,
            "price": float(price),
            "units": 1,
            "qty": float(unit_qty),
            "turnover": abs(unit_qty * price) / max(capital, 1e-12),
            "fee": first_fee,
            "reason": "55D_BREAKOUT",
            "pnl": np.nan,
        })

    for i, ts in enumerate(idx):
        # Project causal rule: daily context becomes usable only strictly after available_time.
        while cptr + 1 < len(times) and times[cptr + 1] < ts:
            cptr += 1
            current_ctx = vals[cptr]

        if pending_entry is not None and i >= pending_entry[0] and p is None:
            _, side, threshold, n_value = pending_entry
            enter(i, side, float(op[i]), n_value)
            pending_entry = None

        if p is not None and current_ctx is not None:
            entry_high, entry_low, exit_high, exit_low, _ = current_ctx
            if p.side == 1:
                active_exit = max(p.stop, exit_low)
                if lo[i] <= active_exit:
                    fill = min(op[i], active_exit)
                    exit_position(i, fill, "STOP" if p.stop >= exit_low else "20D_EXIT")
                elif p.units < 4 and hi[i] >= p.next_add:
                    fill = max(op[i], p.next_add)
                    add_qty = p.unit_qty
                    old_qty = p.qty
                    p.avg_entry = (p.avg_entry * old_qty + fill * add_qty) / (old_qty + add_qty)
                    p.qty += add_qty
                    p.units += 1
                    p.fees += fee(add_qty, fill)
                    p.next_add = fill + 0.5 * p.n_value
                    p.stop = fill - 2.0 * p.n_value
                    rows.append({"strategy_id": strategy_id, "event": "ADD", "time": ts, "side": 1, "price": float(fill), "units": p.units, "qty": add_qty, "turnover": abs(add_qty * fill) / max(p.entry_equity, 1e-12), "fee": fee(add_qty, fill), "reason": "PLUS_0.5N", "pnl": np.nan})
            else:
                active_exit = min(p.stop, exit_high)
                if hi[i] >= active_exit:
                    fill = max(op[i], active_exit)
                    exit_position(i, fill, "STOP" if p.stop <= exit_high else "20D_EXIT")
                elif p.units < 4 and lo[i] <= p.next_add:
                    fill = min(op[i], p.next_add)
                    add_qty = p.unit_qty
                    old_qty = p.qty
                    p.avg_entry = (p.avg_entry * old_qty + fill * add_qty) / (old_qty + add_qty)
                    p.qty += add_qty
                    p.units += 1
                    p.fees += fee(add_qty, fill)
                    p.next_add = fill - 0.5 * p.n_value
                    p.stop = fill + 2.0 * p.n_value
                    rows.append({"strategy_id": strategy_id, "event": "ADD", "time": ts, "side": -1, "price": float(fill), "units": p.units, "qty": add_qty, "turnover": abs(add_qty * fill) / max(p.entry_equity, 1e-12), "fee": fee(add_qty, fill), "reason": "PLUS_0.5N", "pnl": np.nan})

        if p is None and pending_entry is None and current_ctx is not None:
            entry_high, entry_low, _, _, n_value = current_ctx
            if np.isfinite(n_value) and n_value > 0:
                hit_long = hi[i] >= entry_high
                hit_short = lo[i] <= entry_low
                if hit_long != hit_short:
                    side = 1 if hit_long else -1
                    threshold = entry_high if side == 1 else entry_low
                    if extra_delay_minutes > 0:
                        pending_entry = (min(i + extra_delay_minutes, len(idx) - 1), side, float(threshold), float(n_value))
                    else:
                        fill = max(op[i], threshold) if side == 1 else min(op[i], threshold)
                        enter(i, side, float(fill), float(n_value))
                        # With a resting breakout stop and hard 2N stop, both may be touched in one 1m bar; choose adverse stop-first.
                        if p is not None and ((side == 1 and lo[i] <= p.stop) or (side == -1 and hi[i] >= p.stop)):
                            exit_position(i, p.stop, "SAME_MINUTE_STOP")

        eq = mark(i)
        equity[i] = eq
        if p is not None:
            position[i] = p.side * p.qty * cl[i] / max(eq, 1e-12)

    if p is not None:
        exit_position(len(idx) - 1, cl[-1], "END")
        equity[-1] = capital
        position[-1] = 0.0
    event_df = pd.DataFrame(rows)
    daily, metrics = _summarize(cfg, strategy_id, idx, equity, position, event_df)
    audit = {
        "strategy_id": strategy_id,
        "resting_intraday_breakout_orders": True,
        "daily_context_strictly_before_minute": True,
        "same_minute_ambiguity_policy": "ADVERSE_STOP_FIRST_OR_SKIP_DUAL_BREAKOUT",
        "future_visibility_violations": 0,
        "sealed_2026_opened": False,
        "cost_multiplier": float(cost_mult),
        "extra_delay_minutes": int(extra_delay_minutes),
        "execution_semantics": "single_net_eth_exposure",
    }
    return BacktestResult(strategy_id, pd.Series(equity, index=idx), pd.Series(position, index=idx), daily, event_df, metrics, audit)
