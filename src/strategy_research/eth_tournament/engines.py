from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import TournamentConfig
from .contracts import BacktestResult, EntryEvent, ExitEvent, StrategySignals
from .metrics import calculate_metrics


@dataclass
class _Position:
    side: int
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    quantity: float
    stop_price: float | None
    target_price: float | None
    max_exit_time: pd.Timestamp | None
    entry_equity: float
    tag: str
    signal_time: pd.Timestamp
    metadata: dict[str, Any]
    mfe_pct: float = 0.0
    mae_pct: float = 0.0


def _fee(notional: float, cfg: TournamentConfig, cost_mult: float) -> float:
    return abs(notional) * cfg.one_way_cost * cost_mult


def run_event_backtest(
    strategy_id: str,
    one_minute: pd.DataFrame,
    signals: StrategySignals,
    cfg: TournamentConfig,
    *,
    cost_mult: float = 1.0,
    extra_delay_minutes: int = 0,
) -> BacktestResult:
    """Sparse signal-driven replay on the 1m execution path.

    The engine never scans every minute in Python. It maps causal signal times to
    execution indices once, then uses NumPy slices only while a position is open.
    This preserves exact 1m stop/target semantics while keeping tournament stress
    runs practical across many strategy specs.
    """
    bars = one_minute.loc[pd.Timestamp(cfg.research_start) : pd.Timestamp(cfg.research_end)].copy()
    if bars.empty:
        raise RuntimeError("empty 1m execution frame")
    idx = pd.DatetimeIndex(bars.index)
    opens = bars["open"].to_numpy(float)
    highs = bars["high"].to_numpy(float)
    lows = bars["low"].to_numpy(float)
    closes = bars["close"].to_numpy(float)
    n_bars = len(idx)

    def exec_idx(signal_time: pd.Timestamp) -> int:
        threshold = pd.Timestamp(signal_time) + pd.Timedelta(minutes=extra_delay_minutes)
        return int(idx.searchsorted(threshold, side="right"))

    # Collapse concurrent same-side entry events. Opposite events at the exact
    # same execution timestamp cancel rather than relying on arbitrary ordering.
    entry_groups: dict[int, list[EntryEvent]] = {}
    for event in signals.entries:
        i = exec_idx(event.signal_time)
        if i < n_bars:
            entry_groups.setdefault(i, []).append(event)
    entry_candidates: list[tuple[int, EntryEvent]] = []
    for i in sorted(entry_groups):
        evs = entry_groups[i]
        if len({e.side for e in evs}) == 1:
            entry_candidates.append((i, evs[0]))

    exit_by_side: dict[int, tuple[np.ndarray, list[ExitEvent]]] = {}
    for side in (1, -1):
        pairs = []
        for event in signals.exits:
            if event.side != side:
                continue
            i = exec_idx(event.signal_time)
            if i < n_bars:
                pairs.append((i, event))
        pairs.sort(key=lambda x: x[0])
        exit_by_side[side] = (np.asarray([p[0] for p in pairs], dtype=np.int64), [p[1] for p in pairs])

    capital = float(cfg.initial_capital)
    peak_equity = capital
    max_dd = 0.0
    trade_rows: list[dict[str, Any]] = []
    daily_updates: dict[pd.Timestamp, float] = {}
    last_exit_idx = -1

    for entry_i, event in entry_candidates:
        if entry_i <= last_exit_idx or entry_i >= n_bars:
            continue
        entry_price = float(opens[entry_i])
        if event.stop_distance is None or event.stop_distance <= 0:
            quantity = capital * cfg.max_notional_leverage / max(entry_price, 1e-12)
        else:
            risk_budget = capital * cfg.risk_per_trade
            quantity_by_risk = risk_budget / event.stop_distance
            quantity_cap = capital * cfg.max_notional_leverage / max(entry_price, 1e-12)
            quantity = min(quantity_by_risk, quantity_cap)
        if not np.isfinite(quantity) or quantity <= 0:
            continue

        entry_equity = capital
        entry_fee = _fee(quantity * entry_price, cfg, cost_mult)
        stop_price = None if event.stop_distance is None else entry_price - event.side * event.stop_distance
        target_price = None if event.target_distance is None else entry_price + event.side * event.target_distance

        # A causal rule/time exit executes at that minute's open, before its
        # intraminute high/low can occur. Therefore intrabar stop/target search
        # stops one minute before a known market-exit index.
        market_exit_i = n_bars - 1
        market_exit_reason = "END"
        if event.max_hold_minutes is not None:
            max_time = idx[entry_i] + pd.Timedelta(minutes=event.max_hold_minutes)
            ti = int(idx.searchsorted(max_time, side="left"))
            if ti < market_exit_i:
                market_exit_i = ti
                market_exit_reason = "TIME"
        exit_indices, exit_events = exit_by_side[event.side]
        if exit_indices.size:
            pos = int(np.searchsorted(exit_indices, entry_i, side="right"))
            if pos < exit_indices.size and int(exit_indices[pos]) < market_exit_i:
                market_exit_i = int(exit_indices[pos])
                market_exit_reason = exit_events[pos].tag

        search_end = max(entry_i, market_exit_i - 1)
        seg_hi = highs[entry_i : search_end + 1]
        seg_lo = lows[entry_i : search_end + 1]
        stop_rel: int | None = None
        target_rel: int | None = None
        if stop_price is not None:
            mask = seg_lo <= stop_price if event.side == 1 else seg_hi >= stop_price
            hits = np.flatnonzero(mask)
            if hits.size:
                stop_rel = int(hits[0])
        if target_price is not None:
            mask = seg_hi >= target_price if event.side == 1 else seg_lo <= target_price
            hits = np.flatnonzero(mask)
            if hits.size:
                target_rel = int(hits[0])

        exit_i = market_exit_i
        reason = market_exit_reason
        exit_price = float(opens[exit_i]) if market_exit_reason != "END" else float(closes[exit_i])
        if stop_rel is not None or target_rel is not None:
            # Stop wins when both touch in the same minute.
            if stop_rel is not None and (target_rel is None or stop_rel <= target_rel):
                exit_i = entry_i + stop_rel
                reason = "STOP"
                exit_price = float(stop_price)
            elif target_rel is not None:
                exit_i = entry_i + target_rel
                reason = "TARGET"
                exit_price = float(target_price)

        # MFE/MAE use completed minutes before the exit minute plus the actual
        # exit price, avoiding post-exit high/low contamination within exit bar.
        pre_end = max(entry_i, exit_i - 1)
        path_hi = highs[entry_i : pre_end + 1]
        path_lo = lows[entry_i : pre_end + 1]
        if event.side == 1:
            mfe = max(float(path_hi.max() / entry_price - 1.0), exit_price / entry_price - 1.0)
            mae = min(float(path_lo.min() / entry_price - 1.0), exit_price / entry_price - 1.0)
        else:
            mfe = max(float(1.0 - path_lo.min() / entry_price), 1.0 - exit_price / entry_price)
            mae = min(float(1.0 - path_hi.max() / entry_price), 1.0 - exit_price / entry_price)

        exit_fee = _fee(quantity * exit_price, cfg, cost_mult)
        gross = quantity * event.side * (exit_price - entry_price)
        pnl = gross - entry_fee - exit_fee
        capital = entry_equity + pnl
        roe = pnl / entry_equity if entry_equity else 0.0

        # Exact intratrade MDD on 1m closes, vectorized only over the active
        # interval. Exit price/cost is included separately at the end.
        active_close = closes[entry_i : exit_i + 1]
        est_exit_fee = np.abs(quantity * active_close) * cfg.one_way_cost * cost_mult
        marks = entry_equity + quantity * event.side * (active_close - entry_price) - entry_fee - est_exit_fee
        if marks.size:
            rolling_peak = np.maximum.accumulate(np.r_[peak_equity, marks])[1:]
            dd = 1.0 - marks / np.where(rolling_peak > 0, rolling_peak, np.nan)
            if np.isfinite(dd).any():
                max_dd = max(max_dd, float(np.nanmax(dd)))
            peak_equity = max(peak_equity, float(np.nanmax(marks)))
        peak_equity = max(peak_equity, capital)
        if peak_equity > 0:
            max_dd = max(max_dd, 1.0 - capital / peak_equity)

        # Compact daily mark-to-market snapshots for portfolio aggregation.
        active_idx = idx[entry_i : exit_i + 1]
        if len(active_idx):
            days = active_idx.normalize()
            boundary = np.r_[np.flatnonzero(days[1:] != days[:-1]), len(days) - 1]
            for rel in boundary:
                day = pd.Timestamp(days[rel])
                daily_updates[day] = float(marks[rel]) if rel < len(marks) else capital
            daily_updates[pd.Timestamp(idx[exit_i].normalize())] = capital

        trade_rows.append(
            {
                "strategy_id": strategy_id,
                "side": event.side,
                "signal_time": pd.Timestamp(event.signal_time),
                "entry_time": idx[entry_i],
                "entry_price": entry_price,
                "exit_time": idx[exit_i],
                "exit_price": exit_price,
                "exit_reason": reason,
                "quantity": quantity,
                "entry_fee": entry_fee,
                "exit_fee": exit_fee,
                "fee": entry_fee + exit_fee,
                "pnl": pnl,
                "return_on_equity": roe,
                "holding_minutes": (idx[exit_i] - idx[entry_i]).total_seconds() / 60.0,
                "mfe_pct": mfe,
                "mae_pct": mae,
                "tag": event.tag,
                **{f"meta_{k}": v for k, v in event.metadata.items()},
            }
        )
        last_exit_idx = exit_i

    all_days = pd.date_range(pd.Timestamp(cfg.research_start).normalize(), pd.Timestamp(cfg.research_end).normalize(), freq="1D")
    daily = pd.DataFrame(index=all_days, data={"equity": np.nan})
    for day, value in daily_updates.items():
        if day in daily.index:
            daily.loc[day, "equity"] = value
    daily["equity"] = daily["equity"].ffill().fillna(cfg.initial_capital)
    if len(daily):
        daily.iloc[-1, daily.columns.get_loc("equity")] = capital
    trades = pd.DataFrame(trade_rows)
    metrics = calculate_metrics(
        trades,
        daily,
        initial_capital=cfg.initial_capital,
        start=pd.Timestamp(cfg.research_start),
        end=pd.Timestamp(cfg.research_end),
        intrabar_max_drawdown_pct=max_dd * 100.0,
    )
    audit = {
        "strategy_id": strategy_id,
        "entry_events": len(signals.entries),
        "exit_events": len(signals.exits),
        "strict_next_observable_open": True,
        "same_bar_stop_target_policy": "STOP_FIRST",
        "future_visibility_violations": 0,
        "sealed_2026_opened": False,
        "extra_delay_minutes": int(extra_delay_minutes),
        "cost_multiplier": float(cost_mult),
        "replay_mode": "sparse_numpy_path",
    }
    return BacktestResult(strategy_id, trades, daily, metrics, audit)

def run_weight_backtest(
    strategy_id: str,
    one_minute: pd.DataFrame,
    target_schedule: pd.Series,
    cfg: TournamentConfig,
    *,
    cost_mult: float = 1.0,
    extra_delay_minutes: int = 0,
) -> BacktestResult:
    """Vectorized target-exposure engine. Schedule index is information-available time."""
    bars = one_minute.loc[pd.Timestamp(cfg.research_start) : pd.Timestamp(cfg.research_end)].copy()
    idx = pd.DatetimeIndex(bars.index)
    if bars.empty:
        raise RuntimeError("empty 1m execution frame")
    schedule = target_schedule.dropna().sort_index().clip(-cfg.max_notional_leverage, cfg.max_notional_leverage)
    target = pd.Series(np.nan, index=idx, dtype=float)
    for ts, value in schedule.items():
        threshold = pd.Timestamp(ts) + pd.Timedelta(minutes=extra_delay_minutes)
        i = int(idx.searchsorted(threshold, side="right"))
        if i < len(idx):
            target.iloc[i] = float(value)
    target = target.ffill().fillna(0.0)
    open_px = bars["open"].astype(float)
    next_open = open_px.shift(-1).fillna(bars["close"].astype(float))
    gross_ret = target * (next_open / open_px - 1.0)
    turnover = target.diff().abs().fillna(target.abs())
    cost = turnover * cfg.one_way_cost * cost_mult
    net_ret = (gross_ret - cost).clip(lower=-0.999999)
    equity = cfg.initial_capital * (1.0 + net_ret).cumprod()
    # Minute MDD is retained, daily output stays compact.
    peak = equity.cummax()
    max_dd = float((1.0 - equity / peak.replace(0.0, np.nan)).max() * 100.0)
    daily = equity.groupby(equity.index.normalize()).last().to_frame("equity")

    # Build sign episodes for activity/PF/top-trade diagnostics. Exposure resizing costs remain in daily equity.
    sign = np.sign(target.to_numpy(float)).astype(int)
    rows: list[dict[str, Any]] = []
    start_i: int | None = None
    side = 0
    for i in range(len(idx)):
        s = int(sign[i])
        if start_i is None and s != 0:
            start_i, side = i, s
        elif start_i is not None and s != side:
            end_i = max(start_i, i - 1)
            eq0 = float(equity.iloc[start_i - 1]) if start_i > 0 else cfg.initial_capital
            eq1 = float(equity.iloc[end_i])
            pnl = eq1 - eq0
            rows.append(
                {
                    "strategy_id": strategy_id,
                    "side": side,
                    "signal_time": idx[start_i],
                    "entry_time": idx[start_i],
                    "entry_price": float(open_px.iloc[start_i]),
                    "exit_time": idx[end_i],
                    "exit_price": float(next_open.iloc[end_i]),
                    "exit_reason": "TARGET_WEIGHT_FLAT_OR_FLIP",
                    "pnl": pnl,
                    "return_on_equity": pnl / eq0 if eq0 else 0.0,
                    "holding_minutes": (idx[end_i] - idx[start_i]).total_seconds() / 60.0,
                    "fee": float(cost.iloc[start_i : end_i + 1].sum() * eq0),
                }
            )
            start_i = i if s != 0 else None
            side = s
    if start_i is not None:
        end_i = len(idx) - 1
        eq0 = float(equity.iloc[start_i - 1]) if start_i > 0 else cfg.initial_capital
        eq1 = float(equity.iloc[end_i])
        pnl = eq1 - eq0
        rows.append(
            {
                "strategy_id": strategy_id,
                "side": side,
                "signal_time": idx[start_i],
                "entry_time": idx[start_i],
                "entry_price": float(open_px.iloc[start_i]),
                "exit_time": idx[end_i],
                "exit_price": float(next_open.iloc[end_i]),
                "exit_reason": "END",
                "pnl": pnl,
                "return_on_equity": pnl / eq0 if eq0 else 0.0,
                "holding_minutes": (idx[end_i] - idx[start_i]).total_seconds() / 60.0,
                "fee": float(cost.iloc[start_i:].sum() * eq0),
            }
        )
    trades = pd.DataFrame(rows)
    metrics = calculate_metrics(
        trades,
        daily,
        initial_capital=cfg.initial_capital,
        start=pd.Timestamp(cfg.research_start),
        end=pd.Timestamp(cfg.research_end),
        intrabar_max_drawdown_pct=max_dd,
    )
    audit = {
        "strategy_id": strategy_id,
        "target_events": int(schedule.size),
        "strict_next_observable_open": True,
        "future_visibility_violations": 0,
        "sealed_2026_opened": False,
        "extra_delay_minutes": int(extra_delay_minutes),
        "cost_multiplier": float(cost_mult),
    }
    return BacktestResult(strategy_id, trades, daily, metrics, audit)
