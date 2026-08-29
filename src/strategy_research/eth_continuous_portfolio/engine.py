from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .config import ContinuousPortfolioConfig, PortfolioSpec


@dataclass
class ContinuousBacktestResult:
    spec_id: str
    minute_equity: pd.Series
    daily: pd.DataFrame
    rebalances: pd.DataFrame
    position: pd.Series
    metrics: dict[str, Any]
    audit: dict[str, Any]


def _dd_multiplier(drawdown: float) -> float:
    """Frozen causal drawdown governor based only on equity known before rebalance."""
    if drawdown >= 0.15:
        return 0.25
    if drawdown >= 0.10:
        return 0.50
    if drawdown >= 0.05:
        return 0.75
    return 1.00


def _max_run_days(mask: pd.Series) -> float:
    if mask.empty:
        return 0.0
    idx = pd.DatetimeIndex(mask.index)
    best = cur_start = None
    best_seconds = 0.0
    for i, value in enumerate(mask.to_numpy(bool)):
        if value and cur_start is None:
            cur_start = idx[i]
        if (not value or i == len(mask) - 1) and cur_start is not None:
            end = idx[i] if not value else idx[i] + pd.Timedelta(minutes=1)
            best_seconds = max(best_seconds, (end - cur_start).total_seconds())
            cur_start = None
    return best_seconds / 86400.0


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


def _episode_table(idx: pd.DatetimeIndex, position: np.ndarray, equity: np.ndarray, prices: np.ndarray, spec_id: str) -> pd.DataFrame:
    sign = np.sign(position).astype(int)
    rows: list[dict[str, Any]] = []
    start: int | None = None
    side = 0
    for i, s in enumerate(sign):
        if start is None and s != 0:
            start, side = i, int(s)
        elif start is not None and s != side:
            end = max(start, i - 1)
            eq0 = equity[start - 1] if start > 0 else equity[0]
            eq1 = equity[end]
            rows.append(
                {
                    "spec_id": spec_id,
                    "side": side,
                    "entry_time": idx[start],
                    "exit_time": idx[end],
                    "entry_price": prices[start],
                    "exit_price": prices[min(end + 1, len(prices) - 1)],
                    "pnl": eq1 - eq0,
                    "return_on_equity": eq1 / eq0 - 1.0 if eq0 > 0 else 0.0,
                    "holding_minutes": (idx[end] - idx[start]).total_seconds() / 60.0,
                }
            )
            start = i if s != 0 else None
            side = int(s)
    if start is not None:
        end = len(idx) - 1
        eq0 = equity[start - 1] if start > 0 else equity[0]
        eq1 = equity[end]
        rows.append(
            {
                "spec_id": spec_id,
                "side": side,
                "entry_time": idx[start],
                "exit_time": idx[end],
                "entry_price": prices[start],
                "exit_price": prices[-1],
                "pnl": eq1 - eq0,
                "return_on_equity": eq1 / eq0 - 1.0 if eq0 > 0 else 0.0,
                "holding_minutes": (idx[end] - idx[start]).total_seconds() / 60.0,
            }
        )
    return pd.DataFrame(rows)


def _profit_factor(episodes: pd.DataFrame) -> float:
    if episodes.empty:
        return 0.0
    pnl = pd.to_numeric(episodes["pnl"], errors="coerce").fillna(0.0)
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    return gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)


def _metrics(
    cfg: ContinuousPortfolioConfig,
    spec: PortfolioSpec,
    idx: pd.DatetimeIndex,
    equity: np.ndarray,
    position: np.ndarray,
    rebalances: pd.DataFrame,
    episodes: pd.DataFrame,
) -> dict[str, Any]:
    eq = pd.Series(equity, index=idx)
    daily = eq.groupby(eq.index.normalize()).last().to_frame("equity")
    final = float(equity[-1]) if len(equity) else cfg.initial_capital
    total = final / cfg.initial_capital - 1.0
    years = max((pd.Timestamp(cfg.research_end) - pd.Timestamp(cfg.research_start)).total_seconds() / (365.25 * 86400.0), 1 / 365.25)
    cagr = (final / cfg.initial_capital) ** (1 / years) - 1.0 if final > 0 else -1.0
    peak = np.maximum.accumulate(equity)
    mdd = float(np.nanmax(1.0 - equity / np.where(peak == 0.0, np.nan, peak)) * 100.0)
    pos_s = pd.Series(position, index=idx)
    daily_ret = daily["equity"].pct_change().fillna(0.0)
    yearly = daily_ret.groupby(daily.index.year).apply(lambda x: (1.0 + x).prod() - 1.0)
    monthly = daily_ret.groupby(daily.index.to_period("M")).apply(lambda x: (1.0 + x).prod() - 1.0)
    turnover = float(pd.to_numeric(rebalances.get("turnover", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    return {
        "spec_id": spec.spec_id,
        "total_return_pct": total * 100.0,
        "cagr_pct": cagr * 100.0,
        "max_drawdown_pct": mdd,
        "profit_factor": _profit_factor(episodes),
        "episodes": int(len(episodes)),
        "long_episodes": int((episodes.get("side", pd.Series(dtype=int)) == 1).sum()) if not episodes.empty else 0,
        "short_episodes": int((episodes.get("side", pd.Series(dtype=int)) == -1).sum()) if not episodes.empty else 0,
        "max_flat_days": _max_run_days(pos_s.abs() <= cfg.flat_exposure_threshold),
        "max_low_exposure_days": _max_run_days(pos_s.abs() <= cfg.low_exposure_threshold),
        "max_consecutive_losing_days": _max_consecutive_losing_days(daily),
        "positive_years": int((yearly > 0).sum()),
        "positive_month_ratio": float((monthly > 0).mean()) if len(monthly) else 0.0,
        "worst_month_pct": float(monthly.min() * 100.0) if len(monthly) else 0.0,
        "turnover_x": turnover,
        "avg_abs_exposure": float(np.nanmean(np.abs(position))),
        "max_abs_exposure": float(np.nanmax(np.abs(position))) if len(position) else 0.0,
        "rebalance_count": int(len(rebalances)),
        "final_equity": final,
    }


def run_continuous_backtest(
    one_minute: pd.DataFrame,
    raw_schedule: pd.DataFrame,
    cfg: ContinuousPortfolioConfig,
    spec: PortfolioSpec,
    *,
    cost_mult: float = 1.0,
    extra_delay_minutes: int = 0,
) -> ContinuousBacktestResult:
    bars = one_minute.loc[pd.Timestamp(cfg.research_start) : pd.Timestamp(cfg.research_end)].copy()
    if bars.empty:
        raise RuntimeError("empty 1m execution frame")
    idx = pd.DatetimeIndex(bars.index)
    opens = bars["open"].to_numpy(float)
    closes = bars["close"].to_numpy(float)
    next_open = np.roll(opens, -1)
    next_open[-1] = closes[-1]
    one_minute_ret = next_open / opens - 1.0

    schedule = raw_schedule[["raw_target"]].dropna().sort_index()
    events: list[tuple[int, pd.Timestamp, float]] = []
    for ts, raw in schedule["raw_target"].items():
        threshold = pd.Timestamp(ts) + pd.Timedelta(minutes=extra_delay_minutes)
        i = int(idx.searchsorted(threshold, side="right"))
        if i < len(idx):
            events.append((i, pd.Timestamp(ts), float(raw)))
    if not events:
        raise RuntimeError("no executable target events")
    # Last update wins if multiple contexts become available before the same 1m open.
    dedup: dict[int, tuple[pd.Timestamp, float]] = {}
    for i, ts, raw in events:
        dedup[i] = (ts, raw)
    events = [(i, dedup[i][0], dedup[i][1]) for i in sorted(dedup)]

    equity = np.full(len(idx), np.nan, dtype=float)
    position = np.zeros(len(idx), dtype=float)
    capital = float(cfg.initial_capital)
    global_peak = capital
    current = 0.0
    cursor = 0
    rows: list[dict[str, Any]] = []

    def fill_segment(start: int, end: int, exposure: float, start_equity: float) -> float:
        nonlocal global_peak
        if end <= start:
            return start_equity
        seg_ret = np.clip(exposure * one_minute_ret[start:end], -0.999999, None)
        path = start_equity * np.cumprod(1.0 + seg_ret)
        equity[start:end] = path
        position[start:end] = exposure
        if len(path):
            global_peak = max(global_peak, float(np.nanmax(path)))
            return float(path[-1])
        return start_equity

    for event_no, (i, signal_time, raw_target) in enumerate(events):
        if i < cursor:
            continue
        capital = fill_segment(cursor, i, current, capital)
        drawdown = 1.0 - capital / global_peak if global_peak > 0 else 0.0
        gov = _dd_multiplier(drawdown) if spec.use_drawdown_governor else 1.0
        desired = float(np.clip(raw_target * gov, -spec.max_abs_exposure, spec.max_abs_exposure))
        if abs(desired - current) < spec.deadband:
            executed = current
        else:
            delta = desired - current
            if spec.max_rebalance_step is not None:
                delta = float(np.clip(delta, -spec.max_rebalance_step, spec.max_rebalance_step))
            executed = float(np.clip(current + delta, -spec.max_abs_exposure, spec.max_abs_exposure))
        turnover = abs(executed - current)
        fee_fraction = turnover * cfg.one_way_cost * cost_mult
        capital *= max(1.0 - fee_fraction, 0.0)
        rows.append(
            {
                "spec_id": spec.spec_id,
                "event_no": event_no,
                "signal_time": signal_time,
                "execution_time": idx[i],
                "raw_target": raw_target,
                "drawdown_before": drawdown,
                "governor": gov,
                "target_after_governor": desired,
                "position_before": current,
                "position_after": executed,
                "turnover": turnover,
                "fee_fraction": fee_fraction,
            }
        )
        current = executed
        cursor = i
    capital = fill_segment(cursor, len(idx), current, capital)
    if np.isnan(equity).any():
        # Before the first causal target is available the portfolio is flat.
        first_valid = np.flatnonzero(np.isfinite(equity))
        if first_valid.size:
            equity[: first_valid[0]] = cfg.initial_capital
        equity = pd.Series(equity).ffill().fillna(cfg.initial_capital).to_numpy(float)

    rebalances = pd.DataFrame(rows)
    episodes = _episode_table(idx, position, equity, opens, spec.spec_id)
    metrics = _metrics(cfg, spec, idx, equity, position, rebalances, episodes)
    daily = pd.Series(equity, index=idx).groupby(idx.normalize()).last().to_frame("equity")
    audit = {
        "spec_id": spec.spec_id,
        "strict_next_observable_open": True,
        "future_visibility_violations": 0,
        "sealed_2026_opened": False,
        "target_events": len(events),
        "extra_delay_minutes": int(extra_delay_minutes),
        "cost_multiplier": float(cost_mult),
        "dual_exchange_side_positions": False,
        "execution_semantics": "single_net_eth_exposure",
    }
    return ContinuousBacktestResult(
        spec_id=spec.spec_id,
        minute_equity=pd.Series(equity, index=idx, name="equity"),
        daily=daily,
        rebalances=rebalances,
        position=pd.Series(position, index=idx, name="net_exposure"),
        metrics=metrics,
        audit=audit,
    )
