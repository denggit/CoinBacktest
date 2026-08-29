#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal 1m replay and portfolio metrics for R01 opportunity strategies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from .contracts import PortfolioSelectionKey
from .opportunity import TradeTemplate


@dataclass(frozen=True)
class StrategyTrade:
    signal_time: str
    entry_time: str
    exit_time: str
    side: str
    entry_price: float
    exit_price: float
    exit_reason: str
    horizon_minutes: int
    take_profit: float
    stop_loss: float
    long_score: float
    short_score: float
    gross_price_return: float
    mfe_price_return: float
    mae_price_return: float
    notional_multiple: float
    same_bar_both_hit: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _prepare_path(path: pd.DataFrame) -> pd.DataFrame:
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in path.columns]
    if missing:
        raise ValueError(f"1m replay path missing: {missing}")
    bars = path.copy()
    bars.index = pd.DatetimeIndex(pd.to_datetime(bars.index, errors="coerce"))
    bars = bars.loc[~bars.index.isna()]
    if bars.index.tz is not None:
        bars.index = bars.index.tz_localize(None)
    bars = bars[~bars.index.duplicated(keep="last")].sort_index()
    for col in required:
        bars[col] = pd.to_numeric(bars[col], errors="coerce")
    valid = (bars[required] > 0).all(axis=1)
    return bars.loc[valid, required]


def replay_strategy(
    *,
    decision_times_ns: np.ndarray,
    long_scores: np.ndarray,
    short_scores: np.ndarray,
    path_1m: pd.DataFrame,
    template: TradeTemplate,
    long_threshold: float,
    short_threshold: float,
    round_trip_cost: float,
    risk_per_trade: float = 0.01,
    max_notional_multiple: float = 2.0,
    entry_delay_minutes: int = 0,
) -> pd.DataFrame:
    """Replay one non-overlapping ETH sleeve on the exact 1m OHLC path.

    State is already frozen at decision_time by R00. Base entry uses the open
    of the 1m bar beginning at decision_time; stress runs may delay that entry
    by whole minutes. If TP and SL occur in the same 1m bar, the stop is assumed
    first. This is intentionally conservative.
    """

    template.validate()
    decisions = pd.to_datetime(np.asarray(decision_times_ns, dtype=np.int64), unit="ns")
    long_scores = np.asarray(long_scores, dtype=float)
    short_scores = np.asarray(short_scores, dtype=float)
    if len(decisions) != len(long_scores) or len(decisions) != len(short_scores):
        raise ValueError("decision_times and score arrays must have equal length")

    bars = _prepare_path(path_1m)
    if bars.empty or len(decisions) == 0:
        return pd.DataFrame(columns=list(StrategyTrade.__dataclass_fields__))
    bar_ns = bars.index.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    opens = bars["open"].to_numpy(dtype=float)
    highs = bars["high"].to_numpy(dtype=float)
    lows = bars["low"].to_numpy(dtype=float)
    closes = bars["close"].to_numpy(dtype=float)
    pos_lookup = {int(ts): i for i, ts in enumerate(bar_ns)}
    h = int(template.horizon_minutes)
    delay = int(entry_delay_minutes)
    if delay < 0:
        raise ValueError("entry_delay_minutes must be >= 0")
    delay_ns = int(pd.Timedelta(minutes=delay).value)
    tp = float(template.take_profit)
    sl = float(template.stop_loss)
    # Sizing includes base round-trip cost in the stop-risk denominator so the
    # intended 1% risk is not silently 1% + fees.
    notional = min(float(max_notional_multiple), float(risk_per_trade) / max(sl + float(round_trip_cost), 1e-12))
    busy_until_ns = -1
    trades: list[StrategyTrade] = []

    for i, ts in enumerate(decisions.to_numpy(dtype="datetime64[ns]").astype(np.int64)):
        if int(ts) <= busy_until_ns:
            continue
        ls = float(long_scores[i])
        ss = float(short_scores[i])
        long_ok = np.isfinite(ls) and ls >= float(long_threshold) and ls > 0.0
        short_ok = np.isfinite(ss) and ss >= float(short_threshold) and ss > 0.0
        if not long_ok and not short_ok:
            continue
        side = "LONG" if long_ok and (not short_ok or ls >= ss) else "SHORT"
        entry_ns = int(ts) + delay_ns
        p = pos_lookup.get(entry_ns)
        if p is None or p + h > len(bars):
            continue
        entry = float(opens[p])
        if not np.isfinite(entry) or entry <= 0:
            continue
        end = p + h
        hi = highs[p:end]
        lo = lows[p:end]
        if side == "LONG":
            tp_mask = hi >= entry * (1.0 + tp)
            sl_mask = lo <= entry * (1.0 - sl)
        else:
            tp_mask = lo <= entry * (1.0 - tp)
            sl_mask = hi >= entry * (1.0 + sl)
        tp_hits = np.flatnonzero(tp_mask)
        sl_hits = np.flatnonzero(sl_mask)
        first_tp = int(tp_hits[0]) if len(tp_hits) else h + 1
        first_sl = int(sl_hits[0]) if len(sl_hits) else h + 1
        same_bar = first_tp == first_sl and first_tp <= h - 1

        if first_sl <= first_tp and first_sl <= h - 1:
            k = first_sl
            reason = "SL"
            exit_price = entry * (1.0 - sl) if side == "LONG" else entry * (1.0 + sl)
            gross = -sl
        elif first_tp < first_sl and first_tp <= h - 1:
            k = first_tp
            reason = "TP"
            exit_price = entry * (1.0 + tp) if side == "LONG" else entry * (1.0 - tp)
            gross = tp
        else:
            k = h - 1
            reason = "HORIZON_EXIT"
            exit_price = float(closes[p + k])
            gross = (exit_price / entry - 1.0) if side == "LONG" else (1.0 - exit_price / entry)
        observed_hi = hi[: k + 1]
        observed_lo = lo[: k + 1]
        if side == "LONG":
            mfe = float(np.nanmax(observed_hi / entry - 1.0))
            mae = float(np.nanmin(observed_lo / entry - 1.0))
        else:
            mfe = float(np.nanmax(1.0 - observed_lo / entry))
            mae = float(np.nanmin(1.0 - observed_hi / entry))
        if reason == "SL":
            mae = -sl
        if reason == "TP":
            mfe = tp
        exit_ns = int(bar_ns[p + k])
        busy_until_ns = exit_ns
        trades.append(
            StrategyTrade(
                signal_time=str(pd.Timestamp(ts)),
                entry_time=str(pd.Timestamp(entry_ns)),
                exit_time=str(pd.Timestamp(exit_ns)),
                side=side,
                entry_price=entry,
                exit_price=float(exit_price),
                exit_reason=reason,
                horizon_minutes=h,
                take_profit=tp,
                stop_loss=sl,
                long_score=ls,
                short_score=ss,
                gross_price_return=float(gross),
                mfe_price_return=float(mfe),
                mae_price_return=float(mae),
                notional_multiple=float(notional),
                same_bar_both_hit=bool(same_bar),
            )
        )
    return pd.DataFrame([x.to_dict() for x in trades])


def _max_consecutive_losing_days(daily_pnl: pd.Series) -> int:
    if daily_pnl.empty:
        return 0
    idx = pd.date_range(daily_pnl.index.min(), daily_pnl.index.max(), freq="1D")
    pnl = daily_pnl.reindex(idx, fill_value=0.0)
    best = run = 0
    for value in pnl.to_numpy(dtype=float):
        if value < 0:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return int(best)


def _max_flat_days(trades: pd.DataFrame, start: pd.Timestamp, end_exclusive: pd.Timestamp) -> float:
    if trades.empty:
        return float((end_exclusive - start) / pd.Timedelta(days=1))
    entries = pd.to_datetime(trades["entry_time"])
    exits = pd.to_datetime(trades["exit_time"])
    gaps = [max(pd.Timedelta(0), entries.iloc[0] - start)]
    for prev_exit, next_entry in zip(exits.iloc[:-1], entries.iloc[1:]):
        gaps.append(max(pd.Timedelta(0), next_entry - prev_exit))
    gaps.append(max(pd.Timedelta(0), end_exclusive - exits.iloc[-1]))
    return float(max(gaps) / pd.Timedelta(days=1))


def evaluate_trades(
    trades: pd.DataFrame,
    *,
    start: str | pd.Timestamp,
    end_exclusive: str | pd.Timestamp,
    round_trip_cost: float,
) -> dict[str, float | int]:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end_exclusive)
    years = max(float((end_ts - start_ts) / pd.Timedelta(days=365.25)), 1e-9)
    if trades is None or trades.empty:
        return {
            "trades": 0, "trades_per_month": 0.0, "positive_month_ratio": 0.0,
            "win_rate": 0.0, "profit_factor": 0.0,
            "max_flat_days": float((end_ts - start_ts) / pd.Timedelta(days=1)),
            "max_consecutive_losing_days": 0, "max_drawdown_pct": 0.0,
            "cagr_pct": 0.0, "total_return_pct": 0.0,
            "same_bar_both_hit_count": 0,
        }
    frame = trades.copy()
    frame["exit_time"] = pd.to_datetime(frame["exit_time"])
    frame["net_price_return"] = pd.to_numeric(frame["gross_price_return"], errors="coerce") - float(round_trip_cost)
    frame["equity_return"] = frame["net_price_return"] * pd.to_numeric(frame["notional_multiple"], errors="coerce")
    r = frame["equity_return"].fillna(0.0).to_numpy(dtype=float)
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    mae_series = pd.to_numeric(frame.get("mae_price_return", frame["gross_price_return"]), errors="coerce").to_numpy(dtype=float)
    notionals = pd.to_numeric(frame["notional_multiple"], errors="coerce").to_numpy(dtype=float)
    for final_r, mae_price, notional in zip(r, mae_series, notionals):
        if not np.isfinite(mae_price):
            mae_price = min(float(final_r) / max(float(notional), 1e-12), 0.0)
        adverse_equity_return = float(notional) * (min(float(mae_price), 0.0) - float(round_trip_cost))
        trough = equity * max(1.0 + adverse_equity_return, 1e-12)
        max_dd = max(max_dd, 1.0 - trough / max(peak, 1e-12))
        equity *= max(1.0 + float(final_r), 1e-12)
        max_dd = max(max_dd, 1.0 - equity / max(peak, 1e-12))
        peak = max(peak, equity)
    total = float(equity - 1.0)
    cagr = float((max(1.0 + total, 1e-12) ** (1.0 / years) - 1.0))
    wins = r[r > 0]
    losses = r[r < 0]
    pf = float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 1e-12 else (float("inf") if len(wins) else 0.0)
    daily = frame.assign(day=frame["exit_time"].dt.floor("D")).groupby("day")["equity_return"].sum()
    month_index = pd.period_range(start_ts.to_period("M"), (end_ts - pd.Timedelta(nanoseconds=1)).to_period("M"), freq="M")
    monthly_growth: list[float] = []
    for period in month_index:
        g = frame.loc[frame["exit_time"].dt.to_period("M") == period, "equity_return"].fillna(0.0).to_numpy(dtype=float)
        monthly_growth.append(float(np.prod(1.0 + g) - 1.0) if len(g) else 0.0)
    positive_month_ratio = float(np.mean(np.asarray(monthly_growth) > 0)) if monthly_growth else 0.0
    return {
        "trades": int(len(frame)),
        "trades_per_month": float(len(frame) / max(len(month_index), 1)),
        "positive_month_ratio": positive_month_ratio,
        "win_rate": float((r > 0).mean()),
        "profit_factor": pf,
        "max_flat_days": _max_flat_days(frame, start_ts, end_ts),
        "max_consecutive_losing_days": _max_consecutive_losing_days(daily),
        "max_drawdown_pct": float(max_dd * 100.0),
        "cagr_pct": cagr * 100.0,
        "total_return_pct": total * 100.0,
        "same_bar_both_hit_count": int(pd.Series(frame.get("same_bar_both_hit", False)).astype(bool).sum()),
    }


def selection_key(metrics: dict[str, float | int]) -> PortfolioSelectionKey:
    return PortfolioSelectionKey.from_metrics(
        max_flat_days=float(metrics["max_flat_days"]),
        max_consecutive_losing_days=int(metrics["max_consecutive_losing_days"]),
        max_drawdown_pct=float(metrics["max_drawdown_pct"]),
        cagr_pct=float(metrics["cagr_pct"]),
        total_return_pct=float(metrics["total_return_pct"]),
    )
