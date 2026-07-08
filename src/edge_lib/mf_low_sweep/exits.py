#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Time48 exit simulation for ETH MF Low Sweep."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.edge_lib.mf_low_sweep.config import MarketCache, UpgradeVariant


def entry_cost(args: Any, cost_mult: float = 1.0) -> float:
    return float(args.entry_fee_rate + args.entry_slippage_pct) * float(cost_mult)


def exit_cost(args: Any, cost_mult: float = 1.0) -> float:
    return float(args.exit_fee_rate + args.exit_slippage_pct) * float(cost_mult)


def payoff_ratio(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    wins = vals[vals > 0]
    losses = vals[vals < 0]
    if wins.empty or losses.empty:
        return float("nan")
    return float(wins.mean() / abs(losses.mean()))


def top_winner_share(x: pd.Series, top_n: int = 5) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    pos = vals[vals > 0].sort_values(ascending=False)
    if pos.empty:
        return float("nan")
    denom = float(pos.sum())
    return float(pos.head(top_n).sum() / denom) if denom > 0 else float("nan")


def max_consecutive_losses(x: pd.Series) -> int:
    vals = pd.to_numeric(x, errors="coerce").dropna().to_numpy(dtype=float)
    best = cur = 0
    for v in vals:
        if v < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def profit_factor(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    gp = float(vals[vals > 0].sum())
    gl = float(-vals[vals < 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else float("nan")
    return gp / gl


def equity_and_dd(returns: pd.Series, starting_equity: float = 1.0) -> tuple[pd.Series, pd.Series]:
    vals = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    equity = float(starting_equity) * (1.0 + vals).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return equity, dd


def build_market_cache(bars: pd.DataFrame, args: Any) -> MarketCache:
    volume = pd.to_numeric(bars.get("volume", pd.Series(np.nan, index=bars.index)), errors="coerce")
    vol_window = int(args.reclaim_volume_window)
    vol_min = max(3, vol_window // 3)
    return MarketCache(
        index=pd.DatetimeIndex(bars.index),
        open=pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float),
        high=pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float),
        low=pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float),
        close=pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float),
        volume=volume.to_numpy(dtype=float),
        reclaim_vol_base=volume.shift(1).rolling(vol_window, min_periods=vol_min).mean().to_numpy(dtype=float),
    )


def horizon_from_exit_mode(exit_mode: str) -> int:
    if str(exit_mode) == "time48":
        return 48
    raise ValueError(f"Unsupported MF low-sweep exit_mode={exit_mode}")


def timing_stats(mtm_low: list[float], mtm_high: list[float]) -> dict[str, object]:
    if not mtm_low or not mtm_high:
        return {
            "mae_time_bars": np.nan,
            "mfe_time_bars": np.nan,
            "first_positive_high_bars": np.nan,
            "mae_before_mfe_flag": np.nan,
        }
    low_arr = np.asarray(mtm_low, dtype=float)
    high_arr = np.asarray(mtm_high, dtype=float)
    mae_pos = int(np.nanargmin(low_arr)) if np.isfinite(low_arr).any() else -1
    mfe_pos = int(np.nanargmax(high_arr)) if np.isfinite(high_arr).any() else -1
    pos_high = np.where(high_arr > 0)[0]
    return {
        "mae_time_bars": mae_pos,
        "mfe_time_bars": mfe_pos,
        "first_positive_high_bars": int(pos_high[0]) if len(pos_high) else np.nan,
        "mae_before_mfe_flag": bool(mae_pos <= mfe_pos) if mae_pos >= 0 and mfe_pos >= 0 else np.nan,
    }


def simulate_time48_trade(
    bars: pd.DataFrame,
    event: pd.Series | dict[str, object],
    signal_pos: int,
    variant: UpgradeVariant,
    args: Any,
    *,
    cost_mult: float = 1.0,
    market: MarketCache | None = None,
) -> dict[str, object]:
    if variant.entry_mode != "next_open":
        raise ValueError(f"Unsupported MF low-sweep entry_mode={variant.entry_mode}")
    if variant.stop_spec.name != "no_stop":
        raise ValueError(f"Unsupported MF low-sweep stop={variant.stop_spec.name}")
    if market is None:
        market = build_market_cache(bars, args)

    opens = market.open
    highs = market.high
    lows = market.low
    closes = market.close
    idx = market.index
    n = len(idx)

    entry_pos = int(signal_pos) + int(args.entry_delay_bars)
    horizon = horizon_from_exit_mode(variant.exit_mode)
    planned_exit_pos = int(signal_pos) + int(horizon)
    if entry_pos >= n or planned_exit_pos >= n or planned_exit_pos <= entry_pos:
        return {"valid": False, "invalid_reason": "insufficient_future_bars"}

    entry_price = float(opens[entry_pos])
    exit_pos = int(planned_exit_pos)
    exit_price = float(closes[exit_pos])
    mtm_low: list[float] = []
    mtm_high: list[float] = []
    for pos in range(entry_pos, exit_pos + 1):
        mtm_low.append(float(lows[pos]) / entry_price - 1.0)
        mtm_high.append(float(highs[pos]) / entry_price - 1.0)

    gross = float(exit_price) / entry_price - 1.0
    net = gross - entry_cost(args, cost_mult) - exit_cost(args, cost_mult)
    timing = timing_stats(mtm_low, mtm_high)

    return {
        "valid": True,
        "variant_name": variant.variant_name,
        "candidate_layer": variant.candidate_layer,
        "support_mode": variant.support_mode,
        "entry_mode": variant.entry_mode,
        "exit_mode": variant.exit_mode,
        "stop_name": variant.stop_spec.name,
        "stop_mode": variant.stop_spec.mode,
        "cost_mult": float(cost_mult),
        "signal_time": event.get("signal_time"),
        "entry_time": idx[int(entry_pos)],
        "exit_time": idx[int(exit_pos)],
        "signal_pos": int(signal_pos),
        "entry_pos": int(entry_pos),
        "exit_pos": int(exit_pos),
        "entry_delay_bars_actual": int(entry_pos - signal_pos),
        "bars_held": int(exit_pos - entry_pos),
        "entry_reason": "next_open",
        "exit_reason": "time_exit_h48",
        "stop_hit": False,
        "target_hit": False,
        "partial_exit_done": False,
        "partial_exit_pos": np.nan,
        "partial_exit_price": np.nan,
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "target_price": np.nan,
        "stop_price": np.nan,
        "stop_pct": np.nan,
        "gross_return_on_equity": float(gross),
        "net_return_on_equity": float(net),
        "mae_on_equity": float(np.nanmin(mtm_low)) if mtm_low else np.nan,
        "mfe_on_equity": float(np.nanmax(mtm_high)) if mtm_high else np.nan,
        "mae_time_bars": timing["mae_time_bars"],
        "mfe_time_bars": timing["mfe_time_bars"],
        "first_positive_high_bars": timing["first_positive_high_bars"],
        "mae_before_mfe_flag": timing["mae_before_mfe_flag"],
        "signal_close": float(event.get("close", np.nan)),
        "signal_open": float(event.get("open", np.nan)),
        "signal_low": float(event.get("low", np.nan)),
        "signal_high": float(event.get("high", np.nan)),
        "swing_level": float(event.get("swing_level", np.nan)),
        "swing_age": float(event.get("swing_age", np.nan)),
        "down_spike_pct": float(event.get("down_spike_pct", np.nan)),
        "close_pos_in_bar": float(event.get("close_pos_in_bar", np.nan)),
        "large_trade_share": float(event.get("large_trade_share", np.nan)),
        "atr_pct": float(event.get("atr_pct", np.nan)),
        "session_bucket": event.get("session_bucket", "NA"),
        "cluster_touch_count_020": event.get("cluster_touch_count_020", np.nan),
        "cluster_touch_count_030": event.get("cluster_touch_count_030", np.nan),
    }


def summarize_trades(trades: pd.DataFrame, args: Any, extra: dict[str, int] | None = None) -> dict[str, object]:
    extra = extra or {}
    if trades.empty:
        out: dict[str, object] = dict(extra)
        out.update({"trades": 0})
        return out
    x = pd.to_numeric(trades["net_return_on_equity"], errors="coerce").fillna(0.0)
    equity, dd = equity_and_dd(x, float(args.starting_equity))
    first_entry = pd.Timestamp(trades["entry_time"].iloc[0])
    last_exit = pd.Timestamp(trades["exit_time"].iloc[-1])
    days = max(1e-9, (last_exit - first_entry).total_seconds() / 86400.0)
    total_ret = float(equity.iloc[-1] / float(args.starting_equity) - 1.0)
    ann_ret = float((1.0 + total_ret) ** (365.0 / days) - 1.0) if total_ret > -1.0 else -1.0
    wins = x[x > 0]
    losses = x[x < 0]
    out = {
        "trades": int(len(trades)),
        "return_total": total_ret,
        "return_annualized": ann_ret,
        "mean_return": float(x.mean()),
        "median_return": float(x.median()),
        "win_rate": float((x > 0).mean()),
        "avg_win": float(wins.mean()) if not wins.empty else np.nan,
        "avg_loss": float(losses.mean()) if not losses.empty else np.nan,
        "payoff_ratio": payoff_ratio(x),
        "profit_factor": profit_factor(x),
        "max_drawdown": float(dd.min()),
        "max_consecutive_losses": max_consecutive_losses(x),
        "top5_winner_share": top_winner_share(x),
        "worst_trade": float(x.min()),
        "best_trade": float(x.max()),
        "mae_mean": float(pd.to_numeric(trades["mae_on_equity"], errors="coerce").mean()),
        "mfe_mean": float(pd.to_numeric(trades["mfe_on_equity"], errors="coerce").mean()),
        "avg_bars_held": float(pd.to_numeric(trades["bars_held"], errors="coerce").mean()),
    }
    out.update(extra)
    return out

