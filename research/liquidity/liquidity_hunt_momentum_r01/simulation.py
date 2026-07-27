#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Conservative Range-Bar execution and forward-path labels."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .features import EPS, _ensure_range_frame, _numeric, datetime_index_to_ns_int64
from .models import LiquidityHuntConfig, StrategyVariant

def _target_and_stop(event: pd.Series, entry_price: float, cfg: LiquidityHuntConfig) -> tuple[float, float, float]:
    side = int(event["side"])
    mode = str(event["mode"])
    if mode == "M1":
        anchor = float(event["sweep_price"])
        stop = anchor * (1.0 - cfg.mode1_stop_buffer_pct) if side == 1 else anchor * (1.0 + cfg.mode1_stop_buffer_pct)
    else:
        anchor = float(event["first_impulse_low"] if side == 1 else event["first_impulse_high"])
        stop = anchor * (1.0 - cfg.mode2_stop_buffer_pct) if side == 1 else anchor * (1.0 + cfg.mode2_stop_buffer_pct)
    risk = side * (entry_price - stop)
    if not np.isfinite(risk) or risk <= 0:
        return np.nan, np.nan, np.nan
    liquidity_target = float(event.get("opposite_liquidity_price", np.nan))
    valid_target = np.isfinite(liquidity_target) and side * (liquidity_target - entry_price) > 0
    fallback = entry_price + side * cfg.fallback_target_r * risk
    target = liquidity_target if valid_target else fallback
    rr = side * (target - entry_price) / risk
    if not np.isfinite(rr) or rr < cfg.minimum_raw_rr:
        target = fallback
        rr = cfg.fallback_target_r
    return float(stop), float(target), float(rr)


def _exit_fill(price: float, side: int, reason: str) -> float:
    # The range-bar path cannot reconstruct queue position.  Stops/targets are
    # filled at their trigger price; dynamic/time exits use the later bar open.
    return float(price)


def _causal_entry_position(
    start_ns: np.ndarray,
    *,
    signal_pos: int,
    signal_time_ns: int,
    entry_delay_bars: int,
) -> int:
    """Return an open whose timestamp is strictly later than the signal.

    Consecutive raw trades can share the same millisecond.  CoinBacktest's
    RangeBarBuilder may therefore produce a next bar whose ``start_ts`` equals
    the completed signal bar's ``end_ts``.  Its open cannot be proven to occur
    after the signal using range-bar data alone, so skip every such bar.
    """

    pos = int(signal_pos) + 1
    while pos < len(start_ns) and int(start_ns[pos]) <= int(signal_time_ns):
        pos += 1
    return pos + int(entry_delay_bars) - 1


def simulate_events(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    cfg: LiquidityHuntConfig,
    variant: StrategyVariant,
    *,
    non_overlapping: bool = True,
) -> pd.DataFrame:
    """Simulate fixed structural exits conservatively on range bars."""

    if events is None or events.empty:
        return pd.DataFrame()
    variant.validate()
    market = _ensure_range_frame(bars)
    market = market.copy()
    market["buy_ratio"] = (_numeric(market, "buy_ratio") if "buy_ratio" in market.columns else _numeric(market, "taker_buy_ratio", 0.5)).fillna(0.5)
    market["sell_ratio"] = 1.0 - market["buy_ratio"]
    if "book_obi_5s" not in market.columns:
        market["book_obi_5s"] = np.nan

    end_ns = datetime_index_to_ns_int64(market["end_ts"])
    open_arr = _numeric(market, "open").to_numpy(dtype=float)
    high_arr = _numeric(market, "high").to_numpy(dtype=float)
    low_arr = _numeric(market, "low").to_numpy(dtype=float)
    close_arr = _numeric(market, "close").to_numpy(dtype=float)
    buy_ratio = _numeric(market, "buy_ratio", 0.5).fillna(0.5).to_numpy(dtype=float)
    sell_ratio = 1.0 - buy_ratio
    obi = _numeric(market, "book_obi_5s").to_numpy(dtype=float)
    start_times = pd.DatetimeIndex(market["start_ts"])
    end_times = pd.DatetimeIndex(market["end_ts"])
    start_ns = datetime_index_to_ns_int64(start_times)

    rows: list[dict[str, object]] = []
    blocked_until = pd.Timestamp.min
    ordered = events.sort_values(["signal_time", "event_id"], kind="stable")
    for event in ordered.itertuples(index=False):
        event_s = pd.Series(event._asdict())
        signal_time = pd.Timestamp(event_s["signal_time"])
        if non_overlapping and signal_time < blocked_until:
            continue
        signal_pos = int(np.searchsorted(end_ns, signal_time.value, side="left"))
        if signal_pos >= len(market) or pd.Timestamp(end_times[signal_pos]) != signal_time:
            continue
        entry_pos = _causal_entry_position(
            start_ns,
            signal_pos=signal_pos,
            signal_time_ns=signal_time.value,
            entry_delay_bars=int(variant.entry_delay_bars),
        )
        if entry_pos >= len(market):
            continue
        entry_time = pd.Timestamp(start_times[entry_pos])
        entry_price = float(open_arr[entry_pos])
        side = int(event_s["side"])
        stop, target, raw_rr = _target_and_stop(event_s, entry_price, cfg)
        if not np.isfinite(stop) or not np.isfinite(target):
            continue

        exit_pos: int | None = None
        exit_price = np.nan
        exit_reason = "max_holding"
        both_hit = False
        decay_count = 0
        mfe = 0.0
        mae = 0.0
        max_end = entry_time + pd.Timedelta(minutes=int(cfg.max_holding_minutes))
        time_stop_at = entry_time + pd.Timedelta(minutes=int(cfg.time_stop_minutes))

        for pos in range(entry_pos, len(market)):
            if start_times[pos] > max_end:
                break
            if side == 1:
                hit_target = high_arr[pos] >= target
                hit_stop = low_arr[pos] <= stop
                favorable = high_arr[pos] / entry_price - 1.0
                adverse = low_arr[pos] / entry_price - 1.0
                flow_decay = buy_ratio[pos] < cfg.decay_flow_ratio
                obi_decay = np.isfinite(obi[pos]) and obi[pos] <= cfg.obi_neutral
                profitable_close = close_arr[pos] > entry_price
            else:
                hit_target = low_arr[pos] <= target
                hit_stop = high_arr[pos] >= stop
                favorable = 1.0 - low_arr[pos] / entry_price
                adverse = 1.0 - high_arr[pos] / entry_price
                flow_decay = sell_ratio[pos] < cfg.decay_flow_ratio
                obi_decay = np.isfinite(obi[pos]) and obi[pos] >= -cfg.obi_neutral
                profitable_close = close_arr[pos] < entry_price
            mfe = max(mfe, float(favorable))
            mae = min(mae, float(adverse))

            if hit_target and hit_stop:
                both_hit = True
                exit_pos = pos
                exit_price = stop
                exit_reason = "same_bar_both_stop_conservative"
                break
            if hit_stop:
                exit_pos = pos
                exit_price = stop
                exit_reason = "hard_stop"
                break
            if hit_target and variant.use_liquidity_target:
                exit_pos = pos
                exit_price = target
                exit_reason = "opposite_liquidity_or_fallback_target"
                break

            if variant.use_dynamic_decay_exit:
                decay_count = decay_count + 1 if (flow_decay or obi_decay) else 0
                if decay_count >= 2:
                    next_pos = pos + 1
                    if next_pos < len(market):
                        exit_pos = next_pos
                        exit_price = open_arr[next_pos]
                        exit_reason = "two_bar_flow_or_obi_decay_next_open"
                    else:
                        exit_pos = pos
                        exit_price = close_arr[pos]
                        exit_reason = "two_bar_flow_or_obi_decay_terminal"
                    break

            if variant.use_time_stop and end_times[pos] >= time_stop_at and not profitable_close:
                next_pos = pos + 1
                if next_pos < len(market):
                    exit_pos = next_pos
                    exit_price = open_arr[next_pos]
                    exit_reason = "time_stop_no_profit_next_open"
                else:
                    exit_pos = pos
                    exit_price = close_arr[pos]
                    exit_reason = "time_stop_no_profit_terminal"
                break

            if end_times[pos] >= max_end:
                exit_pos = pos
                exit_price = close_arr[pos]
                exit_reason = "max_holding"
                break

        if exit_pos is None:
            exit_pos = min(len(market) - 1, int(np.searchsorted(end_ns, max_end.value, side="left")))
            exit_price = close_arr[exit_pos]
            exit_reason = "max_holding_terminal"
        exit_price = _exit_fill(float(exit_price), side, exit_reason)
        exit_at_next_open = exit_reason.endswith("_next_open")
        exit_time = pd.Timestamp(start_times[exit_pos] if exit_at_next_open else end_times[exit_pos])
        gross = side * (exit_price / entry_price - 1.0)
        cost = float(cfg.round_trip_cost) * float(variant.cost_multiplier)
        net = gross - cost
        risk_return = abs(entry_price - stop) / entry_price
        r_multiple = net / risk_return if risk_return > 0 else np.nan
        rows.append(
            {
                "event_id": int(event_s["event_id"]),
                "range_tag": event_s["range_tag"],
                "mode": event_s["mode"],
                "stage": event_s["stage"],
                "side": side,
                "side_name": event_s["side_name"],
                "variant": variant.name,
                "signal_time": signal_time,
                "entry_time": entry_time,
                "entry_price": entry_price,
                "stop_price": stop,
                "target_price": target,
                "raw_rr": raw_rr,
                "exit_time": exit_time,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "gross_return": gross,
                "round_trip_cost": cost,
                "net_return": net,
                "risk_return": risk_return,
                "r_multiple": r_multiple,
                "mfe": mfe,
                "mae": mae,
                "same_bar_both_hit_flag": both_hit,
                "entry_not_after_signal_flag": entry_time <= signal_time,
                "book_available_after_signal_flag": bool(event_s.get("book_available_after_signal_flag", False)),
            }
        )
        if non_overlapping:
            blocked_until = exit_time
    return pd.DataFrame(rows)


def attach_forward_time_outcomes(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    horizons_minutes: Sequence[int] = (5, 15, 30, 60),
    entry_delay_bars: int = 1,
    round_trip_cost: float = 0.0011,
) -> pd.DataFrame:
    """Attach strictly-post-signal open forward close/MFE/MAE labels."""

    if events is None or events.empty:
        return pd.DataFrame()
    market = _ensure_range_frame(bars)
    end_times = pd.DatetimeIndex(market["end_ts"])
    start_times = pd.DatetimeIndex(market["start_ts"])
    end_ns = datetime_index_to_ns_int64(end_times)
    start_ns = datetime_index_to_ns_int64(start_times)
    open_arr = _numeric(market, "open").to_numpy(dtype=float)
    high_arr = _numeric(market, "high").to_numpy(dtype=float)
    low_arr = _numeric(market, "low").to_numpy(dtype=float)
    close_arr = _numeric(market, "close").to_numpy(dtype=float)
    out = events.copy().reset_index(drop=True)
    for horizon in horizons_minutes:
        h = int(horizon)
        gross_values = np.full(len(out), np.nan)
        mfe_values = np.full(len(out), np.nan)
        mae_values = np.full(len(out), np.nan)
        exit_times = np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]")
        for i, row in enumerate(out.itertuples(index=False)):
            signal_time = pd.Timestamp(row.signal_time)
            signal_pos = int(np.searchsorted(end_ns, signal_time.value, side="left"))
            if (
                signal_pos >= len(market)
                or pd.Timestamp(end_times[signal_pos]) != signal_time
            ):
                continue
            entry_pos = _causal_entry_position(
                start_ns,
                signal_pos=signal_pos,
                signal_time_ns=signal_time.value,
                entry_delay_bars=int(entry_delay_bars),
            )
            if entry_pos >= len(market):
                continue
            entry_price = open_arr[entry_pos]
            horizon_time = pd.Timestamp(start_times[entry_pos]) + pd.Timedelta(minutes=h)
            exit_pos = int(np.searchsorted(end_ns, horizon_time.value, side="left"))
            if exit_pos >= len(market):
                continue
            side = int(row.side)
            gross_values[i] = side * (close_arr[exit_pos] / entry_price - 1.0)
            path_high = np.nanmax(high_arr[entry_pos : exit_pos + 1])
            path_low = np.nanmin(low_arr[entry_pos : exit_pos + 1])
            if side == 1:
                mfe_values[i] = path_high / entry_price - 1.0
                mae_values[i] = path_low / entry_price - 1.0
            else:
                mfe_values[i] = 1.0 - path_low / entry_price
                mae_values[i] = 1.0 - path_high / entry_price
            exit_times[i] = end_times[exit_pos].to_datetime64()
        out[f"h{h}_gross_return"] = gross_values
        out[f"h{h}_net_return"] = gross_values - float(round_trip_cost)
        out[f"h{h}_mfe"] = mfe_values
        out[f"h{h}_mae"] = mae_values
        out[f"h{h}_exit_time"] = pd.to_datetime(exit_times)
    return out
