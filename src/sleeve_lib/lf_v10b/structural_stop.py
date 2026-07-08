#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LF V10B all-engine swing structural stop executor."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.portfolio_common.allocator import (
    apply_entry_slippage,
    apply_exit_slippage,
    close_trade,
    protected_stop,
    unit_qty,
    weighted_avg_price,
)
from src.sleeve_lib.lf_v10b import selector


@dataclass(frozen=True)
class StructuralStopConfig:
    enabled: bool = True
    lookback_bars: int = 21
    buffer_atr: float = 0.0
    trigger_mfe_r: float = 0.0
    min_hold_bars: int = 0
    engine_scope: str = "ALL"
    source: str = "swing"
    tighten_only: bool = True


V10B_STRUCTURAL_STOP = StructuralStopConfig()


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _engine_in_scope(entry_engine: str, side: int, scope: str) -> bool:
    scope = str(scope).upper()
    if scope == "ALL":
        return True
    if scope == "BULL":
        return entry_engine == "BULL_RECLAIM_V2"
    if scope == "BEAR":
        return entry_engine == "BEAR_V3_ONLY"
    if scope == "MOMENTUM":
        return entry_engine == "MOMENTUM_V3"
    if scope == "MOM_LONG":
        return entry_engine == "MOMENTUM_V3" and side == 1
    if scope == "MOM_SHORT":
        return entry_engine == "MOMENTUM_V3" and side == -1
    if scope == "LONG":
        return side == 1
    if scope == "SHORT":
        return side == -1
    return False


def _fav_r(side: int, first_entry: float, max_fav: float, risk_per_coin: float) -> float:
    if risk_per_coin <= 0:
        return float("nan")
    return (max_fav - first_entry) / risk_per_coin if side == 1 else (first_entry - max_fav) / risk_per_coin


def add_structural_columns(features: pd.DataFrame, lookback_bars: int = 21) -> pd.DataFrame:
    out = features.copy()
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    out[f"struct_low_{lookback_bars}"] = low.rolling(lookback_bars, min_periods=lookback_bars).min()
    out[f"struct_high_{lookback_bars}"] = high.rolling(lookback_bars, min_periods=lookback_bars).max()
    return out


def _structural_stop_candidate(row: Any, side: int, atr_value: float, cfg: StructuralStopConfig) -> tuple[float, str]:
    buf = float(cfg.buffer_atr) * float(atr_value)
    n = int(cfg.lookback_bars)
    if side == 1:
        level = _safe_float(getattr(row, f"struct_low_{n}", float("nan"))) - buf
        return level, f"SWING_LOW_{n}"
    level = _safe_float(getattr(row, f"struct_high_{n}", float("nan"))) + buf
    return level, f"SWING_HIGH_{n}"


def _is_structural_source(source: Any) -> bool:
    return str(source or "").upper().startswith("STRUCT_")


def _stop_touch_reason(active_stop_source: Any) -> str:
    return "STRUCTURAL_STOP" if _is_structural_source(active_stop_source) else "PROTECTED_TRAILING_STOP"


def run_v10b_backtest(
    df: pd.DataFrame,
    cfg: Any,
    engine_cfgs: dict[str, Any] | None,
    *,
    global_risk_scale: float,
    args: Any,
    structural_cfg: StructuralStopConfig = V10B_STRUCTURAL_STOP,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    if not structural_cfg.enabled:
        return selector.run_priority_backtest(
            df,
            cfg,
            engine_cfgs=engine_cfgs,
            global_risk_scale=global_risk_scale,
            args=args,
        )

    capital = cfg.initial_capital
    peak = capital
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    in_pos = False
    side = 0
    entry_i = -1
    entry_time = None
    first_entry = 0.0
    avg_entry = 0.0
    initial_sl = 0.0
    stop_price = 0.0
    stop_source = "ATR_INITIAL"
    risk_per_coin = 0.0
    qty = 0.0
    total_entry_fee = 0.0
    units = 0
    max_fav = 0.0
    max_adv = 0.0
    entry_risk_mult = 1.0
    entry_engine = "NONE"
    pos_cfg = cfg
    engine_cfgs = engine_cfgs or {}
    last_exit_i = -10**9
    pending_range_exit_i: int | None = None
    pending_range_exit_reason = ""
    pending_range_exit_meta: dict[str, Any] = {}
    cap_at_entry = capital
    structure_updates = 0
    structural_stop_source = "NONE"

    rows = list(df.itertuples())
    idx = df.index

    for i in range(len(rows) - 1):
        row = rows[i]
        ts = idx[i]

        if in_pos:
            active_cfg = pos_cfg
            hold_bars = i - entry_i

            if pending_range_exit_i is not None and i >= pending_range_exit_i:
                exit_price = apply_exit_slippage(float(row.open), side, active_cfg.slippage_pct)
                exit_time = idx[i]
                reason = pending_range_exit_reason or "RANGE_EXIT_DELAYED_OPEN"
                active_stop_source_at_exit = stop_source
                capital = close_trade(
                    trades=trades,
                    capital=capital,
                    side=side,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    first_entry=first_entry,
                    avg_entry=avg_entry,
                    exit_price=exit_price,
                    initial_sl=initial_sl,
                    stop_price=stop_price,
                    qty=qty,
                    units=units,
                    total_entry_fee=total_entry_fee,
                    fee_rate=active_cfg.fee_rate,
                    max_fav=max_fav,
                    max_adv=max_adv,
                    risk_per_coin=risk_per_coin,
                    holding_bars=hold_bars,
                    reason=reason,
                    risk_mult=entry_risk_mult,
                )
                if trades:
                    trades[-1]["return_pct"] = float(trades[-1]["pnl"]) / max(float(cap_at_entry), 1e-12)
                    trades[-1]["structural_stop_enabled"] = True
                    trades[-1]["structural_stop_variant"] = "all_swing_n21_buf0p0_trig0p0_h0"
                    trades[-1]["structural_stop_source"] = structural_stop_source if structure_updates > 0 else "NONE"
                    trades[-1]["active_stop_source_at_exit"] = active_stop_source_at_exit
                    trades[-1]["structure_updates"] = int(structure_updates)
                    trades[-1].update(pending_range_exit_meta)
                    trades[-1]["range_exit_executed_after_delay"] = True
                peak = max(peak, capital)
                in_pos = False
                side = 0
                last_exit_i = i
                pending_range_exit_i = None
                pending_range_exit_reason = ""
                pending_range_exit_meta = {}
                structure_updates = 0
                structural_stop_source = "NONE"
            else:
                high = float(row.high)
                low = float(row.low)
                close = float(row.close)
                atr_value = float(row.atr)
                active_stop = stop_price
                active_stop_source = stop_source
                current_signal = int(getattr(row, "signal", 0))

                if side == 1:
                    max_fav = max(max_fav, high)
                    max_adv = min(max_adv, low)
                    touched_stop = low <= active_stop
                    channel_exit = selector._entry_exit_channel(row, entry_engine, side)
                    opposite = current_signal == -1
                    next_stop = stop_price
                    next_stop_source = stop_source
                    trailing_candidate = close - active_cfg.trailing_atr_mult * atr_value
                    if trailing_candidate > next_stop:
                        next_stop = trailing_candidate
                        next_stop_source = "TRAILING_ATR"
                    locked = protected_stop(first_entry, avg_entry, side, risk_per_coin, max_fav, active_cfg)
                    if locked is not None and locked > next_stop:
                        next_stop = locked
                        next_stop_source = "PROTECTED_TRAILING_STOP"
                else:
                    max_fav = min(max_fav, low)
                    max_adv = max(max_adv, high)
                    touched_stop = high >= active_stop
                    channel_exit = selector._entry_exit_channel(row, entry_engine, side)
                    opposite = current_signal == 1
                    next_stop = stop_price
                    next_stop_source = stop_source
                    trailing_candidate = close + active_cfg.trailing_atr_mult * atr_value
                    if trailing_candidate < next_stop:
                        next_stop = trailing_candidate
                        next_stop_source = "TRAILING_ATR"
                    locked = protected_stop(first_entry, avg_entry, side, risk_per_coin, max_fav, active_cfg)
                    if locked is not None and locked < next_stop:
                        next_stop = locked
                        next_stop_source = "PROTECTED_TRAILING_STOP"

                range_exit_now, range_exit_reason, range_exit_meta = selector._range_exit_signal(
                    row,
                    side=side,
                    avg_entry=avg_entry,
                    risk_per_coin=risk_per_coin,
                    max_fav=max_fav,
                    hold_bars=hold_bars,
                    args=args,
                )

                exit_now = False
                exit_price = 0.0
                exit_time = ts
                reason = ""
                active_stop_source_at_exit = active_stop_source
                if touched_stop:
                    exit_now = True
                    exit_price = apply_exit_slippage(active_stop, side, active_cfg.slippage_pct)
                    exit_time = ts
                    reason = _stop_touch_reason(active_stop_source)
                elif channel_exit:
                    exit_now = True
                    exit_price = apply_exit_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                    exit_time = idx[i + 1]
                    reason = "DONCHIAN_EXIT_NEXT_OPEN"
                elif opposite:
                    exit_now = True
                    exit_price = apply_exit_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                    exit_time = idx[i + 1]
                    reason = "OPPOSITE_BREAKOUT_NEXT_OPEN"
                elif range_exit_now:
                    delay_bars = int(getattr(args, "range_exit_delay_bars", 0) or 0)
                    if delay_bars <= 0:
                        exit_now = True
                        exit_price = apply_exit_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                        exit_time = idx[i + 1]
                        reason = range_exit_reason
                    else:
                        pending_range_exit_i = min(i + 1 + delay_bars, max(i + 1, len(rows) - 2))
                        pending_range_exit_reason = range_exit_reason
                        pending_range_exit_meta = dict(range_exit_meta)
                        pending_range_exit_meta["range_exit_signal_time"] = str(ts)
                        pending_range_exit_meta["range_exit_scheduled_exit_time"] = str(idx[pending_range_exit_i])
                        pending_range_exit_meta["range_exit_delay_bars"] = float(delay_bars)
                elif hold_bars >= active_cfg.max_hold_bars:
                    exit_now = True
                    exit_price = apply_exit_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                    exit_time = idx[i + 1]
                    reason = "MAX_HOLD_EXIT_NEXT_OPEN"

                if exit_now:
                    capital = close_trade(
                        trades=trades,
                        capital=capital,
                        side=side,
                        entry_time=entry_time,
                        exit_time=exit_time,
                        first_entry=first_entry,
                        avg_entry=avg_entry,
                        exit_price=exit_price,
                        initial_sl=initial_sl,
                        stop_price=stop_price,
                        qty=qty,
                        units=units,
                        total_entry_fee=total_entry_fee,
                        fee_rate=active_cfg.fee_rate,
                        max_fav=max_fav,
                        max_adv=max_adv,
                        risk_per_coin=risk_per_coin,
                        holding_bars=hold_bars,
                        reason=reason,
                        risk_mult=entry_risk_mult,
                    )
                    if trades:
                        trades[-1]["return_pct"] = float(trades[-1]["pnl"]) / max(float(cap_at_entry), 1e-12)
                        trades[-1]["structural_stop_enabled"] = True
                        trades[-1]["structural_stop_variant"] = "all_swing_n21_buf0p0_trig0p0_h0"
                        trades[-1]["structural_stop_source"] = structural_stop_source if structure_updates > 0 else "NONE"
                        trades[-1]["active_stop_source_at_exit"] = active_stop_source_at_exit
                        trades[-1]["structure_updates"] = int(structure_updates)
                        if str(reason).startswith("RANGE_EXIT"):
                            trades[-1].update(range_exit_meta or pending_range_exit_meta)
                    peak = max(peak, capital)
                    in_pos = False
                    side = 0
                    last_exit_i = i
                    pending_range_exit_i = None
                    pending_range_exit_reason = ""
                    pending_range_exit_meta = {}
                    structure_updates = 0
                    structural_stop_source = "NONE"
                else:
                    if pending_range_exit_i is None:
                        peak_r = _fav_r(side, first_entry, max_fav, risk_per_coin)
                        scoped = _engine_in_scope(entry_engine, side, structural_cfg.engine_scope)
                        if (
                            scoped
                            and hold_bars >= int(structural_cfg.min_hold_bars)
                            and math.isfinite(peak_r)
                            and peak_r >= float(structural_cfg.trigger_mfe_r)
                        ):
                            candidate, source = _structural_stop_candidate(row, side, atr_value, structural_cfg)
                            if math.isfinite(candidate):
                                if side == 1:
                                    improved = candidate > next_stop and candidate < close
                                    if improved or not structural_cfg.tighten_only:
                                        next_stop = max(next_stop, candidate) if structural_cfg.tighten_only else candidate
                                        next_stop_source = f"STRUCT_{source}"
                                        structural_stop_source = next_stop_source
                                        structure_updates += 1
                                else:
                                    improved = candidate < next_stop and candidate > close
                                    if improved or not structural_cfg.tighten_only:
                                        next_stop = min(next_stop, candidate) if structural_cfg.tighten_only else candidate
                                        next_stop_source = f"STRUCT_{source}"
                                        structural_stop_source = next_stop_source
                                        structure_updates += 1
                    stop_price = next_stop
                    stop_source = next_stop_source

                if in_pos and pending_range_exit_i is None and units < active_cfg.max_units:
                    next_unit_number = units + 1
                    trigger_r = (next_unit_number - 1) * active_cfg.add_every_r
                    add_triggered = high >= first_entry + trigger_r * risk_per_coin if side == 1 else low <= first_entry - trigger_r * risk_per_coin
                    if add_triggered:
                        add_price = apply_entry_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                        add_stop_dist = max(active_cfg.initial_atr_mult * atr_value, risk_per_coin)
                        add_q = unit_qty(
                            capital,
                            add_price,
                            add_stop_dist,
                            qty,
                            active_cfg,
                            float(getattr(row, "risk_mult", entry_risk_mult))
                            * float(getattr(row, "quality_mult", 1.0))
                            * float(global_risk_scale),
                        )
                        if add_q > 0 and math.isfinite(add_q):
                            total_entry_fee += add_q * add_price * active_cfg.fee_rate
                            avg_entry = weighted_avg_price(avg_entry, qty, add_price, add_q)
                            qty += add_q
                            units += 1

        if not in_pos and i - last_exit_i >= cfg.cooldown_bars:
            signal = int(getattr(row, "signal", 0))
            if signal != 0:
                selected_engine = str(getattr(row, "selected_engine", "UNKNOWN"))
                entry_cfg = engine_cfgs.get(selected_engine, cfg)
                next_open = float(rows[i + 1].open)
                entry = apply_entry_slippage(next_open, signal, entry_cfg.slippage_pct)
                atr_value = float(row.atr)
                sl = entry - entry_cfg.initial_atr_mult * atr_value if signal == 1 else entry + entry_cfg.initial_atr_mult * atr_value
                stop_dist = abs(entry - sl)
                entry_risk_mult = (
                    float(getattr(row, "risk_mult", 1.0))
                    * float(getattr(row, "quality_mult", 1.0))
                    * float(getattr(row, "micro_entry_risk_scale", 1.0))
                    * float(global_risk_scale)
                )
                q = unit_qty(capital, entry, stop_dist, 0.0, entry_cfg, entry_risk_mult)
                if q > 0 and math.isfinite(q):
                    in_pos = True
                    side = signal
                    entry_i = i + 1
                    entry_time = idx[i + 1]
                    first_entry = entry
                    avg_entry = entry
                    initial_sl = sl
                    stop_price = sl
                    stop_source = "ATR_INITIAL"
                    risk_per_coin = stop_dist
                    qty = q
                    total_entry_fee = qty * entry * entry_cfg.fee_rate
                    units = 1
                    max_fav = entry
                    max_adv = entry
                    entry_engine = selected_engine
                    pos_cfg = entry_cfg
                    cap_at_entry = capital
                    pending_range_exit_i = None
                    pending_range_exit_reason = ""
                    pending_range_exit_meta = {}
                    structure_updates = 0
                    structural_stop_source = "NONE"

        equity_rows.append({"time": ts, "capital": capital, "drawdown_pct": (peak - capital) / peak if peak > 0 else 0.0})

    if in_pos:
        ts = idx[-1]
        close = float(df.iloc[-1]["close"])
        exit_price = apply_exit_slippage(close, side, pos_cfg.slippage_pct)
        capital = close_trade(
            trades=trades,
            capital=capital,
            side=side,
            entry_time=entry_time,
            exit_time=ts,
            first_entry=first_entry,
            avg_entry=avg_entry,
            exit_price=exit_price,
            initial_sl=initial_sl,
            stop_price=stop_price,
            qty=qty,
            units=units,
            total_entry_fee=total_entry_fee,
            fee_rate=pos_cfg.fee_rate,
            max_fav=max_fav,
            max_adv=max_adv,
            risk_per_coin=risk_per_coin,
            holding_bars=len(df) - 1 - entry_i,
            reason="FORCE_CLOSE_END",
            risk_mult=entry_risk_mult,
        )
        if trades:
            trades[-1]["return_pct"] = float(trades[-1]["pnl"]) / max(float(cap_at_entry), 1e-12)
            trades[-1]["structural_stop_enabled"] = True
            trades[-1]["structural_stop_variant"] = "all_swing_n21_buf0p0_trig0p0_h0"
            trades[-1]["structural_stop_source"] = structural_stop_source if structure_updates > 0 else "NONE"
            trades[-1]["active_stop_source_at_exit"] = stop_source
            trades[-1]["structure_updates"] = int(structure_updates)
    equity = pd.DataFrame(equity_rows).set_index("time") if equity_rows else pd.DataFrame()
    return trades, equity

