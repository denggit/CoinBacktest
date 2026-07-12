#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETH LF Portfolio V10B All-Engine Swing Structural Stop
======================================================

V10B candidate based on V10A baseline plus one promoted research candidate:
    struct_stop_all_swing_n21_buf0p0_trig0p0_h0

Plain-English rule:
    After a position is open, on each completed 4H bar, compute the latest 21-bar
    swing structure level. For longs, the structural stop candidate is the
    lowest low of the latest 21 completed 4H bars. For shorts, it is the highest
    high of the latest 21 completed 4H bars. If that level tightens the existing
    stop without crossing the current close, move the next active stop to that
    structure level.

No-lookahead timing:
    - V10A signals still use completed 4H bars and execute at the next 4H open.
    - Intrabar stop checks always use the stop that was active before the current
      bar closed.
    - The V10B structural stop is calculated from the current completed 4H bar
      and earlier completed bars, then only becomes active after that bar.
    - No future highs/lows, no full-sample quantiles, no date filters.

Scope:
    - Research-to-backtest candidate only.
    - Does not modify V10A.
    - Does not modify AetherEdge/live trading.
    - Keeps single active portfolio position and V10A engine priority/routing.
"""
from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest.lf import eth_lf_portfolio_v10a_momentum_micro_short_speed_filter_backtest as v10a  # noqa: E402

STRATEGY_NAME = "eth_lf_portfolio_v10b_all_swing_structural_stop"
REPORT_STRATEGY_NAME = "ETH_LF_Portfolio_V10B_AllSwingStructuralStop"


@dataclass(frozen=True)
class StructuralStopConfig:
    """Fixed V10B structural stop candidate.

    This intentionally exposes the research-discovered values as constants in
    one place. The formal V10B candidate should not secretly optimize them at
    runtime. Parameter-neighbourhood testing belongs in research scripts.
    """

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
    """Add completed-bar swing structure columns used by V10B.

    The rolling high/low includes the current completed 4H bar. This is allowed
    because the structural stop update is only applied after the current bar has
    closed; stop touches inside the current bar use the prior active stop.
    """
    out = features.copy()
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    # Strict V10B semantics: do not emit a structural level until a full
    # lookback window is available. Earlier research used min_periods=3 for
    # exploration, but the promoted V10B candidate is explicitly n=21.
    out[f"struct_low_{lookback_bars}"] = low.rolling(
        lookback_bars,
        min_periods=lookback_bars,
    ).min()
    out[f"struct_high_{lookback_bars}"] = high.rolling(
        lookback_bars,
        min_periods=lookback_bars,
    ).max()
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
    """Keep V10A stop semantics while separating true structural-stop exits."""
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
    """V10A executor plus all-engine swing structural stop tightening.

    Timing invariant:
      1. Snapshot old active stop at bar start.
      2. Decide all current-bar exits using that old active stop and V10A exits.
      3. Only when no exit is taken, use the completed current bar to tighten the
         next active structural stop.
      4. The tightened stop can only affect following bars.
    """
    if not structural_cfg.enabled:
        return v10a.run_priority_backtest(
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

            # Keep V10A's non-default delayed range-exit semantics: if a delayed
            # range exit is due at this bar, exit immediately at this bar open
            # before reading the bar's completed high/low/close.
            if pending_range_exit_i is not None and i >= pending_range_exit_i:
                exit_price = v10a.apply_exit_slippage(float(row.open), side, active_cfg.slippage_pct)
                exit_time = idx[i]
                reason = pending_range_exit_reason or "RANGE_EXIT_DELAYED_OPEN"
                active_stop_source_at_exit = stop_source
                capital = v10a.close_trade(
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
                    channel_exit = v10a._entry_exit_channel(row, entry_engine, side)
                    opposite = current_signal == -1
                    next_stop = stop_price
                    next_stop_source = stop_source
                    trailing_candidate = close - active_cfg.trailing_atr_mult * atr_value
                    if trailing_candidate > next_stop:
                        next_stop = trailing_candidate
                        next_stop_source = "TRAILING_ATR"
                    locked = v10a.protected_stop(first_entry, avg_entry, side, risk_per_coin, max_fav, active_cfg)
                    if locked is not None and locked > next_stop:
                        next_stop = locked
                        next_stop_source = "PROTECTED_TRAILING_STOP"
                else:
                    max_fav = min(max_fav, low)
                    max_adv = max(max_adv, high)
                    touched_stop = high >= active_stop
                    channel_exit = v10a._entry_exit_channel(row, entry_engine, side)
                    opposite = current_signal == 1
                    next_stop = stop_price
                    next_stop_source = stop_source
                    trailing_candidate = close + active_cfg.trailing_atr_mult * atr_value
                    if trailing_candidate < next_stop:
                        next_stop = trailing_candidate
                        next_stop_source = "TRAILING_ATR"
                    locked = v10a.protected_stop(first_entry, avg_entry, side, risk_per_coin, max_fav, active_cfg)
                    if locked is not None and locked < next_stop:
                        next_stop = locked
                        next_stop_source = "PROTECTED_TRAILING_STOP"

                range_exit_now, range_exit_reason, range_exit_meta = v10a._range_exit_signal(
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
                    exit_price = v10a.apply_exit_slippage(active_stop, side, active_cfg.slippage_pct)
                    exit_time = ts
                    reason = _stop_touch_reason(active_stop_source)
                elif channel_exit:
                    exit_now = True
                    exit_price = v10a.apply_exit_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                    exit_time = idx[i + 1]
                    reason = "DONCHIAN_EXIT_NEXT_OPEN"
                elif opposite:
                    exit_now = True
                    exit_price = v10a.apply_exit_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                    exit_time = idx[i + 1]
                    reason = "OPPOSITE_BREAKOUT_NEXT_OPEN"
                elif range_exit_now:
                    delay_bars = int(getattr(args, "range_exit_delay_bars", 0) or 0)
                    if delay_bars <= 0:
                        exit_now = True
                        exit_price = v10a.apply_exit_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
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
                    exit_price = v10a.apply_exit_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                    exit_time = idx[i + 1]
                    reason = "MAX_HOLD_EXIT_NEXT_OPEN"

                if exit_now:
                    capital = v10a.close_trade(
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
                    # Only now, after all current-bar exits are confirmed false,
                    # compute and commit the completed-bar structural stop for
                    # future bars. Do not record structural updates on bars that
                    # already exited or that have just scheduled a delayed range exit.
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
                        add_price = v10a.apply_entry_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                        add_stop_dist = max(active_cfg.initial_atr_mult * atr_value, risk_per_coin)
                        add_q = v10a.unit_qty(
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
                            avg_entry = v10a.weighted_avg_price(avg_entry, qty, add_price, add_q)
                            qty += add_q
                            units += 1

        if not in_pos and i - last_exit_i >= cfg.cooldown_bars:
            signal = int(getattr(row, "signal", 0))
            if signal != 0:
                selected_engine = str(getattr(row, "selected_engine", "UNKNOWN"))
                entry_cfg = engine_cfgs.get(selected_engine, cfg)
                next_open = float(rows[i + 1].open)
                entry = v10a.apply_entry_slippage(next_open, signal, entry_cfg.slippage_pct)
                atr_value = float(row.atr)
                sl = entry - entry_cfg.initial_atr_mult * atr_value if signal == 1 else entry + entry_cfg.initial_atr_mult * atr_value
                stop_dist = abs(entry - sl)
                entry_risk_mult = (
                    float(getattr(row, "risk_mult", 1.0))
                    * float(getattr(row, "quality_mult", 1.0))
                    * float(getattr(row, "micro_entry_risk_scale", 1.0))
                    * float(global_risk_scale)
                )
                q = v10a.unit_qty(capital, entry, stop_dist, 0.0, entry_cfg, entry_risk_mult)
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
        exit_price = v10a.apply_exit_slippage(close, side, pos_cfg.slippage_pct)
        capital = v10a.close_trade(
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

def write_outputs(trades: list[dict[str, Any]], equity: pd.DataFrame, features: pd.DataFrame, summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(out_dir / f"{STRATEGY_NAME}_trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / f"{STRATEGY_NAME}_equity.csv")
    audit_cols = [
        "open", "high", "low", "close", "volume", "atr", "atr_pct", "adx",
        "risk_mult", "quality_mult", "momentum_signal", "bear_signal", "bull_signal", "signal",
        "momentum_long_not_aligned_blocked", "momentum_long_not_aligned_block_reason",
        "momentum_short_fast_speed_blocked", "momentum_short_fast_speed_block_reason",
        "selected_engine", "selected_priority", "momentum_selected", "bear_only", "bull_reclaim",
        "long_signal", "short_signal",
        "micro_context_available", "micro_aligned", "micro_contra", "micro_entry_risk_scale", "micro_filter_action",
        "rf_bar_count", "rf_micro_return_pct", "rf_close_pos", "rf_delta_sum", "rf_imbalance", "rf_taker_buy_ratio",
        "rf_max_sell_bucket_share", "rf_max_buy_bucket_share",
        "momentum_long_exit_channel", "momentum_short_exit_channel", "bear_short_exit_channel", "bull_long_exit_channel",
        "struct_low_21", "struct_high_21",
    ]
    features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / f"{STRATEGY_NAME}_signal_audit.csv")
    with (out_dir / f"{STRATEGY_NAME}_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)


def print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print("\n" + "=" * 92)
    print("ETH LF Portfolio V10B All-Engine Swing Structural Stop Backtest Summary")
    print("=" * 92)
    for k, v in summary.items():
        print(f"{k:>42}: {v}")
    print("-" * 92)
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 92 + "\n")


def print_deep_report(trades: list[dict[str, Any]], features: pd.DataFrame, cfg: Any, out_dir: Path) -> None:
    if not trades or features.empty:
        return
    total_days = max((features.index[-1] - features.index[0]).total_seconds() / 86400.0, 1e-9)
    v10a.print_full_report(
        trade_history=v10a.build_report_trades(trades),
        df=features,
        initial_capital=cfg.initial_capital,
        capital=float(pd.DataFrame(trades).iloc[-1]["capital"]),
        strategy_name=REPORT_STRATEGY_NAME,
        total_days=total_days,
        ai_enabled=False,
        symbol=cfg.symbol,
        report_dir=out_dir,
    )


def main() -> int:
    args = v10a.parse_args()
    mom_cfg = v10a.make_momentum_config(args)
    bear_cfg = v10a.make_bear_config(args)
    bull_cfg = v10a.make_bull_config(args)
    exec_cfg = v10a.make_exec_config(mom_cfg)
    bull_exec_cfg = v10a.bull_to_exec_config(bull_cfg) if args.bull_execution_mode == "own" else exec_cfg
    out_dir = Path(args.out_dir) if args.out_dir else Path(PROJECT_ROOT) / "data/reports/lf" / STRATEGY_NAME / args.preset

    trade_start = pd.Timestamp(args.start_date)
    if args.warmup_start_date:
        load_start = pd.Timestamp(args.warmup_start_date)
    elif args.warmup_days and args.warmup_days > 0:
        load_start = trade_start - pd.Timedelta(days=int(args.warmup_days))
    else:
        load_start = trade_start
    load_start_str = load_start.strftime("%Y-%m-%d")

    print(f"Loading {args.symbol} 4H for warmup: {load_start_str} -> {args.end_date}; trade_start={args.start_date}")
    base = v10a.load_data(args.symbol, load_start_str, args.end_date, "4H")
    print(f"Loaded {len(base)} rows: {base.index[0]} -> {base.index[-1]}")

    momentum = v10a.build_momentum_features(base, mom_cfg)
    bear = v10a.build_bear_features(base, bear_cfg)
    bull = v10a.build_bull_features(base, bull_cfg)
    micro_ctx = v10a.load_range_footprint_context(args, load_start_str, args.end_date)
    momentum = v10a.apply_momentum_long_not_aligned_block(momentum, micro_ctx, args)
    momentum = v10a.apply_momentum_short_fast_speed_block(momentum, micro_ctx, args)
    features = v10a.select_portfolio_signals(momentum, bear, bull, args)
    features = v10a.apply_micro_context_filter(features, micro_ctx, args)

    before_slice_rows = len(features)
    # Compute the structural columns on the warmup-inclusive feature frame first,
    # then slice to the trade period. This keeps the first tradeable bars from
    # accidentally using a shorter-than-21 structural window.
    features = add_structural_columns(features, lookback_bars=V10B_STRUCTURAL_STOP.lookback_bars)
    features = features.loc[trade_start: pd.Timestamp(args.end_date)].copy()
    print(
        f"Feature rows after warmup-inclusive structural columns + trade slice: {len(features)} / {before_slice_rows}; "
        f"first tradeable bar={features.index[0] if not features.empty else 'NA'}"
    )
    print("V10B structural stop:", V10B_STRUCTURAL_STOP)
    print("Signal counts:", {
        "momentum_long": int((features.momentum_signal == 1).sum()),
        "momentum_short": int((features.momentum_signal == -1).sum()),
        "bear_short": int((features.bear_signal == -1).sum()),
        "bull_long": int((features.bull_signal == 1).sum()),
        "portfolio_long": int((features.signal == 1).sum()),
        "portfolio_short": int((features.signal == -1).sum()),
        "bear_only": int(features.bear_only.sum()),
        "bull_reclaim": int(features.bull_reclaim.sum()),
        "portfolio_conflict": int(features.get("portfolio_conflict", pd.Series(False, index=features.index)).sum()),
        "momentum_long_not_aligned_blocked": int(features.get("momentum_long_not_aligned_blocked", pd.Series(False, index=features.index)).sum()),
        "momentum_short_fast_speed_blocked": int(features.get("momentum_short_fast_speed_blocked", pd.Series(False, index=features.index)).sum()),
        "priority_mode": args.priority_mode,
        "global_risk_scale": args.global_risk_scale,
    })

    trades, equity = run_v10b_backtest(
        features,
        exec_cfg,
        engine_cfgs={"MOMENTUM_V3": exec_cfg, "BEAR_V3_ONLY": exec_cfg, "BULL_RECLAIM_V2": bull_exec_cfg},
        global_risk_scale=args.global_risk_scale,
        args=args,
    )
    trades = v10a.attach_engine_to_trades(trades, features)
    summary = v10a.summarize(trades, equity, exec_cfg.initial_capital)

    if trades:
        tdf = pd.DataFrame(trades)
        summary["engine_counts"] = tdf["engine"].value_counts().to_dict()
        summary["momentum_trade_count"] = int(tdf.get("momentum_selected", pd.Series(dtype=bool)).sum())
        summary["bear_only_trade_count"] = int(tdf.get("bear_only", pd.Series(dtype=bool)).sum())
        summary["bull_reclaim_trade_count"] = int(tdf.get("engine", pd.Series(dtype=str)).eq("BULL_RECLAIM_V2").sum())
        note_col = tdf.get("note", pd.Series(dtype=str)).astype(str)
        summary["range_exit_trade_count"] = int(note_col.str.startswith("RANGE_EXIT").sum())
        summary["structural_stop_trade_count"] = int(pd.to_numeric(tdf.get("structure_updates", 0), errors="coerce").fillna(0).gt(0).sum())
        summary["structural_stop_total_updates"] = int(pd.to_numeric(tdf.get("structure_updates", 0), errors="coerce").fillna(0).sum())
        summary["structural_stop_exit_count"] = int(note_col.eq("STRUCTURAL_STOP").sum())
        summary["protected_trailing_stop_exit_count"] = int(note_col.eq("PROTECTED_TRAILING_STOP").sum())
    else:
        summary["range_exit_trade_count"] = 0
        summary["structural_stop_trade_count"] = 0
        summary["structural_stop_total_updates"] = 0
        summary["structural_stop_exit_count"] = 0
        summary["protected_trailing_stop_exit_count"] = 0

    summary["strategy_name"] = STRATEGY_NAME
    summary["base_strategy"] = "V10A Momentum Micro + Short Speed Filter"
    summary["candidate_origin"] = "struct_stop_all_swing_n21_buf0p0_trig0p0_h0"
    summary["structural_stop_enabled"] = True
    summary["structural_stop_scope"] = V10B_STRUCTURAL_STOP.engine_scope
    summary["structural_stop_source"] = V10B_STRUCTURAL_STOP.source
    summary["structural_stop_lookback_bars"] = V10B_STRUCTURAL_STOP.lookback_bars
    summary["structural_stop_buffer_atr"] = V10B_STRUCTURAL_STOP.buffer_atr
    summary["structural_stop_trigger_mfe_r"] = V10B_STRUCTURAL_STOP.trigger_mfe_r
    summary["structural_stop_min_hold_bars"] = V10B_STRUCTURAL_STOP.min_hold_bars
    summary["structural_stop_timing_note"] = "Strict V10B v2: full 21-bar warmup-inclusive structural windows; current bar exits are decided before structural update; completed current bar may tighten only future active stops."
    summary["preset"] = args.preset
    summary["bear_preset"] = args.bear_preset
    summary["bull_preset"] = args.bull_preset
    summary["bull_execution_mode"] = args.bull_execution_mode
    summary["priority_mode"] = args.priority_mode
    summary["priority_order"] = v10a.PRIORITY_MODES[args.priority_mode]
    summary["global_risk_scale"] = args.global_risk_scale
    summary["micro_filter_mode"] = args.micro_filter_mode
    summary["momentum_long_not_aligned_block_enabled"] = not bool(args.disable_momentum_long_not_aligned_block)
    summary["momentum_long_not_aligned_block_count"] = int(features.get("momentum_long_not_aligned_blocked", pd.Series(False, index=features.index)).sum())
    summary["momentum_short_fast_speed_block_enabled"] = not bool(args.disable_momentum_short_fast_speed_block)
    summary["momentum_short_fast_speed_block_count"] = int(features.get("momentum_short_fast_speed_blocked", pd.Series(False, index=features.index)).sum())
    summary["range_pct"] = args.range_pct
    summary["price_step"] = args.price_step
    summary["range_exit_mode"] = args.range_exit_mode
    summary["range_exit_min_mfe_r"] = args.range_exit_min_mfe_r
    summary["range_exit_giveback_frac"] = args.range_exit_giveback_frac
    summary["range_exit_min_hold_bars"] = args.range_exit_min_hold_bars
    summary["fee_rate_per_side"] = args.fee_rate
    summary["slippage_pct"] = args.slippage_pct
    summary["warmup_start_date"] = load_start_str
    summary["trade_start_date"] = args.start_date
    summary["warmup_days"] = int(args.warmup_days or 0)
    summary["single_active_position"] = True
    summary["no_lookahead_note"] = "Structural swing high/low uses a full completed 21-bar rolling window computed on warmup-inclusive features and affects future stop checks only."

    write_outputs(trades, equity, features, summary, out_dir)
    print_summary(summary, out_dir)
    print_deep_report(trades, features, exec_cfg, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
