#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V10A Structural Stop Grid Research
==================================

Research-only grid for ETH LF Portfolio V10A structural stops / failure exits.

Purpose:
    - Do NOT promote the small Bear weak-footprint tweak to production.
    - Do NOT test Partial TP.
    - Do NOT test independent per-engine books.
    - Search many structural-stop / failure-exit variants that may improve win-rate
      while preserving the V10A high-convexity return profile.

No-lookahead rules:
    - V10A signal still comes from a completed 4H bar.
    - Entry still executes on the next 4H open.
    - Structural stop updates use only completed bars and become effective after the
      current closed bar, just like the official trailing/protected stop path.
    - Close-confirm structure failure exits execute on next 4H open.

This script depends only on the official V10A backtest and the integrated research
helpers included in this patch. It does not modify official strategy behavior.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest.lf import eth_lf_portfolio_v10a_momentum_micro_short_speed_filter_backtest as v10a  # noqa: E402
from research import v10a_integrated_signal_research_suite as suite  # noqa: E402

ENGINE_MOM = suite.ENGINE_MOM
ENGINE_BEAR = suite.ENGINE_BEAR
ENGINE_BULL = suite.ENGINE_BULL
BASELINE = "baseline_v10a"
OUT_NAME = "v10a_structural_stop_grid_research"


@dataclass(frozen=True)
class StructuralStopSpec:
    name: str
    enabled: bool = True
    source: str = "swing"  # signal_bar | swing | current_bar | rf_extreme | hybrid_tighter | hybrid_looser
    action: str = "stop"  # stop | close_confirm
    engine_scope: str = "ALL"  # ALL | BULL | BEAR | MOMENTUM | MOM_LONG | MOM_SHORT
    trigger_mfe_r: float = 0.0
    min_hold_bars: int = 0
    lookback: int = 5
    buffer_atr: float = 0.25
    require_giveback_frac: float = 0.0
    close_confirm_bars: int = 1
    initial_struct_stop: bool = False
    min_initial_atr_mult: float = 0.80
    max_initial_atr_mult: float = 3.50
    only_if_in_profit: bool = False
    tighten_only: bool = True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V10A structural stop / failure-exit grid research.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-15")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--warmup-days", type=int, default=365)
    p.add_argument("--initial-capital", type=float, default=1000.0)

    p.add_argument("--preset", choices=sorted(v10a.MOMENTUM_PRESETS), default="turbo")
    p.add_argument("--unit-risk-per-trade", type=float, default=None)
    p.add_argument("--max-total-notional-mult", type=float, default=None)
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--min-risk-mult", type=float, default=0.35)
    p.add_argument("--max-risk-mult", type=float, default=None)
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--disable-short", action="store_true")

    p.add_argument("--bear-preset", choices=sorted(v10a.BEAR_PRESETS), default="high")
    p.add_argument("--bear-min-risk-mult", type=float, default=0.25)
    p.add_argument("--bear-standalone-risk-scale", type=float, default=1.0)
    p.add_argument("--bear-standalone-quality-scale", type=float, default=1.0)
    p.add_argument("--disable-bear-standalone", action="store_true")

    p.add_argument("--bull-preset", choices=sorted(v10a.BULL_PRESETS), default="high")
    p.add_argument("--bull-min-risk-mult", type=float, default=0.25)
    p.add_argument("--bull-reclaim-risk-scale", type=float, default=1.0)
    p.add_argument("--bull-reclaim-quality-scale", type=float, default=1.0)
    p.add_argument("--bull-execution-mode", choices=["inherit", "own"], default="inherit")
    p.add_argument("--disable-bull-reclaim", action="store_true")

    p.add_argument("--priority-mode", choices=sorted(v10a.PRIORITY_MODES), default="reclaim_first")
    p.add_argument("--global-risk-scale", type=float, default=1.30)
    p.add_argument("--quality-mult-cap", type=float, default=2.20)

    p.add_argument("--micro-filter-mode", choices=["off", "soft", "strict"], default="soft")
    p.add_argument("--range-pct", type=float, default=0.002)
    p.add_argument("--price-step", type=float, default=1.0)
    p.add_argument("--range-data-dir", default=None)
    p.add_argument("--disable-footprint-context", action="store_true")
    p.add_argument("--micro-min-range-bars", type=int, default=5)
    p.add_argument("--micro-contra-imbalance", type=float, default=0.05)
    p.add_argument("--micro-aligned-imbalance", type=float, default=0.05)
    p.add_argument("--micro-bad-close-pos", type=float, default=0.35)
    p.add_argument("--micro-good-close-pos", type=float, default=0.65)
    p.add_argument("--micro-contra-risk-scale", type=float, default=0.50)
    p.add_argument("--micro-not-aligned-risk-scale", type=float, default=0.50)

    # Keep the official V9E/V10A range exit defaults; structural variants are layered research-only.
    p.add_argument("--range-exit-mode", choices=["off", "soft"], default="soft")
    p.add_argument("--range-exit-min-mfe-r", type=float, default=2.0)
    p.add_argument("--range-exit-giveback-frac", type=float, default=0.65)
    p.add_argument("--range-exit-min-hold-bars", type=int, default=2)
    p.add_argument("--range-exit-delay-bars", type=int, default=0)
    p.add_argument("--range-exit-contra-imbalance", type=float, default=0.05)
    p.add_argument("--range-exit-bad-close-pos", type=float, default=0.35)
    p.add_argument("--range-exit-no-reversal-required", dest="range_exit_require_reversal", action="store_false")
    p.set_defaults(range_exit_require_reversal=True)

    p.add_argument("--disable-momentum-long-not-aligned-block", action="store_true")
    p.add_argument("--disable-momentum-short-fast-speed-block", action="store_true")
    p.add_argument("--rf-speed-rolling-window-bars", type=int, default=1080)
    p.add_argument("--rf-speed-min-periods", type=int, default=100)
    p.add_argument("--rf-speed-fast-quantile", type=float, default=0.75)

    p.add_argument("--out-dir", default=f"data/reports/research/{OUT_NAME}")
    p.add_argument("--fast", action="store_true", help="Run a small but representative structural stop grid.")
    p.add_argument("--full", action="store_true", help="Run the full structural stop grid. Default if neither --fast nor --full.")
    p.add_argument("--max-variants", type=int, default=None)
    p.add_argument("--write-trades", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=25, help="Write checkpoint CSVs every N scenarios so a final reporting error does not lose a long run.")
    p.add_argument("--scoreboard-only", action="store_true", help="Rebuild compare/scoreboard/meta from existing output or checkpoint CSVs without rerunning scenarios.")
    return p.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "UNKNOWN"


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
        return entry_engine == ENGINE_BULL
    if scope == "BEAR":
        return entry_engine == ENGINE_BEAR
    if scope == "MOMENTUM":
        return entry_engine == ENGINE_MOM
    if scope == "MOM_LONG":
        return entry_engine == ENGINE_MOM and side == 1
    if scope == "MOM_SHORT":
        return entry_engine == ENGINE_MOM and side == -1
    if scope == "LONG":
        return side == 1
    if scope == "SHORT":
        return side == -1
    return False


def _fav_r(side: int, first_entry: float, max_fav: float, risk_per_coin: float) -> float:
    if risk_per_coin <= 0:
        return float("nan")
    return (max_fav - first_entry) / risk_per_coin if side == 1 else (first_entry - max_fav) / risk_per_coin


def _current_r(side: int, avg_entry: float, close: float, risk_per_coin: float) -> float:
    if risk_per_coin <= 0:
        return float("nan")
    return (close - avg_entry) / risk_per_coin if side == 1 else (avg_entry - close) / risk_per_coin


def _giveback_frac(peak_r: float, current_r: float) -> float:
    if not (math.isfinite(peak_r) and math.isfinite(current_r)) or abs(peak_r) < 1e-12:
        return float("nan")
    return max(0.0, (peak_r - current_r) / abs(peak_r))


def add_structural_columns(features: pd.DataFrame, max_lookback: int = 34) -> pd.DataFrame:
    out = features.copy()
    high = pd.to_numeric(out["high"], errors="coerce")
    low = pd.to_numeric(out["low"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")
    open_ = pd.to_numeric(out["open"], errors="coerce")
    atr = pd.to_numeric(out["atr"], errors="coerce")
    out["signal_bar_low"] = low
    out["signal_bar_high"] = high
    out["signal_bar_mid"] = (high + low) / 2.0
    span = (high - low).replace(0, np.nan)
    out["signal_close_pos"] = ((close - low) / span).clip(0.0, 1.0)
    out["signal_body_pct"] = ((close - open_).abs() / span).clip(0.0, 1.0)
    out["signal_upper_wick_pct"] = ((high - np.maximum(open_, close)) / span).clip(0.0, 1.0)
    out["signal_lower_wick_pct"] = ((np.minimum(open_, close) - low) / span).clip(0.0, 1.0)
    for n in sorted(set([3, 5, 8, 13, 21, 34, max_lookback])):
        if n <= 0:
            continue
        out[f"struct_low_{n}"] = low.rolling(n, min_periods=max(2, min(n, 3))).min()
        out[f"struct_high_{n}"] = high.rolling(n, min_periods=max(2, min(n, 3))).max()
        out[f"struct_mid_{n}"] = (out[f"struct_low_{n}"] + out[f"struct_high_{n}"]) / 2.0
        out[f"range_span_atr_{n}"] = (out[f"struct_high_{n}"] - out[f"struct_low_{n}"]) / atr.replace(0, np.nan)
    return out


def _row_value(row: Any, name: str) -> float:
    return _safe_float(getattr(row, name, float("nan")))


def _struct_level_from_row(
    row: Any,
    *,
    side: int,
    spec: StructuralStopSpec,
    signal_low: float,
    signal_high: float,
    atr_value: float,
) -> tuple[float, str]:
    buf = float(spec.buffer_atr) * float(atr_value)
    source = str(spec.source)

    if side == 1:
        signal_level = signal_low - buf
        current_level = _row_value(row, "low") - buf
        swing_level = _row_value(row, f"struct_low_{int(spec.lookback)}") - buf
        rf_level = _row_value(row, "rf_low") - buf
        if not math.isfinite(rf_level):
            rf_level = swing_level
        if source == "signal_bar":
            return signal_level, "SIGNAL_BAR_LOW"
        if source == "current_bar":
            return current_level, "CURRENT_BAR_LOW"
        if source == "rf_extreme":
            return rf_level, "RF_LOW"
        if source == "hybrid_tighter":
            return max(x for x in [signal_level, swing_level, rf_level] if math.isfinite(x)), "HYBRID_TIGHTER_LOW"
        if source == "hybrid_looser":
            return min(x for x in [signal_level, swing_level, rf_level] if math.isfinite(x)), "HYBRID_LOOSER_LOW"
        return swing_level, f"SWING_LOW_{int(spec.lookback)}"

    signal_level = signal_high + buf
    current_level = _row_value(row, "high") + buf
    swing_level = _row_value(row, f"struct_high_{int(spec.lookback)}") + buf
    rf_level = _row_value(row, "rf_high") + buf
    if not math.isfinite(rf_level):
        rf_level = swing_level
    if source == "signal_bar":
        return signal_level, "SIGNAL_BAR_HIGH"
    if source == "current_bar":
        return current_level, "CURRENT_BAR_HIGH"
    if source == "rf_extreme":
        return rf_level, "RF_HIGH"
    if source == "hybrid_tighter":
        return min(x for x in [signal_level, swing_level, rf_level] if math.isfinite(x)), "HYBRID_TIGHTER_HIGH"
    if source == "hybrid_looser":
        return max(x for x in [signal_level, swing_level, rf_level] if math.isfinite(x)), "HYBRID_LOOSER_HIGH"
    return swing_level, f"SWING_HIGH_{int(spec.lookback)}"


def _clamp_initial_stop(entry: float, side: int, proposed: float, atr_value: float, spec: StructuralStopSpec, fallback: float) -> float:
    if not math.isfinite(proposed) or not math.isfinite(entry) or not math.isfinite(atr_value) or atr_value <= 0:
        return fallback
    dist = abs(entry - proposed)
    min_dist = float(spec.min_initial_atr_mult) * atr_value
    max_dist = float(spec.max_initial_atr_mult) * atr_value
    dist = max(min_dist, min(max_dist, dist))
    return entry - dist if side == 1 else entry + dist


def run_structural_backtest(
    df: pd.DataFrame,
    cfg: Any,
    engine_cfgs: dict[str, Any] | None,
    *,
    global_risk_scale: float,
    args: argparse.Namespace,
    spec: StructuralStopSpec,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    if not spec.enabled:
        return v10a.run_priority_backtest(df, cfg, engine_cfgs=engine_cfgs, global_risk_scale=global_risk_scale, args=args)

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
    signal_low = float("nan")
    signal_high = float("nan")
    structure_updates = 0
    structure_confirm_count = 0

    rows = list(df.itertuples())
    idx = df.index

    for i in range(len(rows) - 1):
        row = rows[i]
        ts = idx[i]
        if in_pos:
            high = float(row.high)
            low = float(row.low)
            close = float(row.close)
            atr_value = float(row.atr)
            hold_bars = i - entry_i
            active_stop = stop_price
            current_signal = int(getattr(row, "signal", 0))
            active_cfg = pos_cfg

            if side == 1:
                max_fav = max(max_fav, high)
                max_adv = min(max_adv, low)
                touched_stop = low <= active_stop
                channel_exit = v10a._entry_exit_channel(row, entry_engine, side)
                opposite = current_signal == -1
                next_stop = max(stop_price, close - active_cfg.trailing_atr_mult * atr_value)
                locked = v10a.protected_stop(first_entry, avg_entry, side, risk_per_coin, max_fav, active_cfg)
                if locked is not None:
                    next_stop = max(next_stop, locked)
            else:
                max_fav = min(max_fav, low)
                max_adv = max(max_adv, high)
                touched_stop = high >= active_stop
                channel_exit = v10a._entry_exit_channel(row, entry_engine, side)
                opposite = current_signal == 1
                next_stop = min(stop_price, close + active_cfg.trailing_atr_mult * atr_value)
                locked = v10a.protected_stop(first_entry, avg_entry, side, risk_per_coin, max_fav, active_cfg)
                if locked is not None:
                    next_stop = min(next_stop, locked)

            peak_r = _fav_r(side, first_entry, max_fav, risk_per_coin)
            cur_r = _current_r(side, avg_entry, close, risk_per_coin)
            giveback = _giveback_frac(peak_r, cur_r)
            scoped = _engine_in_scope(entry_engine, side, spec.engine_scope)
            triggered = (
                scoped
                and hold_bars >= int(spec.min_hold_bars)
                and math.isfinite(peak_r)
                and peak_r >= float(spec.trigger_mfe_r)
                and (not spec.only_if_in_profit or cur_r > 0)
                and (float(spec.require_giveback_frac) <= 0 or (math.isfinite(giveback) and giveback >= float(spec.require_giveback_frac)))
            )

            struct_candidate = float("nan")
            struct_source = "NONE"
            if triggered:
                struct_candidate, struct_source = _struct_level_from_row(
                    row,
                    side=side,
                    spec=spec,
                    signal_low=signal_low,
                    signal_high=signal_high,
                    atr_value=atr_value,
                )
                if math.isfinite(struct_candidate):
                    if spec.action == "stop":
                        if side == 1:
                            improved = struct_candidate > next_stop and struct_candidate < close
                            if improved or not spec.tighten_only:
                                next_stop = max(next_stop, struct_candidate) if spec.tighten_only else struct_candidate
                                stop_source = f"STRUCT_{struct_source}"
                                structure_updates += 1
                        else:
                            improved = struct_candidate < next_stop and struct_candidate > close
                            if improved or not spec.tighten_only:
                                next_stop = min(next_stop, struct_candidate) if spec.tighten_only else struct_candidate
                                stop_source = f"STRUCT_{struct_source}"
                                structure_updates += 1
                    elif spec.action == "close_confirm":
                        failed = close < struct_candidate if side == 1 else close > struct_candidate
                        if failed:
                            structure_confirm_count += 1

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
            reason = ""
            exit_price = 0.0
            exit_time = ts
            if touched_stop:
                exit_now = True
                exit_price = v10a.apply_exit_slippage(active_stop, side, active_cfg.slippage_pct)
                reason = "STRUCTURE_STOP" if str(stop_source).startswith("STRUCT_") else "PROTECTED_TRAILING_STOP"
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
            elif spec.action == "close_confirm" and triggered and math.isfinite(struct_candidate):
                failed = close < struct_candidate if side == 1 else close > struct_candidate
                if failed:
                    exit_now = True
                    exit_price = v10a.apply_exit_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                    exit_time = idx[i + 1]
                    reason = "STRUCTURE_CLOSE_CONFIRM_NEXT_OPEN"
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
            elif hold_bars >= active_cfg.max_hold_bars:
                exit_now = True
                exit_price = v10a.apply_exit_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                exit_time = idx[i + 1]
                reason = "MAX_HOLD_EXIT_NEXT_OPEN"

            if pending_range_exit_i is not None and i >= pending_range_exit_i and not exit_now:
                exit_now = True
                exit_price = v10a.apply_exit_slippage(float(row.open), side, active_cfg.slippage_pct)
                exit_time = idx[i]
                reason = pending_range_exit_reason or "RANGE_EXIT_DELAYED_OPEN"

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
                    trades[-1]["research_exit_variant"] = spec.name
                    trades[-1]["structural_stop_source"] = stop_source
                    trades[-1]["structure_updates"] = int(structure_updates)
                    trades[-1]["structure_confirm_count"] = int(structure_confirm_count)
                    trades[-1]["structure_peak_r"] = float(peak_r) if math.isfinite(peak_r) else float("nan")
                    trades[-1]["structure_current_r"] = float(cur_r) if math.isfinite(cur_r) else float("nan")
                    trades[-1]["structure_giveback_frac"] = float(giveback) if math.isfinite(giveback) else float("nan")
                    if str(reason).startswith("RANGE_EXIT"):
                        trades[-1].update(range_exit_meta)
                peak = max(peak, capital)
                in_pos = False
                side = 0
                last_exit_i = i
                pending_range_exit_i = None
                structure_updates = 0
                structure_confirm_count = 0
            else:
                stop_price = next_stop

            if in_pos and units < active_cfg.max_units:
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
                        float(getattr(row, "risk_mult", entry_risk_mult)) * float(getattr(row, "quality_mult", 1.0)) * float(global_risk_scale),
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
                atr_sl = entry - entry_cfg.initial_atr_mult * atr_value if signal == 1 else entry + entry_cfg.initial_atr_mult * atr_value
                signal_low = float(row.low)
                signal_high = float(row.high)
                if spec.initial_struct_stop and _engine_in_scope(selected_engine, signal, spec.engine_scope):
                    proposed, src = _struct_level_from_row(
                        row,
                        side=signal,
                        spec=spec,
                        signal_low=signal_low,
                        signal_high=signal_high,
                        atr_value=atr_value,
                    )
                    sl = _clamp_initial_stop(entry, signal, proposed, atr_value, spec, atr_sl)
                    stop_source = f"STRUCT_INITIAL_{src}" if math.isfinite(proposed) else "ATR_INITIAL"
                else:
                    sl = atr_sl
                    stop_source = "ATR_INITIAL"
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
                    structure_confirm_count = 0

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
            trades[-1]["research_exit_variant"] = spec.name
            trades[-1]["structural_stop_source"] = stop_source
            trades[-1]["structure_updates"] = int(structure_updates)
            trades[-1]["structure_confirm_count"] = int(structure_confirm_count)
    equity = pd.DataFrame(equity_rows).set_index("time") if equity_rows else pd.DataFrame()
    return trades, equity


def build_structural_specs(fast: bool) -> list[StructuralStopSpec]:
    specs: list[StructuralStopSpec] = [StructuralStopSpec(name=BASELINE, enabled=False)]

    if fast:
        for scope in ["ALL", "BULL", "BEAR", "MOMENTUM"]:
            for src in ["signal_bar", "swing", "hybrid_tighter"]:
                for trig in [0.5, 1.0]:
                    specs.append(StructuralStopSpec(
                        name=f"struct_stop_{scope.lower()}_{src}_trig{str(trig).replace('.', 'p')}_n8_buf0p25",
                        source=src,
                        action="stop",
                        engine_scope=scope,
                        trigger_mfe_r=trig,
                        min_hold_bars=1,
                        lookback=8,
                        buffer_atr=0.25,
                    ))
        for scope in ["ALL", "BULL", "BEAR"]:
            for src in ["signal_bar", "swing"]:
                specs.append(StructuralStopSpec(
                    name=f"struct_fail_{scope.lower()}_{src}_hold2_n8_buf0p10",
                    source=src,
                    action="close_confirm",
                    engine_scope=scope,
                    trigger_mfe_r=0.0,
                    min_hold_bars=2,
                    lookback=8,
                    buffer_atr=0.10,
                ))
        for scope in ["BULL", "BEAR", "ALL"]:
            specs.append(StructuralStopSpec(
                name=f"initial_struct_{scope.lower()}_signal_bar_buf0p10",
                source="signal_bar",
                action="stop",
                engine_scope=scope,
                trigger_mfe_r=0.0,
                min_hold_bars=0,
                lookback=5,
                buffer_atr=0.10,
                initial_struct_stop=True,
            ))
        return _dedupe_specs(specs)

    scopes = ["ALL", "BULL", "BEAR", "MOMENTUM", "MOM_LONG", "MOM_SHORT"]
    stop_sources = ["signal_bar", "swing", "current_bar", "hybrid_tighter", "hybrid_looser"]
    lookbacks = [3, 5, 8, 13, 21]
    buffers = [0.0, 0.10, 0.25, 0.50]
    triggers = [0.0, 0.5, 1.0, 1.5]
    min_holds = [0, 1, 2, 3]

    # 1) Structural stop tightening grid. Not too huge, but enough to find broad zones.
    for scope in scopes:
        for src in stop_sources:
            lbs = [8] if src in {"signal_bar", "current_bar"} else lookbacks
            for n in lbs:
                for buf in buffers:
                    for trig in triggers:
                        hold = 1 if trig > 0 else 0
                        specs.append(StructuralStopSpec(
                            name=f"struct_stop_{scope.lower()}_{src}_n{n}_buf{_fmt(buf)}_trig{_fmt(trig)}_h{hold}",
                            source=src,
                            action="stop",
                            engine_scope=scope,
                            trigger_mfe_r=trig,
                            min_hold_bars=hold,
                            lookback=n,
                            buffer_atr=buf,
                        ))

    # 2) Failure exits on close back through structure. These target win-rate without fixed TP.
    for scope in ["ALL", "BULL", "BEAR", "MOMENTUM"]:
        for src in ["signal_bar", "swing", "hybrid_tighter"]:
            lbs = [5, 8, 13] if src != "signal_bar" else [8]
            for n in lbs:
                for buf in [0.0, 0.10, 0.25]:
                    for hold in [1, 2, 3, 4]:
                        specs.append(StructuralStopSpec(
                            name=f"struct_fail_{scope.lower()}_{src}_n{n}_buf{_fmt(buf)}_h{hold}",
                            source=src,
                            action="close_confirm",
                            engine_scope=scope,
                            trigger_mfe_r=0.0,
                            min_hold_bars=hold,
                            lookback=n,
                            buffer_atr=buf,
                        ))

    # 3) Giveback-sensitive structural stops: only tighten after the trade has had life then given back.
    for scope in ["ALL", "BULL", "BEAR", "MOMENTUM"]:
        for n in [5, 8, 13, 21]:
            for trig in [1.0, 1.5, 2.0]:
                for gb in [0.35, 0.50, 0.65]:
                    specs.append(StructuralStopSpec(
                        name=f"struct_giveback_{scope.lower()}_swing_n{n}_trig{_fmt(trig)}_gb{_fmt(gb)}",
                        source="swing",
                        action="stop",
                        engine_scope=scope,
                        trigger_mfe_r=trig,
                        min_hold_bars=2,
                        lookback=n,
                        buffer_atr=0.25,
                        require_giveback_frac=gb,
                    ))

    # 4) Initial structural stop. This changes initial R/size, so keep it separate in the scoreboard.
    for scope in ["ALL", "BULL", "BEAR", "MOMENTUM"]:
        for src in ["signal_bar", "swing", "hybrid_tighter"]:
            lbs = [5, 8, 13] if src != "signal_bar" else [8]
            for n in lbs:
                for buf in [0.0, 0.10, 0.25, 0.50]:
                    specs.append(StructuralStopSpec(
                        name=f"initial_struct_{scope.lower()}_{src}_n{n}_buf{_fmt(buf)}",
                        source=src,
                        action="stop",
                        engine_scope=scope,
                        trigger_mfe_r=0.0,
                        min_hold_bars=0,
                        lookback=n,
                        buffer_atr=buf,
                        initial_struct_stop=True,
                    ))

    return _dedupe_specs(specs)


def _fmt(x: float) -> str:
    return str(x).replace(".", "p").replace("-", "m")


def _dedupe_specs(specs: list[StructuralStopSpec]) -> list[StructuralStopSpec]:
    seen: set[str] = set()
    out: list[StructuralStopSpec] = []
    for s in specs:
        if s.name in seen:
            continue
        seen.add(s.name)
        out.append(s)
    return out


def compare_to_baseline(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    base = summary_df.loc[summary_df["scenario"].eq(BASELINE)]
    if base.empty:
        return summary_df.copy()
    b = base.iloc[0]
    out = summary_df.copy()
    for col in ["total_return_pct", "max_drawdown_pct", "win_rate", "profit_factor", "total_trades", "mfe_ge_1r_ended_loss", "mfe_ge_2r_ended_loss"]:
        if col in out.columns:
            out[f"{col}_baseline"] = b.get(col, np.nan)
            out[f"{col}_delta"] = pd.to_numeric(out[col], errors="coerce") - _safe_float(b.get(col), 0.0)
    out["return_ratio_vs_baseline"] = pd.to_numeric(out.get("total_return_pct", 0), errors="coerce") / max(_safe_float(b.get("total_return_pct"), 0.0), 1e-12)
    dd_base = max(_safe_float(b.get("max_drawdown_pct"), 0.0), 1e-12)
    out["drawdown_ratio_vs_baseline"] = pd.to_numeric(out.get("max_drawdown_pct", 0), errors="coerce") / dd_base
    return out


def _normalize_yearly_columns(yearly_df: pd.DataFrame) -> pd.DataFrame:
    """Accept both integrated-suite and structural-suite yearly column names.

    The shared suite.yearly_metrics() returns year_return_pct / win_rate.
    The first structural scoreboard version expected yearly_return_pct / yearly_win_rate,
    which caused a KeyError after long full-grid runs. Keep both aliases so old and
    future CSVs can be consumed safely.
    """
    if yearly_df is None or yearly_df.empty:
        return pd.DataFrame()
    out = yearly_df.copy()
    if "yearly_return_pct" not in out.columns and "year_return_pct" in out.columns:
        out["yearly_return_pct"] = out["year_return_pct"]
    if "year_return_pct" not in out.columns and "yearly_return_pct" in out.columns:
        out["year_return_pct"] = out["yearly_return_pct"]
    if "yearly_win_rate" not in out.columns and "win_rate" in out.columns:
        out["yearly_win_rate"] = out["win_rate"]
    if "win_rate" not in out.columns and "yearly_win_rate" in out.columns:
        out["win_rate"] = out["yearly_win_rate"]
    return out


def _read_existing_csv(out_dir: Path, *names: str) -> pd.DataFrame:
    for name in names:
        path = out_dir / name
        if path.exists():
            try:
                return pd.read_csv(path)
            except pd.errors.EmptyDataError:
                return pd.DataFrame()
    return pd.DataFrame()


def _write_checkpoint(
    out_dir: Path,
    summary_rows: list[dict[str, Any]],
    yearly_rows: list[pd.DataFrame],
    top_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
) -> None:
    """Persist enough state to rebuild scoreboards without rerunning all variants."""
    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(out_dir / "_checkpoint_02_structural_stop_grid_summary.csv", index=False)
    if yearly_rows:
        yearly_df = pd.concat([x for x in yearly_rows if x is not None and not x.empty], ignore_index=True) if yearly_rows else pd.DataFrame()
        yearly_df.to_csv(out_dir / "_checkpoint_07_variant_yearly.csv", index=False)
    if top_rows:
        pd.DataFrame(top_rows).to_csv(out_dir / "_checkpoint_08_top_trade_dependency.csv", index=False)
    if audit_rows:
        pd.DataFrame(audit_rows).to_csv(out_dir / "_checkpoint_05_structural_stop_audit.csv", index=False)


def _rebuild_outputs_from_existing(out_dir: Path, args: argparse.Namespace) -> int:
    summary_df = _read_existing_csv(out_dir, "02_structural_stop_grid_summary.csv", "_checkpoint_02_structural_stop_grid_summary.csv")
    yearly_df = _normalize_yearly_columns(_read_existing_csv(out_dir, "07_variant_yearly.csv", "_checkpoint_07_variant_yearly.csv"))
    top_df = _read_existing_csv(out_dir, "08_top_trade_dependency.csv", "_checkpoint_08_top_trade_dependency.csv")
    audit_df = _read_existing_csv(out_dir, "05_structural_stop_audit.csv", "_checkpoint_05_structural_stop_audit.csv")
    if summary_df.empty:
        raise RuntimeError(
            "No existing summary/checkpoint CSV found. Re-run without --scoreboard-only. "
            "Expected 02_structural_stop_grid_summary.csv or _checkpoint_02_structural_stop_grid_summary.csv."
        )
    compare_df = compare_to_baseline(summary_df)
    scoreboard_df = build_structural_scoreboard(compare_df, yearly_df, top_df)
    baseline_row = summary_df.loc[summary_df["scenario"].eq(BASELINE)] if "scenario" in summary_df.columns else pd.DataFrame()
    baseline_row.to_csv(out_dir / "01_baseline_summary.csv", index=False)
    summary_df.to_csv(out_dir / "02_structural_stop_grid_summary.csv", index=False)
    compare_df.to_csv(out_dir / "03_compare_to_v10a.csv", index=False)
    if not audit_df.empty:
        audit_df.to_csv(out_dir / "05_structural_stop_audit.csv", index=False)
    scoreboard_df.to_csv(out_dir / "06_candidate_scoreboard.csv", index=False)
    yearly_df.to_csv(out_dir / "07_variant_yearly.csv", index=False)
    top_df.to_csv(out_dir / "08_top_trade_dependency.csv", index=False)
    meta = {
        "script": "research/v10a_structural_stop_grid_research.py",
        "generated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "args": vars(args),
        "scenario_count": int(len(summary_df)),
        "scoreboard_only": True,
        "bugfix_notes": [
            "Rebuilt scoreboards with yearly column aliases: year_return_pct/yearly_return_pct and win_rate/yearly_win_rate.",
        ],
    }
    with (out_dir / "09_research_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)
    print(f"Rebuilt scoreboard from existing CSVs: {out_dir}", flush=True)
    return 0


def build_structural_scoreboard(compare_df: pd.DataFrame, yearly_df: pd.DataFrame, top_dep_df: pd.DataFrame) -> pd.DataFrame:
    if compare_df.empty:
        return pd.DataFrame()
    out = compare_df.copy()
    out = out.merge(top_dep_df, on="scenario", how="left") if not top_dep_df.empty else out
    yearly_df = _normalize_yearly_columns(yearly_df)
    if not yearly_df.empty and "scenario" in yearly_df.columns:
        agg_spec: dict[str, tuple[str, Any]] = {}
        if "yearly_return_pct" in yearly_df.columns:
            agg_spec["yearly_return_pct_min"] = ("yearly_return_pct", "min")
            agg_spec["yearly_return_pct_median"] = ("yearly_return_pct", "median")
        if "yearly_win_rate" in yearly_df.columns:
            agg_spec["yearly_win_rate_median"] = ("yearly_win_rate", "median")
        if agg_spec:
            y = yearly_df.groupby("scenario").agg(**agg_spec).reset_index()
            out = out.merge(y, on="scenario", how="left")
    ret_ratio = pd.to_numeric(out.get("return_ratio_vs_baseline", 0), errors="coerce").fillna(0)
    win_delta = pd.to_numeric(out.get("win_rate_delta", 0), errors="coerce").fillna(0)
    dd_delta = pd.to_numeric(out.get("max_drawdown_pct_delta", 0), errors="coerce").fillna(0)
    pf_delta = pd.to_numeric(out.get("profit_factor_delta", 0), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0)
    mfe1_delta = pd.to_numeric(out.get("mfe_ge_1r_ended_loss_delta", 0), errors="coerce").fillna(0)
    top3 = pd.to_numeric(out.get("top_3_trade_dependency_pct", 100), errors="coerce").fillna(100)
    yearly_min = pd.to_numeric(out.get("yearly_return_pct_min", 0), errors="coerce").fillna(0)

    # Screening, not fitting: prefer clear win-rate improvement without killing return.
    out["candidate_pass_basic"] = (
        ret_ratio.ge(0.85)
        & win_delta.ge(3.0)
        & dd_delta.le(5.0)
        & pf_delta.ge(-1.0)
    )
    out["candidate_pass_strict"] = (
        ret_ratio.ge(0.95)
        & win_delta.ge(5.0)
        & dd_delta.le(2.0)
        & pf_delta.ge(-0.25)
    )
    out["score"] = (
        ret_ratio.clip(0, 1.5) * 35.0
        + win_delta.clip(-20, 30) * 2.2
        - dd_delta.clip(-20, 20) * 1.2
        + pf_delta.clip(-5, 5) * 4.0
        - mfe1_delta.clip(-20, 20) * 1.5
        - (top3 / 100.0).clip(0, 1) * 8.0
        + (yearly_min.gt(0).astype(float) * 6.0)
    )
    sort_cols = ["candidate_pass_strict", "candidate_pass_basic", "score", "win_rate_delta", "return_ratio_vs_baseline"]
    return out.sort_values(sort_cols, ascending=[False, False, False, False, False])


def engine_exit_breakdown(all_trades: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario, tdf in all_trades.items():
        if tdf.empty:
            continue
        note = tdf.get("note", pd.Series("UNKNOWN", index=tdf.index)).astype(str)
        engine = tdf.get("engine", pd.Series("UNKNOWN", index=tdf.index)).astype(str)
        for (e, n), g in tdf.assign(_engine=engine, _note=note).groupby(["_engine", "_note"], dropna=False):
            ret = pd.to_numeric(g.get("return_pct", pd.Series(0, index=g.index)), errors="coerce").fillna(0) * 100.0
            rows.append({
                "scenario": scenario,
                "engine": e,
                "exit_reason": n,
                "trades": int(len(g)),
                "win_rate": float((ret > 0).mean() * 100.0) if len(g) else 0.0,
                "sum_return_pct": float(ret.sum()),
                "avg_return_pct": float(ret.mean()) if len(g) else 0.0,
            })
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    fast = bool(args.fast and not args.full)
    if not args.fast and not args.full:
        fast = False

    out_dir = Path(PROJECT_ROOT) / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.scoreboard_only:
        return _rebuild_outputs_from_existing(out_dir, args)

    print("=" * 96, flush=True)
    print("V10A Structural Stop Grid Research", flush=True)
    print("No Partial TP. No independent books. No formal strategy modification.", flush=True)
    print("=" * 96, flush=True)

    data = suite.load_inputs(args)
    flags = suite.build_flags(data["raw"], data["micro_ctx"], args)
    baseline_features = suite.make_features(
        data["raw"],
        data["micro_ctx"],
        args,
        flags,
        scenario=BASELINE,
        mom_long_block_mask=flags["v10_mom_long_not_aligned"],
        mom_short_block_mask=flags["v10a_mom_short_fast_speed"],
    )
    baseline_features = suite.slice_trade_window(baseline_features, args)
    baseline_features = add_structural_columns(baseline_features)

    specs = build_structural_specs(fast=fast)
    if args.max_variants is not None:
        baseline = [s for s in specs if s.name == BASELINE]
        others = [s for s in specs if s.name != BASELINE][: max(0, int(args.max_variants))]
        specs = baseline + others

    print(f"Running structural scenarios: {len(specs):,} | fast={fast}", flush=True)

    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[pd.DataFrame] = []
    top_rows: list[dict[str, Any]] = []
    all_trades: dict[str, pd.DataFrame] = {}
    audit_rows: list[dict[str, Any]] = []

    for n, spec in enumerate(specs, start=1):
        if spec.name == BASELINE:
            trades, equity = v10a.run_priority_backtest(
                baseline_features,
                data["exec_cfg"],
                engine_cfgs=data["engine_cfgs"],
                global_risk_scale=args.global_risk_scale,
                args=args,
            )
        else:
            trades, equity = run_structural_backtest(
                baseline_features,
                data["exec_cfg"],
                data["engine_cfgs"],
                global_risk_scale=args.global_risk_scale,
                args=args,
                spec=spec,
            )
        trades = v10a.attach_engine_to_trades(trades, baseline_features)
        extra = {
            "scenario_type": "structural_stop_grid",
            "struct_source": spec.source,
            "struct_action": spec.action,
            "struct_engine_scope": spec.engine_scope,
            "struct_trigger_mfe_r": spec.trigger_mfe_r,
            "struct_min_hold_bars": spec.min_hold_bars,
            "struct_lookback": spec.lookback,
            "struct_buffer_atr": spec.buffer_atr,
            "struct_initial_stop": spec.initial_struct_stop,
            "rule_note": json.dumps(asdict(spec), ensure_ascii=False),
        }
        sm = suite.summary_metrics(spec.name, trades, equity, data["exec_cfg"].initial_capital, extra=extra)
        summary_rows.append(sm)
        yearly_rows.append(suite.yearly_metrics(spec.name, trades, equity))
        top_rows.append(suite.top_trade_dependency(spec.name, trades))
        tdf = pd.DataFrame(trades)
        all_trades[spec.name] = tdf
        if not tdf.empty:
            note = tdf.get("note", pd.Series("", index=tdf.index)).astype(str)
            audit_rows.append({
                "scenario": spec.name,
                "structural_stop_exits": int(note.eq("STRUCTURE_STOP").sum()),
                "structural_close_confirm_exits": int(note.eq("STRUCTURE_CLOSE_CONFIRM_NEXT_OPEN").sum()),
                "structure_update_trades": int(pd.to_numeric(tdf.get("structure_updates", pd.Series(0, index=tdf.index)), errors="coerce").fillna(0).gt(0).sum()),
                "avg_structure_updates": float(pd.to_numeric(tdf.get("structure_updates", pd.Series(0, index=tdf.index)), errors="coerce").fillna(0).mean()),
                "return_pct_sum": float(pd.to_numeric(tdf.get("return_pct", pd.Series(0, index=tdf.index)), errors="coerce").fillna(0).sum() * 100.0),
            })
        else:
            audit_rows.append({"scenario": spec.name, "structural_stop_exits": 0, "structural_close_confirm_exits": 0, "structure_update_trades": 0, "avg_structure_updates": 0.0, "return_pct_sum": 0.0})
        if args.write_trades and (spec.name == BASELINE or n <= 10):
            tdf.to_csv(out_dir / f"{spec.name}__trades.csv", index=False)
            if not equity.empty:
                equity.to_csv(out_dir / f"{spec.name}__equity.csv")
        if n == 1 or n % 25 == 0 or n == len(specs):
            print(f"  completed {n:,}/{len(specs):,}: {spec.name}", flush=True)
        if args.checkpoint_every and (n == 1 or n % max(1, int(args.checkpoint_every)) == 0 or n == len(specs)):
            _write_checkpoint(out_dir, summary_rows, yearly_rows, top_rows, audit_rows)

    summary_df = pd.DataFrame(summary_rows)
    yearly_df = pd.concat([x for x in yearly_rows if x is not None and not x.empty], ignore_index=True) if yearly_rows else pd.DataFrame()
    yearly_df = _normalize_yearly_columns(yearly_df)
    top_df = pd.DataFrame(top_rows)
    compare_df = compare_to_baseline(summary_df)
    audit_df = pd.DataFrame(audit_rows)
    exit_df = engine_exit_breakdown(all_trades)

    # Write primary outputs before final scoreboard construction so a reporting bug never
    # discards a long completed grid run. The scoreboard can be rebuilt with --scoreboard-only.
    baseline_row = summary_df.loc[summary_df["scenario"].eq(BASELINE)]
    baseline_row.to_csv(out_dir / "01_baseline_summary.csv", index=False)
    summary_df.to_csv(out_dir / "02_structural_stop_grid_summary.csv", index=False)
    compare_df.to_csv(out_dir / "03_compare_to_v10a.csv", index=False)
    exit_df.to_csv(out_dir / "04_engine_exit_breakdown.csv", index=False)
    audit_df.to_csv(out_dir / "05_structural_stop_audit.csv", index=False)
    yearly_df.to_csv(out_dir / "07_variant_yearly.csv", index=False)
    top_df.to_csv(out_dir / "08_top_trade_dependency.csv", index=False)

    scoreboard_df = build_structural_scoreboard(compare_df, yearly_df, top_df)
    scoreboard_df.to_csv(out_dir / "06_candidate_scoreboard.csv", index=False)

    meta = {
        "script": "research/v10a_structural_stop_grid_research.py",
        "generated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "args": vars(args),
        "scenario_count": int(len(summary_df)),
        "fast": bool(fast),
        "no_lookahead_notes": [
            "Signal bar is closed before next-open entry.",
            "Structural stop updates use completed current bar and affect subsequent stop checks.",
            "Close-confirm structural failure exits execute next 4H open.",
            "No Partial TP and no independent per-engine book scenarios are included.",
        ],
        "outputs": [
            "01_baseline_summary.csv",
            "02_structural_stop_grid_summary.csv",
            "03_compare_to_v10a.csv",
            "04_engine_exit_breakdown.csv",
            "05_structural_stop_audit.csv",
            "06_candidate_scoreboard.csv",
            "07_variant_yearly.csv",
            "08_top_trade_dependency.csv",
        ],
    }
    with (out_dir / "09_research_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    print("\nTop structural candidates:", flush=True)
    show_cols = [
        "scenario", "candidate_pass_strict", "candidate_pass_basic", "score",
        "total_return_pct", "return_ratio_vs_baseline", "win_rate", "win_rate_delta",
        "max_drawdown_pct", "profit_factor", "mfe_ge_1r_ended_loss", "rule_note",
    ]
    existing = [c for c in show_cols if c in scoreboard_df.columns]
    if not scoreboard_df.empty:
        print(scoreboard_df[existing].head(20).to_string(index=False), flush=True)
    print(f"\nOutputs written to: {out_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
