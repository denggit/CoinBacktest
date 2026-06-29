#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V10A Integrated Signal Research Suite
=====================================

Research-only suite for ETH LF Portfolio V10A.

Goals:
    1) Analyze every independent signal, not only final portfolio summary.
    2) Compare V10A-safe candidate variants using percentage/R metrics.
    3) Keep Momentum Long in the research universe; never treat direct deletion as a candidate.
    4) Test whether independent per-engine books can improve the portfolio versus single-position priority routing.

Important boundaries:
    - Does NOT modify the official V10A strategy/backtest file.
    - Does NOT modify AetherEdge/live code.
    - Candidate rules are research-only and past-only where used for routing.
    - Future return/MFE/MAE labels are exported only for diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest.lf import eth_lf_portfolio_v10a_momentum_micro_short_speed_filter_backtest as v10a  # noqa: E402

ENGINE_MOM = "MOMENTUM_V3"
ENGINE_BEAR = "BEAR_V3_ONLY"
ENGINE_BULL = "BULL_RECLAIM_V2"
ENGINES = (ENGINE_MOM, ENGINE_BEAR, ENGINE_BULL)
BASELINE = "baseline_v10a"


# -----------------------------------------------------------------------------
# CLI / small helpers
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V10A integrated signal-level and portfolio research suite.")
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

    p.add_argument("--out-dir", default="data/reports/research/v10a_integrated_signal_research_suite")
    p.add_argument("--write-trades", action="store_true")
    p.add_argument("--fast", action="store_true", help="Run a smaller scenario set for quick iteration.")
    p.add_argument("--full", action="store_true", help="Run the full scenario set. Default if neither --fast nor --full is passed.")
    p.add_argument("--max-variants", type=int, default=None)
    return p.parse_args()


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(float(default), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(float(default))


def _num_nan(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _bool(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype("boolean").fillna(False).astype(bool)


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _safe_pf_from_return_pct(ret_pct: pd.Series) -> float:
    ret = pd.to_numeric(ret_pct, errors="coerce").fillna(0.0)
    gp = float(ret[ret > 0].sum())
    gl = float(-ret[ret <= 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else 0.0
    return gp / gl


def _ts_now() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "UNKNOWN"


# -----------------------------------------------------------------------------
# Load/reuse V10A data once
# -----------------------------------------------------------------------------

def load_inputs(args: argparse.Namespace) -> dict[str, Any]:
    mom_cfg = v10a.make_momentum_config(args)
    bear_cfg = v10a.make_bear_config(args)
    bull_cfg = v10a.make_bull_config(args)
    exec_cfg = v10a.make_exec_config(mom_cfg)
    bull_exec_cfg = v10a.bull_to_exec_config(bull_cfg) if args.bull_execution_mode == "own" else exec_cfg

    trade_start = pd.Timestamp(args.start_date)
    if args.warmup_start_date:
        load_start = pd.Timestamp(args.warmup_start_date)
    elif args.warmup_days and args.warmup_days > 0:
        load_start = trade_start - pd.Timedelta(days=int(args.warmup_days))
    else:
        load_start = trade_start
    load_start_str = load_start.strftime("%Y-%m-%d")

    print(f"[1/6] Loading 4H data once: {args.symbol} {load_start_str}->{args.end_date}; trade_start={args.start_date}", flush=True)
    base = v10a.load_data(args.symbol, load_start_str, args.end_date, "4H")
    if base.empty:
        raise RuntimeError("No 4H data loaded. Please check local CoinBacktest data DB/files.")
    print(f"      Loaded rows={len(base):,} {base.index[0]} -> {base.index[-1]}", flush=True)

    print("[2/6] Building raw engine features once...", flush=True)
    raw_mom = v10a.build_momentum_features(base, mom_cfg)
    raw_bear = v10a.build_bear_features(base, bear_cfg)
    raw_bull = v10a.build_bull_features(base, bull_cfg)

    print("[3/6] Loading range/footprint context once...", flush=True)
    micro_ctx = v10a.load_range_footprint_context(args, load_start_str, args.end_date)

    raw = {ENGINE_MOM: raw_mom, ENGINE_BEAR: raw_bear, ENGINE_BULL: raw_bull}
    engine_cfgs = {ENGINE_MOM: exec_cfg, ENGINE_BEAR: exec_cfg, ENGINE_BULL: bull_exec_cfg}

    return {
        "base": base,
        "raw": raw,
        "micro_ctx": micro_ctx,
        "exec_cfg": exec_cfg,
        "engine_cfgs": engine_cfgs,
        "trade_start": trade_start,
        "trade_end": pd.Timestamp(args.end_date),
        "load_start": load_start,
    }


# -----------------------------------------------------------------------------
# Past-only feature / flag construction
# -----------------------------------------------------------------------------

def _add_signal_bar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    high = _num_nan(out, "high")
    low = _num_nan(out, "low")
    open_ = _num_nan(out, "open")
    close = _num_nan(out, "close")
    span = (high - low).replace(0, np.nan)
    body = (close - open_).abs()
    out["candle_close_pos"] = ((close - low) / span).clip(0.0, 1.0)
    out["candle_body_pct"] = (body / span).clip(0.0, 1.0)
    out["upper_wick_pct"] = ((high - np.maximum(open_, close)) / span).clip(0.0, 1.0)
    out["lower_wick_pct"] = ((np.minimum(open_, close) - low) / span).clip(0.0, 1.0)
    out["candle_body_dir"] = np.where(close >= open_, "UP", "DOWN")

    vol = _num(out, "volume", 0.0)
    vol_base = vol.shift(1).rolling(180, min_periods=30).median().replace(0, np.nan)
    out["volume_ratio_past"] = vol / vol_base
    out["high_volume_past_q75"] = vol.ge(vol.shift(1).rolling(180, min_periods=30).quantile(0.75))
    out["volume_ratio_bin"] = pd.cut(
        out["volume_ratio_past"],
        bins=[-np.inf, 0.75, 1.25, 2.0, np.inf],
        labels=["LOW", "NORMAL", "HIGH", "EXTREME"],
    ).astype(str)

    prev_high_n = high.shift(1).rolling(6, min_periods=3).max()
    prev_low_n = low.shift(1).rolling(6, min_periods=3).min()
    out["break_prev_high_n"] = high.gt(prev_high_n)
    out["close_above_prev_high_n"] = close.gt(prev_high_n)
    out["failed_up_break_n"] = out["break_prev_high_n"] & (~out["close_above_prev_high_n"])
    out["break_prev_low_n"] = low.lt(prev_low_n)
    out["close_below_prev_low_n"] = close.lt(prev_low_n)
    out["failed_down_break_n"] = out["break_prev_low_n"] & (~out["close_below_prev_low_n"])
    return out


def _add_rf_speed_features(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy()
    if "rf_bar_count" not in out.columns:
        out["rf_speed_threshold_past"] = np.nan
        out["rf_bar_count_ratio_past"] = np.nan
        out["rf_speed_bin"] = "UNKNOWN"
        out["mom_short_fast_speed"] = False
        return out
    rf_count = pd.to_numeric(out["rf_bar_count"], errors="coerce")
    window = int(args.rf_speed_rolling_window_bars or 1080)
    min_periods = int(args.rf_speed_min_periods or 100)
    q = float(args.rf_speed_fast_quantile or 0.75)
    threshold = rf_count.shift(1).rolling(window, min_periods=min_periods).quantile(q)
    out["rf_speed_threshold_past"] = threshold
    out["rf_bar_count_ratio_past"] = rf_count / threshold.replace(0, np.nan)
    ratio = out["rf_bar_count_ratio_past"]
    out["rf_speed_bin"] = np.select(
        [ratio.isna(), ratio < 0.70, ratio < 1.00, ratio < 1.30, ratio >= 1.30],
        ["UNKNOWN", "SLOW", "NORMAL", "FAST_Q4", "EXTREME"],
        default="UNKNOWN",
    )
    return out


def build_flags(raw: dict[str, pd.DataFrame], micro_ctx: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    mom = raw[ENGINE_MOM]
    # Use a no-block selected frame only to carry market/micro columns over the common index.
    tmp_mom = mom.copy()
    tmp_selected = v10a.select_portfolio_signals(tmp_mom, raw[ENGINE_BEAR], raw[ENGINE_BULL], args)
    tmp_selected = v10a.apply_micro_context_filter(tmp_selected, micro_ctx, args)
    flags = _add_signal_bar_features(tmp_selected)
    flags = _add_rf_speed_features(flags, args)

    long_na = v10a.build_momentum_long_not_aligned_block_mask(mom, micro_ctx, args)
    short_fast = v10a.build_momentum_short_fast_speed_block_mask(mom, micro_ctx, args)
    flags["v10_mom_long_not_aligned"] = long_na.reindex(flags.index).fillna(False).astype(bool)
    flags["v10a_mom_short_fast_speed"] = short_fast.reindex(flags.index).fillna(False).astype(bool)

    flags["mom_signal_raw"] = _num(raw[ENGINE_MOM].reindex(flags.index), "signal", 0.0).astype(int)
    flags["bear_signal_raw"] = _num(raw[ENGINE_BEAR].reindex(flags.index), "signal", 0.0).astype(int)
    flags["bull_signal_raw"] = _num(raw[ENGINE_BULL].reindex(flags.index), "signal", 0.0).astype(int)

    flags["mom_long_raw"] = flags["mom_signal_raw"].eq(1)
    flags["mom_short_raw"] = flags["mom_signal_raw"].eq(-1)
    flags["bear_short_raw"] = flags["bear_signal_raw"].eq(-1)
    flags["bull_long_raw"] = flags["bull_signal_raw"].eq(1)

    # Long-side quality exceptions, all based on completed signal bar / past-only flags.
    flags["mom_long_exception_strong_bull"] = (
        flags["v10_mom_long_not_aligned"]
        & flags["candle_body_dir"].astype(str).eq("UP")
        & _num(flags, "candle_body_pct", 0.0).ge(0.45)
        & _num(flags, "candle_close_pos", 0.0).ge(0.65)
        & _num(flags, "upper_wick_pct", 1.0).le(0.45)
    )
    flags["mom_long_exception_high_volume"] = flags["v10_mom_long_not_aligned"] & _bool(flags, "high_volume_past_q75")
    flags["mom_long_exception_range_acceptance"] = (
        flags["v10_mom_long_not_aligned"]
        & (_num(flags, "rf_close_pos", 0.0).ge(0.65))
        & ((_num(flags, "rf_imbalance", 0.0).ge(0.0)) | (_num(flags, "rf_taker_buy_ratio", 0.0).ge(0.50)))
    )
    flags["mom_long_exception_broad_quality"] = (
        flags["mom_long_exception_strong_bull"]
        | flags["mom_long_exception_high_volume"]
        | flags["mom_long_exception_range_acceptance"]
    )

    # Short-side crash exceptions, all based on completed signal bar / past-only flags.
    flags["mom_short_exception_strong_breakdown"] = (
        flags["v10a_mom_short_fast_speed"]
        & flags["candle_body_dir"].astype(str).eq("DOWN")
        & _num(flags, "candle_body_pct", 0.0).ge(0.45)
        & _num(flags, "candle_close_pos", 1.0).le(0.35)
        & _num(flags, "lower_wick_pct", 1.0).le(0.35)
    )
    flags["mom_short_exception_sell_imbalance"] = (
        flags["v10a_mom_short_fast_speed"]
        & _num(flags, "rf_imbalance", 0.0).le(-0.05)
        & _num(flags, "rf_close_pos", 1.0).le(0.35)
    )
    flags["mom_short_exception_broad_crash"] = (
        flags["mom_short_exception_strong_breakdown"]
        | flags["mom_short_exception_sell_imbalance"]
        | _bool(flags, "close_below_prev_low_n")
    )

    flags["bear_weak_footprint"] = (
        flags["bear_short_raw"]
        & ((_num(flags, "rf_imbalance", -1.0).gt(-0.025)) | (_num(flags, "rf_close_pos", 0.0).gt(0.41)))
    )
    flags["bull_weak_reclaim"] = (
        flags["bull_long_raw"]
        & ((_num(flags, "rf_imbalance", 1.0).lt(-0.03)) | (_num(flags, "rf_close_pos", 1.0).lt(0.45)))
    )
    flags["bull_upper_wick"] = flags["bull_long_raw"] & _num(flags, "upper_wick_pct", 0.0).ge(0.55)
    return flags


# -----------------------------------------------------------------------------
# Scenario feature construction
# -----------------------------------------------------------------------------

def _copy_with_block(df: pd.DataFrame, mask: pd.Series, reason_col: str, reason: str) -> pd.DataFrame:
    out = df.copy()
    m = mask.reindex(out.index).fillna(False).astype(bool)
    for col in ["signal", "momentum_signal"]:
        if col in out.columns:
            out.loc[m, col] = 0
    for col in ["long_signal", "short_signal"]:
        if col in out.columns:
            out.loc[m, col] = False
    out[reason_col] = False
    out.loc[m, reason_col] = True
    out[f"{reason_col}_reason"] = "NONE"
    out.loc[m, f"{reason_col}_reason"] = reason
    return out


def make_features(
    raw: dict[str, pd.DataFrame],
    micro_ctx: pd.DataFrame,
    args: argparse.Namespace,
    flags: pd.DataFrame,
    *,
    scenario: str,
    mom_long_block_mask: pd.Series | None = None,
    mom_short_block_mask: pd.Series | None = None,
    risk_downs: list[tuple[pd.Series, str, float]] | None = None,
) -> pd.DataFrame:
    mom = raw[ENGINE_MOM].copy()
    mom["momentum_long_not_aligned_blocked"] = False
    mom["momentum_long_not_aligned_block_reason"] = "NONE"
    mom["momentum_short_fast_speed_blocked"] = False
    mom["momentum_short_fast_speed_block_reason"] = "NONE"
    if mom_long_block_mask is not None:
        m = mom_long_block_mask.reindex(mom.index).fillna(False).astype(bool)
        for col in ["signal", "momentum_signal"]:
            if col in mom.columns:
                mom.loc[m, col] = 0
        for col in ["long_signal", "short_signal"]:
            if col in mom.columns:
                mom.loc[m, col] = False
        mom.loc[m, "momentum_long_not_aligned_blocked"] = True
        mom.loc[m, "momentum_long_not_aligned_block_reason"] = "MOM_LONG_NOT_ALIGNED_BLOCKED_RESEARCH"
    if mom_short_block_mask is not None:
        m = mom_short_block_mask.reindex(mom.index).fillna(False).astype(bool)
        for col in ["signal", "momentum_signal"]:
            if col in mom.columns:
                mom.loc[m, col] = 0
        for col in ["long_signal", "short_signal"]:
            if col in mom.columns:
                mom.loc[m, col] = False
        mom.loc[m, "momentum_short_fast_speed_blocked"] = True
        mom.loc[m, "momentum_short_fast_speed_block_reason"] = "MOM_SHORT_FAST_SPEED_BLOCKED_RESEARCH"

    features = v10a.select_portfolio_signals(mom, raw[ENGINE_BEAR], raw[ENGINE_BULL], args)
    features = v10a.apply_micro_context_filter(features, micro_ctx, args)
    features = _add_signal_bar_features(features)
    features = _add_rf_speed_features(features, args)
    features["router_variant"] = scenario
    features["router_risk_adjustment"] = 1.0
    features["router_note"] = "BASELINE_RULES"

    if risk_downs:
        for mask, note, scale in risk_downs:
            m = mask.reindex(features.index).fillna(False).astype(bool)
            if bool(m.any()):
                features.loc[m, "risk_mult"] = (_num(features, "risk_mult", 1.0).loc[m] * float(scale)).clip(args.min_risk_mult, args.max_risk_mult or 10.0)
                features.loc[m, "router_risk_adjustment"] = _num(features, "router_risk_adjustment", 1.0).loc[m] * float(scale)
                features.loc[m, "router_note"] = note
    return features


def slice_trade_window(features: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = features.loc[pd.Timestamp(args.start_date): pd.Timestamp(args.end_date)].copy().sort_index()
    if out.empty:
        raise RuntimeError("No tradeable feature rows after warmup slicing.")
    return out


# -----------------------------------------------------------------------------
# Research executor: V10A-compatible plus optional partial TP / profit lock / no-progress exit
# -----------------------------------------------------------------------------

def _fav_r(side: int, first_entry: float, max_fav: float, risk_per_coin: float) -> float:
    if risk_per_coin <= 0:
        return float("nan")
    return (max_fav - first_entry) / risk_per_coin if side == 1 else (first_entry - max_fav) / risk_per_coin


def _current_r(side: int, avg_entry: float, close: float, risk_per_coin: float) -> float:
    if risk_per_coin <= 0:
        return float("nan")
    return (close - avg_entry) / risk_per_coin if side == 1 else (avg_entry - close) / risk_per_coin


def run_research_backtest(
    df: pd.DataFrame,
    cfg: Any,
    engine_cfgs: dict[str, Any] | None = None,
    *,
    global_risk_scale: float,
    args: argparse.Namespace,
    scenario: str,
    partial_tp_r: float | None = None,
    partial_tp_frac: float = 0.0,
    profit_lock_r: float | None = None,
    profit_lock_to_r: float = 0.0,
    no_progress_bars: int | None = None,
    no_progress_units: int = 1,
    no_progress_max_fav_r: float = 0.5,
    no_progress_current_r: float = 0.2,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    if not any([partial_tp_r is not None, profit_lock_r is not None, no_progress_bars is not None]):
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
    pending_partial_i: int | None = None
    partial_done = False
    partial_realized_pnl = 0.0
    partial_realized_fee = 0.0
    partial_exit_count = 0
    cap_at_entry = capital

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

            if pending_partial_i is not None and i >= pending_partial_i and not partial_done and qty > 0:
                frac = max(0.0, min(float(partial_tp_frac), 0.95))
                close_qty = qty * frac
                if close_qty > 0:
                    px = v10a.apply_exit_slippage(float(row.open), side, active_cfg.slippage_pct)
                    exit_fee = close_qty * px * active_cfg.fee_rate
                    entry_fee_part = total_entry_fee * frac
                    if side == 1:
                        pnl = (px - avg_entry) * close_qty - entry_fee_part - exit_fee
                    else:
                        pnl = (avg_entry - px) * close_qty - entry_fee_part - exit_fee
                    capital += pnl
                    peak = max(peak, capital)
                    qty -= close_qty
                    total_entry_fee -= entry_fee_part
                    partial_realized_pnl += pnl
                    partial_realized_fee += entry_fee_part + exit_fee
                    partial_exit_count += 1
                    partial_done = True
                pending_partial_i = None

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
            if profit_lock_r is not None and math.isfinite(peak_r) and peak_r >= float(profit_lock_r):
                lock_stop = first_entry + side * float(profit_lock_to_r) * risk_per_coin
                next_stop = max(next_stop, lock_stop) if side == 1 else min(next_stop, lock_stop)

            if partial_tp_r is not None and (not partial_done) and pending_partial_i is None and math.isfinite(peak_r) and peak_r >= float(partial_tp_r):
                pending_partial_i = i + 1

            range_exit_now, range_exit_reason, range_exit_meta = v10a._range_exit_signal(
                row,
                side=side,
                avg_entry=avg_entry,
                risk_per_coin=risk_per_coin,
                max_fav=max_fav,
                hold_bars=hold_bars,
                args=args,
            )

            no_progress_now = False
            if no_progress_bars is not None and hold_bars >= int(no_progress_bars):
                no_progress_now = (units <= int(no_progress_units)) and (peak_r < float(no_progress_max_fav_r)) and (cur_r < float(no_progress_current_r))

            exit_now = False
            reason = ""
            exit_price = 0.0
            exit_time = ts
            if touched_stop:
                exit_now = True
                exit_price = v10a.apply_exit_slippage(active_stop, side, active_cfg.slippage_pct)
                reason = "PROTECTED_TRAILING_STOP"
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
            elif no_progress_now:
                exit_now = True
                exit_price = v10a.apply_exit_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                exit_time = idx[i + 1]
                reason = "NO_PROGRESS_EXIT_NEXT_OPEN"
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
                    trades[-1]["pnl"] = float(trades[-1]["pnl"]) + float(partial_realized_pnl)
                    trades[-1]["fee"] = float(trades[-1]["fee"]) + float(partial_realized_fee)
                    trades[-1]["return_pct"] = float(trades[-1]["pnl"]) / max(float(cap_at_entry), 1e-12)
                    trades[-1]["partial_exit_count"] = int(partial_exit_count)
                    trades[-1]["partial_realized_pnl"] = float(partial_realized_pnl)
                    trades[-1]["partial_realized_fee"] = float(partial_realized_fee)
                    trades[-1]["research_exit_variant"] = scenario
                    if str(reason).startswith("RANGE_EXIT"):
                        trades[-1].update(range_exit_meta)
                    if reason == "NO_PROGRESS_EXIT_NEXT_OPEN":
                        trades[-1]["no_progress_peak_r"] = float(peak_r)
                        trades[-1]["no_progress_current_r"] = float(cur_r)
                peak = max(peak, capital)
                in_pos = False
                side = 0
                last_exit_i = i
                pending_range_exit_i = None
                pending_partial_i = None
                partial_done = False
                partial_realized_pnl = 0.0
                partial_realized_fee = 0.0
                partial_exit_count = 0
            else:
                stop_price = next_stop

            if in_pos and units < active_cfg.max_units:
                next_unit_number = units + 1
                trigger_r = (next_unit_number - 1) * active_cfg.add_every_r
                add_triggered = high >= first_entry + trigger_r * risk_per_coin if side == 1 else low <= first_entry - trigger_r * risk_per_coin
                if add_triggered:
                    add_price = v10a.apply_entry_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                    add_stop_dist = max(active_cfg.initial_atr_mult * atr_value, risk_per_coin)
                    add_q = v10a.unit_qty(capital, add_price, add_stop_dist, qty, active_cfg, float(getattr(row, "risk_mult", entry_risk_mult)) * float(getattr(row, "quality_mult", 1.0)) * float(global_risk_scale))
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
                    pending_partial_i = None
                    partial_done = False
                    partial_realized_pnl = 0.0
                    partial_realized_fee = 0.0
                    partial_exit_count = 0

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
            trades[-1]["pnl"] = float(trades[-1]["pnl"]) + float(partial_realized_pnl)
            trades[-1]["fee"] = float(trades[-1]["fee"]) + float(partial_realized_fee)
            trades[-1]["return_pct"] = float(trades[-1]["pnl"]) / max(float(cap_at_entry), 1e-12)
            trades[-1]["partial_exit_count"] = int(partial_exit_count)
            trades[-1]["partial_realized_pnl"] = float(partial_realized_pnl)
            trades[-1]["partial_realized_fee"] = float(partial_realized_fee)
            trades[-1]["research_exit_variant"] = scenario
    equity = pd.DataFrame(equity_rows).set_index("time") if equity_rows else pd.DataFrame()
    return trades, equity


# -----------------------------------------------------------------------------
# Independent engine books
# -----------------------------------------------------------------------------

def make_single_engine_features(
    engine: str,
    raw: dict[str, pd.DataFrame],
    micro_ctx: pd.DataFrame,
    flags: pd.DataFrame,
    args: argparse.Namespace,
    *,
    use_v10a_blocks: bool = True,
) -> pd.DataFrame:
    base = raw[ENGINE_MOM].copy()
    idx = base.index
    base["signal"] = 0
    base["selected_engine"] = "NONE"
    base["selected_priority"] = 0
    base["momentum_signal"] = _num(raw[ENGINE_MOM].reindex(idx), "signal", 0.0).astype(int)
    base["bear_signal"] = _num(raw[ENGINE_BEAR].reindex(idx), "signal", 0.0).astype(int)
    base["bull_signal"] = _num(raw[ENGINE_BULL].reindex(idx), "signal", 0.0).astype(int)
    base["momentum_long_exit_channel"] = _bool(raw[ENGINE_MOM].reindex(idx), "long_exit_channel")
    base["momentum_short_exit_channel"] = _bool(raw[ENGINE_MOM].reindex(idx), "short_exit_channel")
    base["bear_short_exit_channel"] = _bool(raw[ENGINE_BEAR].reindex(idx), "short_exit_channel")
    base["bull_long_exit_channel"] = _bool(raw[ENGINE_BULL].reindex(idx), "long_exit_channel")
    base["momentum_selected"] = False
    base["bear_only"] = False
    base["bull_reclaim"] = False
    base["portfolio_conflict"] = False

    if engine == ENGINE_MOM:
        sig = base["momentum_signal"].copy()
        if use_v10a_blocks:
            block = flags["v10_mom_long_not_aligned"].reindex(idx).fillna(False).astype(bool) | flags["v10a_mom_short_fast_speed"].reindex(idx).fillna(False).astype(bool)
            sig.loc[block] = 0
        base["signal"] = sig.astype(int)
        active = base["signal"].ne(0)
        base.loc[active, "selected_engine"] = ENGINE_MOM
        base.loc[active, "momentum_selected"] = True
    elif engine == ENGINE_BEAR:
        sig = base["bear_signal"].where(base["bear_signal"].eq(-1), 0)
        base["signal"] = sig.astype(int)
        active = base["signal"].ne(0)
        base.loc[active, "selected_engine"] = ENGINE_BEAR
        base.loc[active, "bear_only"] = True
        bear = raw[ENGINE_BEAR].reindex(idx)
        base.loc[active, "risk_mult"] = (_num(bear, "risk_mult", 1.0).loc[active] * float(args.bear_standalone_risk_scale)).clip(args.min_risk_mult, args.max_risk_mult or 10.0)
        base.loc[active, "quality_mult"] = (_num(bear, "quality_mult", 1.0).loc[active] * float(args.bear_standalone_quality_scale)).clip(0.20, args.quality_mult_cap)
    elif engine == ENGINE_BULL:
        sig = base["bull_signal"].where(base["bull_signal"].eq(1), 0)
        base["signal"] = sig.astype(int)
        active = base["signal"].ne(0)
        base.loc[active, "selected_engine"] = ENGINE_BULL
        base.loc[active, "bull_reclaim"] = True
        bull = raw[ENGINE_BULL].reindex(idx)
        base.loc[active, "risk_mult"] = (_num(bull, "risk_mult", 1.0).loc[active] * float(args.bull_reclaim_risk_scale)).clip(args.min_risk_mult, args.max_risk_mult or 10.0)
        base.loc[active, "quality_mult"] = (_num(bull, "quality_mult", 1.0).loc[active] * float(args.bull_reclaim_quality_scale)).clip(0.10, args.quality_mult_cap)
    else:
        raise ValueError(engine)

    base["long_signal"] = base["signal"].eq(1)
    base["short_signal"] = base["signal"].eq(-1)
    base = v10a.apply_micro_context_filter(base, micro_ctx, args)
    base = _add_signal_bar_features(base)
    base = _add_rf_speed_features(base, args)
    base["router_variant"] = f"independent_{engine}"
    return base


def combine_independent_books(runs: list[tuple[str, list[dict[str, Any]], pd.DataFrame]], initial_capital: float) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    all_trades: list[dict[str, Any]] = []
    eq_frames: list[pd.DataFrame] = []
    for engine, trades, equity in runs:
        for tr in trades:
            item = dict(tr)
            item["engine_book"] = engine
            item["engine"] = item.get("engine", engine)
            all_trades.append(item)
        if not equity.empty:
            sub = equity[["capital"]].copy()
            sub = sub.rename(columns={"capital": f"capital_{engine}"})
            eq_frames.append(sub)
    all_trades.sort(key=lambda x: pd.Timestamp(x.get("exit_time")))
    if not eq_frames:
        return all_trades, pd.DataFrame()
    combined = pd.concat(eq_frames, axis=1).sort_index().ffill().bfill()
    combined["capital"] = combined.sum(axis=1)
    combined["peak"] = combined["capital"].cummax()
    combined["drawdown_pct"] = (combined["peak"] - combined["capital"]) / combined["peak"].replace(0, np.nan)
    return all_trades, combined[["capital", "drawdown_pct"]]


# -----------------------------------------------------------------------------
# Metrics / diagnostics
# -----------------------------------------------------------------------------

def annotate_trades(trades: list[dict[str, Any]], features: pd.DataFrame) -> list[dict[str, Any]]:
    try:
        return v10a.attach_engine_to_trades(trades, features)
    except Exception:
        return trades


def summary_metrics(scenario: str, trades: list[dict[str, Any]], equity: pd.DataFrame, initial_capital: float, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"scenario": scenario}
    if equity.empty:
        final_cap = initial_capital
        max_dd = 0.0
    else:
        final_cap = float(pd.to_numeric(equity["capital"], errors="coerce").dropna().iloc[-1])
        max_dd = float(pd.to_numeric(equity.get("drawdown_pct", pd.Series(dtype=float)), errors="coerce").fillna(0.0).max() * 100.0)
    out.update({
        "final_capital": final_cap,
        "total_return_pct": (final_cap / max(initial_capital, 1e-12) - 1.0) * 100.0,
        "max_drawdown_pct": max_dd,
    })
    if not trades:
        out.update({
            "total_trades": 0, "long_trades": 0, "short_trades": 0, "win_rate": 0.0,
            "profit_factor": 0.0, "expectancy_pct": 0.0, "avg_return_pct": 0.0, "median_return_pct": 0.0,
            "avg_mfe_r": 0.0, "avg_mae_r": 0.0, "avg_units": 0.0,
            "mfe_ge_1r_ended_loss": 0, "mfe_ge_2r_ended_loss": 0,
            "force_close_end_count": 0,
        })
    else:
        tdf = pd.DataFrame(trades).copy()
        pnl = pd.to_numeric(tdf.get("pnl", pd.Series(0.0, index=tdf.index)), errors="coerce").fillna(0.0)
        ret = pd.to_numeric(tdf.get("return_pct", pd.Series(0.0, index=tdf.index)), errors="coerce").fillna(0.0) * 100.0
        mfe = pd.to_numeric(tdf.get("mfe_r", pd.Series(0.0, index=tdf.index)), errors="coerce").fillna(0.0)
        mae = pd.to_numeric(tdf.get("mae_r", pd.Series(0.0, index=tdf.index)), errors="coerce").fillna(0.0)
        wins = pnl > 0
        notes = tdf.get("note", pd.Series("", index=tdf.index)).astype(str)
        out.update({
            "total_trades": int(len(tdf)),
            "long_trades": int(tdf.get("type", pd.Series("", index=tdf.index)).astype(str).eq("LONG").sum()),
            "short_trades": int(tdf.get("type", pd.Series("", index=tdf.index)).astype(str).eq("SHORT").sum()),
            "win_rate": float(wins.mean() * 100.0),
            "profit_factor": _safe_pf_from_return_pct(ret),
            "expectancy_pct": float(ret.mean()),
            "avg_return_pct": float(ret.mean()),
            "median_return_pct": float(ret.median()),
            "avg_mfe_r": float(mfe.mean()),
            "avg_mae_r": float(mae.mean()),
            "avg_units": float(pd.to_numeric(tdf.get("units", pd.Series(0.0, index=tdf.index)), errors="coerce").fillna(0.0).mean()),
            "mfe_ge_1r_ended_loss": int(((mfe >= 1.0) & (~wins)).sum()),
            "mfe_ge_2r_ended_loss": int(((mfe >= 2.0) & (~wins)).sum()),
            "force_close_end_count": int(notes.eq("FORCE_CLOSE_END").sum()),
            "partial_exit_trade_count": int(pd.to_numeric(tdf.get("partial_exit_count", pd.Series(0, index=tdf.index)), errors="coerce").fillna(0).gt(0).sum()),
            "no_progress_exit_count": int(notes.eq("NO_PROGRESS_EXIT_NEXT_OPEN").sum()),
        })
        closed = tdf.loc[~notes.eq("FORCE_CLOSE_END")].copy()
        if not closed.empty:
            closed_ret = pd.to_numeric(closed.get("return_pct", pd.Series(0.0, index=closed.index)), errors="coerce").fillna(0.0) * 100.0
            closed_pnl = pd.to_numeric(closed.get("pnl", pd.Series(0.0, index=closed.index)), errors="coerce").fillna(0.0)
            out["no_force_close_trades"] = int(len(closed))
            out["no_force_close_win_rate"] = float((closed_pnl > 0).mean() * 100.0)
            out["no_force_close_pf_pct"] = _safe_pf_from_return_pct(closed_ret)
            out["no_force_close_expectancy_pct"] = float(closed_ret.mean())
        else:
            out["no_force_close_trades"] = 0
            out["no_force_close_win_rate"] = 0.0
            out["no_force_close_pf_pct"] = 0.0
            out["no_force_close_expectancy_pct"] = 0.0
    if extra:
        out.update(extra)
    return out


def yearly_metrics(scenario: str, trades: list[dict[str, Any]], equity: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if equity.empty:
        return pd.DataFrame()
    eq = equity.copy()
    eq.index = pd.to_datetime(eq.index)
    eq["year"] = eq.index.year
    tdf = pd.DataFrame(trades).copy() if trades else pd.DataFrame()
    if not tdf.empty:
        tdf["exit_time"] = pd.to_datetime(tdf["exit_time"])
        tdf["year"] = tdf["exit_time"].dt.year
    for year, g in eq.groupby("year"):
        start_cap = float(g["capital"].iloc[0])
        end_cap = float(g["capital"].iloc[-1])
        yt = tdf[tdf["year"].eq(year)] if not tdf.empty else pd.DataFrame()
        ret = pd.to_numeric(yt.get("return_pct", pd.Series(dtype=float)), errors="coerce").fillna(0.0) * 100.0 if len(yt) else pd.Series(dtype=float)
        pnl = pd.to_numeric(yt.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0) if len(yt) else pd.Series(dtype=float)
        rows.append({
            "scenario": scenario,
            "year": int(year),
            "year_return_pct": (end_cap / max(start_cap, 1e-12) - 1.0) * 100.0,
            "max_drawdown_pct": float(pd.to_numeric(g.get("drawdown_pct", pd.Series(0.0, index=g.index)), errors="coerce").fillna(0.0).max() * 100.0),
            "trades": int(len(yt)),
            "win_rate": float((pnl > 0).mean() * 100.0) if len(yt) else 0.0,
            "expectancy_pct": float(ret.mean()) if len(ret) else 0.0,
            "pf_pct": _safe_pf_from_return_pct(ret) if len(ret) else 0.0,
        })
    return pd.DataFrame(rows)


def top_trade_dependency(scenario: str, trades: list[dict[str, Any]]) -> dict[str, Any]:
    if not trades:
        return {"scenario": scenario, "top_1_trade_dependency_pct": 0.0, "top_3_trade_dependency_pct": 0.0, "remove_top1_sum_return_pct": 0.0, "remove_top3_sum_return_pct": 0.0}
    tdf = pd.DataFrame(trades)
    ret = pd.to_numeric(tdf.get("return_pct", pd.Series(0.0, index=tdf.index)), errors="coerce").fillna(0.0) * 100.0
    total_pos = float(ret[ret > 0].sum())
    sorted_win = ret[ret > 0].sort_values(ascending=False)
    top1 = float(sorted_win.head(1).sum())
    top3 = float(sorted_win.head(3).sum())
    return {
        "scenario": scenario,
        "top_1_trade_dependency_pct": top1 / total_pos * 100.0 if total_pos > 0 else 0.0,
        "top_3_trade_dependency_pct": top3 / total_pos * 100.0 if total_pos > 0 else 0.0,
        "remove_top1_sum_return_pct": float(ret.sum() - top1),
        "remove_top3_sum_return_pct": float(ret.sum() - top3),
    }


def build_signal_event_table(features: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    f = features.copy()
    common = flags.reindex(f.index)
    cols = [
        "mom_signal_raw", "bear_signal_raw", "bull_signal_raw", "v10_mom_long_not_aligned", "v10a_mom_short_fast_speed",
        "candle_close_pos", "candle_body_pct", "upper_wick_pct", "lower_wick_pct", "candle_body_dir", "volume_ratio_past",
        "volume_ratio_bin", "high_volume_past_q75", "rf_speed_bin", "rf_bar_count_ratio_past", "rf_speed_threshold_past",
        "break_prev_high_n", "close_above_prev_high_n", "failed_up_break_n", "break_prev_low_n", "close_below_prev_low_n", "failed_down_break_n",
        "mom_long_exception_strong_bull", "mom_long_exception_high_volume", "mom_long_exception_range_acceptance", "mom_long_exception_broad_quality",
        "mom_short_exception_strong_breakdown", "mom_short_exception_sell_imbalance", "mom_short_exception_broad_crash",
        "bear_weak_footprint", "bull_weak_reclaim", "bull_upper_wick",
    ]
    for col in cols:
        if col in common.columns:
            f[col] = common[col]
    event_mask = (
        _num(f, "mom_signal_raw", 0).ne(0)
        | _num(f, "bear_signal_raw", 0).ne(0)
        | _num(f, "bull_signal_raw", 0).ne(0)
        | _num(f, "signal", 0).ne(0)
    )
    out = f.loc[event_mask].copy()
    out["timestamp"] = out.index
    # Future diagnostic labels only. They are not used by any router rule.
    close = _num(f, "close", np.nan)
    high = _num(f, "high", np.nan)
    low = _num(f, "low", np.nan)
    side = _num(f, "signal", 0.0).astype(int)
    for n in [1, 3, 6, 12]:
        fut_close = close.shift(-n)
        fwd = (fut_close / close - 1.0) * 100.0
        out[f"forward_return_{n}bar_pct"] = (fwd * side).reindex(out.index)
    roll_high = high.shift(-1).rolling(12, min_periods=1).max().shift(-11)
    roll_low = low.shift(-1).rolling(12, min_periods=1).min().shift(-11)
    long_mfe = (roll_high / close - 1.0) * 100.0
    long_mae = (close / roll_low - 1.0) * 100.0
    short_mfe = (close / roll_low - 1.0) * 100.0
    short_mae = (roll_high / close - 1.0) * 100.0
    out["forward_mfe_12bar_pct"] = np.where(side.reindex(out.index).eq(-1), short_mfe.reindex(out.index), long_mfe.reindex(out.index))
    out["forward_mae_12bar_pct"] = np.where(side.reindex(out.index).eq(-1), short_mae.reindex(out.index), long_mae.reindex(out.index))
    keep = [
        "timestamp", "open", "high", "low", "close", "volume", "atr", "atr_pct", "adx",
        "signal", "selected_engine", "portfolio_conflict", "momentum_signal", "bear_signal", "bull_signal",
        "mom_signal_raw", "bear_signal_raw", "bull_signal_raw",
        "momentum_long_not_aligned_blocked", "momentum_short_fast_speed_blocked",
        "v10_mom_long_not_aligned", "v10a_mom_short_fast_speed",
        "micro_context_available", "micro_aligned", "micro_contra", "micro_filter_action", "micro_entry_risk_scale",
        "rf_bar_count", "rf_imbalance", "rf_close_pos", "rf_taker_buy_ratio", "rf_micro_return_pct", "rf_speed_bin",
        "candle_close_pos", "candle_body_pct", "upper_wick_pct", "lower_wick_pct", "candle_body_dir", "volume_ratio_past", "volume_ratio_bin",
        "mom_long_exception_strong_bull", "mom_long_exception_high_volume", "mom_long_exception_range_acceptance", "mom_long_exception_broad_quality",
        "mom_short_exception_strong_breakdown", "mom_short_exception_sell_imbalance", "mom_short_exception_broad_crash",
        "bear_weak_footprint", "bull_weak_reclaim", "bull_upper_wick",
        "forward_return_1bar_pct", "forward_return_3bar_pct", "forward_return_6bar_pct", "forward_return_12bar_pct",
        "forward_mfe_12bar_pct", "forward_mae_12bar_pct",
    ]
    return out[[c for c in keep if c in out.columns]].reset_index(drop=True)


def signal_level_diagnostics(trades: list[dict[str, Any]], features: pd.DataFrame, signal_events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trades:
        tdf = pd.DataFrame(trades).copy()
        groups = {
            "executed_all": pd.Series(True, index=tdf.index),
            "executed_momentum_long": tdf.get("engine", pd.Series("", index=tdf.index)).astype(str).eq(ENGINE_MOM) & tdf.get("type", pd.Series("", index=tdf.index)).astype(str).eq("LONG"),
            "executed_momentum_short": tdf.get("engine", pd.Series("", index=tdf.index)).astype(str).eq(ENGINE_MOM) & tdf.get("type", pd.Series("", index=tdf.index)).astype(str).eq("SHORT"),
            "executed_bull": tdf.get("engine", pd.Series("", index=tdf.index)).astype(str).eq(ENGINE_BULL),
            "executed_bear": tdf.get("engine", pd.Series("", index=tdf.index)).astype(str).eq(ENGINE_BEAR),
            "executed_micro_aligned": tdf.get("micro_aligned", pd.Series(False, index=tdf.index)).astype("boolean").fillna(False).astype(bool),
            "executed_micro_contra": tdf.get("micro_contra", pd.Series(False, index=tdf.index)).astype("boolean").fillna(False).astype(bool),
        }
        for name, mask in groups.items():
            sub = tdf.loc[mask].copy()
            if sub.empty:
                rows.append({"bucket": name, "kind": "trade", "count": 0})
                continue
            pnl = pd.to_numeric(sub.get("pnl", pd.Series(0.0, index=sub.index)), errors="coerce").fillna(0.0)
            ret = pd.to_numeric(sub.get("return_pct", pd.Series(0.0, index=sub.index)), errors="coerce").fillna(0.0) * 100.0
            mfe = pd.to_numeric(sub.get("mfe_r", pd.Series(0.0, index=sub.index)), errors="coerce").fillna(0.0)
            mae = pd.to_numeric(sub.get("mae_r", pd.Series(0.0, index=sub.index)), errors="coerce").fillna(0.0)
            rows.append({
                "bucket": name,
                "kind": "trade",
                "count": int(len(sub)),
                "win_rate": float((pnl > 0).mean() * 100.0),
                "sum_return_pct": float(ret.sum()),
                "avg_return_pct": float(ret.mean()),
                "median_return_pct": float(ret.median()),
                "pf_pct": _safe_pf_from_return_pct(ret),
                "avg_mfe_r": float(mfe.mean()),
                "avg_mae_r": float(mae.mean()),
                "mfe_ge_1r_ended_loss": int(((mfe >= 1.0) & (pnl <= 0)).sum()),
                "mfe_ge_2r_ended_loss": int(((mfe >= 2.0) & (pnl <= 0)).sum()),
            })
    if not signal_events.empty:
        se = signal_events.copy()
        event_groups = {
            "raw_momentum_long": _num(se, "mom_signal_raw", 0).eq(1),
            "raw_momentum_short": _num(se, "mom_signal_raw", 0).eq(-1),
            "raw_bull": _num(se, "bull_signal_raw", 0).eq(1),
            "raw_bear": _num(se, "bear_signal_raw", 0).eq(-1),
            "blocked_momentum_long_not_aligned": _bool(se, "v10_mom_long_not_aligned"),
            "blocked_momentum_short_fast_speed": _bool(se, "v10a_mom_short_fast_speed"),
            "portfolio_conflict": _bool(se, "portfolio_conflict"),
        }
        for name, mask in event_groups.items():
            sub = se.loc[mask].copy()
            fwd = pd.to_numeric(sub.get("forward_return_3bar_pct", pd.Series(dtype=float)), errors="coerce").dropna()
            rows.append({
                "bucket": name,
                "kind": "signal_event",
                "count": int(len(sub)),
                "avg_forward_return_3bar_pct": float(fwd.mean()) if len(fwd) else 0.0,
                "median_forward_return_3bar_pct": float(fwd.median()) if len(fwd) else 0.0,
                "positive_forward_rate_3bar": float((fwd > 0).mean() * 100.0) if len(fwd) else 0.0,
            })
    return pd.DataFrame(rows)


def blocked_signal_opportunity(signal_events: pd.DataFrame) -> pd.DataFrame:
    if signal_events.empty:
        return pd.DataFrame()
    rows = []
    for name, mask_col in [
        ("v10_blocked_momentum_long_not_aligned", "v10_mom_long_not_aligned"),
        ("v10a_blocked_momentum_short_fast_speed", "v10a_mom_short_fast_speed"),
    ]:
        if mask_col not in signal_events.columns:
            continue
        sub = signal_events.loc[_bool(signal_events, mask_col)].copy()
        for label_col in ["forward_return_1bar_pct", "forward_return_3bar_pct", "forward_return_6bar_pct", "forward_return_12bar_pct", "forward_mfe_12bar_pct", "forward_mae_12bar_pct"]:
            vals = pd.to_numeric(sub.get(label_col, pd.Series(dtype=float)), errors="coerce").dropna()
            rows.append({
                "blocked_bucket": name,
                "label": label_col,
                "count": int(len(vals)),
                "mean": float(vals.mean()) if len(vals) else 0.0,
                "median": float(vals.median()) if len(vals) else 0.0,
                "positive_rate": float((vals > 0).mean() * 100.0) if len(vals) else 0.0,
                "p25": float(vals.quantile(0.25)) if len(vals) else 0.0,
                "p75": float(vals.quantile(0.75)) if len(vals) else 0.0,
            })
    return pd.DataFrame(rows)


def conflict_signal_diagnostics(signal_events: pd.DataFrame) -> pd.DataFrame:
    if signal_events.empty or "portfolio_conflict" not in signal_events.columns:
        return pd.DataFrame()
    sub = signal_events.loc[_bool(signal_events, "portfolio_conflict")].copy()
    if sub.empty:
        return pd.DataFrame()
    sub["raw_combo"] = (
        np.where(_num(sub, "mom_signal_raw", 0).eq(1), "MOM_L", np.where(_num(sub, "mom_signal_raw", 0).eq(-1), "MOM_S", ""))
        + "|" + np.where(_num(sub, "bear_signal_raw", 0).eq(-1), "BEAR_S", "")
        + "|" + np.where(_num(sub, "bull_signal_raw", 0).eq(1), "BULL_L", "")
    )
    rows = []
    for combo, g in sub.groupby("raw_combo"):
        vals = pd.to_numeric(g.get("forward_return_3bar_pct", pd.Series(dtype=float)), errors="coerce").dropna()
        rows.append({
            "raw_combo": combo,
            "count": int(len(g)),
            "selected_engine_counts": json.dumps(g.get("selected_engine", pd.Series(dtype=str)).astype(str).value_counts().to_dict(), ensure_ascii=False),
            "avg_forward_return_3bar_pct": float(vals.mean()) if len(vals) else 0.0,
            "positive_forward_rate_3bar": float((vals > 0).mean() * 100.0) if len(vals) else 0.0,
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False)


# -----------------------------------------------------------------------------
# Scenario registry
# -----------------------------------------------------------------------------

def build_scenario_specs(flags: pd.DataFrame, args: argparse.Namespace, fast: bool) -> list[dict[str, Any]]:
    long_na = flags["v10_mom_long_not_aligned"].astype(bool)
    short_fast = flags["v10a_mom_short_fast_speed"].astype(bool)

    def sel_mom_long(f: pd.DataFrame) -> pd.Series:
        return f.get("selected_engine", pd.Series("", index=f.index)).astype(str).eq(ENGINE_MOM) & _num(f, "signal", 0).eq(1)

    def sel_mom_short(f: pd.DataFrame) -> pd.Series:
        return f.get("selected_engine", pd.Series("", index=f.index)).astype(str).eq(ENGINE_MOM) & _num(f, "signal", 0).eq(-1)

    def sel_bear(f: pd.DataFrame) -> pd.Series:
        return f.get("selected_engine", pd.Series("", index=f.index)).astype(str).eq(ENGINE_BEAR)

    def sel_bull(f: pd.DataFrame) -> pd.Series:
        return f.get("selected_engine", pd.Series("", index=f.index)).astype(str).eq(ENGINE_BULL)

    specs: list[dict[str, Any]] = [
        {"name": BASELINE, "long_block": long_na, "short_block": short_fast, "note": "Official V10A routing."},
    ]

    long_scales = [0.35] if fast else [0.25, 0.35, 0.50, 0.65]
    for scale in long_scales:
        specs.append({
            "name": f"mom_long_not_aligned_risk_down_{int(scale*100):02d}",
            "long_block": None,
            "short_block": short_fast,
            "risk_down_builder": lambda f, s=scale: [(sel_mom_long(f) & long_na.reindex(f.index).fillna(False).astype(bool), f"MOM_LONG_NOT_ALIGNED_RISK_DOWN_{s:.2f}", s)],
            "note": "Momentum Long NOT_ALIGNED is kept but risk-reduced; not deleted.",
        })
    for col, name in [
        ("mom_long_exception_strong_bull", "mom_long_not_aligned_allow_strong_bull"),
        ("mom_long_exception_high_volume", "mom_long_not_aligned_allow_high_volume"),
        ("mom_long_exception_range_acceptance", "mom_long_not_aligned_allow_range_acceptance"),
        ("mom_long_exception_broad_quality", "mom_long_not_aligned_allow_broad_quality"),
    ]:
        if fast and name not in {"mom_long_not_aligned_allow_broad_quality"}:
            continue
        allow = _bool(flags, col)
        specs.append({"name": name, "long_block": long_na & (~allow), "short_block": short_fast, "note": f"V10A but Momentum Long exception={col}."})

    short_scales = [0.35] if fast else [0.25, 0.35, 0.50]
    for scale in short_scales:
        specs.append({
            "name": f"mom_short_fast_speed_risk_down_{int(scale*100):02d}",
            "long_block": long_na,
            "short_block": None,
            "risk_down_builder": lambda f, s=scale: [(sel_mom_short(f) & short_fast.reindex(f.index).fillna(False).astype(bool), f"MOM_SHORT_FAST_SPEED_RISK_DOWN_{s:.2f}", s)],
            "note": "Momentum Short FAST_Q4 is kept but risk-reduced instead of blocked.",
        })
    for col, name in [
        ("mom_short_exception_strong_breakdown", "mom_short_fast_speed_allow_strong_breakdown"),
        ("mom_short_exception_sell_imbalance", "mom_short_fast_speed_allow_sell_imbalance"),
        ("mom_short_exception_broad_crash", "mom_short_fast_speed_allow_broad_crash"),
    ]:
        if fast and name not in {"mom_short_fast_speed_allow_broad_crash"}:
            continue
        allow = _bool(flags, col)
        specs.append({"name": name, "long_block": long_na, "short_block": short_fast & (~allow), "note": f"V10A but Momentum Short crash exception={col}."})

    bear_scales = [0.50] if fast else [0.50, 0.35]
    for scale in bear_scales:
        specs.append({
            "name": f"bear_weak_footprint_risk_down_{int(scale*100):02d}",
            "long_block": long_na,
            "short_block": short_fast,
            "risk_down_builder": lambda f, s=scale: [(sel_bear(f) & _bool(flags.reindex(f.index), "bear_weak_footprint"), f"BEAR_WEAK_FOOTPRINT_RISK_DOWN_{s:.2f}", s)],
            "note": "Bear weak footprint is risk-reduced only; no hard block.",
        })
    if not fast:
        specs.extend([
            {
                "name": "bull_weak_reclaim_risk_down_50",
                "long_block": long_na,
                "short_block": short_fast,
                "risk_down_builder": lambda f: [(sel_bull(f) & _bool(flags.reindex(f.index), "bull_weak_reclaim"), "BULL_WEAK_RECLAIM_RISK_DOWN_0.50", 0.50)],
                "note": "Bull weak reclaim risk-down.",
            },
            {
                "name": "bull_upper_wick_risk_down_50",
                "long_block": long_na,
                "short_block": short_fast,
                "risk_down_builder": lambda f: [(sel_bull(f) & _bool(flags.reindex(f.index), "bull_upper_wick"), "BULL_UPPER_WICK_RISK_DOWN_0.50", 0.50)],
                "note": "Bull upper-wick risk-down.",
            },
        ])

    # Exit-only specs use baseline features and research executor.
    exit_specs = [
        ("partial_tp_1r_15", {"partial_tp_r": 1.0, "partial_tp_frac": 0.15}),
        ("partial_tp_1r_20", {"partial_tp_r": 1.0, "partial_tp_frac": 0.20}),
        ("partial_tp_1p2r_15", {"partial_tp_r": 1.2, "partial_tp_frac": 0.15}),
        ("partial_tp_1p2r_20", {"partial_tp_r": 1.2, "partial_tp_frac": 0.20}),
        ("profit_lock_after_1r_to_minus_0p2r", {"profit_lock_r": 1.0, "profit_lock_to_r": -0.2}),
        ("profit_lock_after_1r_to_0r", {"profit_lock_r": 1.0, "profit_lock_to_r": 0.0}),
        ("no_progress_3bar_unit1", {"no_progress_bars": 3, "no_progress_units": 1, "no_progress_max_fav_r": 0.5, "no_progress_current_r": 0.2}),
        ("no_progress_4bar_unit2", {"no_progress_bars": 4, "no_progress_units": 2, "no_progress_max_fav_r": 0.75, "no_progress_current_r": 0.0}),
    ]
    for name, kwargs in exit_specs:
        if fast and name not in {"partial_tp_1p2r_20", "profit_lock_after_1r_to_0r", "no_progress_3bar_unit1"}:
            continue
        specs.append({"name": name, "long_block": long_na, "short_block": short_fast, "executor_kwargs": kwargs, "note": "Research-only exit/position-management variant."})

    specs.extend([
        {
            "name": "combo_mom_long_riskdown35_mom_short_crash_exception",
            "long_block": None,
            "short_block": short_fast & (~_bool(flags, "mom_short_exception_broad_crash")),
            "risk_down_builder": lambda f: [(sel_mom_long(f) & long_na.reindex(f.index).fillna(False).astype(bool), "MOM_LONG_NOT_ALIGNED_RISK_DOWN_0.35", 0.35)],
            "note": "Momentum Long risk-down + Momentum Short broad crash exception.",
        },
        {
            "name": "combo_partial1p2r20_no_progress3bar",
            "long_block": long_na,
            "short_block": short_fast,
            "executor_kwargs": {"partial_tp_r": 1.2, "partial_tp_frac": 0.20, "no_progress_bars": 3, "no_progress_units": 1, "no_progress_max_fav_r": 0.5, "no_progress_current_r": 0.2},
            "note": "Partial TP + no-progress exit.",
        },
        {
            "name": "combo_profitlock0r_no_progress3bar",
            "long_block": long_na,
            "short_block": short_fast,
            "executor_kwargs": {"profit_lock_r": 1.0, "profit_lock_to_r": 0.0, "no_progress_bars": 3, "no_progress_units": 1, "no_progress_max_fav_r": 0.5, "no_progress_current_r": 0.2},
            "note": "Profit lock + no-progress exit.",
        },
        {
            "name": "combo_mom_filters_plus_profitlock0r",
            "long_block": None,
            "short_block": short_fast & (~_bool(flags, "mom_short_exception_broad_crash")),
            "risk_down_builder": lambda f: [(sel_mom_long(f) & long_na.reindex(f.index).fillna(False).astype(bool), "MOM_LONG_NOT_ALIGNED_RISK_DOWN_0.35", 0.35)],
            "executor_kwargs": {"profit_lock_r": 1.0, "profit_lock_to_r": 0.0},
            "note": "Momentum filters + 1R breakeven lock.",
        },
    ])
    return specs


def build_scoreboard(summary_df: pd.DataFrame, yearly_df: pd.DataFrame, top_dep_df: pd.DataFrame, stress_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    base = summary_df.loc[summary_df["scenario"].eq(BASELINE)]
    if base.empty:
        base_row = summary_df.iloc[0]
    else:
        base_row = base.iloc[0]
    out = summary_df.copy()
    for col in ["total_return_pct", "max_drawdown_pct", "profit_factor", "win_rate", "mfe_ge_1r_ended_loss"]:
        if col not in out.columns:
            out[col] = 0.0
    out["return_vs_base_pct"] = out["total_return_pct"] - float(base_row.get("total_return_pct", 0.0))
    out["return_ratio_vs_base"] = out["total_return_pct"] / max(abs(float(base_row.get("total_return_pct", 0.0))), 1e-12)
    out["dd_delta_vs_base"] = out["max_drawdown_pct"] - float(base_row.get("max_drawdown_pct", 0.0))
    out["pf_ratio_vs_base"] = out["profit_factor"] / max(float(base_row.get("profit_factor", 0.0)), 1e-12)
    out["win_rate_delta_vs_base"] = out["win_rate"] - float(base_row.get("win_rate", 0.0))
    out["mfe_1r_loss_reduction"] = float(base_row.get("mfe_ge_1r_ended_loss", 0.0)) - out["mfe_ge_1r_ended_loss"]
    if not top_dep_df.empty:
        out = out.merge(top_dep_df, on="scenario", how="left")
    else:
        out["top_1_trade_dependency_pct"] = np.nan
        out["top_3_trade_dependency_pct"] = np.nan
    if not yearly_df.empty:
        y = yearly_df.groupby("scenario").agg(
            yearly_positive_rate=("year_return_pct", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean() * 100.0)),
            yearly_return_std=("year_return_pct", lambda s: float(pd.to_numeric(s, errors="coerce").std(ddof=0))),
            worst_year_return_pct=("year_return_pct", lambda s: float(pd.to_numeric(s, errors="coerce").min())),
        ).reset_index()
        out = out.merge(y, on="scenario", how="left")
    else:
        out["yearly_positive_rate"] = np.nan
        out["yearly_return_std"] = np.nan
        out["worst_year_return_pct"] = np.nan
    if not stress_df.empty:
        stress = stress_df.groupby("scenario").agg(
            worst_stress_return_pct=("total_return_pct", "min"),
            worst_stress_pf=("profit_factor", "min"),
            worst_stress_dd_pct=("max_drawdown_pct", "max"),
        ).reset_index()
        out = out.merge(stress, on="scenario", how="left")
    out["candidate_pass_basic"] = (
        (out["return_ratio_vs_base"] >= 0.80)
        & (out["dd_delta_vs_base"] <= 5.0)
        & (out["pf_ratio_vs_base"] >= 0.75)
    )
    # Conservative score: do not rank by raw return alone.
    out["score"] = (
        out["return_ratio_vs_base"].clip(-2, 2) * 25.0
        + out["win_rate_delta_vs_base"].clip(-20, 20) * 1.0
        + out["mfe_1r_loss_reduction"].clip(-20, 20) * 1.5
        - out["dd_delta_vs_base"].clip(-20, 20) * 1.5
        + out["pf_ratio_vs_base"].clip(0, 2) * 10.0
        + out.get("yearly_positive_rate", pd.Series(0.0, index=out.index)).fillna(0.0) * 0.05
        - out.get("top_1_trade_dependency_pct", pd.Series(0.0, index=out.index)).fillna(0.0) * 0.05
    )
    return out.sort_values(["candidate_pass_basic", "score"], ascending=[False, False])


# -----------------------------------------------------------------------------
# Main run
# -----------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    if not args.fast and not args.full:
        args.full = True
    out_dir = Path(PROJECT_ROOT) / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_inputs(args)
    raw = data["raw"]
    micro_ctx = data["micro_ctx"]
    exec_cfg = data["exec_cfg"]
    engine_cfgs = data["engine_cfgs"]

    print("[4/6] Building flags and V10A baseline features...", flush=True)
    flags = build_flags(raw, micro_ctx, args)
    specs = build_scenario_specs(flags, args, fast=args.fast and not args.full)
    if args.max_variants:
        specs = specs[: int(args.max_variants)]

    baseline_features_full = make_features(
        raw, micro_ctx, args, flags,
        scenario=BASELINE,
        mom_long_block_mask=flags["v10_mom_long_not_aligned"],
        mom_short_block_mask=flags["v10a_mom_short_fast_speed"],
    )
    baseline_features = slice_trade_window(baseline_features_full, args)
    signal_events = build_signal_event_table(baseline_features, flags)

    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[pd.DataFrame] = []
    top_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    stress_rows: list[dict[str, Any]] = []
    all_trades_for_optional: dict[str, pd.DataFrame] = {}

    print(f"[5/6] Running {len(specs)} single-position research scenarios...", flush=True)
    baseline_summary: dict[str, Any] | None = None
    scenario_cache: dict[str, tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]] = {}
    for n, spec in enumerate(specs, start=1):
        name = str(spec["name"])
        features_full = make_features(
            raw, micro_ctx, args, flags,
            scenario=name,
            mom_long_block_mask=spec.get("long_block"),
            mom_short_block_mask=spec.get("short_block"),
        )
        features = slice_trade_window(features_full, args)
        risk_down_builder = spec.get("risk_down_builder")
        if callable(risk_down_builder):
            risk_downs = risk_down_builder(features)
            # Re-apply risk-downs after final selection.
            for mask, note, scale in risk_downs:
                m = mask.reindex(features.index).fillna(False).astype(bool)
                if bool(m.any()):
                    features.loc[m, "risk_mult"] = (_num(features, "risk_mult", 1.0).loc[m] * float(scale)).clip(args.min_risk_mult, args.max_risk_mult or 10.0)
                    features.loc[m, "router_risk_adjustment"] = _num(features, "router_risk_adjustment", 1.0).loc[m] * float(scale)
                    features.loc[m, "router_note"] = note
        executor_kwargs = spec.get("executor_kwargs", {}) or {}
        trades, equity = run_research_backtest(
            features,
            exec_cfg,
            engine_cfgs=engine_cfgs,
            global_risk_scale=args.global_risk_scale,
            args=args,
            scenario=name,
            **executor_kwargs,
        )
        trades = annotate_trades(trades, features)
        extra = {"rule_note": str(spec.get("note", "")), "scenario_type": "single_position"}
        summary = summary_metrics(name, trades, equity, exec_cfg.initial_capital, extra=extra)
        if name == BASELINE:
            baseline_summary = dict(summary)
        summary_rows.append(summary)
        yearly_rows.append(yearly_metrics(name, trades, equity))
        top_rows.append(top_trade_dependency(name, trades))
        scenario_cache[name] = (features, trades, equity)
        if args.write_trades:
            all_trades_for_optional[name] = pd.DataFrame(trades)
            if not equity.empty:
                equity.to_csv(out_dir / f"{name}__equity.csv")
        if n % 5 == 0 or n == len(specs):
            print(f"      completed {n}/{len(specs)} scenarios", flush=True)

    # Independent engine books: research-only, not live-ready by default.
    print("[5b/6] Running independent per-engine books research scenarios...", flush=True)
    for scenario_name, grs in [
        ("independent_engines_equal_capital_v10a_blocks", args.global_risk_scale),
        ("independent_engines_equal_capital_global_risk_0p70", 0.70),
    ]:
        runs: list[tuple[str, list[dict[str, Any]], pd.DataFrame]] = []
        for engine in ENGINES:
            eng_features_full = make_single_engine_features(engine, raw, micro_ctx, flags, args, use_v10a_blocks=True)
            eng_features = slice_trade_window(eng_features_full, args)
            sub_init = exec_cfg.initial_capital / float(len(ENGINES))
            sub_cfg = replace(exec_cfg, initial_capital=sub_init)
            sub_engine_cfgs = {k: replace(v, initial_capital=sub_init) for k, v in engine_cfgs.items()}
            trades, equity = run_research_backtest(
                eng_features,
                sub_cfg,
                engine_cfgs=sub_engine_cfgs,
                global_risk_scale=grs,
                args=args,
                scenario=scenario_name,
            )
            trades = annotate_trades(trades, eng_features)
            runs.append((engine, trades, equity))
        combined_trades, combined_equity = combine_independent_books(runs, exec_cfg.initial_capital)
        summary_rows.append(summary_metrics(
            scenario_name,
            combined_trades,
            combined_equity,
            exec_cfg.initial_capital,
            extra={
                "rule_note": "Research-only: each engine has its own isolated sub-book and capital allocation. This allows overlapping/hedged exposures and is not the current AetherEdge live model.",
                "scenario_type": "independent_engine_books",
                "independent_engine_books": True,
                "independent_engine_global_risk_scale": float(grs),
            },
        ))
        yearly_rows.append(yearly_metrics(scenario_name, combined_trades, combined_equity))
        top_rows.append(top_trade_dependency(scenario_name, combined_trades))
        if args.write_trades:
            all_trades_for_optional[scenario_name] = pd.DataFrame(combined_trades)
            if not combined_equity.empty:
                combined_equity.to_csv(out_dir / f"{scenario_name}__equity.csv")

    summary_df = pd.DataFrame(summary_rows)
    yearly_df = pd.concat([x for x in yearly_rows if not x.empty], ignore_index=True) if yearly_rows else pd.DataFrame()
    top_df = pd.DataFrame(top_rows)

    if baseline_summary is None:
        baseline_summary = summary_rows[0] if summary_rows else {}
    for row in summary_rows:
        compare = dict(row)
        compare["baseline_total_return_pct"] = baseline_summary.get("total_return_pct", 0.0)
        compare["baseline_max_drawdown_pct"] = baseline_summary.get("max_drawdown_pct", 0.0)
        compare["baseline_win_rate"] = baseline_summary.get("win_rate", 0.0)
        compare["baseline_profit_factor"] = baseline_summary.get("profit_factor", 0.0)
        compare["return_delta_pct"] = row.get("total_return_pct", 0.0) - baseline_summary.get("total_return_pct", 0.0)
        compare["drawdown_delta_pct"] = row.get("max_drawdown_pct", 0.0) - baseline_summary.get("max_drawdown_pct", 0.0)
        compare["win_rate_delta"] = row.get("win_rate", 0.0) - baseline_summary.get("win_rate", 0.0)
        compare_rows.append(compare)
    compare_df = pd.DataFrame(compare_rows)

    # Lightweight stress: recompute baseline and top candidate under higher fee/slippage without reloading data.
    print("[5c/6] Running lightweight fee/slippage stress for baseline candidates...", flush=True)
    stress_names = [BASELINE]
    if len(compare_df) > 0:
        # Add up to 5 promising non-independent candidates by score proxy.
        tmp = compare_df.loc[~compare_df["scenario"].astype(str).str.startswith("independent_")].copy()
        tmp["stress_pick_score"] = tmp.get("win_rate_delta", 0.0) + tmp.get("return_delta_pct", 0.0) / 100.0 - tmp.get("drawdown_delta_pct", 0.0)
        stress_names += [x for x in tmp.sort_values("stress_pick_score", ascending=False)["scenario"].head(5).tolist() if x not in stress_names]
    stress_specs = {str(s["name"]): s for s in specs}
    for stress_name in stress_names:
        spec = stress_specs.get(stress_name)
        if spec is None:
            continue
        for fee, slip in [(0.00075, args.slippage_pct), (0.00100, args.slippage_pct), (args.fee_rate, 0.00050), (args.fee_rate, 0.00100)]:
            sargs = argparse.Namespace(**vars(args))
            sargs.fee_rate = fee
            sargs.slippage_pct = slip
            s_mom_cfg = v10a.make_momentum_config(sargs)
            s_exec_cfg = v10a.make_exec_config(s_mom_cfg)
            s_bull_cfg = v10a.make_bull_config(sargs)
            s_bull_exec_cfg = v10a.bull_to_exec_config(s_bull_cfg) if sargs.bull_execution_mode == "own" else s_exec_cfg
            s_engine_cfgs = {ENGINE_MOM: s_exec_cfg, ENGINE_BEAR: s_exec_cfg, ENGINE_BULL: s_bull_exec_cfg}
            features_full = make_features(
                raw, micro_ctx, sargs, flags,
                scenario=stress_name,
                mom_long_block_mask=spec.get("long_block"),
                mom_short_block_mask=spec.get("short_block"),
            )
            features = slice_trade_window(features_full, sargs)
            risk_down_builder = spec.get("risk_down_builder")
            if callable(risk_down_builder):
                for mask, note, scale in risk_down_builder(features):
                    m = mask.reindex(features.index).fillna(False).astype(bool)
                    features.loc[m, "risk_mult"] = (_num(features, "risk_mult", 1.0).loc[m] * float(scale)).clip(sargs.min_risk_mult, sargs.max_risk_mult or 10.0)
            trades, equity = run_research_backtest(
                features,
                s_exec_cfg,
                engine_cfgs=s_engine_cfgs,
                global_risk_scale=sargs.global_risk_scale,
                args=sargs,
                scenario=stress_name,
                **(spec.get("executor_kwargs", {}) or {}),
            )
            stress_summary = summary_metrics(stress_name, trades, equity, s_exec_cfg.initial_capital, extra={"fee_rate": fee, "slippage_pct": slip})
            stress_rows.append(stress_summary)
    stress_df = pd.DataFrame(stress_rows)
    scoreboard_df = build_scoreboard(summary_df, yearly_df, top_df, stress_df)

    print("[6/6] Writing research outputs...", flush=True)
    pd.DataFrame([baseline_summary]).to_csv(out_dir / "01_baseline_summary.csv", index=False)
    signal_events.to_csv(out_dir / "02_signal_event_table.csv", index=False)
    # Use baseline trades for signal diagnostics.
    b_features, b_trades, _b_equity = scenario_cache.get(BASELINE, (baseline_features, [], pd.DataFrame()))
    signal_level_diagnostics(b_trades, b_features, signal_events).to_csv(out_dir / "03_signal_level_diagnostics.csv", index=False)
    if not yearly_df.empty:
        yearly_df.to_csv(out_dir / "04_signal_level_yearly.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "04_signal_level_yearly.csv", index=False)
    blocked_signal_opportunity(signal_events).to_csv(out_dir / "05_blocked_signal_opportunity.csv", index=False)
    conflict_signal_diagnostics(signal_events).to_csv(out_dir / "06_conflict_signal_diagnostics.csv", index=False)
    summary_df.to_csv(out_dir / "07_variant_summary.csv", index=False)
    compare_df.to_csv(out_dir / "08_variant_compare_to_v10a.csv", index=False)
    yearly_df.to_csv(out_dir / "09_variant_yearly.csv", index=False)
    stress_df.to_csv(out_dir / "10_variant_stress.csv", index=False)
    top_df.to_csv(out_dir / "11_variant_top_trade_dependency.csv", index=False)
    scoreboard_df.to_csv(out_dir / "12_candidate_scoreboard.csv", index=False)
    if args.write_trades:
        trades_dir = out_dir / "trades"
        trades_dir.mkdir(exist_ok=True)
        for name, tdf in all_trades_for_optional.items():
            tdf.to_csv(trades_dir / f"{name}__trades.csv", index=False)
    meta = {
        "generated_at": _ts_now(),
        "git_commit": _git_commit(),
        "args": vars(args),
        "strategy_baseline": "V10A Momentum Micro + Momentum Short Speed Filter",
        "no_lookahead_notes": [
            "Routing rules use completed 4H signal bar data and shifted/past rolling range-speed thresholds.",
            "Forward returns/MFE/MAE in signal_event_table are diagnostic labels only and are not used by scenario masks.",
            "Partial TP/profit lock/no-progress exits are research-only and preserve closed-bar -> next-open decision timing.",
            "Independent engine books are research-only and not the current AetherEdge live execution model.",
        ],
        "scenario_count": int(len(summary_df)),
        "scenarios": summary_df["scenario"].astype(str).tolist() if "scenario" in summary_df.columns else [],
        "output_files": [
            "01_baseline_summary.csv", "02_signal_event_table.csv", "03_signal_level_diagnostics.csv",
            "04_signal_level_yearly.csv", "05_blocked_signal_opportunity.csv", "06_conflict_signal_diagnostics.csv",
            "07_variant_summary.csv", "08_variant_compare_to_v10a.csv", "09_variant_yearly.csv",
            "10_variant_stress.csv", "11_variant_top_trade_dependency.csv", "12_candidate_scoreboard.csv", "13_research_meta.json",
        ],
    }
    with (out_dir / "13_research_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    print("\nDone.")
    print(f"Output directory: {out_dir.resolve()}")
    if not scoreboard_df.empty:
        cols = [c for c in ["scenario", "candidate_pass_basic", "score", "total_return_pct", "max_drawdown_pct", "win_rate", "profit_factor", "mfe_ge_1r_ended_loss", "rule_note"] if c in scoreboard_df.columns]
        print(scoreboard_df[cols].head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
