#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V9E Momentum Signal Event Study Lab
===================================

Research-only event study for Momentum_V3 inside ETH_LF_Portfolio_V9E.

This script deliberately does NOT run the portfolio backtest. It treats every raw
Momentum signal as an independent event and measures what happened after it:
    - forward signed close-to-close returns
    - forward signed next-open-to-future-close returns
    - forward MFE / MAE from next open
    - yearly equal-weight signal quality
    - condition vs complement contrasts
    - rule ablation on signal events, e.g. all Momentum except Long+NotAligned

Use this to validate whether a proposed Momentum condition is intrinsically bad
or good, without portfolio state, router priority, compounding, or existing
position interactions.

This script does NOT change V9E strategy logic and does NOT place orders.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest.lf import eth_lf_portfolio_v9e_range_exit_overlay_backtest as v9e  # noqa: E402

ENGINE = "MOMENTUM_V3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Momentum raw-signal event study for V9E.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-15")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--warmup-days", type=int, default=365)
    p.add_argument("--initial-capital", type=float, default=1000.0)

    p.add_argument("--preset", choices=sorted(v9e.MOMENTUM_PRESETS), default="turbo")
    p.add_argument("--unit-risk-per-trade", type=float, default=None)
    p.add_argument("--max-total-notional-mult", type=float, default=None)
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--min-risk-mult", type=float, default=0.35)
    p.add_argument("--max-risk-mult", type=float, default=None)
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--disable-short", action="store_true")

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

    # Keep range-exit args for compatibility with v9e.make_exec_config / shared CLI patterns.
    p.add_argument("--range-exit-mode", choices=["off", "soft"], default="soft")
    p.add_argument("--range-exit-min-mfe-r", type=float, default=2.0)
    p.add_argument("--range-exit-giveback-frac", type=float, default=0.65)
    p.add_argument("--range-exit-min-hold-bars", type=int, default=2)
    p.add_argument("--range-exit-delay-bars", type=int, default=0)
    p.add_argument("--range-exit-contra-imbalance", type=float, default=0.05)
    p.add_argument("--range-exit-bad-close-pos", type=float, default=0.35)
    p.add_argument("--range-exit-no-reversal-required", dest="range_exit_require_reversal", action="store_false")
    p.set_defaults(range_exit_require_reversal=True)

    p.add_argument("--out-dir", default="data/reports/research/v9e_momentum_signal_event_study_lab")
    p.add_argument("--horizons", default="1,3,6,12", help="Forward return horizons in 4H bars.")
    p.add_argument("--mfe-horizon", type=int, default=12, help="Forward MFE/MAE horizon in 4H bars.")
    p.add_argument("--min-count", type=int, default=5)
    p.add_argument("--candidate-horizon", type=int, default=12)
    return p.parse_args()


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _side_name(side: int) -> str:
    if int(side) == 1:
        return "LONG"
    if int(side) == -1:
        return "SHORT"
    return "FLAT"


def _num_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _str_series(df: pd.DataFrame, col: str, default: str = "NA") -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="object")
    return df[col].astype("object").where(df[col].notna(), default).astype(str)


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype("boolean").fillna(False).astype(bool)


def _qcut_label(s: pd.Series, q: int = 4, prefix: str = "Q") -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    out = pd.Series("NA", index=s.index, dtype="object")
    valid = x.dropna()
    if valid.nunique() < 2 or len(valid) < q:
        return out
    try:
        binned = pd.qcut(valid, q=q, duplicates="drop")
    except ValueError:
        return out
    label_map = {cat: f"{prefix}{i + 1}" for i, cat in enumerate(binned.cat.categories)}
    out.loc[valid.index] = binned.map(label_map).astype(str)
    return out


def _signed_close_to_close_return(close: pd.Series, signal: pd.Series, horizon: int) -> pd.Series:
    future = close.shift(-int(horizon))
    return (future / close - 1.0) * signal.astype(float)


def _signed_next_open_return(df: pd.DataFrame, signal: pd.Series, horizon: int) -> pd.Series:
    # Entry anchor is next bar open after the signal bar closes.
    next_open = pd.to_numeric(df["open"], errors="coerce").shift(-1)
    future_close = pd.to_numeric(df["close"], errors="coerce").shift(-int(horizon))
    return (future_close / next_open - 1.0) * signal.astype(float)


def _forward_mfe_mae(df: pd.DataFrame, signal: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
    idx = df.index
    highs = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    opens = pd.to_numeric(df["open"], errors="coerce").to_numpy(dtype=float)
    sig = signal.astype(int).to_numpy()
    mfe = np.full(len(df), np.nan, dtype=float)
    mae = np.full(len(df), np.nan, dtype=float)
    for i in range(len(df)):
        side = sig[i]
        if side == 0 or i + 1 >= len(df):
            continue
        end = min(len(df), i + 1 + int(horizon))
        if end <= i + 1:
            continue
        entry = opens[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        hi = np.nanmax(highs[i + 1:end])
        lo = np.nanmin(lows[i + 1:end])
        if side == 1:
            mfe[i] = hi / entry - 1.0 if np.isfinite(hi) else np.nan
            mae[i] = lo / entry - 1.0 if np.isfinite(lo) else np.nan
        elif side == -1:
            mfe[i] = entry / lo - 1.0 if np.isfinite(lo) and lo > 0 else np.nan
            mae[i] = entry / hi - 1.0 if np.isfinite(hi) and hi > 0 else np.nan
    return pd.Series(mfe, index=idx), pd.Series(mae, index=idx)


def _profit_factor(ret_pct: pd.Series) -> float:
    x = pd.to_numeric(ret_pct, errors="coerce").dropna()
    if x.empty:
        return np.nan
    gp = float(x[x > 0].sum())
    gl = float(-x[x <= 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else np.nan
    return gp / gl


def _payoff_ratio(ret_pct: pd.Series) -> float:
    x = pd.to_numeric(ret_pct, errors="coerce").dropna()
    if x.empty:
        return np.nan
    wins = x[x > 0]
    losses = x[x <= 0]
    if wins.empty or losses.empty:
        return np.nan
    avg_loss = abs(float(losses.mean()))
    return float(wins.mean()) / avg_loss if avg_loss > 0 else np.nan


def build_momentum_features(args: argparse.Namespace) -> tuple[pd.DataFrame, Any, Any]:
    mom_cfg = v9e.make_momentum_config(args)
    exec_cfg = v9e.make_exec_config(mom_cfg)
    trade_start = pd.Timestamp(args.start_date)
    if args.warmup_start_date:
        load_start = pd.Timestamp(args.warmup_start_date)
    elif args.warmup_days and args.warmup_days > 0:
        load_start = trade_start - pd.Timedelta(days=int(args.warmup_days))
    else:
        load_start = trade_start
    load_start_str = load_start.strftime("%Y-%m-%d")

    print(f"Loading {args.symbol} 4H for warmup: {load_start_str} -> {args.end_date}; trade_start={args.start_date}", flush=True)
    base = v9e.load_data(args.symbol, load_start_str, args.end_date, "4H")
    print(f"Loaded {len(base):,} rows: {base.index[0]} -> {base.index[-1]}", flush=True)

    momentum = v9e.build_momentum_features(base, mom_cfg).copy()
    momentum["selected_engine"] = ENGINE
    momentum["selected_priority"] = 100
    momentum["momentum_signal"] = _num_series(momentum, "signal", 0).astype(int)
    momentum["momentum_selected"] = momentum["momentum_signal"] != 0
    momentum["bear_signal"] = 0
    momentum["bull_signal"] = 0
    momentum["bear_only"] = False
    momentum["bull_reclaim"] = False

    micro_ctx = v9e.load_range_footprint_context(args, load_start_str, args.end_date)
    momentum = v9e.apply_micro_context_filter(momentum, micro_ctx, args)
    momentum = momentum.loc[trade_start: pd.Timestamp(args.end_date)].copy().sort_index()
    print(f"Momentum feature rows after warmup slice: {len(momentum):,}; first={momentum.index[0]}", flush=True)
    return momentum, mom_cfg, exec_cfg


def build_event_table(features: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    horizons = [int(x.strip()) for x in str(args.horizons).split(",") if x.strip()]
    signal = _num_series(features, "signal", 0).astype(int)
    out = features.loc[signal != 0].copy()
    sig_events = signal.reindex(out.index).astype(int)

    out["timestamp"] = out.index
    out["year"] = out.index.year.astype(int)
    out["side"] = sig_events
    out["side_name"] = out["side"].map(_side_name)

    # Regime labels are calculated across raw Momentum signals only, not across all bars.
    out["adx_q"] = _qcut_label(_num_series(out, "adx"), 4, "ADX_Q")
    out["atr_pct_q"] = _qcut_label(_num_series(out, "atr_pct"), 4, "ATR_Q")
    out["d1_distance_abs_q"] = _qcut_label(_num_series(out, "d1_distance").abs(), 4, "D1DIST_Q")
    out["quality_mult_q"] = _qcut_label(_num_series(out, "quality_mult"), 4, "QUALITY_Q")
    out["risk_mult_q"] = _qcut_label(_num_series(out, "risk_mult"), 4, "RISK_Q")
    volume_ratio = _num_series(out, "volume") / _num_series(out, "volume_median")
    out["volume_ratio"] = volume_ratio.replace([np.inf, -np.inf], np.nan)
    out["volume_ratio_q"] = _qcut_label(out["volume_ratio"], 4, "VOL_Q")
    out["rf_imbalance_q"] = _qcut_label(_num_series(out, "rf_imbalance"), 4, "RFIMB_Q")
    out["rf_close_pos_q"] = _qcut_label(_num_series(out, "rf_close_pos"), 4, "RFCLOSE_Q")
    out["rf_taker_buy_ratio_q"] = _qcut_label(_num_series(out, "rf_taker_buy_ratio"), 4, "RFTAKER_Q")

    out["micro_filter_action"] = _str_series(out, "micro_filter_action", "NA")
    out["micro_context_available"] = _bool_series(out, "micro_context_available")
    out["micro_aligned"] = _bool_series(out, "micro_aligned")
    out["micro_contra"] = _bool_series(out, "micro_contra")
    out["long_quality_full"] = _bool_series(out, "long_quality_full")
    out["long_quality_weak"] = _bool_series(out, "long_quality_weak")
    out["long_mature_breakout"] = _bool_series(out, "long_mature_breakout")

    # Condition flags used in current research thread.
    is_long = out["side"].eq(1)
    is_short = out["side"].eq(-1)
    micro_not_aligned = out["micro_filter_action"].eq("NOT_ALIGNED_RISK_REDUCED")
    micro_neutral = out["micro_filter_action"].eq("NEUTRAL")
    low_volume = out["volume_ratio_q"].eq("VOL_Q1")
    quality_q2 = out["quality_mult_q"].eq("QUALITY_Q2")
    rfclose_q2 = out["rf_close_pos_q"].eq("RFCLOSE_Q2")
    rfclose_q3 = out["rf_close_pos_q"].eq("RFCLOSE_Q3")
    rfimb_q2 = out["rf_imbalance_q"].eq("RFIMB_Q2")
    adx_q2 = out["adx_q"].eq("ADX_Q2")

    out["cond_long_micro_not_aligned"] = is_long & micro_not_aligned
    out["cond_long_micro_neutral"] = is_long & micro_neutral
    out["cond_long_low_volume"] = is_long & low_volume
    out["cond_long_quality_q2"] = is_long & quality_q2
    out["cond_long_bad_any"] = is_long & (micro_not_aligned | low_volume | quality_q2)
    out["cond_short_adx_q2"] = is_short & adx_q2
    out["cond_short_rfclose_q2"] = is_short & rfclose_q2
    out["cond_short_rfclose_q3"] = is_short & rfclose_q3
    out["cond_short_rfimb_q2_rfclose_q2"] = is_short & rfimb_q2 & rfclose_q2

    full_signal = signal.astype(int)
    for h in horizons:
        close_ret = _signed_close_to_close_return(features["close"], full_signal, h).reindex(out.index)
        nopen_ret = _signed_next_open_return(features, full_signal, h).reindex(out.index)
        out[f"fwd_{h}bar_close_ret_pct"] = close_ret * 100.0
        out[f"fwd_{h}bar_next_open_ret_pct"] = nopen_ret * 100.0
        out[f"fwd_{h}bar_next_open_win"] = nopen_ret > 0
    mfe, mae = _forward_mfe_mae(features, full_signal, int(args.mfe_horizon))
    out[f"mfe_{args.mfe_horizon}bar_pct"] = mfe.reindex(out.index) * 100.0
    out[f"mae_{args.mfe_horizon}bar_pct"] = mae.reindex(out.index) * 100.0

    keep = [
        "timestamp", "year", "side", "side_name", "open", "high", "low", "close", "volume",
        "adx", "adx_q", "atr_pct", "atr_pct_q", "d1_distance", "d1_distance_abs_q",
        "risk_mult", "risk_mult_q", "quality_mult", "quality_mult_q", "volume_ratio", "volume_ratio_q",
        "micro_filter_action", "micro_context_available", "micro_aligned", "micro_contra",
        "rf_bar_count", "rf_imbalance", "rf_imbalance_q", "rf_close_pos", "rf_close_pos_q",
        "rf_taker_buy_ratio", "rf_taker_buy_ratio_q", "rf_micro_return_pct",
        "long_quality_full", "long_quality_weak", "long_mature_breakout",
        "cond_long_micro_not_aligned", "cond_long_micro_neutral", "cond_long_low_volume",
        "cond_long_quality_q2", "cond_long_bad_any", "cond_short_adx_q2", "cond_short_rfclose_q2",
        "cond_short_rfclose_q3", "cond_short_rfimb_q2_rfclose_q2",
        f"mfe_{args.mfe_horizon}bar_pct", f"mae_{args.mfe_horizon}bar_pct",
    ]
    for h in horizons:
        keep.extend([f"fwd_{h}bar_close_ret_pct", f"fwd_{h}bar_next_open_ret_pct", f"fwd_{h}bar_next_open_win"])
    return out[[c for c in keep if c in out.columns]].reset_index(drop=True)


def _subset_stats(events: pd.DataFrame, mask: pd.Series, name: str, args: argparse.Namespace, group_name: str = "condition") -> dict[str, Any]:
    horizons = [int(x.strip()) for x in str(args.horizons).split(",") if x.strip()]
    g = events.loc[mask.fillna(False)].copy()
    row: dict[str, Any] = {
        group_name: name,
        "count": int(len(g)),
        "year_count": int(g["year"].nunique()) if not g.empty and "year" in g else 0,
        "years": ";".join(map(str, sorted(g["year"].dropna().astype(int).unique().tolist()))) if not g.empty and "year" in g else "",
        "long_count": int((g["side_name"] == "LONG").sum()) if not g.empty and "side_name" in g else 0,
        "short_count": int((g["side_name"] == "SHORT").sum()) if not g.empty and "side_name" in g else 0,
        f"mfe_{args.mfe_horizon}bar_avg_pct": float(pd.to_numeric(g.get(f"mfe_{args.mfe_horizon}bar_pct", pd.Series(np.nan, index=g.index)), errors="coerce").mean()) if not g.empty else np.nan,
        f"mae_{args.mfe_horizon}bar_avg_pct": float(pd.to_numeric(g.get(f"mae_{args.mfe_horizon}bar_pct", pd.Series(np.nan, index=g.index)), errors="coerce").mean()) if not g.empty else np.nan,
    }
    for h in horizons:
        ret_col = f"fwd_{h}bar_next_open_ret_pct"
        close_col = f"fwd_{h}bar_close_ret_pct"
        ret = pd.to_numeric(g.get(ret_col, pd.Series(np.nan, index=g.index)), errors="coerce")
        close_ret = pd.to_numeric(g.get(close_col, pd.Series(np.nan, index=g.index)), errors="coerce")
        valid = ret.dropna()
        row[f"h{h}_valid_count"] = int(valid.shape[0])
        row[f"h{h}_win_rate"] = float((valid > 0).mean() * 100.0) if not valid.empty else np.nan
        row[f"h{h}_avg_ret_pct"] = float(valid.mean()) if not valid.empty else np.nan
        row[f"h{h}_median_ret_pct"] = float(valid.median()) if not valid.empty else np.nan
        row[f"h{h}_profit_factor"] = _profit_factor(valid)
        row[f"h{h}_payoff_ratio"] = _payoff_ratio(valid)
        row[f"h{h}_close_to_close_avg_ret_pct"] = float(close_ret.mean()) if not close_ret.dropna().empty else np.nan
    # Primary columns for sorting / comparison.
    h0 = int(args.candidate_horizon)
    row["primary_horizon"] = h0
    row["primary_avg_ret_pct"] = row.get(f"h{h0}_avg_ret_pct", np.nan)
    row["primary_win_rate"] = row.get(f"h{h0}_win_rate", np.nan)
    row["primary_profit_factor"] = row.get(f"h{h0}_profit_factor", np.nan)
    row["score"] = (
        _safe_float(row.get("primary_avg_ret_pct"), 0.0)
        + 0.04 * _safe_float(row.get("primary_win_rate"), 0.0)
        + 0.10 * _safe_float(row.get("year_count"), 0.0)
    )
    return row


def condition_masks(events: pd.DataFrame) -> dict[str, pd.Series]:
    side = events["side_name"].astype(str)
    micro = events["micro_filter_action"].astype(str)
    masks: dict[str, pd.Series] = {
        "all_momentum": pd.Series(True, index=events.index),
        "long_all": side.eq("LONG"),
        "short_all": side.eq("SHORT"),
        "long_micro_not_aligned": events.get("cond_long_micro_not_aligned", pd.Series(False, index=events.index)).astype(bool),
        "long_micro_neutral": events.get("cond_long_micro_neutral", pd.Series(False, index=events.index)).astype(bool),
        "long_low_volume": events.get("cond_long_low_volume", pd.Series(False, index=events.index)).astype(bool),
        "long_quality_q2": events.get("cond_long_quality_q2", pd.Series(False, index=events.index)).astype(bool),
        "long_bad_any": events.get("cond_long_bad_any", pd.Series(False, index=events.index)).astype(bool),
        "short_adx_q2": events.get("cond_short_adx_q2", pd.Series(False, index=events.index)).astype(bool),
        "short_rfclose_q2": events.get("cond_short_rfclose_q2", pd.Series(False, index=events.index)).astype(bool),
        "short_rfclose_q3": events.get("cond_short_rfclose_q3", pd.Series(False, index=events.index)).astype(bool),
        "short_rfimb_q2_rfclose_q2": events.get("cond_short_rfimb_q2_rfclose_q2", pd.Series(False, index=events.index)).astype(bool),
    }
    for value in sorted(micro.dropna().unique().tolist()):
        safe = str(value).replace(" ", "_").replace("/", "_")
        masks[f"micro_action_{safe}"] = micro.eq(value)
        masks[f"long_micro_action_{safe}"] = side.eq("LONG") & micro.eq(value)
        masks[f"short_micro_action_{safe}"] = side.eq("SHORT") & micro.eq(value)
    for col in ["adx_q", "atr_pct_q", "quality_mult_q", "risk_mult_q", "rf_imbalance_q", "rf_close_pos_q", "volume_ratio_q"]:
        if col not in events.columns:
            continue
        for value in sorted(events[col].dropna().astype(str).unique().tolist()):
            if value == "NA":
                continue
            safe = value.replace(" ", "_")
            masks[f"long_{col}_{safe}"] = side.eq("LONG") & events[col].astype(str).eq(value)
            masks[f"short_{col}_{safe}"] = side.eq("SHORT") & events[col].astype(str).eq(value)
    return masks


def build_condition_stats(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    masks = condition_masks(events)
    for name, mask in masks.items():
        if int(mask.fillna(False).sum()) < int(args.min_count):
            continue
        rows.append(_subset_stats(events, mask, name, args, group_name="condition"))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["score", "primary_avg_ret_pct", "primary_win_rate"], ascending=[False, False, False]).reset_index(drop=True)


def build_yearly_stats(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    masks = condition_masks(events)
    for name, mask in masks.items():
        if int(mask.fillna(False).sum()) < int(args.min_count):
            continue
        sub = events.loc[mask.fillna(False)].copy()
        for year, gy in sub.groupby("year"):
            if len(gy) < 1:
                continue
            rows.append(_subset_stats(gy, pd.Series(True, index=gy.index), f"{name}__{int(year)}", args, group_name="condition_year"))
            rows[-1]["condition"] = name
            rows[-1]["year"] = int(year)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["condition", "year"]).reset_index(drop=True)


def build_condition_contrast(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    masks = condition_masks(events)
    h = int(args.candidate_horizon)
    for name, mask in masks.items():
        n = int(mask.fillna(False).sum())
        if n < int(args.min_count):
            continue
        # Complement is within the same side if condition is side-specific, otherwise all other momentum events.
        if name.startswith("long_"):
            base_mask = events["side_name"].eq("LONG")
        elif name.startswith("short_"):
            base_mask = events["side_name"].eq("SHORT")
        else:
            base_mask = pd.Series(True, index=events.index)
        comp_mask = base_mask & (~mask.fillna(False))
        if int(comp_mask.sum()) < int(args.min_count):
            continue
        a = _subset_stats(events, mask, name, args, group_name="condition")
        b = _subset_stats(events, comp_mask, f"complement_of_{name}", args, group_name="condition")
        rows.append({
            "condition": name,
            "condition_count": a["count"],
            "complement_count": b["count"],
            "base_scope": "LONG" if name.startswith("long_") else ("SHORT" if name.startswith("short_") else "ALL"),
            f"h{h}_condition_win_rate": a.get(f"h{h}_win_rate"),
            f"h{h}_complement_win_rate": b.get(f"h{h}_win_rate"),
            f"h{h}_delta_win_rate": _safe_float(a.get(f"h{h}_win_rate"), np.nan) - _safe_float(b.get(f"h{h}_win_rate"), np.nan),
            f"h{h}_condition_avg_ret_pct": a.get(f"h{h}_avg_ret_pct"),
            f"h{h}_complement_avg_ret_pct": b.get(f"h{h}_avg_ret_pct"),
            f"h{h}_delta_avg_ret_pct": _safe_float(a.get(f"h{h}_avg_ret_pct"), np.nan) - _safe_float(b.get(f"h{h}_avg_ret_pct"), np.nan),
            f"h{h}_condition_pf": a.get(f"h{h}_profit_factor"),
            f"h{h}_complement_pf": b.get(f"h{h}_profit_factor"),
            "condition_year_count": a.get("year_count"),
            "complement_year_count": b.get("year_count"),
            "condition_years": a.get("years"),
            "complement_years": b.get("years"),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values([f"h{h}_delta_avg_ret_pct", f"h{h}_delta_win_rate"], ascending=[False, False]).reset_index(drop=True)


def build_rule_ablation_stats(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    masks = condition_masks(events)
    rules: dict[str, pd.Series] = {}
    all_mask = pd.Series(True, index=events.index)
    long_mask = events["side_name"].eq("LONG")
    short_mask = events["side_name"].eq("SHORT")

    # Keep-all / side baseline.
    rules["keep_all_momentum"] = all_mask
    rules["keep_all_long"] = long_mask
    rules["keep_all_short"] = short_mask

    # Candidate removals from the current research discussion.
    for cond in [
        "long_micro_not_aligned",
        "long_low_volume",
        "long_quality_q2",
        "long_bad_any",
        "short_rfclose_q3",
    ]:
        if cond in masks:
            rules[f"keep_all_except_{cond}"] = all_mask & (~masks[cond].fillna(False))
            if cond.startswith("long_"):
                rules[f"keep_long_except_{cond}"] = long_mask & (~masks[cond].fillna(False))
            if cond.startswith("short_"):
                rules[f"keep_short_except_{cond}"] = short_mask & (~masks[cond].fillna(False))

    # Candidate keep-only sets.
    for cond in [
        "long_micro_neutral",
        "short_adx_q2",
        "short_rfclose_q2",
        "short_rfimb_q2_rfclose_q2",
    ]:
        if cond in masks:
            rules[f"keep_only_{cond}"] = masks[cond].fillna(False)

    rows = []
    for name, mask in rules.items():
        if int(mask.fillna(False).sum()) < int(args.min_count):
            continue
        rows.append(_subset_stats(events, mask, name, args, group_name="rule"))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Add deltas versus keep_all_momentum and side baselines.
    h = int(args.candidate_horizon)
    base_all = out.loc[out["rule"].eq("keep_all_momentum")]
    base_long = out.loc[out["rule"].eq("keep_all_long")]
    base_short = out.loc[out["rule"].eq("keep_all_short")]
    for prefix, base in [("vs_all", base_all), ("vs_long", base_long), ("vs_short", base_short)]:
        if base.empty:
            continue
        b = base.iloc[0]
        out[f"{prefix}_h{h}_avg_ret_delta"] = pd.to_numeric(out[f"h{h}_avg_ret_pct"], errors="coerce") - _safe_float(b.get(f"h{h}_avg_ret_pct"), np.nan)
        out[f"{prefix}_h{h}_win_rate_delta"] = pd.to_numeric(out[f"h{h}_win_rate"], errors="coerce") - _safe_float(b.get(f"h{h}_win_rate"), np.nan)
        out[f"{prefix}_h{h}_pf_delta"] = pd.to_numeric(out[f"h{h}_profit_factor"], errors="coerce") - _safe_float(b.get(f"h{h}_profit_factor"), np.nan)
    return out.sort_values(["score", "primary_avg_ret_pct", "primary_win_rate"], ascending=[False, False, False]).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features, _mom_cfg, _exec_cfg = build_momentum_features(args)
    events = build_event_table(features, args)
    events.to_csv(out_dir / "v9e_momentum_all_signal_events.csv", index=False, encoding="utf-8-sig")

    condition_stats = build_condition_stats(events, args)
    condition_stats.to_csv(out_dir / "v9e_momentum_condition_signal_stats.csv", index=False, encoding="utf-8-sig")

    yearly_stats = build_yearly_stats(events, args)
    yearly_stats.to_csv(out_dir / "v9e_momentum_condition_yearly_signal_stats.csv", index=False, encoding="utf-8-sig")

    contrast = build_condition_contrast(events, args)
    contrast.to_csv(out_dir / "v9e_momentum_condition_contrast.csv", index=False, encoding="utf-8-sig")

    ablation = build_rule_ablation_stats(events, args)
    ablation.to_csv(out_dir / "v9e_momentum_rule_ablation_signal_stats.csv", index=False, encoding="utf-8-sig")

    h = int(args.candidate_horizon)
    overview_rows = []
    for name, mask in {
        "all_momentum": pd.Series(True, index=events.index),
        "long_all": events["side_name"].eq("LONG"),
        "short_all": events["side_name"].eq("SHORT"),
        "long_micro_not_aligned": events.get("cond_long_micro_not_aligned", pd.Series(False, index=events.index)).astype(bool),
        "long_micro_neutral": events.get("cond_long_micro_neutral", pd.Series(False, index=events.index)).astype(bool),
    }.items():
        overview_rows.append(_subset_stats(events, mask, name, args, group_name="scope"))
    overview = pd.DataFrame(overview_rows)
    overview.to_csv(out_dir / "v9e_momentum_signal_event_overview.csv", index=False, encoding="utf-8-sig")

    meta = {
        "script": "v9e_momentum_signal_event_study_lab.py",
        "mode": "raw_signal_event_study_not_chronological_backtest",
        "symbol": args.symbol,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "signal_event_count": int(len(events)),
        "candidate_horizon": h,
        "horizons": args.horizons,
        "important_note": "Each Momentum signal is treated as an independent event. This ignores portfolio state, router priority, compounding, and overlapping positions.",
        "outputs": [
            "v9e_momentum_all_signal_events.csv",
            "v9e_momentum_signal_event_overview.csv",
            "v9e_momentum_condition_signal_stats.csv",
            "v9e_momentum_condition_yearly_signal_stats.csv",
            "v9e_momentum_condition_contrast.csv",
            "v9e_momentum_rule_ablation_signal_stats.csv",
        ],
    }
    (out_dir / "v9e_momentum_signal_event_study_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Momentum Signal Event Study Done ===", flush=True)
    print(f"Output dir: {out_dir}", flush=True)
    print(f"Signal events: {len(events):,}", flush=True)
    print("Key files:", flush=True)
    for name in meta["outputs"]:
        print(f"  - {name}", flush=True)
    if not overview.empty:
        cols = ["scope", "count", "year_count", f"h{h}_win_rate", f"h{h}_avg_ret_pct", f"h{h}_profit_factor", f"mfe_{args.mfe_horizon}bar_avg_pct", f"mae_{args.mfe_horizon}bar_avg_pct"]
        print("\nOverview:", flush=True)
        print(overview[[c for c in cols if c in overview.columns]].to_string(index=False), flush=True)
    if not ablation.empty:
        print("\nTop rule ablations by event-study score:", flush=True)
        cols = ["rule", "count", "year_count", f"h{h}_win_rate", f"h{h}_avg_ret_pct", f"h{h}_profit_factor", "score"]
        print(ablation[[c for c in cols if c in ablation.columns]].head(12).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
