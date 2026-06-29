#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V9E Engine Bad-Environment Event Study Lab
==========================================

Research-only raw signal event study for all V9E entry engines:
    - MOMENTUM_V3
    - BULL_RECLAIM_V2
    - BEAR_V3_ONLY

This script deliberately does NOT run the chronological portfolio backtest.
It treats every raw engine signal as an independent event and measures forward
signed return, MFE, and MAE after the signal.

Purpose in the current research thread:
    1. Validate whether the Momentum bad environments are intrinsically bad.
    2. Check whether the same environments also hurt Bull/Bear, or whether they
       are engine-specific.
    3. Avoid blindly blocking all engines with a condition that was only proven
       bad for Momentum Long.

No strategy files are modified and no orders are placed.
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

ENGINE_ORDER = ["MOMENTUM_V3", "BULL_RECLAIM_V2", "BEAR_V3_ONLY"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Raw-signal bad-environment event study for all V9E engines.")
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-15")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--warmup-days", type=int, default=365)
    p.add_argument("--initial-capital", type=float, default=1000.0)

    # V9E-compatible config args.
    p.add_argument("--preset", choices=sorted(v9e.MOMENTUM_PRESETS), default="turbo")
    p.add_argument("--unit-risk-per-trade", type=float, default=None)
    p.add_argument("--max-total-notional-mult", type=float, default=None)
    p.add_argument("--max-units", type=int, default=None)
    p.add_argument("--min-risk-mult", type=float, default=0.35)
    p.add_argument("--max-risk-mult", type=float, default=None)
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--disable-short", action="store_true")

    p.add_argument("--bear-preset", choices=sorted(v9e.BEAR_PRESETS), default="high")
    p.add_argument("--bear-min-risk-mult", type=float, default=0.25)
    p.add_argument("--bear-standalone-risk-scale", type=float, default=1.0)
    p.add_argument("--bear-standalone-quality-scale", type=float, default=1.0)
    p.add_argument("--disable-bear-standalone", action="store_true")

    p.add_argument("--bull-preset", choices=sorted(v9e.BULL_PRESETS), default="high")
    p.add_argument("--bull-min-risk-mult", type=float, default=0.25)
    p.add_argument("--bull-reclaim-risk-scale", type=float, default=1.0)
    p.add_argument("--bull-reclaim-quality-scale", type=float, default=1.0)
    p.add_argument("--bull-execution-mode", choices=["inherit", "own"], default="inherit")
    p.add_argument("--disable-bull-reclaim", action="store_true")

    p.add_argument("--priority-mode", choices=sorted(v9e.PRIORITY_MODES), default="reclaim_first")
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

    # Kept for compatibility with shared V9E config shape.
    p.add_argument("--range-exit-mode", choices=["off", "soft"], default="soft")
    p.add_argument("--range-exit-min-mfe-r", type=float, default=2.0)
    p.add_argument("--range-exit-giveback-frac", type=float, default=0.65)
    p.add_argument("--range-exit-min-hold-bars", type=int, default=2)
    p.add_argument("--range-exit-delay-bars", type=int, default=0)
    p.add_argument("--range-exit-contra-imbalance", type=float, default=0.05)
    p.add_argument("--range-exit-bad-close-pos", type=float, default=0.35)
    p.add_argument("--range-exit-no-reversal-required", dest="range_exit_require_reversal", action="store_false")
    p.set_defaults(range_exit_require_reversal=True)

    p.add_argument("--out-dir", default="data/reports/research/v9e_engine_bad_environment_event_study_lab")
    p.add_argument("--horizons", default="1,3,6,12", help="Forward return horizons in 4H bars.")
    p.add_argument("--mfe-horizon", type=int, default=12)
    p.add_argument("--candidate-horizon", type=int, default=12)
    p.add_argument("--min-count", type=int, default=5)
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


def _signed_next_open_return(df: pd.DataFrame, signal: pd.Series, horizon: int) -> pd.Series:
    next_open = pd.to_numeric(df["open"], errors="coerce").shift(-1)
    future_close = pd.to_numeric(df["close"], errors="coerce").shift(-int(horizon))
    return (future_close / next_open - 1.0) * signal.astype(float)


def _signed_close_to_close_return(close: pd.Series, signal: pd.Series, horizon: int) -> pd.Series:
    future = close.shift(-int(horizon))
    return (future / close - 1.0) * signal.astype(float)


def _forward_mfe_mae(df: pd.DataFrame, signal: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
    idx = df.index
    highs = pd.to_numeric(df["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(df["low"], errors="coerce").to_numpy(dtype=float)
    opens = pd.to_numeric(df["open"], errors="coerce").to_numpy(dtype=float)
    sig = signal.astype(int).to_numpy()
    mfe = np.full(len(df), np.nan, dtype=float)
    mae = np.full(len(df), np.nan, dtype=float)
    for i, side in enumerate(sig):
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
        if int(side) == 1:
            mfe[i] = hi / entry - 1.0 if np.isfinite(hi) else np.nan
            mae[i] = lo / entry - 1.0 if np.isfinite(lo) else np.nan
        elif int(side) == -1:
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


def _load_base(args: argparse.Namespace) -> pd.DataFrame:
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
    return base


def _prep_engine_frame(engine: str, frame: pd.DataFrame, micro_ctx: pd.DataFrame, args: argparse.Namespace, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    f = frame.copy().sort_index()
    if "volume_median" not in f.columns and "volume" in f.columns:
        f["volume_median"] = pd.to_numeric(f["volume"], errors="coerce").rolling(90, min_periods=20).median()
    f["selected_engine"] = engine
    f["selected_priority"] = 100
    sig = _num_series(f, "signal", 0).fillna(0).astype(int)
    f["long_signal"] = sig == 1
    f["short_signal"] = sig == -1
    f = v9e.apply_micro_context_filter(f, micro_ctx, args)
    f["engine"] = engine
    return f.loc[start:end].copy().sort_index()


def build_engine_features(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    base = _load_base(args)
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    load_start_str = base.index[0].strftime("%Y-%m-%d")

    mom_cfg = v9e.make_momentum_config(args)
    bear_cfg = v9e.make_bear_config(args)
    bull_cfg = v9e.make_bull_config(args)

    momentum = v9e.build_momentum_features(base, mom_cfg)
    bear = v9e.build_bear_features(base, bear_cfg)
    bull = v9e.build_bull_features(base, bull_cfg)

    micro_ctx = v9e.load_range_footprint_context(args, load_start_str, args.end_date)

    frames = {
        "MOMENTUM_V3": _prep_engine_frame("MOMENTUM_V3", momentum, micro_ctx, args, start, end),
        "BULL_RECLAIM_V2": _prep_engine_frame("BULL_RECLAIM_V2", bull, micro_ctx, args, start, end),
        "BEAR_V3_ONLY": _prep_engine_frame("BEAR_V3_ONLY", bear, micro_ctx, args, start, end),
    }
    for engine, df in frames.items():
        sig_count = int((_num_series(df, "signal", 0).fillna(0).astype(int) != 0).sum())
        print(f"{engine} feature rows={len(df):,}; raw_signal_count={sig_count:,}", flush=True)
    return frames


def _add_event_labels(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["micro_filter_action"] = out["micro_filter_action"].fillna("NA").astype(str)
    out["micro_not_aligned"] = out["micro_filter_action"].eq("NOT_ALIGNED_RISK_REDUCED")
    out["micro_neutral"] = out["micro_filter_action"].eq("NEUTRAL")
    out["micro_contra_reduced"] = out["micro_filter_action"].eq("CONTRA_RISK_REDUCED")

    # Quantile labels are engine-local. This avoids comparing Momentum's low-volume quartile
    # against Bull/Bear distributions with different signal timing.
    for label_col, value_col, prefix in [
        ("adx_q", "adx", "ADX_Q"),
        ("atr_pct_q", "atr_pct", "ATR_Q"),
        ("d1_distance_abs_q", "d1_distance_abs", "D1DIST_Q"),
        ("quality_mult_q", "quality_mult", "QUALITY_Q"),
        ("risk_mult_q", "risk_mult", "RISK_Q"),
        ("volume_ratio_q", "volume_ratio", "VOL_Q"),
        ("rf_imbalance_q", "rf_imbalance", "RFIMB_Q"),
        ("rf_close_pos_q", "rf_close_pos", "RFCLOSE_Q"),
        ("rf_taker_buy_ratio_q", "rf_taker_buy_ratio", "RFTAKER_Q"),
    ]:
        out[label_col] = "NA"
        if value_col not in out.columns:
            continue
        for engine, idx in out.groupby("engine").groups.items():
            out.loc[idx, label_col] = _qcut_label(out.loc[idx, value_col], 4, prefix).values

    out["low_volume"] = out["volume_ratio_q"].eq("VOL_Q1")
    out["quality_q2"] = out["quality_mult_q"].eq("QUALITY_Q2")
    out["rfclose_q2"] = out["rf_close_pos_q"].eq("RFCLOSE_Q2")
    out["rfclose_q3"] = out["rf_close_pos_q"].eq("RFCLOSE_Q3")
    out["adx_q2"] = out["adx_q"].eq("ADX_Q2")
    out["rfimb_q2"] = out["rf_imbalance_q"].eq("RFIMB_Q2")

    # Generic bad-environment definitions from the Momentum discussion.
    out["bad_micro_or_low_volume"] = out["micro_not_aligned"] | out["low_volume"]
    out["bad_any"] = out["micro_not_aligned"] | out["low_volume"] | out["quality_q2"]
    return out


def build_engine_events(frames: dict[str, pd.DataFrame], args: argparse.Namespace) -> pd.DataFrame:
    horizons = [int(x.strip()) for x in str(args.horizons).split(",") if x.strip()]
    rows: list[pd.DataFrame] = []
    for engine in ENGINE_ORDER:
        f = frames[engine].copy().sort_index()
        signal = _num_series(f, "signal", 0).fillna(0).astype(int)
        ev = f.loc[signal != 0].copy()
        if ev.empty:
            continue
        ev["timestamp"] = ev.index
        ev["year"] = ev.index.year.astype(int)
        ev["engine"] = engine
        ev["side"] = signal.reindex(ev.index).astype(int)
        ev["side_name"] = ev["side"].map(_side_name)
        ev["d1_distance_abs"] = _num_series(ev, "d1_distance").abs()
        vol = _num_series(ev, "volume")
        vol_med = _num_series(ev, "volume_median")
        fallback_med = _num_series(f, "volume").rolling(90, min_periods=20).median().reindex(ev.index)
        vol_med = vol_med.where(vol_med.notna() & (vol_med > 0), fallback_med)
        ev["volume_ratio"] = (vol / vol_med).replace([np.inf, -np.inf], np.nan)
        ev["micro_filter_action"] = _str_series(ev, "micro_filter_action", "NA")
        ev["micro_context_available"] = _bool_series(ev, "micro_context_available")
        ev["micro_aligned"] = _bool_series(ev, "micro_aligned")
        ev["micro_contra"] = _bool_series(ev, "micro_contra")

        for h in horizons:
            ev[f"fwd_{h}bar_next_open_ret_pct"] = _signed_next_open_return(f, signal, h).reindex(ev.index) * 100.0
            ev[f"fwd_{h}bar_close_ret_pct"] = _signed_close_to_close_return(f["close"], signal, h).reindex(ev.index) * 100.0
            ev[f"fwd_{h}bar_next_open_win"] = ev[f"fwd_{h}bar_next_open_ret_pct"] > 0
        mfe, mae = _forward_mfe_mae(f, signal, int(args.mfe_horizon))
        ev[f"mfe_{args.mfe_horizon}bar_pct"] = mfe.reindex(ev.index) * 100.0
        ev[f"mae_{args.mfe_horizon}bar_pct"] = mae.reindex(ev.index) * 100.0
        rows.append(ev)

    if not rows:
        return pd.DataFrame()
    events = pd.concat(rows, axis=0, ignore_index=False).sort_index()
    events = _add_event_labels(events.reset_index(drop=True))
    keep = [
        "timestamp", "year", "engine", "side", "side_name",
        "open", "high", "low", "close", "volume", "volume_median", "volume_ratio", "volume_ratio_q",
        "adx", "adx_q", "atr_pct", "atr_pct_q", "d1_distance", "d1_distance_abs", "d1_distance_abs_q",
        "risk_mult", "risk_mult_q", "quality_mult", "quality_mult_q",
        "micro_filter_action", "micro_context_available", "micro_aligned", "micro_contra",
        "rf_bar_count", "rf_micro_return_pct", "rf_close_pos", "rf_close_pos_q", "rf_delta_sum", "rf_imbalance", "rf_imbalance_q", "rf_taker_buy_ratio", "rf_taker_buy_ratio_q",
        "micro_not_aligned", "micro_neutral", "micro_contra_reduced", "low_volume", "quality_q2", "bad_micro_or_low_volume", "bad_any",
        f"mfe_{args.mfe_horizon}bar_pct", f"mae_{args.mfe_horizon}bar_pct",
    ]
    for h in horizons:
        keep.extend([f"fwd_{h}bar_next_open_ret_pct", f"fwd_{h}bar_close_ret_pct", f"fwd_{h}bar_next_open_win"])
    return events[[c for c in keep if c in events.columns]].reset_index(drop=True)


def _subset_stats(events: pd.DataFrame, mask: pd.Series, name: str, args: argparse.Namespace, label_col: str = "group") -> dict[str, Any]:
    horizons = [int(x.strip()) for x in str(args.horizons).split(",") if x.strip()]
    g = events.loc[mask.fillna(False)].copy()
    row: dict[str, Any] = {
        label_col: name,
        "count": int(len(g)),
        "year_count": int(g["year"].nunique()) if not g.empty and "year" in g else 0,
        "years": ";".join(map(str, sorted(g["year"].dropna().astype(int).unique().tolist()))) if not g.empty and "year" in g else "",
        "engine_count": int(g["engine"].nunique()) if not g.empty and "engine" in g else 0,
        "engines": ";".join(sorted(g["engine"].dropna().astype(str).unique().tolist())) if not g.empty and "engine" in g else "",
        "long_count": int((g.get("side_name") == "LONG").sum()) if not g.empty else 0,
        "short_count": int((g.get("side_name") == "SHORT").sum()) if not g.empty else 0,
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
    h0 = int(args.candidate_horizon)
    row["primary_horizon"] = h0
    row["primary_avg_ret_pct"] = row.get(f"h{h0}_avg_ret_pct", np.nan)
    row["primary_win_rate"] = row.get(f"h{h0}_win_rate", np.nan)
    row["primary_profit_factor"] = row.get(f"h{h0}_profit_factor", np.nan)
    row["score"] = _safe_float(row.get("primary_avg_ret_pct"), 0.0) + 0.04 * _safe_float(row.get("primary_win_rate"), 0.0) + 0.08 * _safe_float(row.get("year_count"), 0.0)
    return row


def _condition_masks(events: pd.DataFrame) -> dict[str, pd.Series]:
    e = events["engine"].astype(str)
    side = events["side_name"].astype(str)
    masks: dict[str, pd.Series] = {
        "all_events": pd.Series(True, index=events.index),
        "micro_not_aligned": events["micro_not_aligned"].astype(bool),
        "low_volume": events["low_volume"].astype(bool),
        "quality_q2": events["quality_q2"].astype(bool),
        "bad_micro_or_low_volume": events["bad_micro_or_low_volume"].astype(bool),
        "bad_any": events["bad_any"].astype(bool),
        "micro_neutral": events["micro_neutral"].astype(bool),
    }
    for engine in ENGINE_ORDER:
        eng_mask = e.eq(engine)
        masks[f"{engine}__all"] = eng_mask
        for s in sorted(side[eng_mask].dropna().unique().tolist()):
            masks[f"{engine}__{s}__all"] = eng_mask & side.eq(s)
        for cond in ["micro_not_aligned", "low_volume", "quality_q2", "bad_micro_or_low_volume", "bad_any", "micro_neutral"]:
            cm = masks[cond]
            masks[f"{engine}__{cond}"] = eng_mask & cm
            for s in sorted(side[eng_mask].dropna().unique().tolist()):
                masks[f"{engine}__{s}__{cond}"] = eng_mask & side.eq(s) & cm
    return masks


def build_summary(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for name, mask in _condition_masks(events).items():
        if int(mask.fillna(False).sum()) < int(args.min_count):
            continue
        rows.append(_subset_stats(events, mask, name, args, label_col="group"))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["score", "primary_avg_ret_pct", "primary_win_rate"], ascending=[False, False, False]).reset_index(drop=True)


def build_contrast(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    h = int(args.candidate_horizon)
    masks = _condition_masks(events)
    for name, mask in masks.items():
        n = int(mask.fillna(False).sum())
        if n < int(args.min_count):
            continue
        parts = name.split("__")
        # Complement is within the same engine+side for engine-side conditions,
        # within engine for engine conditions, or all events for generic conditions.
        if len(parts) >= 3 and parts[0] in ENGINE_ORDER and parts[1] in {"LONG", "SHORT"}:
            base = events["engine"].eq(parts[0]) & events["side_name"].eq(parts[1])
        elif len(parts) >= 2 and parts[0] in ENGINE_ORDER:
            base = events["engine"].eq(parts[0])
        else:
            base = pd.Series(True, index=events.index)
        comp = base & (~mask.fillna(False))
        if int(comp.sum()) < int(args.min_count):
            continue
        a = _subset_stats(events, mask, name, args, label_col="condition")
        b = _subset_stats(events, comp, f"complement_of_{name}", args, label_col="condition")
        rows.append({
            "condition": name,
            "base_scope": "engine_side" if len(parts) >= 3 and parts[0] in ENGINE_ORDER else ("engine" if len(parts) >= 2 and parts[0] in ENGINE_ORDER else "all"),
            "condition_count": a["count"],
            "complement_count": b["count"],
            "condition_year_count": a["year_count"],
            "complement_year_count": b["year_count"],
            "condition_years": a["years"],
            "complement_years": b["years"],
            f"h{h}_condition_win_rate": a.get(f"h{h}_win_rate"),
            f"h{h}_complement_win_rate": b.get(f"h{h}_win_rate"),
            f"h{h}_delta_win_rate": _safe_float(a.get(f"h{h}_win_rate"), np.nan) - _safe_float(b.get(f"h{h}_win_rate"), np.nan),
            f"h{h}_condition_avg_ret_pct": a.get(f"h{h}_avg_ret_pct"),
            f"h{h}_complement_avg_ret_pct": b.get(f"h{h}_avg_ret_pct"),
            f"h{h}_delta_avg_ret_pct": _safe_float(a.get(f"h{h}_avg_ret_pct"), np.nan) - _safe_float(b.get(f"h{h}_avg_ret_pct"), np.nan),
            f"h{h}_condition_pf": a.get(f"h{h}_profit_factor"),
            f"h{h}_complement_pf": b.get(f"h{h}_profit_factor"),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values([f"h{h}_delta_avg_ret_pct", f"h{h}_delta_win_rate"], ascending=[True, True]).reset_index(drop=True)


def build_yearly(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    for name, mask in _condition_masks(events).items():
        if int(mask.fillna(False).sum()) < int(args.min_count):
            continue
        sub = events.loc[mask.fillna(False)].copy()
        for year, gy in sub.groupby("year"):
            rows.append(_subset_stats(gy, pd.Series(True, index=gy.index), f"{name}__{int(year)}", args, label_col="group_year"))
            rows[-1]["group"] = name
            rows[-1]["year"] = int(year)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["group", "year"]).reset_index(drop=True)


def build_ablation(events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    h = int(args.candidate_horizon)
    all_mask = pd.Series(True, index=events.index)
    for cond in ["micro_not_aligned", "low_volume", "quality_q2", "bad_micro_or_low_volume", "bad_any"]:
        cm = events[cond].astype(bool)
        rows.append(_subset_stats(events, all_mask & (~cm), f"keep_all_except_{cond}", args, label_col="rule"))
        for engine in ENGINE_ORDER:
            em = events["engine"].eq(engine)
            rows.append(_subset_stats(events, all_mask & (~(em & cm)), f"keep_all_except_{engine}__{cond}", args, label_col="rule"))
            for side in sorted(events.loc[em, "side_name"].dropna().astype(str).unique().tolist()):
                sm = em & events["side_name"].eq(side)
                rows.append(_subset_stats(events, all_mask & (~(sm & cm)), f"keep_all_except_{engine}__{side}__{cond}", args, label_col="rule"))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    base = _subset_stats(events, all_mask, "keep_all_events", args, label_col="rule")
    for col in [f"h{h}_avg_ret_pct", f"h{h}_win_rate", f"h{h}_profit_factor"]:
        out[f"vs_all_{col}_delta"] = pd.to_numeric(out[col], errors="coerce") - _safe_float(base.get(col), np.nan)
    return out.sort_values(["score", "primary_avg_ret_pct", "primary_win_rate"], ascending=[False, False, False]).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = build_engine_features(args)
    events = build_engine_events(frames, args)
    if events.empty:
        raise RuntimeError("No engine events found. Check data/config.")

    events.to_csv(out_dir / "v9e_all_engine_signal_events.csv", index=False, encoding="utf-8-sig")
    summary = build_summary(events, args)
    summary.to_csv(out_dir / "v9e_engine_bad_environment_summary.csv", index=False, encoding="utf-8-sig")
    contrast = build_contrast(events, args)
    contrast.to_csv(out_dir / "v9e_engine_bad_environment_contrast.csv", index=False, encoding="utf-8-sig")
    yearly = build_yearly(events, args)
    yearly.to_csv(out_dir / "v9e_engine_bad_environment_yearly.csv", index=False, encoding="utf-8-sig")
    ablation = build_ablation(events, args)
    ablation.to_csv(out_dir / "v9e_engine_bad_environment_ablation_signal_stats.csv", index=False, encoding="utf-8-sig")

    h = int(args.candidate_horizon)
    overview_rows = []
    for engine in ENGINE_ORDER:
        em = events["engine"].eq(engine)
        overview_rows.append(_subset_stats(events, em, engine, args, label_col="scope"))
        for side in sorted(events.loc[em, "side_name"].dropna().astype(str).unique().tolist()):
            overview_rows.append(_subset_stats(events, em & events["side_name"].eq(side), f"{engine}__{side}", args, label_col="scope"))
    overview = pd.DataFrame(overview_rows)
    overview.to_csv(out_dir / "v9e_engine_signal_event_overview.csv", index=False, encoding="utf-8-sig")

    meta = {
        "script": "v9e_engine_bad_environment_event_study_lab.py",
        "mode": "raw_signal_event_study_not_chronological_portfolio_backtest",
        "symbol": args.symbol,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "event_count": int(len(events)),
        "candidate_horizon": h,
        "horizons": args.horizons,
        "important_note": "Quantile labels are engine-local. This is an event study only; it ignores portfolio state, router priority, exits, compounding, and overlapping positions.",
        "outputs": [
            "v9e_all_engine_signal_events.csv",
            "v9e_engine_signal_event_overview.csv",
            "v9e_engine_bad_environment_summary.csv",
            "v9e_engine_bad_environment_contrast.csv",
            "v9e_engine_bad_environment_yearly.csv",
            "v9e_engine_bad_environment_ablation_signal_stats.csv",
        ],
    }
    (out_dir / "v9e_engine_bad_environment_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== V9E Engine Bad-Environment Event Study Done ===", flush=True)
    print(f"Output dir: {out_dir}", flush=True)
    print(f"Signal events: {len(events):,}", flush=True)
    print("Key files:", flush=True)
    for name in meta["outputs"]:
        print(f"  - {name}", flush=True)
    if not overview.empty:
        cols = ["scope", "count", "year_count", f"h{h}_win_rate", f"h{h}_avg_ret_pct", f"h{h}_profit_factor", f"mfe_{args.mfe_horizon}bar_avg_pct", f"mae_{args.mfe_horizon}bar_avg_pct"]
        print("\nOverview:", flush=True)
        print(overview[[c for c in cols if c in overview.columns]].to_string(index=False), flush=True)
    if not contrast.empty:
        cols = ["condition", "base_scope", "condition_count", "complement_count", f"h{h}_condition_avg_ret_pct", f"h{h}_complement_avg_ret_pct", f"h{h}_delta_avg_ret_pct", f"h{h}_condition_pf", f"h{h}_complement_pf"]
        print("\nWorst condition contrasts by delta avg ret:", flush=True)
        print(contrast[[c for c in cols if c in contrast.columns]].head(12).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
