#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V9E Momentum Regime Lab
=======================

Research-only diagnostics for Momentum_V3 inside ETH_LF_Portfolio_V9E.

Goal:
    Find when Momentum is actually good, instead of cutting it by LONG/SHORT blindly.

Outputs:
    - all Momentum signal labels with forward returns/MFE/MAE
    - group stats by side/year/ADX/ATR/micro/quality/risk/daily-distance/range-footprint
    - stable high-win/high-expectancy candidate regimes
    - bad candidate regimes
    - standalone Momentum trade regime stats

This script does NOT change V9E strategy logic and does NOT place orders.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest.lf import eth_lf_portfolio_v9e_range_exit_overlay_backtest as v9e  # noqa: E402

ENGINE = "MOMENTUM_V3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Momentum regime diagnostics for V9E.")
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

    p.add_argument("--range-exit-mode", choices=["off", "soft"], default="soft")
    p.add_argument("--range-exit-min-mfe-r", type=float, default=2.0)
    p.add_argument("--range-exit-giveback-frac", type=float, default=0.65)
    p.add_argument("--range-exit-min-hold-bars", type=int, default=2)
    p.add_argument("--range-exit-delay-bars", type=int, default=0)
    p.add_argument("--range-exit-contra-imbalance", type=float, default=0.05)
    p.add_argument("--range-exit-bad-close-pos", type=float, default=0.35)
    p.add_argument("--range-exit-no-reversal-required", dest="range_exit_require_reversal", action="store_false")
    p.set_defaults(range_exit_require_reversal=True)

    p.add_argument("--out-dir", default="data/reports/research/v9e_momentum_regime_lab")
    p.add_argument("--horizons", default="1,3,6,12", help="Forward close-return horizons in 4H bars.")
    p.add_argument("--mfe-horizon", type=int, default=12, help="Forward MFE/MAE horizon in 4H bars.")
    p.add_argument("--min-count", type=int, default=8, help="Minimum group sample count for candidate regimes.")
    p.add_argument("--min-years", type=int, default=2, help="Minimum distinct years for candidate regimes.")
    p.add_argument("--candidate-horizon", type=int, default=12, help="Horizon used for candidate ranking.")
    p.add_argument("--skip-standalone-trades", action="store_true")
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
    return df[col].astype(str).fillna(default)


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
    cats = binned.cat.categories
    label_map = {cat: f"{prefix}{i + 1}" for i, cat in enumerate(cats)}
    out.loc[valid.index] = binned.map(label_map).astype(str)
    return out


def _signed_forward_return(close: pd.Series, signal: pd.Series, horizon: int) -> pd.Series:
    future = close.shift(-int(horizon))
    return (future / close - 1.0) * signal.astype(float)


def _forward_mfe_mae(df: pd.DataFrame, signal: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
    # Label uses next open as executable entry anchor and future high/low over the next horizon bars.
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
        entry = opens[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        hi = np.nanmax(highs[i + 1:end]) if end > i + 1 else np.nan
        lo = np.nanmin(lows[i + 1:end]) if end > i + 1 else np.nan
        if side == 1:
            mfe[i] = hi / entry - 1.0 if np.isfinite(hi) else np.nan
            mae[i] = lo / entry - 1.0 if np.isfinite(lo) else np.nan
        elif side == -1:
            mfe[i] = entry / lo - 1.0 if np.isfinite(lo) and lo > 0 else np.nan
            mae[i] = entry / hi - 1.0 if np.isfinite(hi) and hi > 0 else np.nan
    return pd.Series(mfe, index=idx), pd.Series(mae, index=idx)


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

    momentum = v9e.build_momentum_features(base, mom_cfg)
    momentum = momentum.copy()
    momentum["selected_engine"] = ENGINE
    momentum["selected_priority"] = 100
    momentum["momentum_selected"] = momentum["signal"].fillna(0).astype(int) != 0
    momentum["momentum_signal"] = momentum["signal"].fillna(0).astype(int)
    momentum["bear_signal"] = 0
    momentum["bull_signal"] = 0
    momentum["bear_only"] = False
    momentum["bull_reclaim"] = False

    micro_ctx = v9e.load_range_footprint_context(args, load_start_str, args.end_date)
    momentum = v9e.apply_micro_context_filter(momentum, micro_ctx, args)
    momentum = momentum.loc[trade_start: pd.Timestamp(args.end_date)].copy().sort_index()
    print(f"Momentum feature rows after warmup slice: {len(momentum):,}; first={momentum.index[0]}", flush=True)
    return momentum, mom_cfg, exec_cfg


def build_signal_table(features: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    horizons = [int(x.strip()) for x in str(args.horizons).split(",") if x.strip()]
    out = features.loc[_num_series(features, "signal", 0).astype(int) != 0].copy()
    out["timestamp"] = out.index
    out["year"] = out.index.year.astype(int)
    out["side"] = _num_series(out, "signal", 0).astype(int)
    out["side_name"] = out["side"].map(lambda x: _side_name(int(x)))

    # Regime dimensions.
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
    out["micro_context_available_label"] = np.where(_bool_series(out, "micro_context_available"), "CTX_AVAILABLE", "CTX_MISSING")
    out["micro_aligned_label"] = np.where(_bool_series(out, "micro_aligned"), "MICRO_ALIGNED", "MICRO_NOT_ALIGNED")
    out["micro_contra_label"] = np.where(_bool_series(out, "micro_contra"), "MICRO_CONTRA", "MICRO_NOT_CONTRA")
    out["long_quality_full_label"] = np.where(_bool_series(out, "long_quality_full"), "LONG_QUALITY_FULL", "NOT_LONG_QUALITY_FULL")
    out["long_quality_weak_label"] = np.where(_bool_series(out, "long_quality_weak"), "LONG_QUALITY_WEAK", "NOT_LONG_QUALITY_WEAK")
    out["long_mature_breakout_label"] = np.where(_bool_series(out, "long_mature_breakout"), "LONG_MATURE", "NOT_LONG_MATURE")

    # Forward labels.
    sig_full = _num_series(features, "signal", 0).astype(int)
    for h in horizons:
        fwd = _signed_forward_return(features["close"], sig_full, h).reindex(out.index)
        out[f"fwd_{h}bar_ret_pct"] = fwd * 100.0
        out[f"fwd_{h}bar_win"] = fwd > 0
    mfe, mae = _forward_mfe_mae(features, sig_full, int(args.mfe_horizon))
    out[f"mfe_{args.mfe_horizon}bar_pct"] = mfe.reindex(out.index) * 100.0
    out[f"mae_{args.mfe_horizon}bar_pct"] = mae.reindex(out.index) * 100.0

    # Executable next-open one-bar return is useful for immediate signal quality.
    next_open = features["open"].shift(-1)
    next_close = features["close"].shift(-1)
    out["next_bar_open_to_close_ret_pct"] = ((next_close / next_open - 1.0) * sig_full).reindex(out.index) * 100.0

    keep = [
        "timestamp", "year", "side", "side_name", "open", "high", "low", "close", "volume",
        "adx", "adx_q", "atr_pct", "atr_pct_q", "d1_distance", "d1_distance_abs_q",
        "risk_mult", "risk_mult_q", "quality_mult", "quality_mult_q", "volume_ratio", "volume_ratio_q",
        "micro_filter_action", "micro_context_available_label", "micro_aligned_label", "micro_contra_label",
        "rf_bar_count", "rf_imbalance", "rf_imbalance_q", "rf_close_pos", "rf_close_pos_q",
        "rf_taker_buy_ratio", "rf_taker_buy_ratio_q", "rf_micro_return_pct",
        "long_quality_full_label", "long_quality_weak_label", "long_mature_breakout_label",
        "next_bar_open_to_close_ret_pct", f"mfe_{args.mfe_horizon}bar_pct", f"mae_{args.mfe_horizon}bar_pct",
    ]
    for h in horizons:
        keep += [f"fwd_{h}bar_ret_pct", f"fwd_{h}bar_win"]
    return out[[c for c in keep if c in out.columns]].reset_index(drop=True)


def _pf_from_returns(ret: pd.Series) -> float:
    x = pd.to_numeric(ret, errors="coerce").dropna()
    if x.empty:
        return np.nan
    gp = float(x[x > 0].sum())
    gl = float(-x[x <= 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else np.nan
    return gp / gl


def summarize_groups(signal_table: pd.DataFrame, group_cols: list[str], horizon: int, min_count: int) -> pd.DataFrame:
    ret_col = f"fwd_{horizon}bar_ret_pct"
    win_col = f"fwd_{horizon}bar_win"
    rows: list[dict[str, Any]] = []
    cols = [c for c in group_cols if c in signal_table.columns]
    if not cols or ret_col not in signal_table.columns:
        return pd.DataFrame()
    for keys, g in signal_table.groupby(cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = int(g[ret_col].notna().sum())
        if n < int(min_count):
            continue
        ret = pd.to_numeric(g[ret_col], errors="coerce")
        years = sorted(pd.to_numeric(g["year"], errors="coerce").dropna().astype(int).unique().tolist()) if "year" in g else []
        row: dict[str, Any] = {
            "group_cols": "+".join(cols),
            "group_key": " | ".join(f"{c}={v}" for c, v in zip(cols, keys)),
            "count": n,
            "year_count": int(len(years)),
            "years": ";".join(map(str, years)),
            "win_rate": float(pd.to_numeric(g[win_col], errors="coerce").fillna(False).mean() * 100.0),
            "avg_ret_pct": float(ret.mean()),
            "median_ret_pct": float(ret.median()),
            "profit_factor": _pf_from_returns(ret),
            "avg_mfe_pct": float(pd.to_numeric(g.get("mfe_12bar_pct", pd.Series(np.nan, index=g.index)), errors="coerce").mean()),
            "avg_mae_pct": float(pd.to_numeric(g.get("mae_12bar_pct", pd.Series(np.nan, index=g.index)), errors="coerce").mean()),
        }
        # Check yearly consistency inside this group.
        yearly = []
        for y, gy in g.groupby("year"):
            yret = pd.to_numeric(gy[ret_col], errors="coerce")
            yearly.append({"year": int(y), "count": int(len(gy)), "win_rate": float(pd.to_numeric(gy[win_col], errors="coerce").fillna(False).mean() * 100.0), "avg_ret_pct": float(yret.mean())})
        positive_years = sum(1 for item in yearly if item["avg_ret_pct"] > 0)
        row["positive_year_count"] = int(positive_years)
        row["negative_year_count"] = int(len(yearly) - positive_years)
        row["min_year_avg_ret_pct"] = float(min([item["avg_ret_pct"] for item in yearly], default=np.nan))
        row["max_single_year_share"] = float(max([item["count"] for item in yearly], default=0) / max(n, 1))
        row["yearly_detail"] = json.dumps(yearly, ensure_ascii=False)
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["score"] = (
        out["avg_ret_pct"].fillna(0.0)
        + 0.05 * out["win_rate"].fillna(0.0)
        + 0.20 * out["positive_year_count"].fillna(0.0)
        - 0.50 * out["negative_year_count"].fillna(0.0)
        - 2.0 * out["max_single_year_share"].fillna(0.0)
    )
    return out.sort_values(["score", "avg_ret_pct", "win_rate"], ascending=[False, False, False]).reset_index(drop=True)


def build_group_stats(signal_table: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    h = int(args.candidate_horizon)
    group_sets = [
        ["side_name"],
        ["side_name", "year"],
        ["side_name", "adx_q"],
        ["side_name", "atr_pct_q"],
        ["side_name", "d1_distance_abs_q"],
        ["side_name", "quality_mult_q"],
        ["side_name", "risk_mult_q"],
        ["side_name", "volume_ratio_q"],
        ["side_name", "micro_filter_action"],
        ["side_name", "micro_aligned_label"],
        ["side_name", "micro_contra_label"],
        ["side_name", "rf_imbalance_q"],
        ["side_name", "rf_close_pos_q"],
        ["side_name", "rf_taker_buy_ratio_q"],
        ["side_name", "long_quality_full_label"],
        ["side_name", "long_quality_weak_label"],
        ["side_name", "long_mature_breakout_label"],
        ["side_name", "adx_q", "atr_pct_q"],
        ["side_name", "adx_q", "micro_filter_action"],
        ["side_name", "atr_pct_q", "micro_filter_action"],
        ["side_name", "adx_q", "d1_distance_abs_q"],
        ["side_name", "quality_mult_q", "micro_filter_action"],
        ["side_name", "risk_mult_q", "micro_filter_action"],
        ["side_name", "rf_imbalance_q", "rf_close_pos_q"],
    ]
    frames = [summarize_groups(signal_table, cols, h, int(args.min_count)) for cols in group_sets]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True).sort_values(["score", "avg_ret_pct"], ascending=[False, False]).reset_index(drop=True) if frames else pd.DataFrame()


def candidate_regimes(group_stats: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    if group_stats.empty:
        return pd.DataFrame(), pd.DataFrame()
    stable = group_stats[
        (pd.to_numeric(group_stats["count"], errors="coerce") >= int(args.min_count))
        & (pd.to_numeric(group_stats["year_count"], errors="coerce") >= int(args.min_years))
        & (pd.to_numeric(group_stats["max_single_year_share"], errors="coerce") <= 0.70)
    ].copy()
    good = stable[
        (pd.to_numeric(stable["avg_ret_pct"], errors="coerce") > 0)
        & (pd.to_numeric(stable["win_rate"], errors="coerce") >= 45.0)
        & (pd.to_numeric(stable["positive_year_count"], errors="coerce") >= np.maximum(1, pd.to_numeric(stable["year_count"], errors="coerce") - 1))
    ].copy()
    bad = stable[
        (pd.to_numeric(stable["avg_ret_pct"], errors="coerce") < 0)
        & (pd.to_numeric(stable["win_rate"], errors="coerce") <= 40.0)
    ].copy()
    return (
        good.sort_values(["score", "avg_ret_pct", "win_rate"], ascending=[False, False, False]).reset_index(drop=True),
        bad.sort_values(["avg_ret_pct", "win_rate"], ascending=[True, True]).reset_index(drop=True),
    )


def _closed_metrics(trades: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    if not trades:
        return {"closed_final_capital": initial_capital, "closed_total_trades": 0, "closed_win_rate": 0.0, "closed_profit_factor": 0.0, "force_close_count": 0, "force_close_pnl": 0.0}
    tdf = pd.DataFrame(trades).copy()
    note = tdf.get("note", pd.Series("", index=tdf.index)).astype(str)
    force = note.eq("FORCE_CLOSE_END")
    force_pnl = float(pd.to_numeric(tdf.loc[force, "pnl"], errors="coerce").fillna(0.0).sum()) if "pnl" in tdf.columns else 0.0
    closed = tdf.loc[~force].copy()
    if closed.empty:
        return {"closed_final_capital": initial_capital, "closed_total_trades": 0, "closed_win_rate": 0.0, "closed_profit_factor": 0.0, "force_close_count": int(force.sum()), "force_close_pnl": force_pnl}
    wins = closed[closed["pnl"] > 0]
    losses = closed[closed["pnl"] <= 0]
    gross_profit = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(-losses["pnl"].sum()) if not losses.empty else 0.0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return {
        "closed_final_capital": float(closed.iloc[-1]["capital"]),
        "closed_total_trades": int(len(closed)),
        "closed_win_rate": float((closed["pnl"] > 0).mean() * 100.0),
        "closed_profit_factor": pf,
        "closed_expectancy_pct": float(closed["return_pct"].mean() * 100.0),
        "closed_avg_win_pct": float(wins["return_pct"].mean() * 100.0) if not wins.empty else 0.0,
        "closed_avg_loss_pct": float(losses["return_pct"].mean() * 100.0) if not losses.empty else 0.0,
        "force_close_count": int(force.sum()),
        "force_close_pnl": force_pnl,
    }


def run_standalone_momentum(features: pd.DataFrame, exec_cfg: Any, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    scenarios = {
        "momentum_all": features.copy(),
        "momentum_long_only": features.assign(signal=np.where(_num_series(features, "signal", 0).astype(int) == 1, 1, 0)),
        "momentum_short_only": features.assign(signal=np.where(_num_series(features, "signal", 0).astype(int) == -1, -1, 0)),
    }
    for name, feat in scenarios.items():
        feat = feat.copy()
        feat["long_signal"] = _num_series(feat, "signal", 0).astype(int) == 1
        feat["short_signal"] = _num_series(feat, "signal", 0).astype(int) == -1
        feat["selected_engine"] = np.where(_num_series(feat, "signal", 0).astype(int) != 0, ENGINE, "NONE")
        trades, equity = v9e.run_priority_backtest(feat, exec_cfg, {ENGINE: exec_cfg}, global_risk_scale=float(args.global_risk_scale), args=args)
        summary = v9e.summarize(trades, equity, float(args.initial_capital))
        closed = _closed_metrics(trades, float(args.initial_capital))
        row = {"scenario": name}
        row.update(summary)
        row.update(closed)
        if not equity.empty and "drawdown_pct" in equity.columns:
            row["max_drawdown_pct"] = float(pd.to_numeric(equity["drawdown_pct"], errors="coerce").fillna(0.0).max() * 100.0)
        rows.append(row)
        for t in trades:
            item = dict(t)
            item["scenario"] = name
            signal_time = pd.Timestamp(item["entry_time"]) - pd.Timedelta(hours=4)
            item["signal_time"] = signal_time
            if signal_time in feat.index:
                sigrow = feat.loc[signal_time]
                item["signal_year"] = int(signal_time.year)
                item["signal_side_name"] = _side_name(int(sigrow.get("signal", 0)))
                item["adx_q"] = str(sigrow.get("adx_q", "NA")) if "adx_q" in feat.columns else "NA"
                item["atr_pct_q"] = str(sigrow.get("atr_pct_q", "NA")) if "atr_pct_q" in feat.columns else "NA"
                item["micro_filter_action"] = str(sigrow.get("micro_filter_action", "NA"))
                item["micro_aligned"] = bool(sigrow.get("micro_aligned", False))
                item["quality_mult"] = _safe_float(sigrow.get("quality_mult", np.nan))
                item["risk_mult"] = _safe_float(sigrow.get("risk_mult", np.nan))
            trade_rows.append(item)
    return pd.DataFrame(rows), pd.DataFrame(trade_rows)


def trade_regime_stats(trades: pd.DataFrame, min_count: int = 3) -> pd.DataFrame:
    if trades.empty or "return_pct" not in trades.columns:
        return pd.DataFrame()
    t = trades.loc[trades["scenario"] == "momentum_all"].copy()
    if t.empty:
        return pd.DataFrame()
    # Exclude force-close from closed-trade regime stats.
    t = t.loc[t.get("note", pd.Series("", index=t.index)).astype(str) != "FORCE_CLOSE_END"].copy()
    if t.empty:
        return pd.DataFrame()
    q = lambda s, prefix: _qcut_label(pd.to_numeric(s, errors="coerce"), 4, prefix)
    t["quality_mult_q"] = q(t.get("quality_mult", pd.Series(np.nan, index=t.index)), "QUALITY_Q")
    t["risk_mult_q"] = q(t.get("risk_mult", pd.Series(np.nan, index=t.index)), "RISK_Q")
    group_sets = [
        ["signal_side_name"],
        ["signal_side_name", "signal_year"],
        ["signal_side_name", "adx_q"],
        ["signal_side_name", "atr_pct_q"],
        ["signal_side_name", "micro_filter_action"],
        ["signal_side_name", "quality_mult_q"],
        ["signal_side_name", "risk_mult_q"],
    ]
    rows = []
    for cols in group_sets:
        cols = [c for c in cols if c in t.columns]
        if not cols:
            continue
        for keys, g in t.groupby(cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            n = len(g)
            if n < min_count:
                continue
            ret = pd.to_numeric(g["return_pct"], errors="coerce") * 100.0
            rows.append({
                "group_cols": "+".join(cols),
                "group_key": " | ".join(f"{c}={v}" for c, v in zip(cols, keys)),
                "count": int(n),
                "win_rate": float((pd.to_numeric(g["pnl"], errors="coerce") > 0).mean() * 100.0) if "pnl" in g else float((ret > 0).mean() * 100.0),
                "avg_return_pct": float(ret.mean()),
                "median_return_pct": float(ret.median()),
                "profit_factor_ret_pct": _pf_from_returns(ret),
                "avg_mfe_r": float(pd.to_numeric(g.get("mfe_r", pd.Series(np.nan, index=g.index)), errors="coerce").mean()),
                "avg_mae_r": float(pd.to_numeric(g.get("mae_r", pd.Series(np.nan, index=g.index)), errors="coerce").mean()),
                "years": ";".join(map(str, sorted(pd.to_numeric(g.get("signal_year", pd.Series([], dtype=float)), errors="coerce").dropna().astype(int).unique().tolist()))),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["profit_factor_ret_pct", "avg_return_pct", "win_rate"], ascending=[False, False, False]).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    features, mom_cfg, exec_cfg = build_momentum_features(args)
    signal_table = build_signal_table(features, args)
    signal_table.to_csv(out_dir / "v9e_momentum_signal_regime_table.csv", index=False, encoding="utf-8-sig")

    group_stats = build_group_stats(signal_table, args)
    group_stats.to_csv(out_dir / "v9e_momentum_signal_regime_group_stats.csv", index=False, encoding="utf-8-sig")

    good, bad = candidate_regimes(group_stats, args)
    good.to_csv(out_dir / "v9e_momentum_good_regime_candidates.csv", index=False, encoding="utf-8-sig")
    bad.to_csv(out_dir / "v9e_momentum_bad_regime_candidates.csv", index=False, encoding="utf-8-sig")

    overview_rows = []
    for side_name, g in signal_table.groupby("side_name"):
        overview_rows.append({
            "side_name": side_name,
            "signal_count": int(len(g)),
            "year_count": int(g["year"].nunique()),
            "fwd_12bar_win_rate": float(g.get("fwd_12bar_win", pd.Series(False, index=g.index)).mean() * 100.0),
            "fwd_12bar_avg_ret_pct": float(pd.to_numeric(g.get("fwd_12bar_ret_pct", pd.Series(np.nan, index=g.index)), errors="coerce").mean()),
            "fwd_6bar_win_rate": float(g.get("fwd_6bar_win", pd.Series(False, index=g.index)).mean() * 100.0) if "fwd_6bar_win" in g else np.nan,
            "fwd_6bar_avg_ret_pct": float(pd.to_numeric(g.get("fwd_6bar_ret_pct", pd.Series(np.nan, index=g.index)), errors="coerce").mean()) if "fwd_6bar_ret_pct" in g else np.nan,
            "avg_mfe_12bar_pct": float(pd.to_numeric(g.get("mfe_12bar_pct", pd.Series(np.nan, index=g.index)), errors="coerce").mean()),
            "avg_mae_12bar_pct": float(pd.to_numeric(g.get("mae_12bar_pct", pd.Series(np.nan, index=g.index)), errors="coerce").mean()),
        })
    overview = pd.DataFrame(overview_rows).sort_values("signal_count", ascending=False)
    overview.to_csv(out_dir / "v9e_momentum_side_overview.csv", index=False, encoding="utf-8-sig")

    if not args.skip_standalone_trades:
        # Add quantile labels to features for trade join.
        features = features.copy()
        features["adx_q"] = _qcut_label(_num_series(features, "adx"), 4, "ADX_Q")
        features["atr_pct_q"] = _qcut_label(_num_series(features, "atr_pct"), 4, "ATR_Q")
        summary, trades = run_standalone_momentum(features, exec_cfg, args)
        summary.to_csv(out_dir / "v9e_momentum_standalone_summary.csv", index=False, encoding="utf-8-sig")
        trades.to_csv(out_dir / "v9e_momentum_standalone_trades.csv", index=False, encoding="utf-8-sig")
        tstats = trade_regime_stats(trades, min_count=max(3, int(args.min_count // 2)))
        tstats.to_csv(out_dir / "v9e_momentum_trade_regime_stats.csv", index=False, encoding="utf-8-sig")

    meta = {
        "script": "v9e_momentum_regime_lab.py",
        "symbol": args.symbol,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "signal_count": int(len(signal_table)),
        "candidate_horizon": int(args.candidate_horizon),
        "min_count": int(args.min_count),
        "min_years": int(args.min_years),
        "outputs": [
            "v9e_momentum_signal_regime_table.csv",
            "v9e_momentum_signal_regime_group_stats.csv",
            "v9e_momentum_good_regime_candidates.csv",
            "v9e_momentum_bad_regime_candidates.csv",
            "v9e_momentum_side_overview.csv",
            "v9e_momentum_standalone_summary.csv",
            "v9e_momentum_trade_regime_stats.csv",
        ],
    }
    (out_dir / "v9e_momentum_regime_lab_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Momentum Regime Lab Done ===", flush=True)
    print(f"Output dir: {out_dir}", flush=True)
    print("Key files:", flush=True)
    for name in meta["outputs"]:
        print(f"  - {name}", flush=True)
    if not good.empty:
        print("\nTop good candidates:", flush=True)
        print(good.head(10)[["group_cols", "group_key", "count", "year_count", "win_rate", "avg_ret_pct", "profit_factor", "score"]].to_string(index=False), flush=True)
    else:
        print("\nNo stable good candidate found under current thresholds.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
