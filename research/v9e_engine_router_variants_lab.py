#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V9E Engine-Aware Router Variants Lab
====================================

Research-only probe for ETH_LF_Portfolio_V9E_RangeExitOverlay.

Purpose:
    Test whether MOMENTUM_V3 should remain a normal entry engine, or be used as:
    - same-bar confirmation for BULL/BEAR entries
    - reduced-risk entry engine
    - short-only entry engine
    - in-position short trend continuation add-on signal

This script does NOT modify the V9E strategy file and does NOT place orders.
All variants keep V9E's closed-bar -> next-open timing and range-exit overlay.
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

ENGINE_MOM = "MOMENTUM_V3"
ENGINE_BEAR = "BEAR_V3_ONLY"
ENGINE_BULL = "BULL_RECLAIM_V2"
ENGINES = (ENGINE_BULL, ENGINE_BEAR, ENGINE_MOM)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V9E engine-aware router variants research lab.")

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

    # Baseline-compatible parameters.
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

    p.add_argument("--range-exit-mode", choices=["off", "soft"], default="soft")
    p.add_argument("--range-exit-min-mfe-r", type=float, default=2.0)
    p.add_argument("--range-exit-giveback-frac", type=float, default=0.65)
    p.add_argument("--range-exit-min-hold-bars", type=int, default=2)
    p.add_argument("--range-exit-delay-bars", type=int, default=0)
    p.add_argument("--range-exit-contra-imbalance", type=float, default=0.05)
    p.add_argument("--range-exit-bad-close-pos", type=float, default=0.35)
    p.add_argument("--range-exit-no-reversal-required", dest="range_exit_require_reversal", action="store_false")
    p.set_defaults(range_exit_require_reversal=True)

    # Research controls.
    p.add_argument("--out-dir", default="data/reports/research/v9e_engine_router_variants_lab")
    p.add_argument("--write-trades", action="store_true", help="Write per-variant trades/equity. Default writes only summaries for speed/space.")
    p.add_argument("--addon-min-current-r", type=float, default=1.5)
    p.add_argument("--addon-risk-scale", type=float, default=0.25)
    p.add_argument("--addon-max-count", type=int, default=1)
    p.add_argument("--entry-confirm-risk-boost", type=float, default=1.25)
    p.add_argument("--momentum-risk-down-scale", type=float, default=0.50)
    return p.parse_args()


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(float(default), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(float(default))


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


def _side_name(side: int) -> str:
    if int(side) == 1:
        return "LONG"
    if int(side) == -1:
        return "SHORT"
    return "FLAT"


def _note_series(tdf: pd.DataFrame) -> pd.Series:
    if "note" not in tdf.columns:
        return pd.Series("", index=tdf.index)
    return tdf["note"].astype(str)


def closed_metrics(trades: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    if not trades:
        return {
            "closed_final_capital": float(initial_capital),
            "closed_total_return_pct": 0.0,
            "closed_total_trades": 0,
            "closed_win_rate": 0.0,
            "closed_profit_factor": 0.0,
            "closed_expectancy_pct": 0.0,
            "closed_avg_win_pct": 0.0,
            "closed_avg_loss_pct": 0.0,
            "force_close_count": 0,
            "force_close_pnl": 0.0,
        }
    tdf = pd.DataFrame(trades).copy()
    notes = _note_series(tdf)
    force = notes.eq("FORCE_CLOSE_END")
    force_pnl = float(pd.to_numeric(tdf.loc[force, "pnl"], errors="coerce").fillna(0.0).sum()) if "pnl" in tdf.columns else 0.0
    closed = tdf.loc[~force].copy()
    if closed.empty:
        return {
            "closed_final_capital": float(initial_capital),
            "closed_total_return_pct": 0.0,
            "closed_total_trades": 0,
            "closed_win_rate": 0.0,
            "closed_profit_factor": 0.0,
            "closed_expectancy_pct": 0.0,
            "closed_avg_win_pct": 0.0,
            "closed_avg_loss_pct": 0.0,
            "force_close_count": int(force.sum()),
            "force_close_pnl": force_pnl,
        }
    pnl = pd.to_numeric(closed["pnl"], errors="coerce").fillna(0.0)
    ret = pd.to_numeric(closed["return_pct"], errors="coerce").fillna(0.0)
    wins = pnl > 0
    gross_profit = float(pnl[wins].sum())
    gross_loss = float(-pnl[~wins].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    final_cap = float(pd.to_numeric(closed["capital"], errors="coerce").iloc[-1])
    return {
        "closed_final_capital": final_cap,
        "closed_total_return_pct": (final_cap / float(initial_capital) - 1.0) * 100.0,
        "closed_total_trades": int(len(closed)),
        "closed_win_rate": float(wins.mean() * 100.0),
        "closed_profit_factor": pf,
        "closed_expectancy_pct": float(ret.mean() * 100.0),
        "closed_avg_win_pct": float(ret[wins].mean() * 100.0) if bool(wins.any()) else 0.0,
        "closed_avg_loss_pct": float(ret[~wins].mean() * 100.0) if bool((~wins).any()) else 0.0,
        "force_close_count": int(force.sum()),
        "force_close_pnl": force_pnl,
    }


def summarize_run(name: str, trades: list[dict[str, Any]], equity: pd.DataFrame, cfg: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    base = v9e.summarize(trades, equity, cfg.initial_capital)
    out: dict[str, Any] = {"scenario": name}
    out.update(base)
    out.update(closed_metrics(trades, cfg.initial_capital))
    if not equity.empty and "drawdown_pct" in equity.columns:
        out["max_drawdown_pct"] = float(pd.to_numeric(equity["drawdown_pct"], errors="coerce").fillna(0.0).max() * 100.0)
    if trades:
        tdf = pd.DataFrame(trades)
        out["total_fees"] = float(pd.to_numeric(tdf.get("fee", pd.Series(0.0, index=tdf.index)), errors="coerce").fillna(0.0).sum())
        out["avg_holding_hours"] = float(pd.to_numeric(tdf.get("holding_hours", pd.Series(0.0, index=tdf.index)), errors="coerce").fillna(0.0).mean())
        out["avg_mfe_r"] = float(pd.to_numeric(tdf.get("mfe_r", pd.Series(0.0, index=tdf.index)), errors="coerce").fillna(0.0).mean())
        out["avg_mae_r"] = float(pd.to_numeric(tdf.get("mae_r", pd.Series(0.0, index=tdf.index)), errors="coerce").fillna(0.0).mean())
        out["signal_addon_trade_count"] = int(pd.to_numeric(tdf.get("router_addon_count", pd.Series(0, index=tdf.index)), errors="coerce").fillna(0).gt(0).sum())
        out["signal_addon_total_count"] = int(pd.to_numeric(tdf.get("router_addon_count", pd.Series(0, index=tdf.index)), errors="coerce").fillna(0).sum())
    else:
        out["signal_addon_trade_count"] = 0
        out["signal_addon_total_count"] = 0
    if extra:
        out.update(extra)
    return out


def yearly_metrics(trades: list[dict[str, Any]], equity: pd.DataFrame, scenario: str, initial_capital: float) -> pd.DataFrame:
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
        notes = _note_series(tdf)
        tdf_closed = tdf.loc[~notes.eq("FORCE_CLOSE_END")].copy()
    else:
        tdf_closed = pd.DataFrame()
    for year, g in eq.groupby("year"):
        start_cap = float(g["capital"].iloc[0])
        end_cap = float(g["capital"].iloc[-1])
        ytrades = tdf_closed[tdf_closed["year"].eq(year)] if not tdf_closed.empty else pd.DataFrame()
        rows.append({
            "scenario": scenario,
            "year": int(year),
            "start_capital": start_cap,
            "end_capital": end_cap,
            "year_return_pct": (end_cap / max(start_cap, 1e-12) - 1.0) * 100.0,
            "max_drawdown_pct": float(pd.to_numeric(g.get("drawdown_pct", pd.Series(0.0, index=g.index)), errors="coerce").fillna(0.0).max() * 100.0),
            "closed_trade_count": int(len(ytrades)),
            "closed_win_rate": float((pd.to_numeric(ytrades.get("pnl", pd.Series(dtype=float)), errors="coerce") > 0).mean() * 100.0) if len(ytrades) else 0.0,
            "closed_pnl": float(pd.to_numeric(ytrades.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()) if len(ytrades) else 0.0,
        })
    return pd.DataFrame(rows)


def build_features(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], Any, dict[str, Any]]:
    mom_cfg = v9e.make_momentum_config(args)
    bear_cfg = v9e.make_bear_config(args)
    bull_cfg = v9e.make_bull_config(args)
    exec_cfg = v9e.make_exec_config(mom_cfg)
    bull_exec_cfg = v9e.bull_to_exec_config(bull_cfg) if args.bull_execution_mode == "own" else exec_cfg

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
    print(f"Loaded {len(base)} rows: {base.index[0]} -> {base.index[-1]}", flush=True)

    momentum = v9e.build_momentum_features(base, mom_cfg)
    bear = v9e.build_bear_features(base, bear_cfg)
    bull = v9e.build_bull_features(base, bull_cfg)

    baseline_selected = v9e.select_portfolio_signals(momentum, bear, bull, args)
    micro_ctx = v9e.load_range_footprint_context(args, load_start_str, args.end_date)
    baseline = v9e.apply_micro_context_filter(baseline_selected, micro_ctx, args)
    baseline = baseline.loc[trade_start: pd.Timestamp(args.end_date)].copy().sort_index()

    raw = {
        ENGINE_MOM: momentum,
        ENGINE_BEAR: bear,
        ENGINE_BULL: bull,
    }
    engine_cfgs = {ENGINE_MOM: exec_cfg, ENGINE_BEAR: exec_cfg, ENGINE_BULL: bull_exec_cfg}
    print(f"Feature rows after warmup slice: {len(baseline)}; first={baseline.index[0] if len(baseline) else 'NA'}", flush=True)
    return baseline, raw, exec_cfg, engine_cfgs


def apply_micro_to_variant(df: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    """Copy already-computed micro context columns from baseline to avoid reloading range DB."""
    out = df.copy()
    micro_cols = [
        "rf_bar_count", "rf_micro_return_pct", "rf_close_pos", "rf_delta_sum", "rf_imbalance", "rf_taker_buy_ratio",
        "rf_max_sell_bucket_share", "rf_max_buy_bucket_share", "micro_context_available", "micro_aligned", "micro_contra",
        "micro_entry_risk_scale", "micro_filter_action",
    ]
    for col in micro_cols:
        if col in baseline.columns:
            out[col] = baseline[col].reindex(out.index)
    # Micro flags depend on final signal side. Recompute aligned/contra tags for final signal using copied context.
    sig = out["signal"].fillna(0).astype(int)
    if "rf_imbalance" in out.columns and "rf_close_pos" in out.columns and "micro_context_available" in out.columns:
        has_ctx = out["micro_context_available"].astype("boolean").fillna(False).astype(bool)
        imb = pd.to_numeric(out["rf_imbalance"], errors="coerce")
        pos = pd.to_numeric(out["rf_close_pos"], errors="coerce")
        long_sig = sig.eq(1)
        short_sig = sig.eq(-1)
        # Use V9E default thresholds because args are not available here; exact risk scale was already copied, only tags are for analysis/add-on filters.
        out["micro_aligned"] = ((long_sig & has_ctx & (imb >= 0.05) & (pos >= 0.65)) | (short_sig & has_ctx & (imb <= -0.05) & (pos <= 0.35))).astype(bool)
        out["micro_contra"] = ((long_sig & has_ctx & (imb <= -0.05) & (pos <= 0.35)) | (short_sig & has_ctx & (imb >= 0.05) & (pos >= 0.65))).astype(bool)
    return out


def make_engine_aware_router(
    baseline: pd.DataFrame,
    raw: dict[str, pd.DataFrame],
    args: argparse.Namespace,
    *,
    scenario: str,
    momentum_entry_mode: str = "all",  # all / none / short_only / long_only
    priority_order: tuple[str, ...] = (ENGINE_BULL, ENGINE_BEAR, ENGINE_MOM),
    momentum_selected_risk_scale: float = 1.0,
    momentum_long_selected_risk_scale: float = 1.0,
    momentum_short_selected_risk_scale: float = 1.0,
    confirm_boost_enabled: bool = False,
    confirm_boost_scale: float = 1.0,
    require_momentum_confirm_for_bull_bear: bool = False,
    no_confirm_risk_scale: float = 1.0,
) -> pd.DataFrame:
    out = baseline.copy()
    mom = raw[ENGINE_MOM].reindex(out.index)
    bear = raw[ENGINE_BEAR].reindex(out.index)
    bull = raw[ENGINE_BULL].reindex(out.index)

    out["momentum_signal"] = _num(mom, "signal", 0.0).astype(int)
    out["bear_signal"] = _num(bear, "signal", 0.0).astype(int)
    out["bull_signal"] = _num(bull, "signal", 0.0).astype(int)
    out["momentum_long_exit_channel"] = _bool(mom, "long_exit_channel")
    out["momentum_short_exit_channel"] = _bool(mom, "short_exit_channel")
    out["bear_short_exit_channel"] = _bool(bear, "short_exit_channel")
    out["bull_long_exit_channel"] = _bool(bull, "long_exit_channel")

    mom_active = out["momentum_signal"].ne(0)
    if momentum_entry_mode == "none":
        mom_active = pd.Series(False, index=out.index)
    elif momentum_entry_mode == "short_only":
        mom_active = out["momentum_signal"].eq(-1)
    elif momentum_entry_mode == "long_only":
        mom_active = out["momentum_signal"].eq(1)
    elif momentum_entry_mode != "all":
        raise ValueError(f"Unsupported momentum_entry_mode={momentum_entry_mode}")

    bear_active = out["bear_signal"].eq(-1) & (not args.disable_bear_standalone)
    bull_active = out["bull_signal"].eq(1) & (not args.disable_bull_reclaim)

    same_side_mom_for_bull = out["momentum_signal"].eq(1) & bull_active
    same_side_mom_for_bear = out["momentum_signal"].eq(-1) & bear_active
    if require_momentum_confirm_for_bull_bear:
        bull_active = bull_active & same_side_mom_for_bull
        bear_active = bear_active & same_side_mom_for_bear

    candidates = {
        ENGINE_BULL: bull_active,
        ENGINE_BEAR: bear_active,
        ENGINE_MOM: mom_active,
    }
    out["signal"] = 0
    out["selected_engine"] = "NONE"
    out["selected_priority"] = 0
    out["momentum_selected"] = False
    out["bear_only"] = False
    out["bull_reclaim"] = False
    out["momentum_same_bar_confirm"] = False
    out["router_variant"] = scenario
    out["router_risk_adjustment"] = 1.0
    out["router_note"] = "NONE"
    out["portfolio_conflict"] = sum(mask.astype(int) for mask in candidates.values()) > 1

    final = pd.Series(0, index=out.index, dtype="int64")
    for rank, engine in enumerate(priority_order):
        mask = candidates[engine] & final.eq(0)
        if not bool(mask.any()):
            continue
        out.loc[mask, "selected_engine"] = engine
        out.loc[mask, "selected_priority"] = int((len(priority_order) - rank) * 50)
        if engine == ENGINE_BULL:
            final.loc[mask] = 1
            out.loc[mask, "bull_reclaim"] = True
            out.loc[mask, "risk_mult"] = (_num(bull, "risk_mult", 1.0).loc[mask] * float(args.bull_reclaim_risk_scale)).clip(args.min_risk_mult, args.max_risk_mult or 10.0)
            out.loc[mask, "quality_mult"] = (_num(bull, "quality_mult", 1.0).loc[mask] * float(args.bull_reclaim_quality_scale)).clip(0.10, args.quality_mult_cap)
        elif engine == ENGINE_BEAR:
            final.loc[mask] = -1
            out.loc[mask, "bear_only"] = True
            out.loc[mask, "risk_mult"] = (_num(bear, "risk_mult", 1.0).loc[mask] * float(args.bear_standalone_risk_scale)).clip(args.min_risk_mult, args.max_risk_mult or 10.0)
            out.loc[mask, "quality_mult"] = (_num(bear, "quality_mult", 1.0).loc[mask] * float(args.bear_standalone_quality_scale)).clip(0.20, args.quality_mult_cap)
        elif engine == ENGINE_MOM:
            final.loc[mask] = out.loc[mask, "momentum_signal"].astype(int)
            out.loc[mask, "momentum_selected"] = True
            scale = pd.Series(float(momentum_selected_risk_scale), index=out.index)
            scale.loc[mask & out["momentum_signal"].eq(1)] *= float(momentum_long_selected_risk_scale)
            scale.loc[mask & out["momentum_signal"].eq(-1)] *= float(momentum_short_selected_risk_scale)
            out.loc[mask, "risk_mult"] = (_num(out, "risk_mult", 1.0).loc[mask] * scale.loc[mask]).clip(args.min_risk_mult, args.max_risk_mult or 10.0)
            out.loc[mask, "router_risk_adjustment"] = scale.loc[mask]
            out.loc[mask, "router_note"] = f"MOMENTUM_ENTRY_RISK_SCALE_{float(momentum_selected_risk_scale):.2f}"

    out["signal"] = final.astype(int)
    out["long_signal"] = out["signal"].eq(1)
    out["short_signal"] = out["signal"].eq(-1)

    if confirm_boost_enabled or no_confirm_risk_scale != 1.0:
        selected_bull = out["selected_engine"].eq(ENGINE_BULL)
        selected_bear = out["selected_engine"].eq(ENGINE_BEAR)
        confirmed = (selected_bull & out["momentum_signal"].eq(1)) | (selected_bear & out["momentum_signal"].eq(-1))
        bb_selected = selected_bull | selected_bear
        out.loc[confirmed, "momentum_same_bar_confirm"] = True
        if no_confirm_risk_scale != 1.0:
            no_conf = bb_selected & (~confirmed)
            out.loc[no_conf, "risk_mult"] = (_num(out, "risk_mult", 1.0).loc[no_conf] * float(no_confirm_risk_scale)).clip(args.min_risk_mult, args.max_risk_mult or 10.0)
            out.loc[no_conf, "router_risk_adjustment"] = out.loc[no_conf, "router_risk_adjustment"].astype(float) * float(no_confirm_risk_scale)
            out.loc[no_conf, "router_note"] = f"NO_MOMENTUM_CONFIRM_RISK_SCALE_{float(no_confirm_risk_scale):.2f}"
        if confirm_boost_enabled and confirm_boost_scale != 1.0:
            out.loc[confirmed, "risk_mult"] = (_num(out, "risk_mult", 1.0).loc[confirmed] * float(confirm_boost_scale)).clip(args.min_risk_mult, args.max_risk_mult or 10.0)
            out.loc[confirmed, "router_risk_adjustment"] = out.loc[confirmed, "router_risk_adjustment"].astype(float) * float(confirm_boost_scale)
            out.loc[confirmed, "router_note"] = f"MOMENTUM_SAME_BAR_CONFIRM_BOOST_{float(confirm_boost_scale):.2f}"

    out.loc[out["signal"].eq(0), ["selected_engine", "router_note"]] = ["NONE", "NO_ENTRY"]
    out = apply_micro_to_variant(out, baseline)
    out["router_variant"] = scenario
    return out


def annotate_last_addon(trades: list[dict[str, Any]], count: int, risk_sum: float, last_reason: str) -> None:
    if not trades:
        return
    trades[-1]["router_addon_count"] = int(count)
    trades[-1]["router_addon_risk_scale_avg"] = float(risk_sum / count) if count > 0 else 0.0
    trades[-1]["router_last_addon_reason"] = str(last_reason)


def run_router_addon_backtest(
    df: pd.DataFrame,
    cfg: Any,
    engine_cfgs: dict[str, Any] | None,
    *,
    global_risk_scale: float,
    args: argparse.Namespace,
    addon_mode: str = "off",  # off / short_confirm / all_confirm
    addon_min_current_r: float = 1.5,
    addon_risk_scale: float = 0.25,
    addon_max_count: int = 1,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """V9E-equivalent executor plus engine-aware in-position confirmation add-on.

    Add-on probe is intentionally conservative:
        - only after current trade has positive R beyond threshold
        - max one add-on by default
        - default mode only supports SHORT confirmation from BEAR/MOMENTUM raw signals
        - add-on uses next 4H open after confirmed closed 4H bar
    """
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
    addon_count = 0
    addon_risk_sum = 0.0
    last_addon_reason = "NONE"

    rows = list(df.itertuples())
    idx = df.index

    for i in range(len(rows) - 1):
        row = rows[i]
        ts = idx[i]
        if in_pos:
            if pending_range_exit_i is not None and i >= pending_range_exit_i:
                active_cfg = pos_cfg
                hold_bars = i - entry_i
                exit_price = v9e.apply_exit_slippage(float(row.open), side, active_cfg.slippage_pct)
                exit_time = idx[i]
                reason = pending_range_exit_reason or "RANGE_EXIT_DELAYED_OPEN"
                capital = v9e.close_trade(
                    trades=trades, capital=capital, side=side, entry_time=entry_time, exit_time=exit_time,
                    first_entry=first_entry, avg_entry=avg_entry, exit_price=exit_price, initial_sl=initial_sl,
                    stop_price=stop_price, qty=qty, units=units, total_entry_fee=total_entry_fee,
                    fee_rate=active_cfg.fee_rate, max_fav=max_fav, max_adv=max_adv,
                    risk_per_coin=risk_per_coin, holding_bars=hold_bars, reason=reason, risk_mult=entry_risk_mult,
                )
                if trades:
                    trades[-1].update(pending_range_exit_meta)
                    trades[-1]["range_exit_executed_after_delay"] = True
                    annotate_last_addon(trades, addon_count, addon_risk_sum, last_addon_reason)
                peak = max(peak, capital)
                in_pos = False
                side = 0
                last_exit_i = i
                pending_range_exit_i = None
                pending_range_exit_reason = ""
                pending_range_exit_meta = {}
                addon_count = 0
                addon_risk_sum = 0.0
                last_addon_reason = "NONE"
            else:
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
                    channel_exit = v9e._entry_exit_channel(row, entry_engine, side)
                    opposite = current_signal == -1
                    next_stop = max(stop_price, close - active_cfg.trailing_atr_mult * atr_value)
                    locked = v9e.protected_stop(first_entry, avg_entry, side, risk_per_coin, max_fav, active_cfg)
                    if locked is not None:
                        next_stop = max(next_stop, locked)
                    current_r = (close - avg_entry) / risk_per_coin if risk_per_coin > 0 else float("nan")
                else:
                    max_fav = min(max_fav, low)
                    max_adv = max(max_adv, high)
                    touched_stop = high >= active_stop
                    channel_exit = v9e._entry_exit_channel(row, entry_engine, side)
                    opposite = current_signal == 1
                    next_stop = min(stop_price, close + active_cfg.trailing_atr_mult * atr_value)
                    locked = v9e.protected_stop(first_entry, avg_entry, side, risk_per_coin, max_fav, active_cfg)
                    if locked is not None:
                        next_stop = min(next_stop, locked)
                    current_r = (avg_entry - close) / risk_per_coin if risk_per_coin > 0 else float("nan")

                range_exit_now, range_exit_reason, range_exit_meta = v9e._range_exit_signal(
                    row, side=side, avg_entry=avg_entry, risk_per_coin=risk_per_coin,
                    max_fav=max_fav, hold_bars=hold_bars, args=args,
                )

                exit_now = False
                reason = ""
                exit_price = 0.0
                exit_time = ts
                if touched_stop:
                    exit_now = True
                    exit_price = v9e.apply_exit_slippage(active_stop, side, active_cfg.slippage_pct)
                    reason = "PROTECTED_TRAILING_STOP"
                elif channel_exit:
                    exit_now = True
                    exit_price = v9e.apply_exit_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                    exit_time = idx[i + 1]
                    reason = "DONCHIAN_EXIT_NEXT_OPEN"
                elif opposite:
                    exit_now = True
                    exit_price = v9e.apply_exit_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                    exit_time = idx[i + 1]
                    reason = "OPPOSITE_BREAKOUT_NEXT_OPEN"
                elif range_exit_now:
                    delay_bars = int(getattr(args, "range_exit_delay_bars", 0) or 0)
                    if delay_bars <= 0:
                        exit_now = True
                        exit_price = v9e.apply_exit_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
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
                    exit_price = v9e.apply_exit_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                    exit_time = idx[i + 1]
                    reason = "MAX_HOLD_EXIT_NEXT_OPEN"

                if exit_now:
                    capital = v9e.close_trade(
                        trades=trades, capital=capital, side=side, entry_time=entry_time, exit_time=exit_time,
                        first_entry=first_entry, avg_entry=avg_entry, exit_price=exit_price, initial_sl=initial_sl,
                        stop_price=stop_price, qty=qty, units=units, total_entry_fee=total_entry_fee,
                        fee_rate=active_cfg.fee_rate, max_fav=max_fav, max_adv=max_adv,
                        risk_per_coin=risk_per_coin, holding_bars=hold_bars, reason=reason, risk_mult=entry_risk_mult,
                    )
                    if trades and str(reason).startswith("RANGE_EXIT"):
                        trades[-1].update(range_exit_meta)
                    annotate_last_addon(trades, addon_count, addon_risk_sum, last_addon_reason)
                    peak = max(peak, capital)
                    in_pos = False
                    side = 0
                    last_exit_i = i
                    pending_range_exit_i = None
                    pending_range_exit_reason = ""
                    pending_range_exit_meta = {}
                    addon_count = 0
                    addon_risk_sum = 0.0
                    last_addon_reason = "NONE"
                else:
                    stop_price = next_stop

                added_this_bar = False
                if in_pos and addon_mode != "off" and pending_range_exit_i is None and units < active_cfg.max_units and addon_count < int(addon_max_count):
                    raw_mom_sig = int(getattr(row, "momentum_signal", 0))
                    raw_bear_sig = int(getattr(row, "bear_signal", 0))
                    raw_bull_sig = int(getattr(row, "bull_signal", 0))
                    confirm = False
                    confirm_reason = "NONE"
                    if addon_mode == "short_confirm":
                        confirm = side == -1 and (raw_mom_sig == -1 or raw_bear_sig == -1)
                        if confirm:
                            confirm_reason = "SHORT_CONFIRMED_BY_" + ("MOMENTUM" if raw_mom_sig == -1 else "BEAR")
                    elif addon_mode == "all_confirm":
                        confirm = (side == -1 and (raw_mom_sig == -1 or raw_bear_sig == -1)) or (side == 1 and (raw_mom_sig == 1 or raw_bull_sig == 1))
                        if confirm:
                            confirm_reason = "SAME_SIDE_ENGINE_CONFIRM"
                    else:
                        raise ValueError(f"Unsupported addon_mode={addon_mode}")

                    if confirm and math.isfinite(current_r) and current_r >= float(addon_min_current_r):
                        add_price = v9e.apply_entry_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                        add_stop_dist = max(active_cfg.initial_atr_mult * atr_value, risk_per_coin)
                        add_risk_mult = (
                            float(getattr(row, "risk_mult", entry_risk_mult))
                            * float(getattr(row, "quality_mult", 1.0))
                            * float(getattr(row, "micro_entry_risk_scale", 1.0))
                            * float(global_risk_scale)
                            * float(addon_risk_scale)
                        )
                        add_q = v9e.unit_qty(capital, add_price, add_stop_dist, qty, active_cfg, add_risk_mult)
                        if add_q > 0 and math.isfinite(add_q):
                            total_entry_fee += add_q * add_price * active_cfg.fee_rate
                            avg_entry = v9e.weighted_avg_price(avg_entry, qty, add_price, add_q)
                            qty += add_q
                            units += 1
                            addon_count += 1
                            addon_risk_sum += float(addon_risk_scale)
                            last_addon_reason = f"{confirm_reason}_CURRENT_R_{current_r:.2f}"
                            added_this_bar = True

                # Keep original V9E pyramiding behavior, but avoid double adding on same bar.
                if in_pos and pending_range_exit_i is None and (not added_this_bar) and units < active_cfg.max_units:
                    next_unit_number = units + 1
                    trigger_r = (next_unit_number - 1) * active_cfg.add_every_r
                    add_triggered = high >= first_entry + trigger_r * risk_per_coin if side == 1 else low <= first_entry - trigger_r * risk_per_coin
                    if add_triggered:
                        add_price = v9e.apply_entry_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                        add_stop_dist = max(active_cfg.initial_atr_mult * atr_value, risk_per_coin)
                        add_q = v9e.unit_qty(
                            capital, add_price, add_stop_dist, qty, active_cfg,
                            float(getattr(row, "risk_mult", entry_risk_mult))
                            * float(getattr(row, "quality_mult", 1.0))
                            * float(global_risk_scale),
                        )
                        if add_q > 0 and math.isfinite(add_q):
                            total_entry_fee += add_q * add_price * active_cfg.fee_rate
                            avg_entry = v9e.weighted_avg_price(avg_entry, qty, add_price, add_q)
                            qty += add_q
                            units += 1

        if not in_pos and i - last_exit_i >= cfg.cooldown_bars:
            signal = int(getattr(row, "signal", 0))
            if signal != 0:
                selected_engine = str(getattr(row, "selected_engine", "UNKNOWN"))
                entry_cfg = engine_cfgs.get(selected_engine, cfg)
                next_open = float(rows[i + 1].open)
                entry = v9e.apply_entry_slippage(next_open, signal, entry_cfg.slippage_pct)
                atr_value = float(row.atr)
                sl = entry - entry_cfg.initial_atr_mult * atr_value if signal == 1 else entry + entry_cfg.initial_atr_mult * atr_value
                stop_dist = abs(entry - sl)
                entry_risk_mult = (
                    float(getattr(row, "risk_mult", 1.0))
                    * float(getattr(row, "quality_mult", 1.0))
                    * float(getattr(row, "micro_entry_risk_scale", 1.0))
                    * float(global_risk_scale)
                )
                q = v9e.unit_qty(capital, entry, stop_dist, 0.0, entry_cfg, entry_risk_mult)
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
                    pending_range_exit_i = None
                    pending_range_exit_reason = ""
                    pending_range_exit_meta = {}
                    addon_count = 0
                    addon_risk_sum = 0.0
                    last_addon_reason = "NONE"

        equity_rows.append({"time": ts, "capital": capital, "drawdown_pct": (peak - capital) / peak if peak > 0 else 0.0})

    if in_pos:
        ts = idx[-1]
        close = float(df.iloc[-1]["close"])
        exit_price = v9e.apply_exit_slippage(close, side, pos_cfg.slippage_pct)
        capital = v9e.close_trade(
            trades=trades, capital=capital, side=side, entry_time=entry_time, exit_time=ts,
            first_entry=first_entry, avg_entry=avg_entry, exit_price=exit_price, initial_sl=initial_sl,
            stop_price=stop_price, qty=qty, units=units, total_entry_fee=total_entry_fee,
            fee_rate=pos_cfg.fee_rate, max_fav=max_fav, max_adv=max_adv,
            risk_per_coin=risk_per_coin, holding_bars=len(df) - 1 - entry_i,
            reason="FORCE_CLOSE_END", risk_mult=entry_risk_mult,
        )
        annotate_last_addon(trades, addon_count, addon_risk_sum, last_addon_reason)

    equity = pd.DataFrame(equity_rows).set_index("time") if equity_rows else pd.DataFrame()
    return trades, equity


def run_variant(name: str, features: pd.DataFrame, cfg: Any, engine_cfgs: dict[str, Any], args: argparse.Namespace, *, addon_mode: str = "off", extra: dict[str, Any] | None = None) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    if addon_mode == "off":
        trades, equity = v9e.run_priority_backtest(features, cfg, engine_cfgs=engine_cfgs, global_risk_scale=args.global_risk_scale, args=args)
    else:
        trades, equity = run_router_addon_backtest(
            features, cfg, engine_cfgs, global_risk_scale=args.global_risk_scale, args=args,
            addon_mode=addon_mode, addon_min_current_r=args.addon_min_current_r,
            addon_risk_scale=args.addon_risk_scale, addon_max_count=args.addon_max_count,
        )
    trades = v9e.attach_engine_to_trades(trades, features)
    summary = summarize_run(name, trades, equity, cfg, extra=extra)
    return summary, pd.DataFrame(trades), trades, equity


def scenario_features(baseline: pd.DataFrame, raw: dict[str, pd.DataFrame], args: argparse.Namespace) -> dict[str, tuple[pd.DataFrame, dict[str, Any], str]]:
    boost = float(args.entry_confirm_risk_boost)
    down = float(args.momentum_risk_down_scale)
    scenarios: dict[str, tuple[pd.DataFrame, dict[str, Any], str]] = {}

    scenarios["baseline_v9e"] = (baseline.copy(), {"variant_type": "baseline"}, "off")

    scenarios["router_bull_bear_primary_momentum_disabled"] = (
        make_engine_aware_router(baseline, raw, args, scenario="router_bull_bear_primary_momentum_disabled", momentum_entry_mode="none", priority_order=(ENGINE_BULL, ENGINE_BEAR, ENGINE_MOM)),
        {"variant_type": "entry_router", "momentum_entry_mode": "none", "priority_order": "BULL>BEAR"},
        "off",
    )
    scenarios["router_bull_bear_primary_mom_confirm_boost"] = (
        make_engine_aware_router(baseline, raw, args, scenario="router_bull_bear_primary_mom_confirm_boost", momentum_entry_mode="none", priority_order=(ENGINE_BULL, ENGINE_BEAR, ENGINE_MOM), confirm_boost_enabled=True, confirm_boost_scale=boost),
        {"variant_type": "entry_router", "momentum_entry_mode": "none", "confirm_boost_scale": boost},
        "off",
    )
    scenarios["router_bull_bear_primary_mom_no_confirm_half_risk"] = (
        make_engine_aware_router(baseline, raw, args, scenario="router_bull_bear_primary_mom_no_confirm_half_risk", momentum_entry_mode="none", priority_order=(ENGINE_BULL, ENGINE_BEAR, ENGINE_MOM), no_confirm_risk_scale=0.50),
        {"variant_type": "entry_router", "momentum_entry_mode": "none", "no_confirm_risk_scale": 0.50},
        "off",
    )
    scenarios["router_momentum_short_only_after_bull_bear"] = (
        make_engine_aware_router(baseline, raw, args, scenario="router_momentum_short_only_after_bull_bear", momentum_entry_mode="short_only", priority_order=(ENGINE_BULL, ENGINE_BEAR, ENGINE_MOM)),
        {"variant_type": "entry_router", "momentum_entry_mode": "short_only", "priority_order": "BULL>BEAR>MOM_SHORT"},
        "off",
    )
    scenarios["router_momentum_long_disabled_keep_short_current_priority"] = (
        make_engine_aware_router(baseline, raw, args, scenario="router_momentum_long_disabled_keep_short_current_priority", momentum_entry_mode="short_only", priority_order=(ENGINE_BULL, ENGINE_MOM, ENGINE_BEAR)),
        {"variant_type": "entry_router", "momentum_entry_mode": "short_only", "priority_order": "BULL>MOM_SHORT>BEAR"},
        "off",
    )
    scenarios["router_momentum_all_after_bull_bear_risk_down"] = (
        make_engine_aware_router(baseline, raw, args, scenario="router_momentum_all_after_bull_bear_risk_down", momentum_entry_mode="all", priority_order=(ENGINE_BULL, ENGINE_BEAR, ENGINE_MOM), momentum_selected_risk_scale=down),
        {"variant_type": "entry_router", "momentum_entry_mode": "all", "priority_order": "BULL>BEAR>MOM", "momentum_risk_down_scale": down},
        "off",
    )
    scenarios["router_momentum_long_risk_down_current_priority"] = (
        make_engine_aware_router(baseline, raw, args, scenario="router_momentum_long_risk_down_current_priority", momentum_entry_mode="all", priority_order=(ENGINE_BULL, ENGINE_MOM, ENGINE_BEAR), momentum_long_selected_risk_scale=down),
        {"variant_type": "entry_router", "momentum_entry_mode": "all", "priority_order": "BULL>MOM>BEAR", "momentum_long_risk_down_scale": down},
        "off",
    )
    scenarios["router_bull_bear_require_momentum_samebar"] = (
        make_engine_aware_router(baseline, raw, args, scenario="router_bull_bear_require_momentum_samebar", momentum_entry_mode="none", priority_order=(ENGINE_BULL, ENGINE_BEAR, ENGINE_MOM), require_momentum_confirm_for_bull_bear=True),
        {"variant_type": "entry_router", "momentum_entry_mode": "none", "bull_bear_requires_samebar_momentum": True},
        "off",
    )
    scenarios["router_bull_bear_primary_short_confirm_addon"] = (
        make_engine_aware_router(baseline, raw, args, scenario="router_bull_bear_primary_short_confirm_addon", momentum_entry_mode="none", priority_order=(ENGINE_BULL, ENGINE_BEAR, ENGINE_MOM)),
        {"variant_type": "in_position_addon", "momentum_entry_mode": "none", "addon_mode": "short_confirm", "addon_min_current_r": args.addon_min_current_r, "addon_risk_scale": args.addon_risk_scale},
        "short_confirm",
    )
    scenarios["router_mom_short_primary_short_confirm_addon"] = (
        make_engine_aware_router(baseline, raw, args, scenario="router_mom_short_primary_short_confirm_addon", momentum_entry_mode="short_only", priority_order=(ENGINE_BULL, ENGINE_BEAR, ENGINE_MOM)),
        {"variant_type": "entry_router+in_position_addon", "momentum_entry_mode": "short_only", "addon_mode": "short_confirm", "addon_min_current_r": args.addon_min_current_r, "addon_risk_scale": args.addon_risk_scale},
        "short_confirm",
    )
    return scenarios


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(PROJECT_ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline, raw, cfg, engine_cfgs = build_features(args)
    scenarios = scenario_features(baseline, raw, args)

    summaries: list[dict[str, Any]] = []
    yearly_frames: list[pd.DataFrame] = []

    for name, (features, extra, addon_mode) in scenarios.items():
        print(f"Running variant: {name}", flush=True)
        summary, trades_df, trades, equity = run_variant(name, features, cfg, engine_cfgs, args, addon_mode=addon_mode, extra=extra)
        selected = features[features["signal"].astype(int).ne(0)] if "signal" in features.columns else pd.DataFrame()
        summary["portfolio_signal_count"] = int(len(selected))
        summary["momentum_selected_count"] = int((features.get("selected_engine", pd.Series("", index=features.index)).astype(str) == ENGINE_MOM).sum())
        summary["bull_selected_count"] = int((features.get("selected_engine", pd.Series("", index=features.index)).astype(str) == ENGINE_BULL).sum())
        summary["bear_selected_count"] = int((features.get("selected_engine", pd.Series("", index=features.index)).astype(str) == ENGINE_BEAR).sum())
        summary["samebar_momentum_confirm_count"] = int(features.get("momentum_same_bar_confirm", pd.Series(False, index=features.index)).astype("boolean").fillna(False).sum())
        summaries.append(summary)
        yearly_frames.append(yearly_metrics(trades, equity, name, cfg.initial_capital))
        if args.write_trades:
            trades_df.to_csv(out_dir / f"{name}_trades.csv", index=False)
            if not equity.empty:
                equity.to_csv(out_dir / f"{name}_equity.csv")
            audit_cols = [
                "open", "high", "low", "close", "atr", "atr_pct", "adx", "signal", "selected_engine",
                "momentum_signal", "bear_signal", "bull_signal", "momentum_same_bar_confirm", "router_note", "router_risk_adjustment",
                "risk_mult", "quality_mult", "micro_entry_risk_scale", "micro_filter_action", "rf_imbalance", "rf_close_pos",
            ]
            features[[c for c in audit_cols if c in features.columns]].to_csv(out_dir / f"{name}_signal_audit.csv")

    summary_df = pd.DataFrame(summaries)
    # Sort by closed final capital but keep baseline in file.
    if not summary_df.empty and "closed_final_capital" in summary_df.columns:
        summary_df = summary_df.sort_values("closed_final_capital", ascending=False)
    summary_df.to_csv(out_dir / "v9e_engine_router_variant_summary.csv", index=False)

    yearly_df = pd.concat([f for f in yearly_frames if not f.empty], ignore_index=True) if yearly_frames else pd.DataFrame()
    yearly_df.to_csv(out_dir / "v9e_engine_router_variant_yearly.csv", index=False)

    # Compact comparison versus baseline.
    if not summary_df.empty:
        base = summary_df[summary_df["scenario"].eq("baseline_v9e")]
        if not base.empty:
            b = base.iloc[0]
            comp = summary_df.copy()
            for col in ["closed_final_capital", "closed_profit_factor", "closed_win_rate", "max_drawdown_pct", "closed_expectancy_pct", "closed_total_trades"]:
                if col in comp.columns:
                    comp[f"delta_{col}"] = pd.to_numeric(comp[col], errors="coerce") - float(b[col])
            comp.to_csv(out_dir / "v9e_engine_router_variant_compare_to_baseline.csv", index=False)

    with (out_dir / "v9e_engine_router_variants_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "scenarios": list(scenarios.keys()), "output_dir": str(out_dir)}, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 92)
    print("V9E Engine-Aware Router Variants Lab completed")
    print("=" * 92)
    print(f"Output directory: {out_dir.resolve()}")
    print("Key files:")
    print("  - v9e_engine_router_variant_summary.csv")
    print("  - v9e_engine_router_variant_compare_to_baseline.csv")
    print("  - v9e_engine_router_variant_yearly.csv")
    print("=" * 92 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
