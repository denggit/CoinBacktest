
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V9E Engine Diagnostics Lab
==========================

Research-only diagnostics for ETH_LF_Portfolio_V9E_RangeExitOverlay.

Goal:
    Diagnose each engine before changing the portfolio router:
    - raw signal quality by engine / side / year / micro context / ATR / ADX
    - standalone backtest per engine and per side
    - router conflict audit: selected engine vs non-selected engine forward behavior
    - in-position swallowed signals: same-side / opposite-side value

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

ENGINES = ("MOMENTUM_V3", "BEAR_V3_ONLY", "BULL_RECLAIM_V2")
ENGINE_LABELS = {
    "MOMENTUM_V3": "momentum",
    "BEAR_V3_ONLY": "bear",
    "BULL_RECLAIM_V2": "bull",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V9E engine-level diagnostics lab.")

    # Keep defaults aligned with V9E / previous research labs.
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

    p.add_argument("--out-dir", default="data/reports/research/v9e_engine_diagnostics_lab")
    p.add_argument("--group-min-count", type=int, default=5)
    p.add_argument("--skip-standalone-backtests", action="store_true")
    return p.parse_args()


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype("boolean").fillna(False).astype(bool)


def _num_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(float(default), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(float(default))


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


def _closed_metrics(trades: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
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
    note = tdf.get("note", pd.Series("", index=tdf.index)).astype(str)
    force = note.eq("FORCE_CLOSE_END")
    force_pnl = float(pd.to_numeric(tdf.loc[force, "pnl"], errors="coerce").fillna(0.0).sum()) if "pnl" in tdf else 0.0
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
    wins = closed.loc[closed["pnl"] > 0]
    losses = closed.loc[closed["pnl"] <= 0]
    gross_profit = float(wins["pnl"].sum()) if not wins.empty else 0.0
    gross_loss = float(-losses["pnl"].sum()) if not losses.empty else 0.0
    final_capital = float(closed.iloc[-1]["capital"])
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return {
        "closed_final_capital": final_capital,
        "closed_total_return_pct": (final_capital / float(initial_capital) - 1.0) * 100.0,
        "closed_total_trades": int(len(closed)),
        "closed_win_rate": float((closed["pnl"] > 0).mean() * 100.0),
        "closed_profit_factor": pf if math.isfinite(pf) else float("inf"),
        "closed_expectancy_pct": float(closed["return_pct"].mean() * 100.0),
        "closed_avg_win_pct": float(wins["return_pct"].mean() * 100.0) if not wins.empty else 0.0,
        "closed_avg_loss_pct": float(losses["return_pct"].mean() * 100.0) if not losses.empty else 0.0,
        "force_close_count": int(force.sum()),
        "force_close_pnl": force_pnl,
    }


def summarize_trades(name: str, trades: list[dict[str, Any]], equity: pd.DataFrame, cfg: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    base = v9e.summarize(trades, equity, cfg.initial_capital)
    out: dict[str, Any] = {"scenario": name}
    out.update(base)
    out.update(_closed_metrics(trades, cfg.initial_capital))
    if not equity.empty and "drawdown_pct" in equity.columns:
        out["max_drawdown_pct"] = float(pd.to_numeric(equity["drawdown_pct"], errors="coerce").fillna(0.0).max() * 100.0)
    if trades:
        tdf = pd.DataFrame(trades)
        out["total_fees"] = float(pd.to_numeric(tdf.get("fee", pd.Series(0.0, index=tdf.index)), errors="coerce").fillna(0.0).sum())
        out["avg_holding_hours"] = float(pd.to_numeric(tdf.get("holding_hours", pd.Series(0.0, index=tdf.index)), errors="coerce").fillna(0.0).mean())
        out["avg_mfe_r"] = float(pd.to_numeric(tdf.get("mfe_r", pd.Series(0.0, index=tdf.index)), errors="coerce").fillna(0.0).mean())
        out["avg_mae_r"] = float(pd.to_numeric(tdf.get("mae_r", pd.Series(0.0, index=tdf.index)), errors="coerce").fillna(0.0).mean())
    if extra:
        out.update(extra)
    return out


def build_feature_frames(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame, Any, dict[str, Any]]:
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
    raw_frames = {"MOMENTUM_V3": momentum, "BEAR_V3_ONLY": bear, "BULL_RECLAIM_V2": bull}

    selected = v9e.select_portfolio_signals(momentum, bear, bull, args)
    micro_ctx = v9e.load_range_footprint_context(args, load_start_str, args.end_date)
    portfolio = v9e.apply_micro_context_filter(selected, micro_ctx, args)
    portfolio = portfolio.loc[trade_start: pd.Timestamp(args.end_date)].copy().sort_index()
    print(f"Portfolio feature rows after warmup slice: {len(portfolio)}; first={portfolio.index[0] if len(portfolio) else 'NA'}", flush=True)

    engine_frames: dict[str, pd.DataFrame] = {}
    for engine in ENGINES:
        engine_df = make_engine_frame(engine, momentum, bear, bull, args)
        engine_df = v9e.apply_micro_context_filter(engine_df, micro_ctx, args)
        engine_df = engine_df.loc[trade_start: pd.Timestamp(args.end_date)].copy().sort_index()
        engine_frames[engine] = engine_df

    engine_cfgs = {"MOMENTUM_V3": exec_cfg, "BEAR_V3_ONLY": exec_cfg, "BULL_RECLAIM_V2": bull_exec_cfg}
    return portfolio, engine_frames, micro_ctx, exec_cfg, engine_cfgs


def make_engine_frame(engine: str, momentum: pd.DataFrame, bear: pd.DataFrame, bull: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = momentum.copy()
    bear_aligned = bear.reindex(out.index)
    bull_aligned = bull.reindex(out.index)

    out["momentum_signal"] = momentum["signal"].fillna(0).astype(int)
    out["bear_signal"] = bear_aligned["signal"].fillna(0).astype(int)
    out["bull_signal"] = bull_aligned["signal"].fillna(0).astype(int)
    out["momentum_long_exit_channel"] = _bool_series(momentum, "long_exit_channel")
    out["momentum_short_exit_channel"] = _bool_series(momentum, "short_exit_channel")
    out["bear_short_exit_channel"] = _bool_series(bear_aligned, "short_exit_channel")
    out["bull_long_exit_channel"] = _bool_series(bull_aligned, "long_exit_channel")
    out["portfolio_conflict"] = (out["momentum_signal"].ne(0).astype(int) + out["bear_signal"].eq(-1).astype(int) + out["bull_signal"].eq(1).astype(int)) > 1
    out["selected_engine"] = "NONE"
    out["selected_priority"] = 0
    out["momentum_selected"] = False
    out["bear_only"] = False
    out["bull_reclaim"] = False

    if engine == "MOMENTUM_V3":
        sig = out["momentum_signal"].copy()
        active = sig != 0
        out.loc[active, "selected_engine"] = engine
        out.loc[active, "selected_priority"] = 100
        out.loc[active, "momentum_selected"] = True
        # Keep momentum risk/quality from its own frame.
    elif engine == "BEAR_V3_ONLY":
        sig = pd.Series(0, index=out.index, dtype="int64")
        active = (out["bear_signal"] == -1) & (not args.disable_bear_standalone)
        sig.loc[active] = -1
        out.loc[active, "selected_engine"] = engine
        out.loc[active, "selected_priority"] = 100
        out.loc[active, "bear_only"] = True
        out.loc[active, "risk_mult"] = (
            _num_series(bear_aligned, "risk_mult", 1.0).loc[active] * float(args.bear_standalone_risk_scale)
        ).clip(args.min_risk_mult, args.max_risk_mult or 10.0)
        out.loc[active, "quality_mult"] = (
            _num_series(bear_aligned, "quality_mult", 1.0).loc[active] * float(args.bear_standalone_quality_scale)
        ).clip(0.20, args.quality_mult_cap)
    elif engine == "BULL_RECLAIM_V2":
        sig = pd.Series(0, index=out.index, dtype="int64")
        active = (out["bull_signal"] == 1) & (not args.disable_bull_reclaim)
        sig.loc[active] = 1
        out.loc[active, "selected_engine"] = engine
        out.loc[active, "selected_priority"] = 100
        out.loc[active, "bull_reclaim"] = True
        out.loc[active, "risk_mult"] = (
            _num_series(bull_aligned, "risk_mult", 1.0).loc[active] * float(args.bull_reclaim_risk_scale)
        ).clip(args.min_risk_mult, args.max_risk_mult or 10.0)
        out.loc[active, "quality_mult"] = (
            _num_series(bull_aligned, "quality_mult", 1.0).loc[active] * float(args.bull_reclaim_quality_scale)
        ).clip(0.10, args.quality_mult_cap)
    else:
        raise ValueError(f"Unknown engine: {engine}")

    out["signal"] = sig.fillna(0).astype(int)
    out["long_signal"] = out["signal"] == 1
    out["short_signal"] = out["signal"] == -1
    return out


def make_side_variant(df: pd.DataFrame, side: int) -> pd.DataFrame:
    out = df.copy()
    out.loc[out["signal"].astype(int) != int(side), "signal"] = 0
    out["long_signal"] = out["signal"] == 1
    out["short_signal"] = out["signal"] == -1
    out.loc[out["signal"].astype(int).eq(0), "selected_engine"] = "NONE"
    out.loc[out["signal"].astype(int).eq(0), "selected_priority"] = 0
    return out


def run_bt(name: str, df: pd.DataFrame, cfg: Any, engine_cfgs: dict[str, Any], args: argparse.Namespace, extra: dict[str, Any] | None = None) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    trades, equity = v9e.run_priority_backtest(
        df,
        cfg,
        engine_cfgs=engine_cfgs,
        global_risk_scale=args.global_risk_scale,
        args=args,
    )
    trades = v9e.attach_engine_to_trades(trades, df)
    summary = summarize_trades(name, trades, equity, cfg, extra=extra)
    return trades, equity, summary


def build_trade_state(features: pd.DataFrame, trades: list[dict[str, Any]]) -> pd.DataFrame:
    idx = features.index
    state = pd.DataFrame(index=idx)
    state["in_position"] = False
    state["position_side"] = 0
    state["position_type"] = "FLAT"
    state["position_engine"] = "NONE"
    state["position_entry_time"] = ""
    state["position_exit_time"] = ""
    for t in trades:
        entry_time = pd.Timestamp(t.get("entry_time"))
        exit_time = pd.Timestamp(t.get("exit_time"))
        side = 1 if str(t.get("type", "")).upper() == "LONG" else -1
        mask = (idx >= entry_time) & (idx < exit_time)
        if not bool(mask.any()):
            continue
        state.loc[mask, "in_position"] = True
        state.loc[mask, "position_side"] = side
        state.loc[mask, "position_type"] = _side_name(side)
        state.loc[mask, "position_engine"] = str(t.get("engine", "UNKNOWN"))
        state.loc[mask, "position_entry_time"] = str(entry_time)
        state.loc[mask, "position_exit_time"] = str(exit_time)
    return state


def build_executed_key_set(trades: list[dict[str, Any]]) -> set[tuple[pd.Timestamp, str, int]]:
    keys: set[tuple[pd.Timestamp, str, int]] = set()
    for t in trades:
        entry_time = pd.Timestamp(t.get("entry_time"))
        signal_time = entry_time - pd.Timedelta(hours=4)
        engine = str(t.get("engine", "UNKNOWN"))
        side = 1 if str(t.get("type", "")).upper() == "LONG" else -1
        keys.add((signal_time, engine, side))
    return keys


def forward_metrics(features: pd.DataFrame, ts: pd.Timestamp, side: int, horizons: tuple[int, ...] = (1, 3, 6, 12)) -> dict[str, float]:
    loc = features.index.get_loc(ts)
    if isinstance(loc, slice) or isinstance(loc, np.ndarray):
        return {}
    rows = features.iloc
    close_now = float(rows[loc]["close"])
    entry_i = loc + 1
    entry_open = float(rows[entry_i]["open"]) if entry_i < len(features) else close_now
    out: dict[str, float] = {"entry_open_next": entry_open}
    for h in horizons:
        target_i = min(loc + h, len(features) - 1)
        if target_i <= loc:
            out[f"fwd_{h}bar_close_ret_pct"] = np.nan
            continue
        exit_close = float(rows[target_i]["close"])
        out[f"fwd_{h}bar_close_ret_pct"] = float(side) * (exit_close / max(entry_open, 1e-12) - 1.0) * 100.0
        window = features.iloc[entry_i: target_i + 1]
        if window.empty:
            out[f"mfe_{h}bar_pct"] = np.nan
            out[f"mae_{h}bar_pct"] = np.nan
        elif side == 1:
            out[f"mfe_{h}bar_pct"] = (float(window["high"].max()) / max(entry_open, 1e-12) - 1.0) * 100.0
            out[f"mae_{h}bar_pct"] = (float(window["low"].min()) / max(entry_open, 1e-12) - 1.0) * 100.0
        else:
            out[f"mfe_{h}bar_pct"] = (1.0 - float(window["low"].min()) / max(entry_open, 1e-12)) * 100.0
            out[f"mae_{h}bar_pct"] = (1.0 - float(window["high"].max()) / max(entry_open, 1e-12)) * 100.0
    return out


def add_bins(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col, label in [("adx", "adx_bin"), ("atr_pct", "atr_pct_bin"), ("quality_mult", "quality_bin"), ("risk_mult", "risk_bin"), ("rf_imbalance", "rf_imbalance_bin"), ("rf_close_pos", "rf_close_pos_bin")]:
        if col not in out.columns:
            out[label] = "NA"
            continue
        s = pd.to_numeric(out[col], errors="coerce")
        valid = s.dropna()
        if len(valid) < 4 or valid.nunique() < 4:
            out[label] = "NA"
            continue
        try:
            out[label] = pd.qcut(s, q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop").astype(str)
        except ValueError:
            out[label] = "NA"
        out[label] = out[label].replace("nan", "NA").fillna("NA")
    return out


def build_engine_signal_table(portfolio: pd.DataFrame, engine_frames: dict[str, pd.DataFrame], baseline_trades: list[dict[str, Any]]) -> pd.DataFrame:
    trade_state = build_trade_state(portfolio, baseline_trades)
    executed_keys = build_executed_key_set(baseline_trades)
    rows: list[dict[str, Any]] = []
    common_index = portfolio.index
    for engine, edf in engine_frames.items():
        edf = edf.reindex(common_index)
        sig_series = edf["signal"].fillna(0).astype(int)
        for ts, sig in sig_series[sig_series.ne(0)].items():
            side = int(sig)
            p_row = portfolio.loc[ts]
            e_row = edf.loc[ts]
            state = trade_state.loc[ts]
            selected_engine = str(p_row.get("selected_engine", "NONE"))
            selected_sig = int(p_row.get("signal", 0) or 0)
            was_selected = selected_engine == engine and selected_sig == side
            in_position = bool(state["in_position"])
            pos_side = int(state["position_side"])
            relation = "NA"
            if in_position:
                relation = "SAME_SIDE" if pos_side == side else "OPPOSITE_SIDE"
            executed = (pd.Timestamp(ts), engine, side) in executed_keys
            if executed:
                action_context = "EXECUTED_ENTRY"
            elif in_position:
                action_context = "IGNORED_IN_POSITION_" + relation
            elif was_selected:
                action_context = "SELECTED_BUT_NOT_EXECUTED"
            else:
                action_context = "ROUTER_DROPPED_FLAT"
            item: dict[str, Any] = {
                "timestamp": ts,
                "year": int(pd.Timestamp(ts).year),
                "engine": engine,
                "side": _side_name(side),
                "side_num": side,
                "was_portfolio_selected": bool(was_selected),
                "portfolio_selected_engine": selected_engine,
                "portfolio_signal": selected_sig,
                "portfolio_conflict": bool(p_row.get("portfolio_conflict", False)),
                "executed_entry": bool(executed),
                "in_position": in_position,
                "position_side": _side_name(pos_side),
                "position_engine": str(state["position_engine"]),
                "relation_to_position": relation,
                "action_context": action_context,
                "open": _safe_float(e_row.get("open")),
                "high": _safe_float(e_row.get("high")),
                "low": _safe_float(e_row.get("low")),
                "close": _safe_float(e_row.get("close")),
                "atr": _safe_float(e_row.get("atr")),
                "atr_pct": _safe_float(e_row.get("atr_pct")),
                "adx": _safe_float(e_row.get("adx")),
                "risk_mult": _safe_float(e_row.get("risk_mult"), 1.0),
                "quality_mult": _safe_float(e_row.get("quality_mult"), 1.0),
                "micro_context_available": bool(e_row.get("micro_context_available", False)),
                "micro_aligned": bool(e_row.get("micro_aligned", False)),
                "micro_contra": bool(e_row.get("micro_contra", False)),
                "micro_entry_risk_scale": _safe_float(e_row.get("micro_entry_risk_scale"), 1.0),
                "micro_filter_action": str(e_row.get("micro_filter_action", "NA")),
                "rf_imbalance": _safe_float(e_row.get("rf_imbalance")),
                "rf_close_pos": _safe_float(e_row.get("rf_close_pos")),
                "rf_bar_count": _safe_float(e_row.get("rf_bar_count")),
            }
            item.update(forward_metrics(portfolio, pd.Timestamp(ts), side))
            rows.append(item)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = add_bins(out).sort_values(["timestamp", "engine"]).reset_index(drop=True)
    return out


def _agg_signal_group(g: pd.DataFrame) -> pd.Series:
    n = len(g)
    f3 = pd.to_numeric(g.get("fwd_3bar_close_ret_pct", pd.Series(dtype=float)), errors="coerce")
    f6 = pd.to_numeric(g.get("fwd_6bar_close_ret_pct", pd.Series(dtype=float)), errors="coerce")
    f12 = pd.to_numeric(g.get("fwd_12bar_close_ret_pct", pd.Series(dtype=float)), errors="coerce")
    return pd.Series({
        "count": int(n),
        "executed_count": int(g.get("executed_entry", pd.Series(False, index=g.index)).astype(bool).sum()),
        "selected_count": int(g.get("was_portfolio_selected", pd.Series(False, index=g.index)).astype(bool).sum()),
        "in_position_count": int(g.get("in_position", pd.Series(False, index=g.index)).astype(bool).sum()),
        "fwd_3bar_win_rate": float((f3 > 0).mean() * 100.0) if n else 0.0,
        "fwd_3bar_avg_ret_pct": float(f3.mean()) if not f3.empty else np.nan,
        "fwd_6bar_win_rate": float((f6 > 0).mean() * 100.0) if n else 0.0,
        "fwd_6bar_avg_ret_pct": float(f6.mean()) if not f6.empty else np.nan,
        "fwd_12bar_win_rate": float((f12 > 0).mean() * 100.0) if n else 0.0,
        "fwd_12bar_avg_ret_pct": float(f12.mean()) if not f12.empty else np.nan,
        "mfe_12bar_avg_pct": float(pd.to_numeric(g.get("mfe_12bar_pct", pd.Series(dtype=float)), errors="coerce").mean()),
        "mae_12bar_avg_pct": float(pd.to_numeric(g.get("mae_12bar_pct", pd.Series(dtype=float)), errors="coerce").mean()),
        "avg_adx": float(pd.to_numeric(g.get("adx", pd.Series(dtype=float)), errors="coerce").mean()),
        "avg_atr_pct": float(pd.to_numeric(g.get("atr_pct", pd.Series(dtype=float)), errors="coerce").mean()),
    })


def build_group_stats(signal_table: pd.DataFrame, min_count: int) -> pd.DataFrame:
    group_defs: list[tuple[str, list[str]]] = [
        ("engine", ["engine"]),
        ("engine_side", ["engine", "side"]),
        ("engine_year", ["engine", "year"]),
        ("engine_action_context", ["engine", "action_context"]),
        ("engine_micro_action", ["engine", "micro_filter_action"]),
        ("engine_atr_bin", ["engine", "atr_pct_bin"]),
        ("engine_adx_bin", ["engine", "adx_bin"]),
        ("engine_quality_bin", ["engine", "quality_bin"]),
        ("engine_risk_bin", ["engine", "risk_bin"]),
        ("engine_rf_imbalance_bin", ["engine", "rf_imbalance_bin"]),
        ("engine_rf_close_pos_bin", ["engine", "rf_close_pos_bin"]),
        ("engine_position_relation", ["engine", "in_position", "relation_to_position"]),
        ("engine_selected", ["engine", "was_portfolio_selected"]),
    ]
    frames: list[pd.DataFrame] = []
    if signal_table.empty:
        return pd.DataFrame()
    for group_type, keys in group_defs:
        present = [k for k in keys if k in signal_table.columns]
        if len(present) != len(keys):
            continue
        grouped = signal_table.groupby(present, dropna=False).apply(_agg_signal_group, include_groups=False).reset_index()
        grouped.insert(0, "group_type", group_type)
        grouped = grouped[grouped["count"] >= int(min_count)].copy()
        frames.append(grouped)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.sort_values(["group_type", "engine" if "engine" in out.columns else "count", "count"], ascending=[True, True, False], inplace=True)
    return out


def build_conflict_audit(signal_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if signal_table.empty:
        return pd.DataFrame(), pd.DataFrame()
    c = signal_table[signal_table["portfolio_conflict"].astype(bool)].copy()
    if c.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for ts, g in c.groupby("timestamp"):
        selected_rows = g[g["was_portfolio_selected"].astype(bool)]
        selected_engine = str(selected_rows.iloc[0]["engine"]) if not selected_rows.empty else "NONE"
        selected_side = str(selected_rows.iloc[0]["side"]) if not selected_rows.empty else "NA"
        best12_idx = pd.to_numeric(g["fwd_12bar_close_ret_pct"], errors="coerce").idxmax()
        best6_idx = pd.to_numeric(g["fwd_6bar_close_ret_pct"], errors="coerce").idxmax()
        best12_engine = str(g.loc[best12_idx, "engine"])
        best6_engine = str(g.loc[best6_idx, "engine"])
        item: dict[str, Any] = {
            "timestamp": ts,
            "candidate_count": int(len(g)),
            "candidate_engines": ";".join(g["engine"].astype(str).tolist()),
            "candidate_sides": ";".join(g["side"].astype(str).tolist()),
            "selected_engine": selected_engine,
            "selected_side": selected_side,
            "best_engine_6bar": best6_engine,
            "best_engine_12bar": best12_engine,
            "selected_is_best_6bar": selected_engine == best6_engine,
            "selected_is_best_12bar": selected_engine == best12_engine,
            "selected_fwd_6bar_ret_pct": float(selected_rows.iloc[0]["fwd_6bar_close_ret_pct"]) if not selected_rows.empty else np.nan,
            "selected_fwd_12bar_ret_pct": float(selected_rows.iloc[0]["fwd_12bar_close_ret_pct"]) if not selected_rows.empty else np.nan,
            "best_fwd_6bar_ret_pct": float(g.loc[best6_idx, "fwd_6bar_close_ret_pct"]),
            "best_fwd_12bar_ret_pct": float(g.loc[best12_idx, "fwd_12bar_close_ret_pct"]),
        }
        for _, r in g.iterrows():
            eng = str(r["engine"])
            item[f"{eng}_side"] = str(r["side"])
            item[f"{eng}_fwd_6bar_ret_pct"] = float(r["fwd_6bar_close_ret_pct"])
            item[f"{eng}_fwd_12bar_ret_pct"] = float(r["fwd_12bar_close_ret_pct"])
        rows.append(item)
    audit = pd.DataFrame(rows).sort_values("timestamp")
    summary = audit.groupby("selected_engine", dropna=False).agg(
        conflict_count=("timestamp", "count"),
        selected_best_6bar_rate=("selected_is_best_6bar", lambda x: float(pd.Series(x).mean() * 100.0)),
        selected_best_12bar_rate=("selected_is_best_12bar", lambda x: float(pd.Series(x).mean() * 100.0)),
        avg_selected_fwd_6bar_ret_pct=("selected_fwd_6bar_ret_pct", "mean"),
        avg_best_fwd_6bar_ret_pct=("best_fwd_6bar_ret_pct", "mean"),
        avg_selected_fwd_12bar_ret_pct=("selected_fwd_12bar_ret_pct", "mean"),
        avg_best_fwd_12bar_ret_pct=("best_fwd_12bar_ret_pct", "mean"),
    ).reset_index()
    return audit, summary


def build_swallowed_signal_summary(signal_table: pd.DataFrame, min_count: int) -> pd.DataFrame:
    if signal_table.empty:
        return pd.DataFrame()
    swallowed = signal_table[signal_table["in_position"].astype(bool)].copy()
    if swallowed.empty:
        return pd.DataFrame()
    keys = ["engine", "relation_to_position", "position_engine"]
    out = swallowed.groupby(keys, dropna=False).apply(_agg_signal_group, include_groups=False).reset_index()
    out = out[out["count"] >= int(min_count)].copy()
    out.sort_values(["relation_to_position", "engine", "count"], ascending=[True, True, False], inplace=True)
    return out


def run_standalone_suite(portfolio: pd.DataFrame, engine_frames: dict[str, pd.DataFrame], cfg: Any, engine_cfgs: dict[str, Any], args: argparse.Namespace, out_dir: Path) -> tuple[pd.DataFrame, dict[str, list[dict[str, Any]]]]:
    summaries: list[dict[str, Any]] = []
    trade_map: dict[str, list[dict[str, Any]]] = {}

    baseline_trades, baseline_equity, baseline_summary = run_bt("portfolio_baseline_v9e", portfolio, cfg, engine_cfgs, args, extra={"scope": "portfolio"})
    summaries.append(baseline_summary)
    trade_map["portfolio_baseline_v9e"] = baseline_trades
    pd.DataFrame(baseline_trades).to_csv(out_dir / "portfolio_baseline_v9e_trades.csv", index=False)
    if not baseline_equity.empty:
        baseline_equity.to_csv(out_dir / "portfolio_baseline_v9e_equity.csv")

    for engine, edf in engine_frames.items():
        scenario = f"standalone_{ENGINE_LABELS[engine]}"
        trades, equity, summary = run_bt(scenario, edf, cfg, engine_cfgs, args, extra={"scope": "standalone", "engine": engine, "side_filter": "BOTH"})
        summaries.append(summary)
        trade_map[scenario] = trades
        pd.DataFrame(trades).to_csv(out_dir / f"{scenario}_trades.csv", index=False)
        if not equity.empty:
            equity.to_csv(out_dir / f"{scenario}_equity.csv")

        for side in (1, -1):
            if engine == "BEAR_V3_ONLY" and side == 1:
                continue
            if engine == "BULL_RECLAIM_V2" and side == -1:
                continue
            side_df = make_side_variant(edf, side)
            if not bool(side_df["signal"].astype(int).ne(0).any()):
                continue
            side_name = _side_name(side).lower()
            scenario = f"standalone_{ENGINE_LABELS[engine]}_{side_name}_only"
            trades, equity, summary = run_bt(scenario, side_df, cfg, engine_cfgs, args, extra={"scope": "standalone", "engine": engine, "side_filter": _side_name(side)})
            summaries.append(summary)
            trade_map[scenario] = trades
            pd.DataFrame(trades).to_csv(out_dir / f"{scenario}_trades.csv", index=False)
            if not equity.empty:
                equity.to_csv(out_dir / f"{scenario}_equity.csv")

    return pd.DataFrame(summaries), trade_map


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(PROJECT_ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    portfolio, engine_frames, _micro_ctx, cfg, engine_cfgs = build_feature_frames(args)

    print("Running portfolio baseline...", flush=True)
    baseline_trades, baseline_equity, baseline_summary = run_bt("portfolio_baseline_v9e", portfolio, cfg, engine_cfgs, args, extra={"scope": "portfolio"})
    baseline_trades_df = pd.DataFrame(baseline_trades)
    baseline_trades_df.to_csv(out_dir / "portfolio_baseline_v9e_trades.csv", index=False)
    if not baseline_equity.empty:
        baseline_equity.to_csv(out_dir / "portfolio_baseline_v9e_equity.csv")

    print("Building engine signal opportunity table...", flush=True)
    signal_table = build_engine_signal_table(portfolio, engine_frames, baseline_trades)
    signal_table.to_csv(out_dir / "v9e_engine_signal_opportunity_table.csv", index=False)

    group_stats = build_group_stats(signal_table, min_count=args.group_min_count)
    group_stats.to_csv(out_dir / "v9e_engine_signal_group_stats.csv", index=False)

    conflict_audit, conflict_summary = build_conflict_audit(signal_table)
    conflict_audit.to_csv(out_dir / "v9e_router_conflict_audit.csv", index=False)
    conflict_summary.to_csv(out_dir / "v9e_router_conflict_summary.csv", index=False)

    swallowed_summary = build_swallowed_signal_summary(signal_table, min_count=args.group_min_count)
    swallowed_summary.to_csv(out_dir / "v9e_swallowed_signal_summary.csv", index=False)

    if args.skip_standalone_backtests:
        standalone_summary = pd.DataFrame([baseline_summary])
    else:
        print("Running standalone engine backtests...", flush=True)
        standalone_summary, _trade_map = run_standalone_suite(portfolio, engine_frames, cfg, engine_cfgs, args, out_dir)

    standalone_summary.to_csv(out_dir / "v9e_engine_standalone_summary.csv", index=False)

    top_rows: list[dict[str, Any]] = []
    if not signal_table.empty:
        for engine, g in signal_table.groupby("engine"):
            top_rows.append({
                "engine": engine,
                "raw_signal_count": int(len(g)),
                "selected_count": int(g["was_portfolio_selected"].astype(bool).sum()),
                "executed_entry_count": int(g["executed_entry"].astype(bool).sum()),
                "in_position_signal_count": int(g["in_position"].astype(bool).sum()),
                "same_side_in_position_count": int(((g["in_position"].astype(bool)) & (g["relation_to_position"] == "SAME_SIDE")).sum()),
                "opposite_side_in_position_count": int(((g["in_position"].astype(bool)) & (g["relation_to_position"] == "OPPOSITE_SIDE")).sum()),
                "avg_fwd_6bar_ret_pct": float(pd.to_numeric(g["fwd_6bar_close_ret_pct"], errors="coerce").mean()),
                "avg_fwd_12bar_ret_pct": float(pd.to_numeric(g["fwd_12bar_close_ret_pct"], errors="coerce").mean()),
                "fwd_6bar_win_rate": float((pd.to_numeric(g["fwd_6bar_close_ret_pct"], errors="coerce") > 0).mean() * 100.0),
                "fwd_12bar_win_rate": float((pd.to_numeric(g["fwd_12bar_close_ret_pct"], errors="coerce") > 0).mean() * 100.0),
            })
    engine_overview = pd.DataFrame(top_rows)
    engine_overview.to_csv(out_dir / "v9e_engine_overview.csv", index=False)

    summary_json = {
        "args": vars(args),
        "baseline_summary": baseline_summary,
        "engine_overview": engine_overview.to_dict(orient="records"),
        "output_files": [
            "v9e_engine_signal_opportunity_table.csv",
            "v9e_engine_signal_group_stats.csv",
            "v9e_engine_standalone_summary.csv",
            "v9e_router_conflict_audit.csv",
            "v9e_router_conflict_summary.csv",
            "v9e_swallowed_signal_summary.csv",
            "v9e_engine_overview.csv",
        ],
    }
    with (out_dir / "v9e_engine_diagnostics_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 92)
    print("V9E Engine Diagnostics Lab completed")
    print("=" * 92)
    print(f"Output directory: {out_dir.resolve()}")
    print("Key files:")
    print("  - v9e_engine_overview.csv")
    print("  - v9e_engine_standalone_summary.csv")
    print("  - v9e_engine_signal_group_stats.csv")
    print("  - v9e_router_conflict_summary.csv")
    print("  - v9e_swallowed_signal_summary.csv")
    print("=" * 92 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
