#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V9E Signal Opportunity Lab
==========================

Research-only lab for ETH_LF_Portfolio_V9E_RangeExitOverlay.

Goals:
    1. Analyze all selected portfolio signals, not only executed trades.
    2. Mine coarse, human-readable bad-entry rule candidates from executed entries
       and forward-return behavior.
    3. Run conservative filter / signal-utilization backtest variants without
       modifying the frozen V9E strategy file.

Important:
    - This file is research-only. It does not place orders.
    - It does not change V9E's entry signal generation.
    - Rule mining is in-sample by default; use it to generate hypotheses, not as
      final live rules without walk-forward / pressure tests.
"""
from __future__ import annotations

import argparse
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backtest.lf import eth_lf_portfolio_v9e_range_exit_overlay_backtest as v9e  # noqa: E402
from src.backtest_common.execution import apply_entry_slippage, apply_exit_slippage  # noqa: E402
from backtest.lf.eth_1d_4h_trend_rider_v8_position_lock_backtest import (  # noqa: E402
    close_trade,
    protected_stop,
    unit_qty,
    weighted_avg_price,
)

ROUNDTRIP_COST_PCT = 0.11


@dataclass(frozen=True)
class GroupRule:
    rule_id: str
    keys: tuple[str, ...]
    values: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "keys": list(self.keys),
            "values": list(self.values),
            "reason": self.reason,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Research lab for V9E signal opportunity / entry filtering / utilization variants.")

    # Core strategy arguments. Defaults intentionally match V9E defaults.
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

    # Lab-only arguments.
    p.add_argument("--out-dir", default="data/reports/research/v9e_signal_opportunity_lab")
    p.add_argument("--min-group-samples", type=int, default=8, help="Minimum signal samples for opportunity group stats.")
    p.add_argument("--min-executed-samples", type=int, default=4, help="Minimum executed trades for bad-entry rule candidates.")
    p.add_argument("--bad-rule-top-n", type=int, default=5, help="How many mined bad-entry groups to test as blocked rules.")
    p.add_argument("--max-bad-entry-win-rate", type=float, default=0.25)
    p.add_argument("--max-bad-entry-pf", type=float, default=1.00)
    p.add_argument("--low-quality-quantile", type=float, default=0.25)
    p.add_argument("--low-adx-quantile", type=float, default=0.25)
    p.add_argument("--signal-addon-min-current-r", type=float, default=1.0)
    p.add_argument("--signal-addon-risk-scale", type=float, default=0.30)
    p.add_argument("--signal-addon-max-count", type=int, default=1)
    p.add_argument("--signal-addon-require-micro-aligned", action="store_true")
    p.add_argument("--signal-addon-block-micro-contra", action="store_true", default=True)
    p.add_argument("--skip-variant-backtests", action="store_true", help="Only build opportunity/group/rule analysis; do not run variant backtests.")
    return p.parse_args()


def _to_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out


def _safe_bool(value: Any) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def _ts(value: Any) -> pd.Timestamp:
    return pd.Timestamp(value).tz_localize(None) if pd.Timestamp(value).tzinfo else pd.Timestamp(value)


def build_features(args: argparse.Namespace) -> tuple[pd.DataFrame, Any, dict[str, Any]]:
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
    features = v9e.select_portfolio_signals(momentum, bear, bull, args)
    micro_ctx = v9e.load_range_footprint_context(args, load_start_str, args.end_date)
    features = v9e.apply_micro_context_filter(features, micro_ctx, args)
    features = features.loc[trade_start: pd.Timestamp(args.end_date)].copy()
    features.sort_index(inplace=True)
    print(f"Feature rows after warmup slice: {len(features)}; first={features.index[0] if len(features) else 'NA'}", flush=True)

    engine_cfgs = {"MOMENTUM_V3": exec_cfg, "BEAR_V3_ONLY": exec_cfg, "BULL_RECLAIM_V2": bull_exec_cfg}
    return features, exec_cfg, engine_cfgs


def _df_numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return a numeric Series aligned to df.index even when the column is absent."""
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype="float64")


def _df_column(df: pd.DataFrame, column: str, default: Any = None) -> pd.Series:
    """Return a Series aligned to df.index even when the column is absent."""
    if column in df.columns:
        return df[column]
    return pd.Series(default, index=df.index)


def closed_only_metrics(trades: list[dict[str, Any]], initial_capital: float) -> dict[str, Any]:
    if not trades:
        return {
            "closed_final_capital": initial_capital,
            "closed_total_trades": 0,
            "closed_win_rate": 0.0,
            "closed_profit_factor": 0.0,
            "closed_expectancy_pct": 0.0,
            "force_close_count": 0,
            "force_close_pnl": 0.0,
        }
    tdf = pd.DataFrame(trades).copy()
    note = _df_column(tdf, "note", "").astype(str)
    force = note.eq("FORCE_CLOSE_END")
    force_count = int(force.sum())
    force_pnl = float(pd.to_numeric(tdf.loc[force, "pnl"], errors="coerce").fillna(0.0).sum()) if "pnl" in tdf.columns else 0.0
    closed = tdf.loc[~force].copy()
    if closed.empty:
        return {
            "closed_final_capital": initial_capital,
            "closed_total_trades": 0,
            "closed_win_rate": 0.0,
            "closed_profit_factor": 0.0,
            "closed_expectancy_pct": 0.0,
            "force_close_count": force_count,
            "force_close_pnl": force_pnl,
        }
    pnl = _df_numeric_column(closed, "pnl", 0.0)
    ret = _df_numeric_column(closed, "return_pct", 0.0)
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    pf = gp / gl if gl > 0 else float("inf") if gp > 0 else 0.0
    wins = ret > 0
    return {
        "closed_final_capital": float(closed.iloc[-1].get("capital", initial_capital)),
        "closed_total_trades": int(len(closed)),
        "closed_win_rate": float(wins.mean() * 100.0),
        "closed_profit_factor": float(pf),
        "closed_expectancy_pct": float(ret.mean()),
        "closed_avg_win_pct": float(ret[wins].mean()) if bool(wins.any()) else 0.0,
        "closed_avg_loss_pct": float(ret[~wins].mean()) if bool((~wins).any()) else 0.0,
        "force_close_count": force_count,
        "force_close_pnl": force_pnl,
    }


def summarize_run(name: str, trades: list[dict[str, Any]], equity: pd.DataFrame, cfg: Any, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    s = v9e.summarize(trades, equity, cfg.initial_capital)
    s.update(closed_only_metrics(trades, cfg.initial_capital))
    s["scenario"] = name
    if trades:
        tdf = pd.DataFrame(trades)
        s["force_close_included"] = bool(_df_column(tdf, "note", "").astype(str).eq("FORCE_CLOSE_END").any())
        signal_addon_count = _df_numeric_column(tdf, "signal_addon_count", 0.0)
        s["signal_addon_trade_count"] = int(signal_addon_count.gt(0).sum())
        s["signal_addon_total_count"] = int(signal_addon_count.sum())
        s["engine_counts"] = tdf.get("engine", pd.Series(dtype=str)).value_counts().to_dict() if "engine" in tdf.columns else {}
    else:
        s["force_close_included"] = False
        s["signal_addon_trade_count"] = 0
        s["signal_addon_total_count"] = 0
        s["engine_counts"] = {}
    if extra:
        s.update(extra)
    return s


def run_v9e_baseline(features: pd.DataFrame, cfg: Any, engine_cfgs: dict[str, Any], args: argparse.Namespace) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[str, Any]]:
    trades, equity = v9e.run_priority_backtest(features, cfg, engine_cfgs=engine_cfgs, global_risk_scale=args.global_risk_scale, args=args)
    trades = v9e.attach_engine_to_trades(trades, features)
    summary = summarize_run("baseline_v9e", trades, equity, cfg)
    return trades, equity, summary


def _signal_side_label(signal: int) -> str:
    if signal == 1:
        return "LONG"
    if signal == -1:
        return "SHORT"
    return "NONE"


def _make_bin(series: pd.Series, q: int = 4, prefix: str = "Q") -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    out = pd.Series("NA", index=series.index, dtype="object")
    valid = numeric.dropna()
    if valid.nunique() < 2 or len(valid) < q:
        out.loc[valid.index] = "ALL"
        return out
    try:
        binned = pd.qcut(valid, q=q, duplicates="drop")
    except ValueError:
        out.loc[valid.index] = "ALL"
        return out
    categories = list(binned.cat.categories)
    mapping = {cat: f"{prefix}{i + 1}:{cat.left:.4g}~{cat.right:.4g}" for i, cat in enumerate(categories)}
    out.loc[valid.index] = binned.map(mapping).astype("object")
    return out


def add_research_bins(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    out["signal_side"] = _df_numeric_column(out, "signal", 0.0).astype(int).map(_signal_side_label)
    out["micro_alignment_bucket"] = "NO_CTX"
    if "micro_context_available" in out.columns:
        out.loc[out["micro_context_available"].astype(bool), "micro_alignment_bucket"] = "CTX_NOT_ALIGNED"
    if "micro_aligned" in out.columns:
        out.loc[out["micro_aligned"].fillna(False).astype(bool), "micro_alignment_bucket"] = "ALIGNED"
    if "micro_contra" in out.columns:
        out.loc[out["micro_contra"].fillna(False).astype(bool), "micro_alignment_bucket"] = "CONTRA"
    for col, q, prefix in [
        ("adx", 4, "ADX_Q"),
        ("atr_pct", 4, "ATR_Q"),
        ("quality_mult", 4, "QUAL_Q"),
        ("risk_mult", 4, "RISK_Q"),
        ("rf_close_pos", 4, "CLOSEPOS_Q"),
        ("rf_imbalance", 4, "IMB_Q"),
        ("rf_bar_count", 4, "RBAR_Q"),
    ]:
        if col in out.columns:
            out[f"{col}_bin"] = _make_bin(out[col], q=q, prefix=prefix)
        else:
            out[f"{col}_bin"] = "NA"
    return out


def _trade_intervals(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    intervals: list[dict[str, Any]] = []
    for trade in trades:
        try:
            entry_time = _ts(trade["entry_time"])
            exit_time = _ts(trade["exit_time"])
        except Exception:
            continue
        side_raw = str(trade.get("type", trade.get("side", ""))).upper()
        side = 1 if side_raw == "LONG" else -1 if side_raw == "SHORT" else int(trade.get("side", 0) or 0)
        intervals.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "side": side,
            "note": str(trade.get("note", "")),
            "return_pct": _to_float(trade.get("return_pct", float("nan"))),
            "pnl": _to_float(trade.get("pnl", float("nan"))),
            "engine": str(trade.get("engine", "UNKNOWN")),
            "entry_signal_time": entry_time - pd.Timedelta(hours=4),
            "exit_signal_time": exit_time - pd.Timedelta(hours=4),
        })
    return intervals


def build_signal_opportunity_table(features: pd.DataFrame, trades: list[dict[str, Any]]) -> pd.DataFrame:
    feats = add_research_bins(features)
    signal_mask = feats["signal"].fillna(0).astype(int) != 0
    signals = feats.loc[signal_mask].copy()
    intervals = _trade_intervals(trades)
    entry_by_signal_time: dict[pd.Timestamp, dict[str, Any]] = {item["entry_signal_time"]: item for item in intervals}
    opposite_exit_signal_times = {item["exit_signal_time"] for item in intervals if item.get("note") == "OPPOSITE_BREAKOUT_NEXT_OPEN"}

    rows: list[dict[str, Any]] = []
    idx_list = list(feats.index)
    index_pos = {ts: i for i, ts in enumerate(idx_list)}
    for ts, row in signals.iterrows():
        signal = int(row.get("signal", 0))
        exec_time = ts + pd.Timedelta(hours=4)
        matched_entry = entry_by_signal_time.get(pd.Timestamp(ts))
        active = None
        for item in intervals:
            if item["entry_time"] <= exec_time < item["exit_time"]:
                active = item
                break
        if matched_entry is not None:
            action = "EXECUTED_ENTRY"
            ignored_reason = ""
            entry_trade_return_pct = matched_entry["return_pct"]
            entry_trade_pnl = matched_entry["pnl"]
        elif active is not None:
            if signal == active["side"]:
                action = "IGNORED_SAME_SIDE_IN_POSITION"
            elif pd.Timestamp(ts) in opposite_exit_signal_times:
                action = "USED_OPPOSITE_EXIT_SIGNAL"
            else:
                action = "IGNORED_OPPOSITE_OR_CONFLICT_IN_POSITION"
            ignored_reason = "IN_POSITION"
            entry_trade_return_pct = float("nan")
            entry_trade_pnl = float("nan")
        else:
            action = "IGNORED_FLAT_NOT_ENTERED"
            ignored_reason = "COOLDOWN_OR_ZERO_QTY_OR_FILTERED_AFTER_SIGNAL"
            entry_trade_return_pct = float("nan")
            entry_trade_pnl = float("nan")

        pos = index_pos.get(ts)
        next_open = float("nan")
        if pos is not None and pos + 1 < len(idx_list):
            next_open = _to_float(feats.iloc[pos + 1].get("open"))
        item: dict[str, Any] = {
            "timestamp": ts,
            "execution_time": exec_time,
            "signal": signal,
            "signal_side": _signal_side_label(signal),
            "selected_engine": str(row.get("selected_engine", "UNKNOWN")),
            "action_taken": action,
            "ignored_reason": ignored_reason,
            "active_position_side": _signal_side_label(active["side"]) if active is not None else "FLAT",
            "active_position_engine": str(active.get("engine", "")) if active is not None else "",
            "entry_trade_return_pct": entry_trade_return_pct,
            "entry_trade_pnl": entry_trade_pnl,
            "next_open": next_open,
        }
        for col in [
            "micro_filter_action", "micro_context_available", "micro_aligned", "micro_contra",
            "micro_alignment_bucket", "portfolio_conflict", "momentum_signal", "bear_signal", "bull_signal",
            "risk_mult", "quality_mult", "adx", "atr_pct", "rf_bar_count", "rf_imbalance", "rf_close_pos",
            "rf_taker_buy_ratio", "rf_max_sell_bucket_share", "rf_max_buy_bucket_share",
            "adx_bin", "atr_pct_bin", "quality_mult_bin", "risk_mult_bin", "rf_close_pos_bin",
            "rf_imbalance_bin", "rf_bar_count_bin",
        ]:
            if col in row.index:
                item[col] = row.get(col)
        for bars in [1, 3, 6, 12, 24]:
            fwd_col = f"future_{bars}bar_signal_return_pct"
            if pos is None or pos + bars >= len(idx_list) or not math.isfinite(next_open) or next_open <= 0:
                item[fwd_col] = float("nan")
            else:
                future_close = _to_float(feats.iloc[pos + bars].get("close"))
                if math.isfinite(future_close):
                    item[fwd_col] = signal * (future_close / next_open - 1.0) * 100.0
                else:
                    item[fwd_col] = float("nan")
        rows.append(item)
    return pd.DataFrame(rows)


def _profit_factor_from_returns(ret: pd.Series) -> float:
    x = pd.to_numeric(ret, errors="coerce").dropna()
    if x.empty:
        return 0.0
    gp = float(x[x > 0].sum())
    gl = float(-x[x < 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else 0.0
    return gp / gl


def group_stats(ops: pd.DataFrame, min_samples: int) -> pd.DataFrame:
    group_specs: list[tuple[str, tuple[str, ...]]] = [
        ("engine", ("selected_engine",)),
        ("side", ("signal_side",)),
        ("engine_side", ("selected_engine", "signal_side")),
        ("engine_micro_action", ("selected_engine", "micro_filter_action")),
        ("engine_micro_bucket", ("selected_engine", "micro_alignment_bucket")),
        ("engine_quality_bin", ("selected_engine", "quality_mult_bin")),
        ("engine_adx_bin", ("selected_engine", "adx_bin")),
        ("engine_atr_bin", ("selected_engine", "atr_pct_bin")),
        ("engine_rf_close_pos_bin", ("selected_engine", "rf_close_pos_bin")),
        ("engine_rf_imb_bin", ("selected_engine", "rf_imbalance_bin")),
        ("engine_side_micro", ("selected_engine", "signal_side", "micro_alignment_bucket")),
        ("engine_side_quality", ("selected_engine", "signal_side", "quality_mult_bin")),
        ("engine_side_adx", ("selected_engine", "signal_side", "adx_bin")),
    ]
    rows: list[dict[str, Any]] = []
    for group_name, keys in group_specs:
        missing = [k for k in keys if k not in ops.columns]
        if missing:
            continue
        grouped = ops.groupby(list(keys), dropna=False)
        for values, g in grouped:
            if not isinstance(values, tuple):
                values = (values,)
            if len(g) < min_samples:
                continue
            executed = g[g["action_taken"].eq("EXECUTED_ENTRY")]
            ret = pd.to_numeric(executed.get("entry_trade_return_pct"), errors="coerce").dropna()
            row: dict[str, Any] = {
                "group_name": group_name,
                "keys": "|".join(keys),
                "values": "|".join(str(v) for v in values),
                "signal_count": int(len(g)),
                "executed_count": int(len(executed)),
                "ignored_same_side_count": int(g["action_taken"].eq("IGNORED_SAME_SIDE_IN_POSITION").sum()),
                "used_opposite_exit_count": int(g["action_taken"].eq("USED_OPPOSITE_EXIT_SIGNAL").sum()),
                "executed_win_rate_pct": float((ret > 0).mean() * 100.0) if len(ret) else float("nan"),
                "executed_avg_return_pct": float(ret.mean()) if len(ret) else float("nan"),
                "executed_sum_return_pct": float(ret.sum()) if len(ret) else 0.0,
                "executed_profit_factor": _profit_factor_from_returns(ret),
            }
            for bars in [1, 3, 6, 12, 24]:
                col = f"future_{bars}bar_signal_return_pct"
                x = pd.to_numeric(g.get(col), errors="coerce").dropna()
                row[f"fwd_{bars}bar_count"] = int(len(x))
                row[f"fwd_{bars}bar_win_rate_pct"] = float((x > ROUNDTRIP_COST_PCT).mean() * 100.0) if len(x) else float("nan")
                row[f"fwd_{bars}bar_avg_pct"] = float(x.mean()) if len(x) else float("nan")
                row[f"fwd_{bars}bar_median_pct"] = float(x.median()) if len(x) else float("nan")
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out.sort_values(["executed_sum_return_pct", "fwd_6bar_avg_pct"], ascending=[True, True], inplace=True)
    return out


def mine_bad_entry_rules(stats: pd.DataFrame, args: argparse.Namespace) -> list[GroupRule]:
    if stats.empty:
        return []
    s = stats.copy()
    s = s[pd.to_numeric(s["executed_count"], errors="coerce").fillna(0).astype(int) >= int(args.min_executed_samples)]
    if s.empty:
        return []
    pf = pd.to_numeric(s["executed_profit_factor"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    win_rate = pd.to_numeric(s["executed_win_rate_pct"], errors="coerce") / 100.0
    avg_ret = pd.to_numeric(s["executed_avg_return_pct"], errors="coerce")
    fwd6 = pd.to_numeric(s["fwd_6bar_avg_pct"], errors="coerce")
    bad = s[(pf <= float(args.max_bad_entry_pf)) | (win_rate <= float(args.max_bad_entry_win_rate)) | ((avg_ret < 0) & (fwd6 < 0))].copy()
    if bad.empty:
        return []
    bad["bad_score"] = (
        pd.to_numeric(bad["executed_sum_return_pct"], errors="coerce").fillna(0.0)
        + pd.to_numeric(bad["fwd_6bar_avg_pct"], errors="coerce").fillna(0.0)
        + pd.to_numeric(bad["fwd_12bar_avg_pct"], errors="coerce").fillna(0.0)
    )
    bad.sort_values(["bad_score", "executed_profit_factor"], ascending=[True, True], inplace=True)
    rules: list[GroupRule] = []
    seen: set[tuple[tuple[str, ...], tuple[str, ...]]] = set()
    for i, row in bad.head(int(args.bad_rule_top_n)).iterrows():
        keys = tuple(str(row["keys"]).split("|"))
        values = tuple(str(row["values"]).split("|"))
        ident = (keys, values)
        if ident in seen:
            continue
        seen.add(ident)
        reason = (
            f"exec_count={int(row['executed_count'])}, "
            f"exec_wr={_to_float(row['executed_win_rate_pct']):.1f}%, "
            f"exec_pf={_to_float(row['executed_profit_factor']):.3f}, "
            f"exec_avg={_to_float(row['executed_avg_return_pct']):.2f}%, "
            f"fwd6={_to_float(row['fwd_6bar_avg_pct']):.2f}%"
        )
        rules.append(GroupRule(rule_id=f"BAD_GROUP_{len(rules) + 1:02d}", keys=keys, values=values, reason=reason))
    return rules


def match_group_rule(features: pd.DataFrame, rule: GroupRule) -> pd.Series:
    mask = pd.Series(True, index=features.index)
    for key, value in zip(rule.keys, rule.values):
        if key not in features.columns:
            return pd.Series(False, index=features.index)
        mask &= features[key].astype(str).fillna("NA").eq(str(value))
    return mask


def apply_filter_variant(features: pd.DataFrame, variant: str, rules: list[GroupRule], args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    feats = add_research_bins(features)
    out = feats.copy()
    signal_mask = out["signal"].fillna(0).astype(int) != 0
    block = pd.Series(False, index=out.index)
    reason_parts: list[str] = []

    if variant == "baseline":
        pass
    elif variant == "block_micro_contra":
        block |= signal_mask & out.get("micro_contra", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        reason_parts.append("micro_contra")
    elif variant == "block_not_aligned":
        action = out.get("micro_filter_action", pd.Series("", index=out.index)).astype(str)
        block |= signal_mask & action.str.contains("NOT_ALIGNED", na=False)
        reason_parts.append("micro_filter_action contains NOT_ALIGNED")
    elif variant == "block_low_quality_q25":
        sig_quality = pd.to_numeric(out.loc[signal_mask, "quality_mult"], errors="coerce").dropna()
        threshold = float(sig_quality.quantile(float(args.low_quality_quantile))) if not sig_quality.empty else float("nan")
        block |= signal_mask & (pd.to_numeric(out.get("quality_mult"), errors="coerce") <= threshold)
        reason_parts.append(f"quality_mult <= q{args.low_quality_quantile:.2f} ({threshold:.6g})")
    elif variant == "block_low_adx_q25":
        sig_adx = pd.to_numeric(out.loc[signal_mask, "adx"], errors="coerce").dropna()
        threshold = float(sig_adx.quantile(float(args.low_adx_quantile))) if not sig_adx.empty else float("nan")
        block |= signal_mask & (pd.to_numeric(out.get("adx"), errors="coerce") <= threshold)
        reason_parts.append(f"adx <= q{args.low_adx_quantile:.2f} ({threshold:.6g})")
    elif variant.startswith("block_bad_groups"):
        for rule in rules:
            rule_mask = match_group_rule(out, rule)
            block |= signal_mask & rule_mask
        reason_parts.append(f"mined_bad_group_rules={len(rules)}")
    else:
        raise ValueError(f"Unsupported filter variant: {variant}")

    blocked_count = int(block.sum())
    out.loc[block, "signal"] = 0
    out.loc[block, "long_signal"] = False
    out.loc[block, "short_signal"] = False
    out.loc[block, "signal_lab_blocked"] = True
    out.loc[~block, "signal_lab_blocked"] = False
    out.loc[block, "signal_lab_block_reason"] = "; ".join(reason_parts) if reason_parts else ""
    return out, {
        "filter_variant": variant,
        "filtered_signal_count": blocked_count,
        "filter_reason": "; ".join(reason_parts),
        "filter_rules": [r.to_dict() for r in rules] if variant.startswith("block_bad_groups") else [],
    }


def annotate_last_trade_signal_addons(trades: list[dict[str, Any]], *, mode: str, count: int, risk_scale_sum: float, last_reason: str) -> None:
    if not trades:
        return
    trades[-1]["signal_addon_mode"] = mode
    trades[-1]["signal_addon_count"] = int(count)
    trades[-1]["signal_addon_risk_scale_avg"] = round(float(risk_scale_sum / count), 6) if count > 0 else 0.0
    trades[-1]["last_signal_addon_reason"] = str(last_reason)


def run_signal_addon_backtest(
    df: pd.DataFrame,
    cfg: Any,
    engine_cfgs: dict[str, Any] | None,
    *,
    global_risk_scale: float,
    args: argparse.Namespace,
    signal_addon_mode: str,
    signal_addon_min_current_r: float,
    signal_addon_risk_scale: float,
    signal_addon_max_count: int,
    signal_addon_require_micro_aligned: bool,
    signal_addon_block_micro_contra: bool,
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    """V9E-equivalent executor plus optional same-side signal add-on probe.

    The only intentional difference from V9E is this conservative add-on condition:
        already in position + current selected portfolio signal is same side + trade has
        at least N current R + max one signal add-on by default.
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
    signal_addon_count = 0
    signal_addon_risk_scale_sum = 0.0
    last_signal_addon_reason = "NONE"

    rows = list(df.itertuples())
    idx = df.index

    for i in range(len(rows) - 1):
        row = rows[i]
        ts = idx[i]
        if in_pos:
            if pending_range_exit_i is not None and i >= pending_range_exit_i:
                active_cfg = pos_cfg
                hold_bars = i - entry_i
                exit_price = apply_exit_slippage(float(row.open), side, active_cfg.slippage_pct)
                exit_time = idx[i]
                reason = pending_range_exit_reason or "RANGE_EXIT_DELAYED_OPEN"
                capital = close_trade(
                    trades=trades, capital=capital, side=side, entry_time=entry_time, exit_time=exit_time,
                    first_entry=first_entry, avg_entry=avg_entry, exit_price=exit_price, initial_sl=initial_sl,
                    stop_price=stop_price, qty=qty, units=units, total_entry_fee=total_entry_fee,
                    fee_rate=active_cfg.fee_rate, max_fav=max_fav, max_adv=max_adv,
                    risk_per_coin=risk_per_coin, holding_bars=hold_bars, reason=reason, risk_mult=entry_risk_mult,
                )
                if trades:
                    trades[-1].update(pending_range_exit_meta)
                    trades[-1]["range_exit_executed_after_delay"] = True
                    annotate_last_trade_signal_addons(
                        trades, mode=signal_addon_mode, count=signal_addon_count,
                        risk_scale_sum=signal_addon_risk_scale_sum, last_reason=last_signal_addon_reason,
                    )
                peak = max(peak, capital)
                in_pos = False
                side = 0
                last_exit_i = i
                pending_range_exit_i = None
                pending_range_exit_reason = ""
                pending_range_exit_meta = {}
                signal_addon_count = 0
                signal_addon_risk_scale_sum = 0.0
                last_signal_addon_reason = "NONE"
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
                    locked = protected_stop(first_entry, avg_entry, side, risk_per_coin, max_fav, active_cfg)
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
                    locked = protected_stop(first_entry, avg_entry, side, risk_per_coin, max_fav, active_cfg)
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
                    exit_price = apply_exit_slippage(active_stop, side, active_cfg.slippage_pct)
                    reason = "PROTECTED_TRAILING_STOP"
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
                        trades=trades, capital=capital, side=side, entry_time=entry_time, exit_time=exit_time,
                        first_entry=first_entry, avg_entry=avg_entry, exit_price=exit_price, initial_sl=initial_sl,
                        stop_price=stop_price, qty=qty, units=units, total_entry_fee=total_entry_fee,
                        fee_rate=active_cfg.fee_rate, max_fav=max_fav, max_adv=max_adv,
                        risk_per_coin=risk_per_coin, holding_bars=hold_bars, reason=reason, risk_mult=entry_risk_mult,
                    )
                    if trades and str(reason).startswith("RANGE_EXIT"):
                        trades[-1].update(range_exit_meta)
                    annotate_last_trade_signal_addons(
                        trades, mode=signal_addon_mode, count=signal_addon_count,
                        risk_scale_sum=signal_addon_risk_scale_sum, last_reason=last_signal_addon_reason,
                    )
                    peak = max(peak, capital)
                    in_pos = False
                    side = 0
                    last_exit_i = i
                    pending_range_exit_i = None
                    pending_range_exit_reason = ""
                    pending_range_exit_meta = {}
                    signal_addon_count = 0
                    signal_addon_risk_scale_sum = 0.0
                    last_signal_addon_reason = "NONE"
                else:
                    stop_price = next_stop

                added_this_bar = False
                if (
                    in_pos
                    and signal_addon_mode != "off"
                    and pending_range_exit_i is None
                    and current_signal == side
                    and units < active_cfg.max_units
                    and signal_addon_count < int(signal_addon_max_count)
                    and math.isfinite(current_r)
                    and current_r >= float(signal_addon_min_current_r)
                ):
                    micro_aligned = _safe_bool(getattr(row, "micro_aligned", False))
                    micro_contra = _safe_bool(getattr(row, "micro_contra", False))
                    allowed = True
                    block_reasons = []
                    if signal_addon_require_micro_aligned and not micro_aligned:
                        allowed = False
                        block_reasons.append("NOT_MICRO_ALIGNED")
                    if signal_addon_block_micro_contra and micro_contra:
                        allowed = False
                        block_reasons.append("MICRO_CONTRA")
                    if allowed:
                        add_price = apply_entry_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                        add_stop_dist = max(active_cfg.initial_atr_mult * atr_value, risk_per_coin)
                        base_add_risk_mult = (
                            float(getattr(row, "risk_mult", entry_risk_mult))
                            * float(getattr(row, "quality_mult", 1.0))
                            * float(getattr(row, "micro_entry_risk_scale", 1.0))
                            * float(global_risk_scale)
                            * float(signal_addon_risk_scale)
                        )
                        add_q = unit_qty(capital, add_price, add_stop_dist, qty, active_cfg, base_add_risk_mult)
                        if add_q > 0 and math.isfinite(add_q):
                            total_entry_fee += add_q * add_price * active_cfg.fee_rate
                            avg_entry = weighted_avg_price(avg_entry, qty, add_price, add_q)
                            qty += add_q
                            units += 1
                            signal_addon_count += 1
                            signal_addon_risk_scale_sum += float(signal_addon_risk_scale)
                            last_signal_addon_reason = f"SAME_SIDE_SIGNAL_ADDON_CURRENT_R_{current_r:.2f}"
                            added_this_bar = True
                    elif block_reasons:
                        last_signal_addon_reason = "BLOCKED_" + "+".join(block_reasons)

                if in_pos and pending_range_exit_i is None and (not added_this_bar) and units < active_cfg.max_units:
                    next_unit_number = units + 1
                    trigger_r = (next_unit_number - 1) * active_cfg.add_every_r
                    add_triggered = high >= first_entry + trigger_r * risk_per_coin if side == 1 else low <= first_entry - trigger_r * risk_per_coin
                    if add_triggered:
                        add_price = apply_entry_slippage(float(rows[i + 1].open), side, active_cfg.slippage_pct)
                        add_stop_dist = max(active_cfg.initial_atr_mult * atr_value, risk_per_coin)
                        add_q = unit_qty(
                            capital, add_price, add_stop_dist, qty, active_cfg,
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
                    signal_addon_count = 0
                    signal_addon_risk_scale_sum = 0.0
                    last_signal_addon_reason = "NONE"

        equity_rows.append({"time": ts, "capital": capital, "drawdown_pct": (peak - capital) / peak if peak > 0 else 0.0})

    if in_pos:
        ts = idx[-1]
        close = float(df.iloc[-1]["close"])
        exit_price = apply_exit_slippage(close, side, pos_cfg.slippage_pct)
        capital = close_trade(
            trades=trades, capital=capital, side=side, entry_time=entry_time, exit_time=ts,
            first_entry=first_entry, avg_entry=avg_entry, exit_price=exit_price, initial_sl=initial_sl,
            stop_price=stop_price, qty=qty, units=units, total_entry_fee=total_entry_fee,
            fee_rate=pos_cfg.fee_rate, max_fav=max_fav, max_adv=max_adv,
            risk_per_coin=risk_per_coin, holding_bars=len(df) - 1 - entry_i,
            reason="FORCE_CLOSE_END", risk_mult=entry_risk_mult,
        )
        annotate_last_trade_signal_addons(
            trades, mode=signal_addon_mode, count=signal_addon_count,
            risk_scale_sum=signal_addon_risk_scale_sum, last_reason=last_signal_addon_reason,
        )
    equity = pd.DataFrame(equity_rows).set_index("time") if equity_rows else pd.DataFrame()
    return trades, equity


def save_run_outputs(run_dir: Path, features: pd.DataFrame, trades: list[dict[str, Any]], equity: pd.DataFrame, summary: dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trades).to_csv(run_dir / "trades.csv", index=False)
    if not equity.empty:
        equity.to_csv(run_dir / "equity.csv")
    with (run_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    # Keep a compact audit instead of exporting all feature columns by default.
    compact_cols = [
        c for c in [
            "open", "high", "low", "close", "atr", "adx", "atr_pct", "signal", "selected_engine",
            "signal_side", "risk_mult", "quality_mult", "micro_filter_action", "micro_aligned",
            "micro_contra", "micro_entry_risk_scale", "rf_imbalance", "rf_close_pos",
            "signal_lab_blocked", "signal_lab_block_reason",
        ] if c in features.columns
    ]
    if compact_cols:
        add_research_bins(features)[compact_cols].to_csv(run_dir / "signal_audit_compact.csv")


def run_filter_scenario(name: str, features: pd.DataFrame, cfg: Any, engine_cfgs: dict[str, Any], args: argparse.Namespace, rules: list[GroupRule]) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    filtered, meta = apply_filter_variant(features, name, rules, args)
    trades, equity = v9e.run_priority_backtest(filtered, cfg, engine_cfgs=engine_cfgs, global_risk_scale=args.global_risk_scale, args=args)
    trades = v9e.attach_engine_to_trades(trades, filtered)
    summary = summarize_run(name, trades, equity, cfg, meta)
    return summary, filtered, trades, equity


def run_signal_addon_scenario(name: str, features: pd.DataFrame, cfg: Any, engine_cfgs: dict[str, Any], args: argparse.Namespace, extra_rules: list[GroupRule] | None = None) -> tuple[dict[str, Any], pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    if extra_rules:
        filtered, filter_meta = apply_filter_variant(features, "block_bad_groups", extra_rules, args)
    else:
        filtered = add_research_bins(features)
        filter_meta = {"filter_variant": "none", "filtered_signal_count": 0, "filter_reason": "", "filter_rules": []}
    trades, equity = run_signal_addon_backtest(
        filtered,
        cfg,
        engine_cfgs,
        global_risk_scale=args.global_risk_scale,
        args=args,
        signal_addon_mode=name,
        signal_addon_min_current_r=args.signal_addon_min_current_r,
        signal_addon_risk_scale=args.signal_addon_risk_scale,
        signal_addon_max_count=args.signal_addon_max_count,
        signal_addon_require_micro_aligned=args.signal_addon_require_micro_aligned,
        signal_addon_block_micro_contra=args.signal_addon_block_micro_contra,
    )
    trades = v9e.attach_engine_to_trades(trades, filtered)
    meta = {
        "signal_addon_mode": name,
        "signal_addon_min_current_r": args.signal_addon_min_current_r,
        "signal_addon_risk_scale": args.signal_addon_risk_scale,
        "signal_addon_max_count": args.signal_addon_max_count,
        "signal_addon_require_micro_aligned": bool(args.signal_addon_require_micro_aligned),
        "signal_addon_block_micro_contra": bool(args.signal_addon_block_micro_contra),
        **filter_meta,
    }
    summary = summarize_run(name, trades, equity, cfg, meta)
    return summary, filtered, trades, equity


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    runs_dir = out_dir / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    features, cfg, engine_cfgs = build_features(args)
    baseline_trades, baseline_equity, baseline_summary = run_v9e_baseline(features, cfg, engine_cfgs, args)
    save_run_outputs(runs_dir / "baseline_v9e", add_research_bins(features), baseline_trades, baseline_equity, baseline_summary)

    ops = build_signal_opportunity_table(features, baseline_trades)
    ops.to_csv(out_dir / "v9e_signal_opportunity_table.csv", index=False)

    stats = group_stats(ops, min_samples=int(args.min_group_samples))
    stats.to_csv(out_dir / "v9e_signal_group_stats.csv", index=False)

    rules = mine_bad_entry_rules(stats, args)
    rules_df = pd.DataFrame([r.to_dict() for r in rules])
    rules_df.to_csv(out_dir / "v9e_bad_entry_rule_candidates.csv", index=False)

    scenario_rows: list[dict[str, Any]] = [baseline_summary]
    if not args.skip_variant_backtests:
        filter_variants = [
            "block_micro_contra",
            "block_not_aligned",
            "block_low_quality_q25",
            "block_low_adx_q25",
        ]
        if rules:
            filter_variants.append("block_bad_groups")
        for variant in filter_variants:
            summary, fts, trades, equity = run_filter_scenario(variant, features, cfg, engine_cfgs, args, rules)
            scenario_rows.append(summary)
            save_run_outputs(runs_dir / variant, fts, trades, equity, summary)

        addon_summary, addon_features, addon_trades, addon_equity = run_signal_addon_scenario(
            "same_side_signal_addon", features, cfg, engine_cfgs, args
        )
        scenario_rows.append(addon_summary)
        save_run_outputs(runs_dir / "same_side_signal_addon", addon_features, addon_trades, addon_equity, addon_summary)

        if rules:
            combo_summary, combo_features, combo_trades, combo_equity = run_signal_addon_scenario(
                "bad_groups_plus_same_side_addon", features, cfg, engine_cfgs, args, extra_rules=rules
            )
            scenario_rows.append(combo_summary)
            save_run_outputs(runs_dir / "bad_groups_plus_same_side_addon", combo_features, combo_trades, combo_equity, combo_summary)

    summary_df = pd.DataFrame(scenario_rows)
    preferred_cols = [
        "scenario", "total_trades", "closed_total_trades", "win_rate", "closed_win_rate",
        "final_capital", "closed_final_capital", "profit_factor", "closed_profit_factor",
        "expectancy_pct", "closed_expectancy_pct", "max_drawdown_pct", "total_return_pct",
        "force_close_count", "force_close_pnl", "filtered_signal_count", "filter_reason",
        "signal_addon_trade_count", "signal_addon_total_count", "signal_addon_min_current_r", "signal_addon_risk_scale",
    ]
    cols = [c for c in preferred_cols if c in summary_df.columns] + [c for c in summary_df.columns if c not in preferred_cols]
    summary_df = summary_df[cols]
    summary_df.to_csv(out_dir / "v9e_variant_summary.csv", index=False)

    lab_summary = {
        "strategy": "V9E signal opportunity lab",
        "symbol": args.symbol,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "portfolio_signal_count": int(len(ops)),
        "executed_entry_count": int(ops["action_taken"].eq("EXECUTED_ENTRY").sum()) if not ops.empty else 0,
        "ignored_same_side_in_position_count": int(ops["action_taken"].eq("IGNORED_SAME_SIDE_IN_POSITION").sum()) if not ops.empty else 0,
        "used_opposite_exit_signal_count": int(ops["action_taken"].eq("USED_OPPOSITE_EXIT_SIGNAL").sum()) if not ops.empty else 0,
        "bad_entry_rule_count": int(len(rules)),
        "bad_entry_rules": [r.to_dict() for r in rules],
        "outputs": {
            "opportunity_table": str(out_dir / "v9e_signal_opportunity_table.csv"),
            "group_stats": str(out_dir / "v9e_signal_group_stats.csv"),
            "bad_rule_candidates": str(out_dir / "v9e_bad_entry_rule_candidates.csv"),
            "variant_summary": str(out_dir / "v9e_variant_summary.csv"),
            "runs_dir": str(runs_dir),
        },
        "warning": "Bad-entry rules are mined in-sample. Treat as hypotheses and validate with walk-forward / no-2026 / fee2x / slippage2x before live use.",
    }
    with (out_dir / "v9e_lab_summary.json").open("w", encoding="utf-8") as f:
        json.dump(lab_summary, f, ensure_ascii=False, indent=2, default=str)

    print("\n" + "=" * 100)
    print("V9E Signal Opportunity Lab completed")
    print("=" * 100)
    print(f"Output directory: {out_dir.resolve()}")
    print("\nTop variant summary:")
    display_cols = [c for c in [
        "scenario", "closed_final_capital", "closed_profit_factor", "closed_win_rate",
        "closed_expectancy_pct", "max_drawdown_pct", "filtered_signal_count", "signal_addon_total_count",
    ] if c in summary_df.columns]
    if display_cols:
        print(summary_df[display_cols].to_string(index=False))
    print("=" * 100 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
