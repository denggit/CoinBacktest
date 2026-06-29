#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V10 Microstructure Feature Discovery Lab
========================================

Research-only feature discovery lab for ETH LF Portfolio V9E/V10 family.

Purpose
-------
Find more micro / range-bar / footprint / 4H candle-structure regimes that may
improve win rate without introducing future leakage or overfitting.

Important no-lookahead design
-----------------------------
1. Candidate input features are computed only from the closed signal bar and
   bars before it. If a feature needs a threshold, it uses fixed thresholds or
   shifted rolling/expanding past-only thresholds.
2. Future returns, MFE, and MAE are written only as labels for evaluation. They
   are never used to compute candidate feature values.
3. The script is a discovery tool, not a final strategy. Candidate rules must be
   validated later by chronological portfolio backtest + walk-forward / stress tests.

Outputs
-------
- v10_micro_all_signal_events.csv
- v10_micro_engine_side_overview.csv
- v10_micro_feature_group_stats.csv
- v10_micro_feature_contrast.csv
- v10_micro_feature_yearly_stats.csv
- v10_micro_bad_filter_candidates.csv
- v10_micro_good_regime_candidates.csv
- v10_micro_top_winner_overlap.csv
- v10_micro_feature_dictionary.csv
- v10_micro_discovery_meta.json
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

ENGINE_MOM = "MOMENTUM_V3"
ENGINE_BEAR = "BEAR_V3_ONLY"
ENGINE_BULL = "BULL_RECLAIM_V2"
ENGINES = (ENGINE_BULL, ENGINE_BEAR, ENGINE_MOM)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Past-only microstructure feature discovery lab for V9E/V10 ETH LF portfolio.")

    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-15")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--warmup-days", type=int, default=365)
    p.add_argument("--initial-capital", type=float, default=1000.0)

    # V9E-compatible strategy params.
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

    # Discovery controls.
    p.add_argument("--out-dir", default="data/reports/research/v10_microstructure_feature_discovery_lab")
    p.add_argument("--horizons", default="1,3,6,12", help="Comma-separated future 4H horizons used only as labels.")
    p.add_argument("--primary-horizon", type=int, default=12)
    p.add_argument("--rolling-window-bars", type=int, default=1080, help="Past-only quantile window. 1080 4H bars ~= 180 days.")
    p.add_argument("--volume-median-window", type=int, default=120)
    p.add_argument("--atr-window", type=int, default=42)
    p.add_argument("--ema-fast", type=int, default=20)
    p.add_argument("--ema-slow", type=int, default=50)
    p.add_argument("--prev-breakout-window", type=int, default=20)
    p.add_argument("--min-count", type=int, default=8)
    p.add_argument("--top-n-winners", type=int, default=5)
    p.add_argument("--write-all-events", action="store_true", default=True)
    p.add_argument("--no-write-all-events", dest="write_all_events", action="store_false")
    return p.parse_args()


def _parse_horizons(text: str) -> list[int]:
    out: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        n = int(part)
        if n <= 0:
            raise ValueError("horizons must be positive integers")
        out.append(n)
    return sorted(set(out)) or [1, 3, 6, 12]


def _num(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")


def _bool(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype("boolean").fillna(False).astype(bool)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce").replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _side_name(side: int) -> str:
    return "LONG" if int(side) == 1 else "SHORT" if int(side) == -1 else "FLAT"


def _pf(ret_pct: pd.Series) -> float:
    x = pd.to_numeric(ret_pct, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    gross_profit = float(x[x > 0].sum())
    gross_loss = float(-x[x < 0].sum())
    if gross_loss <= 0:
        return float("inf") if gross_profit > 0 else float("nan")
    return gross_profit / gross_loss


def _fixed_bin(s: pd.Series, bins: list[float], labels: list[str]) -> pd.Series:
    return pd.cut(pd.to_numeric(s, errors="coerce"), bins=bins, labels=labels, include_lowest=True).astype("object").fillna("NA")


def _rolling_past_q(s: pd.Series, window: int, q: float, min_periods: int | None = None) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mp = min_periods if min_periods is not None else max(20, min(window // 5, window))
    return x.shift(1).rolling(window, min_periods=mp).quantile(q)


def _rolling_past_median(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mp = min_periods if min_periods is not None else max(20, min(window // 5, window))
    return x.shift(1).rolling(window, min_periods=mp).median()


def build_features(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    mom_cfg = v9e.make_momentum_config(args)
    bear_cfg = v9e.make_bear_config(args)
    bull_cfg = v9e.make_bull_config(args)

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
    bear = v9e.build_bear_features(base, bear_cfg)
    bull = v9e.build_bull_features(base, bull_cfg)
    baseline_selected = v9e.select_portfolio_signals(momentum, bear, bull, args)
    micro_ctx = v9e.load_range_footprint_context(args, load_start_str, args.end_date)
    baseline = v9e.apply_micro_context_filter(baseline_selected, micro_ctx, args)
    baseline = baseline.loc[trade_start: pd.Timestamp(args.end_date)].copy().sort_index()
    raw = {ENGINE_MOM: momentum, ENGINE_BEAR: bear, ENGINE_BULL: bull}
    print(f"Feature rows after warmup slice: {len(baseline):,}; first={baseline.index[0] if len(baseline) else 'NA'}", flush=True)
    return baseline, raw


def add_past_only_micro_features(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = df.copy().sort_index()
    span = (_num(out, "high") - _num(out, "low")).replace(0, np.nan)
    body = (_num(out, "close") - _num(out, "open")).abs()
    upper = _num(out, "high") - pd.concat([_num(out, "open"), _num(out, "close")], axis=1).max(axis=1)
    lower = pd.concat([_num(out, "open"), _num(out, "close")], axis=1).min(axis=1) - _num(out, "low")

    out["candle_close_pos"] = _safe_div(_num(out, "close") - _num(out, "low"), span)
    out["candle_body_pct"] = _safe_div(body, span)
    out["upper_wick_pct"] = _safe_div(upper, span)
    out["lower_wick_pct"] = _safe_div(lower, span)
    out["candle_body_dir"] = np.where(_num(out, "close") > _num(out, "open"), "UP", np.where(_num(out, "close") < _num(out, "open"), "DOWN", "DOJI"))
    out["candle_close_pos_bin"] = _fixed_bin(out["candle_close_pos"], [-np.inf, 0.20, 0.35, 0.65, 0.80, np.inf], ["BOTTOM_20", "LOW_20_35", "MID_35_65", "HIGH_65_80", "TOP_20"])
    out["candle_body_bin"] = _fixed_bin(out["candle_body_pct"], [-np.inf, 0.20, 0.45, 0.70, np.inf], ["SMALL_BODY", "MID_BODY", "LARGE_BODY", "FULL_BODY"])
    out["upper_wick_big"] = out["upper_wick_pct"].ge(0.45)
    out["lower_wick_big"] = out["lower_wick_pct"].ge(0.45)

    prev_close = _num(out, "close").shift(1)
    tr = pd.concat([
        _num(out, "high") - _num(out, "low"),
        (_num(out, "high") - prev_close).abs(),
        (_num(out, "low") - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["tr"] = tr
    atr_past = _rolling_past_median(tr, int(args.atr_window), min_periods=max(10, int(args.atr_window) // 2))
    out["tr_to_past_atr"] = _safe_div(tr, atr_past)
    out["range_expansion_past_q75"] = out["tr_to_past_atr"].ge(_rolling_past_q(out["tr_to_past_atr"], int(args.rolling_window_bars), 0.75))
    out["range_compression_past_q25"] = out["tr_to_past_atr"].le(_rolling_past_q(out["tr_to_past_atr"], int(args.rolling_window_bars), 0.25))

    vol = _num(out, "volume")
    vol_med = _rolling_past_median(vol, int(args.volume_median_window), min_periods=max(20, int(args.volume_median_window) // 3))
    out["volume_ratio_past"] = _safe_div(vol, vol_med)
    out["low_volume_past_q25"] = out["volume_ratio_past"].le(_rolling_past_q(out["volume_ratio_past"], int(args.rolling_window_bars), 0.25))
    out["high_volume_past_q75"] = out["volume_ratio_past"].ge(_rolling_past_q(out["volume_ratio_past"], int(args.rolling_window_bars), 0.75))
    out["volume_ratio_bin"] = _fixed_bin(out["volume_ratio_past"], [-np.inf, 0.70, 1.00, 1.50, np.inf], ["LOW_LT_0P7", "NORMAL_0P7_1P0", "HIGH_1P0_1P5", "VERY_HIGH_GT_1P5"])

    n = int(args.prev_breakout_window)
    prev_high_n = _num(out, "high").shift(1).rolling(n, min_periods=max(5, n // 2)).max()
    prev_low_n = _num(out, "low").shift(1).rolling(n, min_periods=max(5, n // 2)).min()
    out["break_prev_high_n"] = _num(out, "high").gt(prev_high_n)
    out["break_prev_low_n"] = _num(out, "low").lt(prev_low_n)
    out["close_above_prev_high_n"] = _num(out, "close").gt(prev_high_n)
    out["close_below_prev_low_n"] = _num(out, "close").lt(prev_low_n)
    out["failed_up_break_n"] = out["break_prev_high_n"] & (~out["close_above_prev_high_n"])
    out["failed_down_break_n"] = out["break_prev_low_n"] & (~out["close_below_prev_low_n"])
    out["inside_bar"] = _num(out, "high").le(_num(out, "high").shift(1)) & _num(out, "low").ge(_num(out, "low").shift(1))
    out["outside_bar"] = _num(out, "high").ge(_num(out, "high").shift(1)) & _num(out, "low").le(_num(out, "low").shift(1))

    close = _num(out, "close")
    ema_fast = close.ewm(span=int(args.ema_fast), min_periods=max(5, int(args.ema_fast) // 2), adjust=False).mean()
    ema_slow = close.ewm(span=int(args.ema_slow), min_periods=max(10, int(args.ema_slow) // 2), adjust=False).mean()
    out["close_above_ema_fast"] = close.gt(ema_fast)
    out["close_below_ema_fast"] = close.lt(ema_fast)
    out["close_above_ema_slow"] = close.gt(ema_slow)
    out["close_below_ema_slow"] = close.lt(ema_slow)
    out["ema_fast_above_slow"] = ema_fast.gt(ema_slow)
    out["dist_ema_fast_pct"] = _safe_div(close - ema_fast, ema_fast) * 100.0

    # Range / footprint features. All are current completed 4H bucket or past-only thresholds.
    out["rf_close_pos_bin"] = _fixed_bin(_num(out, "rf_close_pos"), [-np.inf, 0.20, 0.35, 0.65, 0.80, np.inf], ["BOTTOM_20", "LOW_20_35", "MID_35_65", "HIGH_65_80", "TOP_20"])
    imb = _num(out, "rf_imbalance")
    out["rf_imbalance_bin"] = pd.Series("NA", index=out.index, dtype="object")
    out.loc[imb <= -0.15, "rf_imbalance_bin"] = "SELL_STRONG_LE_-0P15"
    out.loc[(imb > -0.15) & (imb <= -0.05), "rf_imbalance_bin"] = "SELL_-0P15_-0P05"
    out.loc[(imb > -0.05) & (imb < 0.05), "rf_imbalance_bin"] = "NEUTRAL_-0P05_0P05"
    out.loc[(imb >= 0.05) & (imb < 0.15), "rf_imbalance_bin"] = "BUY_0P05_0P15"
    out.loc[imb >= 0.15, "rf_imbalance_bin"] = "BUY_STRONG_GE_0P15"

    rf_count = _num(out, "rf_bar_count")
    rf_count_med = _rolling_past_median(rf_count, int(args.rolling_window_bars), min_periods=100)
    out["rf_bar_count_ratio_past"] = _safe_div(rf_count, rf_count_med)
    out["rf_bar_count_low_past_q25"] = rf_count.le(_rolling_past_q(rf_count, int(args.rolling_window_bars), 0.25, min_periods=100))
    out["rf_bar_count_high_past_q75"] = rf_count.ge(_rolling_past_q(rf_count, int(args.rolling_window_bars), 0.75, min_periods=100))
    out["rf_speed_bin"] = np.select(
        [out["rf_bar_count_low_past_q25"], out["rf_bar_count_high_past_q75"]],
        ["SLOW_Q1", "FAST_Q4"],
        default="NORMAL_Q2_Q3",
    )

    rf_notional = _num(out, "rf_notional_sum")
    if rf_notional.notna().sum() == 0:
        rf_notional = _num(out, "rf_buy_notional_sum").fillna(0.0) + _num(out, "rf_sell_notional_sum").fillna(0.0)
    out["rf_notional_ratio_past"] = _safe_div(rf_notional, _rolling_past_median(rf_notional, int(args.rolling_window_bars), min_periods=100))
    out["rf_notional_high"] = out["rf_notional_ratio_past"].ge(1.50)
    out["rf_result_small"] = _num(out, "rf_micro_return_pct").abs().le(float(args.range_pct) * 0.75)
    out["rf_effort_no_result"] = out["rf_notional_high"] & out["rf_result_small"]
    out["rf_buy_absorption"] = _num(out, "rf_imbalance").ge(0.05) & _num(out, "rf_close_pos").le(0.50)
    out["rf_buy_absorption_strong"] = _num(out, "rf_imbalance").ge(0.15) & _num(out, "rf_close_pos").le(0.50)
    out["rf_sell_absorption"] = _num(out, "rf_imbalance").le(-0.05) & _num(out, "rf_close_pos").ge(0.50)
    out["rf_sell_absorption_strong"] = _num(out, "rf_imbalance").le(-0.15) & _num(out, "rf_close_pos").ge(0.50)

    out["rf_max_buy_bucket_high"] = _num(out, "rf_max_buy_bucket_share").ge(0.35)
    out["rf_max_sell_bucket_high"] = _num(out, "rf_max_sell_bucket_share").ge(0.35)
    return out


def _engine_micro_action(sig: pd.Series, baseline: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    sig = pd.to_numeric(sig, errors="coerce").fillna(0).astype(int)
    has_ctx = _bool(baseline, "micro_context_available")
    imb = _num(baseline, "rf_imbalance")
    pos = _num(baseline, "rf_close_pos")
    aligned_imb = abs(float(args.micro_aligned_imbalance))
    contra_imb = abs(float(args.micro_contra_imbalance))
    good_pos = float(args.micro_good_close_pos)
    bad_pos = float(args.micro_bad_close_pos)
    long_sig = sig.eq(1)
    short_sig = sig.eq(-1)
    aligned = (long_sig & has_ctx & imb.ge(aligned_imb) & pos.ge(good_pos)) | (short_sig & has_ctx & imb.le(-aligned_imb) & pos.le(1.0 - good_pos))
    contra = (long_sig & has_ctx & imb.le(-contra_imb) & pos.le(bad_pos)) | (short_sig & has_ctx & imb.ge(contra_imb) & pos.ge(1.0 - bad_pos))
    action = pd.Series("NEUTRAL", index=baseline.index, dtype="object")
    action.loc[aligned] = "ALIGNED"
    action.loc[contra] = "CONTRA"
    action.loc[sig.ne(0) & has_ctx & (~aligned) & (~contra)] = "NOT_ALIGNED_RISK_REDUCED"
    action.loc[sig.eq(0)] = "NO_SIGNAL"
    return action


def _side_bool(side: int, long_condition: Any, short_condition: Any) -> Any:
    return long_condition if int(side) == 1 else short_condition


def build_signal_events(baseline: pd.DataFrame, raw: dict[str, pd.DataFrame], args: argparse.Namespace, horizons: list[int]) -> pd.DataFrame:
    base = add_past_only_micro_features(baseline, args)
    rows: list[dict[str, Any]] = []
    max_h = max(horizons)
    idx = base.index
    engine_actions: dict[str, pd.Series] = {}
    for engine in ENGINES:
        sig = _num(raw[engine].reindex(idx), "signal", 0.0).fillna(0).astype(int)
        engine_actions[engine] = _engine_micro_action(sig, base, args)

    for engine in ENGINES:
        rdf = raw[engine].reindex(idx)
        sig = _num(rdf, "signal", 0.0).fillna(0).astype(int)
        active_locs = np.flatnonzero(sig.to_numpy() != 0)
        for i in active_locs:
            if i + 1 >= len(base):
                continue
            side = int(sig.iloc[i])
            ts = idx[i]
            row = base.iloc[i]
            entry_open = float(base["open"].iloc[i + 1])
            if not math.isfinite(entry_open) or entry_open <= 0:
                continue
            item: dict[str, Any] = {
                "timestamp": ts,
                "year": int(pd.Timestamp(ts).year),
                "engine": engine,
                "side": _side_name(side),
                "side_int": side,
                "selected_engine_baseline": str(row.get("selected_engine", "NONE")),
                "selected_by_baseline_router": bool(str(row.get("selected_engine", "NONE")) == engine and int(row.get("signal", 0)) == side),
                "portfolio_signal_same_side": bool(int(row.get("signal", 0)) == side),
                "micro_action_engine": str(engine_actions[engine].iloc[i]),
                "risk_mult_raw": float(_num(rdf, "risk_mult", 1.0).iloc[i]) if "risk_mult" in rdf.columns else np.nan,
                "quality_mult_raw": float(_num(rdf, "quality_mult", 1.0).iloc[i]) if "quality_mult" in rdf.columns else np.nan,
                "entry_open_next_bar": entry_open,
            }
            # General candle/range features.
            feature_cols = [
                "candle_close_pos", "candle_body_pct", "upper_wick_pct", "lower_wick_pct", "candle_body_dir",
                "candle_close_pos_bin", "candle_body_bin", "upper_wick_big", "lower_wick_big",
                "tr_to_past_atr", "range_expansion_past_q75", "range_compression_past_q25",
                "volume_ratio_past", "low_volume_past_q25", "high_volume_past_q75", "volume_ratio_bin",
                "break_prev_high_n", "break_prev_low_n", "close_above_prev_high_n", "close_below_prev_low_n",
                "failed_up_break_n", "failed_down_break_n", "inside_bar", "outside_bar",
                "close_above_ema_fast", "close_below_ema_fast", "close_above_ema_slow", "close_below_ema_slow",
                "ema_fast_above_slow", "dist_ema_fast_pct",
                "rf_bar_count", "rf_micro_return_pct", "rf_close_pos", "rf_delta_sum", "rf_imbalance", "rf_taker_buy_ratio",
                "rf_max_sell_bucket_share", "rf_max_buy_bucket_share", "rf_close_pos_bin", "rf_imbalance_bin",
                "rf_bar_count_ratio_past", "rf_bar_count_low_past_q25", "rf_bar_count_high_past_q75", "rf_speed_bin",
                "rf_notional_ratio_past", "rf_notional_high", "rf_result_small", "rf_effort_no_result",
                "rf_buy_absorption", "rf_buy_absorption_strong", "rf_sell_absorption", "rf_sell_absorption_strong",
                "rf_max_buy_bucket_high", "rf_max_sell_bucket_high", "micro_context_available",
            ]
            for col in feature_cols:
                item[col] = row.get(col, np.nan)

            # Side-aware features, still available at the signal close.
            item["signal_body_aligned"] = bool((side == 1 and row.get("candle_body_dir") == "UP") or (side == -1 and row.get("candle_body_dir") == "DOWN"))
            item["signal_close_strong"] = bool((side == 1 and row.get("candle_close_pos", np.nan) >= 0.65) or (side == -1 and row.get("candle_close_pos", np.nan) <= 0.35))
            item["signal_close_weak"] = bool((side == 1 and row.get("candle_close_pos", np.nan) <= 0.50) or (side == -1 and row.get("candle_close_pos", np.nan) >= 0.50))
            item["signal_wick_bad"] = bool((side == 1 and row.get("upper_wick_pct", np.nan) >= 0.45) or (side == -1 and row.get("lower_wick_pct", np.nan) >= 0.45))
            item["signal_failed_breakout"] = bool((side == 1 and bool(row.get("failed_up_break_n", False))) or (side == -1 and bool(row.get("failed_down_break_n", False))))
            item["signal_breakout_close_confirmed"] = bool((side == 1 and bool(row.get("close_above_prev_high_n", False))) or (side == -1 and bool(row.get("close_below_prev_low_n", False))))
            item["signal_ema_fast_aligned"] = bool((side == 1 and bool(row.get("close_above_ema_fast", False))) or (side == -1 and bool(row.get("close_below_ema_fast", False))))
            item["signal_ema_slow_aligned"] = bool((side == 1 and bool(row.get("close_above_ema_slow", False))) or (side == -1 and bool(row.get("close_below_ema_slow", False))))
            item["signal_rf_close_good"] = bool((side == 1 and row.get("rf_close_pos", np.nan) >= 0.65) or (side == -1 and row.get("rf_close_pos", np.nan) <= 0.35))
            item["signal_rf_close_bad"] = bool((side == 1 and row.get("rf_close_pos", np.nan) <= 0.35) or (side == -1 and row.get("rf_close_pos", np.nan) >= 0.65))
            item["signal_rf_imbalance_aligned"] = bool((side == 1 and row.get("rf_imbalance", np.nan) >= 0.05) or (side == -1 and row.get("rf_imbalance", np.nan) <= -0.05))
            item["signal_rf_imbalance_contra"] = bool((side == 1 and row.get("rf_imbalance", np.nan) <= -0.05) or (side == -1 and row.get("rf_imbalance", np.nan) >= 0.05))
            item["signal_rf_return_aligned"] = bool(side * float(row.get("rf_micro_return_pct", 0.0) or 0.0) > 0)
            item["signal_rf_absorption_bad"] = bool((side == 1 and bool(row.get("rf_buy_absorption", False))) or (side == -1 and bool(row.get("rf_sell_absorption", False))))
            item["signal_rf_absorption_strong_bad"] = bool((side == 1 and bool(row.get("rf_buy_absorption_strong", False))) or (side == -1 and bool(row.get("rf_sell_absorption_strong", False))))
            item["signal_max_bucket_absorption_bad"] = bool((side == 1 and bool(row.get("rf_max_buy_bucket_high", False)) and row.get("rf_close_pos", np.nan) <= 0.50) or (side == -1 and bool(row.get("rf_max_sell_bucket_high", False)) and row.get("rf_close_pos", np.nan) >= 0.50))

            for h in horizons:
                if i + h >= len(base):
                    item[f"ret_{h}bar_pct"] = np.nan
                    item[f"win_{h}bar"] = np.nan
                    item[f"mfe_{h}bar_pct"] = np.nan
                    item[f"mae_{h}bar_pct"] = np.nan
                    continue
                future_close = float(base["close"].iloc[i + h])
                window = base.iloc[i + 1: i + h + 1]
                if window.empty:
                    item[f"ret_{h}bar_pct"] = np.nan
                    item[f"win_{h}bar"] = np.nan
                    item[f"mfe_{h}bar_pct"] = np.nan
                    item[f"mae_{h}bar_pct"] = np.nan
                    continue
                ret = side * (future_close / entry_open - 1.0) * 100.0
                max_high = float(window["high"].max())
                min_low = float(window["low"].min())
                if side == 1:
                    mfe = (max_high / entry_open - 1.0) * 100.0
                    mae = (min_low / entry_open - 1.0) * 100.0
                else:
                    mfe = (entry_open / max(min_low, 1e-12) - 1.0) * 100.0
                    mae = (entry_open / max(max_high, 1e-12) - 1.0) * 100.0
                item[f"ret_{h}bar_pct"] = ret
                item[f"win_{h}bar"] = bool(ret > 0)
                item[f"mfe_{h}bar_pct"] = mfe
                item[f"mae_{h}bar_pct"] = mae
            rows.append(item)
    events = pd.DataFrame(rows)
    if not events.empty:
        events = events.sort_values(["timestamp", "engine", "side"]).reset_index(drop=True)
    return events


def _agg_stats(g: pd.DataFrame, h: int) -> dict[str, Any]:
    ret = pd.to_numeric(g[f"ret_{h}bar_pct"], errors="coerce")
    valid = ret.dropna()
    wins = valid > 0
    return {
        "count": int(valid.size),
        "win_rate": float(wins.mean() * 100.0) if valid.size else np.nan,
        "avg_ret_pct": float(valid.mean()) if valid.size else np.nan,
        "median_ret_pct": float(valid.median()) if valid.size else np.nan,
        "profit_factor": _pf(valid),
        "avg_mfe_pct": float(pd.to_numeric(g[f"mfe_{h}bar_pct"], errors="coerce").mean()) if f"mfe_{h}bar_pct" in g else np.nan,
        "avg_mae_pct": float(pd.to_numeric(g[f"mae_{h}bar_pct"], errors="coerce").mean()) if f"mae_{h}bar_pct" in g else np.nan,
    }


def build_overview(events: pd.DataFrame, h: int) -> pd.DataFrame:
    rows = []
    if events.empty:
        return pd.DataFrame()
    for keys, g in events.groupby(["engine", "side"], dropna=False):
        row = {"engine": keys[0], "side": keys[1]}
        row.update(_agg_stats(g, h))
        for short_h in [1, 3, 6, h]:
            col = f"ret_{short_h}bar_pct"
            if col in g.columns:
                row[f"avg_ret_{short_h}bar_pct"] = float(pd.to_numeric(g[col], errors="coerce").mean())
                row[f"win_rate_{short_h}bar"] = float(pd.to_numeric(g[col], errors="coerce").gt(0).mean() * 100.0)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["engine", "side"])


def feature_columns() -> list[str]:
    return [
        # Engine/router context
        "selected_by_baseline_router", "portfolio_signal_same_side", "micro_action_engine",
        # 4H candle structure
        "candle_close_pos_bin", "candle_body_bin", "upper_wick_big", "lower_wick_big",
        "signal_body_aligned", "signal_close_strong", "signal_close_weak", "signal_wick_bad",
        "range_expansion_past_q75", "range_compression_past_q25", "inside_bar", "outside_bar",
        "signal_failed_breakout", "signal_breakout_close_confirmed", "signal_ema_fast_aligned", "signal_ema_slow_aligned",
        # Volume and volatility, all past-only thresholds
        "low_volume_past_q25", "high_volume_past_q75", "volume_ratio_bin",
        # Range/footprint fixed or past-only features
        "rf_close_pos_bin", "rf_imbalance_bin", "rf_speed_bin", "rf_bar_count_low_past_q25", "rf_bar_count_high_past_q75",
        "rf_notional_high", "rf_result_small", "rf_effort_no_result",
        "signal_rf_close_good", "signal_rf_close_bad", "signal_rf_imbalance_aligned", "signal_rf_imbalance_contra",
        "signal_rf_return_aligned", "signal_rf_absorption_bad", "signal_rf_absorption_strong_bad", "signal_max_bucket_absorption_bad",
    ]


def feature_dictionary_rows() -> list[dict[str, str]]:
    return [
        {"feature": "micro_action_engine", "source": "range/footprint", "lookahead_safe": "yes", "description": "Engine-side micro context label computed from the completed 4H range-footprint bucket."},
        {"feature": "low_volume_past_q25", "source": "4H volume", "lookahead_safe": "yes", "description": "Current volume_ratio is below the shifted rolling past 25% quantile."},
        {"feature": "high_volume_past_q75", "source": "4H volume", "lookahead_safe": "yes", "description": "Current volume_ratio is above the shifted rolling past 75% quantile."},
        {"feature": "signal_wick_bad", "source": "4H candle", "lookahead_safe": "yes", "description": "Long: large upper wick; Short: large lower wick. Uses signal bar after close."},
        {"feature": "signal_failed_breakout", "source": "4H candle", "lookahead_safe": "yes", "description": "Long: broke prior N-bar high but did not close above it. Short symmetric. Prior high/low uses shift(1)."},
        {"feature": "rf_speed_bin", "source": "range bars", "lookahead_safe": "yes", "description": "Current 4H range-bar count vs shifted rolling past range-bar count quantiles."},
        {"feature": "rf_effort_no_result", "source": "range/footprint", "lookahead_safe": "yes", "description": "High range notional versus past median but small 4H range micro return."},
        {"feature": "signal_rf_absorption_bad", "source": "range/footprint", "lookahead_safe": "yes", "description": "Long: buy imbalance but weak range close. Short: sell imbalance but strong range close."},
        {"feature": "signal_rf_close_bad", "source": "range/footprint", "lookahead_safe": "yes", "description": "Long: range bucket closes low. Short: range bucket closes high."},
        {"feature": "signal_rf_imbalance_contra", "source": "range/footprint", "lookahead_safe": "yes", "description": "Footprint imbalance is against signal direction using fixed +/-0.05 threshold."},
    ]


def _feature_value_series(events: pd.DataFrame, feature: str) -> pd.Series:
    s = events[feature]
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False).astype(bool).astype(str)
    return s.astype("object").where(~s.isna(), "NA").astype(str)


def build_feature_group_stats(events: pd.DataFrame, h: int, features: list[str], min_count: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        if feature not in events.columns:
            continue
        tmp = events.copy()
        tmp["feature_value"] = _feature_value_series(tmp, feature)
        for keys, g in tmp.groupby(["engine", "side", "feature_value"], dropna=False):
            if len(g) < min_count:
                continue
            row = {"feature": feature, "engine": keys[0], "side": keys[1], "feature_value": keys[2]}
            row.update(_agg_stats(g, h))
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["engine", "side", "feature", "feature_value"]).reset_index(drop=True)
    return out


def build_feature_contrast(events: pd.DataFrame, h: int, features: list[str], min_count: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        if feature not in events.columns:
            continue
        tmp = events.copy()
        tmp["feature_value"] = _feature_value_series(tmp, feature)
        for (engine, side), eg in tmp.groupby(["engine", "side"], dropna=False):
            values = sorted(v for v in eg["feature_value"].dropna().unique().tolist() if v != "NA")
            for value in values:
                in_g = eg[eg["feature_value"].eq(value)]
                out_g = eg[~eg["feature_value"].eq(value)]
                if len(in_g) < min_count or len(out_g) < min_count:
                    continue
                a = _agg_stats(in_g, h)
                b = _agg_stats(out_g, h)
                row = {
                    "feature": feature,
                    "feature_value": value,
                    "engine": engine,
                    "side": side,
                    "condition_count": a["count"],
                    "complement_count": b["count"],
                    "condition_win_rate": a["win_rate"],
                    "complement_win_rate": b["win_rate"],
                    "condition_avg_ret_pct": a["avg_ret_pct"],
                    "complement_avg_ret_pct": b["avg_ret_pct"],
                    "condition_pf": a["profit_factor"],
                    "complement_pf": b["profit_factor"],
                    "condition_avg_mfe_pct": a["avg_mfe_pct"],
                    "condition_avg_mae_pct": a["avg_mae_pct"],
                    "delta_avg_ret_pct": a["avg_ret_pct"] - b["avg_ret_pct"],
                    "delta_win_rate": a["win_rate"] - b["win_rate"],
                    "delta_pf": a["profit_factor"] - b["profit_factor"] if math.isfinite(float(a["profit_factor"])) and math.isfinite(float(b["profit_factor"])) else np.nan,
                }
                rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["engine", "side", "feature", "feature_value"]).reset_index(drop=True)
    return out


def build_feature_yearly(events: pd.DataFrame, h: int, features: list[str], min_count: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature in features:
        if feature not in events.columns:
            continue
        tmp = events.copy()
        tmp["feature_value"] = _feature_value_series(tmp, feature)
        for keys, g in tmp.groupby(["engine", "side", "feature_value", "year"], dropna=False):
            if len(g) < max(2, min_count // 3):
                continue
            row = {"feature": feature, "engine": keys[0], "side": keys[1], "feature_value": keys[2], "year": int(keys[3])}
            row.update(_agg_stats(g, h))
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["engine", "side", "feature", "feature_value", "year"]).reset_index(drop=True)
    return out


def build_top_winner_overlap(events: pd.DataFrame, h: int, features: list[str], top_n: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if events.empty:
        return pd.DataFrame()
    ret_col = f"ret_{h}bar_pct"
    events2 = events.copy()
    events2["_ret"] = pd.to_numeric(events2[ret_col], errors="coerce")
    for (engine, side), eg in events2.dropna(subset=["_ret"]).groupby(["engine", "side"]):
        top = eg.nlargest(int(top_n), "_ret")
        if top.empty:
            continue
        for feature in features:
            if feature not in top.columns:
                continue
            all_values = _feature_value_series(events2.loc[events2["engine"].eq(engine) & events2["side"].eq(side)], feature).unique().tolist()
            top_values = _feature_value_series(top, feature)
            for value in sorted(v for v in all_values if v != "NA"):
                hit = int(top_values.eq(value).sum())
                rows.append({
                    "engine": engine,
                    "side": side,
                    "feature": feature,
                    "feature_value": value,
                    "top_n": int(top_n),
                    "top_winner_hit_count": hit,
                    "top_winner_hit_rate": hit / max(len(top), 1),
                    "top_winner_avg_ret_pct": float(top["_ret"].mean()),
                })
    return pd.DataFrame(rows)


def build_candidates(contrast: pd.DataFrame, yearly: pd.DataFrame, top_overlap: pd.DataFrame, min_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if contrast.empty:
        return pd.DataFrame(), pd.DataFrame()

    yr = yearly.copy()
    if not yr.empty:
        yr_summary = yr.groupby(["engine", "side", "feature", "feature_value"], dropna=False).agg(
            years_with_data=("year", "nunique"),
            positive_years=("avg_ret_pct", lambda x: int((pd.to_numeric(x, errors="coerce") > 0).sum())),
            negative_years=("avg_ret_pct", lambda x: int((pd.to_numeric(x, errors="coerce") < 0).sum())),
            min_year_avg_ret_pct=("avg_ret_pct", "min"),
            max_year_avg_ret_pct=("avg_ret_pct", "max"),
        ).reset_index()
    else:
        yr_summary = pd.DataFrame(columns=["engine", "side", "feature", "feature_value"])

    top = top_overlap[["engine", "side", "feature", "feature_value", "top_winner_hit_count", "top_winner_hit_rate"]].copy() if not top_overlap.empty else pd.DataFrame(columns=["engine", "side", "feature", "feature_value"])
    df = contrast.merge(yr_summary, on=["engine", "side", "feature", "feature_value"], how="left")
    df = df.merge(top, on=["engine", "side", "feature", "feature_value"], how="left")
    df["top_winner_hit_count"] = pd.to_numeric(df.get("top_winner_hit_count"), errors="coerce").fillna(0).astype(int)
    df["top_winner_hit_rate"] = pd.to_numeric(df.get("top_winner_hit_rate"), errors="coerce").fillna(0.0)
    df["years_with_data"] = pd.to_numeric(df.get("years_with_data"), errors="coerce").fillna(0).astype(int)
    df["positive_years"] = pd.to_numeric(df.get("positive_years"), errors="coerce").fillna(0).astype(int)
    df["negative_years"] = pd.to_numeric(df.get("negative_years"), errors="coerce").fillna(0).astype(int)

    bad_mask = (
        (df["condition_count"] >= min_count)
        & (df["condition_avg_ret_pct"] < 0)
        & (df["condition_pf"] < 1.0)
        & (df["delta_avg_ret_pct"] < -0.50)
        & (df["condition_avg_ret_pct"] < df["complement_avg_ret_pct"])
    )
    bad = df.loc[bad_mask].copy()
    if not bad.empty:
        bad["candidate_score"] = (
            (-bad["condition_avg_ret_pct"].clip(upper=0))
            * np.sqrt(bad["condition_count"].clip(lower=1))
            * (1.0 - bad["top_winner_hit_rate"].clip(0, 1))
        )
        bad["warning"] = np.where(bad["top_winner_hit_count"] > 0, "CHECK_TOP_WINNER_OVERLAP", "")
        bad = bad.sort_values(["candidate_score", "condition_count"], ascending=[False, False])

    good_mask = (
        (df["condition_count"] >= min_count)
        & (df["condition_avg_ret_pct"] > 0)
        & (df["condition_pf"] > 1.75)
        & (df["delta_avg_ret_pct"] > 0.75)
        & (df["condition_win_rate"] >= 50.0)
    )
    good = df.loc[good_mask].copy()
    if not good.empty:
        good["candidate_score"] = (
            good["condition_avg_ret_pct"].clip(lower=0)
            * np.sqrt(good["condition_count"].clip(lower=1))
            * (good["condition_win_rate"].clip(0, 100) / 100.0)
        )
        good["warning"] = np.where(good["years_with_data"] < 2, "LOW_YEAR_COVERAGE", "")
        good = good.sort_values(["candidate_score", "condition_count"], ascending=[False, False])
    return bad, good


def write_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    horizons = _parse_horizons(args.horizons)
    if args.primary_horizon not in horizons:
        horizons = sorted(set(horizons + [args.primary_horizon]))
    h = int(args.primary_horizon)

    baseline, raw = build_features(args)
    events = build_signal_events(baseline, raw, args, horizons)
    features = feature_columns()

    overview = build_overview(events, h)
    group_stats = build_feature_group_stats(events, h, features, int(args.min_count))
    contrast = build_feature_contrast(events, h, features, int(args.min_count))
    yearly = build_feature_yearly(events, h, features, int(args.min_count))
    top_overlap = build_top_winner_overlap(events, h, features, int(args.top_n_winners))
    bad, good = build_candidates(contrast, yearly, top_overlap, int(args.min_count))

    if args.write_all_events:
        events.to_csv(out_dir / "v10_micro_all_signal_events.csv", index=False)
    overview.to_csv(out_dir / "v10_micro_engine_side_overview.csv", index=False)
    group_stats.to_csv(out_dir / "v10_micro_feature_group_stats.csv", index=False)
    contrast.to_csv(out_dir / "v10_micro_feature_contrast.csv", index=False)
    yearly.to_csv(out_dir / "v10_micro_feature_yearly_stats.csv", index=False)
    top_overlap.to_csv(out_dir / "v10_micro_top_winner_overlap.csv", index=False)
    bad.to_csv(out_dir / "v10_micro_bad_filter_candidates.csv", index=False)
    good.to_csv(out_dir / "v10_micro_good_regime_candidates.csv", index=False)
    pd.DataFrame(feature_dictionary_rows()).to_csv(out_dir / "v10_micro_feature_dictionary.csv", index=False)

    meta = {
        "symbol": args.symbol,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "horizons": horizons,
        "primary_horizon": h,
        "rolling_window_bars": args.rolling_window_bars,
        "volume_median_window": args.volume_median_window,
        "min_count": args.min_count,
        "top_n_winners": args.top_n_winners,
        "event_count": int(len(events)),
        "feature_count": int(len(features)),
        "bad_candidate_count": int(len(bad)),
        "good_candidate_count": int(len(good)),
        "no_lookahead_policy": [
            "features use current closed signal bar and historical bars only",
            "rolling thresholds are shifted by one bar before quantile/median",
            "future returns/MFE/MAE are labels only and not input features",
            "candidate rules require chronological portfolio backtest and walk-forward before V10/V11 inclusion",
        ],
    }
    write_json(out_dir / "v10_micro_discovery_meta.json", meta)

    print("\n" + "=" * 96)
    print("V10 Microstructure Feature Discovery Lab complete")
    print("=" * 96)
    print(f"events: {len(events):,}")
    print(f"bad filter candidates: {len(bad):,}")
    print(f"good regime candidates: {len(good):,}")
    if not bad.empty:
        print("\nTop bad-filter candidates:")
        cols = ["engine", "side", "feature", "feature_value", "condition_count", "condition_win_rate", "condition_avg_ret_pct", "condition_pf", "delta_avg_ret_pct", "top_winner_hit_count", "candidate_score"]
        print(bad[[c for c in cols if c in bad.columns]].head(15).to_string(index=False))
    if not good.empty:
        print("\nTop good-regime candidates:")
        cols = ["engine", "side", "feature", "feature_value", "condition_count", "condition_win_rate", "condition_avg_ret_pct", "condition_pf", "delta_avg_ret_pct", "candidate_score"]
        print(good[[c for c in cols if c in good.columns]].head(15).to_string(index=False))
    print(f"\nOutput directory: {out_dir.resolve()}")
    print("=" * 96 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
