#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Lead Flywheel Research - single-file CoinBacktest copy-trading research script.

Rangefix build: fixes duplicate-index joins for range-primary research and makes
range-bar evaluation actually runnable instead of failing during context joins.

Design goals
------------
1) Single research file, matching the existing CoinBacktest research style.
2) Reuse CoinBacktest data interfaces; no custom local DB/trades loader here.
3) No future function in tradable signals:
   - rolling thresholds that define events use shift(1) where the current bar
     would otherwise contaminate the threshold;
   - signals are evaluated only after a closed bar;
   - entries are executed at the next bar open, plus optional delay bars;
   - add/scale decisions are also closed-bar decisions executed next bar open;
   - if TP and SL are both inside the same OHLC bar, SL is assumed first.
4) Research factory, not a final live strategy. It exports candidate summaries,
   yearly stats, robustness checks and optional trades for manual review.

Default data source
-------------------
Uses existing project loaders:
    src.data_feed.okx_trade_bar_loader.OKXTradeBarLoader
    src.data_feed.okx_range_bar_loader.OKXRangeBarLoader      (optional context)
    src.data_feed.okx_range_footprint_loader.OKXRangeFootprintLoader (optional, light aggregation)

Typical commands from CoinBacktest root
---------------------------------------
python research/eth_lead_flywheel_research.py --mode smoke --max-specs 30 --out-dir data/reports/research/eth_lead_flywheel_smoke
python research/eth_lead_flywheel_research.py --mode core --max-specs 400 --out-dir data/reports/research/eth_lead_flywheel_core
python research/eth_lead_flywheel_research.py --mode wide --max-specs 1200 --robustness-top-n 80 --write-trades --out-dir data/reports/research/eth_lead_flywheel_wide

Notes
-----
- Default round-trip cost is fee_rate_per_side * 2 = 0.11%, matching the current
  project fee assumption when fee_rate_per_side=0.00055.
- Books are intentionally not used.
- This file is a new research line, not V4 of the earlier signal factory.
- It is optimized for copy-trading metrics: partial TP rate, green-touch rate,
  max days without trade, adverse excursion and robustness under delay/fees.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_FILE))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SCRIPT_NAME = "eth_lead_flywheel_research"


# =============================================================================
# Config / specs
# =============================================================================


@dataclass(frozen=True)
class StrategySpec:
    spec_id: str
    entry_model: str
    regime: str
    structure: str
    system: str = "baseline"
    layer: str = "comfort"  # comfort | expansion | hybrid | sanity
    confirmation: str = "none"
    signal_frame: str = "primary"  # primary | tf5m | tf15m | tf30m | tf1H | tf4H
    side_mode: str = "both"  # both | long_only | short_only
    swing_window: int = 240
    trend_fast: int = 60
    trend_slow: int = 240
    vol_window: int = 240
    stop_atr_mult: float = 1.25
    min_stop_pct: float = 0.0018
    tp_r: float = 1.30
    max_hold_bars: int = 90
    confirm_bars: int = 1
    add_trigger_r1: float = 1.0
    add_trigger_r2: float = 2.0
    add_size_1: float = 0.50
    add_size_2: float = 0.35
    partial_tp_r: float = 1.0
    partial_fraction: float = 0.50
    trail_atr_mult: float = 1.20
    time_bomb_bars: int = 18
    time_bomb_min_mfe_r: float = 0.30
    fail_fast_bars: int = 8
    fail_fast_adverse_r: float = 0.45
    initial_size_mult: float = 1.0


@dataclass(frozen=True)
class RunConfig:
    symbol: str = "ETH-USDT-SWAP"
    timeframe: str = "1m"
    primary_frame: str = "time"  # time | range
    context_timeframes: str = "5m,15m"
    start_date: str = "2023-01-01"
    end_date: str = "2026-06-30"
    warmup_start_date: str = "2022-01-01"
    data_dir: str | None = None
    mode: str = "core"
    max_specs: int | None = None
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.0020
    max_notional_mult: float = 3.0
    fee_rate_per_side: float = 0.00055
    slippage_pct: float = 0.00015
    entry_delay_bars: int = 0
    local_only: bool = False
    build_missing_cache: bool = True
    include_range_context: bool = False
    range_pct: float = 0.0020
    include_footprint_context: bool = False
    price_step: float = 1.0
    chunksize: int = 300_000
    min_signal_gap_bars: int = 5
    max_signals_per_spec: int = 0
    include_sanity_in_core: bool = False
    robustness_top_n: int = 50
    write_trades: bool = False
    verify_fast_exactness: bool = False
    verify_fast_exactness_specs: int = 30
    fail_on_empty_signals: bool = True
    fail_on_empty_audit: bool = True
    out_dir: str = "data/reports/research/eth_lead_flywheel_research"


STRUCTURE_PRESETS: dict[str, dict[str, Any]] = {
    # Baselines kept for comparability with the earlier intraday factory.
    "fixed_fast": dict(tp_r=1.05, max_hold_bars=35, stop_atr_mult=1.15),
    "fixed_balanced": dict(tp_r=1.40, max_hold_bars=75, stop_atr_mult=1.25),
    "fixed_wide": dict(tp_r=2.00, max_hold_bars=150, stop_atr_mult=1.50),
    "time_bomb": dict(tp_r=1.25, max_hold_bars=60, time_bomb_bars=15, time_bomb_min_mfe_r=0.35),
    "fail_fast": dict(tp_r=1.30, max_hold_bars=70, fail_fast_bars=7, fail_fast_adverse_r=0.35),
    "partial_runner": dict(tp_r=3.00, partial_tp_r=1.00, partial_fraction=0.55, max_hold_bars=180, trail_atr_mult=1.10),
    "probe_confirm_add": dict(tp_r=2.20, add_trigger_r1=0.65, add_size_1=0.55, add_size_2=0.0, max_hold_bars=130),
    "anti_martingale_1r": dict(tp_r=3.00, add_trigger_r1=1.0, add_trigger_r2=2.0, add_size_1=0.50, add_size_2=0.35, max_hold_bars=210),
    "breakout_add_runner": dict(tp_r=3.50, add_trigger_r1=1.20, add_size_1=0.40, add_size_2=0.25, max_hold_bars=240, trail_atr_mult=1.35),
    "slow_runner": dict(tp_r=4.00, partial_tp_r=1.20, partial_fraction=0.40, max_hold_bars=360, trail_atr_mult=1.60),

    # Copy-trading flywheel structures: probe -> dopamine partial TP -> BE -> optional pyramid.
    "flywheel_dopamine": dict(initial_size_mult=0.55, tp_r=1.20, partial_tp_r=0.60, partial_fraction=0.60, max_hold_bars=55, stop_atr_mult=1.05, fail_fast_bars=8, fail_fast_adverse_r=0.35),
    "flywheel_probe": dict(initial_size_mult=0.45, tp_r=1.60, partial_tp_r=0.75, partial_fraction=0.55, add_trigger_r1=0.90, add_size_1=0.45, add_size_2=0.0, max_hold_bars=95, stop_atr_mult=1.10, fail_fast_bars=10, fail_fast_adverse_r=0.40),
    "flywheel_runner": dict(initial_size_mult=0.60, tp_r=3.00, partial_tp_r=0.80, partial_fraction=0.45, add_trigger_r1=1.05, add_trigger_r2=2.00, add_size_1=0.40, add_size_2=0.25, max_hold_bars=180, stop_atr_mult=1.20, trail_atr_mult=1.15, fail_fast_bars=12, fail_fast_adverse_r=0.50),
    "comfort_failfast": dict(initial_size_mult=0.60, tp_r=1.05, partial_tp_r=0.55, partial_fraction=0.50, max_hold_bars=45, stop_atr_mult=1.00, fail_fast_bars=6, fail_fast_adverse_r=0.30),
    "expansion_pyramid": dict(initial_size_mult=0.75, tp_r=3.80, partial_tp_r=1.00, partial_fraction=0.35, add_trigger_r1=1.00, add_trigger_r2=2.00, add_size_1=0.50, add_size_2=0.30, max_hold_bars=240, stop_atr_mult=1.25, trail_atr_mult=1.25),
}

SANITY_ENTRY_MODELS = [
    "ema20_cross",
    "vwap_cross",
    "price_momentum_sanity",
    "prior_20_breakout_sanity",
    "ret_impulse_continuation",
    "or_breakout_close",
]

EDGE_ENTRY_MODELS = [
    "sweep_reclaim",
    "failed_breakout_reclaim",
    "trend_pullback_reclaim",
    "vwap_deviation_reversion",
    "cvd_divergence_reversal",
    "liquidation_panic_reversal",
    "compression_breakout_retest",
    "microtrend_continuation",
    "opening_range_fakeout",
    "opening_range_breakout",
    "range_bar_momentum_proxy",
    "range_momentum_burst",
    "range_sweep_reclaim",
    "range_pullback_reclaim",
    "range_speed_reversal",
]

LEAD_ENTRY_MODELS = [
    "lead_trend_pullback_flywheel",
    "lead_failed_breakout_vwap",
    "lead_sweep_absorption_proxy",
    "lead_range_burst_pullback",
    "lead_vwap_rotation_transition",
    "lead_opening_fakeout_comfort",
]

ENTRY_MODELS = SANITY_ENTRY_MODELS + EDGE_ENTRY_MODELS + LEAD_ENTRY_MODELS

REGIMES = [
    "any",
    "trend_aligned",
    "range_only",
    "high_vol",
    "low_vol",
    "asia_session",
    "eu_us_session",
]


# =============================================================================
# Generic helpers
# =============================================================================


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(float(default), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(float(default))


def _bool(df: pd.DataFrame, col: str, default: bool = False) -> pd.Series:
    if col not in df.columns:
        return pd.Series(bool(default), index=df.index)
    return df[col].astype("boolean").fillna(bool(default)).astype(bool)


def _safe_div(num: pd.Series, den: pd.Series, default: float = np.nan) -> pd.Series:
    out = num.astype(float) / den.replace(0, np.nan).astype(float)
    return out.replace([np.inf, -np.inf], default)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity / peak.replace(0, np.nan) - 1.0
    return float(dd.min()) if not dd.empty else 0.0


def _profit_factor(pnl: pd.Series) -> float:
    pnl = pd.to_numeric(pnl, errors="coerce").fillna(0.0)
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl <= 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else 0.0
    return gp / gl


def _parse_ts(x: Any) -> pd.Timestamp:
    return pd.Timestamp(x).tz_localize(None) if getattr(pd.Timestamp(x), "tzinfo", None) else pd.Timestamp(x)


def _window_start_ts(value: Any) -> pd.Timestamp:
    """Inclusive backtest start. Date-only strings mean 00:00:00."""
    return pd.Timestamp(value).tz_localize(None) if pd.Timestamp(value).tzinfo else pd.Timestamp(value)


def _window_end_ts(value: Any) -> pd.Timestamp:
    """Inclusive backtest end. Date-only strings mean the full calendar day.

    Pandas Timestamp('2026-06-30') is midnight, which would otherwise include
    only the first minute of the final day. Existing CoinBacktest research uses
    date ranges as full-day ranges, so match that behavior here.
    """
    ts = pd.Timestamp(value).tz_localize(None) if pd.Timestamp(value).tzinfo else pd.Timestamp(value)
    text = str(value).strip()
    if len(text) <= 10 and ts == ts.normalize():
        return ts + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return ts


def _progress_month_marks(start_date: str, end_date: str) -> list[pd.Timestamp]:
    cur = pd.Timestamp(start_date).normalize() + pd.DateOffset(months=1)
    end = pd.Timestamp(end_date).normalize()
    marks: list[pd.Timestamp] = []
    while cur <= end:
        marks.append(cur)
        cur += pd.DateOffset(months=1)
    if not marks or marks[-1] < end:
        marks.append(end)
    return marks


# =============================================================================
# Existing CoinBacktest data interfaces only
# =============================================================================




def _sanitize_timeframe(value: str) -> str:
    return str(value).replace(" ", "").replace("/", "").replace("-", "").replace("_", "")


def _context_prefix(timeframe: str) -> str:
    return f"tf{_sanitize_timeframe(timeframe)}"


def _parse_context_timeframes(value: str | None) -> list[str]:
    if not value:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for part in str(value).split(","):
        tf = part.strip()
        if not tf or tf in seen:
            continue
        seen.add(tf)
        out.append(tf)
    return out


def _signal_frames_from_cfg(cfg: RunConfig) -> list[str]:
    frames = ["primary"]
    for tf in _parse_context_timeframes(cfg.context_timeframes):
        if cfg.primary_frame == "time" and tf == cfg.timeframe:
            continue
        frames.append(_context_prefix(tf))
    return frames


def _series_name(frame: str, col: str) -> str:
    return col if frame == "primary" else f"{frame}_{col}"


def _entry_uses_primary_only(entry_model: str) -> bool:
    return entry_model.startswith("range_") or entry_model == "range_bar_momentum_proxy"

def load_trade_bars(cfg: RunConfig) -> pd.DataFrame:
    """Load 1m/5m trade bars through the existing OKXTradeBarLoader."""
    from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: WPS433

    print(
        f"[1/5] Loading OKX trade bars via OKXTradeBarLoader: {cfg.symbol} {cfg.timeframe} "
        f"{cfg.warmup_start_date}->{cfg.end_date}",
        flush=True,
    )
    loader = OKXTradeBarLoader(symbol=cfg.symbol, timeframe=cfg.timeframe, data_dir=cfg.data_dir)
    if cfg.local_only or not cfg.build_missing_cache:
        df = loader.load_local_data(start_date=cfg.warmup_start_date, end_date=cfg.end_date)
        if df.empty:
            raise RuntimeError(
                "No local trade bars found in existing CoinBacktest cache. "
                "Either run tools/prebuild_okx_trade_bars.py first or rerun this research without --local-only."
            )
    else:
        df = loader.fetch_data_by_date_range(
            cfg.warmup_start_date,
            cfg.end_date,
            chunksize=cfg.chunksize,
            force_rebuild=False,
            cvd_mode="range",
        )
    if df.empty:
        raise RuntimeError("OKXTradeBarLoader returned empty data.")

    df = df.copy().sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df.index = pd.to_datetime(df["timestamp"])
        elif "end_ts" in df.columns:
            df.index = pd.to_datetime(df["end_ts"])
        else:
            raise RuntimeError("Trade bars have no DatetimeIndex/timestamp/end_ts.")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise RuntimeError(f"Trade bars missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_index().copy()
    print(f"      rows={len(df):,} index={df.index[0]} -> {df.index[-1]}", flush=True)
    return df



def load_range_bars_primary(cfg: RunConfig) -> pd.DataFrame:
    """Load range bars as the tradable primary frame via existing OKXRangeBarLoader."""
    from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader  # noqa: WPS433

    print(
        f"[1/5] Loading PRIMARY range bars via OKXRangeBarLoader: {cfg.symbol} range_pct={cfg.range_pct} "
        f"{cfg.warmup_start_date}->{cfg.end_date}",
        flush=True,
    )
    loader = OKXRangeBarLoader(symbol=cfg.symbol, range_pct=cfg.range_pct, data_dir=cfg.data_dir)
    if cfg.local_only or not cfg.build_missing_cache:
        df = loader.load_local_data(start_date=cfg.warmup_start_date, end_date=cfg.end_date)
        if df.empty:
            raise RuntimeError(
                "No local range bars found in existing CoinBacktest cache. "
                "Run the range-bar prebuild tool first or rerun without --local-only."
            )
    else:
        df = loader.fetch_data_by_date_range(
            cfg.warmup_start_date,
            cfg.end_date,
            chunksize=cfg.chunksize,
            force_rebuild=False,
            cvd_mode="range",
        )
    if df.empty:
        raise RuntimeError("OKXRangeBarLoader returned empty data.")
    df = df.copy().sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "end_ts" in df.columns:
            df.index = pd.to_datetime(df["end_ts"])
        elif "timestamp" in df.columns:
            df.index = pd.to_datetime(df["timestamp"])
        else:
            raise RuntimeError("Range bars have no DatetimeIndex/end_ts/timestamp.")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise RuntimeError(f"Range bars missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_index().copy()
    print(f"      rows={len(df):,} index={df.index[0]} -> {df.index[-1]}", flush=True)
    return df


def load_primary_bars(cfg: RunConfig) -> pd.DataFrame:
    if cfg.primary_frame == "time":
        return load_trade_bars(cfg)
    if cfg.primary_frame == "range":
        return load_range_bars_primary(cfg)
    raise ValueError("--primary-frame must be time or range")


def _build_context_feature_frame(raw: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Build causal context features on an auxiliary time frame, then prefix columns.

    These features are aligned backward to the primary frame, so a primary bar only
    sees the latest already-closed context bar.  They are context, not a new data
    loader or execution engine.
    """
    x = raw.copy().sort_index()
    x.index = pd.to_datetime(x.index).tz_localize(None)
    x = x[~x.index.duplicated(keep="last")].sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in x.columns:
            x[col] = 0.0
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
    open_ = x["open"]
    high = x["high"]
    low = x["low"]
    close = x["close"]
    volume = x["volume"]
    buy_notional = pd.to_numeric(x.get("buy_notional", 0.0), errors="coerce").fillna(0.0)
    sell_notional = pd.to_numeric(x.get("sell_notional", 0.0), errors="coerce").fillna(0.0)
    delta_notional = pd.to_numeric(x.get("delta_notional", 0.0), errors="coerce").fillna(0.0)
    notional = (buy_notional + sell_notional).replace(0, np.nan)
    if notional.isna().all():
        notional = (close * volume).replace(0, np.nan)
    span = (high - low).replace(0, np.nan)
    feat = pd.DataFrame(index=x.index)
    feat["open"] = open_
    feat["high"] = high
    feat["low"] = low
    feat["close"] = close
    feat["volume"] = volume
    feat["ret_1"] = close.pct_change().fillna(0.0)
    feat["bar_range_pct"] = _safe_div(high - low, close, 0.0).fillna(0.0)
    feat["close_pos"] = _safe_div(close - low, span, 0.5).clip(0.0, 1.0).fillna(0.5)
    feat["upper_wick_pct"] = _safe_div(high - np.maximum(open_, close), span, 0.0).clip(0.0, 1.0).fillna(0.0)
    feat["lower_wick_pct"] = _safe_div(np.minimum(open_, close) - low, span, 0.0).clip(0.0, 1.0).fillna(0.0)
    tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    feat["atr_60"] = tr.rolling(60, min_periods=20).mean()
    feat["ema_20"] = close.ewm(span=20, adjust=False, min_periods=5).mean()
    feat["ema_60"] = close.ewm(span=60, adjust=False, min_periods=15).mean()
    feat["ema_240"] = close.ewm(span=240, adjust=False, min_periods=60).mean()
    feat["ema_slope_60"] = feat["ema_60"].pct_change(20).fillna(0.0)
    feat["trend_up"] = (feat["ema_60"] > feat["ema_240"]) & (feat["ema_slope_60"] > 0)
    feat["trend_down"] = (feat["ema_60"] < feat["ema_240"]) & (feat["ema_slope_60"] < 0)
    day = x.index.date
    px_notional = (close * volume).where(volume > 0, close)
    cum_vol = volume.groupby(day).cumsum().replace(0, np.nan)
    cum_pv = px_notional.groupby(day).cumsum()
    feat["session_vwap"] = (cum_pv / cum_vol).fillna(close)
    feat["vwap_dist_pct"] = _safe_div(close - feat["session_vwap"], close, 0.0).fillna(0.0)
    feat["delta_ratio"] = _safe_div(delta_notional, notional, 0.0).clip(-1.0, 1.0).fillna(0.0)
    feat["cvd"] = pd.to_numeric(x.get("cvd_notional", delta_notional.cumsum()), errors="coerce").fillna(0.0)
    for w in [60, 120, 240, 480]:
        feat[f"prior_high_{w}"] = high.shift(1).rolling(w, min_periods=max(20, w // 4)).max()
        feat[f"prior_low_{w}"] = low.shift(1).rolling(w, min_periods=max(20, w // 4)).min()
    feat["vol_q75_past"] = volume.shift(1).rolling(240, min_periods=60).quantile(0.75)
    feat["vol_q90_past"] = volume.shift(1).rolling(240, min_periods=60).quantile(0.90)
    feat["range_q30_past"] = feat["bar_range_pct"].shift(1).rolling(240, min_periods=60).quantile(0.30)
    feat["range_q70_past"] = feat["bar_range_pct"].shift(1).rolling(240, min_periods=60).quantile(0.70)
    feat["range_q90_past"] = feat["bar_range_pct"].shift(1).rolling(240, min_periods=60).quantile(0.90)
    feat["vwap_dev_q90_past"] = feat["vwap_dist_pct"].abs().shift(1).rolling(480, min_periods=120).quantile(0.90)
    feat["ret_abs_q70_past"] = feat["ret_1"].abs().shift(1).rolling(240, min_periods=60).quantile(0.70)
    feat["ret_abs_q90_past"] = feat["ret_1"].abs().shift(1).rolling(480, min_periods=120).quantile(0.90)
    feat["range_regime"] = (~feat["trend_up"]) & (~feat["trend_down"]) & (feat["bar_range_pct"] < feat["range_q70_past"])
    session_key = pd.Series(pd.DatetimeIndex(feat.index).normalize(), index=feat.index)
    session_min = feat.groupby(session_key, sort=False).cumcount().astype("int32")
    feat["session_minute"] = session_min
    first_hour = session_min < 60
    feat["opening_range_high"] = high.where(first_hour).groupby(session_key, sort=False).transform("max")
    feat["opening_range_low"] = low.where(first_hour).groupby(session_key, sort=False).transform("min")
    feat["after_opening_range"] = session_min >= 60
    feat["prior_20_high"] = high.shift(1).rolling(20, min_periods=5).max()
    feat["prior_20_low"] = low.shift(1).rolling(20, min_periods=5).min()
    return feat.add_prefix(f"{prefix}_")


def load_timeframe_contexts(cfg: RunConfig, base_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Load optional 5m/15m/etc context bars and align backward to primary frame.

    Range bars can have duplicate end timestamps.  A normal DataFrame.join on a
    duplicate primary index can become a many-to-many join and explode memory.
    Therefore this function always returns a frame with exactly len(base_index)
    rows and later joins are done positionally, not via index merge.
    """
    frames = _parse_context_timeframes(cfg.context_timeframes)
    base_index = pd.DatetimeIndex(pd.to_datetime(base_index).tz_localize(None))
    if not frames:
        return pd.DataFrame(index=base_index)

    parts: list[pd.DataFrame] = []
    from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: WPS433
    for tf in frames:
        if cfg.primary_frame == "time" and tf == cfg.timeframe:
            continue
        prefix = _context_prefix(tf)
        print(f"[2c/5] Loading optional time-frame context via OKXTradeBarLoader: {tf} -> {prefix}", flush=True)
        loader = OKXTradeBarLoader(symbol=cfg.symbol, timeframe=tf, data_dir=cfg.data_dir)
        if cfg.local_only or not cfg.build_missing_cache:
            ctx = loader.load_local_data(start_date=cfg.warmup_start_date, end_date=cfg.end_date)
            if ctx.empty:
                print(f"      {tf} context empty; continuing without it", flush=True)
                continue
        else:
            ctx = loader.fetch_data_by_date_range(
                cfg.warmup_start_date,
                cfg.end_date,
                chunksize=cfg.chunksize,
                force_rebuild=False,
                cvd_mode="range",
            )
        if ctx.empty:
            continue
        if not isinstance(ctx.index, pd.DatetimeIndex):
            if "timestamp" in ctx.columns:
                ctx.index = pd.to_datetime(ctx["timestamp"])
            elif "end_ts" in ctx.columns:
                ctx.index = pd.to_datetime(ctx["end_ts"])
            else:
                print(f"      {tf} context has no datetime index; skipped", flush=True)
                continue
        ctx.index = pd.to_datetime(ctx.index).tz_localize(None)
        ctx = ctx[~ctx.index.duplicated(keep="last")].sort_index()
        feat = _build_context_feature_frame(ctx, prefix)
        aligned = feat.sort_index().reindex(base_index, method="ffill").fillna(0.0)
        # Keep row order, but avoid duplicate-index joins later.
        parts.append(aligned.reset_index(drop=True))
        print(f"      {tf} context rows={len(ctx):,}; features={aligned.shape[1]}", flush=True)

    if not parts:
        return pd.DataFrame(index=base_index)
    out = pd.concat(parts, axis=1)
    out.index = base_index
    return out.fillna(0.0)

def load_range_context(cfg: RunConfig, base_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Optional context from the existing OKXRangeBarLoader.

    For time-primary research, range bars are aggregated to minutes and aligned
    to the primary frame.  For range-primary research, the primary bars already
    contain native range-bar columns such as direction/duration/delta, so this
    function intentionally returns empty context to avoid self-joining duplicate
    range-bar timestamps.
    """
    base_index = pd.DatetimeIndex(pd.to_datetime(base_index).tz_localize(None))
    if not cfg.include_range_context or cfg.primary_frame == "range":
        if cfg.include_range_context and cfg.primary_frame == "range":
            print("[2/5] Skipping optional range context because primary frame is already range bars", flush=True)
        return pd.DataFrame(index=base_index)
    from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader  # noqa: WPS433

    print(
        f"[2/5] Loading optional range context via OKXRangeBarLoader: range_pct={cfg.range_pct}",
        flush=True,
    )
    loader = OKXRangeBarLoader(symbol=cfg.symbol, range_pct=cfg.range_pct, data_dir=cfg.data_dir)
    if cfg.local_only or not cfg.build_missing_cache:
        rb = loader.load_local_data(start_date=cfg.warmup_start_date, end_date=cfg.end_date)
        if rb.empty:
            print("      range context empty; continuing without it", flush=True)
            return pd.DataFrame(index=base_index)
    else:
        rb = loader.fetch_data_by_date_range(
            cfg.warmup_start_date,
            cfg.end_date,
            chunksize=cfg.chunksize,
            force_rebuild=False,
            cvd_mode="range",
        )
    if rb.empty:
        return pd.DataFrame(index=base_index)

    x = rb.copy()
    if not isinstance(x.index, pd.DatetimeIndex):
        if "end_ts" in x.columns:
            x.index = pd.to_datetime(x["end_ts"])
        elif "timestamp" in x.columns:
            x.index = pd.to_datetime(x["timestamp"])
        else:
            print("      range context has no datetime index; skipped", flush=True)
            return pd.DataFrame(index=base_index)
    x.index = pd.to_datetime(x.index).tz_localize(None)
    x = x.sort_index()
    minute = x.index.floor("min")

    # Fast vectorized minute aggregation. Avoid groupby.apply over millions of range bars.
    for col in ["direction", "delta_notional", "notional", "close", "low", "high"]:
        if col not in x.columns:
            x[col] = 0.0
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
    span = (x["high"] - x["low"]).replace(0, np.nan)
    x["_rf_close_pos"] = ((x["close"] - x["low"]) / span).clip(0.0, 1.0).fillna(0.5)
    agg = x.groupby(minute, sort=False).agg(
        rf_bar_count=("close", "size"),
        rf_direction_sum=("direction", "sum"),
        rf_delta_notional_sum=("delta_notional", "sum"),
        rf_notional_sum=("notional", "sum"),
        rf_close_pos_mean=("_rf_close_pos", "mean"),
    ).sort_index()
    agg.index = pd.to_datetime(agg.index).tz_localize(None)
    aligned = agg.reindex(base_index.floor("min"), method="ffill").fillna(0.0)
    aligned.index = base_index
    aligned["rf_imbalance"] = _safe_div(aligned["rf_delta_notional_sum"], aligned["rf_notional_sum"].abs(), 0.0).fillna(0.0)
    print(f"      range rows={len(x):,}; aligned_rows={int((aligned['rf_bar_count'] > 0).sum()):,}", flush=True)
    return aligned

def load_footprint_context(cfg: RunConfig, base_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Optional light footprint aggregation via existing OKXRangeFootprintLoader.

    This stays intentionally small: one-row-per-minute max buy/sell bucket pressure.
    It is disabled by default because range-bar context is cheaper and enough for
    first-pass strategy-factory screening.
    """
    base_index = pd.DatetimeIndex(pd.to_datetime(base_index).tz_localize(None))
    if not cfg.include_footprint_context:
        return pd.DataFrame(index=base_index)
    from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader  # noqa: WPS433

    print(
        f"[2b/5] Loading optional footprint context via OKXRangeFootprintLoader: range_pct={cfg.range_pct} step={cfg.price_step}",
        flush=True,
    )
    loader = OKXRangeFootprintLoader(symbol=cfg.symbol, range_pct=cfg.range_pct, price_step=cfg.price_step, data_dir=cfg.data_dir)
    if cfg.local_only or not cfg.build_missing_cache:
        fp = loader.load_local_data(start_date=cfg.warmup_start_date, end_date=cfg.end_date)
        if fp.empty:
            print("      footprint context empty; continuing without it", flush=True)
            return pd.DataFrame(index=base_index)
    else:
        fp = loader.fetch_data_by_date_range(
            cfg.warmup_start_date,
            cfg.end_date,
            chunksize=cfg.chunksize,
            force_rebuild=False,
        )
    if fp.empty:
        return pd.DataFrame(index=base_index)
    x = fp.copy()
    ts_col = "end_ts" if "end_ts" in x.columns else None
    if ts_col is None:
        return pd.DataFrame(index=base_index)
    x[ts_col] = pd.to_datetime(x[ts_col]).dt.tz_localize(None)
    x["minute"] = x[ts_col].dt.floor("min")
    for col in ["buy_notional", "sell_notional", "large_buy_notional", "large_sell_notional", "delta_notional"]:
        if col not in x:
            x[col] = 0.0
        x[col] = pd.to_numeric(x[col], errors="coerce").fillna(0.0)
    g = x.groupby("minute")
    agg = pd.DataFrame({
        "fp_buy_notional_max": g["buy_notional"].max(),
        "fp_sell_notional_max": g["sell_notional"].max(),
        "fp_large_buy_sum": g["large_buy_notional"].sum(),
        "fp_large_sell_sum": g["large_sell_notional"].sum(),
        "fp_delta_sum": g["delta_notional"].sum(),
    })
    agg = agg.sort_index()
    agg.index = pd.to_datetime(agg.index).tz_localize(None)
    ctx = agg.reindex(base_index.floor("min"), method="ffill").fillna(0.0)
    ctx.index = base_index
    ctx["fp_absorption_hint"] = _safe_div(ctx["fp_large_buy_sum"] - ctx["fp_large_sell_sum"], ctx["fp_large_buy_sum"] + ctx["fp_large_sell_sum"], 0.0).fillna(0.0)
    print(f"      footprint rows={len(x):,}; aligned_rows={int((ctx.abs().sum(axis=1) > 0).sum()):,}", flush=True)
    return ctx


# =============================================================================
# Feature engineering: closed-bar / past-only thresholds
# =============================================================================


def _append_context_columns_by_position(base: pd.DataFrame, ctx: pd.DataFrame, label: str) -> pd.DataFrame:
    """Append context columns without index joins.

    Range-bar primary data may have duplicate timestamps.  pandas.DataFrame.join
    on duplicate indexes can create a many-to-many join and allocate enormous
    arrays.  All context loaders in this script return one row per primary row,
    so column assignment by position is the safe and intended operation.
    """
    if ctx is None or ctx.empty or len(ctx.columns) == 0:
        return base
    if len(ctx) != len(base):
        raise RuntimeError(
            f"{label} context length mismatch: base={len(base):,}, context={len(ctx):,}. "
            "Refusing index join because duplicate timestamps can create a many-to-many explosion."
        )
    out = base.copy()
    tmp = ctx.reset_index(drop=True)
    for col in tmp.columns:
        if col in out.columns:
            # Primary OHLCV columns must win. Context columns are already prefixed
            # except range/footprint context, where names are rf_/fp_ and should
            # not collide in normal use.
            continue
        out[col] = tmp[col].to_numpy(copy=False)
    return out


def build_features(bars: pd.DataFrame, range_ctx: pd.DataFrame, footprint_ctx: pd.DataFrame, time_ctx: pd.DataFrame | None = None) -> pd.DataFrame:
    print("[3/5] Building closed-bar feature frame...", flush=True)
    # Do not sort here: context frames were aligned to the exact primary-row
    # order before this point.  Sorting a duplicate range-bar index can reorder
    # equal timestamps and break positional context alignment.
    df = bars.copy()
    if time_ctx is None:
        time_ctx = pd.DataFrame(index=df.index)
    df = _append_context_columns_by_position(df, range_ctx, "range")
    df = _append_context_columns_by_position(df, footprint_ctx, "footprint")
    df = _append_context_columns_by_position(df, time_ctx, "timeframe")
    df = df.loc[:, ~df.columns.duplicated()].copy().fillna(0.0)

    open_ = _num(df, "open")
    high = _num(df, "high")
    low = _num(df, "low")
    close = _num(df, "close")
    volume = _num(df, "volume")
    buy_notional = _num(df, "buy_notional")
    sell_notional = _num(df, "sell_notional")
    delta_notional = _num(df, "delta_notional")
    notional = (buy_notional + sell_notional).replace(0, np.nan)
    if notional.isna().all():
        notional = (close * volume).replace(0, np.nan)
    span = (high - low).replace(0, np.nan)

    df["ret_1"] = close.pct_change().fillna(0.0)
    df["bar_range_pct"] = _safe_div(high - low, close, 0.0).fillna(0.0)
    df["body_pct"] = _safe_div((close - open_).abs(), span, 0.0).fillna(0.0)
    df["close_pos"] = _safe_div(close - low, span, 0.5).clip(0.0, 1.0).fillna(0.5)
    df["upper_wick_pct"] = _safe_div(high - np.maximum(open_, close), span, 0.0).clip(0.0, 1.0).fillna(0.0)
    df["lower_wick_pct"] = _safe_div(np.minimum(open_, close) - low, span, 0.0).clip(0.0, 1.0).fillna(0.0)

    tr = pd.concat([(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    df["atr_60"] = tr.rolling(60, min_periods=20).mean()
    df["atr_pct_60"] = _safe_div(df["atr_60"], close, 0.0).fillna(0.0)

    df["ema_20"] = close.ewm(span=20, adjust=False, min_periods=5).mean()
    df["ema_60"] = close.ewm(span=60, adjust=False, min_periods=15).mean()
    df["ema_240"] = close.ewm(span=240, adjust=False, min_periods=60).mean()
    df["ema_slope_60"] = df["ema_60"].pct_change(20).fillna(0.0)
    df["trend_up"] = (df["ema_60"] > df["ema_240"]) & (df["ema_slope_60"] > 0)
    df["trend_down"] = (df["ema_60"] < df["ema_240"]) & (df["ema_slope_60"] < 0)

    # Daily/session VWAP based only on elapsed bars in the same day.
    day = df.index.date
    px_notional = (close * volume).where(volume > 0, close)
    cum_vol = volume.groupby(day).cumsum().replace(0, np.nan)
    cum_pv = px_notional.groupby(day).cumsum()
    df["session_vwap"] = (cum_pv / cum_vol).fillna(close)
    df["vwap_dist_pct"] = _safe_div(close - df["session_vwap"], close, 0.0).fillna(0.0)

    df["delta_ratio"] = _safe_div(delta_notional, notional, 0.0).clip(-1.0, 1.0).fillna(0.0)
    if "cvd_notional" in df.columns:
        df["cvd"] = pd.to_numeric(df["cvd_notional"], errors="coerce").fillna(0.0)
    else:
        df["cvd"] = delta_notional.cumsum()
    df["cvd_slope_30"] = df["cvd"].diff(30).fillna(0.0)
    df["cvd_slope_120"] = df["cvd"].diff(120).fillna(0.0)

    # Shifted rolling reference levels and quantiles: current bar never sets its own threshold.
    for w in [60, 120, 240, 480]:
        df[f"prior_high_{w}"] = high.shift(1).rolling(w, min_periods=max(20, w // 4)).max()
        df[f"prior_low_{w}"] = low.shift(1).rolling(w, min_periods=max(20, w // 4)).min()
    df["vol_q75_past"] = volume.shift(1).rolling(240, min_periods=60).quantile(0.75)
    df["vol_q90_past"] = volume.shift(1).rolling(240, min_periods=60).quantile(0.90)
    df["range_q30_past"] = df["bar_range_pct"].shift(1).rolling(240, min_periods=60).quantile(0.30)
    df["range_q70_past"] = df["bar_range_pct"].shift(1).rolling(240, min_periods=60).quantile(0.70)
    df["range_q90_past"] = df["bar_range_pct"].shift(1).rolling(240, min_periods=60).quantile(0.90)
    df["vwap_dev_q90_past"] = df["vwap_dist_pct"].abs().shift(1).rolling(480, min_periods=120).quantile(0.90)
    df["ret_abs_q70_past"] = df["ret_1"].abs().shift(1).rolling(240, min_periods=60).quantile(0.70)
    df["ret_abs_q90_past"] = df["ret_1"].abs().shift(1).rolling(480, min_periods=120).quantile(0.90)

    df["vol_regime_high"] = df["bar_range_pct"] > df["range_q70_past"]
    df["vol_regime_low"] = df["bar_range_pct"] < df["range_q30_past"]
    df["range_regime"] = (~df["trend_up"]) & (~df["trend_down"]) & (df["bar_range_pct"] < df["range_q70_past"])

    # Opening range: for bars after first 60 minutes, first-hour high/low is already known.
    # Use an explicit session Series instead of a bare numpy date array; this is
    # more robust with duplicate/naive indexes and mirrors the existing research
    # style of keeping all session features aligned to df.index.
    session_key = pd.Series(pd.DatetimeIndex(df.index).normalize(), index=df.index)
    session_min = df.groupby(session_key, sort=False).cumcount().astype("int32")
    df["session_minute"] = session_min
    first_hour = session_min < 60
    first_hour_high = high.where(first_hour)
    first_hour_low = low.where(first_hour)
    df["opening_range_high"] = first_hour_high.groupby(session_key, sort=False).transform("max")
    df["opening_range_low"] = first_hour_low.groupby(session_key, sort=False).transform("min")
    df["after_opening_range"] = session_min >= 60
    # Causal running intraday references for baseline signal diagnostics.
    df["prior_20_high"] = high.shift(1).rolling(20, min_periods=5).max()
    df["prior_20_low"] = low.shift(1).rolling(20, min_periods=5).min()

    # Optional range/footprint context defaults.
    for col in ["rf_bar_count", "rf_direction_sum", "rf_imbalance", "rf_close_pos_mean", "fp_absorption_hint"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["rf_speed_q75_past"] = df["rf_bar_count"].shift(1).rolling(480, min_periods=120).quantile(0.75)

    # Native primary range-bar features. These are non-zero only when primary_frame=range
    # or when a compatible range-like primary frame is supplied. They stay causal.
    if "direction" not in df.columns:
        df["direction"] = 0.0
    df["direction"] = pd.to_numeric(df["direction"], errors="coerce").fillna(0.0)
    df["range_dir_sum_3"] = df["direction"].rolling(3, min_periods=1).sum()
    df["range_dir_sum_5"] = df["direction"].rolling(5, min_periods=1).sum()
    if "duration_seconds" in df.columns:
        df["duration_seconds"] = pd.to_numeric(df["duration_seconds"], errors="coerce").fillna(0.0)
    else:
        df["duration_seconds"] = 0.0
    df["duration_q30_past"] = df["duration_seconds"].replace(0, np.nan).shift(1).rolling(480, min_periods=120).quantile(0.30)
    df["range_primary_fast"] = (df["duration_seconds"] > 0) & (df["duration_seconds"] <= df["duration_q30_past"])

    print(f"      feature rows={len(df):,}", flush=True)
    return df


# =============================================================================
# Signal generation
# =============================================================================



def _series_signal_from_masks(index: pd.Index, long_mask: np.ndarray, short_mask: np.ndarray, side_mode: str = "both") -> pd.Series:
    """Build a {-1,0,1} signal Series from positional numpy masks.

    This avoids a class of pandas boolean-index alignment surprises on large
    DatetimeIndex frames. It is still fully causal: masks are computed from
    closed-bar features, and entries happen next bar in the backtester.
    """
    out = np.zeros(len(index), dtype=np.int8)
    lm = np.asarray(long_mask, dtype=bool)
    sm = np.asarray(short_mask, dtype=bool)
    if side_mode in {"both", "long_only"}:
        out[lm] = 1
    if side_mode in {"both", "short_only"}:
        out[sm] = -1
    return pd.Series(out, index=index, dtype="int8")


def _np_num(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    return _num(df, col, default).to_numpy(dtype="float64", copy=False)


def _shift_np(a: np.ndarray, n: int = 1, fill: float = np.nan) -> np.ndarray:
    out = np.empty_like(a, dtype="float64")
    if n <= 0:
        return a.astype("float64", copy=False)
    out[:n] = fill
    out[n:] = a[:-n]
    return out

def signal_for_entry(df: pd.DataFrame, entry_model: str, spec: StrategySpec) -> pd.Series:
    frame = getattr(spec, "signal_frame", "primary") or "primary"
    def ccol(name: str) -> str:
        return _series_name(frame, name)

    close = _num(df, ccol("close"))
    high = _num(df, ccol("high"))
    low = _num(df, ccol("low"))
    open_ = _num(df, ccol("open"))
    vol = _num(df, ccol("volume"))
    delta = _num(df, ccol("delta_ratio"))
    close_pos = _num(df, ccol("close_pos"), 0.5)
    lower_wick = _num(df, ccol("lower_wick_pct"))
    upper_wick = _num(df, ccol("upper_wick_pct"))
    ret = _num(df, ccol("ret_1"))
    out = pd.Series(0, index=df.index, dtype="int8")

    prior_high = _num(df, ccol(f"prior_high_{spec.swing_window}"), np.nan)
    prior_low = _num(df, ccol(f"prior_low_{spec.swing_window}"), np.nan)
    high_vol = vol > _num(df, ccol("vol_q75_past"), np.inf)
    extreme_vol = vol > _num(df, ccol("vol_q90_past"), np.inf)

    if entry_model == "ema20_cross":
        # Baseline/sanity signal implemented positionally with numpy arrays.
        # If this is zero on multi-year ETH 1m data, the problem is data/window
        # or feature construction, not market opportunity.
        c = _np_num(df, ccol("close"))
        ema20 = _np_num(df, ccol("ema_20"))
        long_m = (c > ema20) & (_shift_np(c) <= _shift_np(ema20)) & np.isfinite(ema20)
        short_m = (c < ema20) & (_shift_np(c) >= _shift_np(ema20)) & np.isfinite(ema20)
        return _series_signal_from_masks(df.index, long_m, short_m, spec.side_mode)
    elif entry_model == "vwap_cross":
        c = _np_num(df, ccol("close"))
        vwap = _np_num(df, ccol("session_vwap"))
        long_m = (c > vwap) & (_shift_np(c) <= _shift_np(vwap)) & np.isfinite(vwap) & (vwap > 0)
        short_m = (c < vwap) & (_shift_np(c) >= _shift_np(vwap)) & np.isfinite(vwap) & (vwap > 0)
        return _series_signal_from_masks(df.index, long_m, short_m, spec.side_mode)
    elif entry_model == "price_momentum_sanity":
        # Causal baseline for engine validation and broad candidate search:
        # current closed-bar return exceeds its shifted rolling median absolute
        # return, then enter next open. This should fire on any non-flat market.
        r = _np_num(df, ccol("ret_1"))
        th = np.maximum(_np_num(df, ccol("ret_abs_q70_past"), 0.0), 0.00025)
        c = _np_num(df, ccol("close"))
        ema20 = _np_num(df, ccol("ema_20"))
        long_m = (r > th) & (c >= ema20)
        short_m = (r < -th) & (c <= ema20)
        return _series_signal_from_masks(df.index, long_m, short_m, spec.side_mode)
    elif entry_model == "prior_20_breakout_sanity":
        c = _np_num(df, ccol("close"))
        ph = _np_num(df, ccol("prior_20_high"), np.nan)
        pl = _np_num(df, ccol("prior_20_low"), np.nan)
        long_m = (c > ph) & (_shift_np(c) <= _shift_np(ph)) & np.isfinite(ph)
        short_m = (c < pl) & (_shift_np(c) >= _shift_np(pl)) & np.isfinite(pl)
        return _series_signal_from_masks(df.index, long_m, short_m, spec.side_mode)
    elif entry_model == "ret_impulse_continuation":
        # Past-only threshold with conservative floor; no orderbook required.
        rth = np.maximum(_num(df, ccol("ret_abs_q90_past"), 0.0), 0.0008)
        long_m = ret.gt(rth) & close.ge(_num(df, ccol("ema_20")))
        short_m = ret.lt(-rth) & close.le(_num(df, ccol("ema_20")))
    elif entry_model == "or_breakout_close":
        orh = _num(df, ccol("opening_range_high"), np.nan)
        orl = _num(df, ccol("opening_range_low"), np.nan)
        active = _bool(df, ccol("after_opening_range")) & _num(df, ccol("session_minute")).le(720)
        # Pure close-cross opening range breakout.  Delta/volume are intentionally
        # not required here, so this reveals whether opening-range features work.
        long_m = active & close.gt(orh) & close.shift(1).le(orh)
        short_m = active & close.lt(orl) & close.shift(1).ge(orl)
    elif entry_model == "sweep_reclaim":
        # Balanced version: the old V3.2 required both large wick and high volume,
        # which can become zero-signal on some cached trade-bar datasets.  This
        # keeps the causal sweep/reclaim definition but lets either volume OR
        # wick confirm the sweep.
        pressure_ok = high_vol | _num(df, ccol("bar_range_pct")).gt(_num(df, ccol("range_q70_past"), np.inf))
        long_m = low.lt(prior_low) & close.gt(prior_low) & close_pos.ge(0.52) & (lower_wick.ge(0.12) | pressure_ok)
        short_m = high.gt(prior_high) & close.lt(prior_high) & close_pos.le(0.48) & (upper_wick.ge(0.12) | pressure_ok)
    elif entry_model == "failed_breakout_reclaim":
        long_m = low.lt(prior_low) & close.gt(open_) & close.gt(prior_low) & close_pos.ge(0.50) & delta.gt(-0.45)
        short_m = high.gt(prior_high) & close.lt(open_) & close.lt(prior_high) & close_pos.le(0.50) & delta.lt(0.45)
    elif entry_model == "trend_pullback_reclaim":
        ema20 = _num(df, ccol("ema_20"))
        ema60 = _num(df, ccol("ema_60"))
        vwap = _num(df, ccol("session_vwap"))
        up_ctx = close.gt(ema60) | _bool(df, ccol("trend_up"))
        dn_ctx = close.lt(ema60) | _bool(df, ccol("trend_down"))
        long_m = up_ctx & low.le(np.minimum(ema20, vwap)) & close.gt(ema20) & delta.gt(-0.35)
        short_m = dn_ctx & high.ge(np.maximum(ema20, vwap)) & close.lt(ema20) & delta.lt(0.35)
    elif entry_model == "vwap_deviation_reversion":
        dev = _num(df, ccol("vwap_dist_pct"))
        # Use a floor so very quiet periods do not need the rolling q90 to be huge.
        th = np.maximum(_num(df, ccol("vwap_dev_q90_past"), 0.0), 0.0012)
        long_m = dev.lt(-th) & close_pos.ge(0.55) & delta.gt(-0.60) & (~_bool(df, ccol("trend_down")))
        short_m = dev.gt(th) & close_pos.le(0.45) & delta.lt(0.60) & (~_bool(df, ccol("trend_up")))
    elif entry_model == "cvd_divergence_reversal":
        cvd = _num(df, ccol("cvd"))
        cvd_low = cvd.shift(1).rolling(240, min_periods=80).min()
        cvd_high = cvd.shift(1).rolling(240, min_periods=80).max()
        price_swept_low = low.lt(prior_low) | low.lt(low.shift(1).rolling(60, min_periods=20).min())
        price_swept_high = high.gt(prior_high) | high.gt(high.shift(1).rolling(60, min_periods=20).max())
        long_m = price_swept_low & cvd.gt(cvd_low) & close_pos.ge(0.52) & delta.gt(-0.70)
        short_m = price_swept_high & cvd.lt(cvd_high) & close_pos.le(0.48) & delta.lt(0.70)
    elif entry_model == "liquidation_panic_reversal":
        rth = np.maximum(_num(df, ccol("ret_abs_q90_past"), 0.0), 0.0015)
        panic = extreme_vol | _num(df, ccol("bar_range_pct")).gt(_num(df, ccol("range_q90_past"), np.inf))
        long_m = ret.lt(-rth) & panic & close_pos.ge(0.50) & lower_wick.ge(0.10)
        short_m = ret.gt(rth) & panic & close_pos.le(0.50) & upper_wick.ge(0.10)
    elif entry_model == "compression_breakout_retest":
        compressed = _num(df, ccol("bar_range_pct")).shift(1).rolling(30, min_periods=10).mean() < _num(df, ccol("range_q30_past"), np.inf)
        long_m = compressed & close.gt(prior_high) & close.shift(1).le(prior_high) & delta.gt(-0.20)
        short_m = compressed & close.lt(prior_low) & close.shift(1).ge(prior_low) & delta.lt(0.20)
    elif entry_model == "microtrend_continuation":
        ema20 = _num(df, ccol("ema_20"))
        ema60 = _num(df, ccol("ema_60"))
        hh = close.gt(close.shift(1)) & close.shift(1).gt(close.shift(2))
        ll = close.lt(close.shift(1)) & close.shift(1).lt(close.shift(2))
        recent_low = low.shift(1).rolling(8, min_periods=3).min()
        recent_high = high.shift(1).rolling(8, min_periods=3).max()
        long_m = close.gt(ema20) & ema20.ge(ema60) & hh & low.ge(recent_low) & delta.gt(-0.35)
        short_m = close.lt(ema20) & ema20.le(ema60) & ll & high.le(recent_high) & delta.lt(0.35)
    elif entry_model == "opening_range_fakeout":
        orh = _num(df, ccol("opening_range_high"), np.nan)
        orl = _num(df, ccol("opening_range_low"), np.nan)
        active = _bool(df, ccol("after_opening_range")) & _num(df, ccol("session_minute")).le(480)
        long_m = active & low.lt(orl) & close.gt(orl) & close_pos.ge(0.50)
        short_m = active & high.gt(orh) & close.lt(orh) & close_pos.le(0.50)
    elif entry_model == "opening_range_breakout":
        orh = _num(df, ccol("opening_range_high"), np.nan)
        orl = _num(df, ccol("opening_range_low"), np.nan)
        active = _bool(df, ccol("after_opening_range")) & _num(df, ccol("session_minute")).le(720)
        # Cross event, not repeated every bar above/below the OR level.  Delta is
        # only a soft confirmation to avoid zero-signal failures on caches where
        # maker/taker side is incomplete.
        long_m = active & close.gt(orh) & close.shift(1).le(orh) & delta.gt(-0.25)
        short_m = active & close.lt(orl) & close.shift(1).ge(orl) & delta.lt(0.25)
    elif entry_model == "range_bar_momentum_proxy":
        rf_count = _num(df, "rf_bar_count")
        rf_fast = rf_count.gt(_num(df, "rf_speed_q75_past", np.inf))
        rf_imb = _num(df, "rf_imbalance")
        # Disabled by default when range context is not loaded; becomes active
        # only with --include-range-context.
        long_m = rf_fast & rf_imb.gt(0.015) & (~_bool(df, ccol("trend_down")))
        short_m = rf_fast & rf_imb.lt(-0.015) & (~_bool(df, ccol("trend_up")))
    elif entry_model == "range_momentum_burst":
        # Native range-bar burst: requires primary range columns. On time bars this
        # naturally stays near zero unless direction/duration exist.
        direction = _num(df, "direction")
        d3 = _num(df, "range_dir_sum_3")
        fast = _bool(df, "range_primary_fast") | _num(df, "bar_range_pct").gt(_num(df, "range_q70_past", np.inf))
        long_m = d3.ge(2) & fast & close.gt(prior_high) & delta.gt(-0.20)
        short_m = d3.le(-2) & fast & close.lt(prior_low) & delta.lt(0.20)
    elif entry_model == "range_sweep_reclaim":
        direction = _num(df, "direction")
        long_m = low.lt(prior_low) & close.gt(prior_low) & close_pos.ge(0.52) & direction.ge(0)
        short_m = high.gt(prior_high) & close.lt(prior_high) & close_pos.le(0.48) & direction.le(0)
    elif entry_model == "range_pullback_reclaim":
        ema20 = _num(df, ccol("ema_20"))
        ema60 = _num(df, ccol("ema_60"))
        direction = _num(df, "direction")
        long_m = close.gt(ema60) & low.le(ema20) & close.gt(ema20) & direction.gt(0)
        short_m = close.lt(ema60) & high.ge(ema20) & close.lt(ema20) & direction.lt(0)
    elif entry_model == "range_speed_reversal":
        fast = _bool(df, "range_primary_fast") | _num(df, "bar_range_pct").gt(_num(df, "range_q90_past", np.inf))
        direction = _num(df, "direction")
        long_m = fast & direction.lt(0) & close_pos.ge(0.60) & lower_wick.ge(0.10)
        short_m = fast & direction.gt(0) & close_pos.le(0.40) & upper_wick.ge(0.10)
    elif entry_model == "lead_trend_pullback_flywheel":
        # Expansion layer: absorb V3.7 finding that tf5m trend_pullback + high_vol had base edge,
        # but make it slightly more selective and less raw-cross dependent.
        ema20 = _num(df, ccol("ema_20"))
        ema60 = _num(df, ccol("ema_60"))
        vwap = _num(df, ccol("session_vwap"))
        rth = np.maximum(_num(df, ccol("ret_abs_q70_past"), 0.0), 0.00035)
        up_ctx = close.gt(ema60) & ema20.ge(ema60)
        dn_ctx = close.lt(ema60) & ema20.le(ema60)
        long_m = up_ctx & low.le(np.minimum(ema20, vwap)) & close.gt(ema20) & ret.gt(-rth) & delta.gt(-0.35)
        short_m = dn_ctx & high.ge(np.maximum(ema20, vwap)) & close.lt(ema20) & ret.lt(rth) & delta.lt(0.35)
    elif entry_model == "lead_failed_breakout_vwap":
        # Comfort layer: false breakout / sweep back into value.  V3.7 saw a low-vol failed-breakout
        # seed; this version explicitly uses VWAP as the mean-reversion target context.
        vwap = _num(df, ccol("session_vwap"))
        value_ok_long = close.lt(vwap) | close.shift(1).lt(vwap)
        value_ok_short = close.gt(vwap) | close.shift(1).gt(vwap)
        long_m = low.lt(prior_low) & close.gt(prior_low) & close.gt(open_) & close_pos.ge(0.52) & value_ok_long & delta.gt(-0.55)
        short_m = high.gt(prior_high) & close.lt(prior_high) & close.lt(open_) & close_pos.le(0.48) & value_ok_short & delta.lt(0.55)
    elif entry_model == "lead_sweep_absorption_proxy":
        # Liquidity reclaim with a footprint/CVD proxy.  Real footprint can be added as context; this
        # version only requires that price reclaims despite one-sided pressure not extending.
        delta_improve = delta.gt(delta.shift(1))
        delta_weaken = delta.lt(delta.shift(1))
        long_m = low.lt(prior_low) & close.gt(prior_low) & close_pos.ge(0.55) & (lower_wick.ge(0.10) | delta_improve) & delta.gt(-0.75)
        short_m = high.gt(prior_high) & close.lt(prior_high) & close_pos.le(0.45) & (upper_wick.ge(0.10) | delta_weaken) & delta.lt(0.75)
    elif entry_model == "lead_range_burst_pullback":
        # Range momentum pullback: native range if available; otherwise a time-bar proxy.
        direction = _num(df, "direction", 0.0)
        d3 = _num(df, "range_dir_sum_3", 0.0)
        has_range = direction.abs().rolling(20, min_periods=1).sum().gt(0)
        ema20 = _num(df, ccol("ema_20"))
        ema60 = _num(df, ccol("ema_60"))
        time_proxy_long = close.gt(ema60) & low.le(ema20) & close.gt(high.shift(1))
        time_proxy_short = close.lt(ema60) & high.ge(ema20) & close.lt(low.shift(1))
        range_long = has_range & d3.ge(2) & low.le(ema20) & close.gt(high.shift(1)) & delta.gt(-0.35)
        range_short = has_range & d3.le(-2) & high.ge(ema20) & close.lt(low.shift(1)) & delta.lt(0.35)
        long_m = range_long | ((~has_range) & time_proxy_long & delta.gt(-0.35))
        short_m = range_short | ((~has_range) & time_proxy_short & delta.lt(0.35))
    elif entry_model == "lead_vwap_rotation_transition":
        # Comfort-to-expansion hybrid: first mean reversion toward VWAP, but keeps candidates
        # that can transition to trend when reclaim is strong.
        dev = _num(df, ccol("vwap_dist_pct"))
        th = np.maximum(_num(df, ccol("vwap_dev_q80_past"), 0.0), 0.0010)
        ema20 = _num(df, ccol("ema_20"))
        long_m = dev.lt(-th) & close_pos.ge(0.55) & close.gt(open_) & close.le(ema20 * 1.003) & delta.gt(-0.60)
        short_m = dev.gt(th) & close_pos.le(0.45) & close.lt(open_) & close.ge(ema20 * 0.997) & delta.lt(0.60)
    elif entry_model == "lead_opening_fakeout_comfort":
        # Session product layer: daily structure, clear stop, usually easy to explain to followers.
        orh = _num(df, ccol("opening_range_high"), np.nan)
        orl = _num(df, ccol("opening_range_low"), np.nan)
        active = _bool(df, ccol("after_opening_range")) & _num(df, ccol("session_minute")).between(30, 480)
        long_m = active & low.lt(orl) & close.gt(orl) & close_pos.ge(0.55) & delta.gt(-0.50)
        short_m = active & high.gt(orh) & close.lt(orh) & close_pos.le(0.45) & delta.lt(0.50)
    else:
        raise ValueError(f"Unknown entry_model: {entry_model}")

    if spec.side_mode in {"both", "long_only"}:
        out.loc[long_m.fillna(False)] = 1
    if spec.side_mode in {"both", "short_only"}:
        out.loc[short_m.fillna(False)] = -1
    return out


def apply_regime(df: pd.DataFrame, raw_signal: pd.Series, regime: str, signal_frame: str = "primary") -> pd.Series:
    sig = raw_signal.copy().astype("int8")
    if regime == "any":
        return sig
    def ccol(name: str) -> str:
        return _series_name(signal_frame, name)
    if regime == "trend_aligned":
        ok_long = _bool(df, ccol("trend_up"))
        ok_short = _bool(df, ccol("trend_down"))
    elif regime == "range_only":
        ok_long = ok_short = _bool(df, ccol("range_regime"))
    elif regime == "high_vol":
        ok_long = ok_short = _num(df, ccol("bar_range_pct")) > _num(df, ccol("range_q70_past"), np.inf)
    elif regime == "low_vol":
        ok_long = ok_short = _bool(df, "vol_regime_low") if signal_frame == "primary" else (_num(df, ccol("bar_range_pct")) < _num(df, ccol("range_q30_past"), np.inf))
    elif regime == "asia_session":
        hour = pd.Series(df.index.hour, index=df.index)
        m = hour.between(0, 8)
        ok_long = ok_short = m
    elif regime == "eu_us_session":
        hour = pd.Series(df.index.hour, index=df.index)
        m = hour.between(12, 23)
        ok_long = ok_short = m
    else:
        raise ValueError(f"Unknown regime: {regime}")
    sig.loc[(sig == 1) & (~ok_long)] = 0
    sig.loc[(sig == -1) & (~ok_short)] = 0
    return sig.astype("int8")


# =============================================================================
# Strategy factory
# =============================================================================


def generate_specs(mode: str, max_specs: int | None = None, signal_frames: list[str] | None = None, include_sanity_in_core: bool = False) -> list[StrategySpec]:
    """Generate system-based copy-trading candidates, not random signal soup.

    The earlier intraday factory was entry_model-first.  This research is system-first:
    each candidate has a market story, a comfort/expansion layer and a flywheel position
    structure.  This makes the output easier to interpret and avoids mixing prop-style
    high-return candidates with copy-trading comfort candidates.
    """
    available_frames = signal_frames or ["primary"]

    systems: list[dict[str, Any]] = []
    if mode == "smoke":
        systems.extend([
            dict(system="sanity_engine_check", layer="sanity", entries=["ema20_cross", "vwap_cross"], frames=["primary"], regimes=["any"], structures=["fixed_fast"], swings=[60]),
            dict(system="lead_trend_pullback_expansion", layer="expansion", entries=["lead_trend_pullback_flywheel"], frames=["tf5m", "primary"], regimes=["high_vol", "eu_us_session"], structures=["flywheel_runner", "expansion_pyramid"], swings=[120, 240], confirmation="trend_vwap_delta"),
            dict(system="lead_failed_breakout_comfort", layer="comfort", entries=["lead_failed_breakout_vwap"], frames=["tf5m", "primary"], regimes=["low_vol", "range_only"], structures=["flywheel_dopamine", "comfort_failfast"], swings=[120, 240], confirmation="vwap_reclaim"),
            dict(system="lead_opening_fakeout_comfort", layer="comfort", entries=["lead_opening_fakeout_comfort"], frames=["primary", "tf5m"], regimes=["eu_us_session"], structures=["flywheel_dopamine"], swings=[60, 120], confirmation="session_or"),
        ])
    else:
        systems.extend([
            # Absorbs the strongest V3.7 finding, but evaluates it as a lead-trading flywheel.
            dict(system="lead_trend_pullback_expansion", layer="expansion", entries=["lead_trend_pullback_flywheel", "trend_pullback_reclaim"], frames=["tf5m", "tf15m", "primary"], regimes=["high_vol", "trend_aligned", "eu_us_session"], structures=["flywheel_runner", "expansion_pyramid", "partial_runner"], swings=[120, 240], confirmation="trend_vwap_delta"),
            # Absorbs the V3.7 failed_breakout low-vol seed and expands it systematically.
            dict(system="lead_failed_breakout_comfort", layer="comfort", entries=["lead_failed_breakout_vwap", "failed_breakout_reclaim"], frames=["tf5m", "primary"], regimes=["low_vol", "range_only", "eu_us_session"], structures=["flywheel_dopamine", "flywheel_probe", "comfort_failfast"], swings=[120, 240, 480], confirmation="vwap_reclaim"),
            dict(system="lead_liquidity_reclaim", layer="comfort", entries=["lead_sweep_absorption_proxy", "sweep_reclaim", "cvd_divergence_reversal"], frames=["primary", "tf5m"], regimes=["range_only", "low_vol", "eu_us_session"], structures=["flywheel_dopamine", "flywheel_probe", "partial_runner"], swings=[120, 240, 480], confirmation="absorption_proxy"),
            dict(system="lead_range_momentum_pullback", layer="expansion", entries=["lead_range_burst_pullback", "range_momentum_burst", "range_pullback_reclaim"], frames=["primary", "tf5m"], regimes=["high_vol", "trend_aligned", "eu_us_session"], structures=["flywheel_runner", "expansion_pyramid", "breakout_add_runner"], swings=[60, 120, 240], confirmation="range_or_proxy"),
            dict(system="lead_vwap_rotation_to_trend", layer="hybrid", entries=["lead_vwap_rotation_transition", "vwap_deviation_reversion"], frames=["primary", "tf5m"], regimes=["range_only", "low_vol", "eu_us_session"], structures=["flywheel_dopamine", "flywheel_probe", "partial_runner"], swings=[120, 240], confirmation="vwap_value"),
            dict(system="lead_opening_fakeout_comfort", layer="comfort", entries=["lead_opening_fakeout_comfort", "opening_range_fakeout"], frames=["primary", "tf5m"], regimes=["eu_us_session", "any"], structures=["flywheel_dopamine", "comfort_failfast", "flywheel_probe"], swings=[60, 120, 240], confirmation="session_or"),
        ])
        if include_sanity_in_core:
            systems.insert(0, dict(system="sanity_engine_check", layer="sanity", entries=SANITY_ENTRY_MODELS, frames=["primary"], regimes=["any"], structures=["fixed_fast"], swings=[60], confirmation="engine_only"))
        if mode == "wide":
            # Add broader baseline variants for context; still system-tagged.
            systems.append(dict(system="baseline_comparison", layer="baseline", entries=EDGE_ENTRY_MODELS, frames=available_frames, regimes=REGIMES, structures=["fixed_balanced", "partial_runner", "anti_martingale_1r"], swings=[60, 120, 240, 480], confirmation="baseline"))

    specs: list[StrategySpec] = []
    sid = 0
    available = set(available_frames)
    for sysdef in systems:
        frames = [f for f in sysdef["frames"] if f in available]
        if not frames:
            # If a requested auxiliary timeframe was not loaded, keep primary candidates rather than silently producing nothing.
            frames = ["primary"] if "primary" in available else list(available)
        for entry in sysdef["entries"]:
            for frame in frames:
                if frame != "primary" and _entry_uses_primary_only(entry):
                    continue
                for regime in sysdef["regimes"]:
                    for structure in sysdef["structures"]:
                        for swing in sysdef["swings"]:
                            base = StrategySpec(
                                spec_id="",
                                system=sysdef["system"],
                                layer=sysdef["layer"],
                                confirmation=sysdef.get("confirmation", "none"),
                                entry_model=entry,
                                regime=regime,
                                structure=structure,
                                signal_frame=frame,
                                swing_window=swing,
                            )
                            spec = replace(base, **STRUCTURE_PRESETS.get(structure, {}))
                            sid += 1
                            specs.append(replace(spec, spec_id=f"L{sid:05d}_{sysdef['system']}_{frame}_{entry}_{regime}_{structure}_sw{swing}"))
                            if max_specs is not None and len(specs) >= int(max_specs):
                                return specs
    return specs


# =============================================================================
# Backtester# =============================================================================
# Backtester
# =============================================================================


@dataclass
class TradeResult:
    spec_id: str
    system: str
    layer: str
    entry_model: str
    regime: str
    structure: str
    side: str
    signal_time: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    gross_pnl_frac: float
    cost_frac: float
    net_pnl_frac: float
    return_pct: float
    exit_reason: str
    hold_bars: int
    mfe_r: float
    mae_r: float
    adds: int
    partial_taken: bool


@dataclass(frozen=True)
class BacktestArrays:
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr: np.ndarray
    index: pd.DatetimeIndex
    index_ns: np.ndarray
    start_i: int
    end_i: int


def build_backtest_arrays(df: pd.DataFrame, cfg: RunConfig) -> BacktestArrays:
    """Build immutable numpy arrays and the inclusive backtest position window.

    Important:
    Earlier fast versions used np.searchsorted over idx_ns. That is only safe
    when the timestamp array is strictly monotonic. A non-monotonic or chunk-
    concatenated cache can make searchsorted return len(df), which silently
    produces an empty backtest window even though timestamp-mask diagnostics
    show millions of rows. Use a boolean time mask + flatnonzero instead; it is
    still O(n) once per run and does not affect per-spec speed.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(df.index).tz_localize(None))
    idx_ns = idx.view("int64")
    start_ts = _window_start_ts(cfg.start_date)
    end_ts = _window_end_ts(cfg.end_date)
    in_window = (idx >= start_ts) & (idx <= end_ts)
    positions = np.flatnonzero(np.asarray(in_window, dtype=bool))
    if positions.size:
        start_i = int(positions[0])
        end_i = int(positions[-1])
    else:
        start_i = 0
        end_i = -1

    return BacktestArrays(
        open=_num(df, "open").to_numpy(dtype="float64", copy=False),
        high=_num(df, "high").to_numpy(dtype="float64", copy=False),
        low=_num(df, "low").to_numpy(dtype="float64", copy=False),
        close=_num(df, "close").to_numpy(dtype="float64", copy=False),
        atr=_num(df, "atr_60").to_numpy(dtype="float64", copy=False),
        index=idx,
        index_ns=idx_ns,
        start_i=start_i,
        end_i=end_i,
    )


class SingleSpecBacktester:
    def __init__(self, df: pd.DataFrame, cfg: RunConfig, spec: StrategySpec, arrays: BacktestArrays | None = None):
        self.df = df
        self.cfg = cfg
        self.spec = spec
        self.arrays = arrays or build_backtest_arrays(df, cfg)
        self.open = self.arrays.open
        self.high = self.arrays.high
        self.low = self.arrays.low
        self.close = self.arrays.close
        self.atr = self.arrays.atr
        self.index = self.arrays.index
        self.cost_per_turnover = float(cfg.fee_rate_per_side) + float(cfg.slippage_pct)

    def run(self, signal: pd.Series | np.ndarray, signal_indices: np.ndarray | None = None) -> list[TradeResult]:
        """Run an event-driven backtest for one spec.

        V2 scanned every 1m bar for every spec. That is acceptable on V9/V10 4H
        frames, but too slow for multi-year 1m research. This runner only visits
        bars where the precomputed signal is non-zero, then simulates the forward
        trade path. It preserves the same no-lookahead timing: signal on closed
        bar i, entry at i+1+delay.
        """
        if isinstance(signal, pd.Series):
            sig = signal.reindex(self.df.index).fillna(0).astype("int8").to_numpy(copy=False)
        else:
            sig = np.asarray(signal, dtype=np.int8)
        n = len(self.df)
        if signal_indices is None:
            idxs = np.flatnonzero(sig)
        else:
            idxs = np.asarray(signal_indices, dtype=np.int64)
        if idxs.size == 0:
            return []

        delay = int(self.cfg.entry_delay_bars)
        lo = int(self.arrays.start_i)
        hi = int(min(self.arrays.end_i, n - 2 - delay))
        if hi < lo:
            return []
        idxs = idxs[(idxs >= lo) & (idxs <= hi)]
        if idxs.size == 0:
            return []

        trades: list[TradeResult] = []
        min_signal_i = lo
        for signal_i in idxs:
            i = int(signal_i)
            if i < min_signal_i:
                continue
            side = int(sig[i])
            if side == 0:
                continue
            entry_i = i + 1 + delay
            if entry_i >= n:
                break
            result, exit_i = self._simulate_trade(signal_i=i, entry_i=entry_i, side=side)
            if result is not None:
                trades.append(result)
                # Single active position: skip any signals that occur before the trade exits.
                min_signal_i = max(int(exit_i) + 1, i + 1)
        return trades

    def run_slow_reference(self, signal: pd.Series | np.ndarray) -> list[TradeResult]:
        """Slow bar-by-bar reference engine used only for exactness audits.

        This intentionally scans every bar from the test window, just like the
        straightforward V2-style implementation. It shares the same trade
        simulator as the fast path, so any mismatch means the event-driven
        signal-index optimization changed execution order and must not be
        trusted.
        """
        if isinstance(signal, pd.Series):
            sig = signal.reindex(self.df.index).fillna(0).astype("int8").to_numpy(copy=False)
        else:
            sig = np.asarray(signal, dtype=np.int8)
        n = len(self.df)
        delay = int(self.cfg.entry_delay_bars)
        lo = int(self.arrays.start_i)
        hi = int(min(self.arrays.end_i, n - 2 - delay))
        if hi < lo:
            return []

        trades: list[TradeResult] = []
        i = lo
        while i <= hi:
            side = int(sig[i])
            if side == 0:
                i += 1
                continue
            entry_i = i + 1 + delay
            if entry_i >= n:
                break
            result, exit_i = self._simulate_trade(signal_i=i, entry_i=entry_i, side=side)
            if result is not None:
                trades.append(result)
                i = max(int(exit_i) + 1, i + 1)
            else:
                i += 1
        return trades

    def _stop_distance(self, signal_i: int, entry_price: float) -> float:
        atr = self.atr[signal_i]
        atr = atr if math.isfinite(atr) and atr > 0 else entry_price * self.spec.min_stop_pct
        return max(float(atr) * self.spec.stop_atr_mult, entry_price * self.spec.min_stop_pct)

    def _notional_for_risk(self, stop_dist: float, entry_price: float, size_mult: float) -> float:
        stop_pct = max(stop_dist / entry_price, 1e-9)
        raw = float(self.cfg.risk_per_trade) / stop_pct * float(size_mult)
        return min(raw, float(self.cfg.max_notional_mult) * float(size_mult))

    def _lot_pnl(self, side: int, entry: float, exit_: float, notional_frac: float) -> float:
        ret = (exit_ / entry - 1.0) * side
        return ret * notional_frac

    def _simulate_trade(self, signal_i: int, entry_i: int, side: int) -> tuple[TradeResult | None, int]:
        n = len(self.df)
        spec = self.spec
        entry_price = float(self.open[entry_i])
        if not math.isfinite(entry_price) or entry_price <= 0:
            return None, entry_i
        stop_dist = self._stop_distance(signal_i, entry_price)
        initial_stop = entry_price - side * stop_dist
        stop_price = initial_stop
        tp_price = entry_price + side * stop_dist * spec.tp_r
        partial_price = entry_price + side * stop_dist * spec.partial_tp_r

        lots: list[tuple[float, float, bool]] = []  # entry_price, notional_frac, open
        initial_notional = self._notional_for_risk(stop_dist, entry_price, float(getattr(spec, "initial_size_mult", 1.0)))
        lots.append((entry_price, initial_notional, True))
        entry_cost = initial_notional * self.cost_per_turnover
        realized_gross = 0.0
        realized_cost = entry_cost
        partial_taken = False
        adds = 0
        add1_done = False
        add2_done = False
        exit_reason = "max_hold"
        exit_i = min(n - 1, entry_i + int(spec.max_hold_bars))
        exit_price = float(self.close[exit_i])
        max_fav = 0.0
        max_adv = 0.0
        pending_add_mult = 0.0
        pending_time_exit = False
        pending_fail_exit = False

        def current_open_lots() -> list[tuple[float, float, bool]]:
            return [lot for lot in lots if lot[2]]

        def close_fraction(price: float, frac: float) -> None:
            nonlocal realized_gross, realized_cost, lots
            new_lots: list[tuple[float, float, bool]] = []
            for ep, nf, is_open in lots:
                if not is_open or frac <= 0:
                    new_lots.append((ep, nf, is_open))
                    continue
                close_nf = nf * frac
                keep_nf = nf - close_nf
                realized_gross += self._lot_pnl(side, ep, price, close_nf)
                realized_cost += close_nf * self.cost_per_turnover
                if keep_nf > 1e-12:
                    new_lots.append((ep, keep_nf, True))
                else:
                    new_lots.append((ep, 0.0, False))
            lots = new_lots

        j = entry_i
        while j < n:
            if j > entry_i and pending_add_mult > 0:
                add_price = float(self.open[j])
                current_notional = sum(nf for _, nf, op in lots if op)
                capacity = max(0.0, float(self.cfg.max_notional_mult) - current_notional)
                add_nf = min(initial_notional * pending_add_mult, capacity)
                if add_nf > 1e-12:
                    lots.append((add_price, add_nf, True))
                    realized_cost += add_nf * self.cost_per_turnover
                    adds += 1
                    # After adding, move stop toward break-even. This prevents add-on risk explosion.
                    if side == 1:
                        stop_price = max(stop_price, entry_price - 0.05 * stop_dist)
                    else:
                        stop_price = min(stop_price, entry_price + 0.05 * stop_dist)
                pending_add_mult = 0.0

            if j > entry_i and (pending_time_exit or pending_fail_exit):
                exit_price = float(self.open[j])
                exit_reason = "time_bomb" if pending_time_exit else "fail_fast"
                exit_i = j
                close_fraction(exit_price, 1.0)
                break

            h = float(self.high[j])
            l = float(self.low[j])
            c = float(self.close[j])
            if not all(math.isfinite(x) for x in [h, l, c]):
                j += 1
                continue

            fav_price = h if side == 1 else l
            adv_price = l if side == 1 else h
            max_fav = max(max_fav, side * (fav_price - entry_price) / stop_dist)
            max_adv = min(max_adv, side * (adv_price - entry_price) / stop_dist)

            stop_hit = l <= stop_price if side == 1 else h >= stop_price
            full_tp_hit = h >= tp_price if side == 1 else l <= tp_price
            partial_hit = h >= partial_price if side == 1 else l <= partial_price

            # Conservative path assumption.
            if stop_hit:
                exit_price = float(stop_price)
                exit_reason = "stop"
                exit_i = j
                close_fraction(exit_price, 1.0)
                break

            if spec.structure in {"partial_runner", "slow_runner", "flywheel_dopamine", "flywheel_probe", "flywheel_runner", "comfort_failfast", "expansion_pyramid"} and (not partial_taken) and partial_hit:
                close_fraction(float(partial_price), float(spec.partial_fraction))
                partial_taken = True
                if side == 1:
                    stop_price = max(stop_price, entry_price)
                else:
                    stop_price = min(stop_price, entry_price)

            if full_tp_hit:
                exit_price = float(tp_price)
                exit_reason = "tp"
                exit_i = j
                close_fraction(exit_price, 1.0)
                break

            # Closed-bar decisions for add/exit are scheduled for next bar open.
            close_r = side * (c - entry_price) / stop_dist
            if spec.structure in {"probe_confirm_add", "anti_martingale_1r", "breakout_add_runner", "flywheel_probe", "flywheel_runner", "expansion_pyramid"}:
                if (not add1_done) and close_r >= spec.add_trigger_r1 and spec.add_size_1 > 0:
                    pending_add_mult = float(spec.add_size_1)
                    add1_done = True
                elif (not add2_done) and close_r >= spec.add_trigger_r2 and spec.add_size_2 > 0:
                    pending_add_mult = float(spec.add_size_2)
                    add2_done = True

            if spec.structure in {"partial_runner", "slow_runner", "breakout_add_runner", "flywheel_runner", "expansion_pyramid"} and max_fav >= 1.0:
                trail_dist = max(float(self.atr[j]) * spec.trail_atr_mult, stop_dist * 0.60)
                if side == 1:
                    stop_price = max(stop_price, c - trail_dist)
                else:
                    stop_price = min(stop_price, c + trail_dist)

            age = j - entry_i + 1
            if spec.structure in {"time_bomb", "flywheel_dopamine", "comfort_failfast"} and age >= spec.time_bomb_bars and max_fav < spec.time_bomb_min_mfe_r:
                pending_time_exit = True
            if spec.structure in {"fail_fast", "flywheel_dopamine", "flywheel_probe", "flywheel_runner", "comfort_failfast", "expansion_pyramid"} and age >= spec.fail_fast_bars and close_r <= -abs(spec.fail_fast_adverse_r):
                pending_fail_exit = True
            if age >= spec.max_hold_bars:
                exit_price = c
                exit_reason = "max_hold"
                exit_i = j
                close_fraction(exit_price, 1.0)
                break
            j += 1
        else:
            exit_i = n - 1
            exit_price = float(self.close[exit_i])
            exit_reason = "eof"
            close_fraction(exit_price, 1.0)

        # If something remains open due to numerical dust, close at exit price.
        if any(op and nf > 1e-12 for _, nf, op in lots):
            close_fraction(exit_price, 1.0)

        net = realized_gross - realized_cost
        return TradeResult(
            spec_id=spec.spec_id,
            system=getattr(spec, "system", "baseline"),
            layer=getattr(spec, "layer", "unknown"),
            entry_model=spec.entry_model,
            regime=spec.regime,
            structure=spec.structure,
            side="LONG" if side == 1 else "SHORT",
            signal_time=str(self.index[signal_i]),
            entry_time=str(self.index[entry_i]),
            exit_time=str(self.index[exit_i]),
            entry_price=round(entry_price, 6),
            exit_price=round(float(exit_price), 6),
            gross_pnl_frac=float(realized_gross),
            cost_frac=float(realized_cost),
            net_pnl_frac=float(net),
            return_pct=float(net * 100.0),
            exit_reason=exit_reason,
            hold_bars=int(exit_i - entry_i + 1),
            mfe_r=float(max_fav),
            mae_r=float(max_adv),
            adds=int(adds),
            partial_taken=bool(partial_taken),
        ), int(exit_i)


# =============================================================================
# Reporting
# =============================================================================


def _max_days_without_trade(entry_times: pd.Series) -> int:
    if entry_times.empty:
        return 999
    days = pd.to_datetime(entry_times).dt.normalize().drop_duplicates().sort_values()
    if len(days) <= 1:
        return 999
    gaps = days.diff().dt.days.dropna()
    return int(gaps.max()) if not gaps.empty else 0


def summarize_trades(spec: StrategySpec, trades: list[TradeResult], cfg: RunConfig) -> dict[str, Any]:
    row = asdict(spec)
    row.update({"trades": len(trades)})
    if not trades:
        row.update({
            "total_return_pct": 0.0,
            "final_capital": cfg.initial_capital,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "avg_return_pct": 0.0,
            "median_return_pct": 0.0,
            "avg_hold_bars": 0.0,
            "avg_mfe_r": 0.0,
            "avg_mae_r": 0.0,
            "green_touch_rate": 0.0,
            "partial_tp_rate": 0.0,
            "add_rate": 0.0,
            "stop_rate": 0.0,
            "tp_rate": 0.0,
            "max_days_without_trade": 999,
            "trades_per_active_day": 0.0,
            "lead_comfort_score": -999.0,
            "score": -999.0,
            "long_trades": 0,
            "short_trades": 0,
        })
        return row
    t = pd.DataFrame([asdict(x) for x in trades])
    pnl = pd.to_numeric(t["net_pnl_frac"], errors="coerce").fillna(0.0)
    equity = (1.0 + pnl).cumprod() * float(cfg.initial_capital)
    pf = _profit_factor(pnl)
    win_rate = float((pnl > 0).mean())
    dd = _max_drawdown(equity / float(cfg.initial_capital))
    total_ret = float(equity.iloc[-1] / float(cfg.initial_capital) - 1.0)
    avg_ret = float(pd.to_numeric(t["return_pct"], errors="coerce").mean())
    mfe = pd.to_numeric(t["mfe_r"], errors="coerce").fillna(0.0)
    mae = pd.to_numeric(t["mae_r"], errors="coerce").fillna(0.0)
    active_days = max(1, pd.to_datetime(t["entry_time"]).dt.normalize().nunique())
    partial_rate = float(t["partial_taken"].astype(bool).mean()) if "partial_taken" in t else 0.0
    add_rate = float((pd.to_numeric(t["adds"], errors="coerce").fillna(0) > 0).mean())
    green_touch_rate = float((mfe >= max(0.25, spec.partial_tp_r * 0.50)).mean())
    stop_rate = float((t["exit_reason"] == "stop").mean())
    tp_rate = float((t["exit_reason"] == "tp").mean())
    max_gap_days = _max_days_without_trade(t["entry_time"])
    trades_per_day = float(len(t) / active_days)
    score = score_candidate(total_ret, pf, dd, len(t), win_rate, avg_ret, partial_rate, green_touch_rate, max_gap_days, stop_rate)
    row.update({
        "total_return_pct": round(total_ret * 100.0, 4),
        "final_capital": round(float(equity.iloc[-1]), 4),
        "win_rate": round(win_rate * 100.0, 4),
        "profit_factor": round(pf, 4) if math.isfinite(pf) else "inf",
        "max_drawdown_pct": round(dd * 100.0, 4),
        "avg_return_pct": round(avg_ret, 5),
        "median_return_pct": round(float(pd.to_numeric(t["return_pct"], errors="coerce").median()), 5),
        "avg_hold_bars": round(float(pd.to_numeric(t["hold_bars"], errors="coerce").mean()), 3),
        "avg_mfe_r": round(float(mfe.mean()), 4),
        "avg_mae_r": round(float(mae.mean()), 4),
        "p95_adverse_r": round(float((-mae).quantile(0.95)), 4),
        "green_touch_rate": round(green_touch_rate * 100.0, 4),
        "partial_tp_rate": round(partial_rate * 100.0, 4),
        "add_rate": round(add_rate * 100.0, 4),
        "stop_rate": round(stop_rate * 100.0, 4),
        "tp_rate": round(tp_rate * 100.0, 4),
        "max_days_without_trade": int(max_gap_days),
        "trades_per_active_day": round(trades_per_day, 4),
        "long_trades": int((t["side"] == "LONG").sum()),
        "short_trades": int((t["side"] == "SHORT").sum()),
        "lead_comfort_score": round(score, 4),
        "score": round(score, 4),
    })
    return row


def score_candidate(total_ret: float, pf: float, dd: float, trades: int, win_rate: float, avg_ret_pct: float, partial_rate: float = 0.0, green_touch_rate: float = 0.0, max_gap_days: int = 999, stop_rate: float = 1.0) -> float:
    # Lead/copy-trading score: not just return. Penalize follower pain: long no-trade gaps,
    # high stop rate, high DD, and too many tiny noisy trades.
    if trades < 20:
        return -100.0 + trades
    pf_score = min(max((pf - 1.0) / 0.60, -1.5), 2.0) if math.isfinite(pf) else 2.0
    ret_score = min(max(total_ret / 0.80, -1.0), 2.0)
    dd_score = min(max((0.15 + dd) / 0.15, -2.0), 1.0)  # dd is negative
    win_score = min(max((win_rate - 0.48) / 0.20, -1.0), 1.2)
    dopamine_score = min(max(partial_rate / 0.55, 0.0), 1.2)
    green_score = min(max(green_touch_rate / 0.70, 0.0), 1.2)
    gap_penalty = min(max((max_gap_days - 3) / 10.0, 0.0), 2.0)
    overtrade_penalty = min(max((trades - 10000) / 10000.0, 0.0), 2.0)
    stop_penalty = min(max((stop_rate - 0.45) / 0.30, 0.0), 1.5)
    avg_score = min(max(avg_ret_pct / 0.08, -1.0), 1.0)
    return (
        22 * pf_score
        + 18 * ret_score
        + 18 * dd_score
        + 14 * win_score
        + 10 * dopamine_score
        + 8 * green_score
        + 5 * avg_score
        - 10 * gap_penalty
        - 8 * overtrade_penalty
        - 8 * stop_penalty
    )


def yearly_stats(trades_df: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame()
    t = trades_df.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"])
    t["year"] = t["entry_time"].dt.year
    rows = []
    for (spec_id, year), g in t.groupby(["spec_id", "year"]):
        pnl = pd.to_numeric(g["net_pnl_frac"], errors="coerce").fillna(0.0)
        equity = (1 + pnl).cumprod()
        rows.append({
            "spec_id": spec_id,
            "year": int(year),
            "trades": int(len(g)),
            "return_pct": round(float((equity.iloc[-1] - 1) * 100.0), 4),
            "win_rate": round(float((pnl > 0).mean() * 100.0), 4),
            "profit_factor": round(_profit_factor(pnl), 4) if math.isfinite(_profit_factor(pnl)) else "inf",
            "max_drawdown_pct": round(_max_drawdown(equity) * 100.0, 4),
        })
    return pd.DataFrame(rows)


def signal_raw_key(spec: StrategySpec) -> tuple[str, int, str, str]:
    return (str(spec.entry_model), int(spec.swing_window), str(spec.side_mode), str(getattr(spec, "signal_frame", "primary")))


def signal_full_key(spec: StrategySpec) -> tuple[str, int, str, str, str]:
    return (str(spec.entry_model), int(spec.swing_window), str(spec.side_mode), str(spec.regime), str(getattr(spec, "signal_frame", "primary")))




def _filter_signal_array(sig: np.ndarray, lo: int, hi: int, cfg: RunConfig) -> tuple[np.ndarray, np.ndarray]:
    """Apply optional signal-gap/cap filters by zeroing the signal array itself.

    Filtering the array, not only the index cache, keeps fast and slow audit
    equivalent because both engines see the exact same tradable signal series.
    """
    sig = np.asarray(sig, dtype=np.int8)
    idxs = np.flatnonzero(sig)
    if idxs.size:
        idxs = idxs[(idxs >= lo) & (idxs <= hi)]
    if idxs.size == 0:
        return sig.copy(), idxs.astype(np.int64)
    min_gap = max(0, int(getattr(cfg, "min_signal_gap_bars", 0) or 0))
    max_count = max(0, int(getattr(cfg, "max_signals_per_spec", 0) or 0))
    if min_gap <= 0 and max_count <= 0:
        return sig.copy(), idxs.astype(np.int64, copy=False)
    keep: list[int] = []
    last = -10**18
    for i in idxs:
        ii = int(i)
        if min_gap > 0 and ii - last < min_gap:
            continue
        keep.append(ii)
        last = ii
        if max_count > 0 and len(keep) >= max_count:
            break
    out = np.zeros_like(sig, dtype=np.int8)
    if keep:
        keep_arr = np.asarray(keep, dtype=np.int64)
        out[keep_arr] = sig[keep_arr]
    else:
        keep_arr = np.asarray([], dtype=np.int64)
    return out, keep_arr

def build_signal_caches(
    df: pd.DataFrame,
    specs: list[StrategySpec],
    cfg: RunConfig,
) -> tuple[
    dict[tuple[str, int, str, str, str], np.ndarray],
    dict[tuple[str, int, str, str, str], np.ndarray],
    dict[tuple[str, int, str, str, str], dict[str, int]],
    dict[tuple[str, int, str, str], dict[str, int]],
]:
    """Precompute and cache signals shared by many StrategySpec variants.

    Most factory specs differ only by position structure. Recomputing the same
    rolling-entry masks and rescanning all bars per structure was the main V2
    bottleneck. This cache computes each entry/swing signal once, each regime
    filter once, and stores non-zero signal indices for event-driven execution.
    """
    arrays = build_backtest_arrays(df, cfg)
    raw_cache: dict[tuple[str, int, str, str], pd.Series] = {}
    raw_count_cache: dict[tuple[str, int, str, str], dict[str, int]] = {}
    signal_cache: dict[tuple[str, int, str, str, str], np.ndarray] = {}
    index_cache: dict[tuple[str, int, str, str, str], np.ndarray] = {}
    count_cache: dict[tuple[str, int, str, str, str], dict[str, int]] = {}

    unique_raw: dict[tuple[str, int, str, str], StrategySpec] = {}
    unique_full: dict[tuple[str, int, str, str, str], StrategySpec] = {}
    for spec in specs:
        unique_raw.setdefault(signal_raw_key(spec), spec)
        unique_full.setdefault(signal_full_key(spec), spec)

    print(f"      precomputing raw signals: {len(unique_raw):,}; regime signals: {len(unique_full):,}", flush=True)

    lo = int(arrays.start_i)
    hi = int(min(arrays.end_i, len(df) - 1))
    has_test_window = 0 <= lo <= hi < len(df)

    for k, spec in unique_raw.items():
        raw_series = signal_for_entry(df, spec.entry_model, spec).astype("int8")
        raw_cache[k] = raw_series
        raw_arr = raw_series.to_numpy(copy=False)
        if has_test_window:
            raw_idxs = np.flatnonzero(raw_arr)
            raw_idxs = raw_idxs[(raw_idxs >= lo) & (raw_idxs <= hi)]
            raw_vals = raw_arr[raw_idxs] if raw_idxs.size else np.asarray([], dtype=np.int8)
            raw_count_cache[k] = {
                "raw_signals": int(raw_idxs.size),
                "raw_long_signals": int(np.sum(raw_vals == 1)) if raw_idxs.size else 0,
                "raw_short_signals": int(np.sum(raw_vals == -1)) if raw_idxs.size else 0,
            }
        else:
            raw_count_cache[k] = {"raw_signals": 0, "raw_long_signals": 0, "raw_short_signals": 0}

    hi = int(min(arrays.end_i, len(df) - 1))
    for k, spec in unique_full.items():
        sig_series = apply_regime(df, raw_cache[signal_raw_key(spec)], spec.regime, getattr(spec, "signal_frame", "primary"))
        sig0 = sig_series.astype("int8").to_numpy(copy=False)
        sig, idxs = _filter_signal_array(sig0, lo, hi, cfg)
        signal_cache[k] = sig
        index_cache[k] = idxs.astype(np.int64, copy=False)
        if idxs.size:
            vals = sig[idxs]
            longs = int(np.sum(vals == 1))
            shorts = int(np.sum(vals == -1))
        else:
            longs = shorts = 0
        count_cache[k] = {"signals": int(longs + shorts), "long_signals": longs, "short_signals": shorts}
    return signal_cache, index_cache, count_cache, raw_count_cache



def write_data_diagnostics(bars: pd.DataFrame, df: pd.DataFrame, cfg: RunConfig, arrays: BacktestArrays, out_dir: Path) -> pd.DataFrame:
    """Write data-window diagnostics so zero-signal runs are explainable."""
    idx = pd.DatetimeIndex(pd.to_datetime(df.index).tz_localize(None))
    start_ts = _window_start_ts(cfg.start_date)
    end_ts = _window_end_ts(cfg.end_date)
    in_window = (idx >= start_ts) & (idx <= end_ts)
    required_cols = [
        "open", "high", "low", "close", "volume", "buy_notional", "sell_notional",
        "delta_notional", "cvd_notional", "trades_count", "vwap",
    ]
    rows = []
    rows.append({"metric": "bars_rows_loaded", "value": int(len(bars))})
    rows.append({"metric": "primary_frame", "value": str(cfg.primary_frame)})
    rows.append({"metric": "timeframe", "value": str(cfg.timeframe)})
    rows.append({"metric": "context_timeframes", "value": str(cfg.context_timeframes)})
    rows.append({"metric": "feature_rows", "value": int(len(df))})
    rows.append({"metric": "data_first_ts", "value": str(idx[0]) if len(idx) else ""})
    rows.append({"metric": "data_last_ts", "value": str(idx[-1]) if len(idx) else ""})
    rows.append({"metric": "requested_start_date", "value": str(start_ts)})
    rows.append({"metric": "requested_end_date", "value": str(end_ts)})
    rows.append({"metric": "backtest_rows_in_requested_window", "value": int(in_window.sum())})
    if int(in_window.sum()) > 0:
        win_idx = idx[in_window]
        rows.append({"metric": "backtest_first_ts", "value": str(win_idx[0])})
        rows.append({"metric": "backtest_last_ts", "value": str(win_idx[-1])})
    rows.append({"metric": "opening_range_active_rows", "value": int(pd.to_numeric(df.get("after_opening_range", pd.Series(False, index=df.index)), errors="coerce").fillna(0).astype(bool).loc[in_window].sum())})
    rows.append({"metric": "ema20_cross_sanity_long", "value": int((_num(df, "close").gt(_num(df, "ema_20")) & _num(df, "close").shift(1).le(_num(df, "ema_20").shift(1))).loc[in_window].sum())})
    rows.append({"metric": "vwap_cross_sanity_long", "value": int((_num(df, "close").gt(_num(df, "session_vwap")) & _num(df, "close").shift(1).le(_num(df, "session_vwap").shift(1))).loc[in_window].sum())})
    rows.append({"metric": "or_breakout_sanity_long", "value": int((_bool(df, "after_opening_range") & _num(df, "close").gt(_num(df, "opening_range_high", np.nan)) & _num(df, "close").shift(1).le(_num(df, "opening_range_high", np.nan))).loc[in_window].sum())})
    rows.append({"metric": "arrays_start_i", "value": int(arrays.start_i)})
    rows.append({"metric": "arrays_end_i", "value": int(arrays.end_i)})
    rows.append({"metric": "arrays_window_rows", "value": int(max(0, arrays.end_i - arrays.start_i + 1))})
    rows.append({"metric": "index_is_monotonic_increasing", "value": bool(idx.is_monotonic_increasing)})
    rows.append({"metric": "index_is_unique", "value": bool(idx.is_unique)})
    if 0 <= arrays.start_i < len(idx) and 0 <= arrays.end_i < len(idx) and arrays.start_i <= arrays.end_i:
        rows.append({"metric": "arrays_window_first_ts", "value": str(idx[arrays.start_i])})
        rows.append({"metric": "arrays_window_last_ts", "value": str(idx[arrays.end_i])})
    if len(idx) and int(in_window.sum()) > 0:
        win_idx = idx[in_window]
        rows.append({"metric": "backtest_first_ts", "value": str(win_idx[0])})
        rows.append({"metric": "backtest_last_ts", "value": str(win_idx[-1])})
    for col in required_cols:
        rows.append({"metric": f"has_col_{col}", "value": bool(col in df.columns)})
        if col in df.columns:
            ser = pd.to_numeric(df[col], errors="coerce")
            rows.append({"metric": f"nan_ratio_{col}", "value": float(ser.isna().mean())})
            rows.append({"metric": f"min_{col}", "value": float(ser.min()) if ser.notna().any() else ""})
            rows.append({"metric": f"max_{col}", "value": float(ser.max()) if ser.notna().any() else ""})
            if col in {"open", "high", "low", "close", "volume", "trades_count", "buy_notional", "sell_notional", "delta_notional"}:
                rows.append({"metric": f"nonzero_ratio_{col}", "value": float((ser.fillna(0.0).abs() > 0).mean())})
    for col in ["prior_high_120", "prior_low_120", "opening_range_high", "opening_range_low", "session_vwap", "ema_20", "ema_60", "range_q70_past", "vol_q75_past", "direction", "range_dir_sum_3", "range_primary_fast", "tf5m_close", "tf15m_close", "tf5m_trend_up", "tf15m_trend_up"]:
        rows.append({"metric": f"has_feature_{col}", "value": bool(col in df.columns)})
        if col in df.columns:
            ser = pd.to_numeric(df[col], errors="coerce")
            rows.append({"metric": f"nonzero_ratio_{col}", "value": float((ser.fillna(0.0).abs() > 0).mean())})
            rows.append({"metric": f"nan_ratio_{col}", "value": float(ser.isna().mean())})
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "00_data_diagnostics.csv", index=False)
    return out



def write_signal_condition_breakdown(df: pd.DataFrame, cfg: RunConfig, out_dir: Path) -> pd.DataFrame:
    """Independent signal sanity counts, bypassing StrategySpec/regime/cache.

    If these counts are positive but raw_signal_diagnostics is zero, the bug is
    inside signal_for_entry/build_signal_caches. If these are also zero, inspect
    the underlying OHLCV feature values and requested window.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(df.index).tz_localize(None))
    in_window = (idx >= _window_start_ts(cfg.start_date)) & (idx <= _window_end_ts(cfg.end_date))
    c = _np_num(df, "close")
    h = _np_num(df, "high")
    l = _np_num(df, "low")
    ema20 = _np_num(df, "ema_20")
    vwap = _np_num(df, "session_vwap")
    ret = _np_num(df, "ret_1")
    ph20 = _np_num(df, "prior_20_high", np.nan)
    pl20 = _np_num(df, "prior_20_low", np.nan)
    r70 = np.maximum(_np_num(df, "ret_abs_q70_past", 0.0), 0.00025)
    rows = []
    def add(name: str, mask: np.ndarray) -> None:
        m = np.asarray(mask, dtype=bool) & np.asarray(in_window, dtype=bool)
        rows.append({"condition": name, "count": int(m.sum())})
    add("close_nonzero", np.isfinite(c) & (c > 0))
    add("close_changed_from_prev", np.abs(c - _shift_np(c)) > 1e-12)
    add("close_above_ema20", c > ema20)
    add("close_below_ema20", c < ema20)
    add("ema20_cross_long_np", (c > ema20) & (_shift_np(c) <= _shift_np(ema20)) & np.isfinite(ema20))
    add("ema20_cross_short_np", (c < ema20) & (_shift_np(c) >= _shift_np(ema20)) & np.isfinite(ema20))
    add("vwap_cross_long_np", (c > vwap) & (_shift_np(c) <= _shift_np(vwap)) & np.isfinite(vwap) & (vwap > 0))
    add("vwap_cross_short_np", (c < vwap) & (_shift_np(c) >= _shift_np(vwap)) & np.isfinite(vwap) & (vwap > 0))
    add("ret_positive", ret > 0)
    add("ret_negative", ret < 0)
    add("price_momentum_long_np", (ret > r70) & (c >= ema20))
    add("price_momentum_short_np", (ret < -r70) & (c <= ema20))
    add("prior20_breakout_long_np", (c > ph20) & (_shift_np(c) <= _shift_np(ph20)) & np.isfinite(ph20))
    add("prior20_breakout_short_np", (c < pl20) & (_shift_np(c) >= _shift_np(pl20)) & np.isfinite(pl20))
    add("high_gt_prior20_high", h > ph20)
    add("low_lt_prior20_low", l < pl20)
    # Context and range sanity counts. Missing columns simply produce zero.
    for prefix in ["tf5m", "tf15m", "tf30m", "tf1H", "tf4H"]:
        if f"{prefix}_close" in df.columns:
            pc = _np_num(df, f"{prefix}_close")
            pe = _np_num(df, f"{prefix}_ema_20")
            add(f"{prefix}_ema20_cross_long_np", (pc > pe) & (_shift_np(pc) <= _shift_np(pe)) & np.isfinite(pe))
            add(f"{prefix}_ema20_cross_short_np", (pc < pe) & (_shift_np(pc) >= _shift_np(pe)) & np.isfinite(pe))
    add("range_momentum_burst_long_np", (_num(df, "range_dir_sum_3").to_numpy() >= 2) & (_num(df, "close").to_numpy() > _num(df, "prior_high_120", np.nan).to_numpy()))
    add("range_momentum_burst_short_np", (_num(df, "range_dir_sum_3").to_numpy() <= -2) & (_num(df, "close").to_numpy() < _num(df, "prior_low_120", np.nan).to_numpy()))
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "10_signal_condition_breakdown.csv", index=False)
    return out

def write_raw_signal_diagnostics(raw_count_cache: dict[tuple[str, int, str, str], dict[str, int]], out_dir: Path) -> pd.DataFrame:
    rows = []
    for (entry_model, swing_window, side_mode, signal_frame), counts in raw_count_cache.items():
        rows.append({
            "entry_model": entry_model,
            "signal_frame": signal_frame,
            "swing_window": swing_window,
            "side_mode": side_mode,
            **counts,
        })
    out = pd.DataFrame(rows).sort_values(["raw_signals", "entry_model"], ascending=[False, True]) if rows else pd.DataFrame()
    out.to_csv(out_dir / "09_raw_signal_diagnostics.csv", index=False)
    return out



def _trade_signature(trades: list[TradeResult]) -> list[tuple[Any, ...]]:
    """Compact deterministic signature for fast-vs-slow trade comparison."""
    sig: list[tuple[Any, ...]] = []
    for t in trades:
        sig.append((
            t.side,
            t.signal_time,
            t.entry_time,
            t.exit_time,
            round(float(t.entry_price), 8),
            round(float(t.exit_price), 8),
            round(float(t.net_pnl_frac), 12),
            t.exit_reason,
            int(t.hold_bars),
            int(t.adds),
            bool(t.partial_taken),
        ))
    return sig

def verify_fast_exactness(
    df: pd.DataFrame,
    specs: list[StrategySpec],
    cfg: RunConfig,
    *,
    arrays: BacktestArrays,
    signal_cache: dict[tuple[str, int, str, str, str], np.ndarray],
    index_cache: dict[tuple[str, int, str, str, str], np.ndarray],
    count_cache: dict[tuple[str, int, str, str, str], dict[str, int]] | None,
    out_dir: Path,
) -> pd.DataFrame:
    """Compare event-driven fast path against a slow bar-by-bar reference.

    Scope: verifies the speed optimization only. It does not prove that the
    strategy idea is profitable, nor does it solve OHLC intrabar ambiguity. For
    OHLC ambiguity this script still uses the conservative SL-first assumption.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    n_verify = max(0, int(cfg.verify_fast_exactness_specs))
    if count_cache:
        ranked_specs = sorted(
            specs,
            key=lambda sp: int(count_cache.get(signal_full_key(sp), {}).get("signals", 0)),
            reverse=True,
        )
        nonzero = [sp for sp in ranked_specs if int(count_cache.get(signal_full_key(sp), {}).get("signals", 0)) > 0]
        verify_specs = (nonzero or ranked_specs)[:n_verify]
    else:
        verify_specs = specs[:n_verify]
    total_verify_signals = sum(int((count_cache or {}).get(signal_full_key(sp), {}).get("signals", 0)) for sp in verify_specs)
    if cfg.fail_on_empty_audit and total_verify_signals <= 0:
        pd.DataFrame([{
            "error": "no_nonzero_signal_specs_for_audit",
            "verify_specs": len(verify_specs),
            "hint": "Check 00_data_diagnostics.csv and 09_raw_signal_diagnostics.csv and 10_signal_condition_breakdown.csv and 10_signal_condition_breakdown.csv; data coverage or signal conditions are wrong.",
        }]).to_csv(out_dir / "08_fast_exactness_check.csv", index=False)
        raise RuntimeError(
            "FAST EXACTNESS CHECK REFUSED: selected audit specs have zero signals/trades. "
            "This would be a meaningless zero-trade pass. Check diagnostics first."
        )
    print(f"[AUDIT] Fast exactness check: specs={len(verify_specs):,} total_verify_signals={total_verify_signals:,}", flush=True)
    for i, spec in enumerate(verify_specs, start=1):
        key = signal_full_key(spec)
        sig = signal_cache[key]
        idxs = index_cache[key]
        bt = SingleSpecBacktester(df, cfg, spec, arrays=arrays)
        fast = bt.run(sig, idxs)
        slow = bt.run_slow_reference(sig)
        fast_sig = _trade_signature(fast)
        slow_sig = _trade_signature(slow)
        matched = fast_sig == slow_sig
        row = {
            "spec_id": spec.spec_id,
            "system": getattr(spec, "system", "baseline"),
            "layer": getattr(spec, "layer", "unknown"),
            "entry_model": spec.entry_model,
            "signal_frame": getattr(spec, "signal_frame", "primary"),
            "regime": spec.regime,
            "structure": spec.structure,
            "fast_trades": len(fast),
            "slow_trades": len(slow),
            "matched": bool(matched),
            "first_mismatch": "",
        }
        if not matched:
            mismatch_at = 0
            for j, (a, b) in enumerate(zip(fast_sig, slow_sig)):
                if a != b:
                    mismatch_at = j
                    break
            else:
                mismatch_at = min(len(fast_sig), len(slow_sig))
            row["first_mismatch"] = str(mismatch_at)
            rows.append(row)
            pd.DataFrame(rows).to_csv(out_dir / "08_fast_exactness_check.csv", index=False)
            raise RuntimeError(
                "FAST EXACTNESS CHECK FAILED: event-driven result differs from slow bar-by-bar reference. "
                f"spec_id={spec.spec_id} mismatch_at_trade={mismatch_at}. "
                f"Report written to {out_dir / '08_fast_exactness_check.csv'}"
            )
        rows.append(row)
        if i == 1 or i % 20 == 0 or i == len(verify_specs):
            print(f"      exactness {i:,}/{len(verify_specs):,}: ok {spec.spec_id} trades={len(fast):,}", flush=True)
    audit = pd.DataFrame(rows)
    audit.to_csv(out_dir / "08_fast_exactness_check.csv", index=False)
    print("[AUDIT] Fast exactness check passed", flush=True)
    return audit

def robustness_runs(
    df: pd.DataFrame,
    specs: list[StrategySpec],
    base_cfg: RunConfig,
    top_summary: pd.DataFrame,
    *,
    arrays: BacktestArrays | None = None,
    signal_cache: dict[tuple[str, int, str, str, str], np.ndarray] | None = None,
    index_cache: dict[tuple[str, int, str, str, str], np.ndarray] | None = None,
) -> pd.DataFrame:
    if top_summary.empty or base_cfg.robustness_top_n <= 0:
        return pd.DataFrame()
    top_ids = top_summary.sort_values("score", ascending=False).head(int(base_cfg.robustness_top_n))["spec_id"].astype(str).tolist()
    spec_map = {s.spec_id: s for s in specs}
    scenarios = [
        ("base", dict()),
        ("fee_2x", dict(fee_rate_per_side=base_cfg.fee_rate_per_side * 2.0)),
        ("slippage_2x", dict(slippage_pct=base_cfg.slippage_pct * 2.0)),
        ("delay_1bar", dict(entry_delay_bars=base_cfg.entry_delay_bars + 1)),
        ("delay_3bar", dict(entry_delay_bars=base_cfg.entry_delay_bars + 3)),
        ("risk_half", dict(risk_per_trade=base_cfg.risk_per_trade * 0.5)),
    ]
    rows: list[dict[str, Any]] = []
    print(f"[5/5] Robustness checks: top_n={len(top_ids)} scenarios={len(scenarios)}", flush=True)
    local_signal_cache: dict[tuple[str, int, str, str, str], np.ndarray] = signal_cache or {}
    local_index_cache: dict[tuple[str, int, str, str, str], np.ndarray] = index_cache or {}
    if signal_cache is None or index_cache is None:
        top_specs = [spec_map[x] for x in top_ids if x in spec_map]
        local_signal_cache, local_index_cache, _, _ = build_signal_caches(df, top_specs, base_cfg)

    base_arrays = arrays or build_backtest_arrays(df, base_cfg)
    for spec_id in top_ids:
        spec = spec_map.get(spec_id)
        if spec is None:
            continue
        key = signal_full_key(spec)
        sig = local_signal_cache.get(key)
        idxs = local_index_cache.get(key)
        if sig is None:
            raw = signal_for_entry(df, spec.entry_model, spec)
            sig = apply_regime(df, raw, spec.regime, getattr(spec, "signal_frame", "primary")).astype("int8").to_numpy(copy=False)
            idxs = np.flatnonzero(sig)
        for scenario, overrides in scenarios:
            cfg = replace(base_cfg, **overrides)
            bt = SingleSpecBacktester(df, cfg, spec, arrays=base_arrays)
            trades = bt.run(sig, idxs)
            s = summarize_trades(spec, trades, cfg)
            rows.append({
                "scenario": scenario,
                "spec_id": spec.spec_id,
                "entry_model": spec.entry_model,
                "regime": spec.regime,
                "structure": spec.structure,
                "trades": s["trades"],
                "total_return_pct": s["total_return_pct"],
                "profit_factor": s["profit_factor"],
                "max_drawdown_pct": s["max_drawdown_pct"],
                "win_rate": s["win_rate"],
                "score": s["score"],
                "fee_rate_per_side": cfg.fee_rate_per_side,
                "slippage_pct": cfg.slippage_pct,
                "entry_delay_bars": cfg.entry_delay_bars,
                "risk_per_trade": cfg.risk_per_trade,
            })
    return pd.DataFrame(rows)


def build_scoreboard(summary: pd.DataFrame, robust: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    board = summary.copy()
    if not robust.empty:
        r = robust.copy()
        r["profit_factor_num"] = pd.to_numeric(r["profit_factor"].replace("inf", np.inf), errors="coerce")
        agg = r.groupby("spec_id").agg(
            robustness_scenarios=("scenario", "count"),
            robust_min_return_pct=("total_return_pct", "min"),
            robust_min_pf=("profit_factor_num", "min"),
            robust_max_dd_pct=("max_drawdown_pct", "min"),
            robust_min_score=("score", "min"),
        ).reset_index()
        board = board.merge(agg, on="spec_id", how="left")
        board["robust_pass_base"] = (board["robust_min_return_pct"].fillna(-999) > 0) & (board["robust_min_pf"].fillna(0) >= 1.0)
        board["robust_penalty"] = np.where(board["robust_pass_base"], 0.0, 35.0)
        board["final_score"] = pd.to_numeric(board["score"], errors="coerce").fillna(-999) + pd.to_numeric(board["robust_min_score"], errors="coerce").fillna(0) * 0.35 - board["robust_penalty"]
    else:
        board["final_score"] = pd.to_numeric(board["score"], errors="coerce").fillna(-999)
    board = board.sort_values("final_score", ascending=False)
    return board


# =============================================================================
# Main runner
# =============================================================================


def run_factory(cfg: RunConfig) -> None:
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bars = load_primary_bars(cfg)
    range_ctx = load_range_context(cfg, bars.index)
    footprint_ctx = load_footprint_context(cfg, bars.index)
    time_ctx = load_timeframe_contexts(cfg, bars.index)
    df = build_features(bars, range_ctx, footprint_ctx, time_ctx)
    arrays = build_backtest_arrays(df, cfg)
    data_diag = write_data_diagnostics(bars, df, cfg, arrays, out_dir)
    condition_diag = write_signal_condition_breakdown(df, cfg, out_dir)
    bt_rows = int(data_diag.loc[data_diag["metric"].eq("backtest_rows_in_requested_window"), "value"].iloc[0])
    if bt_rows <= 0:
        raise RuntimeError(
            "No rows inside requested backtest window. This research would produce fake zero trades. "
            f"Check {out_dir / '00_data_diagnostics.csv'} and your cached data coverage."
        )

    specs = generate_specs(cfg.mode, cfg.max_specs, signal_frames=_signal_frames_from_cfg(cfg), include_sanity_in_core=cfg.include_sanity_in_core)
    print(f"[4/5] Running strategy specs: mode={cfg.mode} count={len(specs):,}", flush=True)
    pd.DataFrame([asdict(s) for s in specs]).to_csv(out_dir / "01_strategy_specs.csv", index=False)

    signal_cache, index_cache, count_cache, raw_count_cache = build_signal_caches(df, specs, cfg)
    write_raw_signal_diagnostics(raw_count_cache, out_dir)

    total_full_signals = sum(int(v.get("signals", 0)) for v in count_cache.values())
    total_raw_signals = sum(int(v.get("raw_signals", 0)) for v in raw_count_cache.values())
    if cfg.fail_on_empty_signals and total_full_signals <= 0:
        pd.DataFrame([
            {
                "error": "no_signals_after_regime_filter",
                "raw_signals_total": int(total_raw_signals),
                "full_signals_total": int(total_full_signals),
                "hint": "If raw_signals_total is also zero, inspect data columns/coverage or loosen entry definitions. If raw > 0 but full = 0, regime filters are too strict.",
            }
        ]).to_csv(out_dir / "03_signal_counts.csv", index=False)
        raise RuntimeError(
            "No signals generated in the requested backtest window. Refusing to output a misleading zero-trade report. "
            f"Check {out_dir / '00_data_diagnostics.csv'} and {out_dir / '09_raw_signal_diagnostics.csv'} and {out_dir / '10_signal_condition_breakdown.csv'}."
        )

    if cfg.verify_fast_exactness:
        verify_fast_exactness(
            df,
            specs,
            cfg,
            arrays=arrays,
            signal_cache=signal_cache,
            index_cache=index_cache,
            count_cache=count_cache,
            out_dir=out_dir,
        )

    summary_rows: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    signal_count_rows: list[dict[str, Any]] = []

    for idx, spec in enumerate(specs, start=1):
        if idx == 1 or idx % 50 == 0 or idx == len(specs):
            print(f"      spec {idx:,}/{len(specs):,}: {spec.spec_id}", flush=True)
        key = signal_full_key(spec)
        sig = signal_cache[key]
        sig_idxs = index_cache[key]
        counts = count_cache.get(key, {"signals": 0, "long_signals": 0, "short_signals": 0})
        signal_count_rows.append({
            "spec_id": spec.spec_id,
            "system": getattr(spec, "system", "baseline"),
            "layer": getattr(spec, "layer", "unknown"),
            "entry_model": spec.entry_model,
            "signal_frame": getattr(spec, "signal_frame", "primary"),
            "regime": spec.regime,
            "structure": spec.structure,
            "signal_frame": getattr(spec, "signal_frame", "primary"),
            **counts,
        })
        bt = SingleSpecBacktester(df, cfg, spec, arrays=arrays)
        trades = bt.run(sig, sig_idxs)
        summary_rows.append(summarize_trades(spec, trades, cfg))
        if trades:
            _tdf = pd.DataFrame([asdict(t) for t in trades])
            _year = yearly_stats(_tdf, cfg)
            if not _year.empty:
                yearly_rows.extend(_year.to_dict("records"))
        if cfg.write_trades and trades:
            all_trades.extend(asdict(t) for t in trades)

    summary = pd.DataFrame(summary_rows).sort_values("score", ascending=False)
    signal_counts = pd.DataFrame(signal_count_rows).sort_values("signals", ascending=False)
    trades_df = pd.DataFrame(all_trades)
    yearly = pd.DataFrame(yearly_rows)
    robust = robustness_runs(
        df,
        specs,
        cfg,
        summary,
        arrays=arrays,
        signal_cache=signal_cache,
        index_cache=index_cache,
    )
    board = build_scoreboard(summary, robust)

    (out_dir / "00_config.json").write_text(json.dumps(asdict(cfg), ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    summary.to_csv(out_dir / "02_summary.csv", index=False)
    signal_counts.to_csv(out_dir / "03_signal_counts.csv", index=False)
    yearly.to_csv(out_dir / "04_yearly.csv", index=False)
    robust.to_csv(out_dir / "05_robustness.csv", index=False)
    board.to_csv(out_dir / "06_candidate_scoreboard.csv", index=False)
    if cfg.write_trades:
        trades_df.to_csv(out_dir / "07_trades.csv", index=False)

    readme = f"""# ETH Intraday Strategy Factory LeadFlywheel MULTIFRAME FAST Report

Generated by `{SCRIPT_NAME}`.

## Files
- `00_config.json`: run config
- `00_data_diagnostics.csv`: loaded data coverage / field diagnostics
- `01_strategy_specs.csv`: generated strategy combinations
- `02_summary.csv`: main backtest summary
- `03_signal_counts.csv`: signal counts by spec
- `04_yearly.csv`: yearly stats, only populated when `--write-trades` is used
- `05_robustness.csv`: fee/slippage/delay/risk stress checks
- `06_candidate_scoreboard.csv`: final sortable candidate board
- `07_trades.csv`: optional full trades, only with `--write-trades`
- `08_fast_exactness_check.csv`: optional fast-vs-slow exactness audit, only with `--verify-fast-exactness`
- `09_raw_signal_diagnostics.csv and 10_signal_condition_breakdown.csv`: raw entry signal counts before regime filtering

## V3 FAST speed changes
- features are built once;
- raw entry signals are cached by entry model / swing window;
- regime-filtered signals are cached and reused across structures;
- execution is event-driven over non-zero signal indices, not full 1m table scans per spec;
- optional `--verify-fast-exactness` compares fast event-driven output against slow bar-by-bar reference;
- range context minute aggregation avoids groupby.apply;
- LeadFlywheel supports time primary, range primary, multi-timeframe context, range context and footprint context;
- optional signal gap/cap filters zero the signal array itself, preserving fast-vs-slow audit exactness.

## Safety assumptions
- closed-bar signals only;
- next-bar open entry;
- add decisions are closed-bar decisions executed next bar open;
- same-bar TP/SL collision assumes SL first;
- default fee per side = {cfg.fee_rate_per_side}, round trip = {cfg.fee_rate_per_side * 2:.5f};
- books are not used.
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print("\nDONE", flush=True)
    print(f"Report dir: {out_dir.resolve()}", flush=True)
    if not board.empty:
        cols = [c for c in ["spec_id", "entry_model", "regime", "structure", "trades", "total_return_pct", "profit_factor", "max_drawdown_pct", "win_rate", "final_score"] if c in board]
        print(board[cols].head(20).to_string(index=False), flush=True)


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ETH intraday strategy factory LeadFlywheel MULTIFRAME FAST, single-file CoinBacktest research script.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m", choices=["1m", "5m", "15m", "30m", "1H", "4H", "1D"], help="Primary time-bar timeframe when --primary-frame=time.")
    p.add_argument("--primary-frame", default="time", choices=["time", "range"], help="Tradable primary frame. Use range to execute on OKX range bars.")
    p.add_argument("--context-timeframes", default="5m,15m", help="Comma-separated auxiliary trade-bar timeframes aligned backward as context, e.g. 5m,15m,1H. Empty string disables.")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--mode", default="core", choices=["smoke", "core", "wide"])
    p.add_argument("--max-specs", type=int, default=None)
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--risk-per-trade", type=float, default=0.0030)
    p.add_argument("--max-notional-mult", type=float, default=3.0)
    p.add_argument("--fee-rate-per-side", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.00015)
    p.add_argument("--entry-delay-bars", type=int, default=0)
    p.add_argument("--local-only", action="store_true", help="Only read existing cached DB rows; do not build missing cache from raw trades.")
    p.add_argument("--no-build-missing-cache", dest="build_missing_cache", action="store_false")
    p.set_defaults(build_missing_cache=True)
    p.add_argument("--include-range-context", action="store_true")
    p.add_argument("--range-pct", type=float, default=0.0020)
    p.add_argument("--include-footprint-context", action="store_true")
    p.add_argument("--price-step", type=float, default=1.0)
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--min-signal-gap-bars", type=int, default=0, help="Optional causal cooldown after a kept signal. 0 disables.")
    p.add_argument("--max-signals-per-spec", type=int, default=0, help="Optional cap on tradable signals per spec after filtering. 0 disables.")
    p.add_argument("--include-sanity-in-core", action="store_true", help="Allow sanity/baseline signals into core/wide. Default excludes them from formal research.")
    p.add_argument("--robustness-top-n", type=int, default=50)
    p.add_argument("--write-trades", action="store_true")
    p.add_argument("--verify-fast-exactness", action="store_true", help="Run slow bar-by-bar reference on non-zero signal specs and fail if fast path differs.")
    p.add_argument("--verify-fast-exactness-specs", type=int, default=30, help="Number of non-zero-signal specs to compare in fast exactness audit.")
    p.add_argument("--allow-empty-signals", dest="fail_on_empty_signals", action="store_false", help="Allow zero-signal reports. Default refuses them because they hide data/signal bugs.")
    p.set_defaults(fail_on_empty_signals=True)
    p.add_argument("--allow-empty-audit", dest="fail_on_empty_audit", action="store_false", help="Allow fast-vs-slow audit to pass with zero signals. Default refuses it.")
    p.set_defaults(fail_on_empty_audit=True)
    p.add_argument("--out-dir", default="data/reports/research/eth_intraday_strategy_factory_lead_flywheel_multiframe_fast_audited")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(**vars(args))
    run_factory(cfg)


if __name__ == "__main__":
    main()
