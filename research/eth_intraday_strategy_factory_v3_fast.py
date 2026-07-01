#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Intraday Strategy Factory V3 FAST - single-file CoinBacktest research script.

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
python research/eth_intraday_strategy_factory_v3_fast.py --mode smoke --max-specs 30 --out-dir data/reports/research/eth_intraday_factory_v2_smoke
python research/eth_intraday_strategy_factory_v3_fast.py --mode core --max-specs 400 --out-dir data/reports/research/eth_intraday_factory_v2_core
python research/eth_intraday_strategy_factory_v3_fast.py --mode wide --max-specs 1200 --robustness-top-n 80 --write-trades --out-dir data/reports/research/eth_intraday_factory_v2_wide

Notes
-----
- Default round-trip cost is fee_rate_per_side * 2 = 0.11%, matching the current
  project fee assumption when fee_rate_per_side=0.00055.
- Books are intentionally not used.
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

SCRIPT_NAME = "eth_intraday_strategy_factory_v3_fast"


# =============================================================================
# Config / specs
# =============================================================================


@dataclass(frozen=True)
class StrategySpec:
    spec_id: str
    entry_model: str
    regime: str
    structure: str
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


@dataclass(frozen=True)
class RunConfig:
    symbol: str = "ETH-USDT-SWAP"
    timeframe: str = "1m"
    start_date: str = "2023-01-01"
    end_date: str = "2026-06-30"
    warmup_start_date: str = "2022-01-01"
    data_dir: str | None = None
    mode: str = "core"
    max_specs: int | None = None
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.0030
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
    robustness_top_n: int = 50
    write_trades: bool = False
    out_dir: str = "data/reports/research/eth_intraday_strategy_factory_v3_fast"


STRUCTURE_PRESETS: dict[str, dict[str, Any]] = {
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
}

ENTRY_MODELS = [
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
]

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
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            raise RuntimeError(f"Trade bars missing required column: {col}")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    print(f"      rows={len(df):,} index={df.index[0]} -> {df.index[-1]}", flush=True)
    return df


def load_range_context(cfg: RunConfig, base_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Optional context from the existing OKXRangeBarLoader, aligned to trade-bar minutes."""
    if not cfg.include_range_context:
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

    x = rb.copy().sort_index()
    x.index = pd.to_datetime(x.index).tz_localize(None)
    minute = x.index.floor("min")
    ctx = pd.DataFrame(index=base_index)
    # Fast vectorized minute aggregation. Avoid groupby.apply over millions of range bars.
    if "direction" not in x.columns:
        x["direction"] = 0.0
    if "delta_notional" not in x.columns:
        x["delta_notional"] = 0.0
    if "notional" not in x.columns:
        x["notional"] = 0.0
    if {"close", "low", "high"}.issubset(x.columns):
        span = (pd.to_numeric(x["high"], errors="coerce") - pd.to_numeric(x["low"], errors="coerce")).replace(0, np.nan)
        x["_rf_close_pos"] = ((pd.to_numeric(x["close"], errors="coerce") - pd.to_numeric(x["low"], errors="coerce")) / span).clip(0.0, 1.0).fillna(0.5)
    else:
        x["_rf_close_pos"] = 0.5
    agg = x.groupby(minute).agg(
        rf_bar_count=("close", "size"),
        rf_direction_sum=("direction", "sum"),
        rf_delta_notional_sum=("delta_notional", "sum"),
        rf_notional_sum=("notional", "sum"),
        rf_close_pos_mean=("_rf_close_pos", "mean"),
    )
    agg.index = pd.to_datetime(agg.index).tz_localize(None)
    ctx = ctx.join(agg, how="left").fillna(0.0)
    ctx["rf_imbalance"] = _safe_div(ctx["rf_delta_notional_sum"], ctx["rf_notional_sum"].abs(), 0.0).fillna(0.0)
    print(f"      range rows={len(x):,}; aligned_minutes={int((ctx['rf_bar_count'] > 0).sum()):,}", flush=True)
    return ctx


def load_footprint_context(cfg: RunConfig, base_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Optional light footprint aggregation via existing OKXRangeFootprintLoader.

    This stays intentionally small: one-row-per-minute max buy/sell bucket pressure.
    It is disabled by default because range-bar context is cheaper and enough for
    first-pass strategy-factory screening.
    """
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
    ctx = pd.DataFrame(index=base_index).join(agg, how="left").fillna(0.0)
    ctx["fp_absorption_hint"] = _safe_div(ctx["fp_large_buy_sum"] - ctx["fp_large_sell_sum"], ctx["fp_large_buy_sum"] + ctx["fp_large_sell_sum"], 0.0).fillna(0.0)
    print(f"      footprint rows={len(x):,}; aligned_minutes={int((ctx.abs().sum(axis=1) > 0).sum()):,}", flush=True)
    return ctx


# =============================================================================
# Feature engineering: closed-bar / past-only thresholds
# =============================================================================


def build_features(bars: pd.DataFrame, range_ctx: pd.DataFrame, footprint_ctx: pd.DataFrame) -> pd.DataFrame:
    print("[3/5] Building closed-bar feature frame...", flush=True)
    df = bars.copy().sort_index()
    df = df.join(range_ctx, how="left").join(footprint_ctx, how="left")
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df.fillna(0.0)

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
    df["ret_abs_q90_past"] = df["ret_1"].abs().shift(1).rolling(480, min_periods=120).quantile(0.90)

    df["vol_regime_high"] = df["bar_range_pct"] > df["range_q70_past"]
    df["vol_regime_low"] = df["bar_range_pct"] < df["range_q30_past"]
    df["range_regime"] = (~df["trend_up"]) & (~df["trend_down"]) & (df["bar_range_pct"] < df["range_q70_past"])

    # Opening range: for bars after first 60 minutes, first-hour high/low is already known.
    session_min = pd.Series(df.groupby(day).cumcount().to_numpy(), index=df.index)
    df["session_minute"] = session_min
    first_hour = session_min < 60
    or_high = high.where(first_hour).groupby(day).transform("max")
    or_low = low.where(first_hour).groupby(day).transform("min")
    df["opening_range_high"] = or_high
    df["opening_range_low"] = or_low
    df["after_opening_range"] = session_min >= 60

    # Optional range/footprint context defaults.
    for col in ["rf_bar_count", "rf_direction_sum", "rf_imbalance", "rf_close_pos_mean", "fp_absorption_hint"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["rf_speed_q75_past"] = df["rf_bar_count"].shift(1).rolling(480, min_periods=120).quantile(0.75)

    print(f"      feature rows={len(df):,}", flush=True)
    return df


# =============================================================================
# Signal generation
# =============================================================================


def signal_for_entry(df: pd.DataFrame, entry_model: str, spec: StrategySpec) -> pd.Series:
    close = _num(df, "close")
    high = _num(df, "high")
    low = _num(df, "low")
    open_ = _num(df, "open")
    vol = _num(df, "volume")
    delta = _num(df, "delta_ratio")
    close_pos = _num(df, "close_pos", 0.5)
    lower_wick = _num(df, "lower_wick_pct")
    upper_wick = _num(df, "upper_wick_pct")
    ret = _num(df, "ret_1")
    out = pd.Series(0, index=df.index, dtype="int8")

    prior_high = _num(df, f"prior_high_{spec.swing_window}", np.nan)
    prior_low = _num(df, f"prior_low_{spec.swing_window}", np.nan)
    high_vol = vol > _num(df, "vol_q75_past", np.inf)
    extreme_vol = vol > _num(df, "vol_q90_past", np.inf)

    if entry_model == "sweep_reclaim":
        long_m = low.lt(prior_low) & close.gt(prior_low) & close_pos.ge(0.58) & lower_wick.ge(0.25) & high_vol
        short_m = high.gt(prior_high) & close.lt(prior_high) & close_pos.le(0.42) & upper_wick.ge(0.25) & high_vol
    elif entry_model == "failed_breakout_reclaim":
        long_m = low.lt(prior_low) & close.gt(open_) & close_pos.ge(0.55) & delta.gt(-0.20)
        short_m = high.gt(prior_high) & close.lt(open_) & close_pos.le(0.45) & delta.lt(0.20)
    elif entry_model == "trend_pullback_reclaim":
        ema = _num(df, "ema_60")
        vwap = _num(df, "session_vwap")
        long_m = _bool(df, "trend_up") & low.lt(np.minimum(ema, vwap)) & close.gt(ema) & delta.gt(-0.10)
        short_m = _bool(df, "trend_down") & high.gt(np.maximum(ema, vwap)) & close.lt(ema) & delta.lt(0.10)
    elif entry_model == "vwap_deviation_reversion":
        dev = _num(df, "vwap_dist_pct")
        th = _num(df, "vwap_dev_q90_past", np.inf)
        long_m = dev.lt(-th) & close_pos.ge(0.62) & delta.gt(-0.35) & _bool(df, "range_regime")
        short_m = dev.gt(th) & close_pos.le(0.38) & delta.lt(0.35) & _bool(df, "range_regime")
    elif entry_model == "cvd_divergence_reversal":
        cvd = _num(df, "cvd")
        cvd_low = cvd.shift(1).rolling(240, min_periods=80).min()
        cvd_high = cvd.shift(1).rolling(240, min_periods=80).max()
        long_m = low.lt(prior_low) & cvd.gt(cvd_low) & close_pos.ge(0.58) & delta.gt(-0.50)
        short_m = high.gt(prior_high) & cvd.lt(cvd_high) & close_pos.le(0.42) & delta.lt(0.50)
    elif entry_model == "liquidation_panic_reversal":
        rth = _num(df, "ret_abs_q90_past", np.inf)
        long_m = ret.lt(-rth) & extreme_vol & close_pos.ge(0.55) & lower_wick.ge(0.20)
        short_m = ret.gt(rth) & extreme_vol & close_pos.le(0.45) & upper_wick.ge(0.20)
    elif entry_model == "compression_breakout_retest":
        compressed = _num(df, "bar_range_pct").shift(1).rolling(30, min_periods=10).mean() < _num(df, "range_q30_past")
        long_m = compressed & close.gt(prior_high) & _bool(df, "trend_up") & delta.gt(0.0)
        short_m = compressed & close.lt(prior_low) & _bool(df, "trend_down") & delta.lt(0.0)
    elif entry_model == "microtrend_continuation":
        hh = close.gt(close.shift(1)) & close.shift(1).gt(close.shift(2))
        ll = close.lt(close.shift(1)) & close.shift(1).lt(close.shift(2))
        long_m = _bool(df, "trend_up") & hh & low.gt(low.shift(3).rolling(8, min_periods=3).min()) & delta.gt(-0.05)
        short_m = _bool(df, "trend_down") & ll & high.lt(high.shift(3).rolling(8, min_periods=3).max()) & delta.lt(0.05)
    elif entry_model == "opening_range_fakeout":
        orh = _num(df, "opening_range_high", np.nan)
        orl = _num(df, "opening_range_low", np.nan)
        active = _bool(df, "after_opening_range") & _num(df, "session_minute").le(360)
        long_m = active & low.lt(orl) & close.gt(orl) & close_pos.ge(0.55)
        short_m = active & high.gt(orh) & close.lt(orh) & close_pos.le(0.45)
    elif entry_model == "opening_range_breakout":
        orh = _num(df, "opening_range_high", np.nan)
        orl = _num(df, "opening_range_low", np.nan)
        active = _bool(df, "after_opening_range") & _num(df, "session_minute").le(480)
        long_m = active & close.gt(orh) & delta.gt(0.0) & high_vol
        short_m = active & close.lt(orl) & delta.lt(0.0) & high_vol
    elif entry_model == "range_bar_momentum_proxy":
        rf_count = _num(df, "rf_bar_count")
        rf_fast = rf_count.gt(_num(df, "rf_speed_q75_past", np.inf))
        rf_imb = _num(df, "rf_imbalance")
        long_m = rf_fast & rf_imb.gt(0.03) & _bool(df, "trend_up")
        short_m = rf_fast & rf_imb.lt(-0.03) & _bool(df, "trend_down")
    else:
        raise ValueError(f"Unknown entry_model: {entry_model}")

    if spec.side_mode in {"both", "long_only"}:
        out.loc[long_m.fillna(False)] = 1
    if spec.side_mode in {"both", "short_only"}:
        out.loc[short_m.fillna(False)] = -1
    return out


def apply_regime(df: pd.DataFrame, raw_signal: pd.Series, regime: str) -> pd.Series:
    sig = raw_signal.copy().astype("int8")
    if regime == "any":
        return sig
    if regime == "trend_aligned":
        ok_long = _bool(df, "trend_up")
        ok_short = _bool(df, "trend_down")
    elif regime == "range_only":
        ok_long = ok_short = _bool(df, "range_regime")
    elif regime == "high_vol":
        ok_long = ok_short = _bool(df, "vol_regime_high")
    elif regime == "low_vol":
        ok_long = ok_short = _bool(df, "vol_regime_low")
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


def generate_specs(mode: str, max_specs: int | None = None) -> list[StrategySpec]:
    if mode == "smoke":
        entries = ["sweep_reclaim", "trend_pullback_reclaim", "vwap_deviation_reversion", "opening_range_fakeout"]
        regimes = ["any", "trend_aligned"]
        structures = ["fixed_fast", "time_bomb", "partial_runner"]
        swing_windows = [120, 240]
    elif mode == "core":
        entries = ENTRY_MODELS
        regimes = ["any", "trend_aligned", "range_only", "high_vol", "eu_us_session"]
        structures = ["fixed_fast", "fixed_balanced", "time_bomb", "fail_fast", "partial_runner", "probe_confirm_add", "anti_martingale_1r"]
        swing_windows = [120, 240]
    elif mode == "wide":
        entries = ENTRY_MODELS
        regimes = REGIMES
        structures = list(STRUCTURE_PRESETS)
        swing_windows = [60, 120, 240, 480]
    else:
        raise ValueError("mode must be smoke/core/wide")

    specs: list[StrategySpec] = []
    sid = 0
    for entry in entries:
        for regime in regimes:
            for structure in structures:
                for swing in swing_windows:
                    base = StrategySpec(
                        spec_id="",
                        entry_model=entry,
                        regime=regime,
                        structure=structure,
                        swing_window=swing,
                    )
                    kwargs = STRUCTURE_PRESETS.get(structure, {})
                    spec = replace(base, **kwargs)
                    sid += 1
                    specs.append(replace(spec, spec_id=f"S{sid:05d}_{entry}_{regime}_{structure}_sw{swing}"))
                    if max_specs is not None and len(specs) >= int(max_specs):
                        return specs
    return specs


# =============================================================================
# Backtester
# =============================================================================


@dataclass
class TradeResult:
    spec_id: str
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
    idx = pd.DatetimeIndex(pd.to_datetime(df.index).tz_localize(None))
    idx_ns = idx.view("int64")
    start_ns = pd.Timestamp(cfg.start_date).tz_localize(None).value
    end_ns = pd.Timestamp(cfg.end_date).tz_localize(None).value
    start_i = int(np.searchsorted(idx_ns, start_ns, side="left"))
    end_i = int(np.searchsorted(idx_ns, end_ns, side="right") - 1)
    return BacktestArrays(
        open=_num(df, "open").to_numpy(dtype="float64", copy=False),
        high=_num(df, "high").to_numpy(dtype="float64", copy=False),
        low=_num(df, "low").to_numpy(dtype="float64", copy=False),
        close=_num(df, "close").to_numpy(dtype="float64", copy=False),
        atr=_num(df, "atr_60").to_numpy(dtype="float64", copy=False),
        index=idx,
        index_ns=idx_ns,
        start_i=max(0, start_i),
        end_i=min(len(idx) - 1, max(0, end_i)),
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
        initial_notional = self._notional_for_risk(stop_dist, entry_price, 1.0)
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

            if spec.structure in {"partial_runner", "slow_runner"} and (not partial_taken) and partial_hit:
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
            if spec.structure in {"probe_confirm_add", "anti_martingale_1r", "breakout_add_runner"}:
                if (not add1_done) and close_r >= spec.add_trigger_r1 and spec.add_size_1 > 0:
                    pending_add_mult = float(spec.add_size_1)
                    add1_done = True
                elif (not add2_done) and close_r >= spec.add_trigger_r2 and spec.add_size_2 > 0:
                    pending_add_mult = float(spec.add_size_2)
                    add2_done = True

            if spec.structure in {"partial_runner", "slow_runner", "breakout_add_runner"} and max_fav >= 1.0:
                trail_dist = max(float(self.atr[j]) * spec.trail_atr_mult, stop_dist * 0.60)
                if side == 1:
                    stop_price = max(stop_price, c - trail_dist)
                else:
                    stop_price = min(stop_price, c + trail_dist)

            age = j - entry_i + 1
            if spec.structure == "time_bomb" and age >= spec.time_bomb_bars and max_fav < spec.time_bomb_min_mfe_r:
                pending_time_exit = True
            if spec.structure == "fail_fast" and age >= spec.fail_fast_bars and close_r <= -abs(spec.fail_fast_adverse_r):
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
            "long_trades": 0,
            "short_trades": 0,
            "score": -999.0,
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
    score = score_candidate(total_ret, pf, dd, len(t), win_rate, avg_ret)
    row.update({
        "total_return_pct": round(total_ret * 100.0, 4),
        "final_capital": round(float(equity.iloc[-1]), 4),
        "win_rate": round(win_rate * 100.0, 4),
        "profit_factor": round(pf, 4) if math.isfinite(pf) else "inf",
        "max_drawdown_pct": round(dd * 100.0, 4),
        "avg_return_pct": round(avg_ret, 5),
        "median_return_pct": round(float(pd.to_numeric(t["return_pct"], errors="coerce").median()), 5),
        "avg_hold_bars": round(float(pd.to_numeric(t["hold_bars"], errors="coerce").mean()), 3),
        "avg_mfe_r": round(float(pd.to_numeric(t["mfe_r"], errors="coerce").mean()), 4),
        "avg_mae_r": round(float(pd.to_numeric(t["mae_r"], errors="coerce").mean()), 4),
        "long_trades": int((t["side"] == "LONG").sum()),
        "short_trades": int((t["side"] == "SHORT").sum()),
        "score": round(score, 4),
    })
    return row


def score_candidate(total_ret: float, pf: float, dd: float, trades: int, win_rate: float, avg_ret_pct: float) -> float:
    if trades < 20:
        return -100.0 + trades
    pf_score = min(max((pf - 1.0) / 1.0, -1.0), 2.0) if math.isfinite(pf) else 2.0
    ret_score = min(max(total_ret / 1.0, -1.0), 3.0)
    dd_score = min(max((0.20 + dd) / 0.20, -2.0), 1.0)  # dd is negative
    trade_score = min(math.log1p(trades) / math.log1p(1000), 1.2)
    win_score = min(max((win_rate - 0.45) / 0.20, -1.0), 1.0)
    avg_score = min(max(avg_ret_pct / 0.10, -1.0), 1.0)
    return 30 * pf_score + 25 * ret_score + 20 * dd_score + 10 * trade_score + 10 * win_score + 5 * avg_score


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


def signal_raw_key(spec: StrategySpec) -> tuple[str, int, str]:
    return (str(spec.entry_model), int(spec.swing_window), str(spec.side_mode))


def signal_full_key(spec: StrategySpec) -> tuple[str, int, str, str]:
    return (str(spec.entry_model), int(spec.swing_window), str(spec.side_mode), str(spec.regime))


def build_signal_caches(
    df: pd.DataFrame,
    specs: list[StrategySpec],
    cfg: RunConfig,
) -> tuple[dict[tuple[str, int, str, str], np.ndarray], dict[tuple[str, int, str, str], np.ndarray], dict[tuple[str, int, str, str], dict[str, int]]]:
    """Precompute and cache signals shared by many StrategySpec variants.

    Most factory specs differ only by position structure. Recomputing the same
    rolling-entry masks and rescanning all bars per structure was the main V2
    bottleneck. This cache computes each entry/swing signal once, each regime
    filter once, and stores non-zero signal indices for event-driven execution.
    """
    arrays = build_backtest_arrays(df, cfg)
    raw_cache: dict[tuple[str, int, str], pd.Series] = {}
    signal_cache: dict[tuple[str, int, str, str], np.ndarray] = {}
    index_cache: dict[tuple[str, int, str, str], np.ndarray] = {}
    count_cache: dict[tuple[str, int, str, str], dict[str, int]] = {}

    unique_raw: dict[tuple[str, int, str], StrategySpec] = {}
    unique_full: dict[tuple[str, int, str, str], StrategySpec] = {}
    for spec in specs:
        unique_raw.setdefault(signal_raw_key(spec), spec)
        unique_full.setdefault(signal_full_key(spec), spec)

    print(f"      precomputing raw signals: {len(unique_raw):,}; regime signals: {len(unique_full):,}", flush=True)
    for k, spec in unique_raw.items():
        raw_cache[k] = signal_for_entry(df, spec.entry_model, spec).astype("int8")

    lo = int(arrays.start_i)
    hi = int(max(arrays.start_i, min(arrays.end_i, len(df) - 1)))
    for k, spec in unique_full.items():
        sig_series = apply_regime(df, raw_cache[signal_raw_key(spec)], spec.regime)
        sig = sig_series.astype("int8").to_numpy(copy=False)
        idxs = np.flatnonzero(sig)
        if idxs.size:
            idxs = idxs[(idxs >= lo) & (idxs <= hi)]
        signal_cache[k] = sig
        index_cache[k] = idxs.astype(np.int64, copy=False)
        if idxs.size:
            vals = sig[idxs]
            longs = int(np.sum(vals == 1))
            shorts = int(np.sum(vals == -1))
        else:
            longs = shorts = 0
        count_cache[k] = {"signals": int(longs + shorts), "long_signals": longs, "short_signals": shorts}
    return signal_cache, index_cache, count_cache


def robustness_runs(
    df: pd.DataFrame,
    specs: list[StrategySpec],
    base_cfg: RunConfig,
    top_summary: pd.DataFrame,
    *,
    arrays: BacktestArrays | None = None,
    signal_cache: dict[tuple[str, int, str, str], np.ndarray] | None = None,
    index_cache: dict[tuple[str, int, str, str], np.ndarray] | None = None,
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
    local_signal_cache: dict[tuple[str, int, str, str], np.ndarray] = signal_cache or {}
    local_index_cache: dict[tuple[str, int, str, str], np.ndarray] = index_cache or {}
    if signal_cache is None or index_cache is None:
        top_specs = [spec_map[x] for x in top_ids if x in spec_map]
        local_signal_cache, local_index_cache, _ = build_signal_caches(df, top_specs, base_cfg)

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
            sig = apply_regime(df, raw, spec.regime).astype("int8").to_numpy(copy=False)
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
        board["robust_penalty"] = np.where(board["robust_min_return_pct"].fillna(-999) < 0, 25.0, 0.0)
        board["final_score"] = pd.to_numeric(board["score"], errors="coerce").fillna(-999) + pd.to_numeric(board["robust_min_score"], errors="coerce").fillna(0) * 0.25 - board["robust_penalty"]
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
    bars = load_trade_bars(cfg)
    range_ctx = load_range_context(cfg, bars.index)
    footprint_ctx = load_footprint_context(cfg, bars.index)
    df = build_features(bars, range_ctx, footprint_ctx)
    arrays = build_backtest_arrays(df, cfg)

    specs = generate_specs(cfg.mode, cfg.max_specs)
    print(f"[4/5] Running strategy specs: mode={cfg.mode} count={len(specs):,}", flush=True)
    pd.DataFrame([asdict(s) for s in specs]).to_csv(out_dir / "01_strategy_specs.csv", index=False)

    signal_cache, index_cache, count_cache = build_signal_caches(df, specs, cfg)

    summary_rows: list[dict[str, Any]] = []
    all_trades: list[dict[str, Any]] = []
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
            "entry_model": spec.entry_model,
            "regime": spec.regime,
            "structure": spec.structure,
            **counts,
        })
        bt = SingleSpecBacktester(df, cfg, spec, arrays=arrays)
        trades = bt.run(sig, sig_idxs)
        summary_rows.append(summarize_trades(spec, trades, cfg))
        if cfg.write_trades and trades:
            all_trades.extend(asdict(t) for t in trades)

    summary = pd.DataFrame(summary_rows).sort_values("score", ascending=False)
    signal_counts = pd.DataFrame(signal_count_rows).sort_values("signals", ascending=False)
    trades_df = pd.DataFrame(all_trades)
    yearly = yearly_stats(trades_df, cfg) if cfg.write_trades else pd.DataFrame()
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

    readme = f"""# ETH Intraday Strategy Factory V3 FAST Report

Generated by `{SCRIPT_NAME}`.

## Files
- `00_config.json`: run config
- `01_strategy_specs.csv`: generated strategy combinations
- `02_summary.csv`: main backtest summary
- `03_signal_counts.csv`: signal counts by spec
- `04_yearly.csv`: yearly stats, only populated when `--write-trades` is used
- `05_robustness.csv`: fee/slippage/delay/risk stress checks
- `06_candidate_scoreboard.csv`: final sortable candidate board
- `07_trades.csv`: optional full trades, only with `--write-trades`

## V3 FAST speed changes
- features are built once;
- raw entry signals are cached by entry model / swing window;
- regime-filtered signals are cached and reused across structures;
- execution is event-driven over non-zero signal indices, not full 1m table scans per spec;
- range context minute aggregation avoids groupby.apply.

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
    p = argparse.ArgumentParser(description="ETH intraday strategy factory V3 FAST, single-file CoinBacktest research script.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m", choices=["1m", "5m", "15m", "30m", "1H", "4H", "1D"])
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
    p.add_argument("--robustness-top-n", type=int, default=50)
    p.add_argument("--write-trades", action="store_true")
    p.add_argument("--out-dir", default="data/reports/research/eth_intraday_strategy_factory_v3_fast")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RunConfig(**vars(args))
    run_factory(cfg)


if __name__ == "__main__":
    main()
