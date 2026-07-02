#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Follower-Friendly Strategy Factory V1
=====================================

Research-only strategy factory for ETH perpetual follower/copy-trading friendly
engines. It is intentionally independent from V10B/portfolio routing.

Scope:
    - Generate many explainable candidate specs.
    - Backtest each spec as a standalone single-position engine.
    - Enforce closed-bar signal -> next-bar-open execution.
    - Causally align higher-timeframe context by available_time, not bar_start.
    - Record MFE/MAE on every trade so weak strategies can be diagnosed instead
      of blindly discarded.
    - Stress shortlisted candidates with fee_2x, slippage_2x and delay_1bar.

Not in scope:
    - No portfolio merge.
    - No V10B changes.
    - No AetherEdge/live trading code.
    - No non-causal high-timeframe ffill.

Run examples:
    python research/follower_friendly_strategy_factory/factory_v1.py --fast
    python research/follower_friendly_strategy_factory/factory_v1.py --max-specs 3000 --write-top-trades

Notes:
    The default fee/slippage is conservative for strategy discovery. You can set
    --fee-rate 0.00055 to approximate a 0.11% round trip fee before slippage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import sys
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

CURRENT_FILE = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.backtest_common.data import load_ohlcv_data as load_data  # noqa: E402
from src.backtest_common.execution import apply_entry_slippage, apply_exit_slippage  # noqa: E402
from src.backtest_common.indicators import adx, atr, ema  # noqa: E402

STRATEGY_NAME = "follower_friendly_strategy_factory_v1"
DEFAULT_SYMBOL = "ETH-USDT-SWAP"
DEFAULT_START_DATE = "2023-01-01"
DEFAULT_END_DATE = "2026-06-30"
DEFAULT_WARMUP_START_DATE = "2022-01-01"

# Current invalid families from prior research. Do not regenerate exact known-bad
# structures in this factory V1. Future versions can explicitly override this
# after changing signal timing, feature definition, or exit structure.
BANNED_BASELINES = {
    "ema20_cross_1m",
    "vwap_cross_1m",
    "price_momentum_1m",
    "prior20_breakout_1m",
    "range_bar_primary_momentum_burst",
    "range_bar_primary_pullback_reclaim",
    "non_causal_tf5m_context",
}


@dataclass(frozen=True)
class StrategySpec:
    spec_id: str
    family: str
    primary_timeframe: str
    context_timeframe: str
    side_mode: str
    fast_ema: int
    slow_ema: int
    atr_period: int
    adx_period: int
    min_adx: float
    max_adx: float
    volume_ratio_min: float
    pullback_atr: float
    zscore_window: int
    zscore_entry: float
    breakout_lookback: int
    retest_atr: float
    squeeze_window: int
    squeeze_quantile: float
    stop_atr_mult: float
    tp_r: float
    max_hold_bars: int
    cooldown_bars: int
    entry_model: str
    exit_model: str = "fixed_sl_tp_time"


@dataclass(frozen=True)
class BacktestConfig:
    symbol: str = DEFAULT_SYMBOL
    initial_capital: float = 1000.0
    fee_rate: float = 0.00055
    slippage_pct: float = 0.0002
    notional_mult: float = 1.0
    allow_short: bool = True
    conservative_same_bar: bool = True
    delay_bars: int = 0
    min_stop_pct: float = 0.0015


def _timeframe_delta(timeframe: str) -> pd.Timedelta:
    mapping = {
        "1m": pd.Timedelta(minutes=1),
        "3m": pd.Timedelta(minutes=3),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "1H": pd.Timedelta(hours=1),
        "4H": pd.Timedelta(hours=4),
        "1D": pd.Timedelta(days=1),
    }
    if timeframe not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return mapping[timeframe]


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "UNKNOWN"




def _format_seconds(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        return "?:??"
    seconds_i = int(seconds)
    if seconds_i < 60:
        return f"{seconds_i}s"
    minutes, sec = divmod(seconds_i, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minute = divmod(minutes, 60)
    return f"{hours}h{minute:02d}m"


def _progress_update(
    label: str,
    done: int,
    total: int,
    started_at: float,
    *,
    every: int = 25,
    enabled: bool = True,
    force: bool = False,
) -> None:
    """Small dependency-free progress bar.

    tqdm is intentionally not required so the script stays portable on Windows
    and Unix. In a terminal it updates one line; in redirected logs it emits
    periodic complete lines.
    """
    if not enabled or total <= 0:
        return
    every = max(1, int(every))
    if not force and done < total and done % every != 0:
        return
    elapsed = max(0.0, time.perf_counter() - started_at)
    rate = done / elapsed if elapsed > 0 else 0.0
    eta = (total - done) / rate if rate > 0 else float("nan")
    width = 28
    frac = min(1.0, max(0.0, done / total))
    filled = int(round(width * frac))
    bar = "#" * filled + "." * (width - filled)
    msg = (
        f"{label} [{bar}] {done:,}/{total:,} "
        f"({frac * 100:5.1f}%) elapsed={_format_seconds(elapsed)} "
        f"eta={_format_seconds(eta)} rate={rate:.2f}/s"
    )
    if sys.stdout.isatty():
        print("\r" + msg, end="\n" if done >= total else "", flush=True)
    else:
        print(msg, flush=True)

def _spec_id(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return "FF" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:10].upper()


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(float(default), index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").fillna(float(default))


def _bool_series(index: pd.Index, value: bool = False) -> pd.Series:
    return pd.Series(bool(value), index=index, dtype=bool)


# -----------------------------------------------------------------------------
# Data / feature preparation
# -----------------------------------------------------------------------------


def load_market_frames(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    """Load primary and higher-timeframe OHLCV once.

    Warmup defaults follow the project convention: data loads from 2022-01-01,
    while trades are evaluated from 2023-01-01 onward.
    """
    if getattr(args, "data_source", "ohlcv") != "ohlcv":
        raise ValueError("V1.2 only supports --data-source ohlcv. Use V2 for trade_bar/range_bar/footprint sources.")
    frames: dict[str, pd.DataFrame] = {}
    needed = {args.primary_timeframe, args.context_timeframe}
    if args.extra_context_timeframe:
        needed.add(args.extra_context_timeframe)
    for tf in sorted(needed, key=lambda x: _timeframe_delta(x)):
        print(f"[load] source={args.data_source} {args.symbol} {tf} {args.warmup_start_date}->{args.end_date}", flush=True)
        df = load_data(args.symbol, args.warmup_start_date, args.end_date, tf)
        if df.empty:
            raise RuntimeError(f"No data loaded for {args.symbol} {tf}")
        frames[tf] = df.sort_index().copy()
        print(f"       rows={len(df):,} {df.index[0]} -> {df.index[-1]}", flush=True)
    return frames


def build_primary_features(primary: pd.DataFrame) -> pd.DataFrame:
    out = primary.sort_index().copy()
    close = _num(out, "close", np.nan)
    high = _num(out, "high", np.nan)
    low = _num(out, "low", np.nan)
    open_ = _num(out, "open", np.nan)
    volume = _num(out, "volume", 0.0)

    for span in sorted({8, 12, 20, 34, 50, 89, 144, 200}):
        out[f"ema{span}"] = ema(close, span)

    for length in sorted({14, 20, 28}):
        out[f"atr{length}"] = atr(out, length)
        out[f"atr_pct{length}"] = out[f"atr{length}"] / close.replace(0, np.nan)

    out["bar_range"] = (high - low).replace(0, np.nan)
    out["close_pos"] = ((close - low) / out["bar_range"]).clip(0.0, 1.0)
    out["body_pct"] = ((close - open_).abs() / out["bar_range"]).clip(0.0, 1.0)
    out["upper_wick_pct"] = ((high - np.maximum(open_, close)) / out["bar_range"]).clip(0.0, 1.0)
    out["lower_wick_pct"] = ((np.minimum(open_, close) - low) / out["bar_range"]).clip(0.0, 1.0)
    out["ret_1"] = close.pct_change()
    out["ret_3"] = close.pct_change(3)

    # Rolling volume is shifted to avoid comparing against the current bar's
    # final volume before the signal bar has closed. Because signals are made at
    # the close of this bar, current volume is allowed in the signal itself, but
    # the baseline must be past-only.
    vol_base = volume.shift(1).rolling(144, min_periods=30).median().replace(0, np.nan)
    out["volume_ratio"] = volume / vol_base

    for win in sorted({48, 72, 96, 144, 288}):
        pv = close * volume
        rv = volume.rolling(win, min_periods=max(10, win // 4)).sum().replace(0, np.nan)
        out[f"rvwap{win}"] = pv.rolling(win, min_periods=max(10, win // 4)).sum() / rv
        spread = close / out[f"rvwap{win}"] - 1.0
        spread_std = spread.shift(1).rolling(win, min_periods=max(20, win // 3)).std().replace(0, np.nan)
        out[f"vwap_z{win}"] = spread / spread_std

    for lb in sorted({12, 24, 48, 72, 96, 144}):
        out[f"prior_high_{lb}"] = high.shift(1).rolling(lb, min_periods=max(5, lb // 3)).max()
        out[f"prior_low_{lb}"] = low.shift(1).rolling(lb, min_periods=max(5, lb // 3)).min()
        out[f"break_high_{lb}"] = close > out[f"prior_high_{lb}"]
        out[f"break_low_{lb}"] = close < out[f"prior_low_{lb}"]
        out[f"recent_break_high_{lb}"] = out[f"break_high_{lb}"].shift(1).rolling(max(3, lb // 6), min_periods=1).max().fillna(0).astype(bool)
        out[f"recent_break_low_{lb}"] = out[f"break_low_{lb}"].shift(1).rolling(max(3, lb // 6), min_periods=1).max().fillna(0).astype(bool)

    for win in sorted({72, 96, 144, 288}):
        range_pct = (high - low) / close.replace(0, np.nan)
        out[f"range_pct_med_{win}"] = range_pct.shift(1).rolling(win, min_periods=max(20, win // 3)).median()
        out[f"range_pct_q_{win}_20"] = range_pct.shift(1).rolling(win, min_periods=max(20, win // 3)).quantile(0.20)
        out[f"squeeze_{win}_q20"] = range_pct <= out[f"range_pct_q_{win}_20"]
        out[f"recent_squeeze_{win}"] = out[f"squeeze_{win}_q20"].shift(1).rolling(max(6, win // 8), min_periods=1).max().fillna(0).astype(bool)

    return out


def build_context_features(ctx: pd.DataFrame, timeframe: str, prefix: str) -> pd.DataFrame:
    out = ctx.sort_index().copy()
    close = _num(out, "close", np.nan)
    out[f"{prefix}_ema20"] = ema(close, 20)
    out[f"{prefix}_ema50"] = ema(close, 50)
    out[f"{prefix}_ema100"] = ema(close, 100)
    out[f"{prefix}_atr14"] = atr(out, 14)
    out[f"{prefix}_atr_pct14"] = out[f"{prefix}_atr14"] / close.replace(0, np.nan)
    out[f"{prefix}_adx14"] = adx(out, 14)
    out[f"{prefix}_trend_up"] = (close > out[f"{prefix}_ema50"]) & (out[f"{prefix}_ema20"] > out[f"{prefix}_ema50"])
    out[f"{prefix}_trend_down"] = (close < out[f"{prefix}_ema50"]) & (out[f"{prefix}_ema20"] < out[f"{prefix}_ema50"])
    out[f"{prefix}_choppy"] = (out[f"{prefix}_adx14"] < 18.0) | ((out[f"{prefix}_ema20"] / out[f"{prefix}_ema50"] - 1.0).abs() < 0.006)
    out[f"{prefix}_bar_start_time"] = out.index
    out[f"{prefix}_available_time"] = out.index + _timeframe_delta(timeframe)
    keep = [c for c in out.columns if c.startswith(prefix)]
    aligned = out[keep].copy()
    aligned.index = aligned[f"{prefix}_available_time"]
    return aligned.sort_index()


def causal_merge_context(primary_features: pd.DataFrame, context_features: pd.DataFrame) -> pd.DataFrame:
    """Merge context using bar_available_time <= primary signal time."""
    left = primary_features.sort_index().copy()
    right = context_features.sort_index().copy()
    merged = pd.merge_asof(
        left,
        right,
        left_index=True,
        right_index=True,
        direction="backward",
    )
    return merged


# -----------------------------------------------------------------------------
# Spec generation
# -----------------------------------------------------------------------------


def _make_spec(family: str, args: argparse.Namespace, **kwargs: Any) -> StrategySpec:
    payload = {
        "family": family,
        "primary_timeframe": args.primary_timeframe,
        "context_timeframe": args.context_timeframe,
        **kwargs,
    }
    return StrategySpec(
        spec_id=_spec_id(payload),
        family=family,
        primary_timeframe=args.primary_timeframe,
        context_timeframe=args.context_timeframe,
        **kwargs,
    )


def generate_specs(args: argparse.Namespace) -> list[StrategySpec]:
    """Generate explainable candidate specs without exact banned baseline forms."""
    specs: list[StrategySpec] = []
    families = set(args.families.split(",")) if args.families else {
        "trend_pullback_continuation",
        "vwap_reversion_regime",
        "breakout_retest_continuation",
        "squeeze_expansion_continuation",
    }
    side_modes = ["both", "long_only", "short_only"] if args.include_side_modes else ["both"]

    if "trend_pullback_continuation" in families:
        for side_mode, fast, slow, min_adx, vol_min, pullback, stop, tp, hold, cooldown in product(
            side_modes,
            [12, 20, 34],
            [50, 89, 144],
            [12.0, 16.0, 20.0],
            [0.0, 1.0, 1.25],
            [0.35, 0.65, 1.0],
            [1.2, 1.8, 2.4],
            [1.0, 1.5, 2.2],
            [24, 48, 96],
            [0, 3, 6],
        ):
            if fast >= slow:
                continue
            specs.append(_make_spec(
                "trend_pullback_continuation",
                args,
                side_mode=side_mode,
                fast_ema=fast,
                slow_ema=slow,
                atr_period=20,
                adx_period=14,
                min_adx=min_adx,
                max_adx=80.0,
                volume_ratio_min=vol_min,
                pullback_atr=pullback,
                zscore_window=144,
                zscore_entry=2.0,
                breakout_lookback=48,
                retest_atr=0.4,
                squeeze_window=144,
                squeeze_quantile=0.20,
                stop_atr_mult=stop,
                tp_r=tp,
                max_hold_bars=hold,
                cooldown_bars=cooldown,
                entry_model="trend_pullback_reclaim",
            ))

    if "vwap_reversion_regime" in families:
        for side_mode, zwin, zentry, vol_min, stop, tp, hold, cooldown, max_adx in product(
            side_modes,
            [72, 144, 288],
            [1.25, 1.75, 2.25, 2.75],
            [0.0, 0.8, 1.1],
            [0.9, 1.3, 1.8],
            [0.65, 0.9, 1.2, 1.6],
            [12, 24, 48],
            [0, 3, 6],
            [16.0, 20.0, 24.0],
        ):
            specs.append(_make_spec(
                "vwap_reversion_regime",
                args,
                side_mode=side_mode,
                fast_ema=20,
                slow_ema=89,
                atr_period=14,
                adx_period=14,
                min_adx=0.0,
                max_adx=max_adx,
                volume_ratio_min=vol_min,
                pullback_atr=0.0,
                zscore_window=zwin,
                zscore_entry=zentry,
                breakout_lookback=48,
                retest_atr=0.4,
                squeeze_window=144,
                squeeze_quantile=0.20,
                stop_atr_mult=stop,
                tp_r=tp,
                max_hold_bars=hold,
                cooldown_bars=cooldown,
                entry_model="vwap_extreme_reversal_in_chop",
            ))

    if "breakout_retest_continuation" in families:
        for side_mode, lb, retest_atr, min_adx, vol_min, stop, tp, hold, cooldown in product(
            side_modes,
            [24, 48, 72, 96],
            [0.2, 0.4, 0.7, 1.0],
            [12.0, 16.0, 20.0],
            [0.0, 1.0, 1.3],
            [1.2, 1.8, 2.5],
            [1.0, 1.5, 2.0, 2.8],
            [24, 48, 96],
            [0, 3, 6],
        ):
            specs.append(_make_spec(
                "breakout_retest_continuation",
                args,
                side_mode=side_mode,
                fast_ema=20,
                slow_ema=89,
                atr_period=20,
                adx_period=14,
                min_adx=min_adx,
                max_adx=80.0,
                volume_ratio_min=vol_min,
                pullback_atr=0.0,
                zscore_window=144,
                zscore_entry=2.0,
                breakout_lookback=lb,
                retest_atr=retest_atr,
                squeeze_window=144,
                squeeze_quantile=0.20,
                stop_atr_mult=stop,
                tp_r=tp,
                max_hold_bars=hold,
                cooldown_bars=cooldown,
                entry_model="breakout_retest_not_raw_breakout",
            ))

    if "squeeze_expansion_continuation" in families:
        for side_mode, swin, min_adx, vol_min, stop, tp, hold, cooldown in product(
            side_modes,
            [72, 96, 144, 288],
            [10.0, 14.0, 18.0],
            [1.0, 1.3, 1.6],
            [1.1, 1.6, 2.2],
            [1.0, 1.5, 2.2],
            [18, 36, 72],
            [0, 3, 6],
        ):
            specs.append(_make_spec(
                "squeeze_expansion_continuation",
                args,
                side_mode=side_mode,
                fast_ema=20,
                slow_ema=89,
                atr_period=14,
                adx_period=14,
                min_adx=min_adx,
                max_adx=80.0,
                volume_ratio_min=vol_min,
                pullback_atr=0.0,
                zscore_window=144,
                zscore_entry=2.0,
                breakout_lookback=48,
                retest_atr=0.4,
                squeeze_window=swin,
                squeeze_quantile=0.20,
                stop_atr_mult=stop,
                tp_r=tp,
                max_hold_bars=hold,
                cooldown_bars=cooldown,
                entry_model="squeeze_then_directional_expansion",
            ))

    # Stable order for reproducibility.
    specs = sorted({s.spec_id: s for s in specs}.values(), key=lambda s: (s.family, s.spec_id))
    if args.fast:
        specs = specs[: min(len(specs), 300)]
    if args.max_specs is not None:
        specs = specs[: max(0, int(args.max_specs))]
    return specs


# -----------------------------------------------------------------------------
# Signal generation
# -----------------------------------------------------------------------------


def build_signal(features: pd.DataFrame, spec: StrategySpec, ctx_prefix: str) -> pd.Series:
    idx = features.index
    close = _num(features, "close", np.nan)
    high = _num(features, "high", np.nan)
    low = _num(features, "low", np.nan)
    open_ = _num(features, "open", np.nan)
    atr_col = f"atr{spec.atr_period}"
    atr_v = _num(features, atr_col, np.nan)
    volume_ratio = _num(features, "volume_ratio", 0.0)
    fast = _num(features, f"ema{spec.fast_ema}", np.nan)
    slow = _num(features, f"ema{spec.slow_ema}", np.nan)
    htf_adx = _num(features, f"{ctx_prefix}_adx14", 0.0)
    htf_up = features.get(f"{ctx_prefix}_trend_up", _bool_series(idx)).astype("boolean").fillna(False).astype(bool)
    htf_down = features.get(f"{ctx_prefix}_trend_down", _bool_series(idx)).astype("boolean").fillna(False).astype(bool)
    htf_choppy = features.get(f"{ctx_prefix}_choppy", _bool_series(idx)).astype("boolean").fillna(False).astype(bool)

    vol_ok = volume_ratio >= spec.volume_ratio_min
    adx_ok = htf_adx.between(spec.min_adx, spec.max_adx)
    sig = pd.Series(0, index=idx, dtype="int8")

    allow_long = spec.side_mode in {"both", "long_only"}
    allow_short = spec.side_mode in {"both", "short_only"}

    if spec.family == "trend_pullback_continuation":
        # Signal bar is closed. Entry is next bar open. Pullback/reclaim uses only
        # the completed signal bar and past EMAs/ATR.
        long_setup = (
            htf_up & adx_ok & vol_ok
            & (fast > slow)
            & (low <= fast - spec.pullback_atr * atr_v)
            & (close > fast)
            & (close > open_)
            & (features["close_pos"] >= 0.55)
        )
        short_setup = (
            htf_down & adx_ok & vol_ok
            & (fast < slow)
            & (high >= fast + spec.pullback_atr * atr_v)
            & (close < fast)
            & (close < open_)
            & (features["close_pos"] <= 0.45)
        )
    elif spec.family == "vwap_reversion_regime":
        zcol = f"vwap_z{spec.zscore_window}"
        z = _num(features, zcol, np.nan)
        long_setup = htf_choppy & adx_ok & vol_ok & (z <= -spec.zscore_entry) & (features["close_pos"] >= 0.55)
        short_setup = htf_choppy & adx_ok & vol_ok & (z >= spec.zscore_entry) & (features["close_pos"] <= 0.45)
    elif spec.family == "breakout_retest_continuation":
        ph = _num(features, f"prior_high_{spec.breakout_lookback}", np.nan)
        pl = _num(features, f"prior_low_{spec.breakout_lookback}", np.nan)
        recent_bh = features.get(f"recent_break_high_{spec.breakout_lookback}", _bool_series(idx)).astype(bool)
        recent_bl = features.get(f"recent_break_low_{spec.breakout_lookback}", _bool_series(idx)).astype(bool)
        long_setup = (
            htf_up & adx_ok & vol_ok & recent_bh
            & (low <= ph + spec.retest_atr * atr_v)
            & (close >= ph)
            & (features["close_pos"] >= 0.55)
        )
        short_setup = (
            htf_down & adx_ok & vol_ok & recent_bl
            & (high >= pl - spec.retest_atr * atr_v)
            & (close <= pl)
            & (features["close_pos"] <= 0.45)
        )
    elif spec.family == "squeeze_expansion_continuation":
        recent_sq = features.get(f"recent_squeeze_{spec.squeeze_window}", _bool_series(idx)).astype(bool)
        long_setup = (
            htf_up & adx_ok & vol_ok & recent_sq
            & (close > fast) & (fast > slow)
            & (close > high.shift(1))
            & (features["body_pct"] >= 0.45)
            & (features["close_pos"] >= 0.65)
        )
        short_setup = (
            htf_down & adx_ok & vol_ok & recent_sq
            & (close < fast) & (fast < slow)
            & (close < low.shift(1))
            & (features["body_pct"] >= 0.45)
            & (features["close_pos"] <= 0.35)
        )
    else:
        long_setup = _bool_series(idx)
        short_setup = _bool_series(idx)

    if allow_long:
        sig.loc[long_setup.fillna(False)] = 1
    if allow_short:
        sig.loc[short_setup.fillna(False)] = -1
    return sig


# -----------------------------------------------------------------------------
# Backtest / audit
# -----------------------------------------------------------------------------


def _calc_drawdown(equity: pd.DataFrame) -> float:
    if equity.empty or "capital" not in equity:
        return 0.0
    cap = pd.to_numeric(equity["capital"], errors="coerce").dropna()
    if cap.empty:
        return 0.0
    dd = cap / cap.cummax() - 1.0
    return float(dd.min())


def _max_consecutive_losses(pnl: pd.Series) -> int:
    max_run = 0
    run = 0
    for is_loss in (pnl <= 0).fillna(False).tolist():
        if is_loss:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return int(max_run)


def _max_days_without_trade(exit_times: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> float:
    if exit_times.empty:
        return float((end - start).total_seconds() / 86400.0)
    ts = pd.to_datetime(exit_times, errors="coerce").dropna().sort_values()
    points = [start] + list(ts) + [end]
    gaps = [(points[i + 1] - points[i]).total_seconds() / 86400.0 for i in range(len(points) - 1)]
    return float(max(gaps)) if gaps else 0.0


def run_one_spec(features: pd.DataFrame, spec: StrategySpec, cfg: BacktestConfig, *, ctx_prefix: str, trade_start: pd.Timestamp, trade_end: pd.Timestamp) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Run one standalone spec with a lightweight numpy event loop.

    Speed notes:
        - Features are prepared once outside this function.
        - This function does not copy the full feature frame per spec.
        - Equity rows are recorded only when capital changes, not once per bar.
        - MFE/MAE is still computed bar-by-bar while a trade is open.
        - delay_bars uses a pending entry index and does not evaluate bars before
          the delayed fill actually exists.
    """
    signal = build_signal(features, spec, ctx_prefix)
    mask = (features.index >= trade_start) & (features.index <= trade_end)
    f = features.loc[mask]
    if f.empty:
        return [], pd.DataFrame(), pd.DataFrame()

    signal_arr = signal.loc[f.index].fillna(0).astype("int8").to_numpy(copy=False)
    idx = f.index
    open_arr = pd.to_numeric(f["open"], errors="coerce").to_numpy(dtype="float64", copy=False)
    high_arr = pd.to_numeric(f["high"], errors="coerce").to_numpy(dtype="float64", copy=False)
    low_arr = pd.to_numeric(f["low"], errors="coerce").to_numpy(dtype="float64", copy=False)
    close_arr = pd.to_numeric(f["close"], errors="coerce").to_numpy(dtype="float64", copy=False)

    atr_col = f"atr{spec.atr_period}"
    if atr_col not in f.columns:
        return [], pd.DataFrame(), pd.DataFrame()
    atr_arr = pd.to_numeric(f[atr_col], errors="coerce").to_numpy(dtype="float64", copy=False)

    ctx_start_col = f"{ctx_prefix}_bar_start_time"
    ctx_avail_col = f"{ctx_prefix}_available_time"
    ctx_start_arr = f[ctx_start_col].to_numpy(copy=False) if ctx_start_col in f.columns else np.array([pd.NaT] * len(f), dtype=object)
    ctx_avail_arr = f[ctx_avail_col].to_numpy(copy=False) if ctx_avail_col in f.columns else np.array([pd.NaT] * len(f), dtype=object)

    capital = float(cfg.initial_capital)
    peak = capital
    trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = [{"timestamp": idx[0], "capital": capital, "drawdown_pct": 0.0, "spec_id": spec.spec_id}]

    in_pos = False
    side = 0
    entry_i = -1
    entry_time: pd.Timestamp | None = None
    entry_price = 0.0
    raw_entry_open = 0.0
    stop_price = 0.0
    tp_price = 0.0
    risk_per_coin = 0.0
    qty = 0.0
    entry_fee = 0.0
    max_fav = 0.0
    max_adv = 0.0
    last_exit_i = -10**9
    signal_i = -1
    signal_time: pd.Timestamp | None = None
    used_ctx_start = pd.NaT
    used_ctx_available = pd.NaT
    entry_not_next_open_flag = False
    entry_price_mismatch_flag = False

    n_rows = len(f)
    for i in range(n_rows - 1):
        ts = idx[i]

        if in_pos:
            # For delay stress, the position is pending until entry_i. Do not
            # evaluate unrealized MFE/MAE or stop/TP before the delayed fill bar.
            if i < entry_i:
                continue

            high = high_arr[i]
            low = low_arr[i]
            close = close_arr[i]
            if not (math.isfinite(high) and math.isfinite(low) and math.isfinite(close)):
                continue

            if side == 1:
                max_fav = max(max_fav, high)
                max_adv = min(max_adv, low)
                max_fav_price = max_fav
                max_adv_price = max_adv
                stop_hit = low <= stop_price
                tp_hit = high >= tp_price
            else:
                max_fav = max(max_fav, -low)
                max_adv = min(max_adv, -high)
                max_fav_price = -max_fav
                max_adv_price = -max_adv
                stop_hit = high >= stop_price
                tp_hit = low <= tp_price

            same_bar_both = bool(stop_hit and tp_hit)
            exit_reason = ""
            exit_price_raw = 0.0
            if same_bar_both:
                exit_reason = "SL_SAME_BAR_BOTH_CONSERVATIVE"
                exit_price_raw = stop_price
            elif stop_hit:
                exit_reason = "SL"
                exit_price_raw = stop_price
            elif tp_hit:
                exit_reason = "TP"
                exit_price_raw = tp_price
            elif i - entry_i >= spec.max_hold_bars:
                exit_reason = "TIME_STOP"
                exit_price_raw = close

            if exit_reason:
                exit_price = apply_exit_slippage(exit_price_raw, side, cfg.slippage_pct)
                exit_fee = abs(qty * exit_price) * cfg.fee_rate
                gross_pnl = qty * (exit_price - entry_price) * side
                pnl = gross_pnl - entry_fee - exit_fee
                prev_capital = capital
                capital += pnl
                peak = max(peak, capital)
                ret_pct = pnl / max(prev_capital, 1e-12)
                mfe_r = ((max_fav_price - entry_price) * side) / risk_per_coin if risk_per_coin > 0 else float("nan")
                mae_r = ((max_adv_price - entry_price) * side) / risk_per_coin if risk_per_coin > 0 else float("nan")
                trade = {
                    "spec_id": spec.spec_id,
                    "family": spec.family,
                    "system": STRATEGY_NAME,
                    "entry_model": spec.entry_model,
                    "exit_model": spec.exit_model,
                    "signal_frame": spec.primary_timeframe,
                    "signal_time": signal_time,
                    "entry_time": entry_time,
                    "exit_time": ts,
                    "type": "LONG" if side == 1 else "SHORT",
                    "side": side,
                    "entry_price": entry_price,
                    "raw_entry_open": raw_entry_open,
                    "exit_price": exit_price,
                    "exit_price_raw": exit_price_raw,
                    "exit_reason": exit_reason,
                    "stop_price": stop_price,
                    "tp_price": tp_price,
                    "risk_per_coin": risk_per_coin,
                    "qty": qty,
                    "gross_pnl": gross_pnl,
                    "fee": entry_fee + exit_fee,
                    "pnl": pnl,
                    "capital": capital,
                    "return_pct": ret_pct,
                    "mfe_r": mfe_r,
                    "mae_r": mae_r,
                    "mfe_pct": ((max_fav_price - entry_price) * side) / entry_price if entry_price else float("nan"),
                    "mae_pct": ((max_adv_price - entry_price) * side) / entry_price if entry_price else float("nan"),
                    "holding_bars": i - entry_i,
                    "holding_hours": (ts - entry_time).total_seconds() / 3600.0 if entry_time is not None else float("nan"),
                    "expected_entry_time": idx[signal_i + 1] if signal_i + 1 < n_rows else pd.NaT,
                    "expected_entry_price": raw_entry_open,
                    "used_context_timestamp": used_ctx_start,
                    "used_context_available_time": used_ctx_available,
                    "context_available_time_flag": bool(pd.notna(used_ctx_available) and signal_time is not None and used_ctx_available > signal_time),
                    "entry_not_next_open_flag": entry_not_next_open_flag,
                    "entry_price_mismatch_flag": entry_price_mismatch_flag,
                    "same_bar_exit_flag": bool(i == entry_i),
                    "same_bar_stop_tp_both_hit_flag": same_bar_both,
                    "net_return": ret_pct,
                }
                trades.append(trade)
                equity_rows.append({"timestamp": ts, "capital": capital, "drawdown_pct": capital / peak - 1.0, "spec_id": spec.spec_id})
                in_pos = False
                side = 0
                last_exit_i = i
            continue

        sig = int(signal_arr[i])
        if sig == 0:
            continue
        if i <= last_exit_i + spec.cooldown_bars:
            continue

        entry_idx = i + 1 + int(cfg.delay_bars)
        if entry_idx >= n_rows:
            break
        raw_open = open_arr[entry_idx]
        atr_value = atr_arr[i]
        if not math.isfinite(raw_open) or raw_open <= 0 or not math.isfinite(atr_value) or atr_value <= 0:
            continue

        side = sig
        initial_stop_dist = max(float(spec.stop_atr_mult) * atr_value, float(cfg.min_stop_pct) * raw_open)
        fill = apply_entry_slippage(raw_open, side, cfg.slippage_pct)
        risk_per_coin = initial_stop_dist
        if side == 1:
            stop_price = fill - risk_per_coin
            tp_price = fill + risk_per_coin * spec.tp_r
            max_fav = fill
            max_adv = fill
        else:
            stop_price = fill + risk_per_coin
            tp_price = fill - risk_per_coin * spec.tp_r
            max_fav = -fill
            max_adv = -fill
        qty = (capital * cfg.notional_mult) / fill
        entry_fee = abs(qty * fill) * cfg.fee_rate
        in_pos = True
        entry_i = entry_idx
        entry_time = idx[entry_idx]
        entry_price = fill
        raw_entry_open = raw_open
        signal_i = i
        signal_time = ts
        used_ctx_start = ctx_start_arr[i] if i < len(ctx_start_arr) else pd.NaT
        used_ctx_available = ctx_avail_arr[i] if i < len(ctx_avail_arr) else pd.NaT
        expected_time = idx[i + 1] if i + 1 < n_rows else pd.NaT
        entry_not_next_open_flag = bool(cfg.delay_bars == 0 and entry_time != expected_time)
        expected_open = open_arr[i + 1] if i + 1 < n_rows else float("nan")
        entry_price_mismatch_flag = bool(cfg.delay_bars == 0 and math.isfinite(expected_open) and abs(raw_open - expected_open) > max(1e-9, expected_open * 1e-10))

    # Force-close at the last close only for accounting. Mark as EOD so it is not
    # confused with strategy edge. MFE/MAE is still informative.
    if in_pos and n_rows > 0:
        ts = idx[-1]
        exit_price_raw = close_arr[-1]
        if math.isfinite(exit_price_raw) and exit_price_raw > 0:
            exit_price = apply_exit_slippage(exit_price_raw, side, cfg.slippage_pct)
            exit_fee = abs(qty * exit_price) * cfg.fee_rate
            gross_pnl = qty * (exit_price - entry_price) * side
            pnl = gross_pnl - entry_fee - exit_fee
            prev_capital = capital
            capital += pnl
            peak = max(peak, capital)
            max_fav_price = max_fav if side == 1 else -max_fav
            max_adv_price = max_adv if side == 1 else -max_adv
            trades.append({
                "spec_id": spec.spec_id,
                "family": spec.family,
                "system": STRATEGY_NAME,
                "entry_model": spec.entry_model,
                "exit_model": spec.exit_model,
                "signal_frame": spec.primary_timeframe,
                "signal_time": signal_time,
                "entry_time": entry_time,
                "exit_time": ts,
                "type": "LONG" if side == 1 else "SHORT",
                "side": side,
                "entry_price": entry_price,
                "raw_entry_open": raw_entry_open,
                "exit_price": exit_price,
                "exit_price_raw": exit_price_raw,
                "exit_reason": "EOD_FORCE_CLOSE",
                "stop_price": stop_price,
                "tp_price": tp_price,
                "risk_per_coin": risk_per_coin,
                "qty": qty,
                "gross_pnl": gross_pnl,
                "fee": entry_fee + exit_fee,
                "pnl": pnl,
                "capital": capital,
                "return_pct": pnl / max(prev_capital, 1e-12),
                "mfe_r": ((max_fav_price - entry_price) * side) / risk_per_coin if risk_per_coin > 0 else float("nan"),
                "mae_r": ((max_adv_price - entry_price) * side) / risk_per_coin if risk_per_coin > 0 else float("nan"),
                "mfe_pct": ((max_fav_price - entry_price) * side) / entry_price if entry_price else float("nan"),
                "mae_pct": ((max_adv_price - entry_price) * side) / entry_price if entry_price else float("nan"),
                "holding_bars": n_rows - 1 - entry_i,
                "holding_hours": (ts - entry_time).total_seconds() / 3600.0 if entry_time is not None else float("nan"),
                "expected_entry_time": idx[signal_i + 1] if signal_i + 1 < n_rows else pd.NaT,
                "expected_entry_price": raw_entry_open,
                "used_context_timestamp": used_ctx_start,
                "used_context_available_time": used_ctx_available,
                "context_available_time_flag": bool(pd.notna(used_ctx_available) and signal_time is not None and used_ctx_available > signal_time),
                "entry_not_next_open_flag": entry_not_next_open_flag,
                "entry_price_mismatch_flag": entry_price_mismatch_flag,
                "same_bar_exit_flag": False,
                "same_bar_stop_tp_both_hit_flag": False,
                "net_return": pnl / max(prev_capital, 1e-12),
            })
            equity_rows.append({"timestamp": ts, "capital": capital, "drawdown_pct": capital / peak - 1.0, "spec_id": spec.spec_id})

    equity = pd.DataFrame(equity_rows)
    if not equity.empty:
        equity = equity.drop_duplicates(subset=["timestamp"], keep="last").set_index("timestamp").sort_index()
    audit = pd.DataFrame(trades)
    return trades, equity, audit

# -----------------------------------------------------------------------------
# Metrics / scoring / stress
# -----------------------------------------------------------------------------


def summarize_spec(spec: StrategySpec, trades: list[dict[str, Any]], equity: pd.DataFrame, initial_capital: float, trade_start: pd.Timestamp, trade_end: pd.Timestamp, *, scenario: str = "base", extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scenario": scenario,
        "spec_id": spec.spec_id,
        "family": spec.family,
        "entry_model": spec.entry_model,
        "side_mode": spec.side_mode,
        "primary_timeframe": spec.primary_timeframe,
        "context_timeframe": spec.context_timeframe,
        "total_trades": 0,
        "final_capital": initial_capital,
        "total_return_pct": 0.0,
    }
    if extra:
        row.update(extra)
    if not trades:
        row.update({
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "max_consecutive_losses": 0,
            "max_days_without_trade": float((trade_end - trade_start).total_seconds() / 86400.0),
            "active_days": 0,
            "avg_mfe_r": 0.0,
            "avg_mae_r": 0.0,
            "median_mfe_r": 0.0,
            "median_mae_r": 0.0,
            "avg_holding_hours": 0.0,
            "same_bar_both_hit_count": 0,
            "causal_flag_count": 0,
        })
        return row

    t = pd.DataFrame(trades)
    pnl = pd.to_numeric(t["pnl"], errors="coerce").fillna(0.0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    gp = float(wins.sum())
    gl = float(-losses.sum())
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
    final_capital = float(t.iloc[-1]["capital"])
    exit_times = pd.to_datetime(t["exit_time"], errors="coerce")
    row.update({
        "total_trades": int(len(t)),
        "long_trades": int((t["side"] == 1).sum()),
        "short_trades": int((t["side"] == -1).sum()),
        "final_capital": final_capital,
        "total_return_pct": (final_capital / initial_capital - 1.0) * 100.0,
        "win_rate": float((pnl > 0).mean() * 100.0),
        "gross_profit": gp,
        "gross_loss": gl,
        "profit_factor": pf,
        "expectancy_pct": float(pd.to_numeric(t["return_pct"], errors="coerce").mean() * 100.0),
        "max_drawdown_pct": _calc_drawdown(equity) * 100.0,
        "max_consecutive_losses": _max_consecutive_losses(pnl),
        "max_days_without_trade": _max_days_without_trade(exit_times, trade_start, trade_end),
        "active_days": int(exit_times.dt.floor("D").nunique()),
        "avg_mfe_r": float(pd.to_numeric(t["mfe_r"], errors="coerce").mean()),
        "avg_mae_r": float(pd.to_numeric(t["mae_r"], errors="coerce").mean()),
        "median_mfe_r": float(pd.to_numeric(t["mfe_r"], errors="coerce").median()),
        "median_mae_r": float(pd.to_numeric(t["mae_r"], errors="coerce").median()),
        "p75_mfe_r": float(pd.to_numeric(t["mfe_r"], errors="coerce").quantile(0.75)),
        "p25_mae_r": float(pd.to_numeric(t["mae_r"], errors="coerce").quantile(0.25)),
        "avg_holding_hours": float(pd.to_numeric(t["holding_hours"], errors="coerce").mean()),
        "total_fees": float(pd.to_numeric(t["fee"], errors="coerce").sum()),
        "same_bar_both_hit_count": int(t.get("same_bar_stop_tp_both_hit_flag", pd.Series(False, index=t.index)).astype(bool).sum()),
        "causal_flag_count": int(t.get("context_available_time_flag", pd.Series(False, index=t.index)).astype(bool).sum()),
        "entry_not_next_open_count": int(t.get("entry_not_next_open_flag", pd.Series(False, index=t.index)).astype(bool).sum()),
        "entry_price_mismatch_count": int(t.get("entry_price_mismatch_flag", pd.Series(False, index=t.index)).astype(bool).sum()),
    })
    return row


def yearly_metrics(trades: list[dict[str, Any]], initial_capital: float, spec: StrategySpec, scenario: str = "base") -> list[dict[str, Any]]:
    if not trades:
        return []
    t = pd.DataFrame(trades).copy()
    t["exit_time"] = pd.to_datetime(t["exit_time"], errors="coerce")
    t = t.dropna(subset=["exit_time"])
    out: list[dict[str, Any]] = []
    for year, g in t.groupby(t["exit_time"].dt.year):
        pnl = pd.to_numeric(g["pnl"], errors="coerce").fillna(0.0)
        gp = float(pnl[pnl > 0].sum())
        gl = float(-pnl[pnl <= 0].sum())
        out.append({
            "scenario": scenario,
            "spec_id": spec.spec_id,
            "family": spec.family,
            "year": int(year),
            "trades": int(len(g)),
            "pnl": float(pnl.sum()),
            "return_on_initial_pct": float(pnl.sum() / initial_capital * 100.0),
            "win_rate": float((pnl > 0).mean() * 100.0),
            "profit_factor": gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0),
            "avg_mfe_r": float(pd.to_numeric(g["mfe_r"], errors="coerce").mean()),
            "avg_mae_r": float(pd.to_numeric(g["mae_r"], errors="coerce").mean()),
        })
    return out


def build_scoreboard(summary: pd.DataFrame, yearly: pd.DataFrame, stress: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    base = summary[summary["scenario"].eq("base")].copy()
    if base.empty:
        return pd.DataFrame()
    out = base.copy()
    for col in ["profit_factor", "win_rate", "total_return_pct", "max_drawdown_pct", "total_trades", "max_consecutive_losses", "max_days_without_trade", "active_days", "avg_mfe_r", "avg_mae_r"]:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    y = yearly[yearly["scenario"].eq("base")].copy() if not yearly.empty else pd.DataFrame()
    if not y.empty:
        y["positive_year"] = pd.to_numeric(y["pnl"], errors="coerce").fillna(0.0) > 0
        yagg = y.groupby("spec_id").agg(
            years=("year", "nunique"),
            positive_years=("positive_year", "sum"),
            worst_year_return_pct=("return_on_initial_pct", "min"),
        ).reset_index()
        out = out.merge(yagg, on="spec_id", how="left")
    else:
        out["years"] = 0
        out["positive_years"] = 0
        out["worst_year_return_pct"] = 0.0

    if not stress.empty:
        st = stress[~stress["scenario"].eq("base")].copy()
        st["stress_pf_ok"] = pd.to_numeric(st["profit_factor"], errors="coerce").fillna(0.0) >= 1.1
        st["stress_return_ok"] = pd.to_numeric(st["total_return_pct"], errors="coerce").fillna(-999.0) > -5.0
        sagg = st.groupby("spec_id").agg(
            stress_cases=("scenario", "nunique"),
            stress_pf_ok_cases=("stress_pf_ok", "sum"),
            stress_return_ok_cases=("stress_return_ok", "sum"),
            worst_stress_pf=("profit_factor", "min"),
            worst_stress_return_pct=("total_return_pct", "min"),
        ).reset_index()
        out = out.merge(sagg, on="spec_id", how="left")
    else:
        out["stress_cases"] = 0
        out["stress_pf_ok_cases"] = 0
        out["stress_return_ok_cases"] = 0
        out["worst_stress_pf"] = 0.0
        out["worst_stress_return_pct"] = 0.0

    # Follower-friendly score: product curve quality, not just raw return.
    pf_score = np.clip((out["profit_factor"] - 1.0) / 1.0, 0, 2.0) / 2.0
    win_score = np.clip((out["win_rate"] - 35.0) / 25.0, 0, 1.0)
    trade_score = np.clip(out["total_trades"] / 500.0, 0, 1.0)
    dd_score = np.clip(1.0 - (out["max_drawdown_pct"].abs() / 25.0), 0, 1.0)
    loss_score = np.clip(1.0 - (out["max_consecutive_losses"] / 12.0), 0, 1.0)
    gap_score = np.clip(1.0 - (out["max_days_without_trade"] / 10.0), 0, 1.0)
    mfe_quality = np.clip((out["avg_mfe_r"] - out["avg_mae_r"].abs()) / 2.0, 0, 1.0)
    yearly_score = np.where(out["years"] > 0, out["positive_years"] / out["years"].replace(0, np.nan), 0.0)
    stress_score = np.where(out["stress_cases"] > 0, out["stress_pf_ok_cases"] / out["stress_cases"].replace(0, np.nan), 0.0)
    out["follower_score"] = (
        0.18 * pf_score
        + 0.14 * win_score
        + 0.14 * trade_score
        + 0.14 * dd_score
        + 0.10 * loss_score
        + 0.10 * gap_score
        + 0.10 * mfe_quality
        + 0.10 * yearly_score
        + 0.10 * stress_score
    ) * 100.0

    out["candidate_state"] = "REJECT"
    pass_mask = (
        (out["total_trades"] >= 100)
        & (out["profit_factor"] >= 1.25)
        & (out["win_rate"] >= 38.0)
        & (out["max_drawdown_pct"].abs() <= 25.0)
        & (out["causal_flag_count"] == 0)
        & (out["entry_not_next_open_count"] == 0)
        & (out["entry_price_mismatch_count"] == 0)
        & (out["positive_years"] >= np.maximum(1, out["years"] - 1))
    )
    out.loc[pass_mask, "candidate_state"] = "WATCHLIST"
    strong_mask = pass_mask & (out["profit_factor"] >= 1.45) & (out["win_rate"] >= 42.0) & (out["stress_pf_ok_cases"] >= np.maximum(1, out["stress_cases"] - 1))
    out.loc[strong_mask, "candidate_state"] = "STRONG_CANDIDATE_NEEDS_SLOW_REPLAY"
    state_rank = {"STRONG_CANDIDATE_NEEDS_SLOW_REPLAY": 0, "WATCHLIST": 1, "REJECT": 2}
    out["candidate_state_rank"] = out["candidate_state"].map(state_rank).fillna(9).astype(int)
    return out.sort_values(["candidate_state_rank", "follower_score", "profit_factor", "total_return_pct"], ascending=[True, False, False, False]).reset_index(drop=True)


def run_stress_cases(features: pd.DataFrame, specs_by_id: dict[str, StrategySpec], base_scoreboard: pd.DataFrame, cfg: BacktestConfig, *, ctx_prefix: str, trade_start: pd.Timestamp, trade_end: pd.Timestamp, top_n: int, progress_enabled: bool = True, progress_every: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    if base_scoreboard.empty or top_n <= 0:
        return pd.DataFrame(), pd.DataFrame()
    top_ids = base_scoreboard.sort_values("follower_score", ascending=False)["spec_id"].head(top_n).tolist()
    rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    cases = [
        ("fee_2x", BacktestConfig(**{**asdict(cfg), "fee_rate": cfg.fee_rate * 2.0})),
        ("slippage_2x", BacktestConfig(**{**asdict(cfg), "slippage_pct": cfg.slippage_pct * 2.0})),
        ("delay_1bar", BacktestConfig(**{**asdict(cfg), "delay_bars": cfg.delay_bars + 1})),
        ("fee2_slip2_delay1", BacktestConfig(**{**asdict(cfg), "fee_rate": cfg.fee_rate * 2.0, "slippage_pct": cfg.slippage_pct * 2.0, "delay_bars": cfg.delay_bars + 1})),
    ]
    stress_started = time.perf_counter()
    stress_done = 0
    stress_total = len(cases) * len(top_ids)
    for scenario, scfg in cases:
        print(f"[stress] {scenario} top_n={len(top_ids)}", flush=True)
        for spec_id in top_ids:
            spec = specs_by_id[spec_id]
            trades, equity, _audit = run_one_spec(features, spec, scfg, ctx_prefix=ctx_prefix, trade_start=trade_start, trade_end=trade_end)
            rows.append(summarize_spec(spec, trades, equity, scfg.initial_capital, trade_start, trade_end, scenario=scenario, extra={"fee_rate": scfg.fee_rate, "slippage_pct": scfg.slippage_pct, "delay_bars": scfg.delay_bars}))
            yearly_rows.extend(yearly_metrics(trades, scfg.initial_capital, spec, scenario=scenario))
            stress_done += 1
            _progress_update("[stress]", stress_done, stress_total, stress_started, every=progress_every, enabled=progress_enabled, force=(stress_done == 1 or stress_done == stress_total))
    return pd.DataFrame(rows), pd.DataFrame(yearly_rows)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Follower-friendly standalone strategy factory V1")
    p.add_argument("--symbol", default=DEFAULT_SYMBOL)
    p.add_argument("--start-date", default=DEFAULT_START_DATE)
    p.add_argument("--end-date", default=DEFAULT_END_DATE)
    p.add_argument("--warmup-start-date", default=DEFAULT_WARMUP_START_DATE)
    p.add_argument("--data-source", default="ohlcv", choices=["ohlcv"], help="V1.2 uses normal OKX OHLCV only. trade_bar/range_bar/footprint are planned for V2.")
    p.add_argument("--primary-timeframe", default="5m", choices=["1m", "5m", "15m"])
    p.add_argument("--context-timeframe", default="1H", choices=["15m", "1H", "4H"])
    p.add_argument("--extra-context-timeframe", default=None, choices=[None, "1H", "4H"])
    p.add_argument("--initial-capital", type=float, default=1000.0)
    p.add_argument("--fee-rate", type=float, default=0.00055)
    p.add_argument("--slippage-pct", type=float, default=0.0002)
    p.add_argument("--notional-mult", type=float, default=1.0)
    p.add_argument("--min-stop-pct", type=float, default=0.0015)
    p.add_argument("--families", default=None, help="Comma-separated subset of strategy families")
    p.add_argument("--include-side-modes", action="store_true", help="Also generate long_only/short_only variants. Default only both.")
    p.add_argument("--max-specs", type=int, default=1000)
    p.add_argument("--fast", action="store_true", help="Limit to a small deterministic batch for quick sanity checks.")
    p.add_argument("--stress-top-n", type=int, default=50)
    p.add_argument("--write-top-trades", action="store_true")
    p.add_argument("--top-trades-n", type=int, default=30)
    p.add_argument("--progress-every", type=int, default=25, help="Update progress every N specs/cases. Default: 25.")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bar/log updates.")
    p.add_argument("--out-dir", default="data/reports/research/follower_friendly_strategy_factory_v1")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trade_start = pd.Timestamp(args.start_date)
    trade_end = pd.Timestamp(args.end_date)
    frames = load_market_frames(args)
    primary = build_primary_features(frames[args.primary_timeframe])
    ctx_prefix = "ctx"
    ctx = build_context_features(frames[args.context_timeframe], args.context_timeframe, ctx_prefix)
    features = causal_merge_context(primary, ctx)
    features = features.dropna(subset=["open", "high", "low", "close"]).copy()

    specs = generate_specs(args)
    if not specs:
        raise RuntimeError("No specs generated. Check --families/--max-specs.")
    print(f"[factory] generated_specs={len(specs):,}; banned_baselines={sorted(BANNED_BASELINES)}", flush=True)
    pd.DataFrame([asdict(s) for s in specs]).to_csv(out_dir / "01_spec_manifest.csv", index=False, encoding="utf-8-sig")

    cfg = BacktestConfig(
        symbol=args.symbol,
        initial_capital=args.initial_capital,
        fee_rate=args.fee_rate,
        slippage_pct=args.slippage_pct,
        notional_mult=args.notional_mult,
        min_stop_pct=args.min_stop_pct,
    )

    summary_rows: list[dict[str, Any]] = []
    yearly_rows: list[dict[str, Any]] = []
    audit_parts: list[pd.DataFrame] = []
    top_trade_parts: list[pd.DataFrame] = []
    specs_by_id = {s.spec_id: s for s in specs}

    progress_enabled = not bool(args.no_progress)
    factory_started = time.perf_counter()
    for n, spec in enumerate(specs, start=1):
        trades, equity, audit = run_one_spec(features, spec, cfg, ctx_prefix=ctx_prefix, trade_start=trade_start, trade_end=trade_end)
        summary_rows.append(summarize_spec(spec, trades, equity, cfg.initial_capital, trade_start, trade_end, scenario="base", extra={"fee_rate": cfg.fee_rate, "slippage_pct": cfg.slippage_pct, "delay_bars": cfg.delay_bars}))
        yearly_rows.extend(yearly_metrics(trades, cfg.initial_capital, spec, scenario="base"))
        if not audit.empty and len(audit_parts) < 200:
            # Write replay audit for early candidates and all candidates that pass
            # rough positive PF gates after summary is known below. Full all-trades
            # output can be huge, so this keeps default output manageable.
            audit_parts.append(audit.head(200))
        if args.write_top_trades and not audit.empty:
            top_trade_parts.append(audit)
        _progress_update("[factory]", n, len(specs), factory_started, every=args.progress_every, enabled=progress_enabled, force=(n == 1 or n == len(specs)))

    summary_df = pd.DataFrame(summary_rows)
    yearly_df = pd.DataFrame(yearly_rows)
    summary_df.to_csv(out_dir / "02_base_summary.csv", index=False, encoding="utf-8-sig")
    yearly_df.to_csv(out_dir / "03_base_yearly.csv", index=False, encoding="utf-8-sig")

    rough_score = build_scoreboard(summary_df, yearly_df, pd.DataFrame())
    stress_df, stress_yearly_df = run_stress_cases(features, specs_by_id, rough_score, cfg, ctx_prefix=ctx_prefix, trade_start=trade_start, trade_end=trade_end, top_n=args.stress_top_n, progress_enabled=progress_enabled, progress_every=max(1, min(args.progress_every, 10)))
    stress_df.to_csv(out_dir / "04_stress_summary.csv", index=False, encoding="utf-8-sig")
    stress_yearly_df.to_csv(out_dir / "05_stress_yearly.csv", index=False, encoding="utf-8-sig")

    scoreboard = build_scoreboard(summary_df, yearly_df, stress_df)
    scoreboard.to_csv(out_dir / "06_scoreboard.csv", index=False, encoding="utf-8-sig")

    if audit_parts:
        pd.concat(audit_parts, ignore_index=True).to_csv(out_dir / "07_replay_audit_sample.csv", index=False, encoding="utf-8-sig")

    if args.write_top_trades and top_trade_parts:
        all_trades = pd.concat(top_trade_parts, ignore_index=True)
        keep_ids = scoreboard.head(args.top_trades_n)["spec_id"].tolist() if not scoreboard.empty else []
        all_trades[all_trades["spec_id"].isin(keep_ids)].to_csv(out_dir / "08_top_candidate_trades_with_mfe_mae.csv", index=False, encoding="utf-8-sig")

    meta = {
        "strategy_name": STRATEGY_NAME,
        "created_at_utc": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "args": vars(args),
        "config": asdict(cfg),
        "banned_baselines": sorted(BANNED_BASELINES),
        "causal_alignment": "context index = bar_start_time + timeframe_delta; merge_asof backward; signal close -> next open execution",
        "outputs": [
            "01_spec_manifest.csv",
            "02_base_summary.csv",
            "03_base_yearly.csv",
            "04_stress_summary.csv",
            "05_stress_yearly.csv",
            "06_scoreboard.csv",
            "07_replay_audit_sample.csv",
            "08_top_candidate_trades_with_mfe_mae.csv (optional --write-top-trades)",
        ],
    }
    (out_dir / "00_factory_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("\n" + "=" * 100)
    print("Follower-Friendly Strategy Factory V1 complete")
    print("=" * 100)
    print(f"Output directory: {out_dir.resolve()}")
    if not scoreboard.empty:
        cols = [
            "candidate_state", "spec_id", "family", "total_trades", "win_rate", "profit_factor",
            "total_return_pct", "max_drawdown_pct", "max_consecutive_losses", "max_days_without_trade",
            "avg_mfe_r", "avg_mae_r", "follower_score",
        ]
        print(scoreboard[[c for c in cols if c in scoreboard.columns]].head(30).to_string(index=False))
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
