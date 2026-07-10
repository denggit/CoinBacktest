#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH MHF/MF EMA50 Snapback Rejoin V1 event-study replay.

Research-only script. It tests a narrowly pre-declared hypothesis:

    In a one-sided 1m trend, price briefly pierces the wrong side of EMA50,
    quickly reclaims the original trend side, then the trade follows the
    original trend.

This is deliberately not a naked EMA cross strategy. It requires a prior trend,
a fast wrong-side pierce, and a reclaim/rejoin bar. Exits are pre-declared and
causal: fixed time, close back across EMA50, EMA50 slope flip, or first of these.

Timing policy:
    signal_time = current primary closed 1m trade bar
    entry_time  = next primary bar open plus optional delay bars
    entry_price = open[entry_idx]

Dynamic exit policy:
    exit condition is observed on a closed bar; execution is next bar open.
    Fixed-time exit also executes at the open after max_hold bars.

This script does not register a tradable edge, modify portfolio code, or import
from other research scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_NAME = "eth_mhf_ema50_snapback_rejoin_v1.py"
SCRIPT_VERSION = "1.0.1"
EXPERIMENT_ID = "ETH_MHF_EMA50_SNAPBACK_REJOIN_V1"
EDGE_ID = "ETH_EDGE_MHF_EMA50_SNAPBACK_REJOIN_RESEARCH_V1"
DEFAULT_OUT_DIR = "data/reports/research/mhf_ema50_snapback_rejoin_v1"
CAUSAL_POLICY = (
    "closed primary trade bar signal; entry at next primary open plus delay; "
    "dynamic exits are detected on closed bars and executed at next primary open"
)
MATCHED_BASELINE_COLUMNS = ("year", "month", "session", "regime", "volatility_bucket", "trend_side")


@dataclass(frozen=True)
class EventSpec:
    direction: str
    trend_window: int
    reclaim_bars: int
    pierce_pct: float
    pierce_mode: str
    trend_above_ratio: float
    min_slope_pct: float

    @property
    def variant(self) -> str:
        p_bp = int(round(self.pierce_pct * 10000))
        return f"tw{self.trend_window}_rb{self.reclaim_bars}_{self.pierce_mode}_p{p_bp}bp"

    @property
    def event_name(self) -> str:
        return f"ema50_trend_snapback_rejoin__{self.variant}__{self.direction}"


@dataclass(frozen=True)
class ReplaySpec:
    replay_id: str
    exit_model: str
    max_hold_bars: int
    description: str


@dataclass(frozen=True)
class ReplayResult:
    event_pos: int
    signal_idx: int
    signal_time: pd.Timestamp
    entry_idx: int
    entry_time: pd.Timestamp
    entry_price: float
    exit_signal_idx: int
    exit_signal_time: pd.Timestamp
    exit_idx: int
    exit_time: pd.Timestamp
    exit_price: float
    exit_reason: str
    hold_bars: int
    gross_return: float
    mfe: float
    mae: float
    valid: bool
    invalid_reason: str


# Fixed, pre-declared replay exits. This keeps V1 away from TP/SL optimization.
DEFAULT_REPLAY_SPECS: tuple[ReplaySpec, ...] = (
    ReplaySpec("time_5m", "time", 5, "Fixed 5-bar hold, exit next open after bar 5."),
    ReplaySpec("time_10m", "time", 10, "Fixed 10-bar hold, exit next open after bar 10."),
    ReplaySpec("time_15m", "time", 15, "Fixed 15-bar hold, exit next open after bar 15."),
    ReplaySpec("time_30m", "time", 30, "Fixed 30-bar hold, exit next open after bar 30."),
    ReplaySpec("time_45m", "time", 45, "Fixed 45-bar hold, exit next open after bar 45."),
    ReplaySpec("time_60m", "time", 60, "Fixed 60-bar hold, exit next open after bar 60."),
    ReplaySpec("time_120m", "time", 120, "Fixed 120-bar hold, exit next open after bar 120."),
    ReplaySpec("recross_or_time_15m", "recross_or_time", 15, "Exit when close crosses back across EMA50, else 15 bars."),
    ReplaySpec("recross_or_time_30m", "recross_or_time", 30, "Exit when close crosses back across EMA50, else 30 bars."),
    ReplaySpec("recross_or_time_60m", "recross_or_time", 60, "Exit when close crosses back across EMA50, else 60 bars."),
    ReplaySpec("slope_flip_or_time_15m", "slope_flip_or_time", 15, "Exit when EMA50 slope flips, else 15 bars."),
    ReplaySpec("slope_flip_or_time_30m", "slope_flip_or_time", 30, "Exit when EMA50 slope flips, else 30 bars."),
    ReplaySpec("slope_flip_or_time_60m", "slope_flip_or_time", 60, "Exit when EMA50 slope flips, else 60 bars."),
    ReplaySpec("recross_slope_or_time_15m", "recross_slope_or_time", 15, "Exit on EMA50 recross or slope flip, else 15 bars."),
    ReplaySpec("recross_slope_or_time_30m", "recross_slope_or_time", 30, "Exit on EMA50 recross or slope flip, else 30 bars."),
    ReplaySpec("recross_slope_or_time_60m", "recross_slope_or_time", 60, "Exit on EMA50 recross or slope flip, else 60 bars."),
)


# ---------------------------------------------------------------------------
# CLI and small helpers
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Research-only ETH EMA50 snapback/rejoin event-study replay over OKX 1m trade bars.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--primary-timeframe", default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--ema-period", type=int, default=50)
    p.add_argument("--fast-ema-period", type=int, default=20)
    p.add_argument("--slow-ema-period", type=int, default=200)
    p.add_argument("--trend-windows", default="60,120")
    p.add_argument("--reclaim-bars-list", default="3,5")
    p.add_argument("--pierce-pct-list", default="0.0002,0.0005")
    p.add_argument("--pierce-modes", default="wick,close", help="wick means low/high pierce; close means close crosses the wrong side.")
    p.add_argument("--trend-above-ratio", type=float, default=0.65)
    p.add_argument("--min-slope-pct", type=float, default=0.0002, help="Minimum EMA50 slope over slope-window for prior trend.")
    p.add_argument("--slope-window", type=int, default=20)
    p.add_argument("--exit-time-horizons", default="5,10,15,30,45,60,120", help="Fixed-time exit horizons in primary bars.")
    p.add_argument("--dynamic-max-hold-bars-list", default="15,30,60", help="Max hold bars for EMA recross/slope-flip dynamic exits.")
    p.add_argument("--round-trip-cost-pct", type=float, default=0.0011)
    p.add_argument("--cost-multipliers", default="1.0,1.5,2.0,3.0")
    p.add_argument("--delay-bars-list", default="0,1,2,3")
    p.add_argument("--min-count", type=int, default=300)
    p.add_argument("--min-events-per-year", type=float, default=80.0)
    p.add_argument("--baseline-min-count", type=int, default=80)
    p.add_argument("--baseline-samples", type=int, default=100)
    p.add_argument("--baseline-max-events-per-group", type=int, default=1200, help="Cap matched-baseline pseudo events per replay group to keep runtime bounded.")
    p.add_argument("--baseline-prefilter-mean-net", type=float, default=-0.0002, help="Only run matched baseline for replay groups with preliminary mean_net above this level.")
    p.add_argument("--baseline-prefilter-pf", type=float, default=0.95, help="Only run matched baseline for replay groups with preliminary PF at least this value.")
    p.add_argument("--baseline-seed", type=int, default=42)
    p.add_argument("--cooldown-bars", type=int, default=0)
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None, help="Optional data directory passed to OKXTradeBarLoader.")
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--no-build-missing", action="store_true", help="Only read already cached trade bars.")
    p.add_argument("--force-rebuild", action="store_true", help="Pass force_rebuild to OKXTradeBarLoader.")
    p.add_argument("--event-sample-size", type=int, default=5000)
    p.add_argument("--write-full-trades", action="store_true", help="Also write full 03_replay_trades.csv; can be large.")
    p.add_argument("--progress-every", type=int, default=8)
    return p.parse_args(argv)


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    vals: list[int] = []
    for part in str(raw).split(","):
        text = part.strip()
        if text:
            vals.append(int(text))
    return tuple(dict.fromkeys(vals))


def _parse_csv_floats(raw: str) -> tuple[float, ...]:
    vals: list[float] = []
    for part in str(raw).split(","):
        text = part.strip()
        if text:
            vals.append(float(text))
    return tuple(dict.fromkeys(vals))


def _parse_csv_strings(raw: str) -> tuple[str, ...]:
    vals = [part.strip() for part in str(raw).split(",") if part.strip()]
    return tuple(dict.fromkeys(vals))


def _safe_divide(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray) -> pd.Series:
    n = pd.Series(num) if not isinstance(num, pd.Series) else num.astype(float)
    d = pd.Series(den, index=n.index) if not isinstance(den, pd.Series) else den.astype(float)
    out = n.astype(float) / d.replace(0, np.nan).astype(float)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _zscore_current_vs_prior(series: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce").astype(float)
    mp = min_periods or max(5, window // 3)
    mean = x.shift(1).rolling(window, min_periods=mp).mean()
    std = x.shift(1).rolling(window, min_periods=mp).std(ddof=0).replace(0, np.nan)
    return ((x - mean) / std).replace([np.inf, -np.inf], np.nan)


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = 0.0
    return out


def _bool_rolling_any(flag: pd.Series, window: int) -> pd.Series:
    return flag.astype(float).rolling(window, min_periods=1).max().fillna(0).astype(bool)


def _stable_hash_int(text: str, modulo: int) -> int:
    if modulo <= 0:
        return 0
    return int(zlib.crc32(text.encode("utf-8")) % modulo)


def _empty_float() -> float:
    return float("nan")


def generate_replay_specs(args: argparse.Namespace) -> tuple[ReplaySpec, ...]:
    specs: list[ReplaySpec] = []
    for h in _parse_csv_ints(args.exit_time_horizons):
        specs.append(ReplaySpec(f"time_{h}m", "time", int(h), f"Fixed {h}-bar hold, exit next open after bar {h}."))
    for h in _parse_csv_ints(args.dynamic_max_hold_bars_list):
        specs.append(ReplaySpec(f"recross_or_time_{h}m", "recross_or_time", int(h), f"Exit when close crosses back across EMA50, else {h} bars."))
        specs.append(ReplaySpec(f"slope_flip_or_time_{h}m", "slope_flip_or_time", int(h), f"Exit when EMA50 slope flips, else {h} bars."))
        specs.append(ReplaySpec(f"recross_slope_or_time_{h}m", "recross_slope_or_time", int(h), f"Exit on EMA50 recross or slope flip, else {h} bars."))
    # Preserve order while de-duplicating user-supplied values.
    dedup: dict[str, ReplaySpec] = {}
    for spec in specs:
        dedup.setdefault(spec.replay_id, spec)
    return tuple(dedup.values())


# ---------------------------------------------------------------------------
# Data / features
# ---------------------------------------------------------------------------


def load_trade_bars(
    *,
    symbol: str,
    timeframe: str,
    start_date: str,
    end_date: str,
    chunksize: int,
    data_dir: str | None,
    db_name: str,
    build_missing: bool,
    force_rebuild: bool,
) -> pd.DataFrame:
    loader = OKXTradeBarLoader(symbol=symbol, timeframe=timeframe, data_dir=data_dir, db_name=db_name)
    df = loader.fetch_data_by_date_range(
        start_date,
        end_date,
        chunksize=chunksize,
        force_rebuild=force_rebuild,
        cvd_mode="range",
        build_missing=build_missing,
    )
    if df.empty:
        return df
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "timestamp"
    return df.sort_index()


def add_sessions_regimes(features: pd.DataFrame) -> pd.DataFrame:
    out = features.copy()
    hour = out.index.hour
    session = np.where(hour < 8, "asia_00_08", np.where(hour < 16, "asia_europe_08_16", "us_16_24"))
    out["session"] = session
    out["year"] = out.index.year
    out["month"] = out.index.month
    out["weekday"] = out.index.weekday

    close = pd.to_numeric(out["close"], errors="coerce").astype(float)
    ret_120 = close.pct_change(120)
    ret_240 = close.pct_change(240)
    ema200_slope = pd.to_numeric(out.get("ema200_slope", 0.0), errors="coerce").astype(float)
    close_above_ema200 = close > pd.to_numeric(out.get("ema200", close), errors="coerce").astype(float)
    regime = np.full(len(out), "range", dtype=object)
    regime[(ret_240 > 0.004) & (ema200_slope > 0.0002) & close_above_ema200.to_numpy()] = "trend_up"
    regime[(ret_240 < -0.004) & (ema200_slope < -0.0002) & (~close_above_ema200.to_numpy())] = "trend_down"
    regime[(regime == "range") & (ret_120 > 0.0015)] = "normal_up"
    regime[(regime == "range") & (ret_120 < -0.0015)] = "normal_down"
    out["regime"] = regime

    vol = pd.to_numeric(out.get("ret_1", close.pct_change()), errors="coerce").rolling(120, min_periods=30).std()
    try:
        bucket = pd.qcut(vol.rank(method="first"), 4, labels=("vol_q1", "vol_q2", "vol_q3", "vol_q4"))
        out["volatility_bucket"] = bucket.astype(str).replace("nan", "unknown")
    except Exception:
        out["volatility_bucket"] = "unknown"
    return out


def build_features(bars: pd.DataFrame, *, ema_period: int, fast_ema_period: int, slow_ema_period: int, slope_window: int) -> tuple[pd.DataFrame, list[str]]:
    required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "notional",
        "buy_notional",
        "sell_notional",
        "delta_notional",
        "vwap",
        "large_buy_notional",
        "large_sell_notional",
        "large_delta_notional",
        "trades_count",
    ]
    base = _ensure_columns(bars, required)
    open_ = pd.to_numeric(base["open"], errors="coerce").astype(float)
    high = pd.to_numeric(base["high"], errors="coerce").astype(float)
    low = pd.to_numeric(base["low"], errors="coerce").astype(float)
    close = pd.to_numeric(base["close"], errors="coerce").astype(float)
    volume = pd.to_numeric(base.get("volume", 0.0), errors="coerce").astype(float)
    notional = pd.to_numeric(base.get("notional", 0.0), errors="coerce").astype(float)
    buy_notional = pd.to_numeric(base.get("buy_notional", 0.0), errors="coerce").astype(float)
    sell_notional = pd.to_numeric(base.get("sell_notional", 0.0), errors="coerce").astype(float)
    delta = pd.to_numeric(base.get("delta_notional", 0.0), errors="coerce").astype(float)
    large_buy = pd.to_numeric(base.get("large_buy_notional", 0.0), errors="coerce").astype(float)
    large_sell = pd.to_numeric(base.get("large_sell_notional", 0.0), errors="coerce").astype(float)
    large_delta = pd.to_numeric(base.get("large_delta_notional", 0.0), errors="coerce").astype(float)
    trades_count = pd.to_numeric(base.get("trades_count", 0.0), errors="coerce").astype(float)

    tr = (high - low).replace(0, np.nan)
    ema = close.ewm(span=ema_period, adjust=False, min_periods=ema_period).mean()
    fast_ema = close.ewm(span=fast_ema_period, adjust=False, min_periods=fast_ema_period).mean()
    slow_ema = close.ewm(span=slow_ema_period, adjust=False, min_periods=slow_ema_period).mean()
    ema_slope = ema / ema.shift(slope_window).replace(0, np.nan) - 1.0
    fast_slope = fast_ema / fast_ema.shift(slope_window).replace(0, np.nan) - 1.0
    slow_slope = slow_ema / slow_ema.shift(max(1, slope_window * 2)).replace(0, np.nan) - 1.0

    feature_parts: dict[str, pd.Series | np.ndarray] = {
        "ema50": ema,
        "ema20": fast_ema,
        "ema200": slow_ema,
        "ema50_gap": close / ema.replace(0, np.nan) - 1.0,
        "ema20_gap": close / fast_ema.replace(0, np.nan) - 1.0,
        "ema200_gap": close / slow_ema.replace(0, np.nan) - 1.0,
        "ema50_slope": ema_slope,
        "ema20_slope": fast_slope,
        "ema200_slope": slow_slope,
        "ret_1": close.pct_change(1),
        "ret_3": close.pct_change(3),
        "ret_5": close.pct_change(5),
        "ret_15": close.pct_change(15),
        "ret_60": close.pct_change(60),
        "range_pct": (high - low) / open_.replace(0, np.nan),
        "body_pct": (close - open_) / open_.replace(0, np.nan),
        "close_pos": ((close - low) / tr).clip(0.0, 1.0),
        "upper_wick_pct": ((high - np.maximum(open_, close)) / tr).clip(0.0, 1.0),
        "lower_wick_pct": ((np.minimum(open_, close) - low) / tr).clip(0.0, 1.0),
        "delta_ratio": _safe_divide(delta, notional.abs()),
        "large_delta_ratio": _safe_divide(large_delta, notional.abs()),
        "buy_pressure": _safe_divide(buy_notional, notional.abs()),
        "sell_pressure": _safe_divide(sell_notional, notional.abs()),
        "large_buy_share": _safe_divide(large_buy, buy_notional.abs()),
        "large_sell_share": _safe_divide(large_sell, sell_notional.abs()),
        "notional_z_30": _zscore_current_vs_prior(notional, 30),
        "notional_z_60": _zscore_current_vs_prior(notional, 60),
        "delta_z_30": _zscore_current_vs_prior(delta, 30),
        "delta_z_60": _zscore_current_vs_prior(delta, 60),
        "large_buy_z_60": _zscore_current_vs_prior(large_buy, 60),
        "large_sell_z_60": _zscore_current_vs_prior(large_sell, 60),
        "trades_count_z_60": _zscore_current_vs_prior(trades_count, 60),
        "above_ema50": close > ema,
        "below_ema50": close < ema,
        "ema50_wick_pierce_down": low < ema,
        "ema50_wick_pierce_up": high > ema,
        "ema50_close_cross_up": (close > ema) & (close.shift(1) <= ema.shift(1)),
        "ema50_close_cross_down": (close < ema) & (close.shift(1) >= ema.shift(1)),
    }
    features = pd.concat([base, pd.DataFrame(feature_parts, index=base.index)], axis=1)
    for window in (30, 60, 120):
        features[f"above_ema50_ratio_{window}"] = features["above_ema50"].shift(1).rolling(window, min_periods=max(10, window // 3)).mean()
        features[f"below_ema50_ratio_{window}"] = features["below_ema50"].shift(1).rolling(window, min_periods=max(10, window // 3)).mean()
    features = add_sessions_regimes(features)
    features["trend_side"] = np.where(
        (features["above_ema50_ratio_60"] >= 0.65) & (features["ema50_slope"] > 0),
        "up",
        np.where((features["below_ema50_ratio_60"] >= 0.65) & (features["ema50_slope"] < 0), "down", "none"),
    )
    feature_cols = [c for c in features.columns if c not in bars.columns]
    return features, feature_cols


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


def generate_event_specs(args: argparse.Namespace) -> list[EventSpec]:
    specs: list[EventSpec] = []
    for direction in ("long", "short"):
        for tw in _parse_csv_ints(args.trend_windows):
            for rb in _parse_csv_ints(args.reclaim_bars_list):
                for pct in _parse_csv_floats(args.pierce_pct_list):
                    for mode in _parse_csv_strings(args.pierce_modes):
                        if mode not in {"wick", "close"}:
                            raise ValueError(f"unsupported pierce mode: {mode}")
                        specs.append(
                            EventSpec(
                                direction=direction,
                                trend_window=int(tw),
                                reclaim_bars=int(rb),
                                pierce_pct=float(pct),
                                pierce_mode=mode,
                                trend_above_ratio=float(args.trend_above_ratio),
                                min_slope_pct=float(args.min_slope_pct),
                            )
                        )
    return specs


def apply_cooldown(mask: pd.Series, cooldown_bars: int) -> pd.Series:
    if cooldown_bars <= 0:
        return mask.fillna(False).astype(bool)
    arr = mask.fillna(False).to_numpy(dtype=bool)
    keep = np.zeros_like(arr, dtype=bool)
    last = -10**12
    for i, flag in enumerate(arr):
        if flag and i - last > cooldown_bars:
            keep[i] = True
            last = i
    return pd.Series(keep, index=mask.index)


def build_event_mask(features: pd.DataFrame, spec: EventSpec) -> pd.Series:
    close = pd.to_numeric(features["close"], errors="coerce").astype(float)
    high = pd.to_numeric(features["high"], errors="coerce").astype(float)
    low = pd.to_numeric(features["low"], errors="coerce").astype(float)
    ema = pd.to_numeric(features["ema50"], errors="coerce").astype(float)
    fast = pd.to_numeric(features["ema20"], errors="coerce").astype(float)
    ema_slope = pd.to_numeric(features["ema50_slope"], errors="coerce").astype(float)
    close_pos = pd.to_numeric(features["close_pos"], errors="coerce").astype(float)
    delta_ratio = pd.to_numeric(features["delta_ratio"], errors="coerce").astype(float)
    notional_z = pd.to_numeric(features["notional_z_60"], errors="coerce").astype(float)

    rb = spec.reclaim_bars
    tw = spec.trend_window
    if spec.direction == "long":
        prior_trend = (
            (features["above_ema50"].shift(rb).rolling(tw, min_periods=max(20, tw // 3)).mean() >= spec.trend_above_ratio)
            & (fast.shift(rb) > ema.shift(rb))
            & (ema_slope.shift(rb) > spec.min_slope_pct)
        )
        if spec.pierce_mode == "wick":
            pierce = low < ema * (1.0 - spec.pierce_pct)
            recent_pierce = _bool_rolling_any(pierce, rb)
            reclaim_now = (close > ema) & (((close.shift(1) <= ema.shift(1)) | pierce))
        else:
            pierce = close < ema * (1.0 - spec.pierce_pct)
            recent_pierce = _bool_rolling_any(pierce, rb + 1)
            reclaim_now = (close > ema) & (close.shift(1) <= ema.shift(1))
        quality = (close_pos >= 0.55) & (delta_ratio >= -0.35) & (notional_z >= -1.0)
    else:
        prior_trend = (
            (features["below_ema50"].shift(rb).rolling(tw, min_periods=max(20, tw // 3)).mean() >= spec.trend_above_ratio)
            & (fast.shift(rb) < ema.shift(rb))
            & (ema_slope.shift(rb) < -spec.min_slope_pct)
        )
        if spec.pierce_mode == "wick":
            pierce = high > ema * (1.0 + spec.pierce_pct)
            recent_pierce = _bool_rolling_any(pierce, rb)
            reclaim_now = (close < ema) & (((close.shift(1) >= ema.shift(1)) | pierce))
        else:
            pierce = close > ema * (1.0 + spec.pierce_pct)
            recent_pierce = _bool_rolling_any(pierce, rb + 1)
            reclaim_now = (close < ema) & (close.shift(1) >= ema.shift(1))
        quality = (close_pos <= 0.45) & (delta_ratio <= 0.35) & (notional_z >= -1.0)

    return (prior_trend & recent_pierce & reclaim_now & quality).fillna(False).astype(bool)


def build_events(features: pd.DataFrame, specs: Sequence[EventSpec], *, cooldown_bars: int, progress_every: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    with ProgressReporter("[events] specs", total=len(specs), every=max(1, progress_every)) as pr:
        for i, spec in enumerate(specs, start=1):
            mask = apply_cooldown(build_event_mask(features, spec), cooldown_bars)
            if mask.any():
                idx_pos = np.flatnonzero(mask.to_numpy(dtype=bool))
                part = pd.DataFrame(
                    {
                        "event_name": spec.event_name,
                        "family": "ema50_trend_snapback_rejoin",
                        "variant": spec.variant,
                        "direction": spec.direction,
                        "trend_window": spec.trend_window,
                        "reclaim_bars": spec.reclaim_bars,
                        "pierce_pct": spec.pierce_pct,
                        "pierce_mode": spec.pierce_mode,
                        "signal_idx": idx_pos,
                        "signal_time": features.index[idx_pos],
                        "year": features["year"].to_numpy()[idx_pos],
                        "month": features["month"].to_numpy()[idx_pos],
                        "session": features["session"].to_numpy()[idx_pos],
                        "regime": features["regime"].to_numpy()[idx_pos],
                        "volatility_bucket": features["volatility_bucket"].to_numpy()[idx_pos],
                        "trend_side": "up" if spec.direction == "long" else "down",
                        "ema50_gap": features["ema50_gap"].to_numpy()[idx_pos],
                        "ema50_slope": features["ema50_slope"].to_numpy()[idx_pos],
                        "close_pos": features["close_pos"].to_numpy()[idx_pos],
                        "delta_ratio": features["delta_ratio"].to_numpy()[idx_pos],
                        "notional_z_60": features["notional_z_60"].to_numpy()[idx_pos],
                    }
                )
                rows.append(part)
            pr.update(i)
    if not rows:
        return pd.DataFrame()
    events = pd.concat(rows, ignore_index=True)
    return events.sort_values(["signal_time", "event_name"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Replay engine
# ---------------------------------------------------------------------------


def _first_exit_signal_idx(
    *,
    direction: str,
    entry_idx: int,
    max_signal_idx: int,
    exit_model: str,
    close_arr: np.ndarray,
    ema_arr: np.ndarray,
    ema_slope_arr: np.ndarray,
) -> tuple[int, str]:
    if exit_model == "time":
        return max_signal_idx, "time"
    for j in range(entry_idx, max_signal_idx + 1):
        recross = False
        slope_flip = False
        if direction == "long":
            recross = bool(close_arr[j] < ema_arr[j])
            slope_flip = bool(ema_slope_arr[j] <= 0)
        else:
            recross = bool(close_arr[j] > ema_arr[j])
            slope_flip = bool(ema_slope_arr[j] >= 0)
        if exit_model == "recross_or_time" and recross:
            return j, "ema50_recross"
        if exit_model == "slope_flip_or_time" and slope_flip:
            return j, "ema50_slope_flip"
        if exit_model == "recross_slope_or_time":
            if recross and slope_flip:
                return j, "ema50_recross_and_slope_flip"
            if recross:
                return j, "ema50_recross"
            if slope_flip:
                return j, "ema50_slope_flip"
    return max_signal_idx, "time"


def replay_indices(
    *,
    event_positions: np.ndarray,
    signal_indices: np.ndarray,
    signal_times: np.ndarray,
    directions: np.ndarray,
    spec: ReplaySpec,
    delay_bars: int,
    features: pd.DataFrame,
) -> pd.DataFrame:
    index = features.index
    n = len(features)
    open_arr = pd.to_numeric(features["open"], errors="coerce").to_numpy(dtype=float)
    high_arr = pd.to_numeric(features["high"], errors="coerce").to_numpy(dtype=float)
    low_arr = pd.to_numeric(features["low"], errors="coerce").to_numpy(dtype=float)
    close_arr = pd.to_numeric(features["close"], errors="coerce").to_numpy(dtype=float)
    ema_arr = pd.to_numeric(features["ema50"], errors="coerce").to_numpy(dtype=float)
    ema_slope_arr = pd.to_numeric(features["ema50_slope"], errors="coerce").to_numpy(dtype=float)

    out_rows: list[dict[str, object]] = []
    for pos, sig_idx, sig_time, direction in zip(event_positions, signal_indices, signal_times, directions, strict=False):
        sig_i = int(sig_idx)
        entry_idx = sig_i + 1 + int(delay_bars)
        max_signal_idx = entry_idx + int(spec.max_hold_bars) - 1
        exit_idx_time = max_signal_idx + 1
        if entry_idx >= n or exit_idx_time >= n or sig_i < 0:
            out_rows.append(
                {
                    "event_pos": int(pos),
                    "signal_idx": sig_i,
                    "signal_time": pd.Timestamp(sig_time),
                    "entry_idx": entry_idx,
                    "entry_time": pd.NaT,
                    "entry_price": np.nan,
                    "exit_signal_idx": max_signal_idx,
                    "exit_signal_time": pd.NaT,
                    "exit_idx": exit_idx_time,
                    "exit_time": pd.NaT,
                    "exit_price": np.nan,
                    "exit_reason": "invalid_window",
                    "hold_bars": np.nan,
                    "gross_return": np.nan,
                    "mfe": np.nan,
                    "mae": np.nan,
                    "valid": False,
                    "invalid_reason": "insufficient_forward_window",
                }
            )
            continue
        exit_signal_idx, exit_reason = _first_exit_signal_idx(
            direction=str(direction),
            entry_idx=entry_idx,
            max_signal_idx=max_signal_idx,
            exit_model=spec.exit_model,
            close_arr=close_arr,
            ema_arr=ema_arr,
            ema_slope_arr=ema_slope_arr,
        )
        exit_idx = exit_signal_idx + 1
        if exit_idx >= n:
            valid = False
            invalid_reason = "exit_after_data_end"
            entry_price = np.nan
            exit_price = np.nan
            gross = np.nan
            mfe = np.nan
            mae = np.nan
            hold_bars = np.nan
        else:
            valid = True
            invalid_reason = ""
            entry_price = float(open_arr[entry_idx])
            exit_price = float(open_arr[exit_idx])
            if not math.isfinite(entry_price) or not math.isfinite(exit_price) or entry_price <= 0 or exit_price <= 0:
                valid = False
                invalid_reason = "bad_price"
                gross = np.nan
                mfe = np.nan
                mae = np.nan
                hold_bars = np.nan
            else:
                hold_bars = int(exit_idx - entry_idx)
                window_hi = np.nanmax(high_arr[entry_idx:exit_idx]) if exit_idx > entry_idx else np.nan
                window_lo = np.nanmin(low_arr[entry_idx:exit_idx]) if exit_idx > entry_idx else np.nan
                if str(direction) == "long":
                    gross = exit_price / entry_price - 1.0
                    mfe = window_hi / entry_price - 1.0 if math.isfinite(window_hi) else np.nan
                    mae = window_lo / entry_price - 1.0 if math.isfinite(window_lo) else np.nan
                else:
                    gross = entry_price / exit_price - 1.0
                    mfe = entry_price / window_lo - 1.0 if math.isfinite(window_lo) and window_lo > 0 else np.nan
                    mae = entry_price / window_hi - 1.0 if math.isfinite(window_hi) and window_hi > 0 else np.nan
        out_rows.append(
            {
                "event_pos": int(pos),
                "signal_idx": sig_i,
                "signal_time": pd.Timestamp(sig_time),
                "entry_idx": entry_idx,
                "entry_time": index[entry_idx] if 0 <= entry_idx < n else pd.NaT,
                "entry_price": entry_price,
                "exit_signal_idx": int(exit_signal_idx),
                "exit_signal_time": index[exit_signal_idx] if 0 <= exit_signal_idx < n else pd.NaT,
                "exit_idx": int(exit_idx),
                "exit_time": index[exit_idx] if 0 <= exit_idx < n else pd.NaT,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "hold_bars": hold_bars,
                "gross_return": gross,
                "mfe": mfe,
                "mae": mae,
                "valid": bool(valid),
                "invalid_reason": invalid_reason,
            }
        )
    return pd.DataFrame(out_rows)


def replay_events(events: pd.DataFrame, features: pd.DataFrame, replay_specs: Sequence[ReplaySpec], *, delay_bars: int, cost: float, progress_every: int) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    event_positions = events.index.to_numpy(dtype=int)
    sig_indices = events["signal_idx"].to_numpy(dtype=int)
    sig_times = pd.to_datetime(events["signal_time"]).to_numpy()
    dirs = events["direction"].astype(str).to_numpy()
    with ProgressReporter("[forward] replay specs", total=len(replay_specs), every=max(1, progress_every)) as pr:
        for i, spec in enumerate(replay_specs, start=1):
            rr = replay_indices(
                event_positions=event_positions,
                signal_indices=sig_indices,
                signal_times=sig_times,
                directions=dirs,
                spec=spec,
                delay_bars=delay_bars,
                features=features,
            )
            joined = events.reset_index(names="event_pos").merge(rr, on="event_pos", how="inner", suffixes=("", "_replay"))
            joined["replay_id"] = spec.replay_id
            joined["exit_model"] = spec.exit_model
            joined["max_hold_bars"] = spec.max_hold_bars
            joined["delay_bars"] = int(delay_bars)
            joined["net_return"] = joined["gross_return"] - cost
            frames.append(joined)
            pr.update(i)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Aggregation / baseline
# ---------------------------------------------------------------------------


def return_stats(values: pd.Series) -> dict[str, float | int]:
    x = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {
            "count": 0,
            "mean_net": np.nan,
            "median_net": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "top5_winner_share": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p95": np.nan,
        }
    wins = x[x > 0]
    losses = x[x < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(-losses.sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf if gross_profit > 0 else np.nan
    winners = x.sort_values(ascending=False).head(5).sum()
    total_pos = wins.sum()
    top5_share = float(winners / total_pos) if total_pos > 0 else np.nan
    return {
        "count": int(len(x)),
        "mean_net": float(x.mean()),
        "median_net": float(x.median()),
        "win_rate": float((x > 0).mean()),
        "profit_factor": float(pf),
        "top5_winner_share": top5_share,
        "p05": float(x.quantile(0.05)),
        "p25": float(x.quantile(0.25)),
        "p75": float(x.quantile(0.75)),
        "p95": float(x.quantile(0.95)),
    }


def frequency_stats(group: pd.DataFrame, *, start_date: str, end_date: str) -> dict[str, float | int]:
    if group.empty:
        return {
            "events_per_year": np.nan,
            "events_per_month": np.nan,
            "active_months": 0,
            "month_coverage": 0.0,
            "max_days_without_event": np.nan,
            "positive_years": 0,
            "year_count": 0,
        }
    signal_time = pd.to_datetime(group["signal_time"])
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    days = max(1.0, float((end - start).days + 1))
    years = days / 365.25
    months_total = max(1, len(pd.period_range(start=start, end=end, freq="M")))
    ym = signal_time.dt.to_period("M")
    active_months = int(ym.nunique())
    sorted_times = signal_time.sort_values()
    if len(sorted_times) <= 1:
        max_gap = np.nan
    else:
        max_gap = float(sorted_times.diff().dt.total_seconds().dropna().max() / 86400.0)
    yearly = group.groupby(pd.to_datetime(group["signal_time"]).dt.year)["net_return"].mean()
    return {
        "events_per_year": float(len(group) / years),
        "events_per_month": float(len(group) / months_total),
        "active_months": active_months,
        "month_coverage": float(active_months / months_total),
        "max_days_without_event": max_gap,
        "positive_years": int((yearly > 0).sum()),
        "year_count": int(yearly.shape[0]),
    }


def build_summary(trades: pd.DataFrame, *, start_date: str, end_date: str, cost: float, cost_multipliers: Sequence[float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if trades.empty:
        return pd.DataFrame()
    valid = trades[trades["valid"].astype(bool)].copy()
    keys = ["event_name", "family", "variant", "direction", "replay_id", "exit_model", "max_hold_bars"]
    for key, g in valid.groupby(keys, dropna=False):
        row = dict(zip(keys, key, strict=False))
        row.update(return_stats(g["net_return"]))
        row.update(frequency_stats(g, start_date=start_date, end_date=end_date))
        row["mfe_mean"] = float(pd.to_numeric(g["mfe"], errors="coerce").mean())
        row["mae_mean"] = float(pd.to_numeric(g["mae"], errors="coerce").mean())
        row["avg_hold_bars"] = float(pd.to_numeric(g["hold_bars"], errors="coerce").mean())
        row["median_hold_bars"] = float(pd.to_numeric(g["hold_bars"], errors="coerce").median())
        row["time_exit_share"] = float((g["exit_reason"] == "time").mean())
        for mult in cost_multipliers:
            net = pd.to_numeric(g["gross_return"], errors="coerce") - cost * float(mult)
            row[f"cost_{mult:g}x_mean_net"] = float(net.mean())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mean_net", "profit_factor"], ascending=[False, False]).reset_index(drop=True)


def build_breakdowns(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    valid = trades[trades["valid"].astype(bool)].copy()
    rows_year: list[dict[str, object]] = []
    rows_session: list[dict[str, object]] = []
    rows_regime: list[dict[str, object]] = []
    base_keys = ["event_name", "direction", "replay_id"]
    for key, g0 in valid.groupby(base_keys, dropna=False):
        base = dict(zip(base_keys, key, strict=False))
        for year, g in g0.groupby("year"):
            r = dict(base)
            r["year"] = int(year)
            r.update(return_stats(g["net_return"]))
            rows_year.append(r)
        for session, g in g0.groupby("session"):
            r = dict(base)
            r["session"] = session
            r.update(return_stats(g["net_return"]))
            rows_session.append(r)
        for regime, g in g0.groupby("regime"):
            r = dict(base)
            r["regime"] = regime
            r.update(return_stats(g["net_return"]))
            rows_regime.append(r)
    return pd.DataFrame(rows_year), pd.DataFrame(rows_session), pd.DataFrame(rows_regime)


def build_cost_stress(trades: pd.DataFrame, cost: float, cost_multipliers: Sequence[float]) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    valid = trades[trades["valid"].astype(bool)].copy()
    rows: list[dict[str, object]] = []
    keys = ["event_name", "direction", "replay_id"]
    for key, g in valid.groupby(keys, dropna=False):
        for mult in cost_multipliers:
            net = pd.to_numeric(g["gross_return"], errors="coerce") - cost * float(mult)
            r = dict(zip(keys, key, strict=False))
            r["cost_multiplier"] = float(mult)
            r.update(return_stats(net))
            rows.append(r)
    return pd.DataFrame(rows)


def build_delay_stress(events: pd.DataFrame, features: pd.DataFrame, replay_specs: Sequence[ReplaySpec], *, cost: float, delay_bars_list: Sequence[int], progress_every: int) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    total = len(delay_bars_list) * len(replay_specs)
    done = 0
    with ProgressReporter("[aggregate] delay stress", total=total, every=max(1, progress_every)) as pr:
        for delay in delay_bars_list:
            if int(delay) == 0:
                # Still recompute; keeps code simple and avoids stale filtering.
                pass
            for spec in replay_specs:
                rr = replay_indices(
                    event_positions=events.index.to_numpy(dtype=int),
                    signal_indices=events["signal_idx"].to_numpy(dtype=int),
                    signal_times=pd.to_datetime(events["signal_time"]).to_numpy(),
                    directions=events["direction"].astype(str).to_numpy(),
                    spec=spec,
                    delay_bars=int(delay),
                    features=features,
                )
                tmp = events.reset_index(names="event_pos").merge(rr, on="event_pos", how="inner")
                tmp = tmp[tmp["valid"].astype(bool)].copy()
                tmp["net_return"] = tmp["gross_return"] - cost
                for key, g in tmp.groupby(["event_name", "direction"], dropna=False):
                    r = {"event_name": key[0], "direction": key[1], "replay_id": spec.replay_id, "delay_bars": int(delay)}
                    r.update(return_stats(g["net_return"]))
                    rows.append(r)
                done += 1
                pr.update(done)
    return pd.DataFrame(rows)


def make_baseline_universe(features: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Build non-event baseline rows used by matched-baseline sampling.

    Keep this compact: the full feature frame is large, and matched baseline only
    needs coordinates plus signal_idx.  A prior implementation filtered this
    DataFrame once per event row and per sample; on multi-year 1m data that is
    effectively unbounded.  V1.0.1 precomputes grouped numpy pools instead.
    """
    signal_idx_set = set(int(x) for x in events["signal_idx"].dropna().astype(int).tolist()) if not events.empty else set()
    base = features.reset_index(names="signal_time").reset_index(names="signal_idx")
    base["is_event_bar"] = base["signal_idx"].isin(signal_idx_set)
    required = ["signal_idx", "signal_time", "year", "month", "session", "regime", "volatility_bucket", "trend_side"]
    return base[required + ["is_event_bar"]].copy()


def _profit_factor_np(values: pd.Series | np.ndarray) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    if x.size == 0:
        return np.nan
    wins = x[x > 0.0].sum()
    losses = -x[x < 0.0].sum()
    if losses > 0.0:
        return float(wins / losses)
    return float(np.inf) if wins > 0.0 else np.nan


def _pool_dict(universe: pd.DataFrame, columns: Sequence[str]) -> dict[tuple[object, ...], np.ndarray]:
    out: dict[tuple[object, ...], np.ndarray] = {}
    if universe.empty:
        return out
    for key, g in universe.groupby(list(columns), dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        vals = g["signal_idx"].to_numpy(dtype=int)
        if vals.size:
            out[key] = vals
    return out


def _matched_pool_counts(
    event_group: pd.DataFrame,
    *,
    exact_pools: dict[tuple[object, ...], np.ndarray],
    relaxed_vol_pools: dict[tuple[object, ...], np.ndarray],
    relaxed_month_pools: dict[tuple[object, ...], np.ndarray],
    max_events: int,
    seed_text: str,
) -> tuple[list[tuple[np.ndarray, int]], float, int]:
    if event_group.empty:
        return [], 0.0, 0
    eg = event_group
    if max_events > 0 and len(eg) > max_events:
        # Deterministic cap keeps runtime bounded while preserving the matched
        # distribution reasonably well.  This is research baseline sampling, not
        # entry generation, so it does not affect the actual event replay.
        eg = eg.sample(n=int(max_events), random_state=_stable_hash_int(seed_text, 2_000_000_000))
    exact_cols = ["year", "month", "session", "regime", "volatility_bucket", "trend_side"]
    matched: list[tuple[np.ndarray, int]] = []
    matched_count = 0
    for key, sub in eg.groupby(exact_cols, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        cnt = int(len(sub))
        pool = exact_pools.get(key)
        if pool is None or pool.size == 0:
            # Relax volatility bucket first.
            key_relaxed_vol = (key[0], key[1], key[2], key[3], key[5])
            pool = relaxed_vol_pools.get(key_relaxed_vol)
        if pool is None or pool.size == 0:
            # If that is still empty, relax month but keep year/session/regime/trend.
            key_relaxed_month = (key[0], key[2], key[3], key[5])
            pool = relaxed_month_pools.get(key_relaxed_month)
        if pool is not None and pool.size > 0:
            matched.append((pool, cnt))
            matched_count += cnt
    denom = max(1, int(len(eg)))
    return matched, float(matched_count / denom), int(len(eg))


def _sample_from_pool_counts(pool_counts: Sequence[tuple[np.ndarray, int]], rng: np.random.Generator) -> np.ndarray:
    if not pool_counts:
        return np.array([], dtype=int)
    pieces: list[np.ndarray] = []
    for pool, cnt in pool_counts:
        if cnt <= 0 or pool.size == 0:
            continue
        # Replacement is intentional: baseline asks for matched non-event bars,
        # not a unique draw constraint.  It is much faster and avoids exhausting
        # small exact buckets.
        take = pool[rng.integers(0, pool.size, size=int(cnt))]
        pieces.append(take.astype(int, copy=False))
    if not pieces:
        return np.array([], dtype=int)
    return np.concatenate(pieces).astype(int, copy=False)


def build_matched_baseline(
    trades: pd.DataFrame,
    events: pd.DataFrame,
    features: pd.DataFrame,
    replay_specs: Sequence[ReplaySpec],
    *,
    baseline_samples: int,
    baseline_seed: int,
    baseline_min_count: int,
    baseline_max_events_per_group: int,
    baseline_prefilter_mean_net: float,
    baseline_prefilter_pf: float,
    cost: float,
    progress_every: int,
) -> pd.DataFrame:
    if trades.empty or events.empty or baseline_samples <= 0:
        return pd.DataFrame()

    print("[aggregate] matched baseline prep", flush=True)
    universe = make_baseline_universe(features, events)
    non_event_universe = universe[~universe["is_event_bar"].astype(bool)].copy()
    if non_event_universe.empty:
        return pd.DataFrame()
    exact_pools = _pool_dict(non_event_universe, ["year", "month", "session", "regime", "volatility_bucket", "trend_side"])
    relaxed_vol_pools = _pool_dict(non_event_universe, ["year", "month", "session", "regime", "trend_side"])
    relaxed_month_pools = _pool_dict(non_event_universe, ["year", "session", "regime", "trend_side"])
    spec_by_id = {s.replay_id: s for s in replay_specs}

    valid_cols = [
        "event_pos",
        "event_name",
        "direction",
        "replay_id",
        "net_return",
        "valid",
    ]
    valid = trades.loc[trades["valid"].astype(bool), [c for c in valid_cols if c in trades.columns]].copy()
    if valid.empty:
        return pd.DataFrame()

    # Pre-filter: matched baseline is a validator for plausible candidates, not
    # an expensive operation required for every obviously losing replay variant.
    # This avoids repeatedly replaying millions of pseudo-events.
    pre_rows: list[dict[str, object]] = []
    for key, g in valid.groupby(["event_name", "direction", "replay_id"], dropna=False, sort=False):
        if len(g) < baseline_min_count:
            continue
        mean_net = float(pd.to_numeric(g["net_return"], errors="coerce").mean())
        pf = _profit_factor_np(g["net_return"])
        if (math.isfinite(mean_net) and mean_net >= float(baseline_prefilter_mean_net)) and (math.isfinite(pf) and pf >= float(baseline_prefilter_pf)):
            pre_rows.append({"key": key, "count": int(len(g)), "mean_net": mean_net, "pf": pf})
    if not pre_rows:
        print("[aggregate] matched baseline skipped: no groups passed preliminary filters", flush=True)
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    with ProgressReporter("[aggregate] matched baseline", total=len(pre_rows), every=max(1, progress_every)) as pr:
        for i, item in enumerate(pre_rows, start=1):
            event_name, direction, replay_id = item["key"]
            spec = spec_by_id.get(str(replay_id))
            if spec is None:
                pr.update(i)
                continue
            g = valid[
                (valid["event_name"] == event_name)
                & (valid["direction"] == direction)
                & (valid["replay_id"] == replay_id)
            ]
            if g.empty:
                pr.update(i)
                continue
            unique_event_pos = g["event_pos"].dropna().astype(int).unique()
            event_group = events.loc[unique_event_pos].copy()
            if event_group.empty:
                pr.update(i)
                continue
            pool_counts, base_match_rate, baseline_event_count = _matched_pool_counts(
                event_group,
                exact_pools=exact_pools,
                relaxed_vol_pools=relaxed_vol_pools,
                relaxed_month_pools=relaxed_month_pools,
                max_events=int(baseline_max_events_per_group),
                seed_text=f"{event_name}|{direction}|{replay_id}",
            )
            if not pool_counts:
                rows.append(
                    {
                        "event_name": event_name,
                        "direction": direction,
                        "replay_id": replay_id,
                        "count": int(len(g)),
                        "event_mean_net": float(g["net_return"].mean()),
                        "baseline_mean_net": np.nan,
                        "baseline_std_mean_net": np.nan,
                        "matched_excess_mean_net": np.nan,
                        "baseline_p_value": np.nan,
                        "baseline_samples": 0,
                        "baseline_match_rate": 0.0,
                        "baseline_event_count_used": int(baseline_event_count),
                    }
                )
                pr.update(i)
                continue
            sample_means: list[float] = []
            sample_wins = 0
            event_mean = float(pd.to_numeric(g["net_return"], errors="coerce").mean())
            for sample_i in range(int(baseline_samples)):
                rng = np.random.default_rng(int(baseline_seed) + _stable_hash_int(f"{event_name}|{replay_id}|{sample_i}", 10_000_000))
                pseudo_indices = _sample_from_pool_counts(pool_counts, rng)
                if pseudo_indices.size == 0:
                    continue
                pseudo = pd.DataFrame(
                    {
                        "event_pos": np.arange(pseudo_indices.size, dtype=int),
                        "signal_idx": pseudo_indices,
                        "signal_time": features.index[pseudo_indices],
                        "direction": str(direction),
                    }
                )
                rr = replay_indices(
                    event_positions=pseudo["event_pos"].to_numpy(dtype=int),
                    signal_indices=pseudo["signal_idx"].to_numpy(dtype=int),
                    signal_times=pd.to_datetime(pseudo["signal_time"]).to_numpy(),
                    directions=pseudo["direction"].astype(str).to_numpy(),
                    spec=spec,
                    delay_bars=0,
                    features=features,
                )
                rr = rr[rr["valid"].astype(bool)]
                if rr.empty:
                    continue
                net = pd.to_numeric(rr["gross_return"], errors="coerce") - cost
                mean_net = float(net.mean())
                sample_means.append(mean_net)
                if mean_net >= event_mean:
                    sample_wins += 1
            baseline_mean = float(np.nanmean(sample_means)) if sample_means else np.nan
            baseline_std = float(np.nanstd(sample_means, ddof=1)) if len(sample_means) > 1 else np.nan
            p_value = float((sample_wins + 1) / (len(sample_means) + 1)) if sample_means else np.nan
            rows.append(
                {
                    "event_name": event_name,
                    "direction": direction,
                    "replay_id": replay_id,
                    "count": int(len(g)),
                    "event_mean_net": event_mean,
                    "baseline_mean_net": baseline_mean,
                    "baseline_std_mean_net": baseline_std,
                    "matched_excess_mean_net": event_mean - baseline_mean if math.isfinite(baseline_mean) else np.nan,
                    "baseline_p_value": p_value,
                    "baseline_samples": int(len(sample_means)),
                    "baseline_match_rate": float(base_match_rate),
                    "baseline_event_count_used": int(baseline_event_count),
                    "baseline_prefilter_mean_net": float(item["mean_net"]),
                    "baseline_prefilter_pf": float(item["pf"]),
                }
            )
            pr.update(i)
    return pd.DataFrame(rows)

def build_research_decision(
    summary: pd.DataFrame,
    baseline: pd.DataFrame,
    delay: pd.DataFrame,
    *,
    min_count: int,
    min_events_per_year: float,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    out = summary.copy()
    if not baseline.empty:
        out = out.merge(
            baseline[["event_name", "direction", "replay_id", "baseline_mean_net", "matched_excess_mean_net", "baseline_p_value", "baseline_match_rate"]],
            on=["event_name", "direction", "replay_id"],
            how="left",
        )
    else:
        out["baseline_mean_net"] = np.nan
        out["matched_excess_mean_net"] = np.nan
        out["baseline_p_value"] = np.nan
        out["baseline_match_rate"] = np.nan
    if not delay.empty:
        delay1 = delay[delay["delay_bars"] == 1][["event_name", "direction", "replay_id", "mean_net"]].rename(columns={"mean_net": "delay1_mean_net"})
        out = out.merge(delay1, on=["event_name", "direction", "replay_id"], how="left")
    else:
        out["delay1_mean_net"] = np.nan
    reasons: list[str] = []
    decisions: list[str] = []
    for _, r in out.iterrows():
        fail: list[str] = []
        count = int(r.get("count", 0) or 0)
        events_per_year = float(r.get("events_per_year", np.nan))
        fee2 = float(r.get("cost_2x_mean_net", np.nan))
        delay1 = float(r.get("delay1_mean_net", np.nan))
        matched_excess = float(r.get("matched_excess_mean_net", np.nan))
        pf = float(r.get("profit_factor", np.nan))
        wr = float(r.get("win_rate", np.nan))
        top5 = float(r.get("top5_winner_share", np.nan))
        if count < min_count:
            fail.append("count_below_min")
        if not math.isfinite(events_per_year) or events_per_year < min_events_per_year:
            fail.append("frequency_below_mhf_min")
        if not math.isfinite(float(r.get("mean_net", np.nan))) or float(r.get("mean_net", np.nan)) <= 0:
            fail.append("mean_net_nonpositive")
        if not math.isfinite(pf) or pf < 1.15:
            fail.append("pf_below_1_15")
        if not math.isfinite(wr) or wr < 0.52:
            fail.append("win_rate_too_low")
        if not math.isfinite(fee2) or fee2 <= 0:
            fail.append("fee2_not_positive")
        if not math.isfinite(delay1) or delay1 <= 0:
            fail.append("delay1_not_positive")
        if not math.isfinite(matched_excess) or matched_excess <= 0:
            fail.append("matched_excess_nonpositive")
        if int(r.get("positive_years", 0) or 0) < 3:
            fail.append("positive_years_below_3")
        if math.isfinite(top5) and top5 > 0.35:
            fail.append("top5_winner_share_high")
        if fail:
            if count >= max(80, min_count // 2) and math.isfinite(matched_excess) and matched_excess > 0 and math.isfinite(fee2) and fee2 > -0.0005:
                decisions.append("research_continue_weak")
            else:
                decisions.append("rejected")
            reasons.append(";".join(fail))
        else:
            decisions.append("research_continue_candidate")
            reasons.append("passes_predeclared_research_filters_not_tradable_yet")
    out["decision"] = decisions
    out["reason"] = reasons
    return out.sort_values(["decision", "mean_net"], ascending=[True, False]).reset_index(drop=True)


def build_event_summary(events: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for key, g in events.groupby(["event_name", "family", "variant", "direction"], dropna=False):
        row = dict(zip(["event_name", "family", "variant", "direction"], key, strict=False))
        row["count"] = int(len(g))
        freq = frequency_stats(g.assign(net_return=0.0), start_date=start_date, end_date=end_date)
        row.update({k: v for k, v in freq.items() if k in {"events_per_year", "events_per_month", "active_months", "month_coverage", "max_days_without_event"}})
        row["sessions"] = ",".join(sorted(map(str, g["session"].dropna().unique())))
        row["regimes"] = ",".join(sorted(map(str, g["regime"].dropna().unique())))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def build_causal_audit(trades: pd.DataFrame, *, primary_delta: pd.Timedelta) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    aud = trades[[
        "event_name",
        "direction",
        "replay_id",
        "signal_time",
        "entry_time",
        "exit_signal_time",
        "exit_time",
        "signal_idx",
        "entry_idx",
        "exit_signal_idx",
        "exit_idx",
        "valid",
        "invalid_reason",
    ]].copy()
    aud["expected_entry_time"] = pd.to_datetime(aud["signal_time"]) + primary_delta
    aud["entry_not_next_open_flag"] = (pd.to_datetime(aud["entry_time"]) != aud["expected_entry_time"]).astype(int)
    aud["expected_exit_time"] = pd.to_datetime(aud["exit_signal_time"]) + primary_delta
    aud["exit_not_next_open_after_signal_flag"] = (pd.to_datetime(aud["exit_time"]) != aud["expected_exit_time"]).astype(int)
    aud["forward_window_valid_flag"] = aud["valid"].astype(bool).astype(int)
    aud["lookahead_flag"] = (
        (aud["entry_idx"].astype(float) <= aud["signal_idx"].astype(float))
        | (aud["exit_idx"].astype(float) <= aud["exit_signal_idx"].astype(float))
        | (aud["exit_signal_idx"].astype(float) < aud["entry_idx"].astype(float))
    ).astype(int)
    return aud


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    text = str(timeframe).strip()
    unit = text[-1].lower()
    num = int(text[:-1])
    if unit == "s":
        return pd.Timedelta(seconds=num)
    if unit == "m":
        return pd.Timedelta(minutes=num)
    if unit == "h":
        return pd.Timedelta(hours=num)
    if unit == "d":
        return pd.Timedelta(days=num)
    raise ValueError(f"unsupported timeframe: {timeframe}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def write_json(path: Path, obj: object) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cost_multipliers = _parse_csv_floats(args.cost_multipliers)
    delay_bars_list = _parse_csv_ints(args.delay_bars_list)
    build_missing = not bool(args.no_build_missing)

    print(f"[run] {SCRIPT_NAME} version={SCRIPT_VERSION}", flush=True)
    print(
        f"[run] symbol={args.symbol} primary={args.primary_timeframe} range={args.start_date}->{args.end_date} warmup={args.warmup_start_date}",
        flush=True,
    )
    print("[load] primary trade bars", flush=True)
    bars = load_trade_bars(
        symbol=args.symbol,
        timeframe=args.primary_timeframe,
        start_date=args.warmup_start_date,
        end_date=args.end_date,
        chunksize=args.chunksize,
        data_dir=args.data_dir,
        db_name=args.db_name,
        build_missing=build_missing,
        force_rebuild=args.force_rebuild,
    )
    print(f"[load] primary rows={len(bars):,} range={bars.index.min()} -> {bars.index.max()}", flush=True)
    if bars.empty:
        raise SystemExit("No trade bars loaded")

    print("[features] building EMA/orderflow features", flush=True)
    features, feature_cols = build_features(
        bars,
        ema_period=args.ema_period,
        fast_ema_period=args.fast_ema_period,
        slow_ema_period=args.slow_ema_period,
        slope_window=args.slope_window,
    )
    research_start = pd.Timestamp(args.start_date)
    research_end = pd.Timestamp(args.end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    features = features.loc[(features.index >= research_start) & (features.index <= research_end)].copy()
    print(f"[features] research rows={len(features):,} feature_columns={len(feature_cols)}", flush=True)

    print("[events] building EMA50 snapback/rejoin events", flush=True)
    specs = generate_event_specs(args)
    events = build_events(features, specs, cooldown_bars=args.cooldown_bars, progress_every=args.progress_every)
    print(f"[events] rows={len(events):,} specs={len(specs)}", flush=True)

    replay_specs = generate_replay_specs(args)
    print(f"[forward] replay fixed/dynamic exits specs={len(replay_specs)}", flush=True)
    trades = replay_events(
        events,
        features,
        replay_specs,
        delay_bars=0,
        cost=float(args.round_trip_cost_pct),
        progress_every=args.progress_every,
    )
    print(f"[forward] replay rows={len(trades):,}", flush=True)

    print("[aggregate] summaries", flush=True)
    event_summary = build_event_summary(events, start_date=args.start_date, end_date=args.end_date)
    replay_summary = build_summary(trades, start_date=args.start_date, end_date=args.end_date, cost=float(args.round_trip_cost_pct), cost_multipliers=cost_multipliers)
    yearly, session_breakdown, regime_breakdown = build_breakdowns(trades)
    cost_stress = build_cost_stress(trades, float(args.round_trip_cost_pct), cost_multipliers)
    delay_stress = build_delay_stress(events, features, replay_specs, cost=float(args.round_trip_cost_pct), delay_bars_list=delay_bars_list, progress_every=args.progress_every)
    baseline = build_matched_baseline(
        trades,
        events,
        features,
        replay_specs,
        baseline_samples=int(args.baseline_samples),
        baseline_seed=int(args.baseline_seed),
        baseline_min_count=int(args.baseline_min_count),
        baseline_max_events_per_group=int(args.baseline_max_events_per_group),
        baseline_prefilter_mean_net=float(args.baseline_prefilter_mean_net),
        baseline_prefilter_pf=float(args.baseline_prefilter_pf),
        cost=float(args.round_trip_cost_pct),
        progress_every=args.progress_every,
    )
    decision = build_research_decision(
        replay_summary,
        baseline,
        delay_stress,
        min_count=int(args.min_count),
        min_events_per_year=float(args.min_events_per_year),
    )
    causal_audit = build_causal_audit(trades, primary_delta=timeframe_to_timedelta(args.primary_timeframe))

    print("[write] report files", flush=True)
    manifest = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "symbol": args.symbol,
        "primary_timeframe": args.primary_timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "ema_period": args.ema_period,
        "trend_windows": _parse_csv_ints(args.trend_windows),
        "reclaim_bars_list": _parse_csv_ints(args.reclaim_bars_list),
        "pierce_pct_list": _parse_csv_floats(args.pierce_pct_list),
        "pierce_modes": _parse_csv_strings(args.pierce_modes),
        "exit_time_horizons": _parse_csv_ints(args.exit_time_horizons),
        "dynamic_max_hold_bars_list": _parse_csv_ints(args.dynamic_max_hold_bars_list),
        "exit_replay_specs": [spec.__dict__ for spec in replay_specs],
        "round_trip_cost_pct": args.round_trip_cost_pct,
        "cost_multipliers": cost_multipliers,
        "delay_bars_list": delay_bars_list,
        "baseline_max_events_per_group": int(args.baseline_max_events_per_group),
        "baseline_prefilter_mean_net": float(args.baseline_prefilter_mean_net),
        "baseline_prefilter_pf": float(args.baseline_prefilter_pf),
        "input_rows": int(len(bars)),
        "research_rows": int(len(features)),
        "event_spec_count": int(len(specs)),
        "event_count": int(len(events)),
        "replay_row_count": int(len(trades)),
        "research_continue_candidate_count": int((decision.get("decision", pd.Series(dtype=str)) == "research_continue_candidate").sum()) if not decision.empty else 0,
        "causal_lookahead_count": int(causal_audit["lookahead_flag"].sum()) if not causal_audit.empty else 0,
        "causal_policy": CAUSAL_POLICY,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    write_json(out_dir / "00_manifest.json", manifest)
    event_summary.to_csv(out_dir / "01_event_summary.csv", index=False)
    replay_summary.to_csv(out_dir / "02_replay_variant_summary.csv", index=False)
    if args.write_full_trades:
        trades.to_csv(out_dir / "03_replay_trades.csv", index=False)
    else:
        trades.head(int(args.event_sample_size)).to_csv(out_dir / "03_replay_trades_sample.csv", index=False)
    yearly.to_csv(out_dir / "04_yearly_breakdown.csv", index=False)
    session_breakdown.to_csv(out_dir / "05_session_breakdown.csv", index=False)
    regime_breakdown.to_csv(out_dir / "06_regime_breakdown.csv", index=False)
    cost_stress.to_csv(out_dir / "07_cost_stress.csv", index=False)
    delay_stress.to_csv(out_dir / "08_delay_stress.csv", index=False)
    baseline.to_csv(out_dir / "09_matched_baseline_summary.csv", index=False)
    decision.to_csv(out_dir / "10_research_decision.csv", index=False)
    causal_audit.to_csv(out_dir / "11_causal_audit.csv", index=False)
    events.head(int(args.event_sample_size)).to_csv(out_dir / "12_event_sample.csv", index=False)
    write_json(out_dir / "13_feature_columns.json", {"feature_columns": feature_cols})

    print("[review-pack] finalizing", flush=True)
    finalize_research_report(
        out_dir,
        experiment_id=EXPERIMENT_ID,
        edge_id=EDGE_ID,
        title="ETH MHF EMA50 Snapback Rejoin V1",
        print_log=True,
    )
    print(f"[done] report_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
