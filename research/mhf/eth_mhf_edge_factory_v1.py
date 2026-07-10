#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH MHF Edge Factory V1: causal event-study factory for OKX trade bars.

This is a research-only script. It does not register a tradable edge, does not
modify portfolio code, and does not import from other research scripts.

Timing policy:
    signal_time = current primary closed bar timestamp
    entry_time  = next primary bar open plus optional delay bars
    entry_price = open[entry_idx]

Context policy:
    high-timeframe context is shifted from bar-start timestamp to available_time
    before merge_asof, so a 5m bar starting 15:10 becomes usable at 15:15.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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

SCRIPT_NAME = "eth_mhf_edge_factory_v1.py"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_MHF_EDGE_FACTORY_V1"
EDGE_ID = "ETH_EDGE_MHF_RESEARCH_FACTORY_V1"
DEFAULT_OUT_DIR = "data/reports/research/mhf_edge_factory_v1"
CAUSAL_POLICY = (
    "closed primary trade bar signal; entry at next primary open plus delay; "
    "context bars aligned by bar_start + timeframe_delta available_time via merge_asof backward"
)


@dataclass(frozen=True)
class EventSpec:
    family: str
    variant: str
    direction: str
    params: dict[str, float]

    @property
    def event_name(self) -> str:
        return f"{self.family}__{self.variant}__{self.direction.lower()}"


@dataclass(frozen=True)
class ReturnStats:
    count: int
    mean: float
    median: float
    win_rate: float
    profit_factor: float
    top5_winner_share: float
    p05: float
    p25: float
    p75: float
    p95: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Research-only ETH MHF event-study factory over OKX trade bars.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--primary-timeframe", default="1m")
    p.add_argument("--context-timeframes", default="3m,5m,15m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--horizons", default="5,15,30,60,120,240")
    p.add_argument("--primary-horizon", type=int, default=60, help="Horizon used for candidate decisions.")
    p.add_argument("--mfe-mae-horizon", type=int, default=240)
    p.add_argument("--round-trip-cost-pct", type=float, default=0.0011)
    p.add_argument("--cost-multipliers", default="1.0,1.5,2.0,3.0")
    p.add_argument("--delay-bars-list", default="0,1,2,3")
    p.add_argument("--min-count", type=int, default=150)
    p.add_argument("--cooldown-bars", type=int, default=0)
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None, help="Optional data directory passed to OKXTradeBarLoader.")
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--no-build-missing", action="store_true", help="Only read already cached trade bars.")
    p.add_argument("--force-rebuild", action="store_true", help="Pass force_rebuild to OKXTradeBarLoader.")
    p.add_argument("--event-sample-size", type=int, default=5000)
    p.add_argument("--progress-every", type=int, default=4)
    p.add_argument("--write-full-events", action="store_true", help="Also write 15_events_with_forward_returns.csv; can be large.")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Small parsing/time helpers
# ---------------------------------------------------------------------------


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


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    text = str(timeframe).strip()
    if len(text) < 2:
        raise ValueError(f"invalid timeframe: {timeframe!r}")
    unit = text[-1]
    num = text[:-1]
    if not num.isdigit():
        raise ValueError(f"invalid timeframe: {timeframe!r}")
    n = int(num)
    if n <= 0:
        raise ValueError(f"timeframe must be positive: {timeframe!r}")
    u = unit.lower()
    if u == "s":
        return pd.Timedelta(seconds=n)
    if u == "m":
        return pd.Timedelta(minutes=n)
    if u == "h":
        return pd.Timedelta(hours=n)
    if u == "d":
        return pd.Timedelta(days=n)
    raise ValueError(f"unsupported timeframe unit: {timeframe!r}")


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


def _rolling_vwap(vwap: pd.Series, volume: pd.Series, window: int) -> pd.Series:
    v = pd.to_numeric(vwap, errors="coerce").astype(float)
    vol = pd.to_numeric(volume, errors="coerce").astype(float)
    return (v * vol).rolling(window, min_periods=max(5, window // 3)).sum() / vol.rolling(
        window, min_periods=max(5, window // 3)
    ).sum().replace(0, np.nan)


def _ensure_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = 0.0
    return out


def _max_timestamp_frame(cols: list[pd.Series]) -> pd.Series:
    if not cols:
        return pd.Series(pd.NaT)
    frame = pd.concat(cols, axis=1)
    return frame.max(axis=1)


# ---------------------------------------------------------------------------
# Data loading and causal context alignment
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


def build_context_features(ctx: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if ctx.empty:
        return ctx
    tf = str(timeframe)
    out = pd.DataFrame(index=ctx.index)
    close = pd.to_numeric(ctx["close"], errors="coerce")
    notional = pd.to_numeric(ctx.get("notional", 0.0), errors="coerce")
    delta = pd.to_numeric(ctx.get("delta_notional", 0.0), errors="coerce")
    vol = pd.to_numeric(ctx.get("volume", 0.0), errors="coerce")
    vwap = pd.to_numeric(ctx.get("vwap", close), errors="coerce")
    out[f"ctx_{tf}_bar_start_time"] = ctx.index
    out[f"ctx_{tf}_available_time"] = ctx.index + timeframe_to_timedelta(tf)
    out[f"ctx_{tf}_ret_1"] = close.pct_change(1)
    out[f"ctx_{tf}_ret_3"] = close.pct_change(3)
    out[f"ctx_{tf}_notional_z_60"] = _zscore_current_vs_prior(notional, 60)
    out[f"ctx_{tf}_delta_z_60"] = _zscore_current_vs_prior(delta, 60)
    out[f"ctx_{tf}_delta_ratio"] = _safe_divide(delta, notional.abs())
    rvwap = _rolling_vwap(vwap, vol, 30)
    out[f"ctx_{tf}_vwap_gap_30"] = close / rvwap.replace(0, np.nan) - 1.0
    return out


def causal_attach_context(primary: pd.DataFrame, contexts: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    merged = primary.copy().sort_index()
    context_cols: list[str] = []
    for tf, raw_ctx in contexts.items():
        if raw_ctx.empty:
            continue
        ctx = build_context_features(raw_ctx, tf)
        if ctx.empty:
            continue
        delta = timeframe_to_timedelta(tf)
        ctx_asof = ctx.copy()
        ctx_asof.index = pd.to_datetime(raw_ctx.index) + delta
        ctx_asof.index.name = "available_time"
        merged = pd.merge_asof(
            merged.sort_index(),
            ctx_asof.sort_index(),
            left_index=True,
            right_index=True,
            direction="backward",
        )
        context_cols.extend([c for c in ctx.columns if c not in context_cols])
    return merged, context_cols


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------


def build_primary_features(bars: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    required = ["open", "high", "low", "close", "volume", "notional", "buy_notional", "sell_notional", "delta_notional", "vwap"]
    base = _ensure_columns(bars, required)
    idx = base.index
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
    vwap = pd.to_numeric(base.get("vwap", close), errors="coerce").astype(float)

    tr = (high - low).replace(0, np.nan)
    rolling_vwap_30 = _rolling_vwap(vwap, volume, 30)
    rolling_vwap_60 = _rolling_vwap(vwap, volume, 60)
    notional_mean_60 = notional.shift(1).rolling(60, min_periods=20).mean()
    notional_rel_60 = notional / notional_mean_60.replace(0, np.nan)

    prior_high: dict[int, pd.Series] = {}
    prior_low: dict[int, pd.Series] = {}
    current_high: dict[int, pd.Series] = {}
    current_low: dict[int, pd.Series] = {}
    feature_parts: dict[str, pd.Series | np.ndarray] = {
        "ret_1": close.pct_change(1),
        "ret_3": close.pct_change(3),
        "ret_5": close.pct_change(5),
        "ret_15": close.pct_change(15),
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
        "price_impact_abs": (close.pct_change(1).abs() / notional_rel_60.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan),
        "rolling_vwap_30": rolling_vwap_30,
        "rolling_vwap_60": rolling_vwap_60,
        "vwap_gap_30": close / rolling_vwap_30.replace(0, np.nan) - 1.0,
        "vwap_gap_60": close / rolling_vwap_60.replace(0, np.nan) - 1.0,
        "notional_z_30": _zscore_current_vs_prior(notional, 30),
        "notional_z_60": _zscore_current_vs_prior(notional, 60),
        "delta_z_30": _zscore_current_vs_prior(delta, 30),
        "delta_z_60": _zscore_current_vs_prior(delta, 60),
        "large_sell_z_60": _zscore_current_vs_prior(large_sell, 60),
        "large_buy_z_60": _zscore_current_vs_prior(large_buy, 60),
        "notional_rel_60": notional_rel_60,
        "price_efficiency_5": (close.pct_change(5).abs() / notional_rel_60.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan),
        "price_efficiency_15": (close.pct_change(15).abs() / notional_rel_60.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan),
    }

    for w in (30, 60, 120):
        current_high[w] = high.rolling(w, min_periods=max(5, w // 3)).max()
        current_low[w] = low.rolling(w, min_periods=max(5, w // 3)).min()
        prior_high[w] = high.shift(1).rolling(w, min_periods=max(5, w // 3)).max()
        prior_low[w] = low.shift(1).rolling(w, min_periods=max(5, w // 3)).min()
        feature_parts[f"rolling_high_{w}"] = current_high[w]
        feature_parts[f"rolling_low_{w}"] = current_low[w]
        feature_parts[f"prior_rolling_high_{w}"] = prior_high[w]
        feature_parts[f"prior_rolling_low_{w}"] = prior_low[w]
        feature_parts[f"break_high_{w}_flag"] = (high > prior_high[w]).astype(int)
        feature_parts[f"break_low_{w}_flag"] = (low < prior_low[w]).astype(int)

    range_30 = high.shift(1).rolling(30, min_periods=20).max() / low.shift(1).rolling(30, min_periods=20).min().replace(0, np.nan) - 1.0
    range_120 = high.shift(1).rolling(120, min_periods=60).max() / low.shift(1).rolling(120, min_periods=60).min().replace(0, np.nan) - 1.0
    feature_parts["compression_range_30"] = range_30
    feature_parts["compression_range_120"] = range_120
    feature_parts["compression_rank"] = (range_30 / range_120.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    feature_parts["trend_ret_60"] = close.pct_change(60)
    feature_parts["abs_ret_1_ma_60"] = close.pct_change(1).abs().rolling(60, min_periods=20).mean()

    features = pd.DataFrame(feature_parts, index=idx)
    features["session"] = session_labels(idx)
    features["year"] = idx.year.astype(int)
    features["month"] = idx.month.astype(int)
    features["weekday"] = idx.weekday.astype(int)
    features["regime"] = regime_labels(features)

    out = pd.concat([base, features], axis=1, copy=False)
    feature_columns = list(features.columns)
    return out, feature_columns


def session_labels(index: pd.DatetimeIndex) -> pd.Series:
    hour = index.hour
    labels = np.select(
        [hour < 8, (hour >= 8) & (hour < 16), hour >= 16],
        ["asia_00_08", "asia_europe_08_16", "europe_us_16_24"],
        default="unknown",
    )
    return pd.Series(labels, index=index, dtype="object")


def regime_labels(features: pd.DataFrame) -> pd.Series:
    trend = pd.to_numeric(features.get("trend_ret_60", np.nan), errors="coerce")
    vol = pd.to_numeric(features.get("abs_ret_1_ma_60", np.nan), errors="coerce")
    vol_z = _zscore_current_vs_prior(vol.fillna(0.0), 240, min_periods=80)
    high_vol = vol_z > 1.0
    low_vol = vol_z < -0.5
    trend_up = trend > 0.006
    trend_down = trend < -0.006
    labels = np.select(
        [high_vol & trend_up, high_vol & trend_down, high_vol, low_vol, trend_up, trend_down],
        ["high_vol_up", "high_vol_down", "high_vol_range", "low_vol", "normal_up", "normal_down"],
        default="normal_range",
    )
    return pd.Series(labels, index=features.index, dtype="object")


# ---------------------------------------------------------------------------
# Event definitions
# ---------------------------------------------------------------------------


def build_event_specs() -> list[EventSpec]:
    specs: list[EventSpec] = []
    forced_variants = {
        "w60_mild": {"ret5": 0.0040, "delta_z": 1.7, "large_z": 1.4, "large_share": 0.18, "close_pos": 0.42, "wick": 0.22, "swing": 60},
        "w60_strict": {"ret5": 0.0060, "delta_z": 2.1, "large_z": 1.8, "large_share": 0.24, "close_pos": 0.48, "wick": 0.28, "swing": 60},
        "w120_deep": {"ret5": 0.0080, "delta_z": 2.3, "large_z": 2.0, "large_share": 0.28, "close_pos": 0.45, "wick": 0.30, "swing": 120},
    }
    impulse_variants = {
        "fast_mild": {"ret5": 0.0045, "notional_z": 1.5, "delta_z": 1.8, "eff": 0.0035, "close_pos": 0.44},
        "fast_strict": {"ret5": 0.0060, "notional_z": 1.9, "delta_z": 2.2, "eff": 0.0030, "close_pos": 0.48},
        "deep_15m": {"ret15": 0.0100, "notional_z": 1.7, "delta_z": 2.0, "eff15": 0.0060, "close_pos": 0.46},
    }
    vwap_variants = {
        "gap60_mild": {"gap": 0.0035, "delta_z": 1.6, "pressure": 0.60, "close_pos": 0.44, "wick": 0.22},
        "gap60_strict": {"gap": 0.0050, "delta_z": 2.0, "pressure": 0.64, "close_pos": 0.48, "wick": 0.28},
        "gap30_fast": {"gap": 0.0030, "delta_z": 1.8, "pressure": 0.62, "close_pos": 0.46, "wick": 0.24, "use_gap30": 1.0},
    }
    breakout_variants = {
        "brk60_mild": {"swing": 60, "compression": 0.55, "notional_z": 1.4, "delta_z": 1.3, "delta_ratio": 0.15, "close_pos": 0.62},
        "brk60_strict": {"swing": 60, "compression": 0.45, "notional_z": 1.8, "delta_z": 1.7, "delta_ratio": 0.22, "close_pos": 0.68},
        "brk120_deep": {"swing": 120, "compression": 0.50, "notional_z": 1.7, "delta_z": 1.6, "delta_ratio": 0.20, "close_pos": 0.66},
    }
    for name, params in forced_variants.items():
        specs.append(EventSpec("forced_flow_exhaustion_reclaim", name, "LONG", params))
        specs.append(EventSpec("forced_flow_exhaustion_reclaim", name, "SHORT", params))
    for name, params in impulse_variants.items():
        specs.append(EventSpec("impulse_exhaustion_reversal", name, "LONG", params))
        specs.append(EventSpec("impulse_exhaustion_reversal", name, "SHORT", params))
    for name, params in vwap_variants.items():
        specs.append(EventSpec("vwap_extension_orderflow_fade", name, "LONG", params))
        specs.append(EventSpec("vwap_extension_orderflow_fade", name, "SHORT", params))
    for name, params in breakout_variants.items():
        specs.append(EventSpec("compression_breakout_followthrough", name, "LONG", params))
        specs.append(EventSpec("compression_breakout_followthrough", name, "SHORT", params))
    return specs


def build_event_mask(df: pd.DataFrame, spec: EventSpec) -> pd.Series:
    p = spec.params
    direction = spec.direction.upper()
    family = spec.family
    idx = df.index
    false = pd.Series(False, index=idx)

    if family == "forced_flow_exhaustion_reclaim":
        swing = int(p.get("swing", 60))
        if direction == "LONG":
            reclaim = (df["close"] >= df[f"prior_rolling_low_{swing}"]) | (df["close_pos"] >= p["close_pos"])
            mask = (
                (df["ret_5"] <= -p["ret5"])
                & ((df["delta_z_60"] <= -p["delta_z"]) | (df["sell_pressure"] >= 0.62))
                & ((df["large_sell_z_60"] >= p["large_z"]) | (df["large_sell_share"] >= p["large_share"]))
                & (df[f"break_low_{swing}_flag"].astype(bool))
                & reclaim
                & ((df["close_pos"] >= p["close_pos"]) | (df["lower_wick_pct"] >= p["wick"]))
            )
        else:
            reject = (df["close"] <= df[f"prior_rolling_high_{swing}"]) | (df["close_pos"] <= (1.0 - p["close_pos"]))
            mask = (
                (df["ret_5"] >= p["ret5"])
                & ((df["delta_z_60"] >= p["delta_z"]) | (df["buy_pressure"] >= 0.62))
                & ((df["large_buy_z_60"] >= p["large_z"]) | (df["large_buy_share"] >= p["large_share"]))
                & (df[f"break_high_{swing}_flag"].astype(bool))
                & reject
                & ((df["close_pos"] <= (1.0 - p["close_pos"])) | (df["upper_wick_pct"] >= p["wick"]))
            )
        return mask.fillna(False)

    if family == "impulse_exhaustion_reversal":
        if "ret15" in p:
            ret_col = "ret_15"
            ret_thr = p["ret15"]
            eff_col = "price_efficiency_15"
            eff_thr = p["eff15"]
        else:
            ret_col = "ret_5"
            ret_thr = p["ret5"]
            eff_col = "price_efficiency_5"
            eff_thr = p["eff"]
        if direction == "LONG":
            mask = (
                (df[ret_col] <= -ret_thr)
                & (df["notional_z_60"] >= p["notional_z"])
                & (df["delta_z_60"] <= -p["delta_z"])
                & (df[eff_col] <= eff_thr)
                & (df["close_pos"] >= p["close_pos"])
                & (df["delta_ratio"] <= -0.15)
            )
        else:
            mask = (
                (df[ret_col] >= ret_thr)
                & (df["notional_z_60"] >= p["notional_z"])
                & (df["delta_z_60"] >= p["delta_z"])
                & (df[eff_col] <= eff_thr)
                & (df["close_pos"] <= (1.0 - p["close_pos"]))
                & (df["delta_ratio"] >= 0.15)
            )
        return mask.fillna(False)

    if family == "vwap_extension_orderflow_fade":
        gap_col = "vwap_gap_30" if p.get("use_gap30", 0.0) > 0 else "vwap_gap_60"
        if direction == "LONG":
            mask = (
                (df[gap_col] <= -p["gap"])
                & ((df["delta_z_60"] <= -p["delta_z"]) | (df["sell_pressure"] >= p["pressure"]))
                & (df["close_pos"] >= p["close_pos"])
                & ((df["lower_wick_pct"] >= p["wick"]) | (df["price_efficiency_5"] <= 0.0035))
                & (df["delta_ratio"] <= -0.10)
            )
        else:
            mask = (
                (df[gap_col] >= p["gap"])
                & ((df["delta_z_60"] >= p["delta_z"]) | (df["buy_pressure"] >= p["pressure"]))
                & (df["close_pos"] <= (1.0 - p["close_pos"]))
                & ((df["upper_wick_pct"] >= p["wick"]) | (df["price_efficiency_5"] <= 0.0035))
                & (df["delta_ratio"] >= 0.10)
            )
        return mask.fillna(False)

    if family == "compression_breakout_followthrough":
        swing = int(p.get("swing", 60))
        compressed = df["compression_rank"] <= p["compression"]
        if direction == "LONG":
            mask = (
                compressed
                & df[f"break_high_{swing}_flag"].astype(bool)
                & (df["notional_z_60"] >= p["notional_z"])
                & ((df["delta_z_60"] >= p["delta_z"]) | (df["delta_ratio"] >= p["delta_ratio"]))
                & (df["close_pos"] >= p["close_pos"])
                & (df["ret_3"] > 0)
            )
        else:
            mask = (
                compressed
                & df[f"break_low_{swing}_flag"].astype(bool)
                & (df["notional_z_60"] >= p["notional_z"])
                & ((df["delta_z_60"] <= -p["delta_z"]) | (df["delta_ratio"] <= -p["delta_ratio"]))
                & (df["close_pos"] <= (1.0 - p["close_pos"]))
                & (df["ret_3"] < 0)
            )
        return mask.fillna(False)

    return false


def apply_cooldown(pos: np.ndarray, cooldown_bars: int) -> np.ndarray:
    if cooldown_bars <= 0 or len(pos) <= 1:
        return pos
    selected: list[int] = []
    last = -10**18
    gap = int(cooldown_bars)
    for p in pos:
        pi = int(p)
        if pi - last > gap:
            selected.append(pi)
            last = pi
    return np.asarray(selected, dtype=np.int64)


def build_events(df: pd.DataFrame, specs: list[EventSpec], *, start_date: str, end_date: str, cooldown_bars: int, progress_every: int) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if isinstance(end_date, str) and len(end_date.strip()) == 10:
        end_ts = end_ts + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    in_study = (df.index >= start_ts) & (df.index <= end_ts)

    rows: list[pd.DataFrame] = []
    label = "[events] event families"
    progress = ProgressReporter(label=label, total=len(specs), every=max(1, progress_every))
    for i, spec in enumerate(specs, start=1):
        mask = build_event_mask(df, spec) & in_study
        positions = np.flatnonzero(mask.to_numpy(dtype=bool))
        positions = apply_cooldown(positions, cooldown_bars)
        if len(positions):
            part = pd.DataFrame(
                {
                    "event_name": spec.event_name,
                    "family": spec.family,
                    "variant": spec.variant,
                    "direction": spec.direction.upper(),
                    "side": 1 if spec.direction.upper() == "LONG" else -1,
                    "signal_idx": positions.astype(np.int64),
                    "signal_time": df.index[positions],
                    "session": df["session"].iloc[positions].to_numpy(),
                    "year": df["year"].iloc[positions].to_numpy(),
                    "month": df["month"].iloc[positions].to_numpy(),
                    "weekday": df["weekday"].iloc[positions].to_numpy(),
                    "regime": df["regime"].iloc[positions].to_numpy(),
                    "signal_close": pd.to_numeric(df["close"], errors="coerce").iloc[positions].to_numpy(dtype=float),
                    "close_pos": pd.to_numeric(df["close_pos"], errors="coerce").iloc[positions].to_numpy(dtype=float),
                    "ret_5": pd.to_numeric(df["ret_5"], errors="coerce").iloc[positions].to_numpy(dtype=float),
                    "ret_15": pd.to_numeric(df["ret_15"], errors="coerce").iloc[positions].to_numpy(dtype=float),
                    "notional_z_60": pd.to_numeric(df["notional_z_60"], errors="coerce").iloc[positions].to_numpy(dtype=float),
                    "delta_z_60": pd.to_numeric(df["delta_z_60"], errors="coerce").iloc[positions].to_numpy(dtype=float),
                    "vwap_gap_60": pd.to_numeric(df["vwap_gap_60"], errors="coerce").iloc[positions].to_numpy(dtype=float),
                    "compression_rank": pd.to_numeric(df["compression_rank"], errors="coerce").iloc[positions].to_numpy(dtype=float),
                }
            )
            context_start_cols = [c for c in df.columns if c.startswith("ctx_") and c.endswith("_bar_start_time")]
            context_available_cols = [c for c in df.columns if c.startswith("ctx_") and c.endswith("_available_time")]
            for col in context_start_cols + context_available_cols:
                part[col] = df[col].iloc[positions].to_numpy()
            rows.append(part)
        progress.update(i)
    progress.close()
    if not rows:
        return pd.DataFrame(
            columns=["event_name", "family", "variant", "direction", "side", "signal_idx", "signal_time", "session", "year", "month", "weekday", "regime"]
        )
    events = pd.concat(rows, ignore_index=True, copy=False)
    events = events.sort_values(["signal_time", "event_name", "direction"], kind="stable").reset_index(drop=True)
    return events


# ---------------------------------------------------------------------------
# Forward returns / MFE / MAE
# ---------------------------------------------------------------------------


def future_rolling_extreme(values: np.ndarray, window: int, kind: str) -> np.ndarray:
    s = pd.Series(values.astype(float))
    rev = s.iloc[::-1]
    if kind == "max":
        out = rev.rolling(window, min_periods=window).max().iloc[::-1]
    elif kind == "min":
        out = rev.rolling(window, min_periods=window).min().iloc[::-1]
    else:
        raise ValueError("kind must be max or min")
    return out.to_numpy(dtype=float)


def attach_forward_returns(
    events: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    mfe_mae_horizon: int,
    round_trip_cost_pct: float,
    cost_multiplier: float = 1.0,
    delay_bars: int = 0,
    include_mfe_mae: bool = True,
) -> pd.DataFrame:
    out = events.copy()
    n = len(bars)
    if out.empty:
        return out
    signal_idx = pd.to_numeric(out["signal_idx"], errors="coerce").fillna(-1).astype(np.int64).to_numpy()
    side = pd.to_numeric(out["side"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    entry_idx = signal_idx + 1 + int(delay_bars)
    valid_entry = (entry_idx >= 0) & (entry_idx < n)
    opens = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    index = pd.DatetimeIndex(bars.index)

    entry_price = np.full(len(out), np.nan, dtype=float)
    entry_time = np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]")
    entry_price[valid_entry] = opens[entry_idx[valid_entry]]
    entry_time[valid_entry] = index.values[entry_idx[valid_entry]]
    out[f"delay{delay_bars}_entry_idx"] = entry_idx
    out[f"delay{delay_bars}_entry_time"] = entry_time
    out[f"delay{delay_bars}_entry_price"] = entry_price

    for h in horizons:
        exit_idx = entry_idx + int(h) - 1
        valid = valid_entry & (exit_idx >= 0) & (exit_idx < n) & np.isfinite(entry_price) & (entry_price > 0)
        exit_price = np.full(len(out), np.nan, dtype=float)
        exit_time = np.full(len(out), np.datetime64("NaT"), dtype="datetime64[ns]")
        exit_price[valid] = closes[exit_idx[valid]]
        exit_time[valid] = index.values[exit_idx[valid]]
        gross = np.full(len(out), np.nan, dtype=float)
        long_mask = valid & (side == 1) & np.isfinite(exit_price) & (exit_price > 0)
        short_mask = valid & (side == -1) & np.isfinite(exit_price) & (exit_price > 0)
        gross[long_mask] = exit_price[long_mask] / entry_price[long_mask] - 1.0
        gross[short_mask] = entry_price[short_mask] / exit_price[short_mask] - 1.0
        net = gross - float(round_trip_cost_pct) * float(cost_multiplier)
        out[f"delay{delay_bars}_h{h}_exit_time"] = exit_time
        out[f"delay{delay_bars}_h{h}_gross"] = gross
        out[f"delay{delay_bars}_h{h}_net_c{cost_multiplier:g}"] = net
        out[f"delay{delay_bars}_h{h}_valid"] = valid

    if include_mfe_mae:
        fut_max_high = future_rolling_extreme(highs, int(mfe_mae_horizon), "max")
        fut_min_low = future_rolling_extreme(lows, int(mfe_mae_horizon), "min")
        path_high = np.full(len(out), np.nan, dtype=float)
        path_low = np.full(len(out), np.nan, dtype=float)
        path_valid = valid_entry & (entry_idx + int(mfe_mae_horizon) - 1 < n) & np.isfinite(entry_price) & (entry_price > 0)
        path_high[path_valid] = fut_max_high[entry_idx[path_valid]]
        path_low[path_valid] = fut_min_low[entry_idx[path_valid]]
        mfe = np.full(len(out), np.nan, dtype=float)
        mae = np.full(len(out), np.nan, dtype=float)
        long_path = path_valid & (side == 1)
        short_path = path_valid & (side == -1)
        mfe[long_path] = path_high[long_path] / entry_price[long_path] - 1.0
        mae[long_path] = path_low[long_path] / entry_price[long_path] - 1.0
        mfe[short_path] = entry_price[short_path] / path_low[short_path] - 1.0
        mae[short_path] = entry_price[short_path] / path_high[short_path] - 1.0
        out[f"delay{delay_bars}_mfe_h{mfe_mae_horizon}"] = mfe
        out[f"delay{delay_bars}_mae_h{mfe_mae_horizon}"] = mae
        out[f"delay{delay_bars}_mfe_mae_valid"] = path_valid
    return out


# ---------------------------------------------------------------------------
# Statistics and report builders
# ---------------------------------------------------------------------------


def summarize_return_series(values: pd.Series) -> ReturnStats:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return ReturnStats(0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
    wins = x[x > 0]
    losses = x[x < 0]
    gp = float(wins.sum())
    gl = float(-losses.sum())
    if gl > 0:
        pf = gp / gl
    else:
        pf = float("inf") if gp > 0 else np.nan
    if gp > 0 and not wins.empty:
        top5 = float(wins.sort_values(ascending=False).head(5).sum() / gp)
    else:
        top5 = np.nan
    return ReturnStats(
        count=int(len(x)),
        mean=float(x.mean()),
        median=float(x.median()),
        win_rate=float((x > 0).mean()),
        profit_factor=float(pf),
        top5_winner_share=top5,
        p05=float(x.quantile(0.05)),
        p25=float(x.quantile(0.25)),
        p75=float(x.quantile(0.75)),
        p95=float(x.quantile(0.95)),
    )


def _stats_row(prefix: str, stats: ReturnStats) -> dict[str, object]:
    return {
        f"{prefix}count": stats.count,
        f"{prefix}mean": stats.mean,
        f"{prefix}median": stats.median,
        f"{prefix}win_rate": stats.win_rate,
        f"{prefix}profit_factor": stats.profit_factor,
        f"{prefix}top5_winner_share": stats.top5_winner_share,
        f"{prefix}p05": stats.p05,
        f"{prefix}p25": stats.p25,
        f"{prefix}p75": stats.p75,
        f"{prefix}p95": stats.p95,
    }


def summarize_by_groups(df: pd.DataFrame, *, group_cols: list[str], return_col: str, extra: dict[str, object] | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if df.empty or return_col not in df.columns:
        return pd.DataFrame()
    for key, part in df.groupby(group_cols, dropna=False, sort=True):
        key_tuple = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key_tuple))
        if extra:
            row.update(extra)
        stats = summarize_return_series(part[return_col])
        row.update(
            {
                "count": stats.count,
                "mean_net": stats.mean,
                "median_net": stats.median,
                "win_rate": stats.win_rate,
                "profit_factor": stats.profit_factor,
                "top5_winner_share": stats.top5_winner_share,
                "p05": stats.p05,
                "p25": stats.p25,
                "p75": stats.p75,
                "p95": stats.p95,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_forward_return_matrix(events: pd.DataFrame, horizons: tuple[int, ...], *, cost_multiplier: float = 1.0, delay_bars: int = 0) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["event_name", "family", "direction"]
    for h in horizons:
        gross_col = f"delay{delay_bars}_h{h}_gross"
        net_col = f"delay{delay_bars}_h{h}_net_c{cost_multiplier:g}"
        valid_col = f"delay{delay_bars}_h{h}_valid"
        if net_col not in events.columns:
            continue
        valid_df = events.loc[events.get(valid_col, True).astype(bool)].copy()
        for key, part in valid_df.groupby(group_cols, dropna=False, sort=True):
            event_name, family, direction = key
            net_stats = summarize_return_series(part[net_col])
            gross_stats = summarize_return_series(part[gross_col]) if gross_col in part.columns else ReturnStats(0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan)
            rows.append(
                {
                    "event_name": event_name,
                    "family": family,
                    "direction": direction,
                    "horizon": int(h),
                    "count": net_stats.count,
                    "mean_gross": gross_stats.mean,
                    "mean_net": net_stats.mean,
                    "median_net": net_stats.median,
                    "win_rate": net_stats.win_rate,
                    "profit_factor": net_stats.profit_factor,
                    "top5_winner_share": net_stats.top5_winner_share,
                    "p05": net_stats.p05,
                    "p25": net_stats.p25,
                    "p75": net_stats.p75,
                    "p95": net_stats.p95,
                }
            )
    return pd.DataFrame(rows)


def build_cost_stress(events: pd.DataFrame, horizons: tuple[int, ...], cost_multipliers: tuple[float, ...], round_trip_cost_pct: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    base_group_cols = ["event_name", "family", "direction"]
    for h in horizons:
        gross_col = f"delay0_h{h}_gross"
        valid_col = f"delay0_h{h}_valid"
        if gross_col not in events.columns:
            continue
        valid_df = events.loc[events.get(valid_col, True).astype(bool)]
        for mult in cost_multipliers:
            for key, part in valid_df.groupby(base_group_cols, dropna=False, sort=True):
                vals = pd.to_numeric(part[gross_col], errors="coerce") - float(round_trip_cost_pct) * float(mult)
                st = summarize_return_series(vals)
                rows.append(
                    {
                        "event_name": key[0],
                        "family": key[1],
                        "direction": key[2],
                        "horizon": int(h),
                        "cost_multiplier": float(mult),
                        "count": st.count,
                        "mean_net": st.mean,
                        "median_net": st.median,
                        "win_rate": st.win_rate,
                        "profit_factor": st.profit_factor,
                        "top5_winner_share": st.top5_winner_share,
                    }
                )
    return pd.DataFrame(rows)


def build_delay_stress(events_base: pd.DataFrame, bars: pd.DataFrame, horizons: tuple[int, ...], delays: tuple[int, ...], round_trip_cost_pct: float) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    rows: list[dict[str, object]] = []
    delayed_events: dict[int, pd.DataFrame] = {}
    base_cols = [
        "event_name",
        "family",
        "variant",
        "direction",
        "side",
        "signal_idx",
        "signal_time",
        "session",
        "year",
        "month",
        "weekday",
        "regime",
    ]
    minimal = events_base[[c for c in base_cols if c in events_base.columns]].copy()
    for delay in delays:
        ev = attach_forward_returns(
            minimal,
            bars,
            horizons=horizons,
            mfe_mae_horizon=max(horizons),
            round_trip_cost_pct=round_trip_cost_pct,
            cost_multiplier=1.0,
            delay_bars=int(delay),
            include_mfe_mae=False,
        )
        delayed_events[int(delay)] = ev
        for h in horizons:
            net_col = f"delay{delay}_h{h}_net_c1"
            valid_col = f"delay{delay}_h{h}_valid"
            if net_col not in ev.columns:
                continue
            valid_df = ev.loc[ev.get(valid_col, True).astype(bool)]
            for key, part in valid_df.groupby(["event_name", "family", "direction"], dropna=False, sort=True):
                st = summarize_return_series(part[net_col])
                rows.append(
                    {
                        "event_name": key[0],
                        "family": key[1],
                        "direction": key[2],
                        "horizon": int(h),
                        "delay_bars": int(delay),
                        "count": st.count,
                        "mean_net": st.mean,
                        "median_net": st.median,
                        "win_rate": st.win_rate,
                        "profit_factor": st.profit_factor,
                        "top5_winner_share": st.top5_winner_share,
                    }
                )
    return pd.DataFrame(rows), delayed_events


def max_days_without_event(times: pd.Series, start_date: str, end_date: str) -> float:
    vals = pd.to_datetime(times, errors="coerce").dropna().sort_values()
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if isinstance(end_date, str) and len(end_date.strip()) == 10:
        end = end + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    points = [start, *vals.tolist(), end]
    if len(points) < 2:
        return float("nan")
    diffs = [(points[i + 1] - points[i]).total_seconds() / 86400.0 for i in range(len(points) - 1)]
    return float(max(diffs)) if diffs else float("nan")


def build_candidate_tables(
    events: pd.DataFrame,
    forward_matrix: pd.DataFrame,
    cost_stress: pd.DataFrame,
    delay_stress: pd.DataFrame,
    *,
    primary_horizon: int,
    min_count: int,
    start_date: str,
    end_date: str,
    mfe_mae_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = forward_matrix.loc[forward_matrix["horizon"] == int(primary_horizon)].copy() if not forward_matrix.empty else pd.DataFrame()
    if primary.empty:
        columns = [
            "event_name", "family", "direction", "primary_horizon", "count", "mean_net", "median_net", "win_rate",
            "profit_factor", "mfe_mean", "mae_mean", "positive_years", "year_count", "session_best",
            "max_days_without_event", "fee_2x_mean_net", "delay_1bar_mean_net", "top5_winner_share", "decision", "reason",
        ]
        return pd.DataFrame(columns=columns), pd.DataFrame(columns=[*columns, "failed_rules"])

    yearly = summarize_by_groups(events, group_cols=["event_name", "family", "direction", "year"], return_col=f"delay0_h{primary_horizon}_net_c1")
    sessions = summarize_by_groups(events, group_cols=["event_name", "family", "direction", "session"], return_col=f"delay0_h{primary_horizon}_net_c1")
    mfe_col = f"delay0_mfe_h{mfe_mae_horizon}"
    mae_col = f"delay0_mae_h{mfe_mae_horizon}"
    rows: list[dict[str, object]] = []
    for _, row in primary.iterrows():
        event_name = row["event_name"]
        family = row["family"]
        direction = row["direction"]
        mask = (events["event_name"] == event_name) & (events["direction"] == direction)
        part = events.loc[mask]
        y = yearly.loc[(yearly["event_name"] == event_name) & (yearly["direction"] == direction)] if not yearly.empty else pd.DataFrame()
        positive_years = int(((pd.to_numeric(y.get("mean_net", pd.Series(dtype=float)), errors="coerce") > 0) & (pd.to_numeric(y.get("count", pd.Series(dtype=float)), errors="coerce") > 0)).sum()) if not y.empty else 0
        year_count = int(y["year"].nunique()) if not y.empty and "year" in y.columns else 0
        sess = sessions.loc[(sessions["event_name"] == event_name) & (sessions["direction"] == direction)] if not sessions.empty else pd.DataFrame()
        if not sess.empty and "mean_net" in sess.columns:
            best_idx = pd.to_numeric(sess["mean_net"], errors="coerce").idxmax()
            session_best = str(sess.loc[best_idx, "session"])
        else:
            session_best = ""
        fee_2x = np.nan
        if not cost_stress.empty:
            cs = cost_stress.loc[
                (cost_stress["event_name"] == event_name)
                & (cost_stress["direction"] == direction)
                & (cost_stress["horizon"] == int(primary_horizon))
                & (np.isclose(pd.to_numeric(cost_stress["cost_multiplier"], errors="coerce"), 2.0))
            ]
            if not cs.empty:
                fee_2x = float(pd.to_numeric(cs["mean_net"], errors="coerce").iloc[0])
        delay_1 = np.nan
        if not delay_stress.empty:
            ds = delay_stress.loc[
                (delay_stress["event_name"] == event_name)
                & (delay_stress["direction"] == direction)
                & (delay_stress["horizon"] == int(primary_horizon))
                & (delay_stress["delay_bars"] == 1)
            ]
            if not ds.empty:
                delay_1 = float(pd.to_numeric(ds["mean_net"], errors="coerce").iloc[0])
        failed: list[str] = []
        count = int(row.get("count", 0) or 0)
        mean_net = float(row.get("mean_net", np.nan))
        pf = float(row.get("profit_factor", np.nan)) if not isinstance(row.get("profit_factor"), str) else np.nan
        top5 = float(row.get("top5_winner_share", np.nan))
        if count < min_count:
            failed.append(f"count<{min_count}")
        if np.isnan(pf) or pf < 1.15:
            failed.append("profit_factor<1.15")
        if not np.isfinite(mean_net) or mean_net <= 0:
            failed.append("mean_net<=0")
        if positive_years < 3:
            failed.append("positive_years<3")
        if np.isfinite(top5) and top5 > 0.45:
            failed.append("top5_winner_share>0.45")
        if not np.isfinite(fee_2x) or fee_2x <= -0.0005:
            failed.append("fee_2x_mean_net<=-0.0005")
        if not np.isfinite(delay_1) or delay_1 <= -0.0005:
            failed.append("delay_1bar_mean_net<=-0.0005")
        decision = "research_candidate" if not failed else "rejected"
        out = {
            "event_name": event_name,
            "family": family,
            "direction": direction,
            "primary_horizon": int(primary_horizon),
            "count": count,
            "mean_net": mean_net,
            "median_net": row.get("median_net", np.nan),
            "win_rate": row.get("win_rate", np.nan),
            "profit_factor": row.get("profit_factor", np.nan),
            "mfe_mean": float(pd.to_numeric(part.get(mfe_col, pd.Series(dtype=float)), errors="coerce").mean()) if mfe_col in part.columns else np.nan,
            "mae_mean": float(pd.to_numeric(part.get(mae_col, pd.Series(dtype=float)), errors="coerce").mean()) if mae_col in part.columns else np.nan,
            "positive_years": positive_years,
            "year_count": year_count,
            "session_best": session_best,
            "max_days_without_event": max_days_without_event(part["signal_time"], start_date, end_date),
            "fee_2x_mean_net": fee_2x,
            "delay_1bar_mean_net": delay_1,
            "top5_winner_share": row.get("top5_winner_share", np.nan),
            "decision": decision,
            "reason": "pass_v1_research_filters" if not failed else ";".join(failed),
            "failed_rules": ";".join(failed),
        }
        rows.append(out)
    all_decisions = pd.DataFrame(rows).sort_values(["decision", "mean_net", "profit_factor"], ascending=[True, False, False], kind="stable")
    candidates = all_decisions.loc[all_decisions["decision"] == "research_candidate"].copy()
    rejected = all_decisions.loc[all_decisions["decision"] != "research_candidate"].copy()
    return candidates, rejected


def build_causal_audit(events: pd.DataFrame, bars: pd.DataFrame, primary_timeframe: str, primary_horizon: int, context_timeframes: tuple[str, ...]) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "event_name", "direction", "signal_time", "entry_time", "expected_entry_time", "entry_not_next_open_flag",
                "used_context_time", "used_context_available_time", "context_available_time_flag", "forward_window_valid_flag", "lookahead_flag",
            ]
        )
    out = pd.DataFrame(index=events.index)
    out["event_name"] = events["event_name"].values
    out["direction"] = events["direction"].values
    out["signal_time"] = pd.to_datetime(events["signal_time"], errors="coerce")
    signal_idx = pd.to_numeric(events["signal_idx"], errors="coerce").fillna(-1).astype(int).to_numpy()
    expected_entry_idx = signal_idx + 1
    index = pd.DatetimeIndex(bars.index)
    expected = np.full(len(events), np.datetime64("NaT"), dtype="datetime64[ns]")
    in_range = (expected_entry_idx >= 0) & (expected_entry_idx < len(index))
    expected[in_range] = index.values[expected_entry_idx[in_range]]
    out["entry_time"] = pd.to_datetime(events.get("delay0_entry_time", pd.Series(pd.NaT, index=events.index)), errors="coerce")
    out["expected_entry_time"] = expected
    out["entry_not_next_open_flag"] = out["entry_time"] != pd.to_datetime(out["expected_entry_time"], errors="coerce")

    ctx_start_cols: list[pd.Series] = []
    ctx_available_cols: list[pd.Series] = []
    for tf in context_timeframes:
        st_col = f"ctx_{tf}_bar_start_time"
        av_col = f"ctx_{tf}_available_time"
        if st_col in events.columns:
            out[f"used_{tf}_context_time"] = pd.to_datetime(events[st_col], errors="coerce")
            ctx_start_cols.append(out[f"used_{tf}_context_time"])
        if av_col in events.columns:
            out[f"used_{tf}_context_available_time"] = pd.to_datetime(events[av_col], errors="coerce")
            ctx_available_cols.append(out[f"used_{tf}_context_available_time"])
            out[f"{tf}_context_available_time_flag"] = out[f"used_{tf}_context_available_time"].notna() & (out[f"used_{tf}_context_available_time"] > out["signal_time"])
    out["used_context_time"] = _max_timestamp_frame(ctx_start_cols).reindex(out.index) if ctx_start_cols else pd.NaT
    out["used_context_available_time"] = _max_timestamp_frame(ctx_available_cols).reindex(out.index) if ctx_available_cols else pd.NaT
    out["context_available_time_flag"] = out["used_context_available_time"].notna() & (out["used_context_available_time"] > out["signal_time"])
    valid_col = f"delay0_h{primary_horizon}_valid"
    out["forward_window_valid_flag"] = events[valid_col].astype(bool).values if valid_col in events.columns else False
    out["lookahead_flag"] = out["entry_not_next_open_flag"].astype(bool) | out["context_available_time_flag"].astype(bool)
    out["primary_timeframe"] = primary_timeframe
    return out.reset_index(drop=True)


def write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame, columns: list[str] | None = None) -> None:
    if columns is not None:
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[columns]
    df.to_csv(path, index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    horizons = _parse_csv_ints(args.horizons)
    cost_multipliers = _parse_csv_floats(args.cost_multipliers)
    delay_bars_list = _parse_csv_ints(args.delay_bars_list)
    context_timeframes = _parse_csv_strings(args.context_timeframes)
    primary_horizon = int(args.primary_horizon)
    if primary_horizon not in horizons:
        horizons = tuple(sorted(set([*horizons, primary_horizon])))
    build_missing = not bool(args.no_build_missing)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] {SCRIPT_NAME} version={SCRIPT_VERSION}", flush=True)
    print(
        f"[run] symbol={args.symbol} primary={args.primary_timeframe} context={','.join(context_timeframes)} "
        f"range={args.start_date}->{args.end_date} warmup={args.warmup_start_date}",
        flush=True,
    )

    print("[load] primary trade bars", flush=True)
    primary_raw = load_trade_bars(
        symbol=args.symbol,
        timeframe=args.primary_timeframe,
        start_date=args.warmup_start_date,
        end_date=args.end_date,
        chunksize=int(args.chunksize),
        data_dir=args.data_dir,
        db_name=args.db_name,
        build_missing=build_missing,
        force_rebuild=bool(args.force_rebuild),
    )
    if primary_raw.empty:
        raise RuntimeError("primary trade bars are empty; prebuild or enable build_missing first")
    print(f"[load] primary rows={len(primary_raw):,} range={primary_raw.index.min()} -> {primary_raw.index.max()}", flush=True)

    contexts: dict[str, pd.DataFrame] = {}
    if context_timeframes:
        progress = ProgressReporter(label="[load] context trade bars", total=len(context_timeframes), every=1)
        for i, tf in enumerate(context_timeframes, start=1):
            ctx = load_trade_bars(
                symbol=args.symbol,
                timeframe=tf,
                start_date=args.warmup_start_date,
                end_date=args.end_date,
                chunksize=int(args.chunksize),
                data_dir=args.data_dir,
                db_name=args.db_name,
                build_missing=build_missing,
                force_rebuild=bool(args.force_rebuild),
            )
            contexts[tf] = ctx
            print(f"[load] context {tf} rows={len(ctx):,}", flush=True)
            progress.update(i)
        progress.close()

    print("[features] primary features", flush=True)
    primary_features, primary_feature_columns = build_primary_features(primary_raw)
    print(f"[features] primary feature columns={len(primary_feature_columns)}", flush=True)

    print("[features] causal context alignment", flush=True)
    feature_frame, context_feature_columns = causal_attach_context(primary_features, contexts)
    print(f"[features] context feature columns={len(context_feature_columns)}", flush=True)

    print("[events] building event masks", flush=True)
    specs = build_event_specs()
    events = build_events(
        feature_frame,
        specs,
        start_date=args.start_date,
        end_date=args.end_date,
        cooldown_bars=int(args.cooldown_bars),
        progress_every=int(args.progress_every),
    )
    print(f"[events] rows={len(events):,} families={len(set(s.family for s in specs))} specs={len(specs)}", flush=True)

    print("[forward] base cost/delay outcomes", flush=True)
    events = attach_forward_returns(
        events,
        primary_raw,
        horizons=horizons,
        mfe_mae_horizon=int(args.mfe_mae_horizon),
        round_trip_cost_pct=float(args.round_trip_cost_pct),
        cost_multiplier=1.0,
        delay_bars=0,
    )

    print("[aggregate] forward matrix", flush=True)
    forward_matrix = build_forward_return_matrix(events, horizons, cost_multiplier=1.0, delay_bars=0)
    family_summary = summarize_by_groups(events, group_cols=["family", "direction"], return_col=f"delay0_h{primary_horizon}_net_c1")
    if not family_summary.empty:
        family_summary.insert(2, "primary_horizon", int(primary_horizon))
    event_counts = build_event_counts(events, args.start_date, args.end_date)
    event_name_stats = forward_matrix.loc[forward_matrix["horizon"] == int(primary_horizon)].copy() if not forward_matrix.empty else pd.DataFrame()

    yearly = summarize_by_groups(events, group_cols=["event_name", "family", "direction", "year"], return_col=f"delay0_h{primary_horizon}_net_c1")
    sessions = summarize_by_groups(events, group_cols=["event_name", "family", "direction", "session"], return_col=f"delay0_h{primary_horizon}_net_c1")
    regimes = summarize_by_groups(events, group_cols=["event_name", "family", "direction", "regime"], return_col=f"delay0_h{primary_horizon}_net_c1")

    print("[aggregate] cost stress", flush=True)
    cost_stress = build_cost_stress(events, horizons, cost_multipliers, float(args.round_trip_cost_pct))
    print("[aggregate] delay stress", flush=True)
    delay_stress, _delayed = build_delay_stress(events, primary_raw, horizons, delay_bars_list, float(args.round_trip_cost_pct))
    print("[aggregate] candidate filters", flush=True)
    candidates, rejected = build_candidate_tables(
        events,
        forward_matrix,
        cost_stress,
        delay_stress,
        primary_horizon=primary_horizon,
        min_count=int(args.min_count),
        start_date=args.start_date,
        end_date=args.end_date,
        mfe_mae_horizon=int(args.mfe_mae_horizon),
    )
    causal_audit = build_causal_audit(events, primary_raw, args.primary_timeframe, primary_horizon, context_timeframes)

    print("[write] report files", flush=True)
    manifest = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "symbol": args.symbol,
        "primary_timeframe": args.primary_timeframe,
        "context_timeframes": list(context_timeframes),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "horizons": list(horizons),
        "primary_horizon": int(primary_horizon),
        "mfe_mae_horizon": int(args.mfe_mae_horizon),
        "round_trip_cost_pct": float(args.round_trip_cost_pct),
        "cost_multipliers": list(cost_multipliers),
        "delay_bars_list": list(delay_bars_list),
        "input_rows": int(len(primary_raw)),
        "event_family_count": int(len(set(s.family for s in specs))),
        "event_spec_count": int(len(specs)),
        "event_count": int(len(events)),
        "candidate_count": int(len(candidates)),
        "causal_policy": CAUSAL_POLICY,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
    write_json(out_dir / "00_manifest.json", manifest)
    write_csv(out_dir / "01_event_family_summary.csv", family_summary)
    write_csv(out_dir / "02_event_counts.csv", event_counts)
    write_csv(out_dir / "03_forward_return_matrix.csv", forward_matrix)
    write_csv(out_dir / "04_event_name_stats.csv", event_name_stats)
    write_csv(out_dir / "05_yearly_breakdown.csv", yearly)
    write_csv(out_dir / "06_session_breakdown.csv", sessions)
    write_csv(out_dir / "07_regime_breakdown.csv", regimes)
    write_csv(out_dir / "08_cost_stress.csv", cost_stress)
    write_csv(out_dir / "09_delay_stress.csv", delay_stress)
    candidate_cols = [
        "event_name", "family", "direction", "primary_horizon", "count", "mean_net", "median_net", "win_rate",
        "profit_factor", "mfe_mean", "mae_mean", "positive_years", "year_count", "session_best",
        "max_days_without_event", "fee_2x_mean_net", "delay_1bar_mean_net", "top5_winner_share", "decision", "reason",
    ]
    write_csv(out_dir / "10_candidate_shortlist.csv", candidates, columns=candidate_cols)
    write_csv(out_dir / "11_rejected_event_families.csv", rejected)
    write_csv(out_dir / "12_causal_audit.csv", causal_audit)
    sample_n = int(args.event_sample_size)
    event_sample = events.head(sample_n).copy() if sample_n > 0 else events.head(0).copy()
    write_csv(out_dir / "13_event_sample.csv", event_sample)
    write_json(
        out_dir / "14_feature_columns.json",
        {
            "primary_feature_columns": primary_feature_columns,
            "context_feature_columns": context_feature_columns,
            "event_columns": list(events.columns),
            "causal_policy": CAUSAL_POLICY,
        },
    )
    if bool(args.write_full_events):
        write_csv(out_dir / "15_events_with_forward_returns.csv", events)

    print("[review-pack] finalize", flush=True)
    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title="ETH MHF Edge Factory V1")
    print(f"[done] report_dir={out_dir}", flush=True)
    return 0


def build_event_counts(events: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if events.empty:
        return pd.DataFrame(columns=["event_name", "family", "direction", "count", "first_signal", "last_signal", "max_days_without_event"])
    for key, part in events.groupby(["event_name", "family", "direction"], dropna=False, sort=True):
        rows.append(
            {
                "event_name": key[0],
                "family": key[1],
                "direction": key[2],
                "count": int(len(part)),
                "first_signal": pd.to_datetime(part["signal_time"]).min(),
                "last_signal": pd.to_datetime(part["signal_time"]).max(),
                "max_days_without_event": max_days_without_event(part["signal_time"], start_date, end_date),
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "direction", "event_name"], kind="stable")


if __name__ == "__main__":
    raise SystemExit(main())
