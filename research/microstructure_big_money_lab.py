#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH Microstructure Big Money Lab V1.

Four independent microstructure event studies using CoinBacktest local data:

1. Market Impact Lab
   - initiative buying/selling
   - absorbed buying/selling
   - price-impact / delta divergence
2. Absorption Zone Lab
   - repeated low/high tests with CVD divergence and reclaim/reject
3. Liquidity Vacuum Lab
   - price moves fast / far with thin traded liquidity proxy
   - range-bar speed bursts when range data is enabled
4. Lead-Lag Lab
   - optional BTC/other symbol leading ETH using locally cached trade bars

This is an event-study/research script, not a production strategy.  Outcomes are
computed with closed bar signal -> next primary bar open entry -> fixed-horizon
exit using already closed bars.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import warnings
from pandas.errors import PerformanceWarning
warnings.simplefilter("ignore", PerformanceWarning)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader
from src.research_common.progress import ProgressReporter

DEFAULT_HORIZONS = (5, 15, 30, 60, 120, 240)
EPS = 1e-12


def parse_list_int(text: str) -> list[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_list_str(text: str) -> list[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def pct_rank(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    """Rolling percentile rank of latest value inside the trailing window."""
    min_p = min_periods or max(20, window // 5)

    def _rank_last(x: np.ndarray) -> float:
        if len(x) == 0 or not np.isfinite(x[-1]):
            return np.nan
        v = x[-1]
        valid = x[np.isfinite(x)]
        if len(valid) == 0:
            return np.nan
        return float((valid <= v).mean())

    return s.rolling(window, min_periods=min_p).apply(_rank_last, raw=True)


def rolling_z(s: pd.Series, window: int, min_periods: int | None = None) -> pd.Series:
    min_p = min_periods or max(20, window // 5)
    mu = s.rolling(window, min_periods=min_p).mean()
    sd = s.rolling(window, min_periods=min_p).std(ddof=0)
    return (s - mu) / sd.replace(0, np.nan)


def profit_factor(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if vals.empty:
        return np.nan
    gross_profit = vals[vals > 0].sum()
    gross_loss = -vals[vals < 0].sum()
    if gross_loss <= 0:
        return np.inf if gross_profit > 0 else np.nan
    return float(gross_profit / gross_loss)


def top5_winner_share(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    gains = vals[vals > 0].sort_values(ascending=False)
    total = gains.sum()
    if total <= 0 or gains.empty:
        return 0.0
    return float(gains.head(5).sum() / total)


def summarize_group(df: pd.DataFrame, by: list[str], metric: str) -> pd.DataFrame:
    if df.empty or metric not in df.columns:
        return pd.DataFrame()
    g = df.groupby(by, dropna=False)
    rows = []
    for key, sub in g:
        vals = pd.to_numeric(sub[metric], errors="coerce").dropna()
        if vals.empty:
            continue
        if not isinstance(key, tuple):
            key = (key,)
        row = {col: val for col, val in zip(by, key)}
        row.update(
            {
                "count": int(vals.count()),
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "win_rate": float((vals > 0).mean()),
                "profit_factor": profit_factor(vals),
                "q25": float(vals.quantile(0.25)),
                "q75": float(vals.quantile(0.75)),
                "top5_winner_share": top5_winner_share(vals),
            }
        )
        if "mfe_" + metric.split("next_open_ret_")[-1].replace("_net", "") in sub.columns:
            mfe_col = "mfe_" + metric.split("next_open_ret_")[-1].replace("_net", "")
            mae_col = "mae_" + metric.split("next_open_ret_")[-1].replace("_net", "")
            row["mfe_mean"] = float(pd.to_numeric(sub[mfe_col], errors="coerce").mean())
            row["mae_mean"] = float(pd.to_numeric(sub[mae_col], errors="coerce").mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["mean", "profit_factor", "count"], ascending=[False, False, False])
    return out


def load_trade_bars(symbol: str, timeframe: str, start_date: str, end_date: str, *, warmup_start_date: str | None = None, build_missing: bool = True) -> pd.DataFrame:
    start = warmup_start_date or start_date
    loader = OKXTradeBarLoader(symbol=symbol, timeframe=timeframe)
    df = loader.fetch_data_by_date_range(start, end_date, build_missing=build_missing)
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.reset_index(drop=True).copy()
    if "timestamp" not in out.columns:
        # OKXTradeBarLoader usually returns timestamp as index only.
        idx_name = df.index.name or "timestamp"
        out[idx_name] = df.index.to_numpy()
        if idx_name != "timestamp":
            out = out.rename(columns={idx_name: "timestamp"})
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return out


def load_range_bars(symbol: str, range_pct: float, start_date: str, end_date: str, *, warmup_start_date: str | None = None) -> pd.DataFrame:
    start = warmup_start_date or start_date
    loader = OKXRangeBarLoader(symbol=symbol, range_pct=range_pct)
    df = loader.fetch_data_by_date_range(start, end_date)
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.reset_index(drop=True).copy()
    out["start_ts"] = pd.to_datetime(out["start_ts"], errors="coerce")
    out["end_ts"] = pd.to_datetime(out["end_ts"], errors="coerce")
    out["bar_id"] = pd.to_numeric(out["bar_id"], errors="coerce").fillna(0).astype("int64")
    out = out.dropna(subset=["start_ts", "end_ts"]).sort_values(["end_ts", "bar_id"]).reset_index(drop=True)
    return out


def load_footprint(symbol: str, range_pct: float, price_step: float, start_date: str, end_date: str, *, warmup_start_date: str | None = None) -> pd.DataFrame:
    start = warmup_start_date or start_date
    loader = OKXRangeFootprintLoader(symbol=symbol, range_pct=range_pct, price_step=price_step)
    df = loader.fetch_data_by_date_range(start, end_date)
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.reset_index(drop=True).copy()
    out["bar_id"] = pd.to_numeric(out["bar_id"], errors="coerce").fillna(0).astype("int64")
    out["end_ts"] = pd.to_datetime(out["end_ts"], errors="coerce")
    return out.dropna(subset=["end_ts"])


def build_primary_features(df: pd.DataFrame, start_date: str) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = [c for c in out.columns if c not in {"timestamp"}]
    for c in numeric_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out["ret_1"] = out["close"].pct_change().fillna(0.0)
    out["bar_ret"] = out["close"] / out["open"].replace(0, np.nan) - 1.0
    out["hl_range_pct"] = (out["high"] - out["low"]) / out["open"].replace(0, np.nan)
    out["body_to_range"] = (out["close"] - out["open"]).abs() / (out["high"] - out["low"]).replace(0, np.nan)
    out["close_pos"] = (out["close"] - out["low"]) / (out["high"] - out["low"]).replace(0, np.nan)

    # Use columns if present; fill safely.
    for c in [
        "notional",
        "delta_notional",
        "large_delta_notional",
        "large_trades_count",
        "max_trade_notional",
        "taker_buy_ratio",
        "trades_count",
        "buy_notional",
        "sell_notional",
    ]:
        if c not in out.columns:
            out[c] = 0.0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    out["abs_delta_notional"] = out["delta_notional"].abs()
    out["abs_large_delta_notional"] = out["large_delta_notional"].abs()
    out["delta_ratio"] = out["delta_notional"] / out["notional"].replace(0, np.nan)
    out["large_delta_ratio"] = out["large_delta_notional"] / out["notional"].replace(0, np.nan)
    out["signed_ret_by_delta"] = out["ret_1"] * np.sign(out["delta_notional"].replace(0, np.nan))
    out["impact_per_million"] = out["ret_1"].abs() / (out["abs_delta_notional"] / 1_000_000.0).replace(0, np.nan)
    out["impact_signed"] = out["ret_1"] / (out["delta_notional"] / 1_000_000.0).replace(0, np.nan)
    out["liquidity_thin_proxy"] = out["ret_1"].abs() / (out["notional"] / 1_000_000.0).replace(0, np.nan)

    for w in [5, 15, 30, 60, 120, 240]:
        out[f"ret_{w}"] = out["close"].pct_change(w)
        out[f"delta_sum_{w}"] = out["delta_notional"].rolling(w, min_periods=max(3, w // 4)).sum()
        out[f"notional_sum_{w}"] = out["notional"].rolling(w, min_periods=max(3, w // 4)).sum()
        out[f"abs_delta_sum_{w}"] = out["abs_delta_notional"].rolling(w, min_periods=max(3, w // 4)).sum()
        out[f"range_sum_{w}"] = out["hl_range_pct"].rolling(w, min_periods=max(3, w // 4)).sum()
        out[f"cvd_change_{w}"] = out.get("cvd_notional", out["delta_notional"].cumsum()).diff(w)
        out[f"realized_vol_{w}"] = out["ret_1"].rolling(w, min_periods=max(3, w // 4)).std(ddof=0)

    # Rolling z / percentiles: fixed windows to avoid parameter fishing.
    z_window = 240
    for c in ["notional", "delta_notional", "abs_delta_notional", "large_delta_notional", "abs_large_delta_notional", "max_trade_notional", "trades_count", "impact_per_million", "liquidity_thin_proxy", "hl_range_pct"]:
        out[f"{c}_z"] = rolling_z(out[c].replace([np.inf, -np.inf], np.nan), z_window)
        out[f"{c}_pct_rank"] = pct_rank(out[c].replace([np.inf, -np.inf], np.nan), z_window)

    out["ema_60"] = out["close"].ewm(span=60, adjust=False, min_periods=20).mean()
    out["ema_240"] = out["close"].ewm(span=240, adjust=False, min_periods=60).mean()
    out["above_ema60"] = out["close"] > out["ema_60"]
    out["above_ema240"] = out["close"] > out["ema_240"]
    out["trend_up_240"] = (out["close"] > out["ema_240"]) & (out["ema_240"].diff(30) > 0)
    out["trend_down_240"] = (out["close"] < out["ema_240"]) & (out["ema_240"].diff(30) < 0)

    out["hour"] = out["timestamp"].dt.hour
    out["date"] = out["timestamp"].dt.date.astype(str)
    out["in_us_open"] = out["hour"].isin([21, 22, 23, 0])
    out["in_asia"] = out["hour"].between(8, 15)
    out["in_europe"] = out["hour"].between(15, 21)

    out["in_sample"] = out["timestamp"] >= pd.Timestamp(start_date)
    return out


@dataclass(frozen=True)
class EventSpec:
    name: str
    family: str
    side: str  # long or short
    mask_col: str
    description: str


def add_market_impact_masks(f: pd.DataFrame) -> list[EventSpec]:
    specs: list[EventSpec] = []
    # initiative = strong delta + price moves with it + high impact
    f["mi_initiative_buy"] = (f["delta_notional_z"] > 1.5) & (f["ret_1"] > 0) & (f["impact_per_million_pct_rank"] > 0.70) & (f["close_pos"] > 0.65)
    f["mi_initiative_sell"] = (f["delta_notional_z"] < -1.5) & (f["ret_1"] < 0) & (f["impact_per_million_pct_rank"] > 0.70) & (f["close_pos"] < 0.35)
    specs += [
        EventSpec("initiative_buying", "market_impact", "long", "mi_initiative_buy", "Strong active buy delta creates upside price impact."),
        EventSpec("initiative_selling", "market_impact", "short", "mi_initiative_sell", "Strong active sell delta creates downside price impact."),
    ]
    # absorbed = strong active pressure but close rejects direction.
    f["mi_absorbed_selling"] = (f["delta_notional_z"] < -1.5) & (f["bar_ret"] >= -0.0005) & (f["close_pos"] > 0.60) & (f["impact_per_million_pct_rank"] < 0.45)
    f["mi_absorbed_buying"] = (f["delta_notional_z"] > 1.5) & (f["bar_ret"] <= 0.0005) & (f["close_pos"] < 0.40) & (f["impact_per_million_pct_rank"] < 0.45)
    specs += [
        EventSpec("absorbed_selling", "market_impact", "long", "mi_absorbed_selling", "Large sell pressure does not push price lower; close rejects low."),
        EventSpec("absorbed_buying", "market_impact", "short", "mi_absorbed_buying", "Large buy pressure does not push price higher; close rejects high."),
    ]
    # exhaustion: large cumulative delta opposed to price over 15m/30m.
    f["mi_bullish_cvd_divergence_30"] = (f["ret_30"] > -0.001) & (f["cvd_change_30"] < -f["abs_delta_notional"].rolling(240, min_periods=60).median()) & (f["close"] > f["low"].rolling(30, min_periods=10).min() * 1.001)
    f["mi_bearish_cvd_divergence_30"] = (f["ret_30"] < 0.001) & (f["cvd_change_30"] > f["abs_delta_notional"].rolling(240, min_periods=60).median()) & (f["close"] < f["high"].rolling(30, min_periods=10).max() * 0.999)
    specs += [
        EventSpec("bullish_cvd_divergence_30m", "market_impact", "long", "mi_bullish_cvd_divergence_30", "CVD falls while price refuses to make/hold new lows."),
        EventSpec("bearish_cvd_divergence_30m", "market_impact", "short", "mi_bearish_cvd_divergence_30", "CVD rises while price refuses to make/hold new highs."),
    ]
    return specs


def add_absorption_zone_masks(f: pd.DataFrame) -> list[EventSpec]:
    specs: list[EventSpec] = []
    # Repeated low tests over 30/60 bars, large sell delta, close reclaim.
    for w in [30, 60]:
        low_roll = f["low"].rolling(w, min_periods=max(10, w // 3)).min()
        high_roll = f["high"].rolling(w, min_periods=max(10, w // 3)).max()
        # Count touches within 8bp of rolling extremes.
        low_touch = f["low"] <= low_roll.shift(1) * 1.0008
        high_touch = f["high"] >= high_roll.shift(1) * 0.9992
        f[f"az_low_touches_{w}"] = low_touch.rolling(w, min_periods=max(10, w // 3)).sum()
        f[f"az_high_touches_{w}"] = high_touch.rolling(w, min_periods=max(10, w // 3)).sum()
        f[f"az_sell_absorption_{w}"] = (
            (f[f"az_low_touches_{w}"] >= 3)
            & (f[f"delta_sum_{w}"] < 0)
            & (f[f"cvd_change_{w}"] < 0)
            & (f["close"] > low_roll * 1.001)
            & (f["close_pos"] > 0.55)
        )
        f[f"az_buy_absorption_{w}"] = (
            (f[f"az_high_touches_{w}"] >= 3)
            & (f[f"delta_sum_{w}"] > 0)
            & (f[f"cvd_change_{w}"] > 0)
            & (f["close"] < high_roll * 0.999)
            & (f["close_pos"] < 0.45)
        )
        specs += [
            EventSpec(f"sell_absorption_zone_{w}m", "absorption_zone", "long", f"az_sell_absorption_{w}", f"Repeated low tests over {w}m with sell CVD absorbed and reclaim."),
            EventSpec(f"buy_absorption_zone_{w}m", "absorption_zone", "short", f"az_buy_absorption_{w}", f"Repeated high tests over {w}m with buy CVD absorbed and reject."),
        ]
    # Stop-run reclaim/reject around local extremes.
    for w in [30, 60, 120]:
        prior_low = f["low"].rolling(w, min_periods=max(10, w // 3)).min().shift(1)
        prior_high = f["high"].rolling(w, min_periods=max(10, w // 3)).max().shift(1)
        f[f"az_stop_run_reclaim_{w}"] = (f["low"] < prior_low * 0.9995) & (f["close"] > prior_low) & (f["delta_notional"] < 0) & (f["close_pos"] > 0.60)
        f[f"az_stop_run_reject_{w}"] = (f["high"] > prior_high * 1.0005) & (f["close"] < prior_high) & (f["delta_notional"] > 0) & (f["close_pos"] < 0.40)
        specs += [
            EventSpec(f"stop_run_reclaim_low_{w}m", "absorption_zone", "long", f"az_stop_run_reclaim_{w}", f"Sweep prior {w}m low, active selling, close back above low."),
            EventSpec(f"stop_run_reject_high_{w}m", "absorption_zone", "short", f"az_stop_run_reject_{w}", f"Sweep prior {w}m high, active buying, close back below high."),
        ]
    return specs


def add_liquidity_vacuum_masks(f: pd.DataFrame) -> list[EventSpec]:
    specs: list[EventSpec] = []
    f["lv_vacuum_up"] = (
        (f["ret_1"] > 0)
        & (f["hl_range_pct_z"] > 1.3)
        & (f["liquidity_thin_proxy_pct_rank"] > 0.80)
        & (f["notional_pct_rank"] < 0.70)
        & (f["close_pos"] > 0.75)
    )
    f["lv_vacuum_down"] = (
        (f["ret_1"] < 0)
        & (f["hl_range_pct_z"] > 1.3)
        & (f["liquidity_thin_proxy_pct_rank"] > 0.80)
        & (f["notional_pct_rank"] < 0.70)
        & (f["close_pos"] < 0.25)
    )
    specs += [
        EventSpec("liquidity_vacuum_up", "liquidity_vacuum", "long", "lv_vacuum_up", "Large up move with thin-liquidity price impact proxy."),
        EventSpec("liquidity_vacuum_down", "liquidity_vacuum", "short", "lv_vacuum_down", "Large down move with thin-liquidity price impact proxy."),
    ]
    f["lv_low_volume_breakout_up"] = (f["ret_5"] > f["realized_vol_60"] * 1.2) & (f["notional_pct_rank"] < 0.50) & (f["close"] > f["high"].rolling(30, min_periods=10).max().shift(1))
    f["lv_low_volume_breakout_down"] = (f["ret_5"] < -f["realized_vol_60"] * 1.2) & (f["notional_pct_rank"] < 0.50) & (f["close"] < f["low"].rolling(30, min_periods=10).min().shift(1))
    specs += [
        EventSpec("low_liquidity_breakout_up_5m", "liquidity_vacuum", "long", "lv_low_volume_breakout_up", "5m breakout with low notional percentile, proxy for liquidity withdrawal."),
        EventSpec("low_liquidity_breakout_down_5m", "liquidity_vacuum", "short", "lv_low_volume_breakout_down", "5m breakdown with low notional percentile, proxy for liquidity withdrawal."),
    ]
    return specs


def add_range_context_features(f: pd.DataFrame, rb: pd.DataFrame) -> tuple[pd.DataFrame, list[EventSpec]]:
    specs: list[EventSpec] = []
    if rb.empty:
        return f, specs
    r = rb.copy().reset_index(drop=True)
    r["end_ts"] = pd.to_datetime(r["end_ts"], errors="coerce")
    r = r.dropna(subset=["end_ts"]).sort_values("end_ts")
    for c in ["duration_seconds", "direction", "notional", "delta_notional", "trades_count", "large_delta_notional"]:
        if c not in r.columns:
            r[c] = 0.0
        r[c] = pd.to_numeric(r[c], errors="coerce").fillna(0.0)
    r["rb_fast"] = r["duration_seconds"] <= r["duration_seconds"].rolling(1000, min_periods=100).quantile(0.15)
    r["rb_notional_pct"] = pct_rank(r["notional"], 1000, 100)
    r["rb_thin_fast"] = r["rb_fast"] & (r["rb_notional_pct"] < 0.70)
    r["rb_delta_aligned"] = np.sign(r["delta_notional"]) == np.sign(r["direction"])
    r["rb_dir_run3"] = (r["direction"].rolling(3, min_periods=3).sum()).fillna(0)
    ctx = r[["end_ts", "direction", "duration_seconds", "rb_fast", "rb_thin_fast", "rb_delta_aligned", "rb_dir_run3", "notional", "delta_notional"]].rename(
        columns={
            "end_ts": "rb_available_time",
            "direction": "rb_direction",
            "duration_seconds": "rb_duration_seconds",
            "notional": "rb_notional",
            "delta_notional": "rb_delta_notional",
        }
    )
    # Range bars are closed at end_ts; available at end_ts. Align backward to 1m signal time.
    merged = pd.merge_asof(
        f.sort_values("timestamp"),
        ctx.sort_values("rb_available_time"),
        left_on="timestamp",
        right_on="rb_available_time",
        direction="backward",
    )
    merged["lv_range_fast_up"] = (merged["rb_direction"] > 0) & merged["rb_thin_fast"].fillna(False)
    merged["lv_range_fast_down"] = (merged["rb_direction"] < 0) & merged["rb_thin_fast"].fillna(False)
    merged["lv_range_run3_up"] = merged["rb_dir_run3"] >= 3
    merged["lv_range_run3_down"] = merged["rb_dir_run3"] <= -3
    specs += [
        EventSpec("range_thin_fast_up", "liquidity_vacuum", "long", "lv_range_fast_up", "Latest 0.2% range bar is fast and thin, up direction."),
        EventSpec("range_thin_fast_down", "liquidity_vacuum", "short", "lv_range_fast_down", "Latest 0.2% range bar is fast and thin, down direction."),
        EventSpec("range_speed_run3_up", "liquidity_vacuum", "long", "lv_range_run3_up", "Latest range context shows 3 consecutive upward range bars."),
        EventSpec("range_speed_run3_down", "liquidity_vacuum", "short", "lv_range_run3_down", "Latest range context shows 3 consecutive downward range bars."),
    ]
    return merged, specs


def add_footprint_context_features(f: pd.DataFrame, fp: pd.DataFrame) -> tuple[pd.DataFrame, list[EventSpec]]:
    specs: list[EventSpec] = []
    if fp.empty:
        return f, specs
    p = fp.copy()
    for c in ["bar_id", "price_bucket", "delta_notional", "large_delta_notional", "notional", "buy_notional", "sell_notional", "trades_count", "end_ts"]:
        if c not in p.columns:
            return f, specs
    p["end_ts"] = pd.to_datetime(p["end_ts"], errors="coerce")
    p = p.dropna(subset=["end_ts"])
    p["delta_notional"] = pd.to_numeric(p["delta_notional"], errors="coerce").fillna(0.0)
    p["large_delta_notional"] = pd.to_numeric(p["large_delta_notional"], errors="coerce").fillna(0.0)
    p["notional"] = pd.to_numeric(p["notional"], errors="coerce").fillna(0.0)
    p["price_bucket"] = pd.to_numeric(p["price_bucket"], errors="coerce").fillna(0.0)
    # Aggregate per range bar: top/bottom bucket pressure and stacked imbalance proxy.
    p = p.sort_values(["bar_id", "price_bucket"])
    grp = p.groupby("bar_id", sort=False)
    agg = grp.agg(
        fp_end_ts=("end_ts", "max"),
        fp_total_delta=("delta_notional", "sum"),
        fp_total_large_delta=("large_delta_notional", "sum"),
        fp_total_notional=("notional", "sum"),
        fp_bucket_count=("price_bucket", "count"),
    ).reset_index()
    # Top/bottom 30% by bucket within each bar. Kept vectorized enough for research.
    p["rank_pct"] = grp.cumcount() / grp["price_bucket"].transform("count").sub(1).replace(0, 1)
    bottom = p[p["rank_pct"] <= 0.30].groupby("bar_id").agg(fp_bottom_delta=("delta_notional", "sum"), fp_bottom_notional=("notional", "sum")).reset_index()
    top = p[p["rank_pct"] >= 0.70].groupby("bar_id").agg(fp_top_delta=("delta_notional", "sum"), fp_top_notional=("notional", "sum")).reset_index()
    agg = agg.merge(bottom, on="bar_id", how="left").merge(top, on="bar_id", how="left").fillna(0.0)
    agg["fp_bottom_absorb_sell"] = (agg["fp_bottom_delta"] < 0) & (agg["fp_total_delta"] < 0)
    agg["fp_top_absorb_buy"] = (agg["fp_top_delta"] > 0) & (agg["fp_total_delta"] > 0)
    agg["fp_delta_ratio"] = agg["fp_total_delta"] / agg["fp_total_notional"].replace(0, np.nan)
    agg = agg.sort_values("fp_end_ts")
    merged = pd.merge_asof(
        f.sort_values("timestamp"),
        agg.sort_values("fp_end_ts"),
        left_on="timestamp",
        right_on="fp_end_ts",
        direction="backward",
    )
    merged["fp_absorbed_selling_context"] = merged["fp_bottom_absorb_sell"].fillna(False) & (merged["close_pos"] > 0.50)
    merged["fp_absorbed_buying_context"] = merged["fp_top_absorb_buy"].fillna(False) & (merged["close_pos"] < 0.50)
    specs += [
        EventSpec("footprint_bottom_sell_absorption", "absorption_zone", "long", "fp_absorbed_selling_context", "Latest range footprint has bottom sell pressure absorbed/rejected."),
        EventSpec("footprint_top_buy_absorption", "absorption_zone", "short", "fp_absorbed_buying_context", "Latest range footprint has top buy pressure absorbed/rejected."),
    ]
    return merged, specs


def add_lead_lag_features(f: pd.DataFrame, lead_df: pd.DataFrame, lead_symbol: str) -> tuple[pd.DataFrame, list[EventSpec]]:
    specs: list[EventSpec] = []
    if lead_df.empty:
        return f, specs
    lead = build_primary_features(lead_df, start_date=str(f["timestamp"].min().date()))
    lead_cols = ["timestamp", "ret_1", "ret_5", "ret_15", "ret_30", "delta_notional", "notional", "large_delta_notional"]
    lead = lead[[c for c in lead_cols if c in lead.columns]].copy()
    prefix = lead_symbol.replace("-", "_").replace("/", "_").lower()
    lead = lead.rename(columns={c: f"lead_{prefix}_{c}" for c in lead.columns if c != "timestamp"})
    # Shift lead by 1 primary bar: lead signal must already be closed before ETH signal bar.
    for c in lead.columns:
        if c != "timestamp":
            lead[c] = lead[c].shift(1)
    m = pd.merge_asof(f.sort_values("timestamp"), lead.sort_values("timestamp"), on="timestamp", direction="backward")
    ret5 = f"lead_{prefix}_ret_5"
    ret15 = f"lead_{prefix}_ret_15"
    dcol = f"lead_{prefix}_delta_notional"
    if ret5 in m.columns:
        m[f"ll_{prefix}_lead_up_eth_lag"] = (m[ret5] > 0.002) & (m["ret_5"].abs() < 0.0015)
        m[f"ll_{prefix}_lead_down_eth_lag"] = (m[ret5] < -0.002) & (m["ret_5"].abs() < 0.0015)
        specs += [
            EventSpec(f"{prefix}_lead_up_eth_lag", "lead_lag", "long", f"ll_{prefix}_lead_up_eth_lag", f"{lead_symbol} already up over 5m while ETH has not moved much."),
            EventSpec(f"{prefix}_lead_down_eth_lag", "lead_lag", "short", f"ll_{prefix}_lead_down_eth_lag", f"{lead_symbol} already down over 5m while ETH has not moved much."),
        ]
    if ret15 in m.columns and dcol in m.columns:
        m[f"ll_{prefix}_lead_delta_up"] = (m[ret15] > 0.003) & (m[dcol] > 0) & (m["ret_15"].abs() < 0.0025)
        m[f"ll_{prefix}_lead_delta_down"] = (m[ret15] < -0.003) & (m[dcol] < 0) & (m["ret_15"].abs() < 0.0025)
        specs += [
            EventSpec(f"{prefix}_lead_delta_up_eth_lag", "lead_lag", "long", f"ll_{prefix}_lead_delta_up", f"{lead_symbol} 15m upside + delta while ETH lags."),
            EventSpec(f"{prefix}_lead_delta_down_eth_lag", "lead_lag", "short", f"ll_{prefix}_lead_delta_down", f"{lead_symbol} 15m downside + delta while ETH lags."),
        ]
    return m, specs


def build_events(f: pd.DataFrame, specs: list[EventSpec], start_date: str) -> pd.DataFrame:
    rows = []
    idx = f.index.to_numpy()
    timestamps = f["timestamp"].to_numpy()
    start_ts = pd.Timestamp(start_date)
    prog = ProgressReporter(label="[events] microstructure scan", total=len(specs), every=max(1, len(specs) // 20))
    for i, spec in enumerate(specs, start=1):
        if spec.mask_col not in f.columns:
            prog.update(i)
            continue
        mask = f[spec.mask_col].fillna(False).astype(bool).to_numpy()
        if not mask.any():
            prog.update(i)
            continue
        pos = np.flatnonzero(mask)
        # Need next bar entry and in-sample signal.
        pos = pos[(pos + 1) < len(f)]
        if len(pos) == 0:
            prog.update(i)
            continue
        ts_pos = pd.to_datetime(timestamps[pos])
        pos = pos[ts_pos >= start_ts]
        if len(pos) == 0:
            prog.update(i)
            continue
        sub = pd.DataFrame(
            {
                "event_name": spec.name,
                "family": spec.family,
                "side": spec.side,
                "description": spec.description,
                "signal_pos": pos.astype("int64"),
                "signal_time": pd.to_datetime(timestamps[pos]),
            }
        )
        rows.append(sub)
        prog.update(i)
    if prog.last_done < prog.total:
        prog.close()
    else:
        prog.closed = True
    if not rows:
        return pd.DataFrame(columns=["event_name", "family", "side", "description", "signal_pos", "signal_time"])
    out = pd.concat(rows, ignore_index=True)
    out["event_date"] = out["signal_time"].dt.date.astype(str)
    out["event_year"] = out["signal_time"].dt.year.astype(int)
    return out.sort_values(["signal_time", "event_name"]).reset_index(drop=True)


def attach_outcomes(events: pd.DataFrame, f: pd.DataFrame, horizons: Iterable[int], fee_rate: float) -> pd.DataFrame:
    out = events.copy()
    if out.empty:
        out["entry_pos"] = pd.Series(dtype="float")
        out["entry_time"] = pd.Series(dtype="datetime64[ns]")
        out["entry_price"] = pd.Series(dtype="float")
        for h in horizons:
            col_suffix = f"h{int(h)}"
            out[f"exit_time_{col_suffix}"] = pd.Series(dtype="datetime64[ns]")
            out[f"exit_bar_time_{col_suffix}"] = pd.Series(dtype="datetime64[ns]")
            out[f"exit_price_{col_suffix}"] = pd.Series(dtype="float")
            out[f"next_open_ret_{col_suffix}_gross"] = pd.Series(dtype="float")
            out[f"next_open_ret_{col_suffix}_net"] = pd.Series(dtype="float")
            out[f"mfe_{col_suffix}"] = pd.Series(dtype="float")
            out[f"mae_{col_suffix}"] = pd.Series(dtype="float")
        return out
    idx_time = pd.to_datetime(f["timestamp"]).to_numpy(dtype="datetime64[ns]")
    open_arr = pd.to_numeric(f["open"], errors="coerce").to_numpy(dtype="float64")
    high_arr = pd.to_numeric(f["high"], errors="coerce").to_numpy(dtype="float64")
    low_arr = pd.to_numeric(f["low"], errors="coerce").to_numpy(dtype="float64")
    close_arr = pd.to_numeric(f["close"], errors="coerce").to_numpy(dtype="float64")
    signal_pos = pd.to_numeric(out["signal_pos"], errors="coerce").fillna(-1).astype("int64").to_numpy()
    entry_pos = signal_pos + 1
    valid_entry = (entry_pos >= 0) & (entry_pos < len(f))
    out["entry_pos"] = pd.Series(entry_pos).where(valid_entry, np.nan).astype("float")
    out["entry_time"] = pd.NaT
    out["entry_price"] = np.nan
    if valid_entry.any():
        out.loc[valid_entry, "entry_time"] = pd.to_datetime(idx_time[entry_pos[valid_entry]])
        out.loc[valid_entry, "entry_price"] = open_arr[entry_pos[valid_entry]]
    side_sign = np.where(out["side"].astype(str).to_numpy() == "long", 1.0, -1.0)
    idx_ns = idx_time.astype("int64")
    for h in horizons:
        h = int(h)
        col_suffix = f"h{h}"
        out[f"exit_time_{col_suffix}"] = pd.NaT
        out[f"exit_bar_time_{col_suffix}"] = pd.NaT
        out[f"exit_price_{col_suffix}"] = np.nan
        out[f"next_open_ret_{col_suffix}_gross"] = np.nan
        out[f"next_open_ret_{col_suffix}_net"] = np.nan
        out[f"mfe_{col_suffix}"] = np.nan
        out[f"mae_{col_suffix}"] = np.nan
        if not valid_entry.any():
            continue
        entry_ns = idx_ns[entry_pos.clip(0, len(f) - 1)]
        target_available_ns = entry_ns + np.int64(pd.Timedelta(minutes=h).value)
        # Exit using the last closed primary bar whose timestamp < target_available_time.
        exit_pos = np.searchsorted(idx_ns, target_available_ns, side="left") - 1
        valid_exit = valid_entry & (exit_pos >= entry_pos) & (exit_pos < len(f))
        if not valid_exit.any():
            continue
        ep = entry_pos[valid_exit]
        xp = exit_pos[valid_exit]
        eprice = open_arr[ep]
        xprice = close_arr[xp]
        gross = side_sign[valid_exit] * (xprice / eprice - 1.0)
        net = gross - float(fee_rate)
        out.loc[valid_exit, f"exit_bar_time_{col_suffix}"] = pd.to_datetime(idx_time[xp])
        out.loc[valid_exit, f"exit_time_{col_suffix}"] = pd.to_datetime(target_available_ns[valid_exit])
        out.loc[valid_exit, f"exit_price_{col_suffix}"] = xprice
        out.loc[valid_exit, f"next_open_ret_{col_suffix}_gross"] = gross
        out.loc[valid_exit, f"next_open_ret_{col_suffix}_net"] = net
        # MFE/MAE loop by horizon only; OK for research and avoids event-row path loop.
        valid_indices = np.flatnonzero(valid_exit)
        mfe_vals = np.empty(len(valid_indices), dtype="float64")
        mae_vals = np.empty(len(valid_indices), dtype="float64")
        for j, (a, b, px, sgn) in enumerate(zip(ep, xp, eprice, side_sign[valid_exit])):
            if sgn > 0:
                mfe_vals[j] = np.nanmax(high_arr[a : b + 1] / px - 1.0)
                mae_vals[j] = np.nanmin(low_arr[a : b + 1] / px - 1.0)
            else:
                mfe_vals[j] = np.nanmax(1.0 - low_arr[a : b + 1] / px)
                mae_vals[j] = np.nanmin(1.0 - high_arr[a : b + 1] / px)
        out.loc[valid_exit, f"mfe_{col_suffix}"] = mfe_vals
        out.loc[valid_exit, f"mae_{col_suffix}"] = mae_vals
    return out


def build_stats(events: pd.DataFrame, horizons: list[int], out_dir: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    rows_overview = []
    for h in horizons:
        metric = f"next_open_ret_h{h}_net"
        if metric not in events.columns:
            continue
        vals = pd.to_numeric(events[metric], errors="coerce").dropna()
        if vals.empty:
            continue
        rows_overview.append(
            {
                "horizon": h,
                "count": int(vals.count()),
                "mean": float(vals.mean()),
                "median": float(vals.median()),
                "win_rate": float((vals > 0).mean()),
                "profit_factor": profit_factor(vals),
                "q25": float(vals.quantile(0.25)),
                "q75": float(vals.quantile(0.75)),
            }
        )
        summarize_group(events, ["family", "event_name", "side"], metric).to_csv(out_dir / f"03_event_stats_h{h}.csv", index=False)
        summarize_group(events, ["family", "side"], metric).to_csv(out_dir / f"04_family_stats_h{h}.csv", index=False)
        summarize_group(events, ["family", "event_name", "side", "event_year"], metric).to_csv(out_dir / f"05_yearly_stats_h{h}.csv", index=False)
        # Daily de-dup: same event/family/side/date keep first signal.
        dedup = events.sort_values("signal_time").drop_duplicates(["event_name", "side", "event_date"], keep="first")
        summarize_group(dedup, ["family", "event_name", "side"], metric).to_csv(out_dir / f"06_daily_dedup_event_stats_h{h}.csv", index=False)
    overview = pd.DataFrame(rows_overview)
    overview.to_csv(out_dir / "02_overview.csv", index=False)
    if not overview.empty:
        meta["overview"] = overview.to_dict(orient="records")
    # Candidate rank focuses on h60/h120/h240 if present.
    rank_frames = []
    for h in [60, 120, 240]:
        metric = f"next_open_ret_h{h}_net"
        if metric in events.columns:
            s = summarize_group(events, ["family", "event_name", "side"], metric)
            if not s.empty:
                s.insert(0, "horizon", h)
                # Basic sample guard; keep all in CSV but rank sensible rows first.
                s["rank_score"] = s["mean"].fillna(0) * np.log1p(s["count"].fillna(0)) * np.minimum(s["profit_factor"].replace(np.inf, 10).fillna(0), 4)
                rank_frames.append(s)
    if rank_frames:
        rank = pd.concat(rank_frames, ignore_index=True).sort_values(["rank_score", "mean", "count"], ascending=[False, False, False])
        rank.to_csv(out_dir / "15_candidate_rank.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "15_candidate_rank.csv", index=False)
    return meta


def build_audit(events: pd.DataFrame, f: pd.DataFrame, args: argparse.Namespace, horizons: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    rows.append({"check": "ordinary_kline_used", "value": 0, "note": "primary uses OKXTradeBarLoader"})
    rows.append({"check": "range_context_used", "value": int(bool(args.include_range_bars)), "note": "range bars only as optional context"})
    rows.append({"check": "footprint_context_used", "value": int(bool(args.include_footprint)), "note": "range footprint only as optional closed context"})
    if not events.empty:
        rows.append({"check": "entry_not_next_open_flag", "value": int((pd.to_numeric(events["entry_pos"], errors="coerce") != pd.to_numeric(events["signal_pos"], errors="coerce") + 1).sum()), "note": "entry must be next primary bar open"})
        for h in horizons:
            et = pd.to_datetime(events.get(f"entry_time"), errors="coerce")
            xt = pd.to_datetime(events.get(f"exit_time_h{h}"), errors="coerce")
            valid = et.notna() & xt.notna()
            mismatch = int(((xt[valid] - et[valid]) != pd.Timedelta(minutes=h)).sum()) if valid.any() else 0
            rows.append({"check": f"horizon_time_mismatch_h{h}", "value": mismatch, "note": f"exit available time - entry time must be {h} minutes"})
    return pd.DataFrame(rows)


def write_brief(out_dir: Path, args: argparse.Namespace, meta: dict[str, Any]) -> None:
    rank_path = out_dir / "15_candidate_rank.csv"
    rank = pd.read_csv(rank_path) if rank_path.exists() and rank_path.stat().st_size > 0 else pd.DataFrame()
    overview = pd.read_csv(out_dir / "02_overview.csv") if (out_dir / "02_overview.csv").exists() else pd.DataFrame()
    lines = []
    lines.append("# ETH Microstructure Big Money Lab V1")
    lines.append("")
    lines.append("## Run config")
    lines.append(f"- symbol: `{args.symbol}`")
    lines.append(f"- primary timeframe: `{args.primary_timeframe}` trade bars")
    lines.append(f"- date range: `{args.start_date}` -> `{args.end_date}`")
    lines.append(f"- fee rate: `{args.fee_rate}`")
    lines.append(f"- range context: `{args.include_range_bars}` range_pct=`{args.range_pct}`")
    lines.append(f"- footprint context: `{args.include_footprint}` price_step=`{args.footprint_price_step}`")
    lines.append(f"- lead symbols: `{args.lead_symbols}` local_only=`{not args.build_missing_lead}`")
    lines.append("")
    lines.append("## Overview")
    if not overview.empty:
        lines.append(overview.to_markdown(index=False))
    else:
        lines.append("No outcome overview generated.")
    lines.append("")
    lines.append("## Top candidates")
    if not rank.empty:
        cols = [c for c in ["horizon", "family", "event_name", "side", "count", "mean", "median", "win_rate", "profit_factor", "top5_winner_share"] if c in rank.columns]
        lines.append(rank.head(20)[cols].to_markdown(index=False))
    else:
        lines.append("No candidates generated.")
    lines.append("")
    lines.append("## Interpretation guardrails")
    lines.append("- This is an event study, not a tradable strategy replay.")
    lines.append("- Daily de-dup files are more reliable than raw event stats when events overlap.")
    lines.append("- Cross-market lead-lag only runs for locally cached lead-symbol trade bars unless `--build-missing-lead` is used.")
    lines.append("- Strong candidates still need stress tests, delay tests, and root-signal de-dup before strategy promotion.")
    (out_dir / "20_research_brief.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="ETH Microstructure Big Money Lab V1")
    ap.add_argument("--symbol", default="ETH-USDT-SWAP")
    ap.add_argument("--primary-timeframe", default="1m")
    ap.add_argument("--start-date", default="2023-01-01")
    ap.add_argument("--end-date", default="2026-06-30")
    ap.add_argument("--warmup-start-date", default="2022-01-01")
    ap.add_argument("--horizons", default="5,15,30,60,120,240")
    ap.add_argument("--fee-rate", type=float, default=0.0011, help="Round-trip fee/slippage cost subtracted from gross return.")
    ap.add_argument("--include-range-bars", action="store_true")
    ap.add_argument("--range-pct", type=float, default=0.0020)
    ap.add_argument("--include-footprint", action="store_true")
    ap.add_argument("--footprint-price-step", type=float, default=1.0)
    ap.add_argument("--lead-symbols", default="BTC-USDT-SWAP", help="Comma-separated optional lead symbols; loaded local-only by default.")
    ap.add_argument("--no-lead-lag", action="store_true")
    ap.add_argument("--build-missing-lead", action="store_true", help="Allow building/downloading missing lead-symbol trade bars. Default false.")
    ap.add_argument("--write-full-events", action="store_true")
    ap.add_argument("--event-sample-size", type=int, default=200_000)
    ap.add_argument("--out-dir", default="data/reports/research/microstructure_big_money_lab_v1")
    args = ap.parse_args(argv)

    out_dir = safe_mkdir(Path(args.out_dir))
    horizons = parse_list_int(args.horizons)

    print(f"[load] primary trade bars symbol={args.symbol} tf={args.primary_timeframe} {args.warmup_start_date}->{args.end_date}", flush=True)
    bars = load_trade_bars(args.symbol, args.primary_timeframe, args.start_date, args.end_date, warmup_start_date=args.warmup_start_date, build_missing=True)
    if bars.empty:
        raise RuntimeError("primary trade bars are empty")
    print(f"[load] primary rows={len(bars):,} range={bars['timestamp'].min()} -> {bars['timestamp'].max()}", flush=True)

    print("[features] building primary microstructure features", flush=True)
    feat = build_primary_features(bars, args.start_date)
    all_specs: list[EventSpec] = []
    all_specs += add_market_impact_masks(feat)
    all_specs += add_absorption_zone_masks(feat)
    all_specs += add_liquidity_vacuum_masks(feat)

    if args.include_range_bars:
        print(f"[load] range bars range_pct={args.range_pct}", flush=True)
        rb = load_range_bars(args.symbol, args.range_pct, args.start_date, args.end_date, warmup_start_date=args.warmup_start_date)
        print(f"[range] rows={len(rb):,}", flush=True)
        feat, specs = add_range_context_features(feat, rb)
        all_specs += specs
    else:
        rb = pd.DataFrame()

    if args.include_footprint:
        print(f"[load] range footprint range_pct={args.range_pct} step={args.footprint_price_step}", flush=True)
        fp = load_footprint(args.symbol, args.range_pct, args.footprint_price_step, args.start_date, args.end_date, warmup_start_date=args.warmup_start_date)
        print(f"[footprint] rows={len(fp):,}", flush=True)
        feat, specs = add_footprint_context_features(feat, fp)
        all_specs += specs
    else:
        fp = pd.DataFrame()

    lead_status = []
    if not args.no_lead_lag:
        for lead_symbol in parse_list_str(args.lead_symbols):
            if not lead_symbol or lead_symbol == args.symbol:
                continue
            print(f"[lead] loading lead symbol={lead_symbol} local_only={not args.build_missing_lead}", flush=True)
            try:
                lead = load_trade_bars(
                    lead_symbol,
                    args.primary_timeframe,
                    args.start_date,
                    args.end_date,
                    warmup_start_date=args.warmup_start_date,
                    build_missing=bool(args.build_missing_lead),
                )
            except Exception as exc:
                print(f"[lead] skip {lead_symbol}: {exc}", flush=True)
                lead_status.append({"symbol": lead_symbol, "status": "error", "error": repr(exc)})
                continue
            if lead.empty:
                print(f"[lead] skip {lead_symbol}: no local trade bars", flush=True)
                lead_status.append({"symbol": lead_symbol, "status": "empty"})
                continue
            print(f"[lead] rows={len(lead):,} symbol={lead_symbol}", flush=True)
            feat, specs = add_lead_lag_features(feat, lead, lead_symbol)
            all_specs += specs
            lead_status.append({"symbol": lead_symbol, "status": "loaded", "rows": int(len(lead))})

    catalog = pd.DataFrame([s.__dict__ for s in all_specs])
    catalog.to_csv(out_dir / "00_event_catalog.csv", index=False)
    print(f"[events] specs={len(all_specs):,}", flush=True)
    events = build_events(feat, all_specs, args.start_date)
    print(f"[events] raw_events={len(events):,} families={events['family'].nunique() if not events.empty else 0}", flush=True)

    print("[outcomes] attaching fixed-horizon outcomes", flush=True)
    events_out = attach_outcomes(events, feat, horizons, args.fee_rate)

    if args.write_full_events:
        events_out.to_csv(out_dir / "01_events.csv", index=False)
    else:
        events_out.head(int(args.event_sample_size)).to_csv(out_dir / "01_events_sample.csv", index=False)

    print("[stats] writing reports", flush=True)
    meta = build_stats(events_out, horizons, out_dir)
    audit = build_audit(events_out, feat, args, horizons)
    audit.to_csv(out_dir / "08_causal_audit.csv", index=False)
    meta.update(
        {
            "symbol": args.symbol,
            "primary_timeframe": args.primary_timeframe,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "warmup_start_date": args.warmup_start_date,
            "primary_rows": int(len(bars)),
            "event_specs": int(len(all_specs)),
            "raw_events": int(len(events_out)),
            "families": sorted(events_out["family"].dropna().unique().tolist()) if not events_out.empty else [],
            "include_range_bars": bool(args.include_range_bars),
            "include_footprint": bool(args.include_footprint),
            "lead_status": lead_status,
        }
    )
    (out_dir / "10_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_brief(out_dir, args, meta)
    print(f"[done] out_dir={out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
