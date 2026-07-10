#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH MHF/MF volatility compression-expansion fade V1 event study.

Research-only script for a narrowly pre-declared hypothesis:

    During an extended low-volatility 1m regime, price suddenly expands in one
    direction. When the expansion starts contracting, fade the expansion and
    enter against the burst direction at the next 1m open.

This is not a naked volatility breakout strategy. It tests post-expansion
exhaustion/reversal after a low-volatility compression. It does not register a
tradable edge, modify portfolio code, or import business logic from other
research scripts.

Timing policy:
    signal_time = current primary closed 1m trade bar
    entry_time  = next primary bar open plus optional delay bars
    entry_price = open[entry_idx]
    fixed-hold exit uses the open after hold bars

Performance policy:
    - feature/event masks are vectorized;
    - forward fixed-hold returns use precomputed forward open/high/low arrays;
    - baseline uses grouped numpy pools and only runs for preliminary survivors;
    - no per-event full-DataFrame filtering.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_NAME = "eth_mhf_volatility_compression_fade_v1.py"
SCRIPT_VERSION = "1.0.1"
EXPERIMENT_ID = "ETH_MHF_VOL_COMPRESSION_FADE_V1"
EDGE_ID = "ETH_EDGE_MHF_VOL_COMPRESSION_FADE_RESEARCH_V1"
DEFAULT_OUT_DIR = "data/reports/research/mhf_volatility_compression_fade_v1"
CAUSAL_POLICY = "closed 1m trade-bar signal; entry at next 1m open plus delay; fixed-hold exit at future open"
MATCHED_BASELINE_COLUMNS = ("year", "month", "session", "regime", "volatility_bucket")


@dataclass(frozen=True)
class EventSpec:
    compression_window: int
    burst_window: int
    burst_mult: float
    contraction_ratio: float
    low_vol_ratio: float
    min_abs_burst_ret_pct: float

    @property
    def variant(self) -> str:
        bm = str(self.burst_mult).replace(".", "p")
        cr = str(self.contraction_ratio).replace(".", "p")
        lv = str(self.low_vol_ratio).replace(".", "p")
        rbp = int(round(self.min_abs_burst_ret_pct * 10000))
        return f"cw{self.compression_window}_bw{self.burst_window}_bm{bm}_cr{cr}_lv{lv}_ret{rbp}bp"

    def event_name(self, direction: str) -> str:
        return f"vol_compression_expansion_fade__{self.variant}__{direction}"


# ---------------------------------------------------------------------------
# CLI and helpers
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Research-only ETH low-vol compression -> sudden expansion -> contraction fade event study.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--primary-timeframe", default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--compression-windows", default="60,120")
    p.add_argument("--burst-windows", default="3,5,8")
    p.add_argument("--burst-multipliers", default="3.0,4.0")
    p.add_argument("--contraction-ratios", default="0.50,0.70")
    p.add_argument("--low-vol-ratios", default="0.65,0.80", help="prior short-range mean / long-range mean must be below this")
    p.add_argument("--min-abs-burst-ret-pct", default="0.0010", help="Absolute cumulative burst return threshold; comma-separated allowed")
    p.add_argument("--long-vol-window", type=int, default=720, help="Long baseline window for low-vol comparison")
    p.add_argument("--horizons", default="5,10,15,30,45,60,120")
    p.add_argument("--mfe-mae-horizon", type=int, default=120)
    p.add_argument("--round-trip-cost-pct", type=float, default=0.0011)
    p.add_argument("--cost-multipliers", default="1.0,1.5,2.0,3.0")
    p.add_argument("--delay-bars-list", default="0,1,2,3")
    p.add_argument("--min-count", type=int, default=500)
    p.add_argument("--min-events-per-year", type=float, default=120.0)
    p.add_argument("--cooldown-bars", type=int, default=0)
    p.add_argument("--baseline-samples", type=int, default=100)
    p.add_argument("--baseline-max-events-per-group", type=int, default=1500)
    p.add_argument("--baseline-prefilter-mean-net", type=float, default=-0.0002)
    p.add_argument("--baseline-prefilter-pf", type=float, default=0.95)
    p.add_argument("--baseline-seed", type=int, default=42)
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--no-build-missing", action="store_true")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--event-sample-size", type=int, default=5000)
    p.add_argument("--trade-sample-size", type=int, default=20000)
    p.add_argument("--write-full-trades", action="store_true")
    p.add_argument("--progress-every", type=int, default=8)
    return p.parse_args(argv)


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    vals = []
    for part in str(raw).split(","):
        text = part.strip()
        if text:
            vals.append(int(text))
    return tuple(dict.fromkeys(vals))


def _parse_csv_floats(raw: str) -> tuple[float, ...]:
    vals = []
    for part in str(raw).split(","):
        text = part.strip()
        if text:
            vals.append(float(text))
    return tuple(dict.fromkeys(vals))


def _safe_divide_np(num: np.ndarray, den: np.ndarray, default: float = 0.0) -> np.ndarray:
    out = np.full(len(num), default, dtype=float)
    np.divide(num, den, out=out, where=np.isfinite(den) & (np.abs(den) > 1e-12))
    out[~np.isfinite(out)] = default
    return out


def _profit_factor(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    gains = v[v > 0].sum()
    losses = -v[v < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def _top_winner_share(values: np.ndarray, top_n: int = 5) -> float:
    v = np.asarray(values, dtype=float)
    winners = np.sort(v[np.isfinite(v) & (v > 0)])[::-1]
    if winners.size == 0:
        return 0.0
    return float(winners[:top_n].sum() / max(winners.sum(), 1e-12))


def _max_days_without_event(times: pd.Series | pd.DatetimeIndex) -> float:
    if len(times) <= 1:
        return float("nan")
    ts = pd.to_datetime(pd.Series(times).dropna()).sort_values()
    if len(ts) <= 1:
        return float("nan")
    gaps = ts.diff().dropna().dt.total_seconds() / 86400.0
    return float(gaps.max()) if not gaps.empty else float("nan")


def _annualized_years(start_date: str, end_date: str) -> float:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    return max((end - start).total_seconds() / (365.25 * 86400.0), 1e-9)


# ---------------------------------------------------------------------------
# Data and features
# ---------------------------------------------------------------------------


def load_trade_bars(args: argparse.Namespace) -> pd.DataFrame:
    loader_kwargs: dict[str, object] = {
        "symbol": args.symbol,
        "timeframe": args.primary_timeframe,
        "db_name": args.db_name,
    }
    if args.data_dir:
        loader_kwargs["data_dir"] = Path(args.data_dir)
    loader = OKXTradeBarLoader(**loader_kwargs)
    df = loader.fetch_data_by_date_range(
        args.warmup_start_date,
        args.end_date,
        chunksize=args.chunksize,
        force_rebuild=bool(args.force_rebuild),
        cvd_mode="range",
        build_missing=not bool(args.no_build_missing),
    )
    if df.empty:
        raise RuntimeError("OKXTradeBarLoader returned no rows")
    df = df.sort_index()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("trade bar dataframe must be indexed by DatetimeIndex")
    df.index = pd.to_datetime(df.index)
    return df


def _assign_session(hour: pd.Series) -> pd.Series:
    h = hour.astype(int)
    return pd.Series(
        np.select(
            [h < 8, h < 16],
            ["asia_00_08", "asia_europe_08_16"],
            default="us_16_24",
        ),
        index=hour.index,
    )


def build_features(df: pd.DataFrame, *, long_vol_window: int) -> tuple[pd.DataFrame, list[str]]:
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required OHLC columns: {missing}")

    base = df.copy()
    for col in required:
        base[col] = pd.to_numeric(base[col], errors="coerce").astype(float)
    open_ = base["open"].replace(0, np.nan)
    high = base["high"]
    low = base["low"]
    close = base["close"]

    ret_1 = close.pct_change().fillna(0.0)
    signed_body_pct = ((close - open_) / open_).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    abs_ret_1 = ret_1.abs()
    range_pct = ((high - low) / open_).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    body_pct = signed_body_pct.abs()
    close_pos = ((close - low) / (high - low).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.5)

    range_mean_15 = range_pct.rolling(15, min_periods=5).mean()
    range_mean_30 = range_pct.rolling(30, min_periods=10).mean()
    range_mean_60 = range_pct.rolling(60, min_periods=20).mean()
    range_mean_120 = range_pct.rolling(120, min_periods=40).mean()
    long_range_mean = range_pct.shift(1).rolling(long_vol_window, min_periods=max(60, long_vol_window // 4)).mean()
    long_range_std = range_pct.shift(1).rolling(long_vol_window, min_periods=max(60, long_vol_window // 4)).std(ddof=0)
    range_z_long = ((range_pct - long_range_mean) / long_range_std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    vol_30 = ret_1.rolling(30, min_periods=10).std(ddof=0)
    vol_120 = ret_1.rolling(120, min_periods=40).std(ddof=0)
    vol_ratio_30_120 = (vol_30 / vol_120.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # Market-regime and matching buckets are deliberately coarse diagnostics.
    vol_bucket = pd.qcut(range_mean_60.rank(method="first"), 5, labels=False, duplicates="drop")
    vol_bucket = pd.Series(vol_bucket, index=base.index).fillna(2).astype(int).map(lambda x: f"vol_q{int(x) + 1}")
    regime = pd.Series(
        np.select(
            [vol_ratio_30_120 >= 1.35, vol_ratio_30_120 <= 0.75],
            ["high_vol", "low_vol"],
            default="normal_vol",
        ),
        index=base.index,
    )

    feature_map: dict[str, pd.Series] = {
        "ret_1": ret_1,
        "abs_ret_1": abs_ret_1,
        "signed_body_pct": signed_body_pct,
        "range_pct": range_pct,
        "body_pct": body_pct,
        "close_pos": close_pos,
        "range_mean_15": range_mean_15,
        "range_mean_30": range_mean_30,
        "range_mean_60": range_mean_60,
        "range_mean_120": range_mean_120,
        "long_range_mean": long_range_mean,
        "long_range_std": long_range_std,
        "range_z_long": range_z_long,
        "vol_30": vol_30,
        "vol_120": vol_120,
        "vol_ratio_30_120": vol_ratio_30_120,
        "year": pd.Series(base.index.year, index=base.index),
        "month": pd.Series(base.index.month, index=base.index),
        "weekday": pd.Series(base.index.weekday, index=base.index),
        "hour": pd.Series(base.index.hour, index=base.index),
        "session": _assign_session(pd.Series(base.index.hour, index=base.index)),
        "regime": regime,
        "volatility_bucket": vol_bucket,
    }

    # Optional order-flow diagnostics, not hard dependencies for the event.
    if "notional" in base.columns:
        notional = pd.to_numeric(base["notional"], errors="coerce").fillna(0.0)
        n_mean = notional.shift(1).rolling(120, min_periods=30).mean()
        n_std = notional.shift(1).rolling(120, min_periods=30).std(ddof=0)
        feature_map["notional_z_120"] = ((notional - n_mean) / n_std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "delta_notional" in base.columns:
        delta = pd.to_numeric(base["delta_notional"], errors="coerce").fillna(0.0)
        d_mean = delta.shift(1).rolling(120, min_periods=30).mean()
        d_std = delta.shift(1).rolling(120, min_periods=30).std(ddof=0)
        feature_map["delta_z_120"] = ((delta - d_mean) / d_std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    features = pd.DataFrame(feature_map, index=base.index)
    out = pd.concat([base, features], axis=1)
    feature_cols = list(features.columns)
    return out, feature_cols


# ---------------------------------------------------------------------------
# Event masks
# ---------------------------------------------------------------------------


def build_event_specs(args: argparse.Namespace) -> list[EventSpec]:
    specs: list[EventSpec] = []
    for cw in _parse_csv_ints(args.compression_windows):
        for bw in _parse_csv_ints(args.burst_windows):
            for bm in _parse_csv_floats(args.burst_multipliers):
                for cr in _parse_csv_floats(args.contraction_ratios):
                    for lv in _parse_csv_floats(args.low_vol_ratios):
                        for ret_thr in _parse_csv_floats(args.min_abs_burst_ret_pct):
                            specs.append(
                                EventSpec(
                                    compression_window=cw,
                                    burst_window=bw,
                                    burst_mult=bm,
                                    contraction_ratio=cr,
                                    low_vol_ratio=lv,
                                    min_abs_burst_ret_pct=ret_thr,
                                )
                            )
    return specs


def _apply_cooldown(signal_pos: np.ndarray, cooldown_bars: int) -> np.ndarray:
    pos = np.asarray(signal_pos, dtype=np.int64)
    if cooldown_bars <= 0 or pos.size <= 1:
        return pos
    kept: list[int] = []
    last = -10**12
    for p in pos:
        if int(p) - last > cooldown_bars:
            kept.append(int(p))
            last = int(p)
    return np.asarray(kept, dtype=np.int64)


def build_events(df: pd.DataFrame, specs: list[EventSpec], *, start_date: str, cooldown_bars: int, progress_every: int) -> pd.DataFrame:
    range_pct = df["range_pct"].astype(float)
    ret_1 = df["ret_1"].astype(float)
    index = df.index
    rows: list[pd.DataFrame] = []
    start_ts = pd.Timestamp(start_date)

    with ProgressReporter("[events] volatility specs", total=len(specs), every=max(1, progress_every)) as prog:
        for i, spec in enumerate(specs, start=1):
            # Low volatility is measured strictly before the burst window.
            pre_short_mean = range_pct.shift(spec.burst_window + 1).rolling(spec.compression_window, min_periods=max(10, spec.compression_window // 3)).mean()
            pre_long_mean = range_pct.shift(spec.burst_window + 1).rolling(max(spec.compression_window * 4, 240), min_periods=max(60, spec.compression_window)).mean()
            low_vol_ratio = (pre_short_mean / pre_long_mean.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            compressed = low_vol_ratio <= spec.low_vol_ratio

            # Expansion happened in the prior burst_window bars; current closed bar is the contraction signal.
            burst_peak = range_pct.shift(1).rolling(spec.burst_window, min_periods=spec.burst_window).max()
            burst_ret = ret_1.shift(1).rolling(spec.burst_window, min_periods=spec.burst_window).sum()
            burst_vs_pre = (burst_peak / pre_short_mean.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
            contraction = (range_pct < burst_peak * spec.contraction_ratio) & (range_pct < range_pct.shift(1))
            burst_ok = (burst_vs_pre >= spec.burst_mult) & (burst_ret.abs() >= spec.min_abs_burst_ret_pct)

            common = compressed & burst_ok & contraction & (pd.Series(index, index=index) >= start_ts)
            common = common.fillna(False).to_numpy(dtype=bool)
            br = burst_ret.to_numpy(dtype=float)

            for direction, dir_mask in (
                ("long", common & (br < 0)),
                ("short", common & (br > 0)),
            ):
                signal_pos = np.flatnonzero(dir_mask)
                signal_pos = _apply_cooldown(signal_pos, cooldown_bars)
                if signal_pos.size == 0:
                    continue
                part = pd.DataFrame(
                    {
                        "event_name": spec.event_name(direction),
                        "family": "volatility_compression_expansion_fade",
                        "direction": direction,
                        "variant": spec.variant,
                        "compression_window": spec.compression_window,
                        "burst_window": spec.burst_window,
                        "burst_mult": spec.burst_mult,
                        "contraction_ratio": spec.contraction_ratio,
                        "low_vol_ratio": spec.low_vol_ratio,
                        "min_abs_burst_ret_pct": spec.min_abs_burst_ret_pct,
                        "signal_idx": signal_pos,
                        "signal_time": index[signal_pos],
                        "burst_ret": br[signal_pos],
                        "signal_range_pct": range_pct.to_numpy(dtype=float)[signal_pos],
                        "burst_peak_range_pct": burst_peak.to_numpy(dtype=float)[signal_pos],
                        "burst_vs_pre_range": burst_vs_pre.to_numpy(dtype=float)[signal_pos],
                    }
                )
                rows.append(part)
            prog.update(i)

    if not rows:
        return pd.DataFrame()
    events = pd.concat(rows, ignore_index=True)
    meta_cols = ["year", "month", "session", "regime", "volatility_bucket"]
    for c in meta_cols:
        if c in df.columns:
            values = df[c].to_numpy()
            events[c] = values[events["signal_idx"].to_numpy(dtype=np.int64)]
    return events.sort_values(["signal_idx", "event_name"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Replay and aggregation
# ---------------------------------------------------------------------------


def _forward_window_extreme(values: np.ndarray, window: int, fn: str) -> np.ndarray:
    s = pd.Series(values)
    rev = s.iloc[::-1]
    if fn == "max":
        out = rev.rolling(window, min_periods=window).max().iloc[::-1]
    elif fn == "min":
        out = rev.rolling(window, min_periods=window).min().iloc[::-1]
    else:
        raise ValueError(fn)
    return out.to_numpy(dtype=float)


def _compute_returns_for_positions(
    signal_idx: np.ndarray,
    directions: np.ndarray,
    *,
    open_arr: np.ndarray,
    high_fwd: np.ndarray,
    low_fwd: np.ndarray,
    horizon: int,
    delay_bars: int,
) -> dict[str, np.ndarray]:
    sig = np.asarray(signal_idx, dtype=np.int64)
    entry_idx = sig + 1 + int(delay_bars)
    exit_idx = entry_idx + int(horizon)
    n = len(open_arr)
    valid = (entry_idx >= 0) & (exit_idx >= 0) & (entry_idx < n) & (exit_idx < n)
    entry_price = np.full(sig.size, np.nan, dtype=float)
    exit_price = np.full(sig.size, np.nan, dtype=float)
    mfe = np.full(sig.size, np.nan, dtype=float)
    mae = np.full(sig.size, np.nan, dtype=float)
    gross = np.full(sig.size, np.nan, dtype=float)
    if valid.any():
        ep = open_arr[entry_idx[valid]]
        xp = open_arr[exit_idx[valid]]
        dirs = directions[valid]
        entry_price[valid] = ep
        exit_price[valid] = xp
        long_mask = dirs == "long"
        short_mask = dirs == "short"
        gr = np.full(ep.size, np.nan, dtype=float)
        gr[long_mask] = xp[long_mask] / ep[long_mask] - 1.0
        gr[short_mask] = ep[short_mask] / xp[short_mask] - 1.0
        gross[valid] = gr
        hw = high_fwd[entry_idx[valid]]
        lw = low_fwd[entry_idx[valid]]
        mfe_v = np.full(ep.size, np.nan, dtype=float)
        mae_v = np.full(ep.size, np.nan, dtype=float)
        mfe_v[long_mask] = hw[long_mask] / ep[long_mask] - 1.0
        mae_v[long_mask] = lw[long_mask] / ep[long_mask] - 1.0
        mfe_v[short_mask] = ep[short_mask] / lw[short_mask] - 1.0
        mae_v[short_mask] = ep[short_mask] / hw[short_mask] - 1.0
        mfe[valid] = mfe_v
        mae[valid] = mae_v
    return {
        "entry_idx": entry_idx,
        "exit_idx": exit_idx,
        "valid": valid,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "gross_return": gross,
        "mfe": mfe,
        "mae": mae,
    }


def build_replay_tables(
    df: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    delay_bars_list: tuple[int, ...],
    round_trip_cost_pct: float,
    cost_multipliers: tuple[float, ...],
    start_date: str,
    end_date: str,
    min_count: int,
    min_events_per_year: float,
    progress_every: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    open_arr = df["open"].to_numpy(dtype=float)
    high_arr = df["high"].to_numpy(dtype=float)
    low_arr = df["low"].to_numpy(dtype=float)
    times = df.index
    years_len = _annualized_years(start_date, end_date)
    event_summary_rows: list[dict[str, object]] = []
    variant_rows: list[pd.DataFrame] = []
    cost_rows: list[dict[str, object]] = []
    delay_rows: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    sample_rows: list[pd.DataFrame] = []

    if events.empty:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    for (event_name, direction), g in events.groupby(["event_name", "direction"], sort=False):
        event_summary_rows.append(
            {
                "event_name": event_name,
                "family": g["family"].iloc[0],
                "direction": direction,
                "variant": g["variant"].iloc[0],
                "count": int(len(g)),
                "events_per_year": float(len(g) / years_len),
                "events_per_month": float(len(g) / (years_len * 12.0)),
                "active_months": int(pd.to_datetime(g["signal_time"]).dt.to_period("M").nunique()),
                "max_days_without_event": _max_days_without_event(g["signal_time"]),
            }
        )

    total_groups = len(events.groupby(["event_name", "direction"], sort=False)) * len(horizons) * len(delay_bars_list)
    with ProgressReporter("[forward] replay groups", total=total_groups, every=max(1, progress_every)) as prog:
        done = 0
        for horizon in horizons:
            high_fwd = _forward_window_extreme(high_arr, horizon, "max")
            low_fwd = _forward_window_extreme(low_arr, horizon, "min")
            for delay_bars in delay_bars_list:
                for (event_name, direction), g in events.groupby(["event_name", "direction"], sort=False):
                    signal_idx = g["signal_idx"].to_numpy(dtype=np.int64)
                    dirs = g["direction"].to_numpy(dtype=object)
                    out = _compute_returns_for_positions(
                        signal_idx,
                        dirs,
                        open_arr=open_arr,
                        high_fwd=high_fwd,
                        low_fwd=low_fwd,
                        horizon=horizon,
                        delay_bars=delay_bars,
                    )
                    valid = out["valid"] & np.isfinite(out["gross_return"])
                    if valid.any():
                        gross = out["gross_return"][valid]
                        net = gross - round_trip_cost_pct
                        rows = pd.DataFrame(
                            {
                                "event_name": event_name,
                                "family": g["family"].iloc[0],
                                "direction": direction,
                                "variant": g["variant"].iloc[0],
                                "horizon": horizon,
                                "delay_bars": delay_bars,
                                "signal_idx": signal_idx[valid],
                                "signal_time": g["signal_time"].to_numpy()[valid],
                                "entry_idx": out["entry_idx"][valid],
                                "entry_time": times[out["entry_idx"][valid]],
                                "entry_price": out["entry_price"][valid],
                                "exit_idx": out["exit_idx"][valid],
                                "exit_time": times[out["exit_idx"][valid]],
                                "exit_price": out["exit_price"][valid],
                                "gross_return": gross,
                                "net_return": net,
                                "mfe": out["mfe"][valid],
                                "mae": out["mae"][valid],
                                "year": g["year"].to_numpy()[valid],
                                "month": g["month"].to_numpy()[valid],
                                "session": g["session"].to_numpy()[valid],
                                "regime": g["regime"].to_numpy()[valid],
                                "volatility_bucket": g["volatility_bucket"].to_numpy()[valid],
                            }
                        )
                        if delay_bars == 0:
                            variant_rows.append(_summarize_replay_rows(rows, min_count, min_events_per_year, years_len))
                            for cm in cost_multipliers:
                                vals = gross - round_trip_cost_pct * cm
                                cost_rows.append(_summary_dict(rows, vals, extra={"cost_multiplier": cm, "horizon": horizon, "delay_bars": delay_bars}))
                            for y, yg in rows.groupby("year"):
                                yearly_rows.append(_summary_dict(yg, yg["net_return"].to_numpy(dtype=float), extra={"year": int(y), "horizon": horizon, "delay_bars": delay_bars}))
                            if len(sample_rows) < 50:
                                sample_rows.append(rows.head(200))
                        if delay_bars in delay_bars_list:
                            delay_rows.append(_summary_dict(rows, net, extra={"horizon": horizon, "delay_bars": delay_bars}))
                    done += 1
                    prog.update(done)

    event_summary = pd.DataFrame(event_summary_rows)
    replay_summary = pd.concat(variant_rows, ignore_index=True) if variant_rows else pd.DataFrame()
    cost_stress = pd.DataFrame(cost_rows)
    delay_stress = pd.DataFrame(delay_rows)
    yearly = pd.DataFrame(yearly_rows)
    trades_sample = pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()
    return event_summary, replay_summary, cost_stress, delay_stress, yearly, trades_sample


def _summary_dict(rows: pd.DataFrame, values: np.ndarray, *, extra: dict[str, object] | None = None) -> dict[str, object]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    out: dict[str, object] = {
        "event_name": rows["event_name"].iloc[0] if not rows.empty else "",
        "family": rows["family"].iloc[0] if not rows.empty else "",
        "direction": rows["direction"].iloc[0] if not rows.empty else "",
        "variant": rows["variant"].iloc[0] if not rows.empty else "",
        "count": int(vals.size),
        "mean_net": float(np.mean(vals)) if vals.size else float("nan"),
        "median_net": float(np.median(vals)) if vals.size else float("nan"),
        "win_rate": float(np.mean(vals > 0)) if vals.size else float("nan"),
        "profit_factor": _profit_factor(vals),
        "top5_winner_share": _top_winner_share(vals),
    }
    if extra:
        out.update(extra)
    return out


def _summarize_replay_rows(rows: pd.DataFrame, min_count: int, min_events_per_year: float, years_len: float) -> pd.DataFrame:
    vals = rows["net_return"].to_numpy(dtype=float)
    positive_years = 0
    year_count = 0
    for _, yg in rows.groupby("year"):
        year_count += 1
        if yg["net_return"].mean() > 0:
            positive_years += 1
    count = int(len(rows))
    events_per_year = float(count / years_len)
    out = _summary_dict(rows, vals, extra={"horizon": int(rows["horizon"].iloc[0]), "delay_bars": int(rows["delay_bars"].iloc[0])})
    out.update(
        {
            "events_per_year": events_per_year,
            "events_per_month": float(events_per_year / 12.0),
            "mfe_mean": float(rows["mfe"].mean()),
            "mae_mean": float(rows["mae"].mean()),
            "positive_years": int(positive_years),
            "year_count": int(year_count),
            "active_months": int(pd.to_datetime(rows["signal_time"]).dt.to_period("M").nunique()),
            "max_days_without_event": _max_days_without_event(rows["signal_time"]),
            "passes_frequency": bool(count >= min_count and events_per_year >= min_events_per_year),
        }
    )
    return pd.DataFrame([out])


# ---------------------------------------------------------------------------
# Baseline, shortlist, audit
# ---------------------------------------------------------------------------


def build_baseline_summary(
    df: pd.DataFrame,
    events: pd.DataFrame,
    replay_summary: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    round_trip_cost_pct: float,
    baseline_samples: int,
    baseline_max_events_per_group: int,
    baseline_prefilter_mean_net: float,
    baseline_prefilter_pf: float,
    seed: int,
    progress_every: int,
) -> pd.DataFrame:
    if replay_summary.empty or events.empty or baseline_samples <= 0:
        print("[aggregate] matched baseline skipped: empty replay/events or baseline_samples<=0", flush=True)
        return pd.DataFrame()
    candidates = replay_summary[
        (replay_summary["count"] >= 100)
        & (replay_summary["mean_net"] >= baseline_prefilter_mean_net)
        & (replay_summary["profit_factor"] >= baseline_prefilter_pf)
    ].copy()
    if candidates.empty:
        print("[aggregate] matched baseline skipped: no groups passed preliminary filters", flush=True)
        return pd.DataFrame()

    print(f"[aggregate] matched baseline groups={len(candidates)}", flush=True)
    rng = np.random.default_rng(seed)
    open_arr = df["open"].to_numpy(dtype=float)
    n = len(df)
    all_event_pos = set(events["signal_idx"].astype(int).tolist())
    all_pos = np.arange(n - max(horizons) - 2, dtype=np.int64)
    base_mask = np.ones_like(all_pos, dtype=bool)
    if all_event_pos:
        base_mask &= ~np.isin(all_pos, np.fromiter(all_event_pos, dtype=np.int64))
    base_pos = all_pos[base_mask]

    key_cols = list(MATCHED_BASELINE_COLUMNS)
    key_frame = df.iloc[base_pos][key_cols].copy()
    pools: dict[tuple[object, ...], np.ndarray] = {}
    for key, pos_idxs in key_frame.groupby(key_cols, sort=False).indices.items():
        # groupby().groups returns index labels; df keeps timestamp index labels here.
        # Use groupby().indices so we get positional offsets into base_pos.
        k = key if isinstance(key, tuple) else (key,)
        pools[k] = base_pos[np.asarray(pos_idxs, dtype=np.int64)]

    high_cache = {h: _forward_window_extreme(df["high"].to_numpy(dtype=float), h, "max") for h in sorted(set(candidates["horizon"].astype(int)))}
    low_cache = {h: _forward_window_extreme(df["low"].to_numpy(dtype=float), h, "min") for h in sorted(set(candidates["horizon"].astype(int)))}

    rows: list[dict[str, object]] = []
    with ProgressReporter("[aggregate] matched baseline", total=len(candidates), every=max(1, progress_every)) as prog:
        for done, cand in enumerate(candidates.itertuples(index=False), start=1):
            event_name = str(cand.event_name)
            direction = str(cand.direction)
            horizon = int(cand.horizon)
            ev = events[(events["event_name"] == event_name) & (events["direction"] == direction)].copy()
            if ev.empty:
                prog.update(done)
                continue
            if len(ev) > baseline_max_events_per_group:
                ev = ev.sample(n=baseline_max_events_per_group, random_state=seed + (zlib.adler32(event_name.encode("utf-8")) % 100000))
            sampled_positions: list[np.ndarray] = []
            for key, kg in ev.groupby(key_cols, sort=False):
                k = key if isinstance(key, tuple) else (key,)
                pool = pools.get(k)
                if pool is None or pool.size == 0:
                    continue
                draws = min(len(kg) * baseline_samples, pool.size)
                replace = pool.size < len(kg) * baseline_samples
                sampled_positions.append(rng.choice(pool, size=draws, replace=replace))
            if not sampled_positions:
                rows.append({"event_name": event_name, "direction": direction, "horizon": horizon, "baseline_count": 0, "baseline_mean_net": np.nan, "matched_excess_mean_net": np.nan, "baseline_match_rate": 0.0})
                prog.update(done)
                continue
            sig = np.concatenate(sampled_positions).astype(np.int64)
            dirs = np.array([direction] * sig.size, dtype=object)
            out = _compute_returns_for_positions(sig, dirs, open_arr=open_arr, high_fwd=high_cache[horizon], low_fwd=low_cache[horizon], horizon=horizon, delay_bars=0)
            valid = out["valid"] & np.isfinite(out["gross_return"])
            vals = out["gross_return"][valid] - round_trip_cost_pct
            event_mean = float(cand.mean_net)
            rows.append(
                {
                    "event_name": event_name,
                    "family": str(cand.family),
                    "direction": direction,
                    "variant": str(cand.variant),
                    "horizon": horizon,
                    "event_count": int(cand.count),
                    "baseline_count": int(vals.size),
                    "event_mean_net": event_mean,
                    "baseline_mean_net": float(np.mean(vals)) if vals.size else np.nan,
                    "matched_excess_mean_net": event_mean - (float(np.mean(vals)) if vals.size else np.nan),
                    "baseline_win_rate": float(np.mean(vals > 0)) if vals.size else np.nan,
                    "baseline_profit_factor": _profit_factor(vals),
                    "baseline_match_rate": float(min(1.0, vals.size / max(len(ev) * baseline_samples, 1))),
                }
            )
            prog.update(done)
    return pd.DataFrame(rows)


def build_decision_table(
    replay_summary: pd.DataFrame,
    cost_stress: pd.DataFrame,
    delay_stress: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    *,
    min_count: int,
    min_events_per_year: float,
) -> pd.DataFrame:
    if replay_summary.empty:
        return pd.DataFrame()
    df = replay_summary.copy()
    if not cost_stress.empty:
        fee2 = cost_stress[cost_stress["cost_multiplier"].astype(float) == 2.0][["event_name", "direction", "horizon", "mean_net"]].rename(columns={"mean_net": "fee2_mean_net"})
        df = df.merge(fee2, on=["event_name", "direction", "horizon"], how="left")
    else:
        df["fee2_mean_net"] = np.nan
    if not delay_stress.empty:
        d1 = delay_stress[delay_stress["delay_bars"].astype(int) == 1][["event_name", "direction", "horizon", "mean_net"]].rename(columns={"mean_net": "delay1_mean_net"})
        df = df.merge(d1, on=["event_name", "direction", "horizon"], how="left")
    else:
        df["delay1_mean_net"] = np.nan
    if not baseline_summary.empty:
        bs = baseline_summary[["event_name", "direction", "horizon", "baseline_mean_net", "matched_excess_mean_net", "baseline_match_rate"]]
        df = df.merge(bs, on=["event_name", "direction", "horizon"], how="left")
    else:
        df["baseline_mean_net"] = np.nan
        df["matched_excess_mean_net"] = np.nan
        df["baseline_match_rate"] = np.nan

    decisions: list[str] = []
    reasons: list[str] = []
    for r in df.itertuples(index=False):
        reason_parts: list[str] = []
        ok = True
        if int(r.count) < min_count:
            ok = False; reason_parts.append("count_lt_min")
        if float(r.events_per_year) < min_events_per_year:
            ok = False; reason_parts.append("frequency_lt_min")
        if float(r.mean_net) <= 0:
            ok = False; reason_parts.append("mean_net_le_0")
        if float(r.profit_factor) < 1.15:
            ok = False; reason_parts.append("pf_lt_1p15")
        if not np.isfinite(float(getattr(r, "fee2_mean_net", np.nan))) or float(r.fee2_mean_net) <= -0.0005:
            ok = False; reason_parts.append("fee2_weak")
        if not np.isfinite(float(getattr(r, "delay1_mean_net", np.nan))) or float(r.delay1_mean_net) <= -0.0005:
            ok = False; reason_parts.append("delay1_weak")
        if int(r.positive_years) < 3:
            ok = False; reason_parts.append("positive_years_lt_3")
        if float(r.top5_winner_share) > 0.35:
            ok = False; reason_parts.append("top5_winner_share_gt_0p35")
        mex = getattr(r, "matched_excess_mean_net", np.nan)
        if np.isfinite(float(mex)):
            if float(mex) <= 0:
                ok = False; reason_parts.append("matched_excess_le_0")
        else:
            # If no baseline was run, it means the group failed preliminary filters.
            ok = False; reason_parts.append("no_matched_baseline_prelim_failed")
        decisions.append("research_continue" if ok else "rejected")
        reasons.append(";".join(reason_parts) if reason_parts else "passes_research_continue_filters")
    df["decision"] = decisions
    df["reason"] = reasons
    return df.sort_values(["decision", "mean_net"], ascending=[True, False]).reset_index(drop=True)


def build_causal_audit(df: pd.DataFrame, events: pd.DataFrame, *, max_horizon: int, sample_size: int) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    sample = events.head(sample_size).copy()
    n = len(df)
    signal_idx = sample["signal_idx"].to_numpy(dtype=np.int64)
    entry_idx = signal_idx + 1
    expected_entry_idx = signal_idx + 1
    forward_valid = entry_idx + max_horizon < n
    out = pd.DataFrame(
        {
            "event_name": sample["event_name"].to_numpy(),
            "direction": sample["direction"].to_numpy(),
            "signal_idx": signal_idx,
            "signal_time": sample["signal_time"].to_numpy(),
            "entry_idx": entry_idx,
            "entry_time": df.index[np.clip(entry_idx, 0, n - 1)],
            "expected_entry_idx": expected_entry_idx,
            "expected_entry_time": df.index[np.clip(expected_entry_idx, 0, n - 1)],
            "entry_not_next_open_flag": entry_idx != expected_entry_idx,
            "used_context_time": sample["signal_time"].to_numpy(),
            "used_context_available_time": sample["signal_time"].to_numpy(),
            "context_available_time_flag": False,
            "forward_window_valid_flag": forward_valid,
        }
    )
    out["lookahead_flag"] = out["entry_not_next_open_flag"] | out["context_available_time_flag"] | (~out["forward_window_valid_flag"])
    return out


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _write_json(path: Path, data: dict[str, object]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_reports(
    args: argparse.Namespace,
    *,
    df: pd.DataFrame,
    feature_cols: list[str],
    specs: list[EventSpec],
    events: pd.DataFrame,
    event_summary: pd.DataFrame,
    replay_summary: pd.DataFrame,
    cost_stress: pd.DataFrame,
    delay_stress: pd.DataFrame,
    yearly: pd.DataFrame,
    trades_sample: pd.DataFrame,
    baseline_summary: pd.DataFrame,
    decision: pd.DataFrame,
    causal_audit: pd.DataFrame,
) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_count = int((decision["decision"] == "research_continue").sum()) if not decision.empty and "decision" in decision else 0
    manifest = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "edge_status": "research_only_not_tradable",
        "symbol": args.symbol,
        "primary_timeframe": args.primary_timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "horizons": list(_parse_csv_ints(args.horizons)),
        "round_trip_cost_pct": args.round_trip_cost_pct,
        "cost_multipliers": list(_parse_csv_floats(args.cost_multipliers)),
        "delay_bars_list": list(_parse_csv_ints(args.delay_bars_list)),
        "input_rows": int(len(df)),
        "event_spec_count": int(len(specs)),
        "event_count": int(len(events)),
        "replay_group_count": int(len(replay_summary)),
        "candidate_count": candidate_count,
        "causal_lookahead_count": int(causal_audit["lookahead_flag"].sum()) if not causal_audit.empty else 0,
        "causal_policy": CAUSAL_POLICY,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    _write_json(out_dir / "00_manifest.json", manifest)
    event_summary.to_csv(out_dir / "01_event_summary.csv", index=False)
    replay_summary.to_csv(out_dir / "02_replay_variant_summary.csv", index=False)
    if args.write_full_trades:
        trades_sample.to_csv(out_dir / "03_replay_trades.csv", index=False)
    else:
        trades_sample.head(int(args.trade_sample_size)).to_csv(out_dir / "03_replay_trades_sample.csv", index=False)
    yearly.to_csv(out_dir / "04_yearly_breakdown.csv", index=False)
    # Useful static breakdowns from delay0 replay groups.
    if not trades_sample.empty:
        session = trades_sample.groupby(["event_name", "direction", "horizon", "session"], sort=False)["net_return"].agg(["count", "mean", "median"]).reset_index()
        session.rename(columns={"mean": "mean_net", "median": "median_net"}, inplace=True)
        regime = trades_sample.groupby(["event_name", "direction", "horizon", "regime"], sort=False)["net_return"].agg(["count", "mean", "median"]).reset_index()
        regime.rename(columns={"mean": "mean_net", "median": "median_net"}, inplace=True)
    else:
        session = pd.DataFrame(); regime = pd.DataFrame()
    session.to_csv(out_dir / "05_session_breakdown.csv", index=False)
    regime.to_csv(out_dir / "06_regime_breakdown.csv", index=False)
    cost_stress.to_csv(out_dir / "07_cost_stress.csv", index=False)
    delay_stress.to_csv(out_dir / "08_delay_stress.csv", index=False)
    baseline_summary.to_csv(out_dir / "09_matched_baseline_summary.csv", index=False)
    decision.to_csv(out_dir / "10_research_decision.csv", index=False)
    # Candidate/rejected aliases for standard review tools.
    if not decision.empty:
        decision[decision["decision"] == "research_continue"].to_csv(out_dir / "10_candidate_shortlist.csv", index=False)
        decision[decision["decision"] != "research_continue"].to_csv(out_dir / "11_rejected_candidates.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "10_candidate_shortlist.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "11_rejected_candidates.csv", index=False)
    causal_audit.to_csv(out_dir / "12_causal_audit.csv", index=False)
    events.head(int(args.event_sample_size)).to_csv(out_dir / "13_event_sample.csv", index=False)
    _write_json(out_dir / "14_feature_columns.json", {"feature_columns": feature_cols})
    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title="ETH MHF Volatility Compression Fade V1")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    print(f"[run] {SCRIPT_NAME} version={SCRIPT_VERSION}", flush=True)
    print(f"[run] symbol={args.symbol} primary={args.primary_timeframe} range={args.start_date}->{args.end_date} warmup={args.warmup_start_date}", flush=True)
    print("[load] primary trade bars", flush=True)
    df_raw = load_trade_bars(args)
    print(f"[load] primary rows={len(df_raw):,} range={df_raw.index.min()} -> {df_raw.index.max()}", flush=True)

    print("[features] building volatility/orderflow features", flush=True)
    df, feature_cols = build_features(df_raw, long_vol_window=int(args.long_vol_window))
    df = df.loc[df.index <= pd.Timestamp(args.end_date) + pd.Timedelta(hours=23, minutes=59)].copy()
    research_df = df.loc[df.index >= pd.Timestamp(args.start_date)]
    print(f"[features] research rows={len(research_df):,} feature_columns={len(feature_cols)}", flush=True)

    print("[events] building volatility compression/expansion fade events", flush=True)
    specs = build_event_specs(args)
    events = build_events(
        df,
        specs,
        start_date=args.start_date,
        cooldown_bars=int(args.cooldown_bars),
        progress_every=int(args.progress_every),
    )
    print(f"[events] rows={len(events):,} specs={len(specs)}", flush=True)

    horizons = _parse_csv_ints(args.horizons)
    delay_bars_list = _parse_csv_ints(args.delay_bars_list)
    cost_multipliers = _parse_csv_floats(args.cost_multipliers)

    print("[forward] fixed-hold replay", flush=True)
    event_summary, replay_summary, cost_stress, delay_stress, yearly, trades_sample = build_replay_tables(
        df,
        events,
        horizons=horizons,
        delay_bars_list=delay_bars_list,
        round_trip_cost_pct=float(args.round_trip_cost_pct),
        cost_multipliers=cost_multipliers,
        start_date=args.start_date,
        end_date=args.end_date,
        min_count=int(args.min_count),
        min_events_per_year=float(args.min_events_per_year),
        progress_every=int(args.progress_every),
    )
    print(f"[forward] replay groups={len(replay_summary):,}", flush=True)

    print("[aggregate] matched baseline prep", flush=True)
    baseline_summary = build_baseline_summary(
        df,
        events,
        replay_summary,
        horizons=horizons,
        round_trip_cost_pct=float(args.round_trip_cost_pct),
        baseline_samples=int(args.baseline_samples),
        baseline_max_events_per_group=int(args.baseline_max_events_per_group),
        baseline_prefilter_mean_net=float(args.baseline_prefilter_mean_net),
        baseline_prefilter_pf=float(args.baseline_prefilter_pf),
        seed=int(args.baseline_seed),
        progress_every=int(args.progress_every),
    )

    print("[aggregate] research decision", flush=True)
    decision = build_decision_table(
        replay_summary,
        cost_stress,
        delay_stress,
        baseline_summary,
        min_count=int(args.min_count),
        min_events_per_year=float(args.min_events_per_year),
    )

    print("[aggregate] causal audit", flush=True)
    causal_audit = build_causal_audit(df, events, max_horizon=max(horizons or (0,)) + max(delay_bars_list or (0,)) + 1, sample_size=max(int(args.event_sample_size), 1000))

    print("[write] report files", flush=True)
    write_reports(
        args,
        df=df,
        feature_cols=feature_cols,
        specs=specs,
        events=events,
        event_summary=event_summary,
        replay_summary=replay_summary,
        cost_stress=cost_stress,
        delay_stress=delay_stress,
        yearly=yearly,
        trades_sample=trades_sample,
        baseline_summary=baseline_summary,
        decision=decision,
        causal_audit=causal_audit,
    )
    print(f"[done] report_dir={args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
