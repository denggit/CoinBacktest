#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
0.2% Range-Bar 5-Bar Directional Run Research
=============================================

Research goal
-------------
Find every 0.2% range-bar sequence where ETH runs 5 consecutive bars in the
same direction (roughly 1%+ directional movement), then study which causal
signals visible before the first run bar could have entered before the 5-bar
move.

Causal timing
-------------
For a signal on range bar i:
    signal_time = bar_i.end_ts                 # bar i is fully closed
    entry_bar   = bar_{i+1}                    # next range bar only
    entry_price = bar_{i+1}.open
    target      = directions of bars i+1..i+5 are all equal to trade side
    exit_price  = bar_{i+5}.close              # fixed 5 range-bar horizon

The target is an evaluation label only. All signal features are calculated from
bar i and older bars. Footprint features, when enabled, are aggregated by bar_id
and shifted by one bar before being used as context, making them visible only
after the range bar has closed.

Example
-------
python research/range5_directional_run_research.py --symbol ETH-USDT-SWAP --range-pct 0.0020 --start-date 2023-01-01 --end-date 2026-06-30 --warmup-start-date 2022-01-01 --include-footprint --footprint-price-step 1 --out-dir data/reports/research/range5_directional_run_r0020
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

warnings.simplefilter("ignore", PerformanceWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader, range_code  # noqa: E402
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402


@dataclass(frozen=True)
class SignalSpec:
    signal_id: str
    family: str
    side: int
    trigger_keys: tuple[str, ...]
    context_keys: tuple[str, ...] = ()
    description: str = ""


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Research causal signals before 5 consecutive same-direction 0.2% range bars.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--range-pct", type=float, default=0.0020)
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--out-dir", default="data/reports/research/range5_directional_run_r0020")
    p.add_argument("--target-run-bars", type=int, default=5)
    p.add_argument("--feature-windows", default="3,5,8,13,21,34,55,89")
    p.add_argument("--min-count", type=int, default=80)
    p.add_argument("--min-year-count", type=int, default=10)
    p.add_argument("--fee-rate", type=float, default=0.0011, help="Round-trip fee deducted from fixed-horizon signed return.")
    p.add_argument("--include-footprint", action="store_true")
    p.add_argument("--footprint-price-step", type=float, default=1.0)
    p.add_argument("--max-contexts-per-trigger", type=int, default=18)
    p.add_argument("--max-combo-contexts", type=int, default=2)
    p.add_argument("--write-full-signal-events", action="store_true", help="Write all signal-level candidate events. Can be large.")
    p.add_argument("--event-sample-size", type=int, default=100_000)
    p.add_argument("--candidate-top-n", type=int, default=500)
    p.add_argument("--progress", action="store_true", default=True)
    return p.parse_args(argv)


def parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v <= 0:
            raise ValueError("windows must be positive")
        values.append(v)
    if not values:
        raise ValueError("feature windows must not be empty")
    return sorted(set(values))


def side_name(side: int) -> str:
    return "LONG" if int(side) == 1 else "SHORT"


def safe_side_token(side: int) -> str:
    return "long" if int(side) == 1 else "short"


def profit_factor(x: pd.Series | np.ndarray) -> float:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    if arr.size == 0:
        return float("nan")
    gp = float(arr[arr > 0].sum())
    gl = float(-arr[arr <= 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else float("nan")
    return gp / gl


def top_winner_share(x: pd.Series | np.ndarray, n: int = 5) -> float:
    arr = pd.to_numeric(pd.Series(x), errors="coerce").dropna().to_numpy(float)
    wins = np.sort(arr[arr > 0])[::-1]
    if wins.size == 0:
        return float("nan")
    total = float(wins.sum())
    return float(wins[: int(n)].sum() / total) if total > 0 else float("nan")


def summarize_values(x: pd.Series | np.ndarray, *, min_count: int = 0) -> dict[str, object]:
    s = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    if s.empty:
        return {
            "count": 0,
            "eligible": False,
            "mean": np.nan,
            "median": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "top5_winner_share": np.nan,
            "p05": np.nan,
            "p25": np.nan,
            "p75": np.nan,
            "p95": np.nan,
        }
    return {
        "count": int(len(s)),
        "eligible": bool(len(s) >= int(min_count)),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "win_rate": float((s > 0).mean()),
        "profit_factor": profit_factor(s),
        "top5_winner_share": top_winner_share(s),
        "p05": float(s.quantile(0.05)),
        "p25": float(s.quantile(0.25)),
        "p75": float(s.quantile(0.75)),
        "p95": float(s.quantile(0.95)),
    }


def signed_simple_return(side: np.ndarray, entry: np.ndarray, exit_: np.ndarray) -> np.ndarray:
    out = np.full(len(side), np.nan, dtype=float)
    ok = np.isfinite(entry) & np.isfinite(exit_) & (entry > 0) & (exit_ > 0) & (side != 0)
    out[ok] = side[ok] * (exit_[ok] / entry[ok] - 1.0)
    return out


def run_length_signed(direction: pd.Series) -> pd.Series:
    d = pd.to_numeric(direction, errors="coerce").fillna(0).astype(int)
    grp = (d != d.shift()).cumsum()
    length = d.groupby(grp).cumcount() + 1
    return (length * d.where(d != 0, 0)).astype(int)


def streak_id(direction: pd.Series) -> pd.Series:
    d = pd.to_numeric(direction, errors="coerce").fillna(0).astype(int)
    return (d != d.shift()).cumsum().astype(int)


def rolling_rank_ratio(s: pd.Series, window: int) -> pd.Series:
    # Approximate rolling percentile of current value versus prior values.
    prior = s.shift(1)
    med = prior.rolling(window, min_periods=max(5, window // 3)).median()
    return pd.to_numeric(s, errors="coerce") / med.replace(0.0, np.nan)


def load_range_bars(args: argparse.Namespace) -> pd.DataFrame:
    loader = OKXRangeBarLoader(symbol=args.symbol, range_pct=float(args.range_pct))
    df = loader.fetch_data_by_date_range(args.warmup_start_date, args.end_date, cvd_mode="range")
    if df.empty:
        raise RuntimeError("No range bars loaded. Build range bars first or check DB path.")

    # OKXRangeBarLoader._finalize_return_df returns end_ts both as index name
    # and as a retained column (set_index(drop=False)).  Reset the index before
    # sorting by the column, otherwise pandas raises:
    #   ValueError: 'end_ts' is both an index level and a column label
    df = df.reset_index(drop=True).copy()
    for col in ["start_ts", "end_ts"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["bar_id"] = pd.to_numeric(df["bar_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["start_ts", "end_ts", "bar_id"])
    df["bar_id"] = df["bar_id"].astype("int64")
    df = df.sort_values(["end_ts", "bar_id"]).set_index("end_ts", drop=False)
    df.index.name = "end_ts"
    # Keep only rows needed for warmup->end. start-date filtering is later.
    print(
        f"[load] range bars range={range_code(args.range_pct)} rows={len(df):,} "
        f"range={df['end_ts'].min()} -> {df['end_ts'].max()}"
    )
    return df


def load_footprint_features(args: argparse.Namespace, bar_ids: Sequence[int]) -> pd.DataFrame:
    loader = OKXRangeFootprintLoader(
        symbol=args.symbol,
        range_pct=float(args.range_pct),
        price_step=float(args.footprint_price_step),
    )
    fp = loader.fetch_data_by_date_range(args.warmup_start_date, args.end_date)
    if fp.empty:
        print("[footprint] empty; continue without footprint features")
        return pd.DataFrame()
    # Normalize defensively: footprint loader can return sorted columns without an
    # end_ts index today, but keep this robust against future loader changes.
    fp = fp.reset_index(drop=True).copy()
    fp["bar_id"] = pd.to_numeric(fp["bar_id"], errors="coerce").astype("Int64")
    fp = fp.dropna(subset=["bar_id"])
    fp["bar_id"] = fp["bar_id"].astype("int64")
    want = pd.Index(pd.Series(bar_ids, dtype="int64").unique())
    fp = fp[fp["bar_id"].isin(want)].copy()
    if fp.empty:
        print("[footprint] no overlapping footprint rows after bar_id filter")
        return pd.DataFrame()
    print(f"[footprint] rows={len(fp):,} unique_bars={fp['bar_id'].nunique():,}")

    # Vectorized per-bar footprint aggregation.  The first version used a Python
    # groupby loop over every bar_id; on a multi-year 0.2% range-bar dataset that
    # can be hundreds of thousands of groups.  Keep the same feature names but do
    # the heavy work with groupby aggregations.
    for col in [
        "price_bucket", "notional", "delta_notional", "large_delta_notional",
    ]:
        fp[col] = pd.to_numeric(fp[col], errors="coerce").fillna(0.0)
    fp = fp.sort_values(["bar_id", "price_bucket"]).reset_index(drop=True)
    g = fp.groupby("bar_id", sort=False)
    size = g["price_bucket"].transform("size").astype(int)
    pos_in_bar = g.cumcount().astype(int)
    k = np.ceil(size.to_numpy(float) * 0.25).astype(int)
    k = np.maximum(k, 1)
    bottom_mask = pos_in_bar.to_numpy() < k
    top_mask = pos_in_bar.to_numpy() >= (size.to_numpy(int) - k)

    delta = fp["delta_notional"].to_numpy(float)
    notional = fp["notional"].to_numpy(float)
    abs_delta = np.abs(delta)
    fp["_abs_delta_notional"] = abs_delta
    fp["_top_delta_notional"] = np.where(top_mask, delta, 0.0)
    fp["_bottom_delta_notional"] = np.where(bottom_mask, delta, 0.0)
    fp["_top_large_delta_notional"] = np.where(top_mask, fp["large_delta_notional"].to_numpy(float), 0.0)
    fp["_bottom_large_delta_notional"] = np.where(bottom_mask, fp["large_delta_notional"].to_numpy(float), 0.0)

    agg = fp.groupby("bar_id", sort=False).agg(
        fp_levels=("price_bucket", "size"),
        fp_delta_notional_sum=("delta_notional", "sum"),
        fp_abs_delta_notional_sum=("_abs_delta_notional", "sum"),
        _fp_abs_delta_max=("_abs_delta_notional", "max"),
        _fp_total_notional=("notional", "sum"),
        fp_top_delta_notional=("_top_delta_notional", "sum"),
        fp_bottom_delta_notional=("_bottom_delta_notional", "sum"),
        fp_top_large_delta_notional=("_top_large_delta_notional", "sum"),
        fp_bottom_large_delta_notional=("_bottom_large_delta_notional", "sum"),
    )
    denom = agg["_fp_total_notional"].replace(0.0, np.nan)
    agg["fp_delta_concentration"] = (agg["_fp_abs_delta_max"] / denom).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    agg = agg.drop(columns=["_fp_abs_delta_max", "_fp_total_notional"])

    # Max consecutive positive/negative delta levels per bar, vectorized enough
    # for this dataset.  This preserves the previous stacked-level semantics.
    sign_pos = (fp["delta_notional"] > 0).astype("int8")
    sign_neg = (fp["delta_notional"] < 0).astype("int8")
    bar_change = fp["bar_id"].ne(fp["bar_id"].shift())

    def _max_consecutive(flag: pd.Series) -> pd.Series:
        run_start = bar_change | flag.ne(flag.shift())
        run_id = run_start.cumsum()
        run_len = flag.groupby([fp["bar_id"], run_id], sort=False).transform("sum")
        run_len = run_len.where(flag.astype(bool), 0)
        return run_len.groupby(fp["bar_id"], sort=False).max().astype(float)

    agg["fp_max_buy_stacked_levels"] = _max_consecutive(sign_pos)
    agg["fp_max_sell_stacked_levels"] = _max_consecutive(sign_neg)
    agg = agg.fillna(0.0)
    return agg.sort_index()


def add_session_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ts = pd.to_datetime(out["end_ts"], errors="coerce")
    out["date"] = ts.dt.date.astype(str)
    out["year"] = ts.dt.year.astype("Int64")
    out["hour"] = ts.dt.hour.astype("Int64")
    # Data is aligned with project timezone, usually UTC+8. Keep sessions simple and deterministic.
    h = ts.dt.hour.fillna(-1).astype(int)
    out["session_asia"] = ((h >= 8) & (h < 16))
    out["session_europe"] = ((h >= 16) & (h < 21))
    out["session_us_open"] = ((h >= 21) | (h < 1))
    out["session_us_late"] = ((h >= 1) & (h < 5))
    return out


def build_features(bars: pd.DataFrame, args: argparse.Namespace, windows: list[int]) -> pd.DataFrame:
    df = bars.copy().reset_index(drop=True).sort_values(["end_ts", "bar_id"]).reset_index(drop=True)
    num_cols = [
        "open", "high", "low", "close", "duration_seconds", "range_pct", "volume", "notional", "trades_count",
        "buy_notional", "sell_notional", "delta_notional", "cvd_notional", "taker_buy_ratio", "large_buy_notional",
        "large_sell_notional", "large_delta_notional", "large_trades_count", "vwap", "max_trade_notional",
    ]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    df["direction"] = pd.to_numeric(df["direction"], errors="coerce").fillna(0).astype(int)
    # Drop zero-direction bars for target sequences but keep basic features stable.
    df["signed_run_len"] = run_length_signed(df["direction"])
    df["abs_run_len"] = df["signed_run_len"].abs().astype(int)
    df["streak_id"] = streak_id(df["direction"])
    df["prev_direction"] = df["direction"].shift(1).fillna(0).astype(int)
    df["reversal_bar"] = (df["direction"] != 0) & (df["prev_direction"] == -df["direction"])

    df["bar_ret"] = df["close"] / df["open"].replace(0.0, np.nan) - 1.0
    df["delta_ratio"] = df["delta_notional"] / df["notional"].replace(0.0, np.nan)
    df["large_delta_ratio"] = df["large_delta_notional"] / df["notional"].replace(0.0, np.nan)
    df["large_trade_ratio"] = df["large_trades_count"] / df["trades_count"].replace(0.0, np.nan)
    df["taker_buy_ratio"] = df["taker_buy_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.5)

    # Rolling context uses shift(1) where it represents prior context, and current bar only where signal bar is closed.
    for w in windows:
        minp = max(3, min(w, w // 2))
        df[f"dir_sum_{w}"] = df["direction"].shift(1).rolling(w, min_periods=minp).sum()
        df[f"dir_consistency_{w}"] = (df[f"dir_sum_{w}"].abs() / float(w)).replace([np.inf, -np.inf], np.nan)
        df[f"ret_{w}"] = df["close"] / df["close"].shift(w) - 1.0
        df[f"delta_sum_{w}"] = df["delta_notional"].shift(1).rolling(w, min_periods=minp).sum()
        df[f"large_delta_sum_{w}"] = df["large_delta_notional"].shift(1).rolling(w, min_periods=minp).sum()
        df[f"notional_med_{w}"] = df["notional"].shift(1).rolling(w, min_periods=minp).median()
        df[f"notional_ratio_{w}"] = df["notional"] / df[f"notional_med_{w}"].replace(0.0, np.nan)
        df[f"duration_med_{w}"] = df["duration_seconds"].shift(1).rolling(w, min_periods=minp).median()
        df[f"duration_ratio_{w}"] = df["duration_seconds"] / df[f"duration_med_{w}"].replace(0.0, np.nan)
        df[f"prior_duration_med_{w}"] = df["duration_seconds"].shift(1).rolling(w, min_periods=minp).median()
        df[f"prior_notional_med_{w}"] = df["notional"].shift(1).rolling(w, min_periods=minp).median()
        df[f"prior_high_{w}"] = df["high"].shift(1).rolling(w, min_periods=minp).max()
        df[f"prior_low_{w}"] = df["low"].shift(1).rolling(w, min_periods=minp).min()
        df[f"break_high_{w}"] = df["close"] > df[f"prior_high_{w}"]
        df[f"break_low_{w}"] = df["close"] < df[f"prior_low_{w}"]
        df[f"sweep_high_reject_{w}"] = (df["high"] > df[f"prior_high_{w}"]) & (df["close"] < df[f"prior_high_{w}"])
        df[f"sweep_low_reclaim_{w}"] = (df["low"] < df[f"prior_low_{w}"]) & (df["close"] > df[f"prior_low_{w}"])

    # Broader reference context.
    for span in [21, 55, 144]:
        df[f"ema_{span}"] = df["close"].ewm(span=span, adjust=False, min_periods=min(span, 30)).mean()
        df[f"close_above_ema_{span}"] = df["close"] > df[f"ema_{span}"]
        df[f"close_below_ema_{span}"] = df["close"] < df[f"ema_{span}"]

    # Rank-like robust flags based on standard windows.
    for w in [21, 55, 89]:
        if w in windows or w in {21, 55, 89}:
            dur_ratio = df["duration_seconds"] / df["duration_seconds"].shift(1).rolling(w, min_periods=max(5, w // 3)).median().replace(0.0, np.nan)
            not_ratio = df["notional"] / df["notional"].shift(1).rolling(w, min_periods=max(5, w // 3)).median().replace(0.0, np.nan)
            df[f"fast_bar_{w}"] = dur_ratio <= 0.55
            df[f"slow_bar_{w}"] = dur_ratio >= 1.80
            df[f"notional_spike_{w}"] = not_ratio >= 1.75
            df[f"notional_dry_{w}"] = not_ratio <= 0.65
            df[f"duration_ratio_now_{w}"] = dur_ratio
            df[f"notional_ratio_now_{w}"] = not_ratio

    return add_session_features(df.copy())


def merge_footprint(feat: pd.DataFrame, fp: pd.DataFrame) -> pd.DataFrame:
    if fp.empty:
        return feat
    out = feat.merge(fp, how="left", left_on="bar_id", right_index=True)
    fp_cols = [c for c in fp.columns if c.startswith("fp_")]
    for col in fp_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        # Make footprint context conservatively visible only after this range bar closes.
        out[f"{col}_prev"] = out[col].shift(1)
    return out


def add_targets(feat: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    out = feat.copy().reset_index(drop=True)
    n = int(args.target_run_bars)
    d = out["direction"].to_numpy(dtype=int)
    m = len(out)
    long_hit = np.zeros(m, dtype=bool)
    short_hit = np.zeros(m, dtype=bool)
    # signal at i, entry starts at i+1, target bars are i+1..i+n
    for i in range(0, max(0, m - n - 1)):
        path = d[i + 1 : i + 1 + n]
        long_hit[i] = bool(path.size == n and np.all(path == 1))
        short_hit[i] = bool(path.size == n and np.all(path == -1))
    out["target_next5_long"] = long_hit
    out["target_next5_short"] = short_hit
    out["target_next5_any"] = long_hit | short_hit
    out["target_next5_side"] = np.select([long_hit, short_hit], [1, -1], default=0).astype(int)

    entry_pos = np.arange(m) + 1
    exit_pos = np.arange(m) + n
    valid = exit_pos < m
    out["entry_bar_pos"] = pd.Series(np.where(entry_pos < m, entry_pos, np.nan), dtype="Float64")
    out["exit_bar_pos"] = pd.Series(np.where(valid, exit_pos, np.nan), dtype="Float64")
    open_arr = out["open"].to_numpy(float)
    close_arr = out["close"].to_numpy(float)
    high_arr = out["high"].to_numpy(float)
    low_arr = out["low"].to_numpy(float)
    start_ts_arr = pd.to_datetime(out["start_ts"]).to_numpy(dtype="datetime64[ns]")
    end_ts_arr = pd.to_datetime(out["end_ts"]).to_numpy(dtype="datetime64[ns]")

    entry_price = np.full(m, np.nan)
    exit_price = np.full(m, np.nan)
    entry_time = np.full(m, np.datetime64("NaT"), dtype="datetime64[ns]")
    exit_time = np.full(m, np.datetime64("NaT"), dtype="datetime64[ns]")
    valid_entry = entry_pos < m
    entry_price[valid_entry] = open_arr[entry_pos[valid_entry]]
    entry_time[valid_entry] = start_ts_arr[entry_pos[valid_entry]]
    exit_price[valid] = close_arr[exit_pos[valid]]
    exit_time[valid] = end_ts_arr[exit_pos[valid]]
    out["entry_time"] = pd.to_datetime(entry_time)
    out["exit_time_h5rb"] = pd.to_datetime(exit_time)
    out["entry_price"] = entry_price
    out["exit_price_h5rb"] = exit_price

    side_long = np.ones(m, dtype=int)
    side_short = -np.ones(m, dtype=int)
    out["ret_h5rb_long_gross"] = signed_simple_return(side_long, entry_price, exit_price)
    out["ret_h5rb_short_gross"] = signed_simple_return(side_short, entry_price, exit_price)
    out["ret_h5rb_long_net"] = out["ret_h5rb_long_gross"] - float(args.fee_rate)
    out["ret_h5rb_short_net"] = out["ret_h5rb_short_gross"] - float(args.fee_rate)

    # Fast path MFE / MAE for fixed n=5. n is small, so loop over offsets only.
    for side, name in [(1, "long"), (-1, "short")]:
        mfe = np.full(m, np.nan)
        mae = np.full(m, np.nan)
        for i in range(0, max(0, m - n - 1)):
            ep = entry_price[i]
            if not np.isfinite(ep) or ep <= 0:
                continue
            sl = slice(i + 1, i + 1 + n)
            mx = np.nanmax(high_arr[sl])
            mn = np.nanmin(low_arr[sl])
            if side == 1:
                mfe[i] = mx / ep - 1.0
                mae[i] = mn / ep - 1.0
            else:
                mfe[i] = ep / mn - 1.0 if mn > 0 else np.nan
                mae[i] = ep / mx - 1.0 if mx > 0 else np.nan
        out[f"mfe_h5rb_{name}"] = mfe
        out[f"mae_h5rb_{name}"] = mae

    return out


def build_actual_run_tables(feat: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = int(args.target_run_bars)
    rows = []
    starts = []
    d = feat["direction"].to_numpy(dtype=int)
    for j in range(0, len(feat) - n + 1):
        path = d[j : j + n]
        side = 1 if np.all(path == 1) else -1 if np.all(path == -1) else 0
        if side == 0:
            continue
        signal_i = j - 1
        if signal_i < 0:
            continue
        prev_same = bool(j > 0 and d[j - 1] == side)
        row = {
            "run_start_pos": j,
            "run_end_pos": j + n - 1,
            "side": side_name(side),
            "side_int": side,
            "is_streak_start": not prev_same,
            "signal_bar_pos": signal_i,
            "signal_time": feat.loc[signal_i, "end_ts"],
            "entry_time": feat.loc[j, "start_ts"],
            "entry_price": feat.loc[j, "open"],
            "run_end_time": feat.loc[j + n - 1, "end_ts"],
            "run_exit_price": feat.loc[j + n - 1, "close"],
            "run_duration_seconds": float((pd.Timestamp(feat.loc[j + n - 1, "end_ts"]) - pd.Timestamp(feat.loc[j, "start_ts"])).total_seconds()),
            "gross_run_return": side * (float(feat.loc[j + n - 1, "close"]) / float(feat.loc[j, "open"]) - 1.0),
            "date": feat.loc[signal_i, "date"],
            "year": feat.loc[signal_i, "year"],
            "pre_abs_run_len": feat.loc[signal_i, "abs_run_len"],
            "pre_signed_run_len": feat.loc[signal_i, "signed_run_len"],
            "pre_direction": feat.loc[signal_i, "direction"],
            "pre_duration_seconds": feat.loc[signal_i, "duration_seconds"],
            "pre_notional": feat.loc[signal_i, "notional"],
            "pre_delta_ratio": feat.loc[signal_i, "delta_ratio"],
            "pre_large_delta_ratio": feat.loc[signal_i, "large_delta_ratio"],
            "pre_taker_buy_ratio": feat.loc[signal_i, "taker_buy_ratio"],
        }
        rows.append(row)
        if not prev_same:
            starts.append(row.copy())
    return pd.DataFrame(rows), pd.DataFrame(starts)


def bool_series(x: pd.Series) -> pd.Series:
    return x.astype("boolean").fillna(False).astype(bool)


def build_mask_library(feat: pd.DataFrame, windows: list[int]) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    d = feat["direction"].to_numpy(dtype=int)
    prev_d = feat["prev_direction"].fillna(0).to_numpy(dtype=int)
    signed_run = feat["signed_run_len"].fillna(0).to_numpy(dtype=int)
    abs_run = np.abs(signed_run)
    taker = feat["taker_buy_ratio"].fillna(0.5).to_numpy(float)
    delta = feat["delta_notional"].fillna(0.0).to_numpy(float)
    large_delta = feat["large_delta_notional"].fillna(0.0).to_numpy(float)
    delta_ratio = feat["delta_ratio"].fillna(0.0).to_numpy(float)
    large_delta_ratio = feat["large_delta_ratio"].fillna(0.0).to_numpy(float)

    for side in [1, -1]:
        st = safe_side_token(side)
        masks[f"cur_{st}"] = d == side
        masks[f"reversal_to_{st}"] = (d == side) & (prev_d == -side)
        masks[f"streak_ge2_{st}"] = (d == side) & (signed_run * side >= 2)
        masks[f"streak_ge3_{st}"] = (d == side) & (signed_run * side >= 3)
        masks[f"fresh_after_opposite_{st}"] = (d == side) & (prev_d == -side) & (abs_run == 1)
        masks[f"delta_aligned_{st}"] = (d == side) & (delta * side > 0)
        masks[f"large_delta_aligned_{st}"] = (d == side) & (large_delta * side > 0)
        masks[f"strong_delta_ratio_{st}"] = delta_ratio * side >= 0.18
        masks[f"strong_large_delta_ratio_{st}"] = large_delta_ratio * side >= 0.08
        if side == 1:
            masks[f"taker_dominant_{st}"] = taker >= 0.58
            masks[f"taker_extreme_{st}"] = taker >= 0.66
        else:
            masks[f"taker_dominant_{st}"] = taker <= 0.42
            masks[f"taker_extreme_{st}"] = taker <= 0.34

        for w in windows:
            if f"break_high_{w}" in feat.columns:
                masks[f"break_{w}_{st}"] = bool_series(feat[f"break_high_{w}" if side == 1 else f"break_low_{w}"]).to_numpy()
                masks[f"sweep_reclaim_{w}_{st}"] = bool_series(feat[f"sweep_low_reclaim_{w}" if side == 1 else f"sweep_high_reject_{w}"]).to_numpy()
            if f"dir_sum_{w}" in feat.columns:
                masks[f"prior_trend_aligned_{w}_{st}"] = (feat[f"dir_sum_{w}"].fillna(0.0).to_numpy(float) * side) >= max(2.0, w * 0.35)
                masks[f"prior_trend_opposed_{w}_{st}"] = (feat[f"dir_sum_{w}"].fillna(0.0).to_numpy(float) * side) <= -max(2.0, w * 0.35)
                masks[f"prior_chop_{w}_{st}"] = feat[f"dir_consistency_{w}"].fillna(0.0).to_numpy(float) <= 0.25
            if f"ret_{w}" in feat.columns:
                masks[f"prior_price_momo_{w}_{st}"] = feat[f"ret_{w}"].fillna(0.0).to_numpy(float) * side >= float(w) * 0.00045
                masks[f"prior_price_fade_{w}_{st}"] = feat[f"ret_{w}"].fillna(0.0).to_numpy(float) * side <= -float(w) * 0.00045
            if f"delta_sum_{w}" in feat.columns:
                masks[f"prior_delta_aligned_{w}_{st}"] = feat[f"delta_sum_{w}"].fillna(0.0).to_numpy(float) * side > 0
                masks[f"prior_large_delta_aligned_{w}_{st}"] = feat[f"large_delta_sum_{w}"].fillna(0.0).to_numpy(float) * side > 0
            if f"duration_ratio_{w}" in feat.columns:
                masks[f"current_fast_vs_{w}_{st}"] = feat[f"duration_ratio_{w}"].fillna(np.nan).to_numpy(float) <= 0.55
                masks[f"current_slow_vs_{w}_{st}"] = feat[f"duration_ratio_{w}"].fillna(np.nan).to_numpy(float) >= 1.80
            if f"notional_ratio_{w}" in feat.columns:
                masks[f"current_notional_spike_{w}_{st}"] = feat[f"notional_ratio_{w}"].fillna(np.nan).to_numpy(float) >= 1.75
                masks[f"current_notional_dry_{w}_{st}"] = feat[f"notional_ratio_{w}"].fillna(np.nan).to_numpy(float) <= 0.65

        for span in [21, 55, 144]:
            masks[f"ema_{span}_aligned_{st}"] = bool_series(feat[f"close_above_ema_{span}" if side == 1 else f"close_below_ema_{span}"]).to_numpy()
            masks[f"ema_{span}_opposed_{st}"] = bool_series(feat[f"close_below_ema_{span}" if side == 1 else f"close_above_ema_{span}"]).to_numpy()

        if "fp_delta_notional_sum_prev" in feat.columns:
            fp_delta = feat["fp_delta_notional_sum_prev"].fillna(0.0).to_numpy(float)
            fp_top = feat.get("fp_top_delta_notional_prev", pd.Series(0, index=feat.index)).fillna(0.0).to_numpy(float)
            fp_bottom = feat.get("fp_bottom_delta_notional_prev", pd.Series(0, index=feat.index)).fillna(0.0).to_numpy(float)
            fp_buy_stack = feat.get("fp_max_buy_stacked_levels_prev", pd.Series(0, index=feat.index)).fillna(0.0).to_numpy(float)
            fp_sell_stack = feat.get("fp_max_sell_stacked_levels_prev", pd.Series(0, index=feat.index)).fillna(0.0).to_numpy(float)
            masks[f"fp_delta_aligned_{st}"] = fp_delta * side > 0
            if side == 1:
                masks[f"fp_bottom_buy_absorb_{st}"] = fp_bottom > 0
                masks[f"fp_stacked_buy_{st}"] = fp_buy_stack >= 3
            else:
                masks[f"fp_top_sell_absorb_{st}"] = fp_top < 0
                masks[f"fp_stacked_sell_{st}"] = fp_sell_stack >= 3

    # Neutral context masks.
    for key in ["session_asia", "session_europe", "session_us_open", "session_us_late"]:
        masks[key] = bool_series(feat[key]).to_numpy()
    for w in [21, 55, 89]:
        if f"fast_bar_{w}" in feat.columns:
            masks[f"fast_bar_{w}"] = bool_series(feat[f"fast_bar_{w}"]).to_numpy()
            masks[f"slow_bar_{w}"] = bool_series(feat[f"slow_bar_{w}"]).to_numpy()
            masks[f"notional_spike_{w}"] = bool_series(feat[f"notional_spike_{w}"]).to_numpy()
            masks[f"notional_dry_{w}"] = bool_series(feat[f"notional_dry_{w}"]).to_numpy()
    return masks


def select_context_keys(mask_library: dict[str, np.ndarray], side: int, windows: list[int], max_contexts: int) -> list[str]:
    st = safe_side_token(side)
    candidates: list[str] = []
    # Broad, structural contexts rather than threshold grids.
    for key in ["session_us_open", "session_europe", "session_asia", "session_us_late"]:
        if key in mask_library:
            candidates.append(key)
    for w in [5, 13, 21, 34, 55, 89]:
        for name in [
            f"prior_trend_aligned_{w}_{st}", f"prior_trend_opposed_{w}_{st}", f"prior_chop_{w}_{st}",
            f"prior_price_momo_{w}_{st}", f"prior_price_fade_{w}_{st}",
            f"prior_delta_aligned_{w}_{st}", f"prior_large_delta_aligned_{w}_{st}",
        ]:
            if name in mask_library:
                candidates.append(name)
    for span in [21, 55, 144]:
        for name in [f"ema_{span}_aligned_{st}", f"ema_{span}_opposed_{st}"]:
            if name in mask_library:
                candidates.append(name)
    for key in ["fast_bar_21", "fast_bar_55", "slow_bar_55", "notional_spike_21", "notional_spike_55", "notional_dry_55"]:
        if key in mask_library:
            candidates.append(key)
    for key in [f"fp_delta_aligned_{st}", f"fp_bottom_buy_absorb_{st}", f"fp_top_sell_absorb_{st}", f"fp_stacked_buy_{st}", f"fp_stacked_sell_{st}"]:
        if key in mask_library:
            candidates.append(key)
    # De-duplicate preserving order and cap for speed/readability.
    seen = set()
    out = []
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
        if len(out) >= int(max_contexts):
            break
    return out


def build_signal_specs(mask_library: dict[str, np.ndarray], windows: list[int], args: argparse.Namespace) -> pd.DataFrame:
    specs: list[SignalSpec] = []
    for side in [1, -1]:
        st = safe_side_token(side)
        trigger_groups = {
            "range_streak_continuation": [f"cur_{st}", f"streak_ge2_{st}", f"streak_ge3_{st}"],
            "range_reversal_start": [f"reversal_to_{st}", f"fresh_after_opposite_{st}"],
            "orderflow_alignment": [f"cur_{st}", f"delta_aligned_{st}", f"large_delta_aligned_{st}"],
            "aggressive_taker_flow": [f"cur_{st}", f"taker_dominant_{st}"],
            "extreme_taker_flow": [f"cur_{st}", f"taker_extreme_{st}"],
            "strong_delta_bar": [f"strong_delta_ratio_{st}"],
            "strong_large_delta_bar": [f"strong_large_delta_ratio_{st}"],
        }
        for w in windows:
            trigger_groups[f"level_breakout_{w}"] = [f"break_{w}_{st}"]
            trigger_groups[f"sweep_reclaim_{w}"] = [f"sweep_reclaim_{w}_{st}"]
            trigger_groups[f"fast_directional_bar_{w}"] = [f"cur_{st}", f"current_fast_vs_{w}_{st}"]
            trigger_groups[f"volume_directional_bar_{w}"] = [f"cur_{st}", f"current_notional_spike_{w}_{st}"]
            trigger_groups[f"dry_liquidity_directional_bar_{w}"] = [f"cur_{st}", f"current_notional_dry_{w}_{st}"]
        if f"fp_delta_aligned_{st}" in mask_library:
            trigger_groups["footprint_delta_alignment"] = [f"cur_{st}", f"fp_delta_aligned_{st}"]
        if side == 1 and f"fp_bottom_buy_absorb_{st}" in mask_library:
            trigger_groups["footprint_bottom_absorption"] = [f"cur_{st}", f"fp_bottom_buy_absorb_{st}"]
            trigger_groups["footprint_stacked_buy"] = [f"cur_{st}", f"fp_stacked_buy_{st}"]
        if side == -1 and f"fp_top_sell_absorb_{st}" in mask_library:
            trigger_groups["footprint_top_absorption"] = [f"cur_{st}", f"fp_top_sell_absorb_{st}"]
            trigger_groups["footprint_stacked_sell"] = [f"cur_{st}", f"fp_stacked_sell_{st}"]

        contexts = select_context_keys(mask_library, side, windows, int(args.max_contexts_per_trigger))
        for family, triggers in trigger_groups.items():
            if not all(k in mask_library for k in triggers):
                continue
            base_id = f"{family}_{st}"
            specs.append(SignalSpec(base_id, family, side, tuple(triggers), (), f"{family} {side_name(side)}"))
            # Single-context variants.
            for ctx in contexts:
                if ctx in triggers or ctx not in mask_library:
                    continue
                sid = f"{base_id}__ctx_{ctx}"
                specs.append(SignalSpec(sid, f"{family}__context", side, tuple(triggers), (ctx,), f"{family} {side_name(side)} with {ctx}"))
            # Limited two-context variants: structural, not dense parameter sweep.
            if int(args.max_combo_contexts) >= 2:
                for i, c1 in enumerate(contexts[:12]):
                    for c2 in contexts[i + 1 : 12]:
                        if c1 in triggers or c2 in triggers or c1 not in mask_library or c2 not in mask_library:
                            continue
                        # Avoid combos that are mostly the same family direction/opposed variants.
                        if c1.rsplit("_", 2)[0] == c2.rsplit("_", 2)[0]:
                            continue
                        sid = f"{base_id}__ctx_{c1}__{c2}"
                        specs.append(SignalSpec(sid, f"{family}__context2", side, tuple(triggers), (c1, c2), f"{family} {side_name(side)} with {c1}+{c2}"))
    rows = [s.__dict__ for s in specs]
    cat = pd.DataFrame(rows)
    if cat.empty:
        return pd.DataFrame(columns=["signal_id", "family", "side", "trigger_keys", "context_keys", "description"])
    cat["side_name"] = cat["side"].map(side_name)
    cat = cat.drop_duplicates("signal_id").sort_values(["family", "side_name", "signal_id"]).reset_index(drop=True)
    return cat


def evaluate_specs(feat: pd.DataFrame, catalog: pd.DataFrame, masks: dict[str, np.ndarray], args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_start = pd.Timestamp(args.start_date)
    sample_end = pd.Timestamp(args.end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    ts = pd.to_datetime(feat["end_ts"])
    base_eligible = (ts >= sample_start) & (ts <= sample_end) & feat["entry_time"].notna() & feat["exit_time_h5rb"].notna()
    base_mask = base_eligible.to_numpy(bool)
    years = feat["year"].astype("Int64").astype(str).to_numpy()
    dates = feat["date"].astype(str).to_numpy()

    rows = []
    yearly_rows = []
    event_rows = []
    prog = ProgressReporter(label="[signals] evaluate", total=len(catalog), every=max(1, len(catalog) // 40), enabled=bool(args.progress))
    for idx, row in catalog.iterrows():
        side = int(row["side"])
        trigger_keys = tuple(row["trigger_keys"])
        context_keys = tuple(row["context_keys"])
        mask = base_mask.copy()
        missing = False
        for key in trigger_keys + context_keys:
            if key not in masks:
                missing = True
                break
            mask &= masks[key]
        if missing:
            prog.update(idx + 1)
            continue
        count = int(mask.sum())
        if count <= 0:
            prog.update(idx + 1)
            continue
        target_col = "target_next5_long" if side == 1 else "target_next5_short"
        ret_col = "ret_h5rb_long_net" if side == 1 else "ret_h5rb_short_net"
        gross_col = "ret_h5rb_long_gross" if side == 1 else "ret_h5rb_short_gross"
        mfe_col = "mfe_h5rb_long" if side == 1 else "mfe_h5rb_short"
        mae_col = "mae_h5rb_long" if side == 1 else "mae_h5rb_short"
        target = feat.loc[mask, target_col].astype(bool)
        ret = pd.to_numeric(feat.loc[mask, ret_col], errors="coerce")
        gross = pd.to_numeric(feat.loc[mask, gross_col], errors="coerce")
        mfe = pd.to_numeric(feat.loc[mask, mfe_col], errors="coerce")
        mae = pd.to_numeric(feat.loc[mask, mae_col], errors="coerce")
        summ = summarize_values(ret, min_count=int(args.min_count))
        baseline = float(feat.loc[base_mask, target_col].mean()) if base_mask.any() else np.nan
        hit_rate = float(target.mean()) if len(target) else np.nan
        r = {
            "signal_id": row["signal_id"],
            "family": row["family"],
            "side": side_name(side),
            "trigger_keys": "|".join(trigger_keys),
            "context_keys": "|".join(context_keys),
            "count": count,
            "hit_rate_next5": hit_rate,
            "baseline_hit_rate_next5": baseline,
            "hit_rate_lift": hit_rate / baseline if baseline and baseline > 0 else np.nan,
            "gross_mean": float(gross.mean()) if len(gross) else np.nan,
            "net_mean": summ["mean"],
            "net_median": summ["median"],
            "net_win_rate": summ["win_rate"],
            "profit_factor": summ["profit_factor"],
            "top5_winner_share": summ["top5_winner_share"],
            "mfe_mean": float(mfe.mean()) if len(mfe) else np.nan,
            "mae_mean": float(mae.mean()) if len(mae) else np.nan,
            "p25": summ["p25"],
            "p75": summ["p75"],
            "p95": summ["p95"],
            "eligible": bool(count >= int(args.min_count)),
        }
        rows.append(r)
        # Yearly rows.
        sub_years = years[mask]
        for y in sorted(set(sub_years)):
            ym = mask & (years == y)
            if int(ym.sum()) <= 0:
                continue
            yret = pd.to_numeric(feat.loc[ym, ret_col], errors="coerce")
            ytarget = feat.loc[ym, target_col].astype(bool)
            ys = summarize_values(yret, min_count=0)
            yearly_rows.append(
                {
                    "signal_id": row["signal_id"],
                    "family": row["family"],
                    "side": side_name(side),
                    "year": y,
                    "count": int(ym.sum()),
                    "hit_rate_next5": float(ytarget.mean()) if int(ym.sum()) else np.nan,
                    "net_mean": ys["mean"],
                    "net_median": ys["median"],
                    "profit_factor": ys["profit_factor"],
                    "net_win_rate": ys["win_rate"],
                }
            )
        # Keep event sample only for good-ish or if full requested.
        keep_events = bool(args.write_full_signal_events) or (count >= int(args.min_count) and hit_rate > baseline and float(summ["mean"]) > 0)
        if keep_events:
            cols = [
                "bar_id", "end_ts", "entry_time", "exit_time_h5rb", "entry_price", "exit_price_h5rb",
                target_col, gross_col, ret_col, mfe_col, mae_col, "date", "year", "direction", "signed_run_len",
                "duration_seconds", "notional", "delta_ratio", "large_delta_ratio", "taker_buy_ratio",
            ]
            part = feat.loc[mask, [c for c in cols if c in feat.columns]].copy()
            part.insert(0, "signal_id", row["signal_id"])
            part.insert(1, "family", row["family"])
            part.insert(2, "side", side_name(side))
            part = part.rename(columns={target_col: "target_next5_hit", gross_col: "gross_return", ret_col: "net_return", mfe_col: "mfe", mae_col: "mae"})
            if not bool(args.write_full_signal_events):
                per_signal = max(20, int(args.event_sample_size) // max(1, min(200, len(catalog))))
                part = part.head(per_signal)
            event_rows.append(part)
        prog.update(idx + 1)
    prog.close()

    stats = pd.DataFrame(rows)
    yearly = pd.DataFrame(yearly_rows)
    events = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()
    if not stats.empty:
        stats["score"] = (
            stats["hit_rate_lift"].fillna(0.0).clip(0, 5) * 0.25
            + stats["net_mean"].fillna(-9.0).clip(-0.02, 0.03) * 50
            + stats["profit_factor"].replace("inf", 99).astype(float).fillna(0.0).clip(0, 5) * 0.15
            + stats["net_median"].fillna(-9.0).clip(-0.02, 0.03) * 30
        )
        stats = stats.sort_values(["eligible", "score", "hit_rate_lift", "net_mean"], ascending=[False, False, False, False])
    return stats, yearly, events


def summarize_family(stats: pd.DataFrame) -> pd.DataFrame:
    if stats.empty:
        return pd.DataFrame()
    rows = []
    for (fam, side), part in stats.groupby(["family", "side"], dropna=False):
        eligible = part[part["eligible"].astype(bool)]
        use = eligible if not eligible.empty else part
        rows.append(
            {
                "family": fam,
                "side": side,
                "specs": int(len(part)),
                "eligible_specs": int(len(eligible)),
                "best_signal_id": use.iloc[0]["signal_id"],
                "best_score": float(use.iloc[0]["score"]),
                "best_count": int(use.iloc[0]["count"]),
                "best_hit_rate_next5": float(use.iloc[0]["hit_rate_next5"]),
                "best_hit_rate_lift": float(use.iloc[0]["hit_rate_lift"]),
                "best_net_mean": float(use.iloc[0]["net_mean"]),
                "best_net_median": float(use.iloc[0]["net_median"]),
                "best_profit_factor": use.iloc[0]["profit_factor"],
            }
        )
    return pd.DataFrame(rows).sort_values(["best_score", "best_hit_rate_lift"], ascending=False)


def build_feature_contrast(feat: pd.DataFrame, run_starts: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if run_starts.empty:
        return pd.DataFrame()
    sample_start = pd.Timestamp(args.start_date)
    sample_end = pd.Timestamp(args.end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    eligible = (pd.to_datetime(feat["end_ts"]) >= sample_start) & (pd.to_datetime(feat["end_ts"]) <= sample_end)
    run_signal_pos_by_side = {
        1: set(run_starts.loc[run_starts["side_int"] == 1, "signal_bar_pos"].astype(int).tolist()),
        -1: set(run_starts.loc[run_starts["side_int"] == -1, "signal_bar_pos"].astype(int).tolist()),
    }
    features = [
        "abs_run_len", "signed_run_len", "duration_seconds", "notional", "trades_count", "delta_ratio", "large_delta_ratio",
        "large_trade_ratio", "taker_buy_ratio", "bar_ret", "session_asia", "session_europe", "session_us_open", "session_us_late",
    ]
    for extra in ["dir_consistency_13", "dir_consistency_21", "ret_13", "ret_21", "notional_ratio_21", "duration_ratio_21"]:
        if extra in feat.columns:
            features.append(extra)
    rows = []
    base = feat.loc[eligible].copy()
    for side in [1, -1]:
        pos_set = run_signal_pos_by_side[side]
        run_mask = feat.index.to_series().isin(pos_set) & eligible
        side_label = side_name(side)
        for col in features:
            if col not in feat.columns:
                continue
            a = pd.to_numeric(feat.loc[run_mask, col], errors="coerce") if feat[col].dtype != bool else feat.loc[run_mask, col].astype(float)
            b = pd.to_numeric(base[col], errors="coerce") if base[col].dtype != bool else base[col].astype(float)
            rows.append(
                {
                    "side": side_label,
                    "feature": col,
                    "run_start_count": int(a.notna().sum()),
                    "all_count": int(b.notna().sum()),
                    "run_start_mean": float(a.mean()) if not a.empty else np.nan,
                    "all_mean": float(b.mean()) if not b.empty else np.nan,
                    "diff": float(a.mean() - b.mean()) if not a.empty and not b.empty else np.nan,
                    "run_start_median": float(a.median()) if not a.empty else np.nan,
                    "all_median": float(b.median()) if not b.empty else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["side", "diff"], ascending=[True, False])


def audit(feat: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    sample_start = pd.Timestamp(args.start_date)
    sample_end = pd.Timestamp(args.end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    ts = pd.to_datetime(feat["end_ts"])
    sample = (ts >= sample_start) & (ts <= sample_end)
    entry_time = pd.to_datetime(feat["entry_time"], errors="coerce")
    signal_time = pd.to_datetime(feat["end_ts"], errors="coerce")
    exit_time = pd.to_datetime(feat["exit_time_h5rb"], errors="coerce")
    auditable = sample & entry_time.notna() & exit_time.notna() & signal_time.notna()
    entry_after_signal = entry_time >= signal_time
    horizon_ok = exit_time > entry_time
    rows = [
        {"check": "rows", "value": int(len(feat))},
        {"check": "sample_rows", "value": int(sample.sum())},
        {"check": "auditable_rows", "value": int(auditable.sum())},
        {"check": "entry_before_signal_flag", "value": int((auditable & ~entry_after_signal).sum())},
        {"check": "exit_not_after_entry_flag", "value": int((auditable & ~horizon_ok).sum())},
        {"check": "ordinary_kline_used", "value": 0},
        {"check": "target_run_bars", "value": int(args.target_run_bars)},
        {"check": "range_pct", "value": float(args.range_pct)},
    ]
    if "fp_delta_notional_sum_prev" in feat.columns:
        rows.append({"check": "footprint_context_shifted_by_one_bar", "value": 1})
    return pd.DataFrame(rows)


def write_brief(out_dir: Path, args: argparse.Namespace, run_windows: pd.DataFrame, run_starts: pd.DataFrame, stats: pd.DataFrame, family: pd.DataFrame, audit_df: pd.DataFrame) -> None:
    top = stats.head(20) if not stats.empty else pd.DataFrame()
    lines = []
    lines.append("# 0.2% Range-Bar 5-Bar Directional Run Research")
    lines.append("")
    lines.append("## Scope")
    lines.append(f"- symbol: `{args.symbol}`")
    lines.append(f"- range_pct: `{args.range_pct}` ({float(args.range_pct)*100:.2f}%)")
    lines.append(f"- target: next `{args.target_run_bars}` range bars all same direction")
    lines.append(f"- date: `{args.start_date}` -> `{args.end_date}`, warmup `{args.warmup_start_date}`")
    lines.append(f"- fee_rate: `{args.fee_rate}`")
    lines.append(f"- footprint: `{bool(args.include_footprint)}`")
    lines.append("")
    lines.append("## Raw 5-bar run labels")
    lines.append(f"- all overlapping five-bar windows: `{len(run_windows):,}`")
    lines.append(f"- streak-start five-bar runs: `{len(run_starts):,}`")
    if not run_starts.empty:
        by_side = run_starts.groupby("side").size().to_dict()
        lines.append(f"- streak starts by side: `{by_side}`")
        lines.append(f"- median run duration seconds: `{run_starts['run_duration_seconds'].median():.1f}`")
    lines.append("")
    lines.append("## Causal audit")
    for _, r in audit_df.iterrows():
        lines.append(f"- {r['check']}: `{r['value']}`")
    lines.append("")
    lines.append("## Top candidate signals")
    if top.empty:
        lines.append("No candidate signals produced.")
    else:
        show_cols = ["signal_id", "side", "count", "hit_rate_next5", "baseline_hit_rate_next5", "hit_rate_lift", "net_mean", "net_median", "profit_factor"]
        lines.append(top[show_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Family best signals")
    if family.empty:
        lines.append("No family summary.")
    else:
        show_cols = ["family", "side", "eligible_specs", "best_signal_id", "best_count", "best_hit_rate_next5", "best_hit_rate_lift", "best_net_mean", "best_profit_factor"]
        lines.append(family.head(30)[show_cols].to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation note")
    lines.append("This is an event-study / hypothesis research report, not a final strategy backtest. Any strong candidate must still be replayed with one-position state, overlap control, stop/TP/time-exit, slippage/delay stress, yearly stability, and parameter-neighbourhood checks.")
    (out_dir / "20_research_brief.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = PROJECT_ROOT / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    windows = parse_int_list(args.feature_windows)

    bars = load_range_bars(args)
    feat = build_features(bars, args, windows)
    if args.include_footprint:
        fp = load_footprint_features(args, feat["bar_id"].astype(int).tolist())
        feat = merge_footprint(feat, fp)
    feat = add_targets(feat, args)
    run_windows, run_starts = build_actual_run_tables(feat, args)
    print(f"[runs] five-bar windows={len(run_windows):,} streak_starts={len(run_starts):,}")

    masks = build_mask_library(feat, windows)
    catalog = build_signal_specs(masks, windows, args)
    print(f"[signals] mask_library={len(masks):,} catalog={len(catalog):,}")
    stats, yearly, signal_events = evaluate_specs(feat, catalog, masks, args)
    family = summarize_family(stats)
    contrast = build_feature_contrast(feat, run_starts, args)
    audit_df = audit(feat, args)

    # Persist outputs.
    catalog_out = catalog.copy()
    for col in ["trigger_keys", "context_keys"]:
        if col in catalog_out.columns:
            catalog_out[col] = catalog_out[col].apply(lambda x: "|".join(x) if isinstance(x, tuple) else str(x))
    catalog_out.to_csv(out_dir / "00_signal_catalog.csv", index=False)
    run_windows.to_csv(out_dir / "01_five_run_windows_all.csv", index=False)
    run_starts.to_csv(out_dir / "02_five_run_streak_starts.csv", index=False)
    stats.to_csv(out_dir / "03_candidate_signal_stats.csv", index=False)
    yearly.to_csv(out_dir / "04_candidate_yearly_stats.csv", index=False)
    family.to_csv(out_dir / "05_candidate_family_best.csv", index=False)
    contrast.to_csv(out_dir / "06_pre_run_feature_contrast.csv", index=False)
    audit_df.to_csv(out_dir / "07_causal_audit.csv", index=False)
    if not signal_events.empty:
        if bool(args.write_full_signal_events):
            signal_events.to_csv(out_dir / "08_signal_events_full.csv", index=False)
        else:
            signal_events.head(int(args.event_sample_size)).to_csv(out_dir / "08_signal_events_sample.csv", index=False)
    tail_cols = [
        "bar_id", "start_ts", "end_ts", "open", "high", "low", "close", "direction", "signed_run_len",
        "duration_seconds", "notional", "delta_ratio", "large_delta_ratio", "taker_buy_ratio",
        "target_next5_long", "target_next5_short", "ret_h5rb_long_net", "ret_h5rb_short_net",
    ]
    feat[[c for c in tail_cols if c in feat.columns]].tail(5000).to_csv(out_dir / "09_feature_tail_sample.csv", index=False)

    meta = {
        "symbol": args.symbol,
        "range_pct": float(args.range_pct),
        "range_code": range_code(float(args.range_pct)),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "target_run_bars": int(args.target_run_bars),
        "fee_rate": float(args.fee_rate),
        "include_footprint": bool(args.include_footprint),
        "rows": int(len(feat)),
        "five_run_windows": int(len(run_windows)),
        "five_run_streak_starts": int(len(run_starts)),
        "signal_catalog": int(len(catalog)),
        "ordinary_kline_used": False,
    }
    (out_dir / "10_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    write_brief(out_dir, args, run_windows, run_starts, stats, family, audit_df)
    print(f"[done] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
