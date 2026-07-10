#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH 5s CVD impulse-stall reversal V1 research.

Research-only event study for the idea:

    On 5s trade bars, accumulated CVD suddenly changes with an almost vertical
    slope. If the next 5s bar's accumulated CVD slope becomes relatively flat,
    fade the CVD impulse and enter in the opposite direction at the next 5s open.
    Take profit is fixed at 0.3%; if TP is not hit, exit by max-hold time.

Important: CVD here is accumulated CVD (`cvd_notional`), not a single-bar CVD
feature. The event uses slopes/changes of the accumulated CVD series. Since a
constant reset of accumulated CVD does not change slopes, monthly chunking is
safe for event detection as long as enough lookback is loaded.

Timing policy:
    impulse bar: t-1 closed 5s bar
    stall bar:   t closed 5s bar
    signal_time = t closed 5s bar timestamp
    entry_time  = next 5s bar open plus optional delay bars
    TP/timeout replay uses bars after entry; TP is filled at the fixed TP price.

Performance policy:
    - monthly chunk processing for 5s data to avoid loading multi-year 5s bars
      into memory;
    - vectorized masks and replay loops by offset, not per-event full scans;
    - matched baseline only runs for preliminary survivors and uses grouped
      numpy pools, not per-event DataFrame filtering.

This script does not register a tradable edge, modify portfolio code, or import
business logic from other research scripts.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
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

SCRIPT_NAME = "eth_hf_cvd_impulse_stall_reversal_5s_v1.py"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_HF_CVD_IMPULSE_STALL_REVERSAL_5S_V1"
EDGE_ID = "ETH_EDGE_HF_CVD_IMPULSE_STALL_REVERSAL_5S_RESEARCH_V1"
DEFAULT_OUT_DIR = "data/reports/research/hf_cvd_impulse_stall_reversal_5s_v1"
CAUSAL_POLICY = "closed 5s trade-bar signal; next 5s open entry; fixed TP and max-hold timeout"
MATCHED_BASELINE_COLUMNS = ("year", "month", "session", "regime", "volatility_bucket", "direction")


@dataclass(frozen=True)
class EventSpec:
    impulse_bars: int
    cvd_slope_mult: float
    flat_ratio: float
    flat_baseline_mult: float
    min_notional_z: float
    require_price_confirm: bool
    min_abs_price_move_pct: float

    @property
    def variant(self) -> str:
        sm = str(self.cvd_slope_mult).replace(".", "p")
        fr = str(self.flat_ratio).replace(".", "p")
        fb = str(self.flat_baseline_mult).replace(".", "p")
        nz = str(self.min_notional_z).replace(".", "p")
        pm = int(round(self.min_abs_price_move_pct * 10000))
        pc = "pc1" if self.require_price_confirm else "pc0"
        return f"ib{self.impulse_bars}_sm{sm}_fr{fr}_fb{fb}_nz{nz}_{pc}_ret{pm}bp"

    def event_name(self, direction: str) -> str:
        return f"cvd_impulse_stall_reversal_5s__{self.variant}__{direction}"


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Research-only ETH 5s accumulated-CVD impulse-stall reversal with fixed 0.3% TP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--primary-timeframe", default="5s")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--impulse-bars-list", default="1,2", help="Accumulated-CVD slope window ending at the impulse bar t-1")
    p.add_argument("--cvd-slope-multipliers", default="8.0,12.0", help="Impulse abs CVD slope must exceed rolling median abs slope times this")
    p.add_argument("--flat-ratios", default="0.15,0.25", help="Stall bar abs CVD delta must be <= abs impulse CVD change times this")
    p.add_argument("--flat-baseline-mult", type=float, default=1.0, help="Stall bar abs CVD delta must be <= rolling median abs 1-bar CVD delta times this")
    p.add_argument("--cvd-norm-window", type=int, default=180, help="Rolling 5s bars used to normalize CVD slopes; 180 bars = 15 minutes")
    p.add_argument("--notional-z-window", type=int, default=180)
    p.add_argument("--min-notional-z-list", default="0.0,1.0")
    p.add_argument("--price-confirm-modes", default="0,1", help="0=no price confirmation; 1=impulse price move must align with CVD impulse")
    p.add_argument("--min-abs-price-move-pct", type=float, default=0.0003)
    p.add_argument("--tp-pct", type=float, default=0.0030)
    p.add_argument("--max-hold-bars-list", default="12,36,60,120,180,360", help="5s bars; 12=1m, 60=5m, 360=30m")
    p.add_argument("--mfe-mae-horizon", type=int, default=360)
    p.add_argument("--round-trip-cost-pct", type=float, default=0.0011)
    p.add_argument("--cost-multipliers", default="1.0,1.5,2.0,3.0")
    p.add_argument("--delay-bars-list", default="0,1,2,3")
    p.add_argument("--min-count", type=int, default=500)
    p.add_argument("--min-events-per-year", type=float, default=120.0)
    p.add_argument("--cooldown-bars", type=int, default=0)
    p.add_argument("--baseline-samples", type=int, default=50)
    p.add_argument("--baseline-max-events-per-group", type=int, default=1000)
    p.add_argument("--baseline-prefilter-mean-net", type=float, default=-0.0002)
    p.add_argument("--baseline-prefilter-pf", type=float, default=0.95)
    p.add_argument("--baseline-seed", type=int, default=42)
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--no-build-missing", action="store_true")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--event-sample-size", type=int, default=5000)
    p.add_argument("--trade-sample-size", type=int, default=20000)
    p.add_argument("--write-full-trades", action="store_true")
    p.add_argument("--progress-every", type=int, default=1)
    return p.parse_args(argv)


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    out: list[int] = []
    for part in str(raw).split(","):
        text = part.strip()
        if text:
            out.append(int(text))
    return tuple(dict.fromkeys(out))


def _parse_csv_floats(raw: str) -> tuple[float, ...]:
    out: list[float] = []
    for part in str(raw).split(","):
        text = part.strip()
        if text:
            out.append(float(text))
    return tuple(dict.fromkeys(out))


def _parse_csv_bools(raw: str) -> tuple[bool, ...]:
    vals: list[bool] = []
    for part in str(raw).split(","):
        text = part.strip().lower()
        if not text:
            continue
        vals.append(text in {"1", "true", "yes", "y"})
    return tuple(dict.fromkeys(vals))


def _annualized_years(start_date: str, end_date: str) -> float:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    return max((end - start).total_seconds() / (365.25 * 86400.0), 1e-9)


def _profit_factor(values: np.ndarray | pd.Series) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    gains = v[v > 0].sum()
    losses = -v[v < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def _top_winner_share(values: np.ndarray | pd.Series, top_n: int = 5) -> float:
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


def _assign_session(idx: pd.DatetimeIndex) -> np.ndarray:
    h = idx.hour.astype(int)
    return np.select([h < 8, h < 16], ["asia_00_08", "asia_europe_08_16"], default="us_16_24")


def _month_chunks(start: str, end: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    start_ts = pd.Timestamp(start).normalize()
    end_ts = pd.Timestamp(end).normalize() + pd.Timedelta(hours=23, minutes=59, seconds=59)
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start_ts
    while cur <= end_ts:
        nxt = (cur + pd.offsets.MonthBegin(1)).normalize()
        chunk_end = min(nxt - pd.Timedelta(seconds=1), end_ts)
        chunks.append((cur, chunk_end))
        cur = nxt
    return chunks


# ---------------------------------------------------------------------------
# Data / features
# ---------------------------------------------------------------------------


def _make_loader(args: argparse.Namespace) -> OKXTradeBarLoader:
    kwargs: dict[str, object] = {
        "symbol": args.symbol,
        "timeframe": args.primary_timeframe,
        "db_name": args.db_name,
    }
    if args.data_dir:
        kwargs["data_dir"] = Path(args.data_dir)
    return OKXTradeBarLoader(**kwargs)


def load_trade_bars_range(args: argparse.Namespace, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    loader = _make_loader(args)
    df = loader.fetch_data_by_date_range(
        start,
        end,
        chunksize=args.chunksize,
        force_rebuild=bool(args.force_rebuild),
        cvd_mode="range",
        build_missing=not bool(args.no_build_missing),
    )
    if df.empty:
        return df
    df = df.sort_index()
    df.index = pd.to_datetime(df.index)
    keep = [
        "open",
        "high",
        "low",
        "close",
        "notional",
        "delta_notional",
        "cvd_notional",
        "buy_notional",
        "sell_notional",
        "trades_count",
    ]
    cols = [c for c in keep if c in df.columns]
    out = df[cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)
    return out


def build_features(df: pd.DataFrame, *, cvd_norm_window: int, notional_z_window: int) -> tuple[pd.DataFrame, list[str]]:
    required = ["open", "high", "low", "close", "cvd_notional"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"missing required 5s trade-bar columns: {missing}")

    base = df.copy()
    idx = base.index
    close = base["close"].astype(float)
    high = base["high"].astype(float)
    low = base["low"].astype(float)
    cvd = base["cvd_notional"].astype(float)
    cvd_delta_1 = cvd.diff()
    rng_pct = (high - low) / close.replace(0, np.nan)
    ret_1 = close.pct_change()

    notional = base.get("notional", pd.Series(0.0, index=idx)).astype(float)
    notional_mean = notional.shift(1).rolling(notional_z_window, min_periods=max(20, notional_z_window // 5)).mean()
    notional_std = notional.shift(1).rolling(notional_z_window, min_periods=max(20, notional_z_window // 5)).std(ddof=0)
    notional_z = (notional - notional_mean) / notional_std.replace(0, np.nan)

    abs_cvd_delta = cvd_delta_1.abs()
    cvd_delta_base = abs_cvd_delta.shift(1).rolling(cvd_norm_window, min_periods=max(20, cvd_norm_window // 5)).median()
    rng_base = rng_pct.shift(1).rolling(720, min_periods=120).median()
    vol_ratio = rng_pct / rng_base.replace(0, np.nan)

    features = pd.DataFrame(
        {
            "cvd": cvd,
            "cvd_delta_1": cvd_delta_1,
            "abs_cvd_delta_1": abs_cvd_delta,
            "cvd_delta_base": cvd_delta_base,
            "ret_1": ret_1,
            "range_pct": rng_pct,
            "vol_ratio": vol_ratio,
            "notional_z": notional_z,
            "year": idx.year.astype(int),
            "month": idx.month.astype(int),
            "session": _assign_session(idx),
            "weekday": idx.dayofweek.astype(int),
        },
        index=idx,
    )
    features["regime"] = np.select(
        [close.pct_change(720).values > 0.01, close.pct_change(720).values < -0.01],
        ["trend_up", "trend_down"],
        default="normal",
    )
    features["volatility_bucket"] = pd.qcut(features["vol_ratio"].rank(method="first"), q=5, labels=False, duplicates="drop").astype("float")
    features["volatility_bucket"] = features["volatility_bucket"].fillna(-1).astype(int)

    out = pd.concat([base, features], axis=1)
    feature_cols = [c for c in features.columns]
    return out, feature_cols


# ---------------------------------------------------------------------------
# Events and replay
# ---------------------------------------------------------------------------


def build_specs(args: argparse.Namespace) -> list[EventSpec]:
    specs: list[EventSpec] = []
    for impulse_bars in _parse_csv_ints(args.impulse_bars_list):
        for slope_mult in _parse_csv_floats(args.cvd_slope_multipliers):
            for flat_ratio in _parse_csv_floats(args.flat_ratios):
                for min_nz in _parse_csv_floats(args.min_notional_z_list):
                    for pc in _parse_csv_bools(args.price_confirm_modes):
                        specs.append(
                            EventSpec(
                                impulse_bars=impulse_bars,
                                cvd_slope_mult=slope_mult,
                                flat_ratio=flat_ratio,
                                flat_baseline_mult=float(args.flat_baseline_mult),
                                min_notional_z=float(min_nz),
                                require_price_confirm=bool(pc),
                                min_abs_price_move_pct=float(args.min_abs_price_move_pct),
                            )
                        )
    return specs


def _apply_cooldown(mask: np.ndarray, cooldown_bars: int) -> np.ndarray:
    if cooldown_bars <= 0 or mask.sum() <= 1:
        return mask
    idxs = np.flatnonzero(mask)
    keep: list[int] = []
    last = -10**18
    for i in idxs:
        if i - last > cooldown_bars:
            keep.append(int(i))
            last = int(i)
    out = np.zeros_like(mask, dtype=bool)
    out[np.asarray(keep, dtype=int)] = True
    return out


def build_events_for_chunk(
    feat: pd.DataFrame,
    specs: Sequence[EventSpec],
    *,
    chunk_start: pd.Timestamp,
    chunk_end: pd.Timestamp,
    cooldown_bars: int,
) -> pd.DataFrame:
    if feat.empty:
        return pd.DataFrame()
    idx = feat.index
    in_chunk = (idx >= chunk_start) & (idx <= chunk_end)
    rows: list[pd.DataFrame] = []

    cvd = feat["cvd"].astype(float)
    cvd_delta_now = feat["cvd_delta_1"].astype(float)
    cvd_delta_base = feat["cvd_delta_base"].astype(float)
    close = feat["close"].astype(float)

    for spec in specs:
        impulse_raw = cvd.diff(spec.impulse_bars)
        impulse_prev = impulse_raw.shift(1)
        # Use the existing 1-bar median baseline scaled by sqrt(window) as a stable slope baseline.
        # This avoids a slow rolling median for every impulse_bars variant.
        slope_base = feat["cvd_delta_base"].astype(float) * math.sqrt(max(spec.impulse_bars, 1))
        price_impulse_prev = close.pct_change(spec.impulse_bars).shift(1)

        extreme = impulse_prev.abs() >= (slope_base * spec.cvd_slope_mult)
        flat_vs_impulse = cvd_delta_now.abs() <= (impulse_prev.abs() * spec.flat_ratio)
        flat_vs_base = cvd_delta_now.abs() <= (feat["cvd_delta_base"].astype(float) * spec.flat_baseline_mult)
        liquid = feat["notional_z"].fillna(-999.0) >= spec.min_notional_z
        base_mask = extreme & flat_vs_impulse & flat_vs_base & liquid & pd.Series(in_chunk, index=idx)
        if spec.require_price_confirm:
            price_ok = (np.sign(impulse_prev) * price_impulse_prev) > 0
            price_ok &= price_impulse_prev.abs() >= spec.min_abs_price_move_pct
            base_mask &= price_ok

        long_mask = (base_mask & (impulse_prev < 0)).fillna(False).to_numpy(dtype=bool)
        short_mask = (base_mask & (impulse_prev > 0)).fillna(False).to_numpy(dtype=bool)
        long_mask = _apply_cooldown(long_mask, cooldown_bars)
        short_mask = _apply_cooldown(short_mask, cooldown_bars)

        for direction, mask in (("long", long_mask), ("short", short_mask)):
            pos = np.flatnonzero(mask)
            if pos.size == 0:
                continue
            ev = pd.DataFrame(
                {
                    "event_name": spec.event_name(direction),
                    "family": "cvd_impulse_stall_reversal_5s",
                    "variant": spec.variant,
                    "direction": direction,
                    "signal_time": idx[pos],
                    "signal_pos": pos.astype(np.int64),
                    "impulse_bars": spec.impulse_bars,
                    "cvd_slope_mult": spec.cvd_slope_mult,
                    "flat_ratio": spec.flat_ratio,
                    "flat_baseline_mult": spec.flat_baseline_mult,
                    "min_notional_z": spec.min_notional_z,
                    "require_price_confirm": spec.require_price_confirm,
                    "impulse_cvd_change": impulse_prev.iloc[pos].to_numpy(dtype=float),
                    "stall_cvd_delta": cvd_delta_now.iloc[pos].to_numpy(dtype=float),
                    "price_impulse_ret": price_impulse_prev.iloc[pos].to_numpy(dtype=float),
                    "year": feat["year"].iloc[pos].to_numpy(dtype=int),
                    "month": feat["month"].iloc[pos].to_numpy(dtype=int),
                    "session": feat["session"].iloc[pos].astype(str).to_numpy(),
                    "regime": feat["regime"].iloc[pos].astype(str).to_numpy(),
                    "volatility_bucket": feat["volatility_bucket"].iloc[pos].to_numpy(dtype=int),
                }
            )
            rows.append(ev)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def replay_positions(
    feat: pd.DataFrame,
    events: pd.DataFrame,
    *,
    max_hold_bars: int,
    delay_bars: int,
    tp_pct: float,
    round_trip_cost_pct: float,
    cost_multiplier: float = 1.0,
    include_detail: bool = True,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    n = len(feat)
    times = feat.index.to_numpy()
    open_px = feat["open"].to_numpy(dtype=float)
    high_px = feat["high"].to_numpy(dtype=float)
    low_px = feat["low"].to_numpy(dtype=float)

    signal_pos = events["signal_pos"].to_numpy(dtype=np.int64)
    entry_pos = signal_pos + 1 + int(delay_bars)
    valid = (entry_pos >= 0) & ((entry_pos + int(max_hold_bars)) < n)
    if not np.any(valid):
        return pd.DataFrame()
    ev = events.loc[valid].reset_index(drop=True)
    entry_pos = entry_pos[valid]
    entry_price = open_px[entry_pos]
    dirs = ev["direction"].astype(str).to_numpy()
    is_long = dirs == "long"
    tp_price = np.where(is_long, entry_price * (1.0 + tp_pct), entry_price * (1.0 - tp_pct))

    m = len(ev)
    hit_offset = np.full(m, -1, dtype=np.int32)
    mfe = np.full(m, np.nan, dtype=float)
    mae = np.full(m, np.nan, dtype=float)
    hi_max = np.full(m, -np.inf, dtype=float)
    lo_min = np.full(m, np.inf, dtype=float)
    active = np.ones(m, dtype=bool)
    for off in range(int(max_hold_bars)):
        p = entry_pos + off
        hi = high_px[p]
        lo = low_px[p]
        hi_max = np.maximum(hi_max, hi)
        lo_min = np.minimum(lo_min, lo)
        hit = np.where(is_long, hi >= tp_price, lo <= tp_price) & active
        if np.any(hit):
            hit_offset[hit] = off
            active[hit] = False
        if not np.any(active):
            # Still have hi/lo for MFE/MAE until hit only; for event study this is enough.
            break

    timeout_pos = entry_pos + int(max_hold_bars)
    exit_pos = np.where(hit_offset >= 0, entry_pos + hit_offset, timeout_pos)
    exit_price = np.where(hit_offset >= 0, tp_price, open_px[timeout_pos])
    gross = np.where(is_long, exit_price / entry_price - 1.0, entry_price / exit_price - 1.0)
    net = gross - float(round_trip_cost_pct) * float(cost_multiplier)
    mfe = np.where(is_long, hi_max / entry_price - 1.0, entry_price / lo_min - 1.0)
    mae = np.where(is_long, lo_min / entry_price - 1.0, entry_price / hi_max - 1.0)

    data: dict[str, object] = {
        "event_name": ev["event_name"].to_numpy(),
        "family": ev["family"].to_numpy(),
        "variant": ev["variant"].to_numpy(),
        "direction": dirs,
        "max_hold_bars": int(max_hold_bars),
        "delay_bars": int(delay_bars),
        "signal_time": pd.to_datetime(ev["signal_time"]).to_numpy(),
        "entry_time": pd.to_datetime(times[entry_pos]).to_numpy(),
        "exit_time": pd.to_datetime(times[exit_pos]).to_numpy(),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "exit_reason": np.where(hit_offset >= 0, "tp", "time"),
        "tp_hit": hit_offset >= 0,
        "tp_pct": float(tp_pct),
        "gross_return": gross,
        "net_return": net,
        "mfe": mfe,
        "mae": mae,
        "year": ev["year"].to_numpy(dtype=int),
        "month": ev["month"].to_numpy(dtype=int),
        "session": ev["session"].astype(str).to_numpy(),
        "regime": ev["regime"].astype(str).to_numpy(),
        "volatility_bucket": ev["volatility_bucket"].to_numpy(dtype=int),
        "expected_entry_time": pd.to_datetime(times[signal_pos[valid] + 1 + int(delay_bars)]).to_numpy(),
        "entry_not_next_open_flag": 0,
        "forward_window_valid_flag": True,
        "lookahead_flag": 0,
    }
    if include_detail:
        data.update(
            {
                "impulse_bars": ev["impulse_bars"].to_numpy(dtype=int),
                "cvd_slope_mult": ev["cvd_slope_mult"].to_numpy(dtype=float),
                "flat_ratio": ev["flat_ratio"].to_numpy(dtype=float),
                "impulse_cvd_change": ev["impulse_cvd_change"].to_numpy(dtype=float),
                "stall_cvd_delta": ev["stall_cvd_delta"].to_numpy(dtype=float),
                "price_impulse_ret": ev["price_impulse_ret"].to_numpy(dtype=float),
            }
        )
    return pd.DataFrame(data)


def summarize_replay(trades: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    years = _annualized_years(start_date, end_date)
    rows: list[dict[str, object]] = []
    keys = ["event_name", "family", "variant", "direction", "max_hold_bars"]
    for key, g in trades.groupby(keys, sort=False):
        net = g["net_return"].to_numpy(dtype=float)
        gross = g["gross_return"].to_numpy(dtype=float)
        count = len(g)
        yearly = g.groupby("year")["net_return"].mean()
        rows.append(
            {
                "event_name": key[0],
                "family": key[1],
                "variant": key[2],
                "direction": key[3],
                "max_hold_bars": int(key[4]),
                "count": int(count),
                "events_per_year": float(count / years),
                "events_per_month": float(count / max(years * 12.0, 1e-9)),
                "mean_gross": float(np.nanmean(gross)),
                "mean_net": float(np.nanmean(net)),
                "median_net": float(np.nanmedian(net)),
                "win_rate": float(np.nanmean(net > 0)),
                "profit_factor": _profit_factor(net),
                "tp_hit_rate": float(g["tp_hit"].mean()),
                "mfe_mean": float(g["mfe"].mean()),
                "mae_mean": float(g["mae"].mean()),
                "positive_years": int((yearly > 0).sum()),
                "year_count": int(yearly.shape[0]),
                "max_days_without_event": _max_days_without_event(g["signal_time"]),
                "top5_winner_share": _top_winner_share(net),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_net", "profit_factor"], ascending=[False, False]).reset_index(drop=True)


def build_cost_stress(base_trades: pd.DataFrame, cost_multipliers: Sequence[float], round_trip_cost_pct: float) -> pd.DataFrame:
    if base_trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    keys = ["event_name", "family", "variant", "direction", "max_hold_bars"]
    for key, g in base_trades.groupby(keys, sort=False):
        gross = g["gross_return"].to_numpy(dtype=float)
        for cm in cost_multipliers:
            net = gross - round_trip_cost_pct * float(cm)
            rows.append(
                {
                    "event_name": key[0],
                    "family": key[1],
                    "variant": key[2],
                    "direction": key[3],
                    "max_hold_bars": int(key[4]),
                    "cost_multiplier": float(cm),
                    "count": int(len(net)),
                    "mean_net": float(np.nanmean(net)),
                    "profit_factor": _profit_factor(net),
                    "win_rate": float(np.nanmean(net > 0)),
                }
            )
    return pd.DataFrame(rows)


def build_delay_stress(delay_trades: pd.DataFrame) -> pd.DataFrame:
    if delay_trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    keys = ["event_name", "family", "variant", "direction", "max_hold_bars", "delay_bars"]
    for key, g in delay_trades.groupby(keys, sort=False):
        net = g["net_return"].to_numpy(dtype=float)
        rows.append(
            {
                "event_name": key[0],
                "family": key[1],
                "variant": key[2],
                "direction": key[3],
                "max_hold_bars": int(key[4]),
                "delay_bars": int(key[5]),
                "count": int(len(g)),
                "mean_net": float(np.nanmean(net)),
                "profit_factor": _profit_factor(net),
                "win_rate": float(np.nanmean(net > 0)),
            }
        )
    return pd.DataFrame(rows)


def _merge_fee_delay(summary: pd.DataFrame, cost_stress: pd.DataFrame, delay_stress: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    if not cost_stress.empty:
        fee2 = cost_stress[np.isclose(cost_stress["cost_multiplier"].astype(float), 2.0)][
            ["event_name", "direction", "max_hold_bars", "mean_net"]
        ].rename(columns={"mean_net": "fee2_mean_net"})
        out = out.merge(fee2, on=["event_name", "direction", "max_hold_bars"], how="left")
    else:
        out["fee2_mean_net"] = np.nan
    if not delay_stress.empty:
        d1 = delay_stress[delay_stress["delay_bars"].astype(int) == 1][
            ["event_name", "direction", "max_hold_bars", "mean_net"]
        ].rename(columns={"mean_net": "delay1_mean_net"})
        out = out.merge(d1, on=["event_name", "direction", "max_hold_bars"], how="left")
    else:
        out["delay1_mean_net"] = np.nan
    return out


# ---------------------------------------------------------------------------
# Matched baseline
# ---------------------------------------------------------------------------


def _baseline_candidate_groups(summary: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if summary.empty:
        return summary
    cond = (
        (summary["count"] >= int(args.min_count))
        & (summary["mean_net"] >= float(args.baseline_prefilter_mean_net))
        & (summary["profit_factor"] >= float(args.baseline_prefilter_pf))
    )
    return summary.loc[cond].copy().reset_index(drop=True)


def build_baseline_summary(
    args: argparse.Namespace,
    prelim: pd.DataFrame,
    base_trades: pd.DataFrame,
    all_event_times: set[pd.Timestamp],
    specs: Sequence[EventSpec],
) -> pd.DataFrame:
    if prelim.empty:
        print("[aggregate] matched baseline skipped: no groups passed preliminary filters", flush=True)
        return pd.DataFrame()
    print(f"[aggregate] matched baseline groups={len(prelim)}", flush=True)
    rng = np.random.default_rng(int(args.baseline_seed))
    chunks = _month_chunks(args.start_date, args.end_date)
    max_hold_needed = int(max(prelim["max_hold_bars"].astype(int).max(), max(_parse_csv_ints(args.max_hold_bars_list))))
    lookback_bars = int(args.cvd_norm_window) + max(_parse_csv_ints(args.impulse_bars_list)) + 20
    load_lookback = pd.Timedelta(seconds=max(lookback_bars * 5, 3600))
    load_forward = pd.Timedelta(seconds=(max_hold_needed + max(_parse_csv_ints(args.delay_bars_list)) + 5) * 5)

    rows: list[dict[str, object]] = []
    with ProgressReporter("[aggregate] matched baseline", total=len(prelim), every=max(1, len(prelim) // 10 or 1)) as pr:
        for gi, g in prelim.iterrows():
            event_name = str(g["event_name"])
            direction = str(g["direction"])
            max_hold = int(g["max_hold_bars"])
            ev = base_trades[
                (base_trades["event_name"] == event_name)
                & (base_trades["direction"] == direction)
                & (base_trades["max_hold_bars"].astype(int) == max_hold)
            ]
            if ev.empty:
                pr.update(int(gi) + 1)
                continue
            sample_ev = ev.sample(n=min(len(ev), int(args.baseline_max_events_per_group)), random_state=int(args.baseline_seed))
            key_counts = sample_ev.groupby(list(MATCHED_BASELINE_COLUMNS), sort=False).size().to_dict()
            target_per_key = {k if isinstance(k, tuple) else (k,): int(v) * int(args.baseline_samples) for k, v in key_counts.items()}
            pseudo_parts: list[pd.DataFrame] = []
            for chunk_start, chunk_end in chunks:
                load_start = max(pd.Timestamp(args.warmup_start_date), chunk_start - load_lookback)
                load_end = chunk_end + load_forward
                raw = load_trade_bars_range(args, load_start, load_end)
                if raw.empty:
                    continue
                feat, _ = build_features(raw, cvd_norm_window=int(args.cvd_norm_window), notional_z_window=int(args.notional_z_window))
                in_chunk = (feat.index >= chunk_start) & (feat.index <= chunk_end)
                if not np.any(in_chunk):
                    continue
                pos_all = np.flatnonzero(in_chunk)
                times = pd.to_datetime(feat.index[pos_all])
                not_event = ~pd.Series(times).isin(all_event_times).to_numpy(dtype=bool)
                pos_all = pos_all[not_event]
                if pos_all.size == 0:
                    continue
                key_frame = pd.DataFrame(
                    {
                        "year": feat["year"].iloc[pos_all].to_numpy(dtype=int),
                        "month": feat["month"].iloc[pos_all].to_numpy(dtype=int),
                        "session": feat["session"].iloc[pos_all].astype(str).to_numpy(),
                        "regime": feat["regime"].iloc[pos_all].astype(str).to_numpy(),
                        "volatility_bucket": feat["volatility_bucket"].iloc[pos_all].to_numpy(dtype=int),
                        "direction": direction,
                    }
                )
                pools = key_frame.groupby(list(MATCHED_BASELINE_COLUMNS), sort=False).indices
                baseline_events: list[pd.DataFrame] = []
                for key, idxs in pools.items():
                    kk = key if isinstance(key, tuple) else (key,)
                    need = target_per_key.get(kk, 0)
                    if need <= 0:
                        continue
                    pool_pos = pos_all[np.asarray(idxs, dtype=np.int64)]
                    take = min(len(pool_pos), need)
                    if take <= 0:
                        continue
                    chosen = rng.choice(pool_pos, size=take, replace=len(pool_pos) < take)
                    be = pd.DataFrame(
                        {
                            "event_name": event_name,
                            "family": "matched_baseline",
                            "variant": "matched_baseline",
                            "direction": direction,
                            "signal_time": feat.index[chosen],
                            "signal_pos": chosen.astype(np.int64),
                            "impulse_bars": 0,
                            "cvd_slope_mult": np.nan,
                            "flat_ratio": np.nan,
                            "impulse_cvd_change": np.nan,
                            "stall_cvd_delta": np.nan,
                            "price_impulse_ret": np.nan,
                            "year": feat["year"].iloc[chosen].to_numpy(dtype=int),
                            "month": feat["month"].iloc[chosen].to_numpy(dtype=int),
                            "session": feat["session"].iloc[chosen].astype(str).to_numpy(),
                            "regime": feat["regime"].iloc[chosen].astype(str).to_numpy(),
                            "volatility_bucket": feat["volatility_bucket"].iloc[chosen].to_numpy(dtype=int),
                        }
                    )
                    baseline_events.append(be)
                if baseline_events:
                    bev = pd.concat(baseline_events, ignore_index=True)
                    pseudo = replay_positions(
                        feat,
                        bev,
                        max_hold_bars=max_hold,
                        delay_bars=0,
                        tp_pct=float(args.tp_pct),
                        round_trip_cost_pct=float(args.round_trip_cost_pct),
                        cost_multiplier=1.0,
                        include_detail=False,
                    )
                    if not pseudo.empty:
                        pseudo_parts.append(pseudo[["net_return", "gross_return", "tp_hit"]])
            if pseudo_parts:
                p = pd.concat(pseudo_parts, ignore_index=True)
                event_mean = float(g["mean_net"])
                base_mean = float(p["net_return"].mean())
                rows.append(
                    {
                        "event_name": event_name,
                        "direction": direction,
                        "max_hold_bars": max_hold,
                        "event_count": int(g["count"]),
                        "baseline_count": int(len(p)),
                        "event_mean_net": event_mean,
                        "baseline_mean_net": base_mean,
                        "matched_excess_mean_net": float(event_mean - base_mean),
                        "baseline_profit_factor": _profit_factor(p["net_return"].to_numpy(dtype=float)),
                        "baseline_win_rate": float((p["net_return"] > 0).mean()),
                        "baseline_tp_hit_rate": float(p["tp_hit"].mean()),
                    }
                )
            pr.update(int(gi) + 1)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def build_decisions(
    summary: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    min_count: int,
    min_events_per_year: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if summary.empty:
        empty = pd.DataFrame()
        return empty, empty, empty
    out = summary.copy()
    if not baseline.empty:
        out = out.merge(
            baseline[["event_name", "direction", "max_hold_bars", "matched_excess_mean_net", "baseline_mean_net", "baseline_count"]],
            on=["event_name", "direction", "max_hold_bars"],
            how="left",
        )
    else:
        out["matched_excess_mean_net"] = np.nan
        out["baseline_mean_net"] = np.nan
        out["baseline_count"] = 0

    decisions: list[str] = []
    reasons: list[str] = []
    for _, r in out.iterrows():
        fails: list[str] = []
        if int(r["count"]) < min_count:
            fails.append("count_lt_min")
        if float(r["events_per_year"]) < min_events_per_year:
            fails.append("events_per_year_lt_min")
        if float(r["mean_net"]) <= 0:
            fails.append("mean_net_le_0")
        if float(r["profit_factor"]) < 1.15:
            fails.append("pf_lt_1p15")
        if pd.notna(r.get("fee2_mean_net")) and float(r["fee2_mean_net"]) <= 0:
            fails.append("fee2_le_0")
        if pd.notna(r.get("delay1_mean_net")) and float(r["delay1_mean_net"]) <= -0.0002:
            fails.append("delay1_too_weak")
        if int(r["positive_years"]) < 3:
            fails.append("positive_years_lt_3")
        if float(r["top5_winner_share"]) > 0.35:
            fails.append("top5_winner_share_gt_0p35")
        if pd.notna(r.get("matched_excess_mean_net")) and float(r["matched_excess_mean_net"]) <= 0:
            fails.append("matched_excess_le_0")
        if not fails:
            decisions.append("promote_to_backtest_candidate")
            reasons.append("passed_research_filters")
        elif float(r["mean_net"]) > 0 and float(r["profit_factor"]) >= 1.05:
            decisions.append("research_continue")
            reasons.append(";".join(fails))
        else:
            decisions.append("rejected")
            reasons.append(";".join(fails))
    out["decision"] = decisions
    out["reason"] = reasons
    candidates = out[out["decision"] == "promote_to_backtest_candidate"].copy()
    rejected = out[out["decision"] == "rejected"].copy()
    return out.sort_values(["decision", "mean_net"], ascending=[True, False]), candidates, rejected


def write_reports(
    args: argparse.Namespace,
    *,
    specs: Sequence[EventSpec],
    feature_columns: list[str],
    input_rows: int,
    events: pd.DataFrame,
    base_trades: pd.DataFrame,
    delay_trades: pd.DataFrame,
    summary: pd.DataFrame,
    cost_stress: pd.DataFrame,
    delay_stress: pd.DataFrame,
    baseline_summary: pd.DataFrame,
) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary2 = _merge_fee_delay(summary, cost_stress, delay_stress)
    decisions, candidates, rejected = build_decisions(
        summary2,
        baseline_summary,
        min_count=int(args.min_count),
        min_events_per_year=float(args.min_events_per_year),
    )

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
        "tp_pct": float(args.tp_pct),
        "max_hold_bars_list": list(_parse_csv_ints(args.max_hold_bars_list)),
        "round_trip_cost_pct": float(args.round_trip_cost_pct),
        "cost_multipliers": list(_parse_csv_floats(args.cost_multipliers)),
        "delay_bars_list": list(_parse_csv_ints(args.delay_bars_list)),
        "input_rows": int(input_rows),
        "event_spec_count": int(len(specs) * 2),
        "event_count": int(len(events)),
        "replay_trade_rows": int(len(base_trades)),
        "replay_group_count": int(len(summary)),
        "candidate_count": int(len(candidates)),
        "causal_lookahead_count": int(base_trades.get("lookahead_flag", pd.Series(dtype=int)).sum()) if not base_trades.empty else 0,
        "causal_policy": CAUSAL_POLICY,
        "matched_baseline_columns": list(MATCHED_BASELINE_COLUMNS),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")

    if not events.empty:
        ev_summary = events.groupby(["event_name", "family", "variant", "direction"], sort=False).agg(
            count=("signal_time", "size"),
            first_signal=("signal_time", "min"),
            last_signal=("signal_time", "max"),
        ).reset_index()
        ev_summary.to_csv(out_dir / "01_event_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "01_event_summary.csv", index=False)
    summary2.to_csv(out_dir / "02_replay_variant_summary.csv", index=False)
    if args.write_full_trades:
        base_trades.to_csv(out_dir / "03_replay_trades.csv", index=False)
    else:
        base_trades.head(int(args.trade_sample_size)).to_csv(out_dir / "03_replay_trades_sample.csv", index=False)
    if not base_trades.empty:
        y = base_trades.groupby(["event_name", "direction", "max_hold_bars", "year"], sort=False).agg(
            count=("net_return", "size"), mean_net=("net_return", "mean"), win_rate=("net_return", lambda s: float((s > 0).mean())),
        ).reset_index()
        y.to_csv(out_dir / "04_yearly_breakdown.csv", index=False)
        s = base_trades.groupby(["event_name", "direction", "max_hold_bars", "session"], sort=False).agg(
            count=("net_return", "size"), mean_net=("net_return", "mean"), win_rate=("net_return", lambda x: float((x > 0).mean())),
        ).reset_index()
        s.to_csv(out_dir / "05_session_breakdown.csv", index=False)
        r = base_trades.groupby(["event_name", "direction", "max_hold_bars", "regime"], sort=False).agg(
            count=("net_return", "size"), mean_net=("net_return", "mean"), win_rate=("net_return", lambda x: float((x > 0).mean())),
        ).reset_index()
        r.to_csv(out_dir / "06_regime_breakdown.csv", index=False)
    else:
        pd.DataFrame().to_csv(out_dir / "04_yearly_breakdown.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "05_session_breakdown.csv", index=False)
        pd.DataFrame().to_csv(out_dir / "06_regime_breakdown.csv", index=False)
    cost_stress.to_csv(out_dir / "07_cost_stress.csv", index=False)
    delay_stress.to_csv(out_dir / "08_delay_stress.csv", index=False)
    baseline_summary.to_csv(out_dir / "09_matched_baseline_summary.csv", index=False)
    decisions.to_csv(out_dir / "10_research_decision.csv", index=False)
    candidates.to_csv(out_dir / "10_candidate_shortlist.csv", index=False)
    rejected.to_csv(out_dir / "11_rejected_candidates.csv", index=False)
    audit_cols = [
        "event_name", "direction", "signal_time", "entry_time", "expected_entry_time",
        "entry_not_next_open_flag", "forward_window_valid_flag", "lookahead_flag", "exit_time", "exit_reason",
        "max_hold_bars", "delay_bars", "gross_return", "net_return",
    ]
    base_trades[[c for c in audit_cols if c in base_trades.columns]].head(100000).to_csv(out_dir / "12_causal_audit.csv", index=False)
    events.head(int(args.event_sample_size)).to_csv(out_dir / "13_event_sample.csv", index=False)
    (out_dir / "14_feature_columns.json").write_text(json.dumps({"feature_columns": feature_columns}, ensure_ascii=False, indent=2), encoding="utf-8")

    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title="ETH HF CVD Impulse Stall Reversal 5s V1")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.primary_timeframe != "5s":
        print(f"[warn] this idea is designed for 5s trade bars; got {args.primary_timeframe}", flush=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {SCRIPT_NAME} version={SCRIPT_VERSION}", flush=True)
    print(f"[run] symbol={args.symbol} primary={args.primary_timeframe} range={args.start_date}->{args.end_date} warmup={args.warmup_start_date}", flush=True)
    print(f"[run] tp_pct={args.tp_pct:.4f} max_hold_bars={args.max_hold_bars_list}", flush=True)

    specs = build_specs(args)
    max_hold_bars = _parse_csv_ints(args.max_hold_bars_list)
    delay_bars_list = _parse_csv_ints(args.delay_bars_list)
    cost_multipliers = _parse_csv_floats(args.cost_multipliers)
    max_hold_max = max(max_hold_bars)
    delay_max = max(delay_bars_list)
    lookback_bars = int(args.cvd_norm_window) + max(_parse_csv_ints(args.impulse_bars_list)) + 20
    load_lookback = pd.Timedelta(seconds=max(lookback_bars * 5, 3600))
    load_forward = pd.Timedelta(seconds=(max_hold_max + delay_max + 5) * 5)
    chunks = _month_chunks(args.start_date, args.end_date)

    all_events_parts: list[pd.DataFrame] = []
    base_trade_parts: list[pd.DataFrame] = []
    delay_trade_parts: list[pd.DataFrame] = []
    feature_columns: list[str] = []
    input_rows = 0

    print(f"[load] processing {len(chunks)} monthly chunks with 5s bars", flush=True)
    with ProgressReporter("[run] monthly chunks", total=len(chunks), every=max(1, int(args.progress_every))) as pr:
        for ci, (chunk_start, chunk_end) in enumerate(chunks, start=1):
            load_start = max(pd.Timestamp(args.warmup_start_date), chunk_start - load_lookback)
            load_end = chunk_end + load_forward
            raw = load_trade_bars_range(args, load_start, load_end)
            input_rows += int(len(raw))
            if raw.empty:
                pr.update(ci)
                continue
            feat, fcols = build_features(raw, cvd_norm_window=int(args.cvd_norm_window), notional_z_window=int(args.notional_z_window))
            feature_columns = fcols
            events = build_events_for_chunk(
                feat,
                specs,
                chunk_start=chunk_start,
                chunk_end=chunk_end,
                cooldown_bars=int(args.cooldown_bars),
            )
            if not events.empty:
                all_events_parts.append(events.drop(columns=["signal_pos"], errors="ignore"))
                for mh in max_hold_bars:
                    base = replay_positions(
                        feat,
                        events,
                        max_hold_bars=mh,
                        delay_bars=0,
                        tp_pct=float(args.tp_pct),
                        round_trip_cost_pct=float(args.round_trip_cost_pct),
                        cost_multiplier=1.0,
                        include_detail=True,
                    )
                    if not base.empty:
                        base_trade_parts.append(base)
                for db in delay_bars_list:
                    if db == 0:
                        continue
                    for mh in max_hold_bars:
                        d = replay_positions(
                            feat,
                            events,
                            max_hold_bars=mh,
                            delay_bars=db,
                            tp_pct=float(args.tp_pct),
                            round_trip_cost_pct=float(args.round_trip_cost_pct),
                            cost_multiplier=1.0,
                            include_detail=False,
                        )
                        if not d.empty:
                            delay_trade_parts.append(
                                d[["event_name", "family", "variant", "direction", "max_hold_bars", "delay_bars", "net_return", "gross_return"]]
                            )
            pr.update(ci)

    events_all = pd.concat(all_events_parts, ignore_index=True) if all_events_parts else pd.DataFrame()
    base_trades = pd.concat(base_trade_parts, ignore_index=True) if base_trade_parts else pd.DataFrame()
    delay_trades = pd.concat(delay_trade_parts, ignore_index=True) if delay_trade_parts else pd.DataFrame()
    print(f"[events] rows={len(events_all):,} specs={len(specs) * 2}", flush=True)
    print(f"[forward] base replay rows={len(base_trades):,} delay rows={len(delay_trades):,}", flush=True)

    print("[aggregate] summaries", flush=True)
    summary = summarize_replay(base_trades, start_date=args.start_date, end_date=args.end_date)
    cost_stress = build_cost_stress(base_trades, cost_multipliers, float(args.round_trip_cost_pct))
    delay_stress = build_delay_stress(pd.concat([base_trades[["event_name", "family", "variant", "direction", "max_hold_bars", "delay_bars", "net_return", "gross_return"]], delay_trades], ignore_index=True) if not base_trades.empty else delay_trades)
    summary_for_prefilter = _merge_fee_delay(summary, cost_stress, delay_stress)
    prelim = _baseline_candidate_groups(summary_for_prefilter, args)
    print("[aggregate] matched baseline prep", flush=True)
    all_event_times = set(pd.to_datetime(events_all["signal_time"])) if not events_all.empty else set()
    baseline_summary = build_baseline_summary(args, prelim, base_trades, all_event_times, specs)

    print("[write] report files", flush=True)
    write_reports(
        args,
        specs=specs,
        feature_columns=feature_columns,
        input_rows=input_rows,
        events=events_all,
        base_trades=base_trades,
        delay_trades=delay_trades,
        summary=summary,
        cost_stress=cost_stress,
        delay_stress=delay_stress,
        baseline_summary=baseline_summary,
    )
    print(f"[done] report_dir={Path(args.out_dir)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
