#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH 1m raw volume-shadow feature/filter probe V1.

Research-only univariate filter probe. This script keeps the raw lower/upper
volume-shadow universe, then tests one causal feature condition at a time to see
which dimensions lift high-frequency, high-win-rate path outcomes.

Scope
-----
- lower volume shadow -> diagnostic LONG return path
- upper volume shadow -> diagnostic SHORT return path
- two fixed shadow-strength layers: standard and extreme
- one filter family/condition at a time: geometry, volume, trend, vol, session,
  delta/taker-buy when present
- no combined condition optimizer, no footprint/range-bar mixing, no live change

Causal policy
-------------
1m bars are left-labeled by bar start time. An event bar is only known after the
bar closes. signal_time = event_bar_start + 1 minute. entry_time is the next
bar open, optionally plus diagnostic entry-delay. All tested filters use only
features available on or before the event bar close. Forward returns/MFE/MAE
are only used as labels for analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
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

SCRIPT_NAME = "eth_1m_raw_volume_shadow_filter_probe_v1"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_MF_1M_RAW_VOLUME_SHADOW_FILTER_PROBE_V1"
EDGE_ID = "RESEARCH_ONLY_ETH_MF_1M_RAW_VOLUME_SHADOW_FILTER_PROBE_V1"
TITLE = "ETH 1m Raw Volume Shadow Filter Probe V1"
DEFAULT_OUT_DIR = "data/reports/research/eth_1m_raw_volume_shadow_filter_probe_v1"
BAR_DELTA = pd.Timedelta(minutes=1)


@dataclass(frozen=True)
class ShadowThresholds:
    wick_share_min: float = 0.50
    wick_atr_min: float = 0.55
    volume_ratio_min: float = 2.0
    extreme_wick_share_min: float = 0.65
    extreme_wick_atr_min: float = 0.85
    extreme_volume_ratio_min: float = 3.0


@dataclass(frozen=True)
class FilterSpec:
    name: str
    family: str
    description: str
    mask: pd.Series


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Raw ETH 1m volume-shadow one-filter-at-a-time probe for high-trade/high-win edge discovery.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--no-build-missing", action="store_true")
    p.add_argument("--force-rebuild", action="store_true")
    p.add_argument("--round-trip-cost-pct", type=float, default=0.0011)
    p.add_argument("--horizons", default="15,30,60,120,240")
    p.add_argument("--entry-delay-bars-list", default="0,1,2")
    p.add_argument("--min-candidate-trades", type=int, default=800)
    p.add_argument("--min-candidate-win-rate", type=float, default=0.55)
    p.add_argument("--event-sample-size", type=int, default=5000)
    p.add_argument("--candidate-sample-size", type=int, default=2000)
    p.add_argument("--progress-every", type=int, default=5000)
    p.add_argument("--write-slim-path-labels", action="store_true", help="Write slim event x delay x horizon labels. Off by default to keep review packs small.")

    # Fixed raw shadow vocabulary. Keep defaults for comparable reports.
    p.add_argument("--wick-share-min", type=float, default=0.50)
    p.add_argument("--wick-atr-min", type=float, default=0.55)
    p.add_argument("--volume-ratio-min", type=float, default=2.0)
    p.add_argument("--extreme-wick-share-min", type=float, default=0.65)
    p.add_argument("--extreme-wick-atr-min", type=float, default=0.85)
    p.add_argument("--extreme-volume-ratio-min", type=float, default=3.0)
    return p.parse_args(argv)


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    vals: list[int] = []
    for part in str(raw).split(","):
        text = part.strip()
        if text:
            vals.append(int(text))
    out = tuple(dict.fromkeys(vals))
    if not out:
        raise ValueError("integer csv must not be empty")
    return out


def _safe_divide(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> pd.Series:
    aa = pd.Series(a) if not isinstance(a, pd.Series) else pd.to_numeric(a, errors="coerce")
    bb = pd.Series(b) if not isinstance(b, pd.Series) else pd.to_numeric(b, errors="coerce")
    return (aa / bb.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def _true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    return pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _research_window_mask(index: pd.DatetimeIndex, start_date: str, end_date: str) -> pd.Series:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if end_ts == end_ts.normalize() and len(str(end_date).strip()) <= 10:
        end_exclusive = end_ts + pd.Timedelta(days=1)
    else:
        end_exclusive = end_ts + BAR_DELTA
    return pd.Series((index >= start_ts) & (index < end_exclusive), index=index)


def session_label(index: pd.DatetimeIndex) -> np.ndarray:
    hour = index.hour
    labels = np.full(len(index), "asia", dtype=object)
    labels[(hour >= 8) & (hour < 16)] = "eu_london"
    labels[(hour >= 16) & (hour < 24)] = "us"
    return labels


def _profit_factor(ret: pd.Series) -> float:
    r = pd.to_numeric(ret, errors="coerce").dropna()
    if r.empty:
        return float("nan")
    gp = float(r[r > 0].sum())
    gl = float(-r[r <= 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else float("nan")
    return gp / gl


def _equity_stats(ret: pd.Series) -> dict[str, float]:
    r = pd.to_numeric(ret, errors="coerce").fillna(0.0)
    if r.empty:
        return {"total_return": np.nan, "equity_end": np.nan, "max_drawdown": np.nan, "return_over_drawdown": np.nan}
    eq = (1.0 + r).cumprod()
    dd = eq / eq.cummax() - 1.0
    total = float(eq.iloc[-1] - 1.0)
    max_dd = float(dd.min())
    return {
        "total_return": total,
        "equity_end": float(eq.iloc[-1]),
        "max_drawdown": max_dd,
        "return_over_drawdown": float(total / abs(max_dd)) if max_dd < 0 else np.nan,
    }


def _top5_winner_share(ret: pd.Series) -> float:
    r = pd.to_numeric(ret, errors="coerce").dropna()
    wins = r[r > 0].sort_values(ascending=False)
    gross = float(wins.sum())
    if gross <= 0 or wins.empty:
        return float("nan")
    return float(wins.head(5).sum() / gross)


def _max_consecutive_losses(ret: pd.Series) -> int:
    r = pd.to_numeric(ret, errors="coerce").dropna().to_numpy(dtype=float)
    best = cur = 0
    for x in r:
        if x <= 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def _max_days_without_event(times: pd.Series) -> float:
    ts = pd.to_datetime(times, errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return float("nan")
    return float(ts.diff().dropna().max() / pd.Timedelta(days=1))


def _events_per_month(times: pd.Series) -> float:
    ts = pd.to_datetime(times, errors="coerce").dropna().sort_values()
    if ts.empty:
        return float("nan")
    months = max((ts.max() - ts.min()) / pd.Timedelta(days=30.4375), 1.0)
    return float(len(ts) / months)


def _thresholds_from_args(args: argparse.Namespace) -> ShadowThresholds:
    return ShadowThresholds(
        wick_share_min=float(args.wick_share_min),
        wick_atr_min=float(args.wick_atr_min),
        volume_ratio_min=float(args.volume_ratio_min),
        extreme_wick_share_min=float(args.extreme_wick_share_min),
        extreme_wick_atr_min=float(args.extreme_wick_atr_min),
        extreme_volume_ratio_min=float(args.extreme_volume_ratio_min),
    )


def load_bars(args: argparse.Namespace) -> pd.DataFrame:
    print(f"[load] OKXTradeBarLoader {args.symbol} {args.timeframe} {args.warmup_start_date}->{args.end_date}", flush=True)
    loader = OKXTradeBarLoader(
        symbol=args.symbol,
        timeframe=args.timeframe,
        data_dir=args.data_dir,
        db_name=args.db_name,
    )
    df = loader.fetch_data_by_date_range(
        args.warmup_start_date,
        args.end_date,
        chunksize=int(args.chunksize),
        force_rebuild=bool(args.force_rebuild),
        build_missing=not bool(args.no_build_missing),
    )
    if df.empty:
        raise RuntimeError(f"No trade bars loaded for {args.symbol} {args.timeframe}")
    out = df.copy().sort_index()
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"Trade bars missing required columns: {missing}")
    print(f"       rows={len(out):,} range={out.index.min()} -> {out.index.max()} cols={len(out.columns)}", flush=True)
    return out


def build_features(bars: pd.DataFrame, th: ShadowThresholds) -> pd.DataFrame:
    print("[features] building causal event-bar features", flush=True)
    df = bars.copy().sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    body_low = df[["open", "close"]].min(axis=1)
    body_high = df[["open", "close"]].max(axis=1)
    out = pd.DataFrame(index=df.index)
    out[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]]
    out["range_pct"] = _safe_divide(rng, df["close"]).to_numpy()
    out["body_pct"] = _safe_divide((df["close"] - df["open"]).abs(), df["close"]).to_numpy()
    out["close_pos"] = _safe_divide(df["close"] - df["low"], rng).to_numpy()
    out["lower_wick"] = (body_low - df["low"]).clip(lower=0.0)
    out["upper_wick"] = (df["high"] - body_high).clip(lower=0.0)
    out["lower_wick_share"] = _safe_divide(out["lower_wick"], rng).to_numpy()
    out["upper_wick_share"] = _safe_divide(out["upper_wick"], rng).to_numpy()
    out["tr"] = _true_range(df)
    out["atr"] = out["tr"].rolling(60, min_periods=30).mean()
    out["atr_pct"] = _safe_divide(out["atr"], df["close"]).to_numpy()
    out["lower_wick_atr"] = _safe_divide(out["lower_wick"], out["atr"]).to_numpy()
    out["upper_wick_atr"] = _safe_divide(out["upper_wick"], out["atr"]).to_numpy()
    vol_base = df["volume"].shift(1).rolling(240, min_periods=60).median()
    out["volume_ratio"] = _safe_divide(df["volume"], vol_base).to_numpy()
    out["ret_30"] = df["close"].pct_change(30)
    out["ret_120"] = df["close"].pct_change(120)
    out["ret_720"] = df["close"].pct_change(720)
    out["ema_240"] = df["close"].ewm(span=240, adjust=False, min_periods=240).mean()
    out["ema240_slope_60"] = out["ema_240"] / out["ema_240"].shift(60) - 1.0

    delta = pd.to_numeric(df.get("delta_notional", np.nan), errors="coerce")
    notional = pd.to_numeric(df.get("notional", np.nan), errors="coerce").abs()
    out["delta_ratio"] = _safe_divide(delta, notional).to_numpy()
    if "taker_buy_ratio" in df.columns:
        out["taker_buy_ratio"] = pd.to_numeric(df["taker_buy_ratio"], errors="coerce")
    elif "buy_volume" in df.columns:
        buy_volume = pd.to_numeric(df["buy_volume"], errors="coerce")
        out["taker_buy_ratio"] = _safe_divide(buy_volume, df["volume"]).to_numpy()
    else:
        out["taker_buy_ratio"] = np.nan

    out["session"] = session_label(out.index)
    out["vol_regime"] = pd.cut(
        out["atr_pct"],
        bins=[-np.inf, 0.0015, 0.0030, 0.0050, np.inf],
        labels=["very_low_vol", "low_mid_vol", "mid_high_vol", "extreme_vol"],
    ).astype("object").fillna("NA")
    out["trend_regime"] = np.select(
        [
            (df["close"] > out["ema_240"]) & (out["ema240_slope_60"] > 0.0005),
            (df["close"] < out["ema_240"]) & (out["ema240_slope_60"] < -0.0005),
        ],
        ["uptrend", "downtrend"],
        default="range_or_transition",
    )

    out["lower_volume_shadow"] = (
        (out["lower_wick_share"] >= th.wick_share_min)
        & (out["lower_wick_atr"] >= th.wick_atr_min)
        & (out["volume_ratio"] >= th.volume_ratio_min)
    )
    out["upper_volume_shadow"] = (
        (out["upper_wick_share"] >= th.wick_share_min)
        & (out["upper_wick_atr"] >= th.wick_atr_min)
        & (out["volume_ratio"] >= th.volume_ratio_min)
    )
    out["lower_extreme_volume_shadow"] = (
        (out["lower_wick_share"] >= th.extreme_wick_share_min)
        & (out["lower_wick_atr"] >= th.extreme_wick_atr_min)
        & (out["volume_ratio"] >= th.extreme_volume_ratio_min)
    )
    out["upper_extreme_volume_shadow"] = (
        (out["upper_wick_share"] >= th.extreme_wick_share_min)
        & (out["upper_wick_atr"] >= th.extreme_wick_atr_min)
        & (out["volume_ratio"] >= th.extreme_volume_ratio_min)
    )
    return out


def _event_frame_from_mask(
    features: pd.DataFrame,
    *,
    mask: pd.Series,
    event_name: str,
    family: str,
    direction: str,
    side: int,
    strength: str,
) -> pd.DataFrame:
    bool_mask = mask.fillna(False).to_numpy(dtype=bool)
    idx = features.index[bool_mask]
    if len(idx) == 0:
        return pd.DataFrame()
    pos_map = pd.Series(np.arange(len(features), dtype=int), index=features.index)
    event_pos = pos_map.reindex(idx).to_numpy(dtype=int)
    valid = event_pos + 1 < len(features)
    if not valid.any():
        return pd.DataFrame()
    idx = idx[valid]
    event_pos = event_pos[valid]
    f = features.loc[idx]
    side_arr = np.full(len(f), int(side), dtype=int)
    favorable_close_pos = f["close_pos"].to_numpy(dtype=float) if side > 0 else 1.0 - f["close_pos"].to_numpy(dtype=float)
    event_wick_share = f["lower_wick_share"].to_numpy(dtype=float) if side > 0 else f["upper_wick_share"].to_numpy(dtype=float)
    event_wick_atr = f["lower_wick_atr"].to_numpy(dtype=float) if side > 0 else f["upper_wick_atr"].to_numpy(dtype=float)
    signed_delta_against_entry = -side_arr * f["delta_ratio"].to_numpy(dtype=float)
    taker_against_entry = np.where(side_arr > 0, 1.0 - f["taker_buy_ratio"].to_numpy(dtype=float), f["taker_buy_ratio"].to_numpy(dtype=float))
    prior_move_against_30 = -side_arr * f["ret_30"].to_numpy(dtype=float)
    prior_move_against_120 = -side_arr * f["ret_120"].to_numpy(dtype=float)
    prior_move_against_720 = -side_arr * f["ret_720"].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "event_name": event_name,
            "family": family,
            "direction": direction,
            "side": int(side),
            "strength": strength,
            "event_bar_time": idx,
            "event_bar_pos": event_pos,
            "signal_bar_time": features.index[event_pos],
            "signal_bar_pos": event_pos,
            "signal_time": features.index[event_pos] + BAR_DELTA,
            "signal_available_time": features.index[event_pos] + BAR_DELTA,
            "session": f["session"].to_numpy(),
            "vol_regime": f["vol_regime"].astype(str).to_numpy(),
            "trend_regime": f["trend_regime"].astype(str).to_numpy(),
            "close_pos": f["close_pos"].to_numpy(dtype=float),
            "favorable_close_pos": favorable_close_pos,
            "volume_ratio": f["volume_ratio"].to_numpy(dtype=float),
            "event_wick_share": event_wick_share,
            "event_wick_atr": event_wick_atr,
            "range_pct": f["range_pct"].to_numpy(dtype=float),
            "atr_pct": f["atr_pct"].to_numpy(dtype=float),
            "ret_30": f["ret_30"].to_numpy(dtype=float),
            "ret_120": f["ret_120"].to_numpy(dtype=float),
            "ret_720": f["ret_720"].to_numpy(dtype=float),
            "prior_move_against_30": prior_move_against_30,
            "prior_move_against_120": prior_move_against_120,
            "prior_move_against_720": prior_move_against_720,
            "delta_ratio": f["delta_ratio"].to_numpy(dtype=float),
            "taker_buy_ratio": f["taker_buy_ratio"].to_numpy(dtype=float),
            "signed_delta_against_entry": signed_delta_against_entry,
            "taker_against_entry": taker_against_entry,
            "direction_with_trend": np.where(
                ((side > 0) & (f["trend_regime"].astype(str).to_numpy() == "uptrend"))
                | ((side < 0) & (f["trend_regime"].astype(str).to_numpy() == "downtrend")),
                True,
                False,
            ),
            "direction_countertrend_exhaustion": np.where(
                ((side > 0) & (f["trend_regime"].astype(str).to_numpy() == "downtrend"))
                | ((side < 0) & (f["trend_regime"].astype(str).to_numpy() == "uptrend")),
                True,
                False,
            ),
            "event_low": f["low"].to_numpy(dtype=float),
            "event_high": f["high"].to_numpy(dtype=float),
            "event_open": f["open"].to_numpy(dtype=float),
            "event_close": f["close"].to_numpy(dtype=float),
        }
    )


def build_raw_shadow_events(features: pd.DataFrame) -> pd.DataFrame:
    print("[events] building raw lower-long and upper-short shadow events", flush=True)
    f = features
    specs = [
        ("lower_volume_shadow_long", "raw_lower_shadow", "LONG", 1, "standard", f["lower_volume_shadow"]),
        ("lower_extreme_volume_shadow_long", "raw_lower_shadow", "LONG", 1, "extreme", f["lower_extreme_volume_shadow"]),
        ("upper_volume_shadow_short", "raw_upper_shadow", "SHORT", -1, "standard", f["upper_volume_shadow"]),
        ("upper_extreme_volume_shadow_short", "raw_upper_shadow", "SHORT", -1, "extreme", f["upper_extreme_volume_shadow"]),
    ]
    frames = [
        _event_frame_from_mask(f, mask=mask, event_name=name, family=family, direction=direction, side=side, strength=strength)
        for name, family, direction, side, strength, mask in specs
    ]
    non_empty = [x for x in frames if not x.empty]
    if not non_empty:
        return pd.DataFrame()
    out = pd.concat(non_empty, ignore_index=True).sort_values(["event_bar_time", "event_name"]).reset_index(drop=True)
    out["event_id"] = np.arange(len(out), dtype=int)
    print(f"[events] total={len(out):,}", flush=True)
    return out


def _forward_roll_max(arr: np.ndarray, horizon: int) -> np.ndarray:
    s = pd.Series(arr)
    return s.iloc[::-1].rolling(horizon + 1, min_periods=horizon + 1).max().iloc[::-1].to_numpy(dtype=float)


def _forward_roll_min(arr: np.ndarray, horizon: int) -> np.ndarray:
    s = pd.Series(arr)
    return s.iloc[::-1].rolling(horizon + 1, min_periods=horizon + 1).min().iloc[::-1].to_numpy(dtype=float)


def _safe_take(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
    out = np.full(len(pos), np.nan, dtype=float)
    valid = (pos >= 0) & (pos < len(arr))
    if valid.any():
        out[valid] = arr[pos[valid]]
    return out


def build_path_labels(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    entry_delays: tuple[int, ...],
    round_trip_cost_pct: float,
) -> pd.DataFrame:
    print("[labels] attaching forward return/MFE/MAE labels for filter tests", flush=True)
    if events.empty:
        return pd.DataFrame()
    max_h = max(horizons)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    open_ = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    index = bars.index

    fwd_high = {h: _forward_roll_max(high, h) for h in horizons}
    fwd_low = {h: _forward_roll_min(low, h) for h in horizons}

    frames: list[pd.DataFrame] = []
    progress = ProgressReporter(label="[labels] delay x horizon", total=len(entry_delays), every=1)
    for j, delay in enumerate(entry_delays, start=1):
        e = events.copy()
        e["entry_delay_bars"] = int(delay)
        entry_pos = e["signal_bar_pos"].to_numpy(dtype=int) + 1 + int(delay)
        e["entry_bar_pos"] = entry_pos
        valid = entry_pos + max_h < len(bars)
        e = e.loc[valid].copy()
        if e.empty:
            progress.update(j)
            continue
        entry_pos = e["entry_bar_pos"].to_numpy(dtype=int)
        side = e["side"].to_numpy(dtype=int)
        entry_price = _safe_take(open_, entry_pos)
        e["entry_time"] = index[entry_pos]
        e["entry_price"] = entry_price
        e["expected_entry_time"] = e["signal_time"] + pd.to_timedelta(int(delay), unit="m")
        e["entry_not_expected_time_flag"] = pd.to_datetime(e["entry_time"]) != pd.to_datetime(e["expected_entry_time"])
        e["lookahead_flag"] = pd.to_datetime(e["entry_time"]) < pd.to_datetime(e["signal_time"])
        e["context_available_time_flag"] = pd.to_datetime(e["signal_available_time"]) > pd.to_datetime(e["signal_time"])

        label_frames: list[pd.DataFrame] = []
        for h in horizons:
            hh = int(h)
            end_pos = entry_pos + hh
            end_close = _safe_take(close, end_pos)
            max_high = _safe_take(fwd_high[hh], entry_pos)
            min_low = _safe_take(fwd_low[hh], entry_pos)
            gross = np.where(side > 0, end_close / entry_price - 1.0, entry_price / end_close - 1.0)
            mfe = np.where(side > 0, max_high / entry_price - 1.0, entry_price / min_low - 1.0)
            mae = np.where(side > 0, min_low / entry_price - 1.0, entry_price / max_high - 1.0)
            part = e.copy()
            part["horizon_bars"] = hh
            part["exit_time"] = index[end_pos]
            part["exit_price"] = end_close
            part["gross_return"] = gross
            part["net_return"] = gross - float(round_trip_cost_pct)
            part["mfe"] = mfe
            part["mae"] = mae
            part["mfe_capture_ratio"] = np.where(mfe > 0, part["net_return"] / mfe, np.nan)
            label_frames.append(part)
        frames.append(pd.concat(label_frames, ignore_index=True))
        progress.update(j)
    progress.close()
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out
    out["direction_label"] = np.where(out["side"].astype(int) > 0, "LONG_lower_shadow", "SHORT_upper_shadow")
    return out


def build_filter_specs(labels: pd.DataFrame) -> list[FilterSpec]:
    print("[filters] building one-condition-at-a-time causal filter specs", flush=True)
    specs: list[FilterSpec] = []

    def add(name: str, family: str, description: str, mask: pd.Series | np.ndarray) -> None:
        m = pd.Series(mask, index=labels.index).fillna(False).astype(bool)
        specs.append(FilterSpec(name=name, family=family, description=description, mask=m))

    # Geometry and event-bar close quality.
    add("fav_close_pos_ge_0p60", "geometry_close", "event close in favorable 60% of event range", labels["favorable_close_pos"] >= 0.60)
    add("fav_close_pos_ge_0p70", "geometry_close", "event close in favorable 70% of event range", labels["favorable_close_pos"] >= 0.70)
    add("fav_close_pos_ge_0p80", "geometry_close", "event close in favorable 80% of event range", labels["favorable_close_pos"] >= 0.80)
    add("fav_close_pos_le_0p40", "geometry_close", "event close remains weak; diagnostic reject condition", labels["favorable_close_pos"] <= 0.40)
    add("event_wick_share_ge_0p65", "geometry_wick", "shadow share >= 65%", labels["event_wick_share"] >= 0.65)
    add("event_wick_share_ge_0p75", "geometry_wick", "shadow share >= 75%", labels["event_wick_share"] >= 0.75)
    add("event_wick_atr_ge_0p85", "geometry_wick", "shadow size >= 0.85 ATR", labels["event_wick_atr"] >= 0.85)
    add("event_wick_atr_ge_1p20", "geometry_wick", "shadow size >= 1.20 ATR", labels["event_wick_atr"] >= 1.20)

    # Volume and volatility magnitude.
    add("volume_ratio_ge_2p5", "volume", "event volume >= 2.5x rolling median", labels["volume_ratio"] >= 2.5)
    add("volume_ratio_ge_3p0", "volume", "event volume >= 3.0x rolling median", labels["volume_ratio"] >= 3.0)
    add("volume_ratio_ge_5p0", "volume", "event volume >= 5.0x rolling median", labels["volume_ratio"] >= 5.0)
    add("range_pct_ge_0p20", "event_range", "event high-low range >= 0.20%", labels["range_pct"] >= 0.0020)
    add("range_pct_ge_0p35", "event_range", "event high-low range >= 0.35%", labels["range_pct"] >= 0.0035)

    # Prior move context, normalized so positive means the move before event went against the proposed entry direction.
    add("prior_move_against_30_ge_0p20", "prior_move", "30m move against entry >= 0.20%", labels["prior_move_against_30"] >= 0.0020)
    add("prior_move_against_30_ge_0p50", "prior_move", "30m move against entry >= 0.50%", labels["prior_move_against_30"] >= 0.0050)
    add("prior_move_against_120_ge_0p50", "prior_move", "120m move against entry >= 0.50%", labels["prior_move_against_120"] >= 0.0050)
    add("prior_move_against_120_ge_1p00", "prior_move", "120m move against entry >= 1.00%", labels["prior_move_against_120"] >= 0.0100)
    add("prior_move_against_720_ge_1p50", "prior_move", "720m move against entry >= 1.50%", labels["prior_move_against_720"] >= 0.0150)

    # Trend and volatility regimes. These are categorical one-feature filters, not combinations.
    add("trend_uptrend", "trend_regime", "event occurs during uptrend regime", labels["trend_regime"].astype(str) == "uptrend")
    add("trend_downtrend", "trend_regime", "event occurs during downtrend regime", labels["trend_regime"].astype(str) == "downtrend")
    add("trend_range_or_transition", "trend_regime", "event occurs during range_or_transition regime", labels["trend_regime"].astype(str) == "range_or_transition")
    add("direction_with_trend", "trend_regime", "entry direction agrees with trend regime", labels["direction_with_trend"].astype(bool))
    add("direction_countertrend_exhaustion", "trend_regime", "entry direction fades a prior trend regime", labels["direction_countertrend_exhaustion"].astype(bool))
    for regime in ["very_low_vol", "low_mid_vol", "mid_high_vol", "extreme_vol"]:
        add(f"vol_regime_{regime}", "vol_regime", f"ATR volatility regime = {regime}", labels["vol_regime"].astype(str) == regime)
    for sess in ["asia", "eu_london", "us"]:
        add(f"session_{sess}", "session", f"event session = {sess}", labels["session"].astype(str) == sess)

    # Flow features when trade bar provides delta/taker columns. Missing values naturally produce false masks.
    delta_avail = labels["signed_delta_against_entry"].notna().mean() > 0.20
    taker_avail = labels["taker_against_entry"].notna().mean() > 0.20
    if delta_avail:
        add("delta_against_entry_ge_0p10", "flow_delta", "signed delta against entry >= 0.10", labels["signed_delta_against_entry"] >= 0.10)
        add("delta_against_entry_ge_0p20", "flow_delta", "signed delta against entry >= 0.20", labels["signed_delta_against_entry"] >= 0.20)
        add("delta_with_entry_ge_0p10", "flow_delta", "signed delta with entry >= 0.10; diagnostic opposite", labels["signed_delta_against_entry"] <= -0.10)
    if taker_avail:
        add("taker_against_entry_ge_0p55", "flow_taker", "taker flow against entry >= 55%", labels["taker_against_entry"] >= 0.55)
        add("taker_against_entry_ge_0p65", "flow_taker", "taker flow against entry >= 65%", labels["taker_against_entry"] >= 0.65)
        add("taker_with_entry_ge_0p55", "flow_taker", "taker flow with entry >= 55%; diagnostic opposite", labels["taker_against_entry"] <= 0.45)

    return specs


def summarize_return(ret: pd.Series) -> dict[str, float | int]:
    r = pd.to_numeric(ret, errors="coerce").dropna()
    if r.empty:
        return {
            "trades": 0,
            "mean_net": np.nan,
            "median_net": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "total_return": np.nan,
            "equity_end": np.nan,
            "max_drawdown": np.nan,
            "return_over_drawdown": np.nan,
            "avg_win": np.nan,
            "avg_loss": np.nan,
            "payoff_ratio": np.nan,
            "p10_net": np.nan,
            "p90_net": np.nan,
            "top5_winner_share": np.nan,
            "max_consecutive_losses": 0,
        }
    eq = _equity_stats(r)
    wins = r[r > 0]
    losses = r[r <= 0]
    avg_win = float(wins.mean()) if len(wins) else np.nan
    avg_loss = float(losses.mean()) if len(losses) else np.nan
    return {
        "trades": int(len(r)),
        "mean_net": float(r.mean()),
        "median_net": float(r.median()),
        "win_rate": float((r > 0).mean()),
        "profit_factor": _profit_factor(r),
        **eq,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "payoff_ratio": float(avg_win / abs(avg_loss)) if np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss < 0 else np.nan,
        "p10_net": float(r.quantile(0.10)),
        "p90_net": float(r.quantile(0.90)),
        "top5_winner_share": _top5_winner_share(r),
        "max_consecutive_losses": _max_consecutive_losses(r),
    }


def _smooth_edge_score(row: pd.Series) -> float:
    trades = float(row.get("trades", 0) or 0)
    wr = float(row.get("win_rate", np.nan))
    pf = float(row.get("profit_factor", np.nan))
    mean_net = float(row.get("mean_net", np.nan))
    max_dd = float(row.get("max_drawdown", np.nan))
    pos_years = float(row.get("positive_years", np.nan))
    if not np.isfinite(wr):
        return float("nan")
    count_score = min(math.log1p(max(trades, 0.0)) / math.log1p(5000.0), 1.0)
    win_score = min(max((wr - 0.50) / 0.15, 0.0), 1.0)
    pf_score = min(max((pf - 0.90) / 0.60, 0.0), 1.0) if np.isfinite(pf) else 0.0
    mean_score = min(max((mean_net + 0.0005) / 0.0015, 0.0), 1.0) if np.isfinite(mean_net) else 0.0
    dd_score = min(max(1.0 - abs(max_dd) / 0.35, 0.0), 1.0) if np.isfinite(max_dd) else 0.0
    year_score = min(max(pos_years / 4.0, 0.0), 1.0) if np.isfinite(pos_years) else 0.0
    return float(0.30 * count_score + 0.30 * win_score + 0.15 * pf_score + 0.10 * mean_score + 0.10 * dd_score + 0.05 * year_score)


def _baseline_key_cols() -> list[str]:
    return ["event_name", "direction", "strength", "entry_delay_bars", "horizon_bars"]


def build_event_counts(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for keys, part in events.groupby(["event_name", "direction", "strength"], dropna=False):
        rows.append({"event_name": keys[0], "direction": keys[1], "strength": keys[2], "events": int(len(part))})
    return pd.DataFrame(rows).sort_values(["direction", "strength", "event_name"]).reset_index(drop=True)


def build_feature_coverage(events: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "close_pos",
        "favorable_close_pos",
        "volume_ratio",
        "event_wick_share",
        "event_wick_atr",
        "range_pct",
        "atr_pct",
        "ret_30",
        "ret_120",
        "ret_720",
        "prior_move_against_30",
        "prior_move_against_120",
        "prior_move_against_720",
        "delta_ratio",
        "signed_delta_against_entry",
        "taker_buy_ratio",
        "taker_against_entry",
    ]
    rows = []
    for col in cols:
        if col not in events.columns:
            continue
        s = pd.to_numeric(events[col], errors="coerce")
        rows.append(
            {
                "feature": col,
                "non_null": int(s.notna().sum()),
                "coverage": float(s.notna().mean()) if len(s) else np.nan,
                "mean": float(s.mean()) if s.notna().any() else np.nan,
                "p25": float(s.quantile(0.25)) if s.notna().any() else np.nan,
                "median": float(s.quantile(0.50)) if s.notna().any() else np.nan,
                "p75": float(s.quantile(0.75)) if s.notna().any() else np.nan,
                "p90": float(s.quantile(0.90)) if s.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_baseline_summary(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()
    rows = []
    for keys, part in labels.groupby(_baseline_key_cols(), dropna=False, sort=False):
        row = dict(zip(_baseline_key_cols(), keys))
        row.update(summarize_return(part["net_return"]))
        row["events_per_month"] = _events_per_month(part["entry_time"])
        row["max_days_without_trade"] = _max_days_without_event(part["entry_time"])
        row["avg_mfe"] = float(pd.to_numeric(part["mfe"], errors="coerce").mean())
        row["avg_mae"] = float(pd.to_numeric(part["mae"], errors="coerce").mean())
        row["mfe_capture_median"] = float(pd.to_numeric(part["mfe_capture_ratio"], errors="coerce").median())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["direction", "strength", "entry_delay_bars", "horizon_bars"]).reset_index(drop=True)


def build_filter_probe_summary(labels: pd.DataFrame, specs: list[FilterSpec]) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = _baseline_key_cols()
    progress = ProgressReporter(label="[filters] summarize", total=len(specs), every=1)
    for i, spec in enumerate(specs, start=1):
        m = spec.mask.reindex(labels.index).fillna(False).to_numpy(dtype=bool)
        selected = labels.loc[m].copy()
        if selected.empty:
            progress.update(i)
            continue
        for keys, part in selected.groupby(group_cols, dropna=False, sort=False):
            row = dict(zip(group_cols, keys))
            row.update(
                {
                    "filter_name": spec.name,
                    "filter_family": spec.family,
                    "filter_description": spec.description,
                }
            )
            row.update(summarize_return(part["net_return"]))
            row["events_per_month"] = _events_per_month(part["entry_time"])
            row["max_days_without_trade"] = _max_days_without_event(part["entry_time"])
            row["avg_mfe"] = float(pd.to_numeric(part["mfe"], errors="coerce").mean())
            row["avg_mae"] = float(pd.to_numeric(part["mae"], errors="coerce").mean())
            row["mfe_capture_median"] = float(pd.to_numeric(part["mfe_capture_ratio"], errors="coerce").median())
            rows.append(row)
        progress.update(i)
    progress.close()
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["filter_family", "filter_name", "direction", "strength", "entry_delay_bars", "horizon_bars"]).reset_index(drop=True)


def build_filter_vs_baseline(summary: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or baseline.empty:
        return pd.DataFrame()
    key = _baseline_key_cols()
    base_cols = key + ["trades", "mean_net", "win_rate", "profit_factor", "total_return", "max_drawdown", "events_per_month"]
    b = baseline[base_cols].rename(
        columns={
            "trades": "baseline_trades",
            "mean_net": "baseline_mean_net",
            "win_rate": "baseline_win_rate",
            "profit_factor": "baseline_profit_factor",
            "total_return": "baseline_total_return",
            "max_drawdown": "baseline_max_drawdown",
            "events_per_month": "baseline_events_per_month",
        }
    )
    out = summary.merge(b, on=key, how="left")
    out["selection_rate"] = out["trades"] / out["baseline_trades"].replace(0, np.nan)
    out["mean_net_lift"] = out["mean_net"] - out["baseline_mean_net"]
    out["win_rate_lift"] = out["win_rate"] - out["baseline_win_rate"]
    out["pf_lift"] = out["profit_factor"] - out["baseline_profit_factor"]
    out["frequency_retention"] = out["events_per_month"] / out["baseline_events_per_month"].replace(0, np.nan)
    return out


def build_filter_yearly(labels: pd.DataFrame, specs: list[FilterSpec]) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    progress = ProgressReporter(label="[filters] yearly", total=len(specs), every=1)
    for i, spec in enumerate(specs, start=1):
        m = spec.mask.reindex(labels.index).fillna(False).to_numpy(dtype=bool)
        selected = labels.loc[m].copy()
        if selected.empty:
            progress.update(i)
            continue
        selected["year"] = pd.to_datetime(selected["entry_time"]).dt.year
        for keys, part in selected.groupby(["event_name", "direction", "strength", "entry_delay_bars", "horizon_bars", "year"], dropna=False, sort=False):
            row = dict(zip(["event_name", "direction", "strength", "entry_delay_bars", "horizon_bars", "year"], keys))
            row.update({"filter_name": spec.name, "filter_family": spec.family})
            row.update(summarize_return(part["net_return"]))
            rows.append(row)
        progress.update(i)
    progress.close()
    return pd.DataFrame(rows)


def build_candidate_tables(vs: pd.DataFrame, yearly: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if vs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    y = yearly.copy()
    if not y.empty:
        y_pos = (
            y.assign(year_positive=y["mean_net"] > 0)
            .groupby(["filter_name", "event_name", "direction", "strength", "entry_delay_bars", "horizon_bars"], dropna=False)["year_positive"]
            .sum()
            .reset_index()
            .rename(columns={"year_positive": "positive_years"})
        )
    else:
        y_pos = pd.DataFrame(columns=["filter_name", "event_name", "direction", "strength", "entry_delay_bars", "horizon_bars", "positive_years"])
    out = vs.merge(y_pos, on=["filter_name", "event_name", "direction", "strength", "entry_delay_bars", "horizon_bars"], how="left")
    out["positive_years"] = out["positive_years"].fillna(0).astype(int)
    out["smooth_edge_score"] = out.apply(_smooth_edge_score, axis=1)
    out["candidate_reason"] = np.select(
        [
            (out["trades"] >= int(args.min_candidate_trades))
            & (out["win_rate"] >= float(args.min_candidate_win_rate))
            & (out["mean_net"] > 0)
            & (out["profit_factor"] >= 1.05)
            & (out["positive_years"] >= 3),
            (out["trades"] >= int(args.min_candidate_trades))
            & (out["win_rate"] >= float(args.min_candidate_win_rate))
            & (out["mean_net"] >= -0.0005)
            & (out["win_rate_lift"] > 0)
            & (out["mean_net_lift"] > 0),
        ],
        ["positive_high_win_filter", "high_win_near_breakeven_diagnostic"],
        default="reject_or_diagnostic",
    )
    candidates = out.loc[out["candidate_reason"] != "reject_or_diagnostic"].copy()
    candidates = candidates.sort_values(
        ["candidate_reason", "smooth_edge_score", "trades", "win_rate", "mean_net"],
        ascending=[True, False, False, False, False],
    ).reset_index(drop=True)
    leaderboard = out.sort_values(["smooth_edge_score", "trades", "win_rate", "mean_net"], ascending=[False, False, False, False]).reset_index(drop=True)
    rejected = out.loc[out["candidate_reason"] == "reject_or_diagnostic"].copy()
    rejected = rejected.sort_values(["smooth_edge_score", "trades"], ascending=[False, False]).reset_index(drop=True)
    return leaderboard, candidates, rejected


def build_causal_audit_summary(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()
    rows = []
    for keys, part in labels.groupby(["event_name", "entry_delay_bars", "horizon_bars"], dropna=False):
        rows.append(
            {
                "event_name": keys[0],
                "entry_delay_bars": keys[1],
                "horizon_bars": keys[2],
                "rows": int(len(part)),
                "lookahead_flags": int(part["lookahead_flag"].fillna(False).sum()),
                "context_available_time_flags": int(part["context_available_time_flag"].fillna(False).sum()),
                "entry_not_expected_time_flags": int(part["entry_not_expected_time_flag"].fillna(False).sum()),
                "min_event_time": str(pd.to_datetime(part["event_bar_time"]).min()),
                "max_event_time": str(pd.to_datetime(part["event_bar_time"]).max()),
            }
        )
    return pd.DataFrame(rows)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path} rows={len(df):,} cols={len(df.columns)}", flush=True)


def _sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty or len(df) <= n:
        return df.copy()
    return df.sample(n=int(n), random_state=42).sort_values([c for c in ["entry_time", "event_name"] if c in df.columns]).reset_index(drop=True)


def _pct_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pct_like = [
        "mean_net",
        "median_net",
        "win_rate",
        "profit_factor",
        "total_return",
        "max_drawdown",
        "return_over_drawdown",
        "avg_win",
        "avg_loss",
        "payoff_ratio",
        "p10_net",
        "p90_net",
        "top5_winner_share",
        "baseline_mean_net",
        "baseline_win_rate",
        "baseline_profit_factor",
        "baseline_total_return",
        "baseline_max_drawdown",
        "selection_rate",
        "mean_net_lift",
        "win_rate_lift",
        "pf_lift",
        "frequency_retention",
        "avg_mfe",
        "avg_mae",
        "mfe_capture_median",
        "smooth_edge_score",
    ]
    for col in pct_like:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def write_reports(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    events: pd.DataFrame,
    labels: pd.DataFrame,
    specs: list[FilterSpec],
) -> None:
    print("[report] building filter probe reports", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script": SCRIPT_NAME,
        "version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "round_trip_cost_pct": args.round_trip_cost_pct,
        "horizons": args.horizons,
        "entry_delay_bars_list": args.entry_delay_bars_list,
        "research_policy": "one causal event-bar filter at a time; no combined optimizer; high-trade/high-win scoring",
        "notes": [
            "Mean return is not the sole ranking key. This report prioritizes trade count, win rate, PF, drawdown, and year robustness for smoother portfolio overlay candidates.",
            "Full event x horizon labels are not written by default to keep review packs uploadable.",
            "Footprint/range-bar features are intentionally excluded from this first filter probe; add them only after 1m trade-bar dimensions identify a promising family.",
        ],
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    event_counts = build_event_counts(events)
    feature_coverage = build_feature_coverage(events)
    baseline = build_baseline_summary(labels)
    summary = build_filter_probe_summary(labels, specs)
    vs = build_filter_vs_baseline(summary, baseline)
    yearly = build_filter_yearly(labels, specs)
    leaderboard, candidates, rejected = build_candidate_tables(vs, yearly, args)
    causal = build_causal_audit_summary(labels)

    _write_csv(event_counts, out_dir / "01_event_counts.csv")
    _write_csv(_pct_columns(feature_coverage), out_dir / "02_feature_coverage.csv")
    _write_csv(_pct_columns(baseline), out_dir / "03_raw_baseline_by_horizon.csv")
    _write_csv(_pct_columns(summary), out_dir / "04_filter_probe_summary.csv")
    _write_csv(_pct_columns(vs), out_dir / "05_filter_vs_baseline.csv")
    _write_csv(_pct_columns(yearly), out_dir / "06_filter_yearly.csv")
    _write_csv(_pct_columns(leaderboard.head(5000)), out_dir / "07_smooth_high_win_leaderboard.csv")
    _write_csv(_pct_columns(candidates), out_dir / "08_research_continue_filters.csv")
    _write_csv(_pct_columns(rejected.head(5000)), out_dir / "09_rejected_or_diagnostic_filters.csv")
    _write_csv(causal, out_dir / "10_causal_audit_summary.csv")

    sample_cols = [
        "event_id",
        "event_name",
        "direction",
        "strength",
        "event_bar_time",
        "entry_delay_bars",
        "horizon_bars",
        "entry_time",
        "net_return",
        "mfe",
        "mae",
        "favorable_close_pos",
        "volume_ratio",
        "event_wick_share",
        "event_wick_atr",
        "prior_move_against_30",
        "prior_move_against_120",
        "signed_delta_against_entry",
        "taker_against_entry",
        "session",
        "vol_regime",
        "trend_regime",
    ]
    sample_cols = [c for c in sample_cols if c in labels.columns]
    _write_csv(_sample(labels[sample_cols], int(args.event_sample_size)), out_dir / "11_label_sample.csv")

    if not candidates.empty:
        top = candidates.head(20)
        cand_frames = []
        for row in top.itertuples(index=False):
            subset = labels[
                (labels["event_name"] == getattr(row, "event_name"))
                & (labels["entry_delay_bars"] == getattr(row, "entry_delay_bars"))
                & (labels["horizon_bars"] == getattr(row, "horizon_bars"))
            ]
            spec = next((s for s in specs if s.name == getattr(row, "filter_name")), None)
            if spec is not None:
                subset = subset.loc[spec.mask.reindex(subset.index).fillna(False).to_numpy(dtype=bool)]
            if not subset.empty:
                cand_frames.append(_sample(subset[sample_cols].assign(filter_name=getattr(row, "filter_name")), max(50, int(args.candidate_sample_size) // max(len(top), 1))))
        cand_sample = pd.concat(cand_frames, ignore_index=True) if cand_frames else pd.DataFrame()
    else:
        cand_sample = pd.DataFrame()
    _write_csv(cand_sample, out_dir / "12_candidate_trade_sample.csv")

    if bool(args.write_slim_path_labels):
        slim_cols = sample_cols
        _write_csv(labels[slim_cols], out_dir / "13_slim_path_labels.csv")

    readme = f"""# {TITLE}

This research tests **one causal event-bar filter at a time** over raw 1m volume-shadow events.

Important reading order:
1. `10_causal_audit_summary.csv` — no lookahead/context flags; entry gap flags are data-gap diagnostics.
2. `03_raw_baseline_by_horizon.csv` — raw lower/upper shadow baseline by horizon.
3. `05_filter_vs_baseline.csv` — each filter's lift versus its own raw baseline.
4. `07_smooth_high_win_leaderboard.csv` — sorted for high trade count + win rate + robustness, not just mean return.
5. `08_research_continue_filters.csv` — candidates worth a follow-up candidate backtest.
6. `12_candidate_trade_sample.csv` — small sample only; no huge full trades in review pack.

Design constraints:
- No combined filters are optimized here.
- No footprint/range-bar features are mixed in yet.
- Mean return is secondary; filters should be judged by trades, win rate, PF, drawdown, positive years, frequency, and cost-adjusted mean together.
"""
    (out_dir / "README_RESEARCH.md").write_text(readme, encoding="utf-8")

    review_prompt = f"""You are reviewing {TITLE}.

Goal: find high-trade-count, high-win-rate, smoother ETH shadow-event edges. Do NOT rank by mean_net alone.

Review tasks:
1. Confirm `10_causal_audit_summary.csv` has no lookahead/context leakage.
2. Use `03_raw_baseline_by_horizon.csv` as the raw baseline.
3. In `05_filter_vs_baseline.csv`, identify which single-feature filters improve win_rate, PF, mean_net, and frequency retention.
4. In `07_smooth_high_win_leaderboard.csv`, focus on rows with enough trades and good win rate, but reject rows with negative mean_net that cannot survive fees unless they clearly indicate a diagnostic direction.
5. In `08_research_continue_filters.csv`, decide which filter families should be tested next as candidate backtests.
6. Do not recommend combining filters yet unless multiple single-filter families independently improve.

User priority: smooth portfolio overlay candidate, high trading frequency, high win rate, low drawdown. Mean return matters, but it is not the first filter.
"""
    (out_dir / "GPT_REVIEW_PROMPT.md").write_text(review_prompt, encoding="utf-8")

    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    print(f"[run] {SCRIPT_NAME} v{SCRIPT_VERSION}", flush=True)
    print(f"[args] out_dir={out_dir}", flush=True)
    print("[scope] one-filter-at-a-time raw shadow probe; high-trade/high-win scoring", flush=True)

    horizons = _parse_int_csv(args.horizons)
    entry_delays = _parse_int_csv(args.entry_delay_bars_list)
    th = _thresholds_from_args(args)

    bars = load_bars(args)
    features = build_features(bars, th)
    mask = _research_window_mask(features.index, args.start_date, args.end_date)
    research_features = features.loc[mask].copy()
    events = build_raw_shadow_events(research_features)
    if events.empty:
        print("[done] no events found", flush=True)
        write_reports(out_dir=out_dir, args=args, events=events, labels=pd.DataFrame(), specs=[])
        return 0

    labels = build_path_labels(
        research_features,
        events,
        horizons=horizons,
        entry_delays=entry_delays,
        round_trip_cost_pct=float(args.round_trip_cost_pct),
    )
    specs = build_filter_specs(labels) if not labels.empty else []
    print(f"[filters] specs={len(specs):,}", flush=True)
    write_reports(out_dir=out_dir, args=args, events=events, labels=labels, specs=specs)
    print("[done] raw volume-shadow filter probe complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
