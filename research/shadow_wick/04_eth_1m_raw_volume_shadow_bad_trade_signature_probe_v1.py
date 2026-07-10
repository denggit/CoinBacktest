#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH 1m raw volume-shadow bad-trade signature probe V1.

Research-only bad-path signature probe. This script labels fast adverse-sweep + low-MFE paths, profiles their causal event-bar commonalities, then tests one exclusion rule at a time.

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
import warnings
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

SCRIPT_NAME = "eth_1m_raw_volume_shadow_bad_trade_signature_probe_v1"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_MF_1M_RAW_VOLUME_SHADOW_BAD_TRADE_SIGNATURE_PROBE_V1"
EDGE_ID = "RESEARCH_ONLY_ETH_MF_1M_RAW_VOLUME_SHADOW_BAD_TRADE_SIGNATURE_PROBE_V1"
TITLE = "ETH 1m Raw Volume Shadow Bad Trade Signature Probe V1"
DEFAULT_OUT_DIR = "data/reports/research/eth_1m_raw_volume_shadow_bad_trade_signature_probe_v1"
BAR_DELTA = pd.Timedelta(minutes=1)
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)


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
        description="Raw ETH 1m volume-shadow bad-trade signature probe: label fast adverse sweep + low-MFE paths, then test one causal exclusion at a time.",
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
    p.add_argument("--bad-mfe-threshold", type=float, default=0.003, help="Bad-path MFE threshold. 0.003 = 0.3 pct.")
    p.add_argument("--fast-sweep-bars", type=int, default=15, help="Fast adverse sweep window after entry, in bars.")
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

    delta = pd.to_numeric(df["delta_notional"], errors="coerce") if "delta_notional" in df.columns else pd.Series(np.nan, index=df.index)
    notional = pd.to_numeric(df["notional"], errors="coerce").abs() if "notional" in df.columns else pd.Series(np.nan, index=df.index)
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
    bad_mfe_threshold: float,
    fast_sweep_bars: int,
) -> pd.DataFrame:
    """Attach forward labels used only for diagnostics.

    bad_fast_sweep_low_mfe is a future-path label, not a causal filter. Follow-up
    exclusion tests may only use event-bar features that are known at signal time.
    """
    print("[labels] attaching MFE/MAE + adverse-sweep bad-path labels", flush=True)
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
        e0 = events.copy()
        e0["entry_delay_bars"] = int(delay)
        entry_pos0 = e0["signal_bar_pos"].to_numpy(dtype=int) + 1 + int(delay)
        e0["entry_bar_pos"] = entry_pos0
        valid0 = entry_pos0 + max_h < len(bars)
        e0 = e0.loc[valid0].copy()
        if e0.empty:
            progress.update(j)
            continue
        entry_pos0 = e0["entry_bar_pos"].to_numpy(dtype=int)
        side0 = e0["side"].to_numpy(dtype=int)
        entry_price0 = _safe_take(open_, entry_pos0)
        e0["entry_time"] = index[entry_pos0]
        e0["entry_price"] = entry_price0
        e0["expected_entry_time"] = e0["signal_time"] + pd.to_timedelta(int(delay), unit="m")
        e0["entry_not_expected_time_flag"] = pd.to_datetime(e0["entry_time"]) != pd.to_datetime(e0["expected_entry_time"])
        e0["lookahead_flag"] = pd.to_datetime(e0["entry_time"]) < pd.to_datetime(e0["signal_time"])
        e0["context_available_time_flag"] = pd.to_datetime(e0["signal_available_time"]) > pd.to_datetime(e0["signal_time"])

        label_frames: list[pd.DataFrame] = []
        for h in horizons:
            hh = int(h)
            entry_pos = e0["entry_bar_pos"].to_numpy(dtype=int)
            side = e0["side"].to_numpy(dtype=int)
            entry_price = e0["entry_price"].to_numpy(dtype=float)
            end_pos = entry_pos + hh
            end_close = _safe_take(close, end_pos)
            max_high = _safe_take(fwd_high[hh], entry_pos)
            min_low = _safe_take(fwd_low[hh], entry_pos)
            gross = np.where(side > 0, end_close / entry_price - 1.0, entry_price / end_close - 1.0)
            mfe = np.where(side > 0, max_high / entry_price - 1.0, entry_price / min_low - 1.0)
            mae = np.where(side > 0, min_low / entry_price - 1.0, entry_price / max_high - 1.0)

            # Adverse sweep means revisiting/breaching the event extreme in the bad direction.
            sweep_count = np.zeros(len(e0), dtype=int)
            first_sweep_bars = np.full(len(e0), np.nan, dtype=float)
            max_adverse_breach_pct = np.full(len(e0), np.nan, dtype=float)
            if len(e0):
                # h is at most a few hundred; per-horizon window matrices are bounded and much faster than per-row loops.
                win_low = np.lib.stride_tricks.sliding_window_view(low, window_shape=hh + 1)[entry_pos]
                win_high = np.lib.stride_tricks.sliding_window_view(high, window_shape=hh + 1)[entry_pos]
                event_low = e0["event_low"].to_numpy(dtype=float)
                event_high = e0["event_high"].to_numpy(dtype=float)
                long_mask = side > 0
                short_mask = side < 0
                adverse = np.zeros_like(win_low, dtype=bool)
                if long_mask.any():
                    adverse[long_mask] = win_low[long_mask] <= event_low[long_mask, None]
                    min_path_low = np.nanmin(win_low[long_mask], axis=1)
                    max_adverse_breach_pct[long_mask] = min_path_low / event_low[long_mask] - 1.0
                if short_mask.any():
                    adverse[short_mask] = win_high[short_mask] >= event_high[short_mask, None]
                    max_path_high = np.nanmax(win_high[short_mask], axis=1)
                    max_adverse_breach_pct[short_mask] = event_high[short_mask] / max_path_high - 1.0
                sweep_count = adverse.sum(axis=1).astype(int)
                has_sweep = sweep_count > 0
                if has_sweep.any():
                    first_idx = np.argmax(adverse[has_sweep], axis=1)
                    first_sweep_bars[has_sweep] = first_idx.astype(float)

            part = e0.copy()
            part["horizon_bars"] = hh
            part["exit_time"] = index[end_pos]
            part["exit_price"] = end_close
            part["gross_return"] = gross
            part["net_return"] = gross - float(round_trip_cost_pct)
            part["mfe"] = mfe
            part["mae"] = mae
            part["mfe_capture_ratio"] = np.where(mfe > 0, part["net_return"] / mfe, np.nan)
            part["adverse_sweep_count"] = sweep_count
            part["adverse_sweep_flag"] = sweep_count > 0
            part["first_adverse_sweep_bars"] = first_sweep_bars
            part["fast_adverse_sweep_flag"] = (sweep_count > 0) & (first_sweep_bars <= int(fast_sweep_bars))
            part["max_adverse_breach_pct"] = max_adverse_breach_pct
            part["bad_low_mfe_flag"] = pd.to_numeric(part["mfe"], errors="coerce") < float(bad_mfe_threshold)
            part["bad_fast_sweep_low_mfe"] = part["fast_adverse_sweep_flag"].astype(bool) & part["bad_low_mfe_flag"].astype(bool)
            part["bad_any_sweep_low_mfe"] = part["adverse_sweep_flag"].astype(bool) & part["bad_low_mfe_flag"].astype(bool)
            label_frames.append(part)
        frames.append(pd.concat(label_frames, ignore_index=True))
        progress.update(j)
    progress.close()
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out
    out["direction_label"] = np.where(out["side"].astype(int) > 0, "LONG_lower_shadow", "SHORT_upper_shadow")
    return out


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
    bad_rate = float(row.get("bad_rate", np.nan))
    if not np.isfinite(wr):
        return float("nan")
    count_score = min(math.log1p(max(trades, 0.0)) / math.log1p(5000.0), 1.0)
    win_score = min(max((wr - 0.50) / 0.15, 0.0), 1.0)
    pf_score = min(max((pf - 0.90) / 0.60, 0.0), 1.0) if np.isfinite(pf) else 0.0
    mean_score = min(max((mean_net + 0.0005) / 0.0015, 0.0), 1.0) if np.isfinite(mean_net) else 0.0
    dd_score = min(max(1.0 - abs(max_dd) / 0.35, 0.0), 1.0) if np.isfinite(max_dd) else 0.0
    year_score = min(max(pos_years / 4.0, 0.0), 1.0) if np.isfinite(pos_years) else 0.0
    bad_score = min(max(1.0 - bad_rate / 0.35, 0.0), 1.0) if np.isfinite(bad_rate) else 0.0
    return float(0.25 * count_score + 0.25 * win_score + 0.15 * pf_score + 0.10 * mean_score + 0.10 * dd_score + 0.10 * bad_score + 0.05 * year_score)


def _baseline_key_cols() -> list[str]:
    return ["event_name", "direction", "strength", "entry_delay_bars", "horizon_bars"]


def build_event_counts(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows = []
    for keys, part in events.groupby(["event_name", "direction", "strength"], dropna=False):
        rows.append({"event_name": keys[0], "direction": keys[1], "strength": keys[2], "events": int(len(part))})
    return pd.DataFrame(rows).sort_values(["direction", "strength", "event_name"]).reset_index(drop=True)


def _add_path_stats(row: dict, part: pd.DataFrame) -> dict:
    if part.empty:
        row.update({
            "bad_trades": 0,
            "bad_rate": np.nan,
            "bad_any_sweep_low_mfe_rate": np.nan,
            "sweep_rate": np.nan,
            "fast_sweep_rate": np.nan,
            "avg_mfe": np.nan,
            "median_mfe": np.nan,
            "avg_mae": np.nan,
            "median_first_sweep_bars": np.nan,
            "events_per_month": np.nan,
            "max_days_without_trade": np.nan,
        })
        return row
    row.update({
        "bad_trades": int(part["bad_fast_sweep_low_mfe"].fillna(False).sum()),
        "bad_rate": float(part["bad_fast_sweep_low_mfe"].fillna(False).mean()),
        "bad_any_sweep_low_mfe_rate": float(part["bad_any_sweep_low_mfe"].fillna(False).mean()),
        "sweep_rate": float(part["adverse_sweep_flag"].fillna(False).mean()),
        "fast_sweep_rate": float(part["fast_adverse_sweep_flag"].fillna(False).mean()),
        "avg_mfe": float(pd.to_numeric(part["mfe"], errors="coerce").mean()),
        "median_mfe": float(pd.to_numeric(part["mfe"], errors="coerce").median()),
        "avg_mae": float(pd.to_numeric(part["mae"], errors="coerce").mean()),
        "median_first_sweep_bars": float(pd.to_numeric(part["first_adverse_sweep_bars"], errors="coerce").median()),
        "events_per_month": _events_per_month(part["entry_time"]),
        "max_days_without_trade": _max_days_without_event(part["entry_time"]),
    })
    return row


def build_bad_label_summary(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()
    rows = []
    for keys, part in labels.groupby(_baseline_key_cols(), dropna=False):
        row = dict(zip(_baseline_key_cols(), keys))
        row.update(summarize_return(part["net_return"]))
        rows.append(_add_path_stats(row, part))
    return pd.DataFrame(rows).sort_values(["direction", "strength", "event_name", "entry_delay_bars", "horizon_bars"]).reset_index(drop=True)


def _bucket_series(s: pd.Series, name: str) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    if name in {"range_pct", "atr_pct"}:
        return pd.cut(x, [-np.inf, 0.0015, 0.0020, 0.0035, 0.0050, np.inf], labels=["lt_0p15", "0p15_0p20", "0p20_0p35", "0p35_0p50", "ge_0p50"]).astype("object").fillna("NA")
    if name == "volume_ratio":
        return pd.cut(x, [-np.inf, 2.5, 3.0, 5.0, 8.0, np.inf], labels=["lt_2p5", "2p5_3p0", "3p0_5p0", "5p0_8p0", "ge_8p0"]).astype("object").fillna("NA")
    if name in {"event_wick_share", "favorable_close_pos"}:
        return pd.cut(x, [-np.inf, 0.40, 0.60, 0.70, 0.80, np.inf], labels=["lt_0p40", "0p40_0p60", "0p60_0p70", "0p70_0p80", "ge_0p80"]).astype("object").fillna("NA")
    if name == "event_wick_atr":
        return pd.cut(x, [-np.inf, 0.85, 1.20, 1.80, 2.50, np.inf], labels=["lt_0p85", "0p85_1p20", "1p20_1p80", "1p80_2p50", "ge_2p50"]).astype("object").fillna("NA")
    if name.startswith("prior_move_against"):
        return pd.cut(x, [-np.inf, 0.0, 0.002, 0.005, 0.010, 0.020, np.inf], labels=["with_entry_or_flat", "0_0p20", "0p20_0p50", "0p50_1p00", "1p00_2p00", "ge_2p00"]).astype("object").fillna("NA")
    if name in {"signed_delta_against_entry", "taker_against_entry"}:
        if name == "signed_delta_against_entry":
            return pd.cut(x, [-np.inf, -0.10, 0.0, 0.10, 0.20, np.inf], labels=["with_entry_strong", "with_entry_weak", "against_0_0p10", "against_0p10_0p20", "against_ge_0p20"]).astype("object").fillna("NA")
        return pd.cut(x, [-np.inf, 0.45, 0.55, 0.65, 0.75, np.inf], labels=["with_entry_ge_0p55", "balanced", "against_0p55_0p65", "against_0p65_0p75", "against_ge_0p75"]).astype("object").fillna("NA")
    return s.astype("object").fillna("NA")


def build_bad_rate_by_feature_bucket(labels: pd.DataFrame, *, min_bucket_trades: int = 100) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()
    features = [
        "favorable_close_pos", "volume_ratio", "event_wick_share", "event_wick_atr", "range_pct", "atr_pct",
        "prior_move_against_30", "prior_move_against_120", "prior_move_against_720",
        "vol_regime", "trend_regime", "session", "signed_delta_against_entry", "taker_against_entry",
    ]
    rows = []
    group_base = ["event_name", "direction", "strength", "entry_delay_bars", "horizon_bars"]
    for feat in features:
        if feat not in labels.columns:
            continue
        bucket = _bucket_series(labels[feat], feat)
        tmp = labels.assign(feature=feat, bucket=bucket)
        for keys, part in tmp.groupby(group_base + ["feature", "bucket"], dropna=False):
            if len(part) < int(min_bucket_trades):
                continue
            row = dict(zip(group_base + ["feature", "bucket"], keys))
            row.update(summarize_return(part["net_return"]))
            rows.append(_add_path_stats(row, part))
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    return out.sort_values(["bad_rate", "trades"], ascending=[False, False]).reset_index(drop=True)


def build_exclusion_specs(labels: pd.DataFrame) -> list[FilterSpec]:
    print("[exclusions] building one-risk-feature-at-a-time exclusion specs", flush=True)
    specs: list[FilterSpec] = []

    def add(name: str, family: str, description: str, exclude_mask: pd.Series | np.ndarray) -> None:
        m = pd.Series(exclude_mask, index=labels.index).fillna(False).astype(bool)
        specs.append(FilterSpec(name=name, family=family, description=description, mask=m))

    add("exclude_fav_close_pos_lt_0p60", "geometry_close", "exclude events whose close failed to recover into favorable 60% of the event range", labels["favorable_close_pos"] < 0.60)
    add("exclude_fav_close_pos_lt_0p70", "geometry_close", "exclude events whose close failed to recover into favorable 70% of the event range", labels["favorable_close_pos"] < 0.70)
    add("exclude_volume_ratio_lt_2p5", "volume", "exclude weak volume expansion < 2.5x rolling median", labels["volume_ratio"] < 2.5)
    add("exclude_volume_ratio_lt_3p0", "volume", "exclude weak volume expansion < 3.0x rolling median", labels["volume_ratio"] < 3.0)
    add("exclude_range_pct_lt_0p20", "event_range", "exclude event high-low range < 0.20%", labels["range_pct"] < 0.0020)
    add("exclude_range_pct_lt_0p35", "event_range", "exclude event high-low range < 0.35%", labels["range_pct"] < 0.0035)
    add("exclude_wick_share_lt_0p65", "geometry_wick", "exclude shadow share < 65%", labels["event_wick_share"] < 0.65)
    add("exclude_wick_atr_lt_0p85", "geometry_wick", "exclude shadow size < 0.85 ATR", labels["event_wick_atr"] < 0.85)
    add("exclude_prior_move_against_120_lt_0p50", "prior_move", "exclude events without at least 0.50% 120m move against entry", labels["prior_move_against_120"] < 0.0050)
    add("exclude_prior_move_against_120_lt_1p00", "prior_move", "exclude events without at least 1.00% 120m move against entry", labels["prior_move_against_120"] < 0.0100)
    add("exclude_very_low_vol", "vol_regime", "exclude very_low_vol events", labels["vol_regime"].astype(str) == "very_low_vol")
    add("exclude_low_mid_vol", "vol_regime", "exclude low_mid_vol events", labels["vol_regime"].astype(str) == "low_mid_vol")
    add("exclude_not_mid_high_vol", "vol_regime", "keep only mid_high_vol; diagnostic univariate categorical regime", labels["vol_regime"].astype(str) != "mid_high_vol")
    add("exclude_not_mid_or_extreme_vol", "vol_regime", "keep only mid_high_vol or extreme_vol", ~labels["vol_regime"].astype(str).isin(["mid_high_vol", "extreme_vol"]))
    for sess in ["asia", "eu_london", "us"]:
        add(f"exclude_session_{sess}", "session", f"exclude session={sess}", labels["session"].astype(str) == sess)
    if labels["signed_delta_against_entry"].notna().mean() > 0.20:
        add("exclude_delta_against_entry_lt_0p10", "flow_delta", "exclude events without delta absorption against entry >= 0.10", labels["signed_delta_against_entry"] < 0.10)
        add("exclude_delta_against_entry_lt_0p20", "flow_delta", "exclude events without delta absorption against entry >= 0.20", labels["signed_delta_against_entry"] < 0.20)
    if labels["taker_against_entry"].notna().mean() > 0.20:
        add("exclude_taker_against_entry_lt_0p55", "flow_taker", "exclude events without taker pressure against entry >= 55%", labels["taker_against_entry"] < 0.55)
        add("exclude_taker_against_entry_lt_0p65", "flow_taker", "exclude events without taker pressure against entry >= 65%", labels["taker_against_entry"] < 0.65)
    return specs


def build_exclusion_probe_summary(labels: pd.DataFrame, specs: list[FilterSpec]) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()
    rows = []
    key_cols = _baseline_key_cols()
    for spec in specs:
        excl = spec.mask.reindex(labels.index).fillna(False).astype(bool)
        labels2 = labels.assign(_exclude=excl.to_numpy(dtype=bool))
        for keys, part0 in labels2.groupby(key_cols, dropna=False):
            kept = part0.loc[~part0["_exclude"].astype(bool)].copy()
            excluded = part0.loc[part0["_exclude"].astype(bool)].copy()
            if kept.empty:
                continue
            row = dict(zip(key_cols, keys))
            row.update({
                "exclusion_name": spec.name,
                "family": spec.family,
                "description": spec.description,
                "excluded_trades": int(len(excluded)),
                "kept_trades": int(len(kept)),
                "frequency_retention": float(len(kept) / len(part0)) if len(part0) else np.nan,
                "excluded_bad_rate": float(excluded["bad_fast_sweep_low_mfe"].mean()) if len(excluded) else np.nan,
                "excluded_mean_net": float(excluded["net_return"].mean()) if len(excluded) else np.nan,
                "excluded_win_rate": float((excluded["net_return"] > 0).mean()) if len(excluded) else np.nan,
            })
            row.update(summarize_return(kept["net_return"]))
            rows.append(_add_path_stats(row, kept))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_exclusion_vs_baseline(summary: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    base_cols = _baseline_key_cols()
    base_keep = baseline[base_cols + [
        "trades", "mean_net", "win_rate", "profit_factor", "total_return", "max_drawdown", "bad_rate", "sweep_rate", "fast_sweep_rate", "avg_mfe", "avg_mae",
    ]].rename(columns={
        "trades": "baseline_trades",
        "mean_net": "baseline_mean_net",
        "win_rate": "baseline_win_rate",
        "profit_factor": "baseline_profit_factor",
        "total_return": "baseline_total_return",
        "max_drawdown": "baseline_max_drawdown",
        "bad_rate": "baseline_bad_rate",
        "sweep_rate": "baseline_sweep_rate",
        "fast_sweep_rate": "baseline_fast_sweep_rate",
        "avg_mfe": "baseline_avg_mfe",
        "avg_mae": "baseline_avg_mae",
    })
    out = summary.merge(base_keep, on=base_cols, how="left")
    out["mean_net_lift"] = out["mean_net"] - out["baseline_mean_net"]
    out["win_rate_lift"] = out["win_rate"] - out["baseline_win_rate"]
    out["pf_lift"] = out["profit_factor"] - out["baseline_profit_factor"]
    out["bad_rate_reduction"] = out["baseline_bad_rate"] - out["bad_rate"]
    out["sweep_rate_reduction"] = out["baseline_sweep_rate"] - out["sweep_rate"]
    return out


def build_exclusion_yearly(labels: pd.DataFrame, specs: list[FilterSpec]) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()
    rows = []
    key_cols = ["exclusion_name", "event_name", "direction", "strength", "entry_delay_bars", "horizon_bars", "year"]
    base_cols = _baseline_key_cols()
    labels2 = labels.copy()
    labels2["year"] = pd.to_datetime(labels2["entry_time"]).dt.year
    for spec in specs:
        excl = spec.mask.reindex(labels2.index).fillna(False).astype(bool)
        kept_all = labels2.loc[~excl].copy()
        if kept_all.empty:
            continue
        for keys, part in kept_all.groupby(base_cols + ["year"], dropna=False):
            row = dict(zip(key_cols, (spec.name, *keys)))
            row.update(summarize_return(part["net_return"]))
            rows.append(_add_path_stats(row, part))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_candidate_tables(vs: pd.DataFrame, yearly: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if vs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if not yearly.empty:
        y = yearly.copy()
        y["year_positive"] = pd.to_numeric(y["mean_net"], errors="coerce") > 0
        y_pos = (
            y.groupby(["exclusion_name", "event_name", "direction", "strength", "entry_delay_bars", "horizon_bars"], dropna=False)["year_positive"]
            .sum()
            .reset_index()
            .rename(columns={"year_positive": "positive_years"})
        )
    else:
        y_pos = pd.DataFrame(columns=["exclusion_name", "event_name", "direction", "strength", "entry_delay_bars", "horizon_bars", "positive_years"])
    out = vs.merge(y_pos, on=["exclusion_name", "event_name", "direction", "strength", "entry_delay_bars", "horizon_bars"], how="left")
    out["positive_years"] = out["positive_years"].fillna(0).astype(int)
    out["smooth_edge_score"] = out.apply(_smooth_edge_score, axis=1)
    out["candidate_reason"] = np.select(
        [
            (out["trades"] >= int(args.min_candidate_trades))
            & (out["win_rate"] >= float(args.min_candidate_win_rate))
            & (out["mean_net"] > 0)
            & (out["profit_factor"] >= 1.05)
            & (out["bad_rate_reduction"] > 0)
            & (out["positive_years"] >= 3),
            (out["trades"] >= int(args.min_candidate_trades))
            & (out["win_rate"] >= float(args.min_candidate_win_rate))
            & (out["mean_net"] >= -0.0005)
            & (out["mean_net_lift"] > 0)
            & (out["win_rate_lift"] > 0)
            & (out["bad_rate_reduction"] > 0),
        ],
        ["positive_bad_filter_exclusion", "high_win_bad_filter_diagnostic"],
        default="reject_or_diagnostic",
    )
    candidates = out.loc[out["candidate_reason"] != "reject_or_diagnostic"].copy()
    candidates = candidates.sort_values(["candidate_reason", "smooth_edge_score", "trades", "win_rate", "mean_net"], ascending=[True, False, False, False, False]).reset_index(drop=True)
    leaderboard = out.sort_values(["smooth_edge_score", "trades", "win_rate", "mean_net"], ascending=[False, False, False, False]).reset_index(drop=True)
    rejected = out.loc[out["candidate_reason"] == "reject_or_diagnostic"].copy().sort_values(["smooth_edge_score", "trades"], ascending=[False, False]).reset_index(drop=True)
    return leaderboard, candidates, rejected


def build_causal_audit_summary(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()
    rows = []
    for keys, part in labels.groupby(["event_name", "entry_delay_bars", "horizon_bars"], dropna=False):
        rows.append({
            "event_name": keys[0],
            "entry_delay_bars": keys[1],
            "horizon_bars": keys[2],
            "rows": int(len(part)),
            "lookahead_flags": int(part["lookahead_flag"].fillna(False).sum()),
            "context_available_time_flags": int(part["context_available_time_flag"].fillna(False).sum()),
            "entry_not_expected_time_flags": int(part["entry_not_expected_time_flag"].fillna(False).sum()),
            "min_event_time": str(pd.to_datetime(part["event_bar_time"]).min()),
            "max_event_time": str(pd.to_datetime(part["event_bar_time"]).max()),
        })
    return pd.DataFrame(rows)


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path} rows={len(df):,} cols={len(df.columns)}", flush=True)


def _sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty or len(df) <= n:
        return df.copy()
    sort_cols = [c for c in ["entry_time", "event_name"] if c in df.columns]
    out = df.sample(n=int(n), random_state=42)
    return out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out.reset_index(drop=True)


def _pct_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if any(k in col for k in ["rate", "return", "net", "drawdown", "mfe", "mae", "ratio", "lift", "reduction", "share", "score", "breach"]):
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
    print("[report] building bad-trade signature reports", flush=True)
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
        "bad_mfe_threshold": args.bad_mfe_threshold,
        "fast_sweep_bars": args.fast_sweep_bars,
        "research_policy": "label bad paths with future MFE/sweep diagnostics; test only causal event-bar exclusion rules one at a time",
        "notes": [
            "bad_fast_sweep_low_mfe is never used directly as an entry filter; it is a diagnostic target label.",
            "Each exclusion rule uses one event-bar feature family at a time and is compared against the raw baseline for the same event/delay/horizon.",
            "No full trades are written by default; review pack is kept uploadable.",
        ],
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    event_counts = build_event_counts(events)
    bad_summary = build_bad_label_summary(labels)
    bucket = build_bad_rate_by_feature_bucket(labels)
    exclusion_summary = build_exclusion_probe_summary(labels, specs)
    vs = build_exclusion_vs_baseline(exclusion_summary, bad_summary)
    yearly = build_exclusion_yearly(labels, specs)
    leaderboard, candidates, rejected = build_candidate_tables(vs, yearly, args)
    causal = build_causal_audit_summary(labels)

    _write_csv(event_counts, out_dir / "01_event_counts.csv")
    _write_csv(_pct_columns(bad_summary), out_dir / "02_bad_label_summary.csv")
    _write_csv(_pct_columns(bucket.head(10000)), out_dir / "03_bad_rate_by_feature_bucket.csv")
    _write_csv(_pct_columns(exclusion_summary), out_dir / "04_exclusion_probe_summary.csv")
    _write_csv(_pct_columns(vs), out_dir / "05_exclusion_vs_baseline.csv")
    _write_csv(_pct_columns(yearly), out_dir / "06_exclusion_yearly.csv")
    _write_csv(_pct_columns(leaderboard.head(5000)), out_dir / "07_bad_filter_smooth_leaderboard.csv")
    _write_csv(_pct_columns(candidates), out_dir / "08_research_continue_exclusions.csv")
    _write_csv(_pct_columns(rejected.head(5000)), out_dir / "09_rejected_or_diagnostic_exclusions.csv")
    _write_csv(causal, out_dir / "10_causal_audit_summary.csv")

    sample_cols = [
        "event_id", "event_name", "direction", "strength", "event_bar_time", "entry_delay_bars", "horizon_bars",
        "entry_time", "net_return", "mfe", "mae", "bad_fast_sweep_low_mfe", "adverse_sweep_count",
        "first_adverse_sweep_bars", "max_adverse_breach_pct", "favorable_close_pos", "volume_ratio",
        "event_wick_share", "event_wick_atr", "range_pct", "prior_move_against_120", "signed_delta_against_entry",
        "taker_against_entry", "session", "vol_regime", "trend_regime",
    ]
    sample_cols = [c for c in sample_cols if c in labels.columns]
    bad_sample = labels.loc[labels["bad_fast_sweep_low_mfe"].fillna(False), sample_cols]
    _write_csv(_sample(bad_sample, int(args.event_sample_size)), out_dir / "11_bad_trade_sample.csv")

    if not candidates.empty:
        top = candidates.head(20)
        cand_frames = []
        for row in top.itertuples(index=False):
            subset = labels[
                (labels["event_name"] == getattr(row, "event_name"))
                & (labels["entry_delay_bars"] == getattr(row, "entry_delay_bars"))
                & (labels["horizon_bars"] == getattr(row, "horizon_bars"))
            ]
            spec = next((s for s in specs if s.name == getattr(row, "exclusion_name")), None)
            if spec is not None:
                subset = subset.loc[~spec.mask.reindex(subset.index).fillna(False).to_numpy(dtype=bool)]
            if not subset.empty:
                cand_frames.append(_sample(subset[sample_cols].assign(exclusion_name=getattr(row, "exclusion_name")), max(50, int(args.candidate_sample_size) // max(len(top), 1))))
        kept_sample = pd.concat(cand_frames, ignore_index=True) if cand_frames else pd.DataFrame()
    else:
        kept_sample = pd.DataFrame()
    _write_csv(kept_sample, out_dir / "12_kept_trade_sample.csv")

    if bool(args.write_slim_path_labels):
        _write_csv(labels[sample_cols], out_dir / "13_slim_path_labels.csv")

    readme = f"""# {TITLE}

This research follows the bad-trade-first idea:

1. Start from raw lower/upper volume-shadow events.
2. Label bad paths as `fast adverse sweep within {args.fast_sweep_bars} bars` + `MFE < {float(args.bad_mfe_threshold):.4f}`.
3. Profile which event-bar features have high bad-path rates.
4. Test one causal exclusion rule at a time to see whether excluding likely-bad trades improves win rate, PF, bad rate, and mean_net while retaining enough trades.

Important reading order:
1. `10_causal_audit_summary.csv` — no lookahead/context flags.
2. `02_bad_label_summary.csv` — how common the bad path is by event/delay/horizon.
3. `03_bad_rate_by_feature_bucket.csv` — which causal feature buckets overproduce bad trades.
4. `05_exclusion_vs_baseline.csv` — whether excluding one risk feature improves the raw baseline.
5. `07_bad_filter_smooth_leaderboard.csv` — high-trade/high-win retained populations, not mean-return-only ranking.
6. `08_research_continue_exclusions.csv` — exclusions worth follow-up.

Causal note: the bad-path label uses future MFE/sweep only for diagnosis. It must not be used directly as a live filter.
"""
    (out_dir / "README_RESEARCH.md").write_text(readme, encoding="utf-8")

    review_prompt = f"""You are reviewing {TITLE}.

Goal: evaluate the user's bad-trade-first hypothesis for raw ETH 1m volume-shadow events.

Review tasks:
1. Confirm `10_causal_audit_summary.csv` has no lookahead/context leakage.
2. Use `02_bad_label_summary.csv` to understand the target bad label: fast adverse sweep + MFE < {float(args.bad_mfe_threshold):.4f}.
3. Use `03_bad_rate_by_feature_bucket.csv` to identify common causal event-bar traits of bad trades.
4. Use `05_exclusion_vs_baseline.csv` to decide whether excluding one risk feature improves win_rate, PF, mean_net, bad_rate, and drawdown while retaining enough trades.
5. Use `08_research_continue_exclusions.csv` only if rows meet high-trade/high-win and bad-rate-reduction criteria.
6. Do not recommend combining exclusions yet unless multiple single exclusions independently improve.

User priority: high trading frequency, high win rate, smoother portfolio overlay. Mean return matters, but it is not the first ranking key.
"""
    (out_dir / "GPT_REVIEW_PROMPT.md").write_text(review_prompt, encoding="utf-8")

    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    print(f"[run] {SCRIPT_NAME} v{SCRIPT_VERSION}", flush=True)
    print(f"[args] out_dir={out_dir}", flush=True)
    print("[scope] bad-trade-first raw shadow probe; one causal exclusion at a time", flush=True)

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
        bad_mfe_threshold=float(args.bad_mfe_threshold),
        fast_sweep_bars=int(args.fast_sweep_bars),
    )
    specs = build_exclusion_specs(labels) if not labels.empty else []
    print(f"[exclusions] specs={len(specs):,}", flush=True)
    write_reports(out_dir=out_dir, args=args, events=events, labels=labels, specs=specs)
    print("[done] raw volume-shadow bad-trade signature probe complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
