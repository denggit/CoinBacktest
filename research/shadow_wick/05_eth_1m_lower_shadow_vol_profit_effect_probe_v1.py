#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH 1m lower-shadow mid/high-vol profit-effect probe V1.

Research-only probe for the strongest currently recorded raw-shadow lead:

    lower_volume_shadow_long + vol_regime in {mid_high_vol, extreme_vol}
    diagnostic horizon = 60 bars

The goal is not to immediately tighten the signal. This script keeps the
validated base universe broad, then scores one feature bucket at a time by
short-term profit effect: MFE, MAE, fast-profit rate, fast-adverse rate, win
rate, PF, bad-path rate, trade count, and yearly stability.

Causal policy
-------------
1m bars are left-labeled by bar start time. An event bar is known only after it
closes. signal_time = event_bar_start + 1 minute. entry_time is the next bar
open plus optional diagnostic delay. All feature buckets use only event-bar or
pre-event rolling context known at signal_time. Forward MFE/MAE/returns are
labels for analysis only.
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402
from src.research_common.review_pack import finalize_research_report  # noqa: E402

SCRIPT_NAME = "eth_1m_lower_shadow_vol_profit_effect_probe_v1"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_MF_1M_LOWER_SHADOW_VOL_PROFIT_EFFECT_PROBE_V1"
EDGE_ID = "RESEARCH_ONLY_ETH_MF_1M_LOWER_SHADOW_VOL_PROFIT_EFFECT_PROBE_V1"
TITLE = "ETH 1m Lower Shadow Vol-Regime Profit Effect Probe V1"
DEFAULT_OUT_DIR = "data/reports/research/eth_1m_lower_shadow_vol_profit_effect_probe_v1"
BAR_DELTA = pd.Timedelta(minutes=1)
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)


@dataclass(frozen=True)
class ShadowThresholds:
    wick_share_min: float = 0.50
    wick_atr_min: float = 0.55
    volume_ratio_min: float = 2.0


@dataclass(frozen=True)
class BucketSpec:
    feature_name: str
    bucket_name: str
    description: str
    mask: pd.Series


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lower-shadow mid/high-vol short-term profit-effect probe: MFE/MAE scoring by single feature bucket.",
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
    p.add_argument("--horizon-bars", type=int, default=60)
    p.add_argument("--entry-delay-bars-list", default="0,1,2")
    p.add_argument("--base-vol-regimes", default="mid_high_vol,extreme_vol")
    p.add_argument("--min-candidate-trades", type=int, default=200)
    p.add_argument("--min-candidate-win-rate", type=float, default=0.55)
    p.add_argument("--bad-mfe-threshold", type=float, default=0.003, help="Bad-path MFE threshold. 0.003 = 0.3%%.")
    p.add_argument("--fast-profit-threshold", type=float, default=0.003, help="Fast/profit MFE threshold. 0.003 = 0.3%%.")
    p.add_argument("--strong-profit-threshold", type=float, default=0.005, help="Strong MFE threshold. 0.005 = 0.5%%.")
    p.add_argument("--mae-control-threshold", type=float, default=-0.005, help="Controlled MAE threshold. -0.005 = -0.5%%.")
    p.add_argument("--tradable-mae-threshold", type=float, default=-0.007, help="Tradable MAE threshold. -0.007 = -0.7%%.")
    p.add_argument("--fast-window-bars", type=int, default=15)
    p.add_argument("--event-sample-size", type=int, default=5000)
    p.add_argument("--candidate-sample-size", type=int, default=2000)
    p.add_argument("--write-slim-labels", action="store_true", help="Write slim per-trade labels; off by default to keep review pack small.")

    # Fixed raw lower-shadow vocabulary. Keep defaults for comparable reports.
    p.add_argument("--wick-share-min", type=float, default=0.50)
    p.add_argument("--wick-atr-min", type=float, default=0.55)
    p.add_argument("--volume-ratio-min", type=float, default=2.0)
    return p.parse_args(argv)


def _parse_int_csv(raw: str) -> tuple[int, ...]:
    vals = []
    for part in str(raw).split(","):
        text = part.strip()
        if text:
            vals.append(int(text))
    out = tuple(dict.fromkeys(vals))
    if not out:
        raise ValueError("integer csv must not be empty")
    return out


def _parse_str_csv(raw: str) -> tuple[str, ...]:
    vals = [x.strip() for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError("string csv must not be empty")
    return tuple(dict.fromkeys(vals))


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
    return out


def build_lower_vol_events(features: pd.DataFrame, base_vol_regimes: tuple[str, ...]) -> pd.DataFrame:
    print("[events] building lower-volume-shadow LONG events inside mid/high vol universe", flush=True)
    f = features
    base_mask = f["lower_volume_shadow"].fillna(False) & f["vol_regime"].astype(str).isin(base_vol_regimes)
    idx = f.index[base_mask.to_numpy(dtype=bool)]
    if len(idx) == 0:
        return pd.DataFrame()
    pos_map = pd.Series(np.arange(len(f), dtype=int), index=f.index)
    event_pos = pos_map.reindex(idx).to_numpy(dtype=int)
    valid = event_pos + 1 < len(f)
    idx = idx[valid]
    event_pos = event_pos[valid]
    ff = f.loc[idx]
    out = pd.DataFrame(
        {
            "event_id": np.arange(len(ff), dtype=int),
            "event_name": "lower_volume_shadow_long_mid_high_or_extreme_vol",
            "family": "lower_shadow_vol_profit_effect",
            "direction": "LONG",
            "side": 1,
            "strength": "standard",
            "event_bar_time": idx,
            "event_bar_pos": event_pos,
            "signal_bar_time": idx,
            "signal_bar_pos": event_pos,
            "signal_time": idx + BAR_DELTA,
            "signal_available_time": idx + BAR_DELTA,
            "session": ff["session"].to_numpy(),
            "vol_regime": ff["vol_regime"].astype(str).to_numpy(),
            "trend_regime": ff["trend_regime"].astype(str).to_numpy(),
            "close_pos": ff["close_pos"].to_numpy(dtype=float),
            "favorable_close_pos": ff["close_pos"].to_numpy(dtype=float),
            "volume_ratio": ff["volume_ratio"].to_numpy(dtype=float),
            "event_wick_share": ff["lower_wick_share"].to_numpy(dtype=float),
            "event_wick_atr": ff["lower_wick_atr"].to_numpy(dtype=float),
            "range_pct": ff["range_pct"].to_numpy(dtype=float),
            "atr_pct": ff["atr_pct"].to_numpy(dtype=float),
            "ret_30": ff["ret_30"].to_numpy(dtype=float),
            "ret_120": ff["ret_120"].to_numpy(dtype=float),
            "ret_720": ff["ret_720"].to_numpy(dtype=float),
            "prior_move_against_30": -ff["ret_30"].to_numpy(dtype=float),
            "prior_move_against_120": -ff["ret_120"].to_numpy(dtype=float),
            "prior_move_against_720": -ff["ret_720"].to_numpy(dtype=float),
            "delta_ratio": ff["delta_ratio"].to_numpy(dtype=float),
            "taker_buy_ratio": ff["taker_buy_ratio"].to_numpy(dtype=float),
            "signed_delta_against_entry": -ff["delta_ratio"].to_numpy(dtype=float),
            "taker_against_entry": 1.0 - ff["taker_buy_ratio"].to_numpy(dtype=float),
            "event_low": ff["low"].to_numpy(dtype=float),
            "event_high": ff["high"].to_numpy(dtype=float),
            "event_mid": ((ff["high"] + ff["low"]) / 2.0).to_numpy(dtype=float),
            "event_open": ff["open"].to_numpy(dtype=float),
            "event_close": ff["close"].to_numpy(dtype=float),
        }
    )
    print(f"[events] total={len(out):,} vol_regimes={','.join(base_vol_regimes)}", flush=True)
    return out.sort_values(["event_bar_time", "event_id"]).reset_index(drop=True)


def _safe_take(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
    out = np.full(len(pos), np.nan, dtype=float)
    valid = (pos >= 0) & (pos < len(arr))
    if valid.any():
        out[valid] = arr[pos[valid]]
    return out


def _first_true_offset(mask: np.ndarray) -> np.ndarray:
    out = np.full(mask.shape[0], np.nan, dtype=float)
    has = mask.any(axis=1)
    if has.any():
        out[has] = np.argmax(mask[has], axis=1).astype(float)
    return out


def build_profit_effect_labels(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizon_bars: int,
    entry_delays: tuple[int, ...],
    round_trip_cost_pct: float,
    bad_mfe_threshold: float,
    fast_profit_threshold: float,
    strong_profit_threshold: float,
    mae_control_threshold: float,
    tradable_mae_threshold: float,
    fast_window_bars: int,
) -> pd.DataFrame:
    print("[labels] attaching 60bar MFE/MAE profit-effect labels", flush=True)
    if events.empty:
        return pd.DataFrame()
    h = int(horizon_bars)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    open_ = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    index = bars.index
    if len(bars) <= h + 2:
        return pd.DataFrame()

    # Sliding windows are bounded by h=60 by default and keep the research fast.
    win_high_all = np.lib.stride_tricks.sliding_window_view(high, window_shape=h + 1)
    win_low_all = np.lib.stride_tricks.sliding_window_view(low, window_shape=h + 1)
    win_close_all = np.lib.stride_tricks.sliding_window_view(close, window_shape=h + 1)

    frames: list[pd.DataFrame] = []
    progress = ProgressReporter(label="[labels] entry delays", total=len(entry_delays), every=1)
    for j, delay in enumerate(entry_delays, start=1):
        part = events.copy()
        part["entry_delay_bars"] = int(delay)
        entry_pos = part["signal_bar_pos"].to_numpy(dtype=int) + 1 + int(delay)
        valid = entry_pos + h < len(bars)
        part = part.loc[valid].copy()
        if part.empty:
            progress.update(j)
            continue
        entry_pos = entry_pos[valid]
        entry_price = _safe_take(open_, entry_pos)
        end_pos = entry_pos + h
        exit_price = _safe_take(close, end_pos)
        win_high = win_high_all[entry_pos]
        win_low = win_low_all[entry_pos]
        win_close = win_close_all[entry_pos]

        gross = exit_price / entry_price - 1.0
        mfe_path = win_high / entry_price[:, None] - 1.0
        mae_path = win_low / entry_price[:, None] - 1.0
        mfe = np.nanmax(mfe_path, axis=1)
        mae = np.nanmin(mae_path, axis=1)
        time_to_mfe = np.nanargmax(np.where(np.isfinite(mfe_path), mfe_path, -np.inf), axis=1).astype(float)
        time_to_mae = np.nanargmin(np.where(np.isfinite(mae_path), mae_path, np.inf), axis=1).astype(float)

        # Price-path diagnostics for long side.
        event_low = part["event_low"].to_numpy(dtype=float)
        event_mid = part["event_mid"].to_numpy(dtype=float)
        event_high = part["event_high"].to_numpy(dtype=float)
        adverse_sweep = win_low <= event_low[:, None]
        reclaim_mid = win_close >= event_mid[:, None]
        reclaim_high = win_high >= event_high[:, None]
        first_sweep_bars = _first_true_offset(adverse_sweep)
        first_reclaim_mid_bars = _first_true_offset(reclaim_mid)
        first_reclaim_high_bars = _first_true_offset(reclaim_high)
        sweep_count = adverse_sweep.sum(axis=1).astype(int)
        fast_adverse_sweep = (sweep_count > 0) & (first_sweep_bars <= int(fast_window_bars))
        fast_profit = time_to_mfe <= int(fast_window_bars)
        max_adverse_breach_pct = np.nanmin(win_low, axis=1) / event_low - 1.0

        part["horizon_bars"] = h
        part["entry_bar_pos"] = entry_pos
        part["entry_time"] = index[entry_pos]
        part["entry_price"] = entry_price
        part["expected_entry_time"] = part["signal_time"] + pd.to_timedelta(int(delay), unit="m")
        part["entry_not_expected_time_flag"] = pd.to_datetime(part["entry_time"]) != pd.to_datetime(part["expected_entry_time"])
        part["lookahead_flag"] = pd.to_datetime(part["entry_time"]) < pd.to_datetime(part["signal_time"])
        part["context_available_time_flag"] = pd.to_datetime(part["signal_available_time"]) > pd.to_datetime(part["signal_time"])
        part["exit_time"] = index[end_pos]
        part["exit_price"] = exit_price
        part["gross_return"] = gross
        part["net_return"] = gross - float(round_trip_cost_pct)
        part["mfe"] = mfe
        part["mae"] = mae
        part["mfe_mae_ratio"] = np.where(np.abs(mae) > 0, mfe / np.abs(mae), np.nan)
        part["mfe_capture_ratio"] = np.where(mfe > 0, part["net_return"] / mfe, np.nan)
        part["time_to_mfe_bars"] = time_to_mfe
        part["time_to_mae_bars"] = time_to_mae
        part["mfe_before_mae"] = time_to_mfe <= time_to_mae
        part["mfe_ge_0p3"] = mfe >= float(fast_profit_threshold)
        part["mfe_ge_0p5"] = mfe >= float(strong_profit_threshold)
        part["mfe_ge_1p0"] = mfe >= 0.010
        part["mae_le_neg_0p3"] = mae <= -0.003
        part["mae_le_neg_0p5"] = mae <= -0.005
        part["mae_le_neg_1p0"] = mae <= -0.010
        part["mae_controlled"] = mae >= float(mae_control_threshold)
        part["tradable_mae"] = mae >= float(tradable_mae_threshold)
        part["fast_profit_flag"] = fast_profit & (mfe >= float(fast_profit_threshold))
        part["strong_short_term_edge"] = (mfe >= float(strong_profit_threshold)) & (mae >= float(mae_control_threshold)) & (time_to_mfe <= time_to_mae)
        part["tradable_short_term_edge"] = (mfe >= float(fast_profit_threshold)) & (mae >= float(tradable_mae_threshold))
        part["adverse_sweep_count"] = sweep_count
        part["adverse_sweep_flag"] = sweep_count > 0
        part["first_adverse_sweep_bars"] = first_sweep_bars
        part["fast_adverse_sweep_flag"] = fast_adverse_sweep
        part["first_reclaim_mid_bars"] = first_reclaim_mid_bars
        part["first_reclaim_high_bars"] = first_reclaim_high_bars
        part["reclaim_mid_flag"] = np.isfinite(first_reclaim_mid_bars)
        part["reclaim_high_flag"] = np.isfinite(first_reclaim_high_bars)
        part["max_adverse_breach_pct"] = max_adverse_breach_pct
        part["bad_low_mfe_flag"] = mfe < float(bad_mfe_threshold)
        part["bad_fast_adverse_low_mfe"] = fast_adverse_sweep & (mfe < float(bad_mfe_threshold))
        frames.append(part)
        progress.update(j)
    progress.close()
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"[labels] rows={len(out):,}", flush=True)
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


def _rate(part: pd.DataFrame, col: str) -> float:
    if part.empty or col not in part.columns:
        return float("nan")
    return float(part[col].fillna(False).astype(bool).mean())


def _q(part: pd.DataFrame, col: str, q: float) -> float:
    if part.empty or col not in part.columns:
        return float("nan")
    return float(pd.to_numeric(part[col], errors="coerce").quantile(q))


def add_profit_effect_stats(row: dict, part: pd.DataFrame) -> dict:
    if part.empty:
        return row
    mfe = pd.to_numeric(part["mfe"], errors="coerce")
    mae = pd.to_numeric(part["mae"], errors="coerce")
    ratio = pd.to_numeric(part["mfe_mae_ratio"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    row.update(
        {
            "events_per_month": _events_per_month(part["entry_time"]),
            "max_days_without_trade": _max_days_without_event(part["entry_time"]),
            "mfe_mean": float(mfe.mean()),
            "mfe_median": float(mfe.median()),
            "mfe_p25": float(mfe.quantile(0.25)),
            "mfe_p75": float(mfe.quantile(0.75)),
            "mfe_p90": float(mfe.quantile(0.90)),
            "mae_mean": float(mae.mean()),
            "mae_median": float(mae.median()),
            "mae_p25": float(mae.quantile(0.25)),
            "mae_p75": float(mae.quantile(0.75)),
            "mae_p90": float(mae.quantile(0.90)),
            "mfe_mae_ratio_mean": float(ratio.mean()),
            "mfe_mae_ratio_median": float(ratio.median()),
            "mfe_ge_0p3_rate": _rate(part, "mfe_ge_0p3"),
            "mfe_ge_0p5_rate": _rate(part, "mfe_ge_0p5"),
            "mfe_ge_1p0_rate": _rate(part, "mfe_ge_1p0"),
            "mae_le_neg_0p3_rate": _rate(part, "mae_le_neg_0p3"),
            "mae_le_neg_0p5_rate": _rate(part, "mae_le_neg_0p5"),
            "mae_le_neg_1p0_rate": _rate(part, "mae_le_neg_1p0"),
            "mae_controlled_rate": _rate(part, "mae_controlled"),
            "tradable_mae_rate": _rate(part, "tradable_mae"),
            "fast_profit_rate": _rate(part, "fast_profit_flag"),
            "strong_short_term_edge_rate": _rate(part, "strong_short_term_edge"),
            "tradable_short_term_edge_rate": _rate(part, "tradable_short_term_edge"),
            "mfe_before_mae_rate": _rate(part, "mfe_before_mae"),
            "bad_fast_adverse_rate": _rate(part, "bad_fast_adverse_low_mfe"),
            "sweep_rate": _rate(part, "adverse_sweep_flag"),
            "reclaim_mid_rate": _rate(part, "reclaim_mid_flag"),
            "reclaim_high_rate": _rate(part, "reclaim_high_flag"),
            "time_to_mfe_median": _q(part, "time_to_mfe_bars", 0.50),
            "time_to_mae_median": _q(part, "time_to_mae_bars", 0.50),
            "first_sweep_median": _q(part, "first_adverse_sweep_bars", 0.50),
            "first_reclaim_mid_median": _q(part, "first_reclaim_mid_bars", 0.50),
            "first_reclaim_high_median": _q(part, "first_reclaim_high_bars", 0.50),
        }
    )
    return row


def build_base_summary(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()
    rows = []
    for keys, part in labels.groupby(["entry_delay_bars", "horizon_bars"], dropna=False):
        row = {"scope": "base_lower_long_mid_high_or_extreme_vol", "entry_delay_bars": keys[0], "horizon_bars": keys[1]}
        row.update(summarize_return(part["net_return"]))
        rows.append(add_profit_effect_stats(row, part))
    return pd.DataFrame(rows).sort_values(["entry_delay_bars", "horizon_bars"]).reset_index(drop=True)


def _add_bucket_specs_from_cut(specs: list[BucketSpec], labels: pd.DataFrame, feature_name: str, bins: Iterable[float], names: Iterable[str]) -> None:
    if feature_name not in labels.columns:
        return
    x = pd.to_numeric(labels[feature_name], errors="coerce")
    bucket = pd.cut(x, list(bins), labels=list(names)).astype("object").fillna("NA")
    for b in [v for v in bucket.dropna().unique() if str(v) != "NA"]:
        specs.append(
            BucketSpec(
                feature_name=feature_name,
                bucket_name=str(b),
                description=f"{feature_name} in {b}",
                mask=(bucket == b),
            )
        )


def _add_bucket_specs_from_category(specs: list[BucketSpec], labels: pd.DataFrame, feature_name: str) -> None:
    if feature_name not in labels.columns:
        return
    values = labels[feature_name].astype("object").fillna("NA")
    for v in [x for x in values.unique() if str(x) != "NA"]:
        specs.append(
            BucketSpec(
                feature_name=feature_name,
                bucket_name=str(v),
                description=f"{feature_name} == {v}",
                mask=(values == v),
            )
        )


def build_bucket_specs(labels: pd.DataFrame) -> list[BucketSpec]:
    specs: list[BucketSpec] = []
    if labels.empty:
        return specs
    _add_bucket_specs_from_category(specs, labels, "vol_regime")
    _add_bucket_specs_from_category(specs, labels, "session")
    _add_bucket_specs_from_category(specs, labels, "trend_regime")
    _add_bucket_specs_from_cut(
        specs,
        labels,
        "range_pct",
        [-np.inf, 0.0020, 0.0035, 0.0050, 0.0080, np.inf],
        ["lt_0p20", "0p20_0p35", "0p35_0p50", "0p50_0p80", "ge_0p80"],
    )
    _add_bucket_specs_from_cut(
        specs,
        labels,
        "volume_ratio",
        [-np.inf, 2.5, 3.0, 5.0, 8.0, np.inf],
        ["lt_2p5", "2p5_3p0", "3p0_5p0", "5p0_8p0", "ge_8p0"],
    )
    _add_bucket_specs_from_cut(
        specs,
        labels,
        "event_wick_share",
        [-np.inf, 0.50, 0.60, 0.70, 0.80, np.inf],
        ["lt_0p50", "0p50_0p60", "0p60_0p70", "0p70_0p80", "ge_0p80"],
    )
    _add_bucket_specs_from_cut(
        specs,
        labels,
        "event_wick_atr",
        [-np.inf, 0.75, 1.0, 1.5, 2.5, np.inf],
        ["lt_0p75", "0p75_1p0", "1p0_1p5", "1p5_2p5", "ge_2p5"],
    )
    _add_bucket_specs_from_cut(
        specs,
        labels,
        "favorable_close_pos",
        [-np.inf, 0.40, 0.60, 0.70, 0.80, np.inf],
        ["lt_0p40", "0p40_0p60", "0p60_0p70", "0p70_0p80", "ge_0p80"],
    )
    for col in ["prior_move_against_30", "prior_move_against_120", "prior_move_against_720"]:
        _add_bucket_specs_from_cut(
            specs,
            labels,
            col,
            [-np.inf, 0.002, 0.005, 0.010, 0.020, np.inf],
            ["lt_0p20", "0p20_0p50", "0p50_1p00", "1p00_2p00", "ge_2p00"],
        )
    _add_bucket_specs_from_cut(
        specs,
        labels,
        "signed_delta_against_entry",
        [-np.inf, -0.20, 0.0, 0.20, 0.50, np.inf],
        ["lt_neg_0p20", "neg_0p20_0", "0_0p20", "0p20_0p50", "ge_0p50"],
    )
    _add_bucket_specs_from_cut(
        specs,
        labels,
        "taker_against_entry",
        [-np.inf, 0.40, 0.50, 0.60, 0.70, np.inf],
        ["lt_0p40", "0p40_0p50", "0p50_0p60", "0p60_0p70", "ge_0p70"],
    )
    return specs


def _profit_effect_score(row: pd.Series) -> float:
    trades = float(row.get("trades", 0) or 0)
    wr = float(row.get("win_rate", np.nan))
    pf = float(row.get("profit_factor", np.nan))
    mean_net = float(row.get("mean_net", np.nan))
    max_dd = float(row.get("max_drawdown", np.nan))
    mfe05 = float(row.get("mfe_ge_0p5_rate", np.nan))
    fast_profit = float(row.get("fast_profit_rate", np.nan))
    mae_ctrl = float(row.get("mae_controlled_rate", np.nan))
    bad = float(row.get("bad_fast_adverse_rate", np.nan))
    pos_years = float(row.get("positive_years", np.nan))
    if not np.isfinite(trades) or trades <= 0:
        return float("nan")
    # High-frequency smooth edge: count/win/MFE speed/MAE control dominate. Mean_net is a floor, not the sole goal.
    count_score = min(math.log1p(max(trades, 0.0)) / math.log1p(1500.0), 1.0)
    win_score = min(max((wr - 0.50) / 0.15, 0.0), 1.0) if np.isfinite(wr) else 0.0
    pf_score = min(max((pf - 0.95) / 0.60, 0.0), 1.0) if np.isfinite(pf) else 0.0
    mean_score = min(max((mean_net + 0.0003) / 0.0020, 0.0), 1.0) if np.isfinite(mean_net) else 0.0
    mfe_score = min(max((mfe05 - 0.25) / 0.35, 0.0), 1.0) if np.isfinite(mfe05) else 0.0
    fast_score = min(max((fast_profit - 0.15) / 0.35, 0.0), 1.0) if np.isfinite(fast_profit) else 0.0
    mae_score = min(max((mae_ctrl - 0.35) / 0.45, 0.0), 1.0) if np.isfinite(mae_ctrl) else 0.0
    bad_score = min(max(1.0 - bad / 0.25, 0.0), 1.0) if np.isfinite(bad) else 0.0
    dd_score = min(max(1.0 - abs(max_dd) / 0.35, 0.0), 1.0) if np.isfinite(max_dd) else 0.0
    year_score = min(max(pos_years / 4.0, 0.0), 1.0) if np.isfinite(pos_years) else 0.0
    return float(
        0.18 * count_score
        + 0.18 * win_score
        + 0.12 * pf_score
        + 0.08 * mean_score
        + 0.14 * mfe_score
        + 0.10 * fast_score
        + 0.08 * mae_score
        + 0.07 * bad_score
        + 0.03 * dd_score
        + 0.02 * year_score
    )


def build_bucket_summary(labels: pd.DataFrame, specs: list[BucketSpec]) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()
    rows = []
    for spec in specs:
        mask = spec.mask.reindex(labels.index).fillna(False).astype(bool)
        if not mask.any():
            continue
        for keys, part in labels.loc[mask].groupby(["entry_delay_bars", "horizon_bars"], dropna=False):
            row = {
                "feature_name": spec.feature_name,
                "bucket_name": spec.bucket_name,
                "description": spec.description,
                "entry_delay_bars": keys[0],
                "horizon_bars": keys[1],
            }
            row.update(summarize_return(part["net_return"]))
            rows.append(add_profit_effect_stats(row, part))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_yearly_summary(labels: pd.DataFrame, specs: list[BucketSpec]) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()
    rows = []
    labels2 = labels.copy()
    labels2["year"] = pd.to_datetime(labels2["entry_time"]).dt.year
    for spec in specs:
        mask = spec.mask.reindex(labels2.index).fillna(False).astype(bool)
        if not mask.any():
            continue
        for keys, part in labels2.loc[mask].groupby(["feature_name" if False else "year", "entry_delay_bars", "horizon_bars"], dropna=False):
            row = {
                "feature_name": spec.feature_name,
                "bucket_name": spec.bucket_name,
                "year": keys[0],
                "entry_delay_bars": keys[1],
                "horizon_bars": keys[2],
            }
            row.update(summarize_return(part["net_return"]))
            rows.append(add_profit_effect_stats(row, part))
    return pd.DataFrame(rows) if rows else pd.DataFrame()


def build_bucket_vs_base(bucket_summary: pd.DataFrame, base_summary: pd.DataFrame) -> pd.DataFrame:
    if bucket_summary.empty or base_summary.empty:
        return pd.DataFrame()
    base_cols = ["entry_delay_bars", "horizon_bars"]
    metrics = [
        "trades", "mean_net", "win_rate", "profit_factor", "max_drawdown", "events_per_month",
        "mfe_mean", "mfe_median", "mfe_ge_0p3_rate", "mfe_ge_0p5_rate", "fast_profit_rate",
        "mae_mean", "mae_median", "mae_controlled_rate", "tradable_mae_rate", "bad_fast_adverse_rate",
        "strong_short_term_edge_rate", "tradable_short_term_edge_rate", "mfe_before_mae_rate",
        "time_to_mfe_median", "time_to_mae_median", "top5_winner_share",
    ]
    keep_base = base_cols + [c for c in metrics if c in base_summary.columns]
    b = base_summary[keep_base].copy().rename(columns={c: f"base_{c}" for c in keep_base if c not in base_cols})
    out = bucket_summary.merge(b, on=base_cols, how="left")
    for c in metrics:
        bc = f"base_{c}"
        if c in out.columns and bc in out.columns:
            out[f"{c}_lift"] = pd.to_numeric(out[c], errors="coerce") - pd.to_numeric(out[bc], errors="coerce")
    out["trade_retention_rate"] = pd.to_numeric(out["trades"], errors="coerce") / pd.to_numeric(out["base_trades"], errors="coerce").replace(0, np.nan)
    return out


def build_candidate_tables(vs: pd.DataFrame, yearly: pd.DataFrame, args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if vs.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    out = vs.copy()
    if not yearly.empty:
        y = yearly.copy()
        y["year_positive"] = pd.to_numeric(y["mean_net"], errors="coerce") > 0
        y_pos = (
            y.groupby(["feature_name", "bucket_name", "entry_delay_bars", "horizon_bars"], dropna=False)["year_positive"]
            .sum()
            .reset_index()
            .rename(columns={"year_positive": "positive_years"})
        )
        # 2025 is explicitly important because prior reports showed weakness there.
        y2025 = y.loc[y["year"] == 2025, ["feature_name", "bucket_name", "entry_delay_bars", "horizon_bars", "mean_net", "win_rate", "profit_factor", "trades"]].copy()
        y2025 = y2025.rename(columns={"mean_net": "mean_net_2025", "win_rate": "win_rate_2025", "profit_factor": "profit_factor_2025", "trades": "trades_2025"})
        out = out.merge(y_pos, on=["feature_name", "bucket_name", "entry_delay_bars", "horizon_bars"], how="left")
        out = out.merge(y2025, on=["feature_name", "bucket_name", "entry_delay_bars", "horizon_bars"], how="left")
    else:
        out["positive_years"] = 0
    out["positive_years"] = out["positive_years"].fillna(0).astype(int)
    out["profit_effect_score"] = out.apply(_profit_effect_score, axis=1)
    out["candidate_reason"] = np.select(
        [
            (out["trades"] >= int(args.min_candidate_trades))
            & (out["trade_retention_rate"] >= 0.20)
            & (out["win_rate"] >= float(args.min_candidate_win_rate))
            & (out["mean_net"] > 0)
            & (out["profit_factor"] >= 1.10)
            & (out["mfe_ge_0p5_rate_lift"] > 0)
            & (out["bad_fast_adverse_rate_lift"] < 0)
            & (out["positive_years"] >= 3),
            (out["trades"] >= int(args.min_candidate_trades))
            & (out["trade_retention_rate"] >= 0.15)
            & (out["win_rate_lift"] > 0.02)
            & (out["fast_profit_rate_lift"] > 0)
            & (out["bad_fast_adverse_rate_lift"] < 0),
        ],
        ["research_continue_profit_effect_bucket", "diagnostic_profit_effect_bucket"],
        default="reject_or_diagnostic",
    )
    leaderboard = out.sort_values(["profit_effect_score", "trades", "win_rate"], ascending=[False, False, False]).reset_index(drop=True)
    candidates = out.loc[out["candidate_reason"] != "reject_or_diagnostic"].copy()
    candidates = candidates.sort_values(["candidate_reason", "profit_effect_score", "trades"], ascending=[True, False, False]).reset_index(drop=True)
    rejected = out.loc[out["candidate_reason"] == "reject_or_diagnostic"].copy().sort_values(["profit_effect_score", "trades"], ascending=[False, False]).reset_index(drop=True)
    return leaderboard, candidates, rejected


def build_causal_audit_summary(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()
    rows = []
    for keys, part in labels.groupby(["entry_delay_bars", "horizon_bars"], dropna=False):
        rows.append(
            {
                "entry_delay_bars": keys[0],
                "horizon_bars": keys[1],
                "rows": int(len(part)),
                "lookahead_flags": int(part["lookahead_flag"].fillna(False).sum()),
                "context_available_time_flags": int(part["context_available_time_flag"].fillna(False).sum()),
                "entry_not_expected_time_flags": int(part["entry_not_expected_time_flag"].fillna(False).sum()),
                "min_event_time": str(pd.to_datetime(part["event_bar_time"]).min()),
                "max_event_time": str(pd.to_datetime(part["event_bar_time"]).max()),
            }
        )
    return pd.DataFrame(rows)


def build_feature_coverage(labels: pd.DataFrame) -> pd.DataFrame:
    if labels.empty:
        return pd.DataFrame()
    rows = []
    for col in [
        "range_pct", "volume_ratio", "event_wick_share", "event_wick_atr", "favorable_close_pos",
        "prior_move_against_30", "prior_move_against_120", "prior_move_against_720",
        "signed_delta_against_entry", "taker_against_entry", "session", "trend_regime", "vol_regime",
    ]:
        if col in labels.columns:
            s = labels[col]
            rows.append({"feature": col, "rows": len(s), "non_null": int(s.notna().sum()), "coverage": float(s.notna().mean())})
    return pd.DataFrame(rows)


def _pct_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if any(k in col for k in ["rate", "return", "net", "drawdown", "mfe", "mae", "ratio", "lift", "score", "share", "breach", "retention"]):
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[write] {path} rows={len(df):,} cols={len(df.columns)}", flush=True)


def _sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty or len(df) <= n:
        return df.copy()
    sort_cols = [c for c in ["entry_time", "event_bar_time"] if c in df.columns]
    out = df.sample(n=int(n), random_state=42)
    return out.sort_values(sort_cols).reset_index(drop=True) if sort_cols else out.reset_index(drop=True)


def write_reports(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    events: pd.DataFrame,
    labels: pd.DataFrame,
    bucket_specs: list[BucketSpec],
) -> None:
    print("[report] building profit-effect reports", flush=True)
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
        "horizon_bars": args.horizon_bars,
        "entry_delay_bars_list": args.entry_delay_bars_list,
        "base_vol_regimes": args.base_vol_regimes,
        "research_policy": "Keep the known mid_high/extreme-vol lower-shadow universe broad; score one causal feature bucket at a time by short-term MFE/MAE profit effect. Do not combine filters yet.",
        "notes": [
            "Main remembered lead: lower_volume_shadow_long + mid_high_vol/extreme_vol + horizon60.",
            "This version diagnoses buckets and soft scores. It does not recommend directly tightening multiple conditions.",
            "MFE/MAE labels are future path diagnostics only; bucket masks use causal event-bar features.",
        ],
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    base_summary = build_base_summary(labels)
    feature_coverage = build_feature_coverage(labels)
    bucket_summary = build_bucket_summary(labels, bucket_specs)
    yearly = build_yearly_summary(labels, bucket_specs)
    vs = build_bucket_vs_base(bucket_summary, base_summary)
    leaderboard, candidates, rejected = build_candidate_tables(vs, yearly, args)
    causal = build_causal_audit_summary(labels)

    event_counts = pd.DataFrame(
        [
            {
                "event_name": "lower_volume_shadow_long_mid_high_or_extreme_vol",
                "events": int(len(events)),
                "base_vol_regimes": args.base_vol_regimes,
                "min_event_time": str(pd.to_datetime(events["event_bar_time"]).min()) if not events.empty else "",
                "max_event_time": str(pd.to_datetime(events["event_bar_time"]).max()) if not events.empty else "",
            }
        ]
    )
    _write_csv(event_counts, out_dir / "01_event_counts.csv")
    _write_csv(_pct_columns(feature_coverage), out_dir / "02_feature_coverage.csv")
    _write_csv(_pct_columns(base_summary), out_dir / "03_base_profit_effect_summary.csv")
    _write_csv(_pct_columns(bucket_summary), out_dir / "04_feature_bucket_profit_effect.csv")
    _write_csv(_pct_columns(vs), out_dir / "05_feature_bucket_vs_base.csv")
    _write_csv(_pct_columns(yearly), out_dir / "06_feature_bucket_yearly.csv")
    _write_csv(_pct_columns(leaderboard.head(5000)), out_dir / "07_profit_effect_leaderboard.csv")
    _write_csv(_pct_columns(candidates), out_dir / "08_research_continue_buckets.csv")
    _write_csv(_pct_columns(rejected.head(5000)), out_dir / "09_rejected_or_diagnostic_buckets.csv")
    _write_csv(causal, out_dir / "10_causal_audit_summary.csv")

    sample_cols = [
        "event_id", "event_bar_time", "entry_delay_bars", "horizon_bars", "entry_time", "net_return",
        "mfe", "mae", "mfe_mae_ratio", "time_to_mfe_bars", "time_to_mae_bars", "mfe_before_mae",
        "mfe_ge_0p3", "mfe_ge_0p5", "fast_profit_flag", "strong_short_term_edge", "tradable_short_term_edge",
        "bad_fast_adverse_low_mfe", "adverse_sweep_count", "first_adverse_sweep_bars", "first_reclaim_mid_bars",
        "first_reclaim_high_bars", "range_pct", "volume_ratio", "event_wick_share", "event_wick_atr",
        "favorable_close_pos", "prior_move_against_30", "prior_move_against_120", "signed_delta_against_entry",
        "taker_against_entry", "session", "vol_regime", "trend_regime",
    ]
    sample_cols = [c for c in sample_cols if c in labels.columns]
    _write_csv(_sample(labels[sample_cols], int(args.event_sample_size)), out_dir / "11_label_sample.csv")

    if not candidates.empty:
        cand_frames = []
        top = candidates.head(30)
        for row in top.itertuples(index=False):
            spec = next((s for s in bucket_specs if s.feature_name == getattr(row, "feature_name") and s.bucket_name == getattr(row, "bucket_name")), None)
            if spec is None:
                continue
            subset = labels.loc[
                spec.mask.reindex(labels.index).fillna(False).to_numpy(dtype=bool)
                & (labels["entry_delay_bars"] == getattr(row, "entry_delay_bars"))
                & (labels["horizon_bars"] == getattr(row, "horizon_bars"))
            ]
            if not subset.empty:
                cand_frames.append(
                    _sample(
                        subset[sample_cols].assign(feature_name=getattr(row, "feature_name"), bucket_name=getattr(row, "bucket_name")),
                        max(50, int(args.candidate_sample_size) // max(len(top), 1)),
                    )
                )
        cand_sample = pd.concat(cand_frames, ignore_index=True) if cand_frames else pd.DataFrame()
    else:
        cand_sample = pd.DataFrame()
    _write_csv(cand_sample, out_dir / "12_candidate_bucket_sample.csv")

    if bool(args.write_slim_labels):
        _write_csv(labels[sample_cols], out_dir / "13_slim_profit_effect_labels.csv")

    readme = f"""# {TITLE}

Scope: `lower_volume_shadow_long` events inside `{args.base_vol_regimes}` only, horizon={int(args.horizon_bars)} bars.

Goal: score short-term profit effect without prematurely tightening the signal.

Reading order:
1. `10_causal_audit_summary.csv` — confirm no lookahead/context leakage.
2. `03_base_profit_effect_summary.csv` — the broad remembered lead baseline by entry delay.
3. `04_feature_bucket_profit_effect.csv` — MFE/MAE, fast-profit, bad-path, win-rate, PF by one feature bucket.
4. `05_feature_bucket_vs_base.csv` — bucket lifts versus the broad base universe.
5. `07_profit_effect_leaderboard.csv` — smooth high-win/MFE/MAE oriented ranking, not mean_net only.
6. `08_research_continue_buckets.csv` — buckets worth further diagnosis; do not combine them yet without another controlled test.

Important: MFE/MAE and bad-path labels are future diagnostics. Bucket definitions use causal event-bar features only.
"""
    (out_dir / "README_RESEARCH.md").write_text(readme, encoding="utf-8")

    review_prompt = f"""You are reviewing {TITLE}.

Known lead to preserve: `lower_volume_shadow_long + mid_high_vol/extreme_vol + horizon60`.

Review tasks:
1. Confirm `10_causal_audit_summary.csv` has no lookahead/context leakage.
2. Use `03_base_profit_effect_summary.csv` as the broad baseline; do not ask to tighten immediately.
3. Use `04_feature_bucket_profit_effect.csv` and `05_feature_bucket_vs_base.csv` to identify which single feature buckets improve short-term MFE/MAE profile, fast-profit rate, bad-fast-adverse rate, win rate, PF, and 2025 stability.
4. Prioritize high trade count and high win rate. Mean_net is a floor, not the only ranking criterion.
5. Do not recommend combining buckets yet. The next step should only combine conditions if independent single-bucket improvements are clear and trade count remains acceptable.
"""
    (out_dir / "GPT_REVIEW_PROMPT.md").write_text(review_prompt, encoding="utf-8")

    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    print(f"[run] {SCRIPT_NAME} v{SCRIPT_VERSION}", flush=True)
    print(f"[args] out_dir={out_dir}", flush=True)
    print("[scope] lower shadow long + mid_high/extreme vol; MFE/MAE profit-effect scoring; no combined filters", flush=True)

    entry_delays = _parse_int_csv(args.entry_delay_bars_list)
    base_vol_regimes = _parse_str_csv(args.base_vol_regimes)
    th = ShadowThresholds(
        wick_share_min=float(args.wick_share_min),
        wick_atr_min=float(args.wick_atr_min),
        volume_ratio_min=float(args.volume_ratio_min),
    )

    bars = load_bars(args)
    features = build_features(bars, th)
    mask = _research_window_mask(features.index, args.start_date, args.end_date)
    research_features = features.loc[mask].copy()
    events = build_lower_vol_events(research_features, base_vol_regimes)
    if events.empty:
        print("[done] no events found", flush=True)
        write_reports(out_dir=out_dir, args=args, events=events, labels=pd.DataFrame(), bucket_specs=[])
        return 0

    labels = build_profit_effect_labels(
        research_features,
        events,
        horizon_bars=int(args.horizon_bars),
        entry_delays=entry_delays,
        round_trip_cost_pct=float(args.round_trip_cost_pct),
        bad_mfe_threshold=float(args.bad_mfe_threshold),
        fast_profit_threshold=float(args.fast_profit_threshold),
        strong_profit_threshold=float(args.strong_profit_threshold),
        mae_control_threshold=float(args.mae_control_threshold),
        tradable_mae_threshold=float(args.tradable_mae_threshold),
        fast_window_bars=int(args.fast_window_bars),
    )
    bucket_specs = build_bucket_specs(labels) if not labels.empty else []
    print(f"[buckets] specs={len(bucket_specs):,}", flush=True)
    write_reports(out_dir=out_dir, args=args, events=events, labels=labels, bucket_specs=bucket_specs)
    print("[done] lower-shadow vol-regime profit-effect probe complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
