#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH 1m lower-shadow vol-regime direction scorecard probe V1.

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

SCRIPT_NAME = "eth_1m_lower_shadow_vol_direction_scorecard_probe_v1"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_MF_1M_LOWER_SHADOW_VOL_DIRECTION_SCORECARD_PROBE_V1"
EDGE_ID = "RESEARCH_ONLY_ETH_MF_1M_LOWER_SHADOW_VOL_DIRECTION_SCORECARD_PROBE_V1"
TITLE = "ETH 1m Lower Shadow Vol-Regime Direction Scorecard Probe V1"
DEFAULT_OUT_DIR = "data/reports/research/eth_1m_lower_shadow_vol_direction_scorecard_probe_v1"
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


@dataclass(frozen=True)
class DirectionSpec:
    direction_name: str
    family: str
    description: str
    intended_use: str
    risk_note: str
    mask: pd.Series


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Lower-shadow mid/high/extreme-vol direction scorecard probe: score different edge families for sizing/risk logic without combining filters yet.",
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
    p.add_argument("--min-tier-a-trades", type=int, default=450, help="Minimum trades for a potential broad core / Tier A direction.")
    p.add_argument("--min-tier-b-trades", type=int, default=200, help="Minimum trades for a potential quality booster / Tier B direction.")

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




def build_direction_specs(labels: pd.DataFrame) -> list[DirectionSpec]:
    """Build named, interpretable direction families.

    These are still single-feature or broad diagnostic directions. They are not
    combined filters and should not be treated as final entry rules. The purpose
    is to decide which edge families may deserve different sizing/exit/risk.
    """
    if labels.empty:
        return []
    idx = labels.index

    def num(col: str) -> pd.Series:
        return pd.to_numeric(labels[col], errors="coerce") if col in labels.columns else pd.Series(np.nan, index=idx)

    def cat(col: str) -> pd.Series:
        return labels[col].astype(str) if col in labels.columns else pd.Series("NA", index=idx)

    directions: list[DirectionSpec] = []

    def add(name: str, family: str, desc: str, use: str, risk: str, mask: pd.Series) -> None:
        directions.append(DirectionSpec(name, family, desc, use, risk, mask.fillna(False)))

    add("base_mid_high_or_extreme_vol", "base_core", "All lower-volume-shadow LONG events in mid_high_vol or extreme_vol.", "Broad baseline / possible base sleeve if later exit-risk improves.", "Do not size from this alone; MAE and drawdown remain large.", pd.Series(True, index=idx))
    add("panic_depth_720_ge_2p00", "panic_depth", "Prior 720-bar move against LONG entry >= 2%; long-context selloff before lower shadow.", "Candidate broad core direction; likely panic-reversal / squeeze-relief edge.", "Good candidate for normal sizing only after full backtest; avoid stacking with redundant downtrend filters too early.", num("prior_move_against_720") >= 0.020)
    add("panic_depth_120_ge_2p00", "panic_depth", "Prior 120-bar move against LONG entry >= 2%; shorter panic move before lower shadow.", "Candidate broad core/booster direction; compare with 720-bar depth.", "May be more reactive but less stable than 720-bar depth; check 2025 separately.", num("prior_move_against_120") >= 0.020)
    add("trend_downtrend", "panic_depth", "EMA/trend regime is downtrend at event time.", "High-retention broad core diagnostic; simple and robust if yearly stability holds.", "Trend filter can lag regime changes; do not combine blindly with prior_move yet.", cat("trend_regime") == "downtrend")
    add("intensity_range_0p50_0p80", "event_intensity", "Event 1m range_pct is 0.50%-0.80%; strong but not most extreme bar.", "Quality booster while retaining some frequency; good for intensity-based confidence scaling.", "Range buckets can be regime-sensitive; later compare relative range vs recent bars.", (num("range_pct") >= 0.0050) & (num("range_pct") < 0.0080))
    add("intensity_range_ge_0p80", "event_intensity", "Event 1m range_pct >= 0.80%; extreme panic candle.", "Profit-max / booster direction; likely higher confidence but lower frequency.", "Do not use as only signal if high-frequency smoothing is the goal; sample count can shrink.", num("range_pct") >= 0.0080)
    add("volume_ratio_3p0_5p0", "event_intensity", "Volume ratio is 3x-5x recent median; strong but not ultra-extreme volume burst.", "Potential balanced intensity direction; useful for confidence scoring.", "Volume is not monotonic; ultra-high volume may mean continuation risk.", (num("volume_ratio") >= 3.0) & (num("volume_ratio") < 5.0))
    add("wick_share_ge_0p80", "event_intensity", "Lower wick share >= 80% of event range; strong rejection shape.", "Quality/booster direction; useful as shape-confidence score.", "Lower frequency; shape alone should not override context.", num("event_wick_share") >= 0.80)
    add("wick_atr_ge_1p50", "event_intensity", "Lower wick is at least 1.5x recent ATR; relative wick shock.", "High-purity booster / future relative-range research bridge.", "Likely low frequency; later compare range_ratio_N and wick_range_ratio_N.", num("event_wick_atr") >= 1.50)
    add("extreme_vol_only", "volatility_quality", "Base signal only during extreme_vol regime.", "Profit-max / high-risk-high-confidence diagnostic direction.", "Sample can be small; risk of crisis-regime behavior and wider slippage.", cat("vol_regime") == "extreme_vol")
    add("moderate_delta_absorption_0_0p20", "absorption_quality", "Signed delta against LONG entry is 0 to 0.20; moderate sell pressure into lower shadow.", "Absorption-quality direction; bridge to footprint/CVD confirmation later.", "Trade-bar delta proxy may be weaker than footprint; do not overfit before footprint review.", (num("signed_delta_against_entry") >= 0.0) & (num("signed_delta_against_entry") < 0.20))
    add("moderate_taker_against_0p50_0p60", "absorption_quality", "Taker flow against LONG entry is 50%-60%; moderate active selling but not extreme.", "Absorption-quality direction; candidate for future order-flow sizing confidence.", "Likely overlaps with delta direction; check overlap before combining.", (num("taker_against_entry") >= 0.50) & (num("taker_against_entry") < 0.60))
    add("asia_session", "session_structure", "Event occurs in Asia session.", "Diagnostic/session-specific direction; may deserve session-specific sizing if robust.", "Session effects can be calendar/regime artifacts; require independent validation before using as core filter.", cat("session") == "asia")
    return directions


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return 0.0
    return float(min(max(x, 0.0), 1.0))


def _direction_scores(row: pd.Series) -> dict[str, float]:
    trades = float(row.get("trades", 0) or 0)
    wr = float(row.get("win_rate", np.nan))
    pf = float(row.get("profit_factor", np.nan))
    mean_net = float(row.get("mean_net", np.nan))
    total = float(row.get("total_return", np.nan))
    max_dd = float(row.get("max_drawdown", np.nan))
    mfe05 = float(row.get("mfe_ge_0p5_rate", np.nan))
    mfe1 = float(row.get("mfe_ge_1p0_rate", np.nan))
    fast_profit = float(row.get("fast_profit_rate", np.nan))
    mae_ctrl = float(row.get("mae_controlled_rate", np.nan))
    bad = float(row.get("bad_fast_adverse_rate", np.nan))
    ratio = float(row.get("mfe_mae_ratio_median", np.nan))
    pos_years = float(row.get("positive_years", np.nan))
    top5 = float(row.get("top5_winner_share", np.nan))
    events_per_month = float(row.get("events_per_month", np.nan))
    count_score = _clip01(math.log1p(max(trades, 0.0)) / math.log1p(1200.0))
    frequency_score = _clip01((events_per_month - 8.0) / 20.0) if np.isfinite(events_per_month) else 0.0
    win_score = _clip01((wr - 0.50) / 0.15) if np.isfinite(wr) else 0.0
    pf_score = _clip01((pf - 1.00) / 0.80) if np.isfinite(pf) else 0.0
    mean_score = _clip01((mean_net + 0.0003) / 0.0040) if np.isfinite(mean_net) else 0.0
    total_score = _clip01(math.log1p(max(total, 0.0)) / math.log1p(5.0)) if np.isfinite(total) else 0.0
    mfe05_score = _clip01((mfe05 - 0.50) / 0.35) if np.isfinite(mfe05) else 0.0
    mfe1_score = _clip01((mfe1 - 0.20) / 0.35) if np.isfinite(mfe1) else 0.0
    fast_score = _clip01((fast_profit - 0.20) / 0.35) if np.isfinite(fast_profit) else 0.0
    mae_score = _clip01((mae_ctrl - 0.35) / 0.45) if np.isfinite(mae_ctrl) else 0.0
    bad_score = _clip01(1.0 - bad / 0.18) if np.isfinite(bad) else 0.0
    ratio_score = _clip01((ratio - 0.80) / 1.20) if np.isfinite(ratio) else 0.0
    dd_score = _clip01(1.0 - abs(max_dd) / 0.45) if np.isfinite(max_dd) else 0.0
    year_score = _clip01(pos_years / 4.0) if np.isfinite(pos_years) else 0.0
    top5_score = _clip01(1.0 - top5 / 0.40) if np.isfinite(top5) else 0.0
    smooth_high_freq_score = float(0.18 * count_score + 0.14 * frequency_score + 0.16 * win_score + 0.12 * pf_score + 0.12 * mfe05_score + 0.10 * fast_score + 0.08 * bad_score + 0.06 * year_score + 0.04 * dd_score)
    profit_max_score = float(0.12 * count_score + 0.14 * win_score + 0.16 * pf_score + 0.18 * mean_score + 0.10 * total_score + 0.14 * mfe1_score + 0.08 * ratio_score + 0.04 * year_score + 0.04 * top5_score)
    defensive_risk_score = float(0.16 * count_score + 0.12 * win_score + 0.10 * pf_score + 0.20 * mae_score + 0.18 * bad_score + 0.10 * dd_score + 0.08 * year_score + 0.06 * top5_score)
    return {"smooth_high_freq_score": smooth_high_freq_score, "profit_max_score": profit_max_score, "defensive_risk_score": defensive_risk_score, "balanced_direction_score": float(0.45 * smooth_high_freq_score + 0.30 * defensive_risk_score + 0.25 * profit_max_score)}


def _assign_tier(row: pd.Series, args: argparse.Namespace) -> tuple[str, str, str]:
    trades = int(row.get("trades", 0) or 0)
    wr = float(row.get("win_rate", np.nan))
    pf = float(row.get("profit_factor", np.nan))
    mean_net = float(row.get("mean_net", np.nan))
    bad = float(row.get("bad_fast_adverse_rate", np.nan))
    pos_years = int(row.get("positive_years", 0) or 0)
    smooth = float(row.get("smooth_high_freq_score", np.nan))
    profit = float(row.get("profit_max_score", np.nan))
    defensive = float(row.get("defensive_risk_score", np.nan))
    if trades >= int(args.min_tier_a_trades) and wr >= 0.58 and pf >= 1.45 and mean_net > 0 and bad <= 0.12 and pos_years >= 4 and smooth >= 0.55:
        return ("Tier_A_core_candidate", "base_or_normal_sleeve", "Use as candidate core only after candidate backtest; prioritize stable exit and conservative sizing.")
    if trades >= int(args.min_tier_b_trades) and wr >= 0.58 and pf >= 1.45 and mean_net > 0 and pos_years >= 3 and (profit >= 0.55 or defensive >= 0.55):
        return ("Tier_B_quality_booster", "confidence_booster_or_larger_tp", "Use as higher-quality overlay or confidence multiplier; monitor sample size and overlap.")
    if trades >= 100 and mean_net > 0 and pf >= 1.25 and wr >= 0.55:
        return ("Tier_C_diagnostic_alpha", "small_sleeve_or_filter_candidate", "Keep for future validation; do not allocate large risk yet.")
    if trades >= 200 and bad <= 0.10 and defensive >= 0.50:
        return ("Tier_D_defensive_filter", "risk_reducer", "May reduce risk or define lower leverage zone; not necessarily a standalone edge.")
    return ("Diagnostic_or_reject", "no_allocation", "Track only; not enough evidence for sizing or entry logic.")


def build_direction_yearly(labels: pd.DataFrame, directions: list[DirectionSpec]) -> pd.DataFrame:
    if labels.empty or not directions:
        return pd.DataFrame()
    rows = []
    years = pd.to_datetime(labels["entry_time"]).dt.year
    for spec in directions:
        mask_all = spec.mask.reindex(labels.index).fillna(False).to_numpy(dtype=bool)
        for keys, part_delay in labels.groupby(["entry_delay_bars", "horizon_bars"], dropna=False):
            p0 = part_delay.loc[mask_all[part_delay.index]].copy()
            if p0.empty:
                continue
            p0["year"] = years.loc[p0.index].to_numpy()
            for year, part in p0.groupby("year"):
                row = {"direction_name": spec.direction_name, "family": spec.family, "entry_delay_bars": int(keys[0]), "horizon_bars": int(keys[1]), "year": int(year)}
                row.update(summarize_return(part["net_return"]))
                row = add_profit_effect_stats(row, part)
                row["is_positive_year"] = bool(row.get("mean_net", np.nan) > 0)
                rows.append(row)
    return pd.DataFrame(rows).sort_values(["direction_name", "entry_delay_bars", "year"]).reset_index(drop=True) if rows else pd.DataFrame()


def build_direction_scorecard(labels: pd.DataFrame, directions: list[DirectionSpec], args: argparse.Namespace) -> pd.DataFrame:
    if labels.empty or not directions:
        return pd.DataFrame()
    rows = []
    base = build_base_summary(labels)
    base_map = {(int(r.entry_delay_bars), int(r.horizon_bars)): r for r in base.itertuples(index=False)}
    for spec in directions:
        mask_all = spec.mask.reindex(labels.index).fillna(False).to_numpy(dtype=bool)
        for keys, part_delay in labels.groupby(["entry_delay_bars", "horizon_bars"], dropna=False):
            part = part_delay.loc[mask_all[part_delay.index]]
            if part.empty:
                continue
            entry_delay, horizon = int(keys[0]), int(keys[1])
            row = {"direction_name": spec.direction_name, "family": spec.family, "description": spec.description, "intended_use": spec.intended_use, "risk_note": spec.risk_note, "entry_delay_bars": entry_delay, "horizon_bars": horizon}
            row.update(summarize_return(part["net_return"]))
            row = add_profit_effect_stats(row, part)
            base_row = base_map.get((entry_delay, horizon))
            if base_row is not None:
                base_trades = float(getattr(base_row, "trades", np.nan))
                row["trade_retention_rate"] = float(row["trades"] / base_trades) if base_trades and np.isfinite(base_trades) else np.nan
                for col in ["mean_net", "win_rate", "profit_factor", "mfe_ge_0p5_rate", "fast_profit_rate", "bad_fast_adverse_rate", "mae_controlled_rate", "mfe_mae_ratio_median", "max_drawdown"]:
                    b = float(getattr(base_row, col, np.nan)) if hasattr(base_row, col) else np.nan
                    v = float(row.get(col, np.nan))
                    row[f"{col}_lift_vs_base"] = v - b if np.isfinite(v) and np.isfinite(b) else np.nan
            rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    yearly = build_direction_yearly(labels, directions)
    if not yearly.empty:
        pos = yearly.groupby(["direction_name", "entry_delay_bars", "horizon_bars"])["is_positive_year"].sum().reset_index(name="positive_years")
        y2025 = yearly.loc[yearly["year"] == 2025, ["direction_name", "entry_delay_bars", "horizon_bars", "mean_net", "win_rate", "profit_factor", "trades"]].rename(columns={"mean_net": "mean_net_2025", "win_rate": "win_rate_2025", "profit_factor": "profit_factor_2025", "trades": "trades_2025"})
        out = out.merge(pos, on=["direction_name", "entry_delay_bars", "horizon_bars"], how="left")
        out = out.merge(y2025, on=["direction_name", "entry_delay_bars", "horizon_bars"], how="left")
    out["positive_years"] = out.get("positive_years", 0).fillna(0).astype(int)
    scores = out.apply(_direction_scores, axis=1).apply(pd.Series)
    out = pd.concat([out, scores], axis=1)
    tier_info = out.apply(lambda r: _assign_tier(r, args), axis=1)
    out["direction_tier"] = [x[0] for x in tier_info]
    out["suggested_role"] = [x[1] for x in tier_info]
    out["tier_note"] = [x[2] for x in tier_info]
    return out.sort_values(["balanced_direction_score", "trades"], ascending=[False, False]).reset_index(drop=True)


def build_direction_overlap(labels: pd.DataFrame, directions: list[DirectionSpec], *, entry_delay: int = 0) -> pd.DataFrame:
    if labels.empty or not directions:
        return pd.DataFrame()
    part = labels.loc[labels["entry_delay_bars"] == int(entry_delay)].copy()
    if part.empty:
        return pd.DataFrame()
    rows = []
    masks = {spec.direction_name: spec.mask.reindex(part.index).fillna(False).to_numpy(dtype=bool) for spec in directions}
    for name_i, mi in masks.items():
        ni = int(mi.sum())
        for name_j, mj in masks.items():
            nj = int(mj.sum())
            inter = int((mi & mj).sum())
            union = int((mi | mj).sum())
            rows.append({"entry_delay_bars": int(entry_delay), "direction_i": name_i, "direction_j": name_j, "n_i": ni, "n_j": nj, "intersection": inter, "pct_of_i": float(inter / ni) if ni else np.nan, "pct_of_j": float(inter / nj) if nj else np.nan, "jaccard": float(inter / union) if union else np.nan})
    return pd.DataFrame(rows)


def build_tier_playbook(scorecard: pd.DataFrame) -> pd.DataFrame:
    if scorecard.empty:
        return pd.DataFrame()
    best = scorecard.loc[scorecard["entry_delay_bars"] == scorecard["entry_delay_bars"].min()].copy()
    if best.empty:
        best = scorecard.copy()
    rows = []
    for row in best.itertuples(index=False):
        tier = getattr(row, "direction_tier")
        if tier == "Tier_A_core_candidate":
            sizing = "normal_candidate_sleeve_after_backtest"
            entry_logic = "can act as base entry family; do not combine with other filters until overlap/backtest confirms"
            exit_logic = "use conservative 60bar/path-protection exit first; avoid chasing 240bar returns"
        elif tier == "Tier_B_quality_booster":
            sizing = "increase_confidence_or_partial_size_only"
            entry_logic = "use as confidence multiplier or quality overlay, not mandatory global filter"
            exit_logic = "allow slightly wider profit target or slower exit only after MFE/MAE backtest"
        elif tier == "Tier_C_diagnostic_alpha":
            sizing = "small_experimental_sleeve_or_watchlist"
            entry_logic = "keep as separate direction; avoid merging into core until independent validation"
            exit_logic = "same as base until direction-specific exit is tested"
        elif tier == "Tier_D_defensive_filter":
            sizing = "risk_reducer_or_leverage_down_zone"
            entry_logic = "better used to reduce size/avoid weak trades than to add leverage"
            exit_logic = "prioritize fast adverse stop / MAE control"
        else:
            sizing = "no_allocation"
            entry_logic = "diagnostic only"
            exit_logic = "not applicable"
        rows.append({"direction_name": getattr(row, "direction_name"), "family": getattr(row, "family"), "direction_tier": tier, "suggested_role": getattr(row, "suggested_role"), "suggested_sizing_policy": sizing, "suggested_entry_policy": entry_logic, "suggested_exit_policy": exit_logic, "risk_note": getattr(row, "risk_note"), "trades": getattr(row, "trades"), "win_rate": getattr(row, "win_rate"), "profit_factor": getattr(row, "profit_factor"), "mean_net": getattr(row, "mean_net"), "bad_fast_adverse_rate": getattr(row, "bad_fast_adverse_rate"), "balanced_direction_score": getattr(row, "balanced_direction_score"), "smooth_high_freq_score": getattr(row, "smooth_high_freq_score"), "profit_max_score": getattr(row, "profit_max_score"), "defensive_risk_score": getattr(row, "defensive_risk_score")})
    return pd.DataFrame(rows).sort_values(["direction_tier", "balanced_direction_score"], ascending=[True, False]).reset_index(drop=True)

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
    direction_specs: list[DirectionSpec],
) -> None:
    print("[report] building direction scorecard reports", flush=True)
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
        "research_policy": "Score multiple interpretable edge directions for different sizing/entry/exit roles without combining them into one rule yet.",
        "notes": [
            "Main remembered lead: lower_volume_shadow_long + mid_high_vol/extreme_vol + horizon60.",
            "Different directions may deserve different position size, risk controls, and exits; this report assigns tiers but does not allocate live capital.",
            "MFE/MAE labels are future diagnostics only. Direction masks use causal event-bar or pre-event features.",
            "Do not force all good buckets into a single combined filter; preserve frequency by treating directions separately.",
        ],
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    base_summary = build_base_summary(labels)
    feature_coverage = build_feature_coverage(labels)
    bucket_summary = build_bucket_summary(labels, bucket_specs)
    bucket_yearly = build_yearly_summary(labels, bucket_specs)
    bucket_vs = build_bucket_vs_base(bucket_summary, base_summary)
    bucket_leaderboard, bucket_candidates, bucket_rejected = build_candidate_tables(bucket_vs, bucket_yearly, args)
    direction_scorecard = build_direction_scorecard(labels, direction_specs, args)
    direction_yearly = build_direction_yearly(labels, direction_specs)
    direction_overlap = build_direction_overlap(labels, direction_specs, entry_delay=0)
    tier_playbook = build_tier_playbook(direction_scorecard)
    causal = build_causal_audit_summary(labels)
    event_counts = pd.DataFrame([{"event_name": "lower_volume_shadow_long_mid_high_or_extreme_vol", "events": int(len(events)), "base_vol_regimes": args.base_vol_regimes, "directions": int(len(direction_specs)), "min_event_time": str(pd.to_datetime(events["event_bar_time"]).min()) if not events.empty else "", "max_event_time": str(pd.to_datetime(events["event_bar_time"]).max()) if not events.empty else ""}])
    _write_csv(event_counts, out_dir / "01_event_counts.csv")
    _write_csv(_pct_columns(feature_coverage), out_dir / "02_feature_coverage.csv")
    _write_csv(_pct_columns(base_summary), out_dir / "03_base_profit_effect_summary.csv")
    _write_csv(_pct_columns(direction_scorecard), out_dir / "04_direction_scorecard.csv")
    _write_csv(_pct_columns(direction_yearly), out_dir / "05_direction_yearly.csv")
    _write_csv(_pct_columns(direction_overlap), out_dir / "06_direction_overlap_delay0.csv")
    _write_csv(_pct_columns(tier_playbook), out_dir / "07_direction_tier_playbook.csv")
    _write_csv(_pct_columns(direction_scorecard.sort_values(["smooth_high_freq_score", "trades"], ascending=[False, False]).head(2000)), out_dir / "08_smooth_high_freq_leaderboard.csv")
    _write_csv(_pct_columns(direction_scorecard.sort_values(["profit_max_score", "trades"], ascending=[False, False]).head(2000)), out_dir / "09_profit_max_leaderboard.csv")
    _write_csv(_pct_columns(direction_scorecard.sort_values(["defensive_risk_score", "trades"], ascending=[False, False]).head(2000)), out_dir / "10_defensive_risk_leaderboard.csv")
    _write_csv(_pct_columns(bucket_summary), out_dir / "11_feature_bucket_profit_effect.csv")
    _write_csv(_pct_columns(bucket_vs), out_dir / "12_feature_bucket_vs_base.csv")
    _write_csv(_pct_columns(bucket_leaderboard.head(5000)), out_dir / "13_bucket_profit_effect_leaderboard.csv")
    _write_csv(_pct_columns(bucket_candidates), out_dir / "14_research_continue_buckets.csv")
    _write_csv(_pct_columns(bucket_rejected.head(5000)), out_dir / "15_rejected_or_diagnostic_buckets.csv")
    _write_csv(causal, out_dir / "16_causal_audit_summary.csv")
    sample_cols = ["event_id", "event_bar_time", "entry_delay_bars", "horizon_bars", "entry_time", "net_return", "mfe", "mae", "mfe_mae_ratio", "time_to_mfe_bars", "time_to_mae_bars", "mfe_before_mae", "mfe_ge_0p3", "mfe_ge_0p5", "fast_profit_flag", "strong_short_term_edge", "tradable_short_term_edge", "bad_fast_adverse_low_mfe", "adverse_sweep_count", "first_adverse_sweep_bars", "first_reclaim_mid_bars", "first_reclaim_high_bars", "range_pct", "volume_ratio", "event_wick_share", "event_wick_atr", "favorable_close_pos", "prior_move_against_30", "prior_move_against_120", "prior_move_against_720", "signed_delta_against_entry", "taker_against_entry", "session", "vol_regime", "trend_regime"]
    sample_cols = [c for c in sample_cols if c in labels.columns]
    _write_csv(_sample(labels[sample_cols], int(args.event_sample_size)), out_dir / "17_label_sample.csv")
    if not direction_scorecard.empty:
        direction_frames = []
        top_dirs = direction_scorecard.loc[direction_scorecard["entry_delay_bars"] == 0].head(20)
        for row in top_dirs.itertuples(index=False):
            spec = next((s for s in direction_specs if s.direction_name == getattr(row, "direction_name")), None)
            if spec is None:
                continue
            subset = labels.loc[spec.mask.reindex(labels.index).fillna(False).to_numpy(dtype=bool) & (labels["entry_delay_bars"] == getattr(row, "entry_delay_bars")) & (labels["horizon_bars"] == getattr(row, "horizon_bars"))]
            if not subset.empty:
                direction_frames.append(_sample(subset[sample_cols].assign(direction_name=getattr(row, "direction_name"), direction_tier=getattr(row, "direction_tier")), max(50, int(args.candidate_sample_size) // max(len(top_dirs), 1))))
        direction_sample = pd.concat(direction_frames, ignore_index=True) if direction_frames else pd.DataFrame()
    else:
        direction_sample = pd.DataFrame()
    _write_csv(direction_sample, out_dir / "18_direction_trade_sample.csv")
    if bool(args.write_slim_labels):
        _write_csv(labels[sample_cols], out_dir / "19_slim_profit_effect_labels.csv")
    readme = f"""# {TITLE}

Scope: `lower_volume_shadow_long` events inside `{args.base_vol_regimes}` only, horizon={int(args.horizon_bars)} bars.

Goal: classify different edge directions by role, not force them into one combined rule.

Reading order:
1. `16_causal_audit_summary.csv` — confirm no lookahead/context leakage.
2. `03_base_profit_effect_summary.csv` — broad remembered lead baseline.
3. `04_direction_scorecard.csv` — main output: named edge directions scored by smooth-high-frequency, profit-max, and defensive-risk roles.
4. `07_direction_tier_playbook.csv` — suggested role/sizing/exit direction for each named direction.
5. `06_direction_overlap_delay0.csv` — check whether directions are redundant before combining anything.
6. `08/09/10_*_leaderboard.csv` — separate rankings for smoothing, profit maximization, and defensive risk.
7. `11-14` bucket reports — keep single-bucket diagnostics for future hypotheses.

Important: direction tiers are research labels, not live allocations. MFE/MAE and bad-path metrics are future diagnostics; direction definitions use causal event-bar or pre-event features only.
"""
    (out_dir / "README_RESEARCH.md").write_text(readme, encoding="utf-8")
    review_prompt = f"""You are reviewing {TITLE}.

Known lead to preserve: `lower_volume_shadow_long + mid_high_vol/extreme_vol + horizon60`.

Review tasks:
1. Confirm `16_causal_audit_summary.csv` has no lookahead/context leakage.
2. Use `04_direction_scorecard.csv` to identify which directions are suited for high-frequency smooth base sleeve, profit-max booster, and defensive risk reduction.
3. Use `06_direction_overlap_delay0.csv` before recommending any combination. If two directions overlap heavily, do not count them as independent edges.
4. Prioritize trade count + win rate + MFE/MAE + bad-rate + yearly stability. Do not rank by mean_net only.
5. Do not recommend directly combining every good direction. The next step should test one role family at a time in candidate backtest.
"""
    (out_dir / "GPT_REVIEW_PROMPT.md").write_text(review_prompt, encoding="utf-8")
    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    print(f"[run] {SCRIPT_NAME} v{SCRIPT_VERSION}", flush=True)
    print(f"[args] out_dir={out_dir}", flush=True)
    print("[scope] lower shadow long + mid_high/extreme vol; direction scorecards; no combined filters", flush=True)

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
        write_reports(out_dir=out_dir, args=args, events=events, labels=pd.DataFrame(), bucket_specs=[], direction_specs=[])
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
    direction_specs = build_direction_specs(labels) if not labels.empty else []
    print(f"[buckets] specs={len(bucket_specs):,}", flush=True)
    print(f"[directions] specs={len(direction_specs):,}", flush=True)
    write_reports(out_dir=out_dir, args=args, events=events, labels=labels, bucket_specs=bucket_specs, direction_specs=direction_specs)
    print("[done] lower-shadow vol-regime direction scorecard probe complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
