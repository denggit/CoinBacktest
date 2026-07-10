#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH 1m panic-wick delayed reversal research V2.

This is the follow-up to V1 volume-shadow research after V1 rejected the broad
"long lower wick / short upper wick" idea. V2 intentionally narrows the scope:

    extreme or mid-high volatility selloff + long lower shadow -> long-only
    delayed confirmation that price stopped making fresh lows -> next-open entry

The script does not run a Cartesian parameter search. It tests a small fixed set
of structural hypotheses that came from the V1 review: panic flush, close reclaim,
negative-flow absorption, next-bar hold, 2-bar hold, 5-bar stabilization, controlled
break-and-reclaim, and 15m stabilization.

Causal policy
-------------
- 1m bars are left-labeled by bar start time.
- Any confirmation bar must be fully closed before the signal is considered known.
- signal_time = signal_bar_start + 1 minute.
- entry_time = next bar open after signal_time.
- For delayed confirmation events, entry_offset is larger than one bar by design;
  this is causal because the signal_bar_pos is shifted forward to the last
  confirmation bar.
- No multi-timeframe context is used.
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

SCRIPT_NAME = "eth_1m_panic_wick_reversal_v2_research"
SCRIPT_VERSION = "2.0.1"
EXPERIMENT_ID = "ETH_MF_1M_PANIC_WICK_REVERSAL_V2"
EDGE_ID = "RESEARCH_ONLY_ETH_MF_1M_PANIC_WICK_REVERSAL_V2"
TITLE = "ETH 1m Panic Wick Delayed Reversal Research V2"
DEFAULT_OUT_DIR = "data/reports/research/eth_1m_panic_wick_reversal_v2"
BAR_DELTA = pd.Timedelta(minutes=1)


@dataclass(frozen=True)
class PanicThresholds:
    """Fixed thresholds for V2 event vocabulary; not swept as a grid."""

    wick_share_min: float = 0.50
    wick_atr_min: float = 0.55
    volume_ratio_min: float = 2.0
    reclaim_close_pos: float = 0.66
    soft_reclaim_close_pos: float = 0.55
    prior_flush_30_min: float = -0.005
    prior_flush_120_min: float = -0.010
    delta_absorption_max: float = -0.10
    taker_buy_absorption_max: float = 0.45
    hold_break_buffer_pct: float = 0.0015
    deep_break_buffer_pct: float = 0.0030


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Research-only ETH 1m panic lower-wick delayed reversal event study V2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    p.add_argument("--data-dir", default=None, help="Optional data directory passed to OKXTradeBarLoader.")
    p.add_argument("--db-name", default="okx_trade_bars.db")
    p.add_argument("--chunksize", type=int, default=300_000)
    p.add_argument("--no-build-missing", action="store_true", help="Only read already cached trade bars.")
    p.add_argument("--force-rebuild", action="store_true", help="Pass force_rebuild to OKXTradeBarLoader.")

    # Fixed study settings, not a sweep. The script only evaluates the supplied
    # single vocabulary once.
    p.add_argument("--horizons", default="5,15,30,60,120,240")
    p.add_argument("--primary-horizon", type=int, default=60)
    p.add_argument("--mfe-mae-horizon", type=int, default=240)
    p.add_argument("--round-trip-cost-pct", type=float, default=0.0011)
    p.add_argument("--cost-multipliers", default="1.0,1.5,2.0,3.0")
    p.add_argument("--delay-bars-list", default="0,1,2,3", help="Extra bars beyond the event's planned causal entry.")
    p.add_argument("--min-count", type=int, default=120)
    p.add_argument("--event-sample-size", type=int, default=5000)
    p.add_argument("--write-full-events", action="store_true")
    p.add_argument("--progress-every", type=int, default=2500)

    # Few fixed structure knobs for reproducibility only. Do not sweep them.
    p.add_argument("--wick-share-min", type=float, default=0.50)
    p.add_argument("--wick-atr-min", type=float, default=0.55)
    p.add_argument("--volume-ratio-min", type=float, default=2.0)
    p.add_argument("--reclaim-close-pos", type=float, default=0.66)
    p.add_argument("--soft-reclaim-close-pos", type=float, default=0.55)
    p.add_argument("--prior-flush-30-min", type=float, default=-0.005)
    p.add_argument("--prior-flush-120-min", type=float, default=-0.010)
    p.add_argument("--delta-absorption-max", type=float, default=-0.10)
    p.add_argument("--taker-buy-absorption-max", type=float, default=0.45)
    p.add_argument("--hold-break-buffer-pct", type=float, default=0.0015)
    p.add_argument("--deep-break-buffer-pct", type=float, default=0.0030)
    return p.parse_args(argv)


def _parse_csv_ints(raw: str) -> tuple[int, ...]:
    vals: list[int] = []
    for part in str(raw).split(","):
        text = part.strip()
        if text:
            vals.append(int(text))
    out = tuple(dict.fromkeys(vals))
    if not out:
        raise ValueError("integer csv must not be empty")
    return out


def _parse_csv_floats(raw: str) -> tuple[float, ...]:
    vals: list[float] = []
    for part in str(raw).split(","):
        text = part.strip()
        if text:
            vals.append(float(text))
    out = tuple(dict.fromkeys(vals))
    if not out:
        raise ValueError("float csv must not be empty")
    return out


def _thresholds_from_args(args: argparse.Namespace) -> PanicThresholds:
    return PanicThresholds(
        wick_share_min=float(args.wick_share_min),
        wick_atr_min=float(args.wick_atr_min),
        volume_ratio_min=float(args.volume_ratio_min),
        reclaim_close_pos=float(args.reclaim_close_pos),
        soft_reclaim_close_pos=float(args.soft_reclaim_close_pos),
        prior_flush_30_min=float(args.prior_flush_30_min),
        prior_flush_120_min=float(args.prior_flush_120_min),
        delta_absorption_max=float(args.delta_absorption_max),
        taker_buy_absorption_max=float(args.taker_buy_absorption_max),
        hold_break_buffer_pct=float(args.hold_break_buffer_pct),
        deep_break_buffer_pct=float(args.deep_break_buffer_pct),
    )


def _research_window_mask(index: pd.DatetimeIndex, start_date: str, end_date: str) -> pd.Series:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    # A date-only end such as 2026-06-30 should include the whole local day.
    # Use half-open filtering to avoid accidentally dropping 23:59 bars.
    if end_ts == end_ts.normalize() and len(str(end_date).strip()) <= 10:
        end_exclusive = end_ts + pd.Timedelta(days=1)
    else:
        end_exclusive = end_ts + BAR_DELTA
    return pd.Series((index >= start_ts) & (index < end_exclusive), index=index)


# ---------------------------------------------------------------------------
# Data / features
# ---------------------------------------------------------------------------


def _safe_divide(a: pd.Series | np.ndarray, b: pd.Series | np.ndarray) -> pd.Series:
    aa = pd.Series(a) if not isinstance(a, pd.Series) else pd.to_numeric(a, errors="coerce")
    bb = pd.Series(b) if not isinstance(b, pd.Series) else pd.to_numeric(b, errors="coerce")
    out = aa / bb.replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


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


def _future_window_extreme(values: pd.Series, *, start_offset: int, end_offset: int, op: str) -> pd.Series:
    if end_offset < start_offset:
        return pd.Series(np.nan, index=values.index)
    shifted = pd.to_numeric(values, errors="coerce").shift(-int(start_offset))
    window = int(end_offset - start_offset + 1)
    rev = shifted.iloc[::-1]
    if op == "max":
        out = rev.rolling(window, min_periods=1).max().iloc[::-1]
    elif op == "min":
        out = rev.rolling(window, min_periods=1).min().iloc[::-1]
    else:
        raise ValueError("op must be max or min")
    return out.reindex(values.index)


def _future_value(values: pd.Series, *, offset: int) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").shift(-int(offset))


def load_trade_bars(args: argparse.Namespace) -> pd.DataFrame:
    print(
        f"[load] OKXTradeBarLoader {args.symbol} {args.timeframe} "
        f"{args.warmup_start_date}->{args.end_date}",
        flush=True,
    )
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
        raise RuntimeError(f"Loaded trade bars missing required columns: {missing}")
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=required)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    print(f"       rows={len(out):,} range={out.index[0]} -> {out.index[-1]}", flush=True)
    return out


def session_label(index: pd.DatetimeIndex) -> pd.Series:
    """UTC+8 style sessions because project trade bars align with OKX loader timezone."""
    hour = pd.Series(index.hour, index=index)
    labels = np.select(
        [
            (hour >= 0) & (hour < 8),
            (hour >= 8) & (hour < 16),
            (hour >= 16) & (hour < 24),
        ],
        ["asia_early", "asia_day", "eu_us"],
        default="unknown",
    )
    return pd.Series(labels, index=index, dtype="object")


def build_features(bars: pd.DataFrame, th: PanicThresholds) -> pd.DataFrame:
    """Build closed-bar/past-only features and future confirmation helpers.

    Future confirmation helper columns are only used by delayed event definitions
    that explicitly move signal_bar_pos to the last confirmation bar. They are
    never used to enter before the confirmation is closed.
    """
    print("[features] building panic wick geometry, flow, regime and delayed-confirm helpers", flush=True)
    df = bars.copy().sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    body_low = df[["open", "close"]].min(axis=1)
    body_high = df[["open", "close"]].max(axis=1)
    df["range_pct"] = _safe_divide(rng, df["close"]).to_numpy()
    df["body_pct"] = (df["close"] / df["open"] - 1.0).replace([np.inf, -np.inf], np.nan)
    df["close_pos"] = _safe_divide(df["close"] - df["low"], rng).to_numpy()
    df["bar_mid"] = (df["high"] + df["low"]) / 2.0
    df["lower_wick"] = (body_low - df["low"]).clip(lower=0.0)
    df["upper_wick"] = (df["high"] - body_high).clip(lower=0.0)
    df["lower_wick_share"] = _safe_divide(df["lower_wick"], rng).to_numpy()
    df["upper_wick_share"] = _safe_divide(df["upper_wick"], rng).to_numpy()

    df["tr"] = _true_range(df)
    df["atr"] = df["tr"].rolling(60, min_periods=30).mean()
    df["atr_pct"] = _safe_divide(df["atr"], df["close"]).to_numpy()
    df["lower_wick_atr"] = _safe_divide(df["lower_wick"], df["atr"]).to_numpy()

    vol_base = df["volume"].shift(1).rolling(240, min_periods=60).median()
    df["volume_ratio"] = _safe_divide(df["volume"], vol_base).to_numpy()
    df["volume_climax"] = df["volume_ratio"] >= th.volume_ratio_min

    df["ret_5"] = df["close"].pct_change(5)
    df["ret_15"] = df["close"].pct_change(15)
    df["ret_30"] = df["close"].pct_change(30)
    df["ret_120"] = df["close"].pct_change(120)
    df["ret_240"] = df["close"].pct_change(240)
    df["ema_60"] = df["close"].ewm(span=60, adjust=False, min_periods=60).mean()
    df["ema_240"] = df["close"].ewm(span=240, adjust=False, min_periods=240).mean()
    df["ema240_slope_60"] = df["ema_240"] / df["ema_240"].shift(60) - 1.0
    df["trend_below_ema240"] = df["close"] < df["ema_240"]

    delta = pd.to_numeric(df.get("delta_notional", np.nan), errors="coerce")
    notional = pd.to_numeric(df.get("notional", np.nan), errors="coerce").abs()
    df["delta_ratio"] = _safe_divide(delta, notional).to_numpy()
    if "taker_buy_ratio" in df.columns:
        df["taker_buy_ratio"] = pd.to_numeric(df["taker_buy_ratio"], errors="coerce")
    else:
        buy_volume = pd.to_numeric(df.get("buy_volume", np.nan), errors="coerce")
        df["taker_buy_ratio"] = _safe_divide(buy_volume, df["volume"]).to_numpy()

    df["long_lower_wick"] = (df["lower_wick_share"] >= th.wick_share_min) & (df["lower_wick_atr"] >= th.wick_atr_min)
    df["long_upper_wick"] = (df["upper_wick_share"] >= th.wick_share_min)
    df["lower_volume_shadow"] = df["long_lower_wick"] & df["volume_climax"]
    df["two_sided_shadow"] = df["lower_volume_shadow"] & df["long_upper_wick"]

    df["session"] = session_label(df.index)
    df["vol_regime"] = pd.cut(
        df["atr_pct"],
        bins=[-np.inf, 0.0015, 0.0030, 0.0050, np.inf],
        labels=["very_low_vol", "low_mid_vol", "mid_high_vol", "extreme_vol"],
    ).astype("object").fillna("NA")
    df["trend_regime"] = np.select(
        [
            (df["close"] > df["ema_240"]) & (df["ema240_slope_60"] > 0.0005),
            (df["close"] < df["ema_240"]) & (df["ema240_slope_60"] < -0.0005),
        ],
        ["uptrend", "downtrend"],
        default="range_or_transition",
    )

    high_vol = df["vol_regime"].isin(["mid_high_vol", "extreme_vol"])
    prior_flush = (df["ret_30"] <= th.prior_flush_30_min) | (df["ret_120"] <= th.prior_flush_120_min)
    df["panic_context"] = high_vol & prior_flush
    df["panic_downtrend_context"] = df["panic_context"] & (df["trend_regime"] == "downtrend")
    df["flow_absorption"] = (df["delta_ratio"] <= th.delta_absorption_max) | (df["taker_buy_ratio"] <= th.taker_buy_absorption_max)

    # Delayed confirmation helpers. These use future bars but the event definitions
    # below shift signal_bar_pos to the final confirmation bar, so entry remains causal.
    for n in (1, 2, 5, 15):
        df[f"fwd{n}_min_low"] = _future_window_extreme(df["low"], start_offset=1, end_offset=n, op="min")
        df[f"fwd{n}_max_high"] = _future_window_extreme(df["high"], start_offset=1, end_offset=n, op="max")
        df[f"close_plus_{n}"] = _future_value(df["close"], offset=n)
        df[f"low_plus_{n}"] = _future_value(df["low"], offset=n)
    return df


# ---------------------------------------------------------------------------
# Event construction
# ---------------------------------------------------------------------------


def _event_frame_from_mask(
    features: pd.DataFrame,
    *,
    mask: pd.Series,
    event_name: str,
    family: str,
    structure: str,
    confirm_bars: int,
) -> pd.DataFrame:
    bool_mask = mask.fillna(False).to_numpy(dtype=bool)
    idx = features.index[bool_mask]
    if len(idx) == 0:
        return pd.DataFrame()
    f = features.loc[idx]
    pos_map = pd.Series(np.arange(len(features), dtype=int), index=features.index)
    event_pos = pos_map.reindex(idx).to_numpy(dtype=int)
    signal_pos = event_pos + int(confirm_bars)
    valid = signal_pos < len(features)
    if not valid.any():
        return pd.DataFrame()
    idx = idx[valid]
    f = f.iloc[np.flatnonzero(valid)]
    event_pos = event_pos[valid]
    signal_pos = signal_pos[valid]
    signal_bar_time = features.index[signal_pos]
    planned_entry_offset = int(confirm_bars) + 1

    out = pd.DataFrame(
        {
            "event_name": event_name,
            "family": family,
            "direction": "LONG",
            "side": 1,
            "structure": structure,
            "event_bar_time": idx,
            "event_bar_pos": event_pos,
            "signal_bar_time": signal_bar_time,
            "signal_bar_pos": signal_pos,
            "confirmation_bars": int(confirm_bars),
            "planned_entry_offset": planned_entry_offset,
            "signal_time": signal_bar_time + BAR_DELTA,
            "signal_available_time": signal_bar_time + BAR_DELTA,
            "session": f["session"].to_numpy(),
            "vol_regime": f["vol_regime"].astype(str).to_numpy(),
            "trend_regime": f["trend_regime"].astype(str).to_numpy(),
            "close_pos": f["close_pos"].to_numpy(dtype=float),
            "volume_ratio": f["volume_ratio"].to_numpy(dtype=float),
            "lower_wick_share": f["lower_wick_share"].to_numpy(dtype=float),
            "lower_wick_atr": f["lower_wick_atr"].to_numpy(dtype=float),
            "ret_30": f["ret_30"].to_numpy(dtype=float),
            "ret_120": f["ret_120"].to_numpy(dtype=float),
            "delta_ratio": f["delta_ratio"].to_numpy(dtype=float),
            "taker_buy_ratio": f["taker_buy_ratio"].to_numpy(dtype=float),
            "panic_context": f["panic_context"].astype(bool).to_numpy(),
            "panic_downtrend_context": f["panic_downtrend_context"].astype(bool).to_numpy(),
            "flow_absorption": f["flow_absorption"].astype(bool).to_numpy(),
            "event_low": f["low"].to_numpy(dtype=float),
            "event_high": f["high"].to_numpy(dtype=float),
            "event_close": f["close"].to_numpy(dtype=float),
        }
    )
    return out


def build_panic_wick_events(features: pd.DataFrame, th: PanicThresholds) -> pd.DataFrame:
    """Build fixed V2 long-only panic wick hypotheses, not a parameter grid."""
    print("[events] building V2 fixed panic-wick delayed confirmation hypotheses", flush=True)
    f = features
    lower = f["lower_volume_shadow"] & ~f["two_sided_shadow"]
    panic = lower & f["panic_context"]
    panic_downtrend = lower & f["panic_downtrend_context"]
    reclaim = f["close_pos"] >= th.reclaim_close_pos
    soft_reclaim = f["close_pos"] >= th.soft_reclaim_close_pos
    negative_flow = f["flow_absorption"]

    hold1 = (f["fwd1_min_low"] >= f["low"] * (1.0 - th.hold_break_buffer_pct)) & (f["close_plus_1"] >= f["bar_mid"])
    hold2 = (f["fwd2_min_low"] >= f["low"] * (1.0 - th.hold_break_buffer_pct)) & (f["close_plus_2"] >= f["close"])
    hold5 = (f["fwd5_min_low"] >= f["low"] * (1.0 - th.deep_break_buffer_pct)) & (f["close_plus_5"] >= f["close"])
    stabilize15 = (f["fwd15_min_low"] >= f["low"] * (1.0 - th.deep_break_buffer_pct)) & (f["close_plus_15"] >= f["close"])
    controlled_break_reclaim = (
        (f["fwd2_min_low"] < f["low"] * (1.0 - th.hold_break_buffer_pct))
        & (f["fwd2_min_low"] >= f["low"] * (1.0 - th.deep_break_buffer_pct))
        & (f["close_plus_2"] >= f["close"])
    )

    # name, family, structure, confirm_bars, mask
    specs: list[tuple[str, str, str, int, pd.Series]] = [
        (
            "panic_lower_reclaim_long",
            "panic_flush_reclaim",
            "high_vol_prior_flush_lower_wick_close_reclaim",
            0,
            panic & reclaim,
        ),
        (
            "panic_downtrend_lower_reclaim_long",
            "panic_downtrend_reclaim",
            "downtrend_high_vol_prior_flush_lower_wick_close_reclaim",
            0,
            panic_downtrend & reclaim,
        ),
        (
            "panic_neg_delta_absorption_long",
            "panic_flow_absorption",
            "panic_lower_wick_soft_reclaim_negative_delta_or_low_taker_buy",
            0,
            panic & soft_reclaim & negative_flow,
        ),
        (
            "panic_downtrend_neg_delta_absorption_long",
            "panic_downtrend_flow_absorption",
            "downtrend_panic_lower_wick_soft_reclaim_negative_delta_or_low_taker_buy",
            0,
            panic_downtrend & soft_reclaim & negative_flow,
        ),
        (
            "panic_next1_hold_long",
            "delayed_no_followthrough",
            "panic_lower_wick_next_1_bar_no_lower_low_and_close_above_mid",
            1,
            panic_downtrend & soft_reclaim & hold1,
        ),
        (
            "panic_next2_hold_long",
            "delayed_no_followthrough",
            "panic_lower_wick_next_2_bars_no_lower_low_and_close_above_event_close",
            2,
            panic_downtrend & soft_reclaim & hold2,
        ),
        (
            "panic_next5_stabilize_long",
            "delayed_stabilization",
            "panic_lower_wick_next_5_bars_no_deep_break_and_close_above_event_close",
            5,
            panic_downtrend & soft_reclaim & hold5,
        ),
        (
            "panic_controlled_break_reclaim_2bar_long",
            "controlled_break_reclaim",
            "panic_lower_wick_small_new_low_then_reclaim_by_second_bar",
            2,
            panic_downtrend & soft_reclaim & controlled_break_reclaim,
        ),
        (
            "panic_15m_stabilize_long",
            "delayed_stabilization",
            "panic_lower_wick_next_15_bars_no_deep_break_and_close_above_event_close",
            15,
            panic_downtrend & soft_reclaim & stabilize15,
        ),
    ]

    frames = [
        _event_frame_from_mask(f, mask=mask, event_name=name, family=family, structure=structure, confirm_bars=confirm_bars)
        for name, family, structure, confirm_bars, mask in specs
    ]
    out = pd.concat([x for x in frames if not x.empty], ignore_index=True) if any(not x.empty for x in frames) else pd.DataFrame()
    if out.empty:
        return out
    out = out.sort_values(["signal_bar_time", "event_name"]).reset_index(drop=True)
    out["event_id"] = np.arange(len(out), dtype=int)
    return out


# ---------------------------------------------------------------------------
# Forward returns / reports
# ---------------------------------------------------------------------------


def _profit_factor(ret: pd.Series) -> float:
    x = pd.to_numeric(ret, errors="coerce").dropna()
    if x.empty:
        return float("nan")
    gp = float(x[x > 0].sum())
    gl = float(-x[x <= 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else float("nan")
    return gp / gl


def _top5_winner_share(ret: pd.Series) -> float:
    x = pd.to_numeric(ret, errors="coerce").dropna()
    wins = x[x > 0].sort_values(ascending=False)
    gross = float(wins.sum())
    if gross <= 0 or wins.empty:
        return float("nan")
    return float(wins.head(5).sum()) / gross


def _max_days_without_event(times: pd.Series) -> float:
    ts = pd.to_datetime(times, errors="coerce").dropna().sort_values()
    if len(ts) < 2:
        return float("nan")
    gaps = ts.diff().dropna()
    if gaps.empty:
        return float("nan")
    return float(gaps.max() / pd.Timedelta(days=1))


def _events_per_month(times: pd.Series) -> float:
    ts = pd.to_datetime(times, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    span_days = max(1.0, float((ts.max() - ts.min()) / pd.Timedelta(days=1)))
    return float(len(ts) / span_days * 30.4375)


def _cost_tag(mult: float) -> str:
    return str(float(mult)).rstrip("0").rstrip(".").replace(".", "p")


def attach_forward_returns(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    cost_multipliers: tuple[float, ...],
    delay_bars_list: tuple[int, ...],
    round_trip_cost_pct: float,
    mfe_mae_horizon: int,
    progress_every: int,
) -> pd.DataFrame:
    """Attach forward labels with event-specific causal entry offsets."""
    print("[forward] attaching event-specific causal entry returns, cost, delay and MFE/MAE", flush=True)
    if events.empty:
        return events.copy()

    frame = bars.copy().sort_index()
    out = events.copy()
    out["event_bar_pos"] = pd.to_numeric(out["event_bar_pos"], errors="coerce").astype("Int64")
    out["signal_bar_pos"] = pd.to_numeric(out["signal_bar_pos"], errors="coerce").astype("Int64")
    out["planned_entry_offset"] = pd.to_numeric(out["planned_entry_offset"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["event_bar_pos", "signal_bar_pos", "planned_entry_offset"]).copy()
    out["event_bar_pos"] = out["event_bar_pos"].astype(int)
    out["signal_bar_pos"] = out["signal_bar_pos"].astype(int)
    out["planned_entry_offset"] = out["planned_entry_offset"].astype(int)
    out["entry_bar_pos"] = out["signal_bar_pos"] + 1

    valid_entry = out["entry_bar_pos"].between(0, len(frame) - 1)
    out["entry_time"] = pd.NaT
    out["entry_price"] = np.nan
    if valid_entry.any():
        entry_pos = out.loc[valid_entry, "entry_bar_pos"].astype(int).to_numpy()
        out.loc[valid_entry, "entry_time"] = frame.index[entry_pos]
        out.loc[valid_entry, "entry_price"] = frame["open"].iloc[entry_pos].to_numpy(dtype=float)
    out["expected_entry_time"] = pd.to_datetime(out["signal_bar_time"]) + BAR_DELTA
    out["expected_entry_price"] = out["entry_price"]

    open_arr = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
    close_arr = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")

    event_pos = out["event_bar_pos"].to_numpy(dtype=int)
    planned_offsets = out["planned_entry_offset"].to_numpy(dtype=int)
    sides = out["side"].to_numpy(dtype=float)
    total = len(horizons) * len(delay_bars_list)
    progress = ProgressReporter(label="[forward] label grid", total=total, every=max(1, min(int(progress_every), total)))
    done = 0
    n_bars = len(frame)
    for extra_delay in delay_bars_list:
        delay_i = int(extra_delay)
        entry_positions = event_pos + planned_offsets + delay_i
        entry_vals = np.full(len(out), np.nan, dtype=float)
        valid_e = (entry_positions >= 0) & (entry_positions < n_bars)
        entry_vals[valid_e] = open_arr[entry_positions[valid_e]]
        for horizon in horizons:
            h = int(horizon)
            exit_positions = event_pos + planned_offsets + delay_i + h - 1
            exit_vals = np.full(len(out), np.nan, dtype=float)
            valid_x = (exit_positions >= 0) & (exit_positions < n_bars)
            exit_vals[valid_x] = close_arr[exit_positions[valid_x]]
            gross = (exit_vals / entry_vals - 1.0) * sides
            invalid = (~np.isfinite(entry_vals)) | (entry_vals <= 0) | (~np.isfinite(exit_vals))
            gross[invalid] = np.nan
            gross_col = f"ret_h{h}_d{delay_i}_gross"
            out[gross_col] = gross
            for mult in cost_multipliers:
                net_col = f"ret_h{h}_d{delay_i}_cost{_cost_tag(mult)}_net"
                out[net_col] = gross - float(round_trip_cost_pct) * float(mult)
            done += 1
            progress.update(done)
    progress.close()

    h = int(mfe_mae_horizon)
    mfe = np.full(len(out), np.nan, dtype=float)
    mae = np.full(len(out), np.nan, dtype=float)
    for offset in sorted(set(int(x) for x in planned_offsets)):
        mask = planned_offsets == offset
        if not mask.any():
            continue
        future_max = _future_window_extreme(high, start_offset=offset, end_offset=offset + h - 1, op="max")
        future_min = _future_window_extreme(low, start_offset=offset, end_offset=offset + h - 1, op="min")
        entry_series = pd.Series(open_arr, index=frame.index).shift(-offset)
        e = entry_series.iloc[event_pos[mask]].to_numpy(dtype=float)
        fmax = future_max.iloc[event_pos[mask]].to_numpy(dtype=float)
        fmin = future_min.iloc[event_pos[mask]].to_numpy(dtype=float)
        local_mfe = fmax / e - 1.0
        local_mae = fmin / e - 1.0
        bad = (~np.isfinite(e)) | (e <= 0) | (~np.isfinite(fmax)) | (~np.isfinite(fmin))
        local_mfe[bad] = np.nan
        local_mae[bad] = np.nan
        mfe[np.flatnonzero(mask)] = local_mfe
        mae[np.flatnonzero(mask)] = local_mae
    out[f"mfe_h{h}"] = mfe
    out[f"mae_h{h}"] = mae
    return out


def build_causal_audit(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    cols = [
        "event_id",
        "event_name",
        "family",
        "direction",
        "event_bar_time",
        "signal_bar_time",
        "signal_time",
        "signal_available_time",
        "confirmation_bars",
        "planned_entry_offset",
        "entry_time",
        "entry_price",
        "expected_entry_time",
        "expected_entry_price",
    ]
    out = events[[c for c in cols if c in events.columns]].copy()
    out["used_context_timestamp"] = pd.NaT
    out["used_context_available_time"] = pd.NaT
    out["context_available_time_flag"] = False
    out["entry_not_next_open_flag"] = pd.to_datetime(out["entry_time"]) != pd.to_datetime(out["expected_entry_time"])
    out["entry_price_mismatch_flag"] = False
    out["lookahead_flag"] = out["context_available_time_flag"] | out["entry_not_next_open_flag"]
    out["audit_notes"] = "event/confirmation bars are closed before next-open entry; no MTF context"
    return out


def _summary_row(part: pd.DataFrame, *, ret_col: str, group: dict[str, object], min_count: int) -> dict[str, object]:
    x = pd.to_numeric(part[ret_col], errors="coerce").dropna()
    return {
        **group,
        "ret_col": ret_col,
        "count": int(len(x)),
        "eligible": bool(len(x) >= int(min_count)),
        "mean_net": float(x.mean()) if len(x) else np.nan,
        "median_net": float(x.median()) if len(x) else np.nan,
        "win_rate": float((x > 0).mean()) if len(x) else np.nan,
        "profit_factor": _profit_factor(x),
        "top5_winner_share": _top5_winner_share(x),
    }


def summarize_groups(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    ret_cols: list[str],
    min_count: int,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, part in df.groupby(group_cols, dropna=False, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group = dict(zip(group_cols, keys))
        for ret_col in ret_cols:
            if ret_col in part.columns:
                rows.append(_summary_row(part, ret_col=ret_col, group=group, min_count=min_count))
    return pd.DataFrame(rows)


def build_context_baseline(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    cost_multipliers: tuple[float, ...],
    delay_bars_list: tuple[int, ...],
    round_trip_cost_pct: float,
    entry_offsets: tuple[int, ...],
) -> pd.DataFrame:
    """Build matched controls from panic context bars without lower-shadow events.

    Controls match the same broad panic regime but remove the actual lower-volume-shadow
    event. Delayed-entry controls use the same planned entry offset as each event family.
    """
    print("[baseline] building matched panic-context controls without lower-shadow signal", flush=True)
    if frame.empty:
        return pd.DataFrame()
    control_mask = frame["panic_context"].fillna(False) & ~frame["lower_volume_shadow"].fillna(False) & ~frame["two_sided_shadow"].fillna(False)
    control = frame.loc[control_mask].copy()
    if control.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    for offset in sorted(set(int(x) for x in entry_offsets)):
        base = pd.DataFrame(
            {
                "control_bar_time": control.index,
                "control_bar_pos": np.arange(len(frame), dtype=int)[control_mask.to_numpy(dtype=bool)],
                "planned_entry_offset": int(offset),
                "direction": "LONG",
                "side": 1,
                "year": control.index.year,
                "month": control.index.to_period("M").astype(str),
                "session": control["session"].to_numpy(),
                "vol_regime": control["vol_regime"].astype(str).to_numpy(),
                "trend_regime": control["trend_regime"].astype(str).to_numpy(),
                "panic_downtrend_context": control["panic_downtrend_context"].astype(bool).to_numpy(),
            }
        )
        rows.append(base)
    baseline = pd.concat(rows, ignore_index=True)
    open_arr = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
    close_arr = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    n_bars = len(frame)
    control_pos = baseline["control_bar_pos"].to_numpy(dtype=int)
    offsets = baseline["planned_entry_offset"].to_numpy(dtype=int)
    sides = baseline["side"].to_numpy(dtype=float)
    for extra_delay in delay_bars_list:
        delay_i = int(extra_delay)
        entry_pos = control_pos + offsets + delay_i
        entry_vals = np.full(len(baseline), np.nan, dtype=float)
        valid_e = (entry_pos >= 0) & (entry_pos < n_bars)
        entry_vals[valid_e] = open_arr[entry_pos[valid_e]]
        for h in horizons:
            h_i = int(h)
            exit_pos = control_pos + offsets + delay_i + h_i - 1
            exit_vals = np.full(len(baseline), np.nan, dtype=float)
            valid_x = (exit_pos >= 0) & (exit_pos < n_bars)
            exit_vals[valid_x] = close_arr[exit_pos[valid_x]]
            gross = (exit_vals / entry_vals - 1.0) * sides
            bad = (~np.isfinite(entry_vals)) | (entry_vals <= 0) | (~np.isfinite(exit_vals))
            gross[bad] = np.nan
            for mult in cost_multipliers:
                baseline[f"baseline_ret_h{h_i}_d{delay_i}_cost{_cost_tag(mult)}_net"] = gross - float(round_trip_cost_pct) * float(mult)
    return baseline


def matched_baseline_summary(
    events: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    primary_horizon: int,
    min_count: int,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    event_ret_col = f"ret_h{int(primary_horizon)}_d0_cost1_net"
    base_ret_col = f"baseline_ret_h{int(primary_horizon)}_d0_cost1_net"
    match_cols = ["year", "session", "vol_regime", "trend_regime", "planned_entry_offset"]
    if baseline.empty or base_ret_col not in baseline.columns:
        return pd.DataFrame(
            {
                "event_name": sorted(events["event_name"].dropna().unique()),
                "count": 0,
                "eligible": False,
                "event_mean_net": np.nan,
                "matched_baseline_mean_net": np.nan,
                "excess_mean_net": np.nan,
                "event_profit_factor": np.nan,
                "baseline_sample_count": 0,
            }
        )
    base_stats = baseline.groupby(match_cols, dropna=False, observed=False)[base_ret_col].agg(["mean", "count"]).reset_index()
    base_stats = base_stats.rename(columns={"mean": "bucket_baseline_mean_net", "count": "baseline_sample_count"})

    rows: list[dict[str, object]] = []
    for event_name, part in events.groupby("event_name", dropna=False):
        p = part.copy()
        p["year"] = pd.to_datetime(p["signal_time"]).dt.year
        event_bucket_counts = p.groupby(match_cols, dropna=False, observed=False).size().reset_index(name="event_bucket_count")
        merged = event_bucket_counts.merge(base_stats, on=match_cols, how="left")
        weighted_baseline = np.nan
        baseline_count = 0
        if not merged.empty:
            valid = merged["bucket_baseline_mean_net"].notna()
            if valid.any():
                weights = merged.loc[valid, "event_bucket_count"]
                weighted_baseline = float((merged.loc[valid, "bucket_baseline_mean_net"] * weights).sum() / weights.sum())
                baseline_count = int(merged.loc[valid, "baseline_sample_count"].sum())
        x = pd.to_numeric(p[event_ret_col], errors="coerce").dropna()
        event_mean = float(x.mean()) if len(x) else np.nan
        rows.append(
            {
                "event_name": event_name,
                "count": int(len(x)),
                "eligible": bool(len(x) >= int(min_count)),
                "event_mean_net": event_mean,
                "matched_baseline_mean_net": weighted_baseline,
                "excess_mean_net": event_mean - weighted_baseline if np.isfinite(event_mean) and np.isfinite(weighted_baseline) else np.nan,
                "event_profit_factor": _profit_factor(x),
                "baseline_sample_count": baseline_count,
            }
        )
    return pd.DataFrame(rows).sort_values(["excess_mean_net", "event_mean_net"], ascending=False, na_position="last")


def _best_bucket(part: pd.DataFrame, *, bucket: str, ret_col: str) -> str:
    if bucket not in part.columns or ret_col not in part.columns:
        return "NA"
    rows = []
    for val, bp in part.groupby(bucket, dropna=False, observed=False):
        x = pd.to_numeric(bp[ret_col], errors="coerce").dropna()
        if len(x) >= 20:
            rows.append((str(val), float(x.mean()), int(len(x))))
    if not rows:
        return "NA"
    rows.sort(key=lambda x: x[1], reverse=True)
    return f"{rows[0][0]}|mean={rows[0][1]:.6f}|n={rows[0][2]}"


def _candidate_decision(
    *,
    count: int,
    min_count: int,
    mean_net: float,
    pf: float,
    fee2_mean: float,
    delay1_mean: float,
    positive_years: int,
    year_count: int,
    top5_share: float,
    excess_mean: float,
    max_days_without_event: float,
) -> tuple[str, str]:
    fails = []
    if count < min_count:
        fails.append("count_below_min")
    if not np.isfinite(mean_net) or mean_net <= 0:
        fails.append("mean_net_not_positive")
    if not np.isfinite(pf) or pf < 1.20:
        fails.append("pf_below_1p20")
    if not np.isfinite(fee2_mean) or fee2_mean <= 0:
        fails.append("fee2_not_positive")
    if not np.isfinite(delay1_mean) or delay1_mean <= 0:
        fails.append("delay1_not_positive")
    if year_count >= 3 and positive_years < 3:
        fails.append("positive_years_below_3")
    if np.isfinite(top5_share) and top5_share > 0.45:
        fails.append("top5_winner_dependency_high")
    if not np.isfinite(excess_mean) or excess_mean <= 0:
        fails.append("matched_baseline_excess_not_positive")
    if np.isfinite(max_days_without_event) and max_days_without_event > 45:
        fails.append("max_event_gap_too_long_for_mf")
    if fails:
        # V2 can continue only when the shape is strong but robustness still needs
        # another pass. This prevents promoting thin post-hoc slices.
        soft_fails = {"fee2_not_positive", "delay1_not_positive", "positive_years_below_3", "top5_winner_dependency_high", "max_event_gap_too_long_for_mf"}
        if (
            count >= min_count
            and np.isfinite(mean_net)
            and mean_net > 0
            and np.isfinite(pf)
            and pf >= 1.10
            and np.isfinite(excess_mean)
            and excess_mean > 0
            and any(f in soft_fails for f in fails)
        ):
            return "research_continue", ";".join(fails)
        return "rejected", ";".join(fails)
    return "promote_to_backtest_candidate", "passes_event_study_gate_only;requires_formal_backtest_and_portfolio_overlay"


def build_candidate_shortlist(
    events: pd.DataFrame,
    matched: pd.DataFrame,
    *,
    primary_horizon: int,
    min_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    base_col = f"ret_h{int(primary_horizon)}_d0_cost1_net"
    fee2_col = f"ret_h{int(primary_horizon)}_d0_cost2_net"
    delay1_col = f"ret_h{int(primary_horizon)}_d1_cost1_net"
    mfe_cols = [c for c in events.columns if c.startswith("mfe_h")]
    mae_cols = [c for c in events.columns if c.startswith("mae_h")]
    mfe_col = mfe_cols[0] if mfe_cols else None
    mae_col = mae_cols[0] if mae_cols else None
    matched_idx = matched.set_index("event_name") if not matched.empty and "event_name" in matched.columns else pd.DataFrame()

    rows: list[dict[str, object]] = []
    for event_name, part in events.groupby("event_name", dropna=False):
        ret = pd.to_numeric(part[base_col], errors="coerce").dropna()
        fee2 = pd.to_numeric(part.get(fee2_col, np.nan), errors="coerce").dropna()
        delay1 = pd.to_numeric(part.get(delay1_col, np.nan), errors="coerce").dropna()
        years = []
        for _, yp in part.groupby(pd.to_datetime(part["signal_time"]).dt.year):
            yret = pd.to_numeric(yp[base_col], errors="coerce").dropna()
            if len(yret) >= max(20, min_count // 5):
                years.append(float(yret.mean()) > 0.0)
        positive_years = int(sum(years))
        year_count = int(len(years))
        m = matched_idx.loc[event_name] if not matched_idx.empty and event_name in matched_idx.index else None
        excess = float(m["excess_mean_net"]) if m is not None and "excess_mean_net" in m else np.nan
        baseline_mean = float(m["matched_baseline_mean_net"]) if m is not None and "matched_baseline_mean_net" in m else np.nan
        count = int(len(ret))
        mean_net = float(ret.mean()) if count else np.nan
        pf = _profit_factor(ret)
        fee2_mean = float(fee2.mean()) if len(fee2) else np.nan
        delay1_mean = float(delay1.mean()) if len(delay1) else np.nan
        top5 = _top5_winner_share(ret)
        max_gap = _max_days_without_event(part["signal_time"])
        decision, reason = _candidate_decision(
            count=count,
            min_count=min_count,
            mean_net=mean_net,
            pf=pf,
            fee2_mean=fee2_mean,
            delay1_mean=delay1_mean,
            positive_years=positive_years,
            year_count=year_count,
            top5_share=top5,
            excess_mean=excess,
            max_days_without_event=max_gap,
        )
        first = part.iloc[0]
        rows.append(
            {
                "candidate_id": event_name,
                "event_name": event_name,
                "family": first.get("family"),
                "direction": first.get("direction"),
                "strategy_class": "MF_panic_wick_delayed_reversal_research",
                "primary_horizon": int(primary_horizon),
                "best_horizon": int(primary_horizon),
                "count": count,
                "events_per_month": _events_per_month(part["signal_time"]),
                "max_days_without_event": max_gap,
                "confirmation_bars": int(first.get("confirmation_bars", 0)),
                "planned_entry_offset": int(first.get("planned_entry_offset", 1)),
                "mean_net": mean_net,
                "median_net": float(ret.median()) if count else np.nan,
                "win_rate": float((ret > 0).mean()) if count else np.nan,
                "profit_factor": pf,
                "mfe_mean": float(pd.to_numeric(part[mfe_col], errors="coerce").mean()) if mfe_col else np.nan,
                "mae_mean": float(pd.to_numeric(part[mae_col], errors="coerce").mean()) if mae_col else np.nan,
                "positive_years": positive_years,
                "year_count": year_count,
                "session_best": _best_bucket(part, bucket="session", ret_col=base_col),
                "regime_best": _best_bucket(part, bucket="vol_regime", ret_col=base_col),
                "trend_best": _best_bucket(part, bucket="trend_regime", ret_col=base_col),
                "matched_baseline_mean_net": baseline_mean,
                "excess_mean_net": excess,
                "fee_2x_mean_net": fee2_mean,
                "delay_1bar_mean_net": delay1_mean,
                "top5_winner_share": top5,
                "decision": decision,
                "reason": reason,
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(["decision", "excess_mean_net", "mean_net"], ascending=[True, False, False], na_position="last")
    rejected = df[df["decision"].eq("rejected")].copy()
    shortlist = df[~df["decision"].eq("rejected")].copy()
    return shortlist, rejected


def build_hypothesis_matrix(events: pd.DataFrame, *, primary_horizon: int) -> pd.DataFrame:
    """Compact structure-level diagnostics for reviewers."""
    if events.empty:
        return pd.DataFrame()
    ret_col = f"ret_h{int(primary_horizon)}_d0_cost1_net"
    rows: list[dict[str, object]] = []
    for event_name, part in events.groupby("event_name", dropna=False):
        x = pd.to_numeric(part[ret_col], errors="coerce").dropna()
        first = part.iloc[0]
        rows.append(
            {
                "event_name": event_name,
                "hypothesis_type": first.get("family"),
                "confirmation_bars": int(first.get("confirmation_bars", 0)),
                "entry_policy": f"enter_next_open_after_{int(first.get('confirmation_bars', 0))}_closed_confirmation_bars",
                "count": int(len(x)),
                "mean_net": float(x.mean()) if len(x) else np.nan,
                "median_net": float(x.median()) if len(x) else np.nan,
                "win_rate": float((x > 0).mean()) if len(x) else np.nan,
                "profit_factor": _profit_factor(x),
                "max_days_without_event": _max_days_without_event(part["signal_time"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_net", "profit_factor"], ascending=False, na_position="last")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_reports(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    thresholds: PanicThresholds,
    features: pd.DataFrame,
    events: pd.DataFrame,
    baseline: pd.DataFrame,
    matched: pd.DataFrame,
    shortlist: pd.DataFrame,
    rejected: pd.DataFrame,
    causal_audit: pd.DataFrame,
    horizons: tuple[int, ...],
    cost_multipliers: tuple[float, ...],
    delay_bars_list: tuple[int, ...],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    primary_col = f"ret_h{int(args.primary_horizon)}_d0_cost1_net"
    ret_cols = [f"ret_h{h}_d0_cost1_net" for h in horizons]

    print("[aggregate] building report tables", flush=True)
    events = events.copy()
    if not events.empty:
        events["year"] = pd.to_datetime(events["signal_time"]).dt.year
        events["month"] = pd.to_datetime(events["signal_time"]).dt.to_period("M").astype(str)

    summary = summarize_groups(events, group_cols=["event_name", "family", "direction"], ret_cols=[primary_col], min_count=int(args.min_count))
    event_counts = (
        events.groupby(["event_name", "family", "direction"], dropna=False, observed=False)
        .agg(
            count=("event_id", "count"),
            first_signal_time=("signal_time", "min"),
            last_signal_time=("signal_time", "max"),
            events_per_month=("signal_time", _events_per_month),
            max_days_without_event=("signal_time", _max_days_without_event),
            confirmation_bars=("confirmation_bars", "first"),
            planned_entry_offset=("planned_entry_offset", "first"),
        )
        .reset_index()
        if not events.empty
        else pd.DataFrame()
    )
    fwd = summarize_groups(events, group_cols=["event_name", "direction"], ret_cols=ret_cols, min_count=int(args.min_count))
    yearly = summarize_groups(events, group_cols=["event_name", "direction", "year"], ret_cols=[primary_col], min_count=max(20, int(args.min_count) // 5))
    monthly = summarize_groups(events, group_cols=["event_name", "direction", "month"], ret_cols=[primary_col], min_count=10)
    session = summarize_groups(events, group_cols=["event_name", "direction", "session"], ret_cols=[primary_col], min_count=30)
    regime = summarize_groups(events, group_cols=["event_name", "direction", "vol_regime", "trend_regime"], ret_cols=[primary_col], min_count=30)

    cost_rows = []
    for mult in cost_multipliers:
        col = f"ret_h{int(args.primary_horizon)}_d0_cost{_cost_tag(mult)}_net"
        if col in events.columns:
            tmp = summarize_groups(events, group_cols=["event_name", "direction"], ret_cols=[col], min_count=int(args.min_count))
            tmp["cost_multiplier"] = float(mult)
            cost_rows.append(tmp)
    cost_stress = pd.concat(cost_rows, ignore_index=True) if cost_rows else pd.DataFrame()

    delay_rows = []
    for delay in delay_bars_list:
        col = f"ret_h{int(args.primary_horizon)}_d{int(delay)}_cost1_net"
        if col in events.columns:
            tmp = summarize_groups(events, group_cols=["event_name", "direction"], ret_cols=[col], min_count=int(args.min_count))
            tmp["extra_delay_bars"] = int(delay)
            delay_rows.append(tmp)
    delay_stress = pd.concat(delay_rows, ignore_index=True) if delay_rows else pd.DataFrame()
    hypothesis_matrix = build_hypothesis_matrix(events, primary_horizon=int(args.primary_horizon))

    sample_cols = [
        "event_id",
        "event_name",
        "family",
        "direction",
        "structure",
        "event_bar_time",
        "signal_bar_time",
        "signal_time",
        "confirmation_bars",
        "planned_entry_offset",
        "entry_time",
        "entry_price",
        "session",
        "vol_regime",
        "trend_regime",
        "close_pos",
        "volume_ratio",
        "lower_wick_share",
        "lower_wick_atr",
        "ret_30",
        "ret_120",
        "delta_ratio",
        "taker_buy_ratio",
        primary_col,
    ]
    sample_cols = [c for c in sample_cols if c in events.columns]
    event_sample = events[sample_cols].head(int(args.event_sample_size)).copy() if not events.empty else pd.DataFrame(columns=sample_cols)

    manifest = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "edge_status": "research_only_not_tradable",
        "title": TITLE,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "primary_timeframe": args.timeframe,
        "context_timeframes": [],
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "horizons": list(horizons),
        "primary_horizon": int(args.primary_horizon),
        "mfe_mae_horizon": int(args.mfe_mae_horizon),
        "round_trip_cost_pct": float(args.round_trip_cost_pct),
        "cost_multipliers": list(cost_multipliers),
        "delay_bars_list": list(delay_bars_list),
        "input_rows": int(len(features)),
        "event_count": int(len(events)),
        "candidate_count": int(len(shortlist)),
        "rejected_count": int(len(rejected)),
        "thresholds": thresholds.__dict__,
        "causal_policy": "closed 1m event/confirmation bars; entry at next bar open after final confirmation; no MTF context",
        "anti_overfit_policy": "V2 tests a fixed long-only panic-wick hypothesis matrix; no Cartesian parameter grid; matched panic-context baseline, cost, delay, year/session/regime audits required",
        "v1_context": "V1 broad long/short volume-shadow research was rejected; V2 only follows the remaining panic-downtrend lower-wick clue.",
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    _write_csv(summary, out_dir / "01_summary.csv")
    _write_csv(event_counts, out_dir / "02_event_counts.csv")
    _write_csv(fwd, out_dir / "03_forward_return_matrix.csv")
    _write_csv(yearly, out_dir / "04_yearly_breakdown.csv")
    _write_csv(monthly, out_dir / "05_monthly_breakdown.csv")
    _write_csv(session, out_dir / "06_session_breakdown.csv")
    _write_csv(regime, out_dir / "07_regime_breakdown.csv")
    _write_csv(cost_stress, out_dir / "08_cost_stress.csv")
    _write_csv(delay_stress, out_dir / "09_delay_stress.csv")
    _write_csv(shortlist, out_dir / "10_candidate_shortlist.csv")
    _write_csv(rejected, out_dir / "11_rejected_candidates.csv")
    _write_csv(causal_audit, out_dir / "12_causal_audit.csv")
    _write_csv(event_sample, out_dir / "13_event_sample.csv")
    _write_csv(matched, out_dir / "14_matched_baseline.csv")
    _write_csv(hypothesis_matrix, out_dir / "15_hypothesis_matrix.csv")
    if bool(args.write_full_events):
        _write_csv(events, out_dir / "16_full_events_with_forward_returns.csv")

    notes = f"""# {TITLE}

This is a research-only pack. It does **not** register a tradable edge.

## Why V2 exists
V1 rejected broad 1m volume-shadow long/short events and the add/flip path. The
only useful clue was that a few lower-wick long buckets inside high/extreme-vol
selloff regimes looked better than the broad sample. V2 therefore narrows to one
long-only question:

> After a high-volatility panic flush, does a volume-climax lower wick become
> useful only when price stops making fresh lows and confirms stabilization?

## What is deliberately not done
- No long/short mirror.
- No same-side add path.
- No wick/volume/ATR parameter grid.
- No post-hoc regime promotion without matched baseline and robustness.

## Causal boundary
Delayed confirmation events only enter after the confirmation bars are closed.
For example, `panic_next5_stabilize_long` uses the five bars after the event bar,
sets the signal to the fifth confirmation bar's close, then enters at the next
bar open.
"""
    (out_dir / "README_RESEARCH.md").write_text(notes, encoding="utf-8")

    print("[review-pack] finalizing GPT review pack", flush=True)
    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    horizons = _parse_csv_ints(args.horizons)
    cost_multipliers = _parse_csv_floats(args.cost_multipliers)
    delay_bars_list = _parse_csv_ints(args.delay_bars_list)
    if int(args.primary_horizon) not in horizons:
        horizons = tuple(sorted((*horizons, int(args.primary_horizon))))
    if 1.0 not in cost_multipliers:
        cost_multipliers = tuple(sorted((*cost_multipliers, 1.0)))
    if 2.0 not in cost_multipliers:
        cost_multipliers = tuple(sorted((*cost_multipliers, 2.0)))
    if 0 not in delay_bars_list:
        delay_bars_list = tuple(sorted((*delay_bars_list, 0)))
    if 1 not in delay_bars_list:
        delay_bars_list = tuple(sorted((*delay_bars_list, 1)))

    thresholds = _thresholds_from_args(args)
    print(f"[run] {SCRIPT_NAME} v{SCRIPT_VERSION}", flush=True)
    print(f"[args] out_dir={args.out_dir}", flush=True)
    print("[scope] V2 long-only panic-wick delayed confirmation; no parameter-grid search", flush=True)

    bars = load_trade_bars(args)
    features = build_features(bars, thresholds)
    research_mask = _research_window_mask(features.index, args.start_date, args.end_date)
    research_features = features.loc[research_mask.to_numpy(dtype=bool)].copy()
    events = build_panic_wick_events(research_features, thresholds)
    # IMPORTANT: event_bar_pos/signal_bar_pos are built on research_features,
    # not on the full warmup frame. Passing the full warmup frame shifts entry
    # timestamps/prices backward by the warmup rows and breaks the causal audit.
    events = attach_forward_returns(
        bars=research_features,
        events=events,
        horizons=horizons,
        cost_multipliers=cost_multipliers,
        delay_bars_list=delay_bars_list,
        round_trip_cost_pct=float(args.round_trip_cost_pct),
        mfe_mae_horizon=int(args.mfe_mae_horizon),
        progress_every=int(args.progress_every),
    )
    causal_audit = build_causal_audit(events)
    if not causal_audit.empty and bool(causal_audit["lookahead_flag"].any()):
        print(f"[causal] WARNING lookahead flags={int(causal_audit['lookahead_flag'].sum())}", flush=True)
    else:
        print("[causal] no lookahead flags in event-specific next-open audit", flush=True)

    entry_offsets = tuple(sorted(set(int(x) for x in events["planned_entry_offset"].dropna().unique()))) if not events.empty else (1,)
    baseline = build_context_baseline(
        research_features,
        horizons=horizons,
        cost_multipliers=cost_multipliers,
        delay_bars_list=delay_bars_list,
        round_trip_cost_pct=float(args.round_trip_cost_pct),
        entry_offsets=entry_offsets,
    )
    matched = matched_baseline_summary(events, baseline, primary_horizon=int(args.primary_horizon), min_count=int(args.min_count))
    shortlist, rejected = build_candidate_shortlist(events, matched, primary_horizon=int(args.primary_horizon), min_count=int(args.min_count))

    write_reports(
        out_dir=Path(args.out_dir),
        args=args,
        thresholds=thresholds,
        features=research_features,
        events=events,
        baseline=baseline,
        matched=matched,
        shortlist=shortlist,
        rejected=rejected,
        causal_audit=causal_audit,
        horizons=horizons,
        cost_multipliers=cost_multipliers,
        delay_bars_list=delay_bars_list,
    )

    print("[done] V2 research completed; inspect gpt_review_pack.zip before deciding next step", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
