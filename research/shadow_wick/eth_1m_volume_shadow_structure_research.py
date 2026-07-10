#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH 1m volume-shadow structure research.

Research purpose
----------------
This script studies the user's idea without turning it into a parameter-grid
strategy:

    volume climax + very long lower shadow -> long bias
    volume climax + very long upper shadow -> short bias
    same-side events can add, opposite-side events can flatten/flip.

The implementation is intentionally hypothesis/structure based, not a broad
parameter optimizer.  Fixed thresholds only define a small event vocabulary; the
main outputs compare structural contexts, matched baselines, cost/delay stress,
and a diagnostic path probe of the add/flip idea.

Causal policy
-------------
- Features use only the closed 1m signal bar and past bars.
- The 1m bar index is treated as bar-start time.
- signal_available_time = signal_bar_time + 1 minute.
- entry_time = next bar open = signal_bar_time + 1 minute.
- Forward returns use next-open entry; no current-bar open execution.
- No multi-timeframe context is used in this V1, so there is no hidden
  high-timeframe left-label ffill risk.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
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

SCRIPT_NAME = "eth_1m_volume_shadow_structure_research"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_MF_1M_VOLUME_SHADOW_STRUCTURE_V1"
EDGE_ID = "RESEARCH_ONLY_ETH_MF_1M_VOLUME_SHADOW_STRUCTURE_V1"
TITLE = "ETH 1m Volume Shadow Structure Research V1"
DEFAULT_OUT_DIR = "data/reports/research/eth_1m_volume_shadow_structure_v1"
BAR_DELTA = pd.Timedelta(minutes=1)


# ---------------------------------------------------------------------------
# Config / parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StructureThresholds:
    """Small fixed event vocabulary thresholds, not a parameter search grid."""

    wick_share_min: float = 0.50
    wick_atr_min: float = 0.55
    volume_ratio_min: float = 2.0
    reclaim_close_pos: float = 0.66
    reject_close_pos: float = 0.34
    prior_move_abs_min: float = 0.005
    repeat_lookback_bars: int = 120
    repeat_price_band_pct: float = 0.0025


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Research-only ETH 1m volume long-shadow event study + add/flip path probe.",
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

    # Fixed study settings. These are exposed so a reviewer can reproduce a
    # stricter/looser *single* research run, but the script does not sweep them.
    p.add_argument("--horizons", default="5,15,30,60,120,240")
    p.add_argument("--primary-horizon", type=int, default=60)
    p.add_argument("--mfe-mae-horizon", type=int, default=240)
    p.add_argument("--round-trip-cost-pct", type=float, default=0.0011)
    p.add_argument("--cost-multipliers", default="1.0,1.5,2.0,3.0")
    p.add_argument("--delay-bars-list", default="0,1,2,3", help="Extra bars beyond normal next-open entry.")
    p.add_argument("--min-count", type=int, default=150)
    p.add_argument("--event-sample-size", type=int, default=5000)
    p.add_argument("--write-full-events", action="store_true")
    p.add_argument("--progress-every", type=int, default=2500)

    # Structure-definition thresholds. They are deliberately few and fixed;
    # no Cartesian product is created from them.
    p.add_argument("--wick-share-min", type=float, default=0.50)
    p.add_argument("--wick-atr-min", type=float, default=0.55)
    p.add_argument("--volume-ratio-min", type=float, default=2.0)
    p.add_argument("--reclaim-close-pos", type=float, default=0.66)
    p.add_argument("--reject-close-pos", type=float, default=0.34)
    p.add_argument("--prior-move-abs-min", type=float, default=0.005)
    p.add_argument("--repeat-lookback-bars", type=int, default=120)
    p.add_argument("--repeat-price-band-pct", type=float, default=0.0025)

    # Diagnostic path probe. Not a full candidate backtest.
    p.add_argument("--path-max-layers", type=int, default=3)
    p.add_argument("--path-max-hold-bars", type=int, default=240)
    p.add_argument("--save-path-trades", type=int, default=10000)
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


def _thresholds_from_args(args: argparse.Namespace) -> StructureThresholds:
    return StructureThresholds(
        wick_share_min=float(args.wick_share_min),
        wick_atr_min=float(args.wick_atr_min),
        volume_ratio_min=float(args.volume_ratio_min),
        reclaim_close_pos=float(args.reclaim_close_pos),
        reject_close_pos=float(args.reject_close_pos),
        prior_move_abs_min=float(args.prior_move_abs_min),
        repeat_lookback_bars=int(args.repeat_lookback_bars),
        repeat_price_band_pct=float(args.repeat_price_band_pct),
    )


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


def build_features(bars: pd.DataFrame, th: StructureThresholds) -> pd.DataFrame:
    """Build closed-bar/past-only features for structural event definitions."""
    print("[features] building candle geometry, volume, order-flow and regime labels", flush=True)
    df = bars.copy().sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    body_high = df[["open", "close"]].max(axis=1)
    body_low = df[["open", "close"]].min(axis=1)
    df["range_pct"] = _safe_divide(rng, df["close"]).to_numpy()
    df["body_pct"] = (df["close"] / df["open"] - 1.0).replace([np.inf, -np.inf], np.nan)
    df["close_pos"] = _safe_divide(df["close"] - df["low"], rng).to_numpy()
    df["lower_wick"] = (body_low - df["low"]).clip(lower=0.0)
    df["upper_wick"] = (df["high"] - body_high).clip(lower=0.0)
    df["lower_wick_share"] = _safe_divide(df["lower_wick"], rng).to_numpy()
    df["upper_wick_share"] = _safe_divide(df["upper_wick"], rng).to_numpy()

    df["tr"] = _true_range(df)
    # ATR includes current closed bar; volume baseline excludes current bar.
    df["atr"] = df["tr"].rolling(60, min_periods=30).mean()
    df["atr_pct"] = _safe_divide(df["atr"], df["close"]).to_numpy()
    df["lower_wick_atr"] = _safe_divide(df["lower_wick"], df["atr"]).to_numpy()
    df["upper_wick_atr"] = _safe_divide(df["upper_wick"], df["atr"]).to_numpy()

    vol_base = df["volume"].shift(1).rolling(240, min_periods=60).median()
    df["volume_ratio"] = _safe_divide(df["volume"], vol_base).to_numpy()
    df["volume_climax"] = df["volume_ratio"] >= th.volume_ratio_min

    df["ret_15"] = df["close"].pct_change(15)
    df["ret_30"] = df["close"].pct_change(30)
    df["ret_120"] = df["close"].pct_change(120)
    df["ema_60"] = df["close"].ewm(span=60, adjust=False, min_periods=60).mean()
    df["ema_240"] = df["close"].ewm(span=240, adjust=False, min_periods=240).mean()
    df["ema_720"] = df["close"].ewm(span=720, adjust=False, min_periods=720).mean()
    df["trend_above_ema240"] = df["close"] > df["ema_240"]
    df["trend_below_ema240"] = df["close"] < df["ema_240"]
    df["ema240_slope_60"] = df["ema_240"] / df["ema_240"].shift(60) - 1.0

    delta = pd.to_numeric(df.get("delta_notional", np.nan), errors="coerce")
    notional = pd.to_numeric(df.get("notional", np.nan), errors="coerce").abs()
    df["delta_ratio"] = _safe_divide(delta, notional).to_numpy()
    if "taker_buy_ratio" in df.columns:
        df["taker_buy_ratio"] = pd.to_numeric(df["taker_buy_ratio"], errors="coerce")
    else:
        buy_volume = pd.to_numeric(df.get("buy_volume", np.nan), errors="coerce")
        df["taker_buy_ratio"] = _safe_divide(buy_volume, df["volume"]).to_numpy()

    df["long_lower_wick"] = (df["lower_wick_share"] >= th.wick_share_min) & (df["lower_wick_atr"] >= th.wick_atr_min)
    df["long_upper_wick"] = (df["upper_wick_share"] >= th.wick_share_min) & (df["upper_wick_atr"] >= th.wick_atr_min)
    df["lower_volume_shadow"] = df["long_lower_wick"] & df["volume_climax"]
    df["upper_volume_shadow"] = df["long_upper_wick"] & df["volume_climax"]
    df["two_sided_shadow"] = df["lower_volume_shadow"] & df["upper_volume_shadow"]

    prior_lower = df["lower_volume_shadow"].shift(1).rolling(th.repeat_lookback_bars, min_periods=1).sum()
    prior_upper = df["upper_volume_shadow"].shift(1).rolling(th.repeat_lookback_bars, min_periods=1).sum()
    prior_lower_low = df["low"].where(df["lower_volume_shadow"]).shift(1).rolling(th.repeat_lookback_bars, min_periods=1).min()
    prior_upper_high = df["high"].where(df["upper_volume_shadow"]).shift(1).rolling(th.repeat_lookback_bars, min_periods=1).max()
    df["repeat_lower_near_low"] = (prior_lower >= 1) & (df["low"] <= prior_lower_low * (1.0 + th.repeat_price_band_pct))
    df["repeat_upper_near_high"] = (prior_upper >= 1) & (df["high"] >= prior_upper_high * (1.0 - th.repeat_price_band_pct))

    df["session"] = session_label(df.index)
    df["vol_regime"] = pd.cut(
        df["atr_pct"],
        bins=[-np.inf, 0.0015, 0.0030, 0.0050, np.inf],
        labels=["very_low_vol", "low_mid_vol", "mid_high_vol", "extreme_vol"],
    ).astype("object").fillna("NA")
    df["trend_regime"] = np.select(
        [
            (df["trend_above_ema240"] & (df["ema240_slope_60"] > 0.0005)),
            (df["trend_below_ema240"] & (df["ema240_slope_60"] < -0.0005)),
        ],
        ["uptrend", "downtrend"],
        default="range_or_transition",
    )
    return df


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


# ---------------------------------------------------------------------------
# Event construction
# ---------------------------------------------------------------------------


def _event_frame_from_mask(
    features: pd.DataFrame,
    *,
    mask: pd.Series,
    event_name: str,
    family: str,
    side: int,
    structure: str,
) -> pd.DataFrame:
    idx = features.index[mask.fillna(False).to_numpy(dtype=bool)]
    if len(idx) == 0:
        return pd.DataFrame()
    f = features.loc[idx]
    out = pd.DataFrame(
        {
            "event_name": event_name,
            "family": family,
            "direction": "LONG" if side == 1 else "SHORT",
            "side": side,
            "structure": structure,
            "signal_bar_time": idx,
            "signal_time": idx + BAR_DELTA,
            "signal_available_time": idx + BAR_DELTA,
            "session": f["session"].to_numpy(),
            "vol_regime": f["vol_regime"].astype(str).to_numpy(),
            "trend_regime": f["trend_regime"].astype(str).to_numpy(),
            "close_pos": f["close_pos"].to_numpy(dtype=float),
            "volume_ratio": f["volume_ratio"].to_numpy(dtype=float),
            "lower_wick_share": f["lower_wick_share"].to_numpy(dtype=float),
            "upper_wick_share": f["upper_wick_share"].to_numpy(dtype=float),
            "lower_wick_atr": f["lower_wick_atr"].to_numpy(dtype=float),
            "upper_wick_atr": f["upper_wick_atr"].to_numpy(dtype=float),
            "ret_30": f["ret_30"].to_numpy(dtype=float),
            "ret_120": f["ret_120"].to_numpy(dtype=float),
            "delta_ratio": f["delta_ratio"].to_numpy(dtype=float),
            "taker_buy_ratio": f["taker_buy_ratio"].to_numpy(dtype=float),
            "two_sided_shadow": f["two_sided_shadow"].astype(bool).to_numpy(),
        }
    )
    return out


def build_structure_events(features: pd.DataFrame, th: StructureThresholds) -> pd.DataFrame:
    """Build a fixed set of structural hypotheses, not a parameter grid."""
    print("[events] building fixed structural event vocabulary", flush=True)
    f = features
    lower = f["lower_volume_shadow"] & ~f["two_sided_shadow"]
    upper = f["upper_volume_shadow"] & ~f["two_sided_shadow"]
    lower_reclaim = lower & (f["close_pos"] >= th.reclaim_close_pos)
    upper_reject = upper & (f["close_pos"] <= th.reject_close_pos)

    specs: list[tuple[str, str, int, str, pd.Series]] = [
        ("lower_any_long", "raw_volume_shadow", 1, "lower_volume_shadow", lower),
        ("upper_any_short", "raw_volume_shadow", -1, "upper_volume_shadow", upper),
        ("lower_reclaim_long", "reclaim_reject", 1, "lower_close_upper_third", lower_reclaim),
        ("upper_reject_short", "reclaim_reject", -1, "upper_close_lower_third", upper_reject),
        (
            "lower_green_reclaim_long",
            "reclaim_body_confirm",
            1,
            "lower_reclaim_and_green_body",
            lower_reclaim & (f["close"] > f["open"]),
        ),
        (
            "upper_red_reject_short",
            "reclaim_body_confirm",
            -1,
            "upper_reject_and_red_body",
            upper_reject & (f["close"] < f["open"]),
        ),
        (
            "lower_panic_absorption_long",
            "prior_move_exhaustion",
            1,
            "lower_reclaim_after_down_move",
            lower_reclaim & (f["ret_30"] <= -th.prior_move_abs_min),
        ),
        (
            "upper_euphoria_absorption_short",
            "prior_move_exhaustion",
            -1,
            "upper_reject_after_up_move",
            upper_reject & (f["ret_30"] >= th.prior_move_abs_min),
        ),
        (
            "lower_trend_pullback_long",
            "trend_pullback",
            1,
            "lower_reclaim_above_ema240_after_pullback",
            lower_reclaim & f["trend_above_ema240"] & (f["ret_30"] < 0.0),
        ),
        (
            "upper_trend_pullback_short",
            "trend_pullback",
            -1,
            "upper_reject_below_ema240_after_bounce",
            upper_reject & f["trend_below_ema240"] & (f["ret_30"] > 0.0),
        ),
        (
            "lower_repeat_absorption_long",
            "repeat_test_absorption",
            1,
            "repeated_lower_shadow_near_prior_low",
            lower_reclaim & f["repeat_lower_near_low"],
        ),
        (
            "upper_repeat_absorption_short",
            "repeat_test_absorption",
            -1,
            "repeated_upper_shadow_near_prior_high",
            upper_reject & f["repeat_upper_near_high"],
        ),
        (
            "lower_failed_reclaim_short",
            "failed_reclaim_continuation",
            -1,
            "lower_shadow_close_weak_continuation_short",
            lower & (f["close_pos"] <= th.reject_close_pos),
        ),
        (
            "upper_failed_reject_long",
            "failed_reclaim_continuation",
            1,
            "upper_shadow_close_strong_continuation_long",
            upper & (f["close_pos"] >= th.reclaim_close_pos),
        ),
        (
            "lower_sell_pressure_absorbed_long",
            "orderflow_absorption",
            1,
            "lower_reclaim_with_negative_delta",
            lower_reclaim & ((f["delta_ratio"] <= -0.10) | (f["taker_buy_ratio"] <= 0.45)),
        ),
        (
            "upper_buy_pressure_absorbed_short",
            "orderflow_absorption",
            -1,
            "upper_reject_with_positive_delta",
            upper_reject & ((f["delta_ratio"] >= 0.10) | (f["taker_buy_ratio"] >= 0.55)),
        ),
    ]
    frames = [
        _event_frame_from_mask(f, mask=mask, event_name=name, family=family, side=side, structure=structure)
        for name, family, side, structure, mask in specs
    ]
    out = pd.concat([x for x in frames if not x.empty], ignore_index=True) if any(not x.empty for x in frames) else pd.DataFrame()
    if out.empty:
        return out
    out = out.sort_values(["signal_bar_time", "event_name"]).reset_index(drop=True)
    out["event_id"] = np.arange(len(out), dtype=int)
    return out


# ---------------------------------------------------------------------------
# Forward returns / MFE-MAE / audit
# ---------------------------------------------------------------------------


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
    """Attach next-open forward labels using vectorized bar arrays."""
    print("[forward] attaching next-open return, cost, delay and MFE/MAE labels", flush=True)
    if events.empty:
        return events.copy()

    frame = bars.copy().sort_index()
    pos_map = pd.Series(np.arange(len(frame), dtype=int), index=frame.index)
    out = events.copy()
    out["signal_bar_pos"] = pos_map.reindex(pd.to_datetime(out["signal_bar_time"])).to_numpy()
    out = out.dropna(subset=["signal_bar_pos"]).copy()
    out["signal_bar_pos"] = out["signal_bar_pos"].astype(int)
    out["entry_bar_pos"] = out["signal_bar_pos"] + 1
    valid_entry = out["entry_bar_pos"].between(0, len(frame) - 1)
    out["entry_time"] = pd.NaT
    out["entry_price"] = np.nan
    if valid_entry.any():
        entry_pos = out.loc[valid_entry, "entry_bar_pos"].astype(int).to_numpy()
        out.loc[valid_entry, "entry_time"] = frame.index[entry_pos]
        out.loc[valid_entry, "entry_price"] = frame["open"].iloc[entry_pos].to_numpy(dtype=float)
    out["expected_entry_time"] = pd.to_datetime(out["signal_bar_time"]) + BAR_DELTA
    out["expected_entry_price"] = np.nan
    if valid_entry.any():
        out.loc[valid_entry, "expected_entry_price"] = out.loc[valid_entry, "entry_price"].to_numpy(dtype=float)

    open_ = pd.to_numeric(frame["open"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    side_by_pos = pd.Series(0.0, index=frame.index)
    pos_arr = out["signal_bar_pos"].to_numpy(dtype=int)
    sides = out["side"].to_numpy(dtype=float)

    for extra_delay in delay_bars_list:
        entry_offset = 1 + int(extra_delay)
        entry_open = open_.shift(-entry_offset)
        for horizon in horizons:
            h = int(horizon)
            gross_col = f"ret_h{h}_d{int(extra_delay)}_gross"
            # Conservative delay stress: exit remains tied to the original signal horizon.
            future_close = close.shift(-h)
            entry_vals = entry_open.iloc[pos_arr].to_numpy(dtype=float)
            exit_vals = future_close.iloc[pos_arr].to_numpy(dtype=float)
            gross = (exit_vals / entry_vals - 1.0) * sides
            invalid = (~np.isfinite(entry_vals)) | (entry_vals <= 0) | (~np.isfinite(exit_vals)) | (h <= extra_delay)
            gross[invalid] = np.nan
            out[gross_col] = gross
            for mult in cost_multipliers:
                net_col = f"ret_h{h}_d{int(extra_delay)}_cost{_cost_tag(mult)}_net"
                out[net_col] = out[gross_col] - float(round_trip_cost_pct) * float(mult)

    entry_offset = 1
    h = int(mfe_mae_horizon)
    future_max = _future_window_extreme(high, start_offset=entry_offset, end_offset=h, op="max")
    future_min = _future_window_extreme(low, start_offset=entry_offset, end_offset=h, op="min")
    entry = open_.shift(-entry_offset)
    e = entry.iloc[pos_arr].to_numpy(dtype=float)
    fmax = future_max.iloc[pos_arr].to_numpy(dtype=float)
    fmin = future_min.iloc[pos_arr].to_numpy(dtype=float)
    mfe = np.where(sides > 0, fmax / e - 1.0, e / fmin - 1.0)
    mae = np.where(sides > 0, fmin / e - 1.0, e / fmax - 1.0)
    bad = (~np.isfinite(e)) | (e <= 0) | (~np.isfinite(fmax)) | (~np.isfinite(fmin)) | (fmin <= 0)
    mfe[bad] = np.nan
    mae[bad] = np.nan
    out[f"mfe_h{h}"] = mfe
    out[f"mae_h{h}"] = mae

    # Small progress indicator for consistency on large runs; labels are vectorized.
    progress = ProgressReporter(label="[forward] label columns", total=len(horizons) * len(delay_bars_list), every=max(1, int(progress_every)))
    progress.close()
    return out


def build_causal_audit(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    out = events[
        [
            "event_id",
            "event_name",
            "signal_bar_time",
            "signal_time",
            "signal_available_time",
            "entry_time",
            "entry_price",
            "expected_entry_time",
            "expected_entry_price",
            "signal_bar_pos",
            "entry_bar_pos",
        ]
    ].copy()
    out["entry_not_next_open_flag"] = pd.to_datetime(out["entry_time"]) != pd.to_datetime(out["expected_entry_time"])
    out["entry_price_mismatch_flag"] = False
    out["context_available_time_flag"] = False
    out["same_bar_entry_flag"] = out["entry_bar_pos"] <= out["signal_bar_pos"]
    out["lookahead_flag"] = out[["entry_not_next_open_flag", "entry_price_mismatch_flag", "context_available_time_flag", "same_bar_entry_flag"]].any(axis=1)
    return out


# ---------------------------------------------------------------------------
# Summaries / matched baseline
# ---------------------------------------------------------------------------


def _cost_tag(mult: float) -> str:
    text = f"{float(mult):g}".replace(".", "p").replace("-", "m")
    return text


def _summary_row(part: pd.DataFrame, *, ret_col: str, group: dict[str, object], min_count: int) -> dict[str, object]:
    x = pd.to_numeric(part.get(ret_col, pd.Series(dtype=float)), errors="coerce").dropna()
    row = dict(group)
    row["metric"] = ret_col
    row["count"] = int(len(x))
    row["eligible"] = bool(len(x) >= int(min_count))
    row["mean_net"] = float(x.mean()) if len(x) else np.nan
    row["median_net"] = float(x.median()) if len(x) else np.nan
    row["win_rate"] = float((x > 0).mean()) if len(x) else np.nan
    row["profit_factor"] = _profit_factor(x)
    row["top5_winner_share"] = _top5_winner_share(x)
    row["p05"] = float(x.quantile(0.05)) if len(x) else np.nan
    row["p25"] = float(x.quantile(0.25)) if len(x) else np.nan
    row["p75"] = float(x.quantile(0.75)) if len(x) else np.nan
    row["p95"] = float(x.quantile(0.95)) if len(x) else np.nan
    return row


def summarize_groups(
    events: pd.DataFrame,
    *,
    group_cols: list[str],
    ret_cols: list[str],
    min_count: int,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    if group_cols:
        grouped = events.groupby(group_cols, dropna=False, observed=False)
        for key, part in grouped:
            key_tuple = key if isinstance(key, tuple) else (key,)
            group = dict(zip(group_cols, key_tuple, strict=False))
            for ret_col in ret_cols:
                rows.append(_summary_row(part, ret_col=ret_col, group=group, min_count=min_count))
    else:
        for ret_col in ret_cols:
            rows.append(_summary_row(events, ret_col=ret_col, group={}, min_count=min_count))
    return pd.DataFrame(rows)


def build_all_bar_baseline(
    features: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    round_trip_cost_pct: float,
    primary_horizon: int,
) -> pd.DataFrame:
    """Build a matched baseline universe over all eligible 1m bars.

    Baseline is not random. For each event bucket, it uses all non-chaotic bars
    with the same year/session/vol/trend buckets and the same directional side.
    """
    frame = features.copy().sort_index()
    rows: list[pd.DataFrame] = []
    for side_name, side in (("LONG", 1), ("SHORT", -1)):
        base = pd.DataFrame(
            {
                "signal_bar_time": frame.index,
                "signal_time": frame.index + BAR_DELTA,
                "side": side,
                "direction": side_name,
                "year": frame.index.year,
                "month": frame.index.to_period("M").astype(str),
                "session": frame["session"].to_numpy(),
                "vol_regime": frame["vol_regime"].astype(str).to_numpy(),
                "trend_regime": frame["trend_regime"].astype(str).to_numpy(),
                "two_sided_shadow": frame["two_sided_shadow"].astype(bool).to_numpy(),
            }
        )
        rows.append(base)
    baseline = pd.concat(rows, ignore_index=True)

    pos_map = pd.Series(np.arange(len(frame), dtype=int), index=frame.index)
    pos = pos_map.reindex(pd.to_datetime(baseline["signal_bar_time"])).to_numpy(dtype=int)
    open_ = pd.to_numeric(frame["open"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    entry = open_.shift(-1).iloc[pos].to_numpy(dtype=float)
    for horizon in horizons:
        h = int(horizon)
        exit_ = close.shift(-h).iloc[pos].to_numpy(dtype=float)
        gross = (exit_ / entry - 1.0) * baseline["side"].to_numpy(dtype=float)
        gross[(~np.isfinite(entry)) | (entry <= 0) | (~np.isfinite(exit_))] = np.nan
        baseline[f"baseline_ret_h{h}_cost1_net"] = gross - float(round_trip_cost_pct)
    # Avoid using ambiguous two-sided climax bars as matched baseline controls.
    baseline = baseline[~baseline["two_sided_shadow"].astype(bool)].copy()
    ret_col = f"baseline_ret_h{int(primary_horizon)}_cost1_net"
    baseline = baseline.dropna(subset=[ret_col])
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
    base_ret_col = f"baseline_ret_h{int(primary_horizon)}_cost1_net"
    match_cols = ["year", "session", "vol_regime", "trend_regime", "direction"]
    base_stats = baseline.groupby(match_cols, dropna=False, observed=False)[base_ret_col].agg(["mean", "count"]).reset_index()
    base_stats = base_stats.rename(columns={"mean": "bucket_baseline_mean_net", "count": "baseline_sample_count"})

    rows: list[dict[str, object]] = []
    for event_name, part in events.groupby("event_name", dropna=False):
        p = part.copy()
        p["year"] = pd.to_datetime(p["signal_time"]).dt.year
        event_bucket_counts = p.groupby(match_cols, dropna=False, observed=False).size().reset_index(name="event_bucket_count")
        merged = event_bucket_counts.merge(base_stats, on=match_cols, how="left")
        total = float(merged["event_bucket_count"].sum())
        weighted_baseline = np.nan
        baseline_count = 0
        if total > 0 and not merged.empty:
            valid = merged["bucket_baseline_mean_net"].notna()
            if valid.any():
                weighted_baseline = float((merged.loc[valid, "bucket_baseline_mean_net"] * merged.loc[valid, "event_bucket_count"]).sum() / merged.loc[valid, "event_bucket_count"].sum())
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
    mfe_col = [c for c in events.columns if c.startswith("mfe_h")]
    mae_col = [c for c in events.columns if c.startswith("mae_h")]
    mfe_col = mfe_col[0] if mfe_col else None
    mae_col = mae_col[0] if mae_col else None

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
        session_best = _best_bucket(part, bucket="session", ret_col=base_col)
        regime_best = _best_bucket(part, bucket="vol_regime", ret_col=base_col)
        trend_best = _best_bucket(part, bucket="trend_regime", ret_col=base_col)
        m = matched_idx.loc[event_name] if not matched_idx.empty and event_name in matched_idx.index else None
        excess = float(m["excess_mean_net"]) if m is not None and "excess_mean_net" in m else np.nan
        baseline_mean = float(m["matched_baseline_mean_net"]) if m is not None and "matched_baseline_mean_net" in m else np.nan
        count = int(len(ret))
        mean_net = float(ret.mean()) if count else np.nan
        pf = _profit_factor(ret)
        fee2_mean = float(fee2.mean()) if len(fee2) else np.nan
        delay1_mean = float(delay1.mean()) if len(delay1) else np.nan
        top5 = _top5_winner_share(ret)
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
        )
        first = part.iloc[0]
        rows.append(
            {
                "candidate_id": event_name,
                "event_name": event_name,
                "family": first.get("family"),
                "direction": first.get("direction"),
                "strategy_class": "MF/MHF_structure_reversal_research",
                "primary_horizon": int(primary_horizon),
                "best_horizon": int(primary_horizon),
                "count": count,
                "events_per_month": _events_per_month(part["signal_time"]),
                "max_days_without_event": _max_days_without_event(part["signal_time"]),
                "mean_net": mean_net,
                "median_net": float(ret.median()) if count else np.nan,
                "win_rate": float((ret > 0).mean()) if count else np.nan,
                "profit_factor": pf,
                "mfe_mean": float(pd.to_numeric(part[mfe_col], errors="coerce").mean()) if mfe_col else np.nan,
                "mae_mean": float(pd.to_numeric(part[mae_col], errors="coerce").mean()) if mae_col else np.nan,
                "positive_years": positive_years,
                "year_count": year_count,
                "session_best": session_best,
                "regime_best": regime_best,
                "trend_best": trend_best,
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
    if np.isfinite(top5_share) and top5_share > 0.50:
        fails.append("top5_winner_dependency_high")
    if not np.isfinite(excess_mean) or excess_mean <= 0:
        fails.append("matched_baseline_excess_not_positive")
    if fails:
        # Research continue only if the shape is promising but not robust enough.
        soft = {"fee2_not_positive", "delay1_not_positive", "positive_years_below_3", "top5_winner_dependency_high"}
        if count >= min_count and np.isfinite(mean_net) and mean_net > 0 and np.isfinite(pf) and pf >= 1.10 and any(f in soft for f in fails):
            return "research_continue", ";".join(fails)
        return "rejected", ";".join(fails)
    return "promote_to_backtest_candidate", "passes_event_study_gate_only;requires_formal_backtest_and_portfolio_overlay"


# ---------------------------------------------------------------------------
# Diagnostic add/flip path probe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PathSpec:
    system_id: str
    long_events: tuple[str, ...]
    short_events: tuple[str, ...]
    notes: str


def default_path_specs() -> tuple[PathSpec, ...]:
    return (
        PathSpec("raw_any_shadow_flip", ("lower_any_long",), ("upper_any_short",), "Original broad lower/upper volume-shadow add/flip idea."),
        PathSpec("reclaim_reject_flip", ("lower_reclaim_long",), ("upper_reject_short",), "Requires close-location rejection/reclaim."),
        PathSpec(
            "panic_euphoria_absorption_flip",
            ("lower_panic_absorption_long",),
            ("upper_euphoria_absorption_short",),
            "Requires prior directional extension before wick rejection.",
        ),
        PathSpec(
            "repeat_absorption_flip",
            ("lower_repeat_absorption_long",),
            ("upper_repeat_absorption_short",),
            "Requires repeated test near prior wick extreme.",
        ),
        PathSpec(
            "failed_reclaim_continuation_flip",
            ("upper_failed_reject_long",),
            ("lower_failed_reclaim_short",),
            "Continuation hypothesis when wick fails to reject.",
        ),
    )


def run_path_probe(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    max_layers: int,
    max_hold_bars: int,
    round_trip_cost_pct: float,
    save_trades: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("[path] running fixed add/flip diagnostic systems", flush=True)
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = bars.copy().sort_index()
    opens = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    all_trades: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    event_names_by_pos: dict[int, set[str]] = {}
    for _, row in events[["signal_bar_pos", "event_name"]].dropna().iterrows():
        event_names_by_pos.setdefault(int(row["signal_bar_pos"]), set()).add(str(row["event_name"]))

    for spec in default_path_specs():
        state_side = 0
        layer_entries: list[float] = []
        layer_entry_pos: list[int] = []
        first_entry_pos: int | None = None
        trades: list[dict[str, object]] = []
        event_positions = sorted(
            pos
            for pos, names in event_names_by_pos.items()
            if bool(set(spec.long_events) & names) or bool(set(spec.short_events) & names)
        )
        for signal_pos in event_positions:
            names = event_names_by_pos[signal_pos]
            long_sig = bool(set(spec.long_events) & names)
            short_sig = bool(set(spec.short_events) & names)
            if long_sig and short_sig:
                continue
            entry_pos = signal_pos + 1
            if entry_pos >= len(frame):
                continue

            # Time stop is diagnostic and fixed, not optimized. It prevents a
            # path probe from carrying a stale position for months with no
            # opposite event.
            if state_side != 0 and first_entry_pos is not None and entry_pos - first_entry_pos > int(max_hold_bars):
                forced_pos = min(first_entry_pos + int(max_hold_bars), len(frame) - 1)
                _close_layers(
                    trades,
                    spec=spec,
                    side=state_side,
                    entry_prices=layer_entries,
                    entry_positions=layer_entry_pos,
                    exit_pos=forced_pos,
                    exit_price=float(opens[forced_pos]),
                    exit_reason="time_stop_before_next_event",
                    frame=frame,
                    round_trip_cost_pct=round_trip_cost_pct,
                )
                state_side = 0
                layer_entries = []
                layer_entry_pos = []
                first_entry_pos = None

            desired_side = 1 if long_sig else -1
            px = float(opens[entry_pos])
            if not np.isfinite(px) or px <= 0:
                continue
            if state_side == 0:
                state_side = desired_side
                layer_entries = [px]
                layer_entry_pos = [entry_pos]
                first_entry_pos = entry_pos
            elif state_side == desired_side:
                if len(layer_entries) < int(max_layers):
                    layer_entries.append(px)
                    layer_entry_pos.append(entry_pos)
            else:
                _close_layers(
                    trades,
                    spec=spec,
                    side=state_side,
                    entry_prices=layer_entries,
                    entry_positions=layer_entry_pos,
                    exit_pos=entry_pos,
                    exit_price=px,
                    exit_reason="opposite_shadow_flip",
                    frame=frame,
                    round_trip_cost_pct=round_trip_cost_pct,
                )
                state_side = desired_side
                layer_entries = [px]
                layer_entry_pos = [entry_pos]
                first_entry_pos = entry_pos

        if state_side != 0 and layer_entries:
            exit_pos = len(frame) - 1
            exit_px = float(closes[exit_pos])
            _close_layers(
                trades,
                spec=spec,
                side=state_side,
                entry_prices=layer_entries,
                entry_positions=layer_entry_pos,
                exit_pos=exit_pos,
                exit_price=exit_px,
                exit_reason="end_of_sample",
                frame=frame,
                round_trip_cost_pct=round_trip_cost_pct,
            )

        ret = pd.Series([t["net_return_sum"] for t in trades], dtype=float)
        summary_rows.append(
            {
                "system_id": spec.system_id,
                "long_events": ",".join(spec.long_events),
                "short_events": ",".join(spec.short_events),
                "notes": spec.notes,
                "trades": int(len(ret)),
                "mean_net_per_closed_position": float(ret.mean()) if len(ret) else np.nan,
                "median_net_per_closed_position": float(ret.median()) if len(ret) else np.nan,
                "sum_net_return_units": float(ret.sum()) if len(ret) else 0.0,
                "win_rate": float((ret > 0).mean()) if len(ret) else np.nan,
                "profit_factor": _profit_factor(ret),
                "max_days_without_trade": _max_days_without_event(pd.Series([t["exit_time"] for t in trades])) if trades else np.nan,
                "avg_layers": float(np.mean([t["layers"] for t in trades])) if trades else np.nan,
                "max_layers_used": int(max([t["layers"] for t in trades], default=0)),
                "max_hold_bars": int(max_hold_bars),
                "decision_hint": _path_decision_hint(ret),
            }
        )
        for t in trades:
            all_trades.append(t)

    trades_df = pd.DataFrame(all_trades)
    if not trades_df.empty and int(save_trades) > 0:
        trades_df = trades_df.sort_values(["system_id", "exit_time"]).head(int(save_trades)).copy()
    return pd.DataFrame(summary_rows).sort_values("sum_net_return_units", ascending=False), trades_df


def _close_layers(
    trades: list[dict[str, object]],
    *,
    spec: PathSpec,
    side: int,
    entry_prices: list[float],
    entry_positions: list[int],
    exit_pos: int,
    exit_price: float,
    exit_reason: str,
    frame: pd.DataFrame,
    round_trip_cost_pct: float,
) -> None:
    if not entry_prices or not np.isfinite(exit_price) or exit_price <= 0:
        return
    gross_layers = []
    for ep in entry_prices:
        gross_layers.append((exit_price / float(ep) - 1.0) * side if side == 1 else (float(ep) / exit_price - 1.0))
    gross_sum = float(np.sum(gross_layers))
    net_sum = gross_sum - float(round_trip_cost_pct) * len(entry_prices)
    trades.append(
        {
            "system_id": spec.system_id,
            "side": "LONG" if side == 1 else "SHORT",
            "entry_time_first": frame.index[int(min(entry_positions))],
            "entry_time_last": frame.index[int(max(entry_positions))],
            "exit_time": frame.index[int(exit_pos)],
            "layers": int(len(entry_prices)),
            "avg_entry_price": float(np.mean(entry_prices)),
            "exit_price": float(exit_price),
            "gross_return_sum": gross_sum,
            "net_return_sum": net_sum,
            "net_return_avg_layer": net_sum / float(len(entry_prices)),
            "hold_bars_first_layer": int(exit_pos - min(entry_positions)),
            "exit_reason": exit_reason,
        }
    )


def _path_decision_hint(ret: pd.Series) -> str:
    x = pd.to_numeric(ret, errors="coerce").dropna()
    if len(x) < 50:
        return "diagnostic_only_low_trade_count"
    if float(x.mean()) > 0 and _profit_factor(x) > 1.2:
        return "path_promising_needs_formal_backtest"
    return "path_not_promising_as_defined"


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"[write] {path.as_posix()} rows={len(df):,}", flush=True)


def write_reports(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    thresholds: StructureThresholds,
    features: pd.DataFrame,
    events: pd.DataFrame,
    baseline: pd.DataFrame,
    matched: pd.DataFrame,
    shortlist: pd.DataFrame,
    rejected: pd.DataFrame,
    causal_audit: pd.DataFrame,
    path_summary: pd.DataFrame,
    path_trades: pd.DataFrame,
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

    sample_cols = [
        "event_id",
        "event_name",
        "family",
        "direction",
        "structure",
        "signal_bar_time",
        "signal_time",
        "entry_time",
        "entry_price",
        "session",
        "vol_regime",
        "trend_regime",
        "close_pos",
        "volume_ratio",
        "lower_wick_share",
        "upper_wick_share",
        "lower_wick_atr",
        "upper_wick_atr",
        "ret_30",
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
        "causal_policy": "closed 1m signal bar, signal_available_time=bar_start+1m, entry at next bar open; no MTF context",
        "anti_overfit_policy": "fixed structural hypotheses only; no Cartesian parameter grid; report matched baseline/cost/delay/year/session/regime",
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
    _write_csv(path_summary, out_dir / "15_sequence_path_probe.csv")
    _write_csv(path_trades, out_dir / "16_sequence_trades_sample.csv")
    if bool(args.write_full_events):
        _write_csv(events, out_dir / "17_full_events_with_forward_returns.csv")

    notes = f"""# {TITLE}

This is a research-only pack. It does **not** register a tradable edge.

## Hypothesis
Volume climax + very long 1m shadows may indicate forced-flow exhaustion, but
only some structural contexts may have value:

- close-location reclaim/reject,
- prior directional extension,
- trend pullback,
- repeated test near the same extreme,
- order-flow absorption,
- failed reclaim/reject continuation.

## Anti-overfit boundary
This script does not run a wick/volume parameter grid. It tests a fixed event
vocabulary and judges it with matched baseline, cost stress, delay stress,
yearly/monthly/session/regime breakdowns and causal audit.

## Causal boundary
1m bars are left-labeled. Each event uses a closed signal bar. The report stores
`signal_available_time = signal_bar_time + 1 minute` and `entry_time = next bar open`.
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
    print("[scope] fixed structural hypotheses; no parameter-grid search", flush=True)

    bars = load_trade_bars(args)
    features = build_features(bars, thresholds)
    # Only generate/report events in the research window. Warmup is feature-only.
    research_features = features.loc[(features.index >= pd.Timestamp(args.start_date)) & (features.index <= pd.Timestamp(args.end_date))].copy()
    events = build_structure_events(research_features, thresholds)
    events = attach_forward_returns(
        bars=features,
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
        print("[causal] no lookahead flags in next-open audit", flush=True)

    baseline = build_all_bar_baseline(
        research_features,
        horizons=horizons,
        round_trip_cost_pct=float(args.round_trip_cost_pct),
        primary_horizon=int(args.primary_horizon),
    )
    matched = matched_baseline_summary(events, baseline, primary_horizon=int(args.primary_horizon), min_count=int(args.min_count))
    shortlist, rejected = build_candidate_shortlist(events, matched, primary_horizon=int(args.primary_horizon), min_count=int(args.min_count))
    path_summary, path_trades = run_path_probe(
        features,
        events,
        max_layers=int(args.path_max_layers),
        max_hold_bars=int(args.path_max_hold_bars),
        round_trip_cost_pct=float(args.round_trip_cost_pct),
        save_trades=int(args.save_path_trades),
    )

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
        path_summary=path_summary,
        path_trades=path_trades,
        horizons=horizons,
        cost_multipliers=cost_multipliers,
        delay_bars_list=delay_bars_list,
    )

    print("[done] research completed; inspect gpt_review_pack.zip before deciding next step", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
