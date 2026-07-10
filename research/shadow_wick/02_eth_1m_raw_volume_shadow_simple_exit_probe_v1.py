#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH 1m raw volume-shadow path atlas V1.

Research-only diagnostic atlas. This script deliberately goes back to the raw
volume long-shadow universe instead of starting from the already narrowed panic
wick candidate. It maps what happens after simple lower/upper volume-shadow
signals using MFE/MAE, sweep/reclaim path features, and holdability labels.

Scope
-----
- lower volume shadow -> diagnostic LONG path
- upper volume shadow -> diagnostic SHORT path
- two fixed strength layers: standard and extreme
- no large parameter grid; thresholds are fixed vocabulary controls
- no live strategy, no portfolio overlay, no AetherEdge changes

Causal policy
-------------
1m bars are left-labeled by bar start time. An event bar is only known after the
bar closes. signal_time = event_bar_start + 1 minute. entry_time is the next
bar open, optionally plus a small diagnostic entry-delay. Path MFE/MAE uses
future bars only after the causal entry time. No multi-timeframe context is used.
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

SCRIPT_NAME = "eth_1m_raw_volume_shadow_simple_exit_probe_v1"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_MF_1M_RAW_VOLUME_SHADOW_SIMPLE_EXIT_PROBE_V1"
EDGE_ID = "RESEARCH_ONLY_ETH_MF_1M_RAW_VOLUME_SHADOW_SIMPLE_EXIT_PROBE_V1"
TITLE = "ETH 1m Raw Volume Shadow Simple Exit Probe V1"
DEFAULT_OUT_DIR = "data/reports/research/eth_1m_raw_volume_shadow_simple_exit_probe_v1"
BAR_DELTA = pd.Timedelta(minutes=1)


@dataclass(frozen=True)
class ShadowThresholds:
    """Fixed raw-shadow vocabulary thresholds. These are not swept."""

    wick_share_min: float = 0.50
    wick_atr_min: float = 0.55
    volume_ratio_min: float = 2.0
    extreme_wick_share_min: float = 0.65
    extreme_wick_atr_min: float = 0.85
    extreme_volume_ratio_min: float = 3.0


@dataclass(frozen=True)
class PathLabelThresholds:
    """Diagnostic label thresholds; used for path atlas labels, not entry rules."""

    worth_holding_mfe: float = 0.006
    quick_rebound_mfe: float = 0.004
    quick_rebound_bars: int = 15
    controlled_mae_floor: float = -0.006
    dead_mfe_ceiling: float = 0.003
    dead_mae_floor: float = -0.010
    adverse_dwell_bars: int = 3
    slow_reclaim_bars: int = 30


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Raw ETH 1m volume-shadow simple exit hypothesis probe for lower-long and upper-short signals.",
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
    p.add_argument("--horizons", default="5,15,30,60,120,240")
    p.add_argument("--max-holding-bars", type=int, default=240, help="Diagnostic path cap for unresolved trades; this is not a promoted time-stop exit.")
    p.add_argument("--entry-delay-bars-list", default="0,1,2")
    p.add_argument("--event-sample-size", type=int, default=5000)
    p.add_argument("--path-sample-size", type=int, default=10000)
    p.add_argument("--progress-every", type=int, default=5000)
    p.add_argument("--skip-full-path", action="store_true", help="Do not write the full slim path table; samples and aggregates are still written.")
    p.add_argument("--skip-full-exit-trades", action="store_true", help="Do not write 11_full_exit_trades.csv; samples and aggregates are still written.")

    # Fixed vocabulary thresholds; keep defaults for comparable reports.
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
        raise RuntimeError(f"Loaded trade bars missing required columns: {missing}")
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    print(f"       rows={len(out):,} range={out.index.min()} -> {out.index.max()}", flush=True)
    return out


def build_features(bars: pd.DataFrame, th: ShadowThresholds) -> pd.DataFrame:
    print("[features] building raw volume-shadow geometry and regimes", flush=True)
    df = bars.copy().sort_index()
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    rng = (df["high"] - df["low"]).replace(0.0, np.nan)
    body_low = df[["open", "close"]].min(axis=1)
    body_high = df[["open", "close"]].max(axis=1)
    out = pd.DataFrame(index=df.index)
    out[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]]
    out["range_pct"] = _safe_divide(rng, df["close"]).to_numpy()
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
    signal_bar_time = features.index[event_pos]
    return pd.DataFrame(
        {
            "event_name": event_name,
            "family": family,
            "direction": direction,
            "side": int(side),
            "strength": strength,
            "event_bar_time": idx,
            "event_bar_pos": event_pos,
            "signal_bar_time": signal_bar_time,
            "signal_bar_pos": event_pos,
            "signal_time": signal_bar_time + BAR_DELTA,
            "signal_available_time": signal_bar_time + BAR_DELTA,
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
            "event_low": f["low"].to_numpy(dtype=float),
            "event_high": f["high"].to_numpy(dtype=float),
            "event_open": f["open"].to_numpy(dtype=float),
            "event_close": f["close"].to_numpy(dtype=float),
        }
    )


def build_raw_shadow_events(features: pd.DataFrame) -> pd.DataFrame:
    print("[events] building raw lower-long and upper-short shadow atlas events", flush=True)
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
    out[valid] = arr[pos[valid]]
    return out


def _bucket_sweep_count(x: pd.Series) -> pd.Series:
    s = pd.to_numeric(x, errors="coerce").fillna(0).astype(int)
    return pd.Series(np.select([s == 0, s == 1, s == 2, s >= 3], ["0", "1", "2", ">=3"], default="NA"), index=x.index)


def _label_holdability(row: pd.Series, th: PathLabelThresholds) -> str:
    mfe = float(row.get("mfe", np.nan))
    mae = float(row.get("mae", np.nan))
    time_to_mfe = row.get("time_to_mfe_bars", np.nan)
    adverse_sweep = bool(row.get("adverse_sweep_flag", False))
    favorable_reclaim = bool(row.get("favorable_reclaim_flag", False))
    adverse_dwell = int(row.get("bars_beyond_adverse_level", 0) or 0)
    bars_to_reclaim = row.get("bars_to_favorable_reclaim", np.nan)
    if np.isfinite(mfe) and np.isfinite(mae):
        if mfe >= th.worth_holding_mfe and mae >= th.controlled_mae_floor and favorable_reclaim:
            return "worth_holding_clean"
        if mfe >= th.quick_rebound_mfe and np.isfinite(time_to_mfe) and int(time_to_mfe) <= th.quick_rebound_bars:
            return "quick_rebound"
        if adverse_sweep and favorable_reclaim and np.isfinite(bars_to_reclaim) and int(bars_to_reclaim) >= th.slow_reclaim_bars:
            return "slow_reclaim_after_sweep"
        if adverse_sweep and (not favorable_reclaim) and adverse_dwell >= th.adverse_dwell_bars:
            return "dump_early_failed_reclaim"
        if mfe <= th.dead_mfe_ceiling and mae <= th.dead_mae_floor:
            return "dead_trade_deep_mae_low_mfe"
    return "mixed_or_unclear"


def build_path_trades(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    entry_delays: tuple[int, ...],
    round_trip_cost_pct: float,
    progress_every: int,
) -> pd.DataFrame:
    print("[path] attaching causal forward MFE/MAE path diagnostics", flush=True)
    if events.empty:
        return pd.DataFrame()
    max_h = max(horizons)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    open_ = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    index = bars.index

    fwd_high: dict[int, np.ndarray] = {}
    fwd_low: dict[int, np.ndarray] = {}
    for h in horizons:
        fwd_high[h] = _forward_roll_max(high, h)
        fwd_low[h] = _forward_roll_min(low, h)

    frames: list[pd.DataFrame] = []
    label_th = PathLabelThresholds()
    total_jobs = len(entry_delays)
    progress = ProgressReporter(label="[path] entry delays", total=total_jobs, every=1)
    job = 0
    for delay in entry_delays:
        e = events.copy()
        e["entry_delay_bars"] = int(delay)
        entry_pos = e["signal_bar_pos"].to_numpy(dtype=int) + 1 + int(delay)
        e["entry_bar_pos"] = entry_pos
        valid = entry_pos + max_h < len(bars)
        e = e.loc[valid].copy()
        if e.empty:
            job += 1
            progress.update(job)
            continue
        entry_pos = e["entry_bar_pos"].to_numpy(dtype=int)
        side = e["side"].to_numpy(dtype=int)
        entry_price = _safe_take(open_, entry_pos)
        e["entry_time"] = index[entry_pos]
        e["entry_price"] = entry_price
        e["expected_entry_time"] = e["signal_time"] + pd.to_timedelta(int(delay), unit="m")
        e["entry_not_expected_time_flag"] = pd.to_datetime(e["entry_time"]) != pd.to_datetime(e["expected_entry_time"])
        e["lookahead_flag"] = pd.to_datetime(e["entry_time"]) < pd.to_datetime(e["signal_time"])

        for h in horizons:
            hh = int(h)
            end_pos = entry_pos + hh
            end_close = _safe_take(close, end_pos)
            max_high = _safe_take(fwd_high[hh], entry_pos)
            min_low = _safe_take(fwd_low[hh], entry_pos)
            gross_close = np.where(side > 0, end_close / entry_price - 1.0, entry_price / end_close - 1.0)
            mfe = np.where(side > 0, max_high / entry_price - 1.0, entry_price / min_low - 1.0)
            mae = np.where(side > 0, min_low / entry_price - 1.0, entry_price / max_high - 1.0)
            e[f"gross_return_h{hh}"] = gross_close
            e[f"net_return_h{hh}"] = gross_close - float(round_trip_cost_pct)
            e[f"mfe_h{hh}"] = mfe
            e[f"mae_h{hh}"] = mae

        # Primary max-horizon path features with one compact Python loop. This is
        # diagnostic only; strategy simulation is not repeated here.
        records: list[dict[str, object]] = []
        ev_low = e["event_low"].to_numpy(dtype=float)
        ev_high = e["event_high"].to_numpy(dtype=float)
        ev_mid = (ev_low + ev_high) / 2.0
        side_arr = side
        prog = ProgressReporter(label=f"[path] detailed delay={delay}", total=len(e), every=max(1, int(progress_every)))
        for i, row in enumerate(e.itertuples(index=False)):
            ep = int(getattr(row, "entry_bar_pos"))
            sd = int(getattr(row, "side"))
            entry = float(getattr(row, "entry_price"))
            hi = high[ep : ep + max_h + 1]
            lo = low[ep : ep + max_h + 1]
            cl = close[ep : ep + max_h + 1]
            if len(hi) == 0 or not np.isfinite(entry) or entry <= 0:
                records.append({})
                continue
            if sd > 0:
                mfe_path = hi / entry - 1.0
                mae_path = lo / entry - 1.0
                adverse_hit = lo < ev_low[i]
                beyond_adverse_close = cl < ev_low[i]
                favorable_reclaim_close = cl >= ev_high[i]
                mid_reclaim_close = cl >= ev_mid[i]
                depth_beyond = np.where(adverse_hit, ev_low[i] / np.maximum(lo, 1e-12) - 1.0, 0.0)
            else:
                mfe_path = entry / np.maximum(lo, 1e-12) - 1.0
                mae_path = entry / np.maximum(hi, 1e-12) - 1.0
                adverse_hit = hi > ev_high[i]
                beyond_adverse_close = cl > ev_high[i]
                favorable_reclaim_close = cl <= ev_low[i]
                mid_reclaim_close = cl <= ev_mid[i]
                depth_beyond = np.where(adverse_hit, hi / max(ev_high[i], 1e-12) - 1.0, 0.0)
            mfe_idx = int(np.nanargmax(mfe_path)) if np.isfinite(mfe_path).any() else -1
            mae_idx = int(np.nanargmin(mae_path)) if np.isfinite(mae_path).any() else -1
            sweep_count = int(np.nansum(adverse_hit))
            first_sweep = int(np.argmax(adverse_hit)) if adverse_hit.any() else -1
            reclaim_flag = bool(favorable_reclaim_close.any())
            mid_reclaim_flag = bool(mid_reclaim_close.any())
            bars_to_reclaim = int(np.argmax(favorable_reclaim_close)) if favorable_reclaim_close.any() else -1
            bars_to_mid_reclaim = int(np.argmax(mid_reclaim_close)) if mid_reclaim_close.any() else -1
            records.append(
                {
                    "mfe": float(np.nanmax(mfe_path)),
                    "mae": float(np.nanmin(mae_path)),
                    "time_to_mfe_bars": mfe_idx,
                    "time_to_mae_bars": mae_idx,
                    "mfe_before_mae": bool(mfe_idx >= 0 and mae_idx >= 0 and mfe_idx <= mae_idx),
                    "adverse_sweep_flag": bool(sweep_count > 0),
                    "adverse_sweep_count": sweep_count,
                    "bars_to_first_adverse_sweep": first_sweep,
                    "bars_beyond_adverse_level": int(np.nansum(beyond_adverse_close)),
                    "max_consecutive_bars_beyond_adverse": _max_consecutive_true(beyond_adverse_close),
                    "max_depth_beyond_adverse_pct": float(np.nanmax(depth_beyond)) if len(depth_beyond) else np.nan,
                    "favorable_reclaim_flag": reclaim_flag,
                    "bars_to_favorable_reclaim": bars_to_reclaim,
                    "mid_reclaim_flag": mid_reclaim_flag,
                    "bars_to_mid_reclaim": bars_to_mid_reclaim,
                }
            )
            prog.update(i + 1)
        prog.close()
        detail = pd.DataFrame(records)
        e = pd.concat([e.reset_index(drop=True), detail], axis=1)
        e["primary_horizon"] = int(max_h)
        e["primary_net_return"] = e[f"net_return_h{max_h}"]
        e["primary_gross_return"] = e[f"gross_return_h{max_h}"]
        e["mfe_capture_ratio"] = np.where(
            pd.to_numeric(e["mfe"], errors="coerce") > 0,
            pd.to_numeric(e["primary_net_return"], errors="coerce") / pd.to_numeric(e["mfe"], errors="coerce"),
            np.nan,
        )
        e["sweep_bucket"] = _bucket_sweep_count(e["adverse_sweep_count"])
        e["holdability_label"] = e.apply(lambda r: _label_holdability(r, label_th), axis=1)
        frames.append(e)
        job += 1
        progress.update(job)
    progress.close()
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if out.empty:
        return out
    out["direction_label"] = np.where(out["side"].astype(int) > 0, "LONG_lower_shadow", "SHORT_upper_shadow")
    return out


def _max_consecutive_true(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask.astype(bool):
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def summarize_return(ret: pd.Series) -> dict[str, float | int]:
    r = pd.to_numeric(ret, errors="coerce").dropna()
    base: dict[str, float | int] = {
        "count": int(len(r)),
        "mean_net": float(r.mean()) if len(r) else np.nan,
        "median_net": float(r.median()) if len(r) else np.nan,
        "win_rate": float((r > 0).mean()) if len(r) else np.nan,
        "profit_factor": _profit_factor(r),
        "top5_winner_share": _top5_winner_share(r),
        "max_consecutive_losses": _max_consecutive_losses(r),
    }
    base.update(_equity_stats(r))
    if len(r):
        wins = r[r > 0]
        losses = r[r <= 0]
        base.update(
            {
                "avg_win": float(wins.mean()) if len(wins) else np.nan,
                "avg_loss": float(losses.mean()) if len(losses) else np.nan,
                "payoff_ratio": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() < 0 else np.nan,
                "p5_return": float(r.quantile(0.05)),
                "p95_return": float(r.quantile(0.95)),
                "max_single_loss": float(r.min()),
                "max_single_win": float(r.max()),
            }
        )
    return base


def _path_stats(part: pd.DataFrame) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in ["mfe", "mae", "time_to_mfe_bars", "time_to_mae_bars", "mfe_capture_ratio", "adverse_sweep_count", "bars_beyond_adverse_level", "max_depth_beyond_adverse_pct"]:
        raw_col = part[col] if col in part.columns else pd.Series(np.nan, index=part.index)
        s = pd.to_numeric(raw_col, errors="coerce")
        out[f"{col}_mean"] = float(s.mean()) if len(s.dropna()) else np.nan
        out[f"{col}_median"] = float(s.median()) if len(s.dropna()) else np.nan
    for col in ["mfe", "mae"]:
        raw_col = part[col] if col in part.columns else pd.Series(np.nan, index=part.index)
        s = pd.to_numeric(raw_col, errors="coerce")
        out[f"{col}_p25"] = float(s.quantile(0.25)) if len(s.dropna()) else np.nan
        out[f"{col}_p75"] = float(s.quantile(0.75)) if len(s.dropna()) else np.nan
        out[f"{col}_p90"] = float(s.quantile(0.90)) if len(s.dropna()) else np.nan
    for col in ["adverse_sweep_flag", "favorable_reclaim_flag", "mid_reclaim_flag", "mfe_before_mae"]:
        if col in part.columns:
            out[f"{col}_share"] = float(pd.Series(part[col]).astype(bool).mean()) if len(part) else np.nan
    return out


def build_event_counts(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, part in events.groupby(["event_name", "direction", "strength"], observed=False):
        event_name, direction, strength = keys
        rows.append(
            {
                "event_name": event_name,
                "direction": direction,
                "strength": strength,
                "count": int(len(part)),
                "events_per_month": _events_per_month(part["signal_time"]),
                "max_days_without_event": _max_days_without_event(part["signal_time"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["direction", "strength", "count"], ascending=[True, True, False])


def build_path_summary(paths: pd.DataFrame, horizons: tuple[int, ...]) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["event_name", "direction", "strength", "entry_delay_bars"]
    for keys, part in paths.groupby(group_cols, observed=False):
        base = dict(zip(group_cols, keys))
        for h in horizons:
            ret_col = f"net_return_h{int(h)}"
            mfe_col = f"mfe_h{int(h)}"
            mae_col = f"mae_h{int(h)}"
            row = {**base, "horizon_bars": int(h), **summarize_return(part[ret_col])}
            row.update(
                {
                    "events_per_month": _events_per_month(part["entry_time"]),
                    "max_days_without_event": _max_days_without_event(part["entry_time"]),
                    "mfe_mean": float(pd.to_numeric(part[mfe_col], errors="coerce").mean()),
                    "mfe_median": float(pd.to_numeric(part[mfe_col], errors="coerce").median()),
                    "mfe_p75": float(pd.to_numeric(part[mfe_col], errors="coerce").quantile(0.75)),
                    "mfe_p90": float(pd.to_numeric(part[mfe_col], errors="coerce").quantile(0.90)),
                    "mae_mean": float(pd.to_numeric(part[mae_col], errors="coerce").mean()),
                    "mae_median": float(pd.to_numeric(part[mae_col], errors="coerce").median()),
                    "mae_p25": float(pd.to_numeric(part[mae_col], errors="coerce").quantile(0.25)),
                    "mae_p10": float(pd.to_numeric(part[mae_col], errors="coerce").quantile(0.10)),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["horizon_bars", "mean_net"], ascending=[True, False], na_position="last")


def build_mfe_mae_by_outcome(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    tmp = paths.copy()
    tmp["outcome"] = np.where(pd.to_numeric(tmp["primary_net_return"], errors="coerce") > 0, "winner", "loser")
    rows: list[dict[str, object]] = []
    for keys, part in tmp.groupby(["event_name", "direction", "strength", "entry_delay_bars", "outcome"], observed=False):
        event_name, direction, strength, delay, outcome = keys
        rows.append(
            {
                "event_name": event_name,
                "direction": direction,
                "strength": strength,
                "entry_delay_bars": int(delay),
                "outcome": outcome,
                **summarize_return(part["primary_net_return"]),
                **_path_stats(part),
            }
        )
    return pd.DataFrame(rows)


def build_sweep_reclaim_summary(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, part in paths.groupby(["event_name", "direction", "strength", "entry_delay_bars", "sweep_bucket", "favorable_reclaim_flag"], observed=False):
        event_name, direction, strength, delay, sweep_bucket, reclaim = keys
        rows.append(
            {
                "event_name": event_name,
                "direction": direction,
                "strength": strength,
                "entry_delay_bars": int(delay),
                "sweep_bucket": sweep_bucket,
                "favorable_reclaim_flag": bool(reclaim),
                **summarize_return(part["primary_net_return"]),
                **_path_stats(part),
            }
        )
    return pd.DataFrame(rows).sort_values(["event_name", "entry_delay_bars", "sweep_bucket", "favorable_reclaim_flag"])


def build_holdability_labels(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, part in paths.groupby(["event_name", "direction", "strength", "entry_delay_bars", "holdability_label"], observed=False):
        event_name, direction, strength, delay, label = keys
        rows.append(
            {
                "event_name": event_name,
                "direction": direction,
                "strength": strength,
                "entry_delay_bars": int(delay),
                "holdability_label": label,
                **summarize_return(part["primary_net_return"]),
                **_path_stats(part),
            }
        )
    return pd.DataFrame(rows).sort_values(["event_name", "entry_delay_bars", "mean_net"], ascending=[True, True, False], na_position="last")


def build_direction_comparison(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, part in paths.groupby(["direction_label", "strength", "entry_delay_bars"], observed=False):
        direction_label, strength, delay = keys
        yearly = part.assign(year=pd.to_datetime(part["entry_time"]).dt.year).groupby("year", observed=False)["primary_net_return"].mean()
        rows.append(
            {
                "direction_label": direction_label,
                "strength": strength,
                "entry_delay_bars": int(delay),
                **summarize_return(part["primary_net_return"]),
                **_path_stats(part),
                "positive_years": int((yearly > 0).sum()),
                "year_count": int(yearly.notna().sum()),
                "events_per_month": _events_per_month(part["entry_time"]),
                "max_days_without_event": _max_days_without_event(part["entry_time"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["strength", "mean_net"], ascending=[True, False], na_position="last")


def build_yearly_path_summary(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    tmp = paths.copy()
    tmp["year"] = pd.to_datetime(tmp["entry_time"]).dt.year
    rows: list[dict[str, object]] = []
    for keys, part in tmp.groupby(["event_name", "direction", "strength", "entry_delay_bars", "year"], observed=False):
        event_name, direction, strength, delay, year = keys
        rows.append(
            {
                "event_name": event_name,
                "direction": direction,
                "strength": strength,
                "entry_delay_bars": int(delay),
                "year": int(year),
                **summarize_return(part["primary_net_return"]),
                **_path_stats(part),
            }
        )
    return pd.DataFrame(rows)


def build_session_regime_summary(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["event_name", "direction", "strength", "entry_delay_bars", "session", "vol_regime", "trend_regime"]
    for keys, part in paths.groupby(group_cols, observed=False):
        rows.append({**dict(zip(group_cols, keys)), **summarize_return(part["primary_net_return"]), **_path_stats(part)})
    return pd.DataFrame(rows).sort_values(["count", "mean_net"], ascending=[False, False], na_position="last")


def build_causal_audit(paths: pd.DataFrame) -> pd.DataFrame:
    if paths.empty:
        return pd.DataFrame()
    cols = [
        "event_name",
        "direction",
        "entry_delay_bars",
        "event_bar_time",
        "signal_time",
        "signal_available_time",
        "entry_time",
        "expected_entry_time",
        "event_bar_pos",
        "signal_bar_pos",
        "entry_bar_pos",
        "entry_not_expected_time_flag",
        "lookahead_flag",
    ]
    out = paths[[c for c in cols if c in paths.columns]].copy()
    out["context_available_time_flag"] = False
    return out


def build_upgrade_decision_map(paths: pd.DataFrame) -> pd.DataFrame:
    """Heuristic diagnostic map. This does not promote a strategy."""
    if paths.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, part in paths.groupby(["event_name", "direction", "strength", "entry_delay_bars"], observed=False):
        event_name, direction, strength, delay = keys
        s = summarize_return(part["primary_net_return"])
        ps = _path_stats(part)
        suggestions: list[str] = []
        reason: list[str] = []
        if ps.get("mfe_mean", np.nan) and ps.get("mfe_capture_ratio_median", np.nan):
            pass
        capture = float(pd.to_numeric(part["mfe_capture_ratio"], errors="coerce").median()) if len(part) else np.nan
        sweep_share = float(pd.Series(part["adverse_sweep_flag"]).astype(bool).mean()) if len(part) else np.nan
        reclaim_after_sweep = part.loc[pd.Series(part["adverse_sweep_flag"]).astype(bool)]
        reclaim_after_sweep_share = float(pd.Series(reclaim_after_sweep["favorable_reclaim_flag"]).astype(bool).mean()) if len(reclaim_after_sweep) else np.nan
        if np.isfinite(capture) and capture < 0.35 and ps.get("mfe_mean", 0) > 0.005:
            suggestions.append("improve_exit_capture")
            reason.append("MFE exists but median capture is low")
        if np.isfinite(sweep_share) and sweep_share > 0.25:
            suggestions.append("analyze_sweep_tolerance")
            reason.append("adverse sweep is common")
        if np.isfinite(reclaim_after_sweep_share) and reclaim_after_sweep_share < 0.55:
            suggestions.append("cut_failed_reclaim_after_sweep")
            reason.append("many swept trades fail to reclaim")
        if s.get("mean_net", 0) > 0 and s.get("profit_factor", 0) > 1.2:
            suggestions.append("candidate_family_has_edge")
            reason.append("primary horizon net expectancy and PF are positive")
        if direction == "SHORT" and s.get("mean_net", 0) <= 0 and ps.get("mfe_mean", 0) > 0.004:
            suggestions.append("short_has_mfe_but_exit_or_confirmation_problem")
            reason.append("short raw shadow has MFE but not net edge")
        rows.append(
            {
                "event_name": event_name,
                "direction": direction,
                "strength": strength,
                "entry_delay_bars": int(delay),
                **s,
                **ps,
                "mfe_capture_ratio_median": capture,
                "adverse_sweep_share": sweep_share,
                "reclaim_after_sweep_share": reclaim_after_sweep_share,
                "upgrade_suggestions": ";".join(dict.fromkeys(suggestions)) if suggestions else "no_clear_upgrade",
                "diagnostic_reason": "; ".join(reason) if reason else "No strong path diagnostic signal.",
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_net", "profit_factor"], ascending=[False, False], na_position="last")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def _sample(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if df.empty or len(df) <= n:
        return df.copy()
    return df.head(int(n)).copy()


def _pct_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    pct_like = [
        "mean_net", "median_net", "total_return", "max_drawdown", "win_rate", "top5_winner_share",
        "mfe_mean", "mfe_median", "mfe_p75", "mfe_p90", "mae_mean", "mae_median", "mae_p25", "mae_p10",
        "mfe", "mae", "primary_net_return", "primary_gross_return", "mfe_capture_ratio",
    ]
    for col in pct_like:
        if col in out.columns and f"{col}_pct" not in out.columns:
            out[f"{col}_pct"] = pd.to_numeric(out[col], errors="coerce") * 100.0
    return out


# ---------------------------------------------------------------------------
# Simple exit hypothesis probe
# ---------------------------------------------------------------------------

EXIT_POLICIES: tuple[str, ...] = (
    "hold_240_benchmark",
    "event_extreme_hard_stop",
    "close_beyond_extreme_stop",
    "allow_1_sweep_stop",
    "allow_2_sweep_stop",
    "sweep_no_reclaim_extreme_15",
    "sweep_no_reclaim_mid_15",
    "sweep_no_reclaim_full_15",
    "reclaim_mid_then_mid_fail",
    "reclaim_full_then_mid_fail",
)


def _exit_policy_family(policy: str) -> str:
    if policy == "hold_240_benchmark":
        return "benchmark_hold"
    if policy in {"event_extreme_hard_stop", "close_beyond_extreme_stop"}:
        return "hard_extreme_stop"
    if policy in {"allow_1_sweep_stop", "allow_2_sweep_stop"}:
        return "sweep_count_stop"
    if policy.startswith("sweep_no_reclaim"):
        return "sweep_reclaim_failure"
    if policy.startswith("reclaim_"):
        return "reclaim_quality_failure"
    return "unknown"


def _favorable_return(side: int, entry: float, exit_price: float) -> float:
    if not np.isfinite(entry) or not np.isfinite(exit_price) or entry <= 0 or exit_price <= 0:
        return float("nan")
    return float(exit_price / entry - 1.0) if side > 0 else float(entry / exit_price - 1.0)


def _path_mfe_mae(side: int, entry: float, hi: np.ndarray, lo: np.ndarray) -> tuple[float, float, int, int]:
    if len(hi) == 0 or not np.isfinite(entry) or entry <= 0:
        return np.nan, np.nan, -1, -1
    if side > 0:
        mfe_path = hi / entry - 1.0
        mae_path = lo / entry - 1.0
    else:
        mfe_path = entry / np.maximum(lo, 1e-12) - 1.0
        mae_path = entry / np.maximum(hi, 1e-12) - 1.0
    mfe_idx = int(np.nanargmax(mfe_path)) if np.isfinite(mfe_path).any() else -1
    mae_idx = int(np.nanargmin(mae_path)) if np.isfinite(mae_path).any() else -1
    return float(np.nanmax(mfe_path)), float(np.nanmin(mae_path)), mfe_idx, mae_idx


def _simulate_policy_on_path(
    *,
    policy: str,
    side: int,
    entry_pos: int,
    entry_price: float,
    event_low: float,
    event_high: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    open_: np.ndarray,
    index: pd.DatetimeIndex,
    max_holding_bars: int,
    round_trip_cost_pct: float,
) -> dict[str, object]:
    """Simulate one simple structural exit policy.

    All structural decisions are made from closed 1m bars after entry. Exits are
    executed at the next available open. `hold_240_benchmark` and unresolved
    cases use the max-holding cap as a diagnostic closure only, not a promoted
    time-stop strategy.
    """
    max_end = min(int(entry_pos) + int(max_holding_bars), len(close) - 2)
    if max_end <= entry_pos or not np.isfinite(entry_price) or entry_price <= 0:
        return {}

    event_mid = (float(event_low) + float(event_high)) / 2.0
    sweep_count = 0
    first_sweep_bar = -1
    last_sweep_bar = -1
    first_extreme_reclaim_bar = -1
    first_mid_reclaim_bar = -1
    first_full_reclaim_bar = -1
    consecutive_beyond = 0
    max_consecutive_beyond = 0
    max_depth_beyond = 0.0
    best_mfe_seen = 0.0
    exit_bar = max_end
    exit_exec_pos = max_end
    exit_reason = "path_cap_unresolved"
    exit_price_kind = "cap_close"
    structural_decision_bar = -1
    reclaimed_mid = False
    reclaimed_full = False

    for pos in range(int(entry_pos), max_end + 1):
        if side > 0:
            adverse_hit = bool(low[pos] < event_low)
            beyond_close = bool(close[pos] < event_low)
            extreme_reclaim = bool(close[pos] >= event_low)
            mid_reclaim = bool(close[pos] >= event_mid)
            full_reclaim = bool(close[pos] >= event_high)
            depth = float(event_low / max(low[pos], 1e-12) - 1.0) if adverse_hit else 0.0
            bar_mfe = float(high[pos] / entry_price - 1.0)
        else:
            adverse_hit = bool(high[pos] > event_high)
            beyond_close = bool(close[pos] > event_high)
            extreme_reclaim = bool(close[pos] <= event_high)
            mid_reclaim = bool(close[pos] <= event_mid)
            full_reclaim = bool(close[pos] <= event_low)
            depth = float(high[pos] / max(event_high, 1e-12) - 1.0) if adverse_hit else 0.0
            bar_mfe = float(entry_price / max(low[pos], 1e-12) - 1.0)

        best_mfe_seen = max(best_mfe_seen, bar_mfe)
        if adverse_hit:
            sweep_count += 1
            last_sweep_bar = pos
            if first_sweep_bar < 0:
                first_sweep_bar = pos
            max_depth_beyond = max(max_depth_beyond, depth)
        if beyond_close:
            consecutive_beyond += 1
            max_consecutive_beyond = max(max_consecutive_beyond, consecutive_beyond)
        else:
            consecutive_beyond = 0
        if extreme_reclaim and first_extreme_reclaim_bar < 0:
            first_extreme_reclaim_bar = pos
        if mid_reclaim and first_mid_reclaim_bar < 0:
            first_mid_reclaim_bar = pos
        if full_reclaim and first_full_reclaim_bar < 0:
            first_full_reclaim_bar = pos
        reclaimed_mid = reclaimed_mid or mid_reclaim
        reclaimed_full = reclaimed_full or full_reclaim

        should_exit = False
        reason = ""
        if policy == "hold_240_benchmark":
            should_exit = False
        elif policy == "event_extreme_hard_stop":
            should_exit = adverse_hit
            reason = "event_extreme_touched"
        elif policy == "close_beyond_extreme_stop":
            should_exit = beyond_close
            reason = "close_beyond_event_extreme"
        elif policy == "allow_1_sweep_stop":
            should_exit = sweep_count >= 2
            reason = "second_adverse_sweep"
        elif policy == "allow_2_sweep_stop":
            should_exit = sweep_count >= 3
            reason = "third_adverse_sweep"
        elif policy == "sweep_no_reclaim_extreme_15":
            if first_sweep_bar >= 0 and pos >= first_sweep_bar + 15 and first_extreme_reclaim_bar < 0:
                should_exit = True
                reason = "sweep_no_extreme_reclaim_15"
        elif policy == "sweep_no_reclaim_mid_15":
            if first_sweep_bar >= 0 and pos >= first_sweep_bar + 15 and first_mid_reclaim_bar < 0:
                should_exit = True
                reason = "sweep_no_mid_reclaim_15"
        elif policy == "sweep_no_reclaim_full_15":
            if first_sweep_bar >= 0 and pos >= first_sweep_bar + 15 and first_full_reclaim_bar < 0:
                should_exit = True
                reason = "sweep_no_full_reclaim_15"
        elif policy == "reclaim_mid_then_mid_fail":
            if reclaimed_mid and pos > max(first_mid_reclaim_bar, entry_pos):
                if side > 0:
                    should_exit = bool(close[pos] < event_mid)
                else:
                    should_exit = bool(close[pos] > event_mid)
                reason = "mid_reclaim_then_mid_fail"
        elif policy == "reclaim_full_then_mid_fail":
            if reclaimed_full and pos > max(first_full_reclaim_bar, entry_pos):
                if side > 0:
                    should_exit = bool(close[pos] < event_mid)
                else:
                    should_exit = bool(close[pos] > event_mid)
                reason = "full_reclaim_then_mid_fail"
        else:
            raise ValueError(f"Unknown exit policy: {policy}")

        if should_exit:
            exec_pos = min(pos + 1, len(open_) - 1)
            exit_bar = pos
            exit_exec_pos = exec_pos
            exit_reason = reason
            exit_price_kind = "next_open_after_closed_bar_signal"
            structural_decision_bar = pos
            break

    if exit_reason == "path_cap_unresolved":
        exit_price = float(close[exit_exec_pos])
        exit_time = index[exit_exec_pos]
    else:
        exit_price = float(open_[exit_exec_pos])
        exit_time = index[exit_exec_pos]
    hi_path = high[int(entry_pos) : int(exit_exec_pos) + 1]
    lo_path = low[int(entry_pos) : int(exit_exec_pos) + 1]
    mfe, mae, time_to_mfe, time_to_mae = _path_mfe_mae(side, entry_price, hi_path, lo_path)
    gross = _favorable_return(side, entry_price, exit_price)
    net = gross - float(round_trip_cost_pct) if np.isfinite(gross) else np.nan
    return {
        "exit_policy": policy,
        "exit_policy_family": _exit_policy_family(policy),
        "exit_bar_pos": int(exit_bar),
        "exit_exec_pos": int(exit_exec_pos),
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": exit_reason,
        "exit_price_kind": exit_price_kind,
        "structural_decision_bar_pos": int(structural_decision_bar),
        "gross_return": gross,
        "net_return": net,
        "mfe": mfe,
        "mae": mae,
        "time_to_mfe_bars": time_to_mfe,
        "time_to_mae_bars": time_to_mae,
        "mfe_capture_ratio": float(net / mfe) if np.isfinite(net) and np.isfinite(mfe) and mfe > 0 else np.nan,
        "holding_bars": int(exit_exec_pos - entry_pos),
        "adverse_sweep_count": int(sweep_count),
        "first_sweep_bars_after_entry": int(first_sweep_bar - entry_pos) if first_sweep_bar >= 0 else -1,
        "last_sweep_bars_after_entry": int(last_sweep_bar - entry_pos) if last_sweep_bar >= 0 else -1,
        "max_depth_beyond_adverse_pct": float(max_depth_beyond),
        "max_consecutive_bars_beyond_adverse": int(max_consecutive_beyond),
        "first_extreme_reclaim_bars_after_entry": int(first_extreme_reclaim_bar - entry_pos) if first_extreme_reclaim_bar >= 0 else -1,
        "first_mid_reclaim_bars_after_entry": int(first_mid_reclaim_bar - entry_pos) if first_mid_reclaim_bar >= 0 else -1,
        "first_full_reclaim_bars_after_entry": int(first_full_reclaim_bar - entry_pos) if first_full_reclaim_bar >= 0 else -1,
        "extreme_reclaim_flag": bool(first_extreme_reclaim_bar >= 0),
        "mid_reclaim_flag": bool(first_mid_reclaim_bar >= 0),
        "full_reclaim_flag": bool(first_full_reclaim_bar >= 0),
    }


def build_exit_probe_trades(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    entry_delays: tuple[int, ...],
    max_holding_bars: int,
    round_trip_cost_pct: float,
    progress_every: int,
) -> pd.DataFrame:
    print("[probe] simulating one-hypothesis-at-a-time simple exits", flush=True)
    if events.empty:
        return pd.DataFrame()
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    open_ = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    index = bars.index
    meta_cols = [
        "event_id", "event_name", "family", "direction", "side", "strength", "event_bar_time", "event_bar_pos",
        "signal_bar_time", "signal_bar_pos", "signal_time", "signal_available_time", "session", "vol_regime", "trend_regime",
        "close_pos", "volume_ratio", "lower_wick_share", "upper_wick_share", "lower_wick_atr", "upper_wick_atr",
        "ret_30", "ret_120", "delta_ratio", "taker_buy_ratio", "event_low", "event_high", "event_open", "event_close",
    ]
    frames: list[pd.DataFrame] = []
    total_jobs = len(entry_delays) * len(EXIT_POLICIES)
    job_progress = ProgressReporter(label="[probe] policy grid", total=total_jobs, every=1)
    job = 0
    for delay in entry_delays:
        base = events.copy()
        entry_pos = base["signal_bar_pos"].to_numpy(dtype=int) + 1 + int(delay)
        valid = entry_pos + 2 < len(bars)
        base = base.loc[valid].copy()
        if base.empty:
            job += len(EXIT_POLICIES)
            job_progress.update(job)
            continue
        entry_pos = base["signal_bar_pos"].to_numpy(dtype=int) + 1 + int(delay)
        base["entry_delay_bars"] = int(delay)
        base["entry_bar_pos"] = entry_pos
        base["entry_time"] = index[entry_pos]
        base["entry_price"] = open_[entry_pos]
        base["expected_entry_time"] = base["signal_time"] + pd.to_timedelta(int(delay), unit="m")
        base["entry_not_expected_time_flag"] = pd.to_datetime(base["entry_time"]) != pd.to_datetime(base["expected_entry_time"])
        base["lookahead_flag"] = pd.to_datetime(base["entry_time"]) < pd.to_datetime(base["signal_time"])
        for policy in EXIT_POLICIES:
            records: list[dict[str, object]] = []
            progress = ProgressReporter(label=f"[probe] {policy} d={delay}", total=len(base), every=max(1, int(progress_every)))
            for i, row in enumerate(base.itertuples(index=False)):
                rec = _simulate_policy_on_path(
                    policy=policy,
                    side=int(getattr(row, "side")),
                    entry_pos=int(getattr(row, "entry_bar_pos")),
                    entry_price=float(getattr(row, "entry_price")),
                    event_low=float(getattr(row, "event_low")),
                    event_high=float(getattr(row, "event_high")),
                    high=high,
                    low=low,
                    close=close,
                    open_=open_,
                    index=index,
                    max_holding_bars=int(max_holding_bars),
                    round_trip_cost_pct=float(round_trip_cost_pct),
                )
                records.append(rec)
                progress.update(i + 1)
            progress.close()
            sim = pd.DataFrame(records)
            out = pd.concat([base[[c for c in meta_cols if c in base.columns]].reset_index(drop=True), base[["entry_delay_bars", "entry_bar_pos", "entry_time", "entry_price", "expected_entry_time", "entry_not_expected_time_flag", "lookahead_flag"]].reset_index(drop=True), sim], axis=1)
            out["is_benchmark"] = policy == "hold_240_benchmark"
            out["is_structural_exit"] = policy != "hold_240_benchmark"
            frames.append(out)
            job += 1
            job_progress.update(job)
    job_progress.close()
    if not frames:
        return pd.DataFrame()
    trades = pd.concat(frames, ignore_index=True)
    trades["direction_label"] = np.where(trades["side"].astype(int) > 0, "LONG_lower_shadow", "SHORT_upper_shadow")
    trades["exit_after_entry_flag"] = pd.to_datetime(trades["exit_time"]) >= pd.to_datetime(trades["entry_time"])
    return trades


def build_policy_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["event_name", "direction", "strength", "entry_delay_bars", "exit_policy", "exit_policy_family"]
    for keys, part in trades.groupby(group_cols, observed=False):
        row = dict(zip(group_cols, keys))
        yearly = part.assign(year=pd.to_datetime(part["entry_time"]).dt.year).groupby("year", observed=False)["net_return"].mean()
        monthly = part.assign(month=pd.to_datetime(part["entry_time"]).dt.to_period("M").astype(str)).groupby("month", observed=False)["net_return"].sum()
        row["is_benchmark"] = bool(row.get("exit_policy") == "hold_240_benchmark")
        row.update(summarize_return(part["net_return"]))
        row.update(_path_stats(part))
        row.update(
            {
                "events_per_month": _events_per_month(part["entry_time"]),
                "max_days_without_event": _max_days_without_event(part["entry_time"]),
                "positive_years": int((yearly > 0).sum()),
                "year_count": int(yearly.notna().sum()),
                "positive_months": int((monthly > 0).sum()),
                "month_count": int(monthly.notna().sum()),
                "monthly_positive_rate": float((monthly > 0).mean()) if len(monthly) else np.nan,
                "avg_holding_bars": float(pd.to_numeric(part["holding_bars"], errors="coerce").mean()),
                "median_holding_bars": float(pd.to_numeric(part["holding_bars"], errors="coerce").median()),
                "path_cap_exit_share": float((part["exit_reason"] == "path_cap_unresolved").mean()),
                "structural_exit_share": float((part["exit_reason"] != "path_cap_unresolved").mean()),
                "reclaim_mid_share": float(pd.Series(part["mid_reclaim_flag"]).astype(bool).mean()),
                "reclaim_full_share": float(pd.Series(part["full_reclaim_flag"]).astype(bool).mean()),
            }
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["is_benchmark", "mean_net"], ascending=[True, False], na_position="last") if "is_benchmark" in out.columns else out.sort_values(["mean_net"], ascending=[False], na_position="last")


def build_policy_yearly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    tmp = trades.copy()
    tmp["year"] = pd.to_datetime(tmp["entry_time"]).dt.year
    rows: list[dict[str, object]] = []
    for keys, part in tmp.groupby(["event_name", "direction", "strength", "entry_delay_bars", "exit_policy", "year"], observed=False):
        rows.append({**dict(zip(["event_name", "direction", "strength", "entry_delay_bars", "exit_policy", "year"], keys)), **summarize_return(part["net_return"])})
    return pd.DataFrame(rows)


def build_policy_monthly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    tmp = trades.copy()
    tmp["month"] = pd.to_datetime(tmp["entry_time"]).dt.to_period("M").astype(str)
    rows: list[dict[str, object]] = []
    for keys, part in tmp.groupby(["event_name", "direction", "strength", "entry_delay_bars", "exit_policy", "month"], observed=False):
        rows.append({**dict(zip(["event_name", "direction", "strength", "entry_delay_bars", "exit_policy", "month"], keys)), **summarize_return(part["net_return"])})
    return pd.DataFrame(rows)


def build_exit_reason_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_cols = ["event_name", "direction", "strength", "entry_delay_bars", "exit_policy", "exit_reason"]
    for keys, part in trades.groupby(group_cols, observed=False):
        rows.append({**dict(zip(group_cols, keys)), **summarize_return(part["net_return"]), **_path_stats(part)})
    return pd.DataFrame(rows).sort_values(["exit_policy", "count"], ascending=[True, False], na_position="last")


def build_policy_vs_benchmark(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    key_cols = ["event_name", "direction", "strength", "entry_delay_bars"]
    bench = summary.loc[summary["exit_policy"] == "hold_240_benchmark"].copy()
    cand = summary.loc[summary["exit_policy"] != "hold_240_benchmark"].copy()
    if bench.empty or cand.empty:
        return pd.DataFrame()
    keep = key_cols + ["mean_net", "total_return", "win_rate", "profit_factor", "max_drawdown", "return_over_drawdown", "mfe_capture_ratio_mean", "avg_holding_bars", "path_cap_exit_share", "positive_years", "monthly_positive_rate"]
    bench = bench[[c for c in keep if c in bench.columns]].rename(columns={c: f"benchmark_{c}" for c in keep if c not in key_cols})
    merged = cand.merge(bench, on=key_cols, how="left")
    for col in ["mean_net", "total_return", "win_rate", "profit_factor", "return_over_drawdown", "mfe_capture_ratio_mean", "avg_holding_bars", "path_cap_exit_share", "positive_years", "monthly_positive_rate"]:
        b = f"benchmark_{col}"
        if col in merged.columns and b in merged.columns:
            merged[f"delta_{col}"] = pd.to_numeric(merged[col], errors="coerce") - pd.to_numeric(merged[b], errors="coerce")
    if "max_drawdown" in merged.columns and "benchmark_max_drawdown" in merged.columns:
        merged["delta_max_drawdown"] = pd.to_numeric(merged["max_drawdown"], errors="coerce") - pd.to_numeric(merged["benchmark_max_drawdown"], errors="coerce")
    return merged.sort_values(["delta_mean_net", "mean_net"], ascending=[False, False], na_position="last")


def build_hypothesis_decision(summary: pd.DataFrame, vs_bench: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows_promote: list[dict[str, object]] = []
    rows_reject: list[dict[str, object]] = []
    candidates = summary.loc[summary["exit_policy"] != "hold_240_benchmark"].copy()
    for _, row in candidates.iterrows():
        reasons: list[str] = []
        failures: list[str] = []
        mean_net = float(row.get("mean_net", np.nan))
        pf = float(row.get("profit_factor", np.nan))
        total = float(row.get("total_return", np.nan))
        pos_years = float(row.get("positive_years", np.nan))
        cap_share = float(row.get("path_cap_exit_share", np.nan))
        trades = int(row.get("count", 0) or 0)
        if np.isfinite(mean_net) and mean_net > 0:
            reasons.append("positive_expectancy")
        else:
            failures.append("negative_or_missing_expectancy")
        if np.isfinite(pf) and pf >= 1.15:
            reasons.append("pf_above_1p15")
        else:
            failures.append("low_profit_factor")
        if np.isfinite(total) and total > 0:
            reasons.append("positive_total_return")
        else:
            failures.append("negative_total_return")
        if np.isfinite(pos_years) and pos_years >= 3:
            reasons.append("yearly_not_bad")
        else:
            failures.append("yearly_unstable")
        if np.isfinite(cap_share) and cap_share <= 0.35:
            reasons.append("not_mostly_cap_exit")
        else:
            failures.append("too_many_unresolved_cap_exits")
        if trades >= 100:
            reasons.append("sample_ok")
        else:
            failures.append("sample_too_small")
        decision = "research_continue" if len(failures) <= 1 and mean_net > 0 and total > 0 else "reject_or_diagnostic_only"
        out = row.to_dict()
        out["decision"] = decision
        out["passed_reasons"] = ";".join(reasons)
        out["failed_reasons"] = ";".join(failures)
        if decision == "research_continue":
            rows_promote.append(out)
        else:
            rows_reject.append(out)
    return pd.DataFrame(rows_promote), pd.DataFrame(rows_reject)


def build_causal_exit_audit(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    cols = [
        "event_id", "event_name", "direction", "entry_delay_bars", "exit_policy", "event_bar_time", "signal_time",
        "signal_available_time", "entry_time", "expected_entry_time", "exit_time", "event_bar_pos", "signal_bar_pos",
        "entry_bar_pos", "exit_bar_pos", "exit_exec_pos", "structural_decision_bar_pos", "entry_not_expected_time_flag", "lookahead_flag",
    ]
    out = trades[[c for c in cols if c in trades.columns]].copy()
    out["context_available_time_flag"] = False
    out["exit_before_entry_flag"] = pd.to_datetime(out["exit_time"]) < pd.to_datetime(out["entry_time"])
    out["exit_decision_before_entry_flag"] = pd.to_numeric(out["structural_decision_bar_pos"], errors="coerce").fillna(-1) >= 0
    out["exit_decision_before_entry_flag"] = out["exit_decision_before_entry_flag"] & (pd.to_numeric(out["structural_decision_bar_pos"], errors="coerce") < pd.to_numeric(out["entry_bar_pos"], errors="coerce"))
    return out


def write_reports(
    out_dir: Path,
    *,
    args: argparse.Namespace,
    th: ShadowThresholds,
    bars: pd.DataFrame,
    events: pd.DataFrame,
    exit_trades: pd.DataFrame,
    entry_delays: tuple[int, ...],
) -> None:
    print("[aggregate] building simple-exit probe report tables", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "script_name": SCRIPT_NAME,
        "script_version": SCRIPT_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "edge_id": EDGE_ID,
        "title": TITLE,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "entry_delay_bars_list": list(entry_delays),
        "exit_policies": list(EXIT_POLICIES),
        "max_holding_bars": int(args.max_holding_bars),
        "round_trip_cost_pct": float(args.round_trip_cost_pct),
        "input_rows": int(len(bars)),
        "event_count": int(len(events)),
        "trade_row_count": int(len(exit_trades)),
        "thresholds": th.__dict__,
        "causal_policy": "closed 1m event bar; next-open entry; closed-bar structural exit decision; next-open exit; no MTF context",
        "created_at": pd.Timestamp.now("UTC").isoformat(),
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    policy_summary = build_policy_summary(exit_trades)
    vs_bench = build_policy_vs_benchmark(policy_summary)
    promoted, rejected = build_hypothesis_decision(policy_summary, vs_bench)
    reports = {
        "01_event_counts.csv": build_event_counts(events),
        "02_policy_summary.csv": policy_summary,
        "03_policy_yearly.csv": build_policy_yearly(exit_trades),
        "04_policy_monthly.csv": build_policy_monthly(exit_trades),
        "05_exit_reason_summary.csv": build_exit_reason_summary(exit_trades),
        "06_policy_vs_hold_benchmark.csv": vs_bench,
        "07_causal_audit.csv": build_causal_exit_audit(exit_trades),
        "08_research_continue_hypotheses.csv": promoted,
        "09_rejected_or_diagnostic_hypotheses.csv": rejected,
        "10_event_sample.csv": _sample(events, int(args.event_sample_size)),
        "11_trade_sample.csv": _sample(exit_trades, int(args.path_sample_size)),
    }
    for filename, df in reports.items():
        _write_csv(_pct_columns(df), out_dir / filename)
    if not bool(args.skip_full_exit_trades):
        slim_cols = [
            "event_id", "event_name", "family", "direction", "side", "strength", "entry_delay_bars", "exit_policy", "exit_policy_family",
            "event_bar_time", "signal_time", "entry_time", "exit_time", "event_low", "event_high", "event_open", "event_close",
            "entry_price", "exit_price", "exit_reason", "exit_price_kind", "gross_return", "net_return", "mfe", "mae",
            "time_to_mfe_bars", "time_to_mae_bars", "mfe_capture_ratio", "holding_bars", "adverse_sweep_count",
            "first_sweep_bars_after_entry", "last_sweep_bars_after_entry", "max_depth_beyond_adverse_pct", "max_consecutive_bars_beyond_adverse",
            "first_extreme_reclaim_bars_after_entry", "first_mid_reclaim_bars_after_entry", "first_full_reclaim_bars_after_entry",
            "extreme_reclaim_flag", "mid_reclaim_flag", "full_reclaim_flag", "session", "vol_regime", "trend_regime", "close_pos",
            "volume_ratio", "ret_30", "ret_120", "delta_ratio", "taker_buy_ratio", "lookahead_flag", "entry_not_expected_time_flag",
        ]
        _write_csv(_pct_columns(exit_trades[[c for c in slim_cols if c in exit_trades.columns]].copy()), out_dir / "12_full_exit_trades.csv")
    prompt = f"""# GPT Review Prompt — {TITLE}

This is a simple-exit hypothesis probe over the raw volume-shadow universe. It is not a final strategy.

Review rules:
1. Evaluate one hypothesis at a time. Do not combine environment filters, entry filters, and exit filters in the conclusion.
2. Compare each structural exit to `hold_240_benchmark`, but do not promote the benchmark itself.
3. Look for improvements across multiple dimensions: mean_net, total_return, PF, win_rate, drawdown proxy, MFE capture, unresolved cap exit share, yearly stability, and trade count.
4. Long and short should both be reviewed; do not assume upper-shadow short has no edge.
5. Structural exits use closed 1m bars and next-open exits. Same-bar stop assumptions are intentionally avoided.
6. If a policy only works by unresolved path-cap exits, mark it diagnostic-only.

Key files:
- 02_policy_summary.csv: main one-policy-at-a-time summary.
- 06_policy_vs_hold_benchmark.csv: candidate versus raw hold benchmark.
- 07_causal_audit.csv: timing checks.
- 08_research_continue_hypotheses.csv: policies that passed loose research-continue checks.
- 09_rejected_or_diagnostic_hypotheses.csv: policies that failed or are diagnostic only.
- 12_full_exit_trades.csv: full slim trade paths unless skipped.
"""
    (out_dir / "GPT_REVIEW_PROMPT.md").write_text(prompt, encoding="utf-8")
    readme = f"""# {TITLE}

This report tests simple structural exit hypotheses derived from the raw path atlas:
- hard event-extreme stop;
- close-beyond-extreme stop;
- allow one/two adverse sweeps;
- sweep requires reclaim within 15 bars;
- reclaim quality failure exits.

It intentionally does not add environment filters, footprint, range-bar context, or new entry filters.
"""
    (out_dir / "README_RESEARCH.md").write_text(readme, encoding="utf-8")
    print("[review-pack] finalizing GPT review pack", flush=True)
    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    entry_delays = _parse_int_csv(args.entry_delay_bars_list)
    th = _thresholds_from_args(args)
    print(f"[run] {SCRIPT_NAME} v{SCRIPT_VERSION}", flush=True)
    print(f"[args] out_dir={args.out_dir}", flush=True)
    print("[scope] raw lower/upper shadow simple-exit hypothesis probe; no environment filters", flush=True)
    bars = load_bars(args)
    features = build_features(bars, th)
    mask = _research_window_mask(features.index, args.start_date, args.end_date)
    research_features = features.loc[mask].copy()
    if research_features.empty:
        raise RuntimeError("Research window produced no rows")
    events = build_raw_shadow_events(research_features)
    if events.empty:
        print("[warn] no raw shadow events found", flush=True)
    trades = build_exit_probe_trades(
        research_features,
        events,
        entry_delays=entry_delays,
        max_holding_bars=int(args.max_holding_bars),
        round_trip_cost_pct=float(args.round_trip_cost_pct),
        progress_every=int(args.progress_every),
    )
    if not trades.empty:
        bad = trades.loc[trades.get("lookahead_flag", pd.Series(False, index=trades.index)).astype(bool)]
        if not bad.empty:
            raise RuntimeError(f"Lookahead flags detected in simple exit probe: {len(bad)}")
    print("[causal] no lookahead flags in simple exit probe", flush=True)
    write_reports(
        Path(args.out_dir),
        args=args,
        th=th,
        bars=bars,
        events=events,
        exit_trades=trades,
        entry_delays=entry_delays,
    )
    print("[done] raw volume-shadow simple exit probe completed; inspect gpt_review_pack.zip", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
