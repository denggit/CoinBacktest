#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETH 1m panic-wick V2 candidate overlap and path probe.

This is a follow-up to the V2 event study. It deliberately does **not** search a
large parameter grid. It only studies four already shortlisted structural
hypotheses and asks three engineering/research questions:

1. How much do the four signals overlap exactly and within the same panic wave?
2. Does each signal add independent trades after removing parent/child overlap?
3. Does a simple causal no-overlap path probe survive fixed exits and costs?

Causal policy
-------------
- 1m bars are left-labeled by bar start time.
- Event bars are known only after they close.
- signal_time = event_bar_start + 1 minute.
- entry_time = next bar open, equal to signal_time for 1m left-labeled bars.
- No multi-timeframe context is used.
- Path simulation skips new long signals while a long position is already open.
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

SCRIPT_NAME = "eth_1m_panic_wick_overlap_path_probe"
SCRIPT_VERSION = "1.0.0"
EXPERIMENT_ID = "ETH_MF_1M_PANIC_WICK_OVERLAP_PATH_V1"
EDGE_ID = "RESEARCH_ONLY_ETH_MF_1M_PANIC_WICK_OVERLAP_PATH_V1"
TITLE = "ETH 1m Panic Wick Candidate Overlap and Path Probe V1"
DEFAULT_OUT_DIR = "data/reports/research/eth_1m_panic_wick_overlap_path_v1"
BAR_DELTA = pd.Timedelta(minutes=1)

CANDIDATE_NAMES: tuple[str, ...] = (
    "panic_lower_reclaim_long",
    "panic_downtrend_lower_reclaim_long",
    "panic_neg_delta_absorption_long",
    "panic_downtrend_neg_delta_absorption_long",
)
STRICT_PRIORITY: tuple[str, ...] = (
    "panic_downtrend_neg_delta_absorption_long",
    "panic_downtrend_lower_reclaim_long",
    "panic_neg_delta_absorption_long",
    "panic_lower_reclaim_long",
)


@dataclass(frozen=True)
class PanicThresholds:
    """Fixed V2 thresholds; they are not swept in this overlap/path probe."""

    wick_share_min: float = 0.50
    wick_atr_min: float = 0.55
    volume_ratio_min: float = 2.0
    reclaim_close_pos: float = 0.66
    soft_reclaim_close_pos: float = 0.55
    prior_flush_30_min: float = -0.005
    prior_flush_120_min: float = -0.010
    delta_absorption_max: float = -0.10
    taker_buy_absorption_max: float = 0.45
    stop_buffer_pct: float = 0.0015


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Research-only overlap/path probe for four ETH 1m panic-wick V2 candidates.",
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

    # Fixed diagnostics, not optimized parameters.
    p.add_argument("--horizons", default="30,60,120,240")
    p.add_argument("--primary-horizon", type=int, default=60)
    p.add_argument("--round-trip-cost-pct", type=float, default=0.0011)
    p.add_argument("--cost-multipliers", default="1.0,2.0,3.0")
    p.add_argument("--delay-bars-list", default="0,1,2")
    p.add_argument("--near-overlap-bars", default="5,60", help="Diagnostic windows for near-overlap, not entry filters.")
    p.add_argument("--time-stop-bars", default="60,120", help="Fixed path-probe exits, not a tuning grid.")
    p.add_argument("--min-count", type=int, default=80)
    p.add_argument("--event-sample-size", type=int, default=5000)
    p.add_argument("--progress-every", type=int, default=2500)
    p.add_argument("--write-full-trades", action="store_true")

    # V2 vocabulary thresholds. Keep defaults unless deliberately reproducing V2.
    p.add_argument("--wick-share-min", type=float, default=0.50)
    p.add_argument("--wick-atr-min", type=float, default=0.55)
    p.add_argument("--volume-ratio-min", type=float, default=2.0)
    p.add_argument("--reclaim-close-pos", type=float, default=0.66)
    p.add_argument("--soft-reclaim-close-pos", type=float, default=0.55)
    p.add_argument("--prior-flush-30-min", type=float, default=-0.005)
    p.add_argument("--prior-flush-120-min", type=float, default=-0.010)
    p.add_argument("--delta-absorption-max", type=float, default=-0.10)
    p.add_argument("--taker-buy-absorption-max", type=float, default=0.45)
    p.add_argument("--stop-buffer-pct", type=float, default=0.0015)
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
        stop_buffer_pct=float(args.stop_buffer_pct),
    )


def _research_window_mask(index: pd.DatetimeIndex, start_date: str, end_date: str) -> pd.Series:
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if end_ts == end_ts.normalize() and len(str(end_date).strip()) <= 10:
        end_exclusive = end_ts + pd.Timedelta(days=1)
    else:
        end_exclusive = end_ts + BAR_DELTA
    return pd.Series((index >= start_ts) & (index < end_exclusive), index=index)


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
    return float(ts.diff().dropna().max() / pd.Timedelta(days=1))


def _events_per_month(times: pd.Series) -> float:
    ts = pd.to_datetime(times, errors="coerce").dropna()
    if ts.empty:
        return 0.0
    span_days = max(1.0, float((ts.max() - ts.min()) / pd.Timedelta(days=1)))
    return float(len(ts) / span_days * 30.4375)


def _cost_tag(mult: float) -> str:
    return str(float(mult)).rstrip("0").rstrip(".").replace(".", "p")


def session_label(index: pd.DatetimeIndex) -> np.ndarray:
    h = index.hour
    return np.select(
        [h < 8, (h >= 8) & (h < 16), h >= 16],
        ["asia_early", "asia_london", "eu_us"],
        default="unknown",
    )


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
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    print(f"       rows={len(out):,} range={out.index.min()} -> {out.index.max()}", flush=True)
    return out


def build_features(bars: pd.DataFrame, th: PanicThresholds) -> pd.DataFrame:
    print("[features] building fixed V2 four-candidate feature frame", flush=True)
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
    vol_base = df["volume"].shift(1).rolling(240, min_periods=60).median()
    out["volume_ratio"] = _safe_divide(df["volume"], vol_base).to_numpy()
    out["volume_climax"] = out["volume_ratio"] >= th.volume_ratio_min
    out["ret_30"] = df["close"].pct_change(30)
    out["ret_120"] = df["close"].pct_change(120)
    out["ema_240"] = df["close"].ewm(span=240, adjust=False, min_periods=240).mean()
    out["ema240_slope_60"] = out["ema_240"] / out["ema_240"].shift(60) - 1.0

    delta = pd.to_numeric(df.get("delta_notional", np.nan), errors="coerce")
    notional = pd.to_numeric(df.get("notional", np.nan), errors="coerce").abs()
    out["delta_ratio"] = _safe_divide(delta, notional).to_numpy()
    if "taker_buy_ratio" in df.columns:
        out["taker_buy_ratio"] = pd.to_numeric(df["taker_buy_ratio"], errors="coerce")
    else:
        buy_volume = pd.to_numeric(df.get("buy_volume", np.nan), errors="coerce")
        out["taker_buy_ratio"] = _safe_divide(buy_volume, df["volume"]).to_numpy()

    out["long_lower_wick"] = (out["lower_wick_share"] >= th.wick_share_min) & (out["lower_wick_atr"] >= th.wick_atr_min)
    out["long_upper_wick"] = out["upper_wick_share"] >= th.wick_share_min
    out["lower_volume_shadow"] = out["long_lower_wick"] & out["volume_climax"]
    out["two_sided_shadow"] = out["lower_volume_shadow"] & out["long_upper_wick"]
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
    high_vol = out["vol_regime"].isin(["mid_high_vol", "extreme_vol"])
    prior_flush = (out["ret_30"] <= th.prior_flush_30_min) | (out["ret_120"] <= th.prior_flush_120_min)
    out["panic_context"] = high_vol & prior_flush
    out["panic_downtrend_context"] = out["panic_context"] & (out["trend_regime"] == "downtrend")
    out["flow_absorption"] = (out["delta_ratio"] <= th.delta_absorption_max) | (out["taker_buy_ratio"] <= th.taker_buy_absorption_max)
    return out


def _event_frame_from_mask(features: pd.DataFrame, *, mask: pd.Series, event_name: str, family: str, structure: str) -> pd.DataFrame:
    bool_mask = mask.fillna(False).to_numpy(dtype=bool)
    idx = features.index[bool_mask]
    if len(idx) == 0:
        return pd.DataFrame()
    pos_map = pd.Series(np.arange(len(features), dtype=int), index=features.index)
    event_pos = pos_map.reindex(idx).to_numpy(dtype=int)
    signal_pos = event_pos
    valid = signal_pos + 1 < len(features)
    if not valid.any():
        return pd.DataFrame()
    idx = idx[valid]
    f = features.loc[idx]
    event_pos = event_pos[valid]
    signal_pos = signal_pos[valid]
    signal_bar_time = features.index[signal_pos]
    return pd.DataFrame(
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
            "confirmation_bars": 0,
            "planned_entry_offset": 1,
            "signal_time": signal_bar_time + BAR_DELTA,
            "signal_available_time": signal_bar_time + BAR_DELTA,
            "entry_bar_pos": signal_pos + 1,
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


def build_four_candidate_events(features: pd.DataFrame, th: PanicThresholds) -> pd.DataFrame:
    print("[events] building four V2 shortlisted candidates", flush=True)
    f = features
    lower = f["lower_volume_shadow"] & ~f["two_sided_shadow"]
    panic = lower & f["panic_context"]
    panic_downtrend = lower & f["panic_downtrend_context"]
    reclaim = f["close_pos"] >= th.reclaim_close_pos
    soft_reclaim = f["close_pos"] >= th.soft_reclaim_close_pos
    negative_flow = f["flow_absorption"]
    specs: list[tuple[str, str, str, pd.Series]] = [
        (
            "panic_lower_reclaim_long",
            "panic_flush_reclaim",
            "high_vol_prior_flush_lower_wick_close_reclaim",
            panic & reclaim,
        ),
        (
            "panic_downtrend_lower_reclaim_long",
            "panic_downtrend_reclaim",
            "downtrend_high_vol_prior_flush_lower_wick_close_reclaim",
            panic_downtrend & reclaim,
        ),
        (
            "panic_neg_delta_absorption_long",
            "panic_flow_absorption",
            "panic_lower_wick_soft_reclaim_negative_delta_or_low_taker_buy",
            panic & soft_reclaim & negative_flow,
        ),
        (
            "panic_downtrend_neg_delta_absorption_long",
            "panic_downtrend_flow_absorption",
            "downtrend_panic_lower_wick_soft_reclaim_negative_delta_or_low_taker_buy",
            panic_downtrend & soft_reclaim & negative_flow,
        ),
    ]
    frames = [_event_frame_from_mask(f, mask=mask, event_name=name, family=family, structure=structure) for name, family, structure, mask in specs]
    out = pd.concat([x for x in frames if not x.empty], ignore_index=True) if any(not x.empty for x in frames) else pd.DataFrame()
    if out.empty:
        return out
    out = out.sort_values(["signal_bar_time", "event_name"]).reset_index(drop=True)
    out["event_id"] = np.arange(len(out), dtype=int)
    return out


def attach_forward_returns(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    cost_multipliers: tuple[float, ...],
    delay_bars_list: tuple[int, ...],
    round_trip_cost_pct: float,
    progress_every: int,
) -> pd.DataFrame:
    print("[forward] attaching fixed-horizon forward labels for overlap diagnostics", flush=True)
    if events.empty:
        return events.copy()
    frame = bars.sort_index()
    out = events.copy()
    open_arr = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
    close_arr = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    event_pos = pd.to_numeric(out["event_bar_pos"], errors="coerce").to_numpy(dtype=int)
    sides = pd.to_numeric(out["side"], errors="coerce").to_numpy(dtype=float)
    n_bars = len(frame)
    new_cols: dict[str, np.ndarray] = {}
    progress = ProgressReporter(label="[forward] label grid", total=len(horizons) * len(delay_bars_list), every=max(1, min(int(progress_every), len(horizons) * len(delay_bars_list))))
    done = 0
    for extra_delay in delay_bars_list:
        delay_i = int(extra_delay)
        entry_pos = event_pos + 1 + delay_i
        entry_vals = np.full(len(out), np.nan, dtype=float)
        valid_e = (entry_pos >= 0) & (entry_pos < n_bars)
        entry_vals[valid_e] = open_arr[entry_pos[valid_e]]
        for h in horizons:
            h_i = int(h)
            exit_pos = event_pos + 1 + delay_i + h_i - 1
            exit_vals = np.full(len(out), np.nan, dtype=float)
            valid_x = (exit_pos >= 0) & (exit_pos < n_bars)
            exit_vals[valid_x] = close_arr[exit_pos[valid_x]]
            gross = (exit_vals / entry_vals - 1.0) * sides
            bad = (~np.isfinite(entry_vals)) | (entry_vals <= 0) | (~np.isfinite(exit_vals))
            gross[bad] = np.nan
            new_cols[f"ret_h{h_i}_d{delay_i}_gross"] = gross
            for mult in cost_multipliers:
                new_cols[f"ret_h{h_i}_d{delay_i}_cost{_cost_tag(mult)}_net"] = gross - float(round_trip_cost_pct) * float(mult)
            done += 1
            progress.update(done)
    progress.close()
    out["entry_time"] = pd.NaT
    out["entry_price"] = np.nan
    entry_pos0 = event_pos + 1
    valid0 = (entry_pos0 >= 0) & (entry_pos0 < n_bars)
    out.loc[valid0, "entry_time"] = frame.index[entry_pos0[valid0]]
    out.loc[valid0, "entry_price"] = open_arr[entry_pos0[valid0]]
    out["expected_entry_time"] = pd.to_datetime(out["signal_bar_time"]) + BAR_DELTA
    out["expected_entry_price"] = out["entry_price"]
    return pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)


def summarize_return(x: pd.Series) -> dict[str, float | int]:
    r = pd.to_numeric(x, errors="coerce").dropna()
    return {
        "count": int(len(r)),
        "mean_net": float(r.mean()) if len(r) else np.nan,
        "median_net": float(r.median()) if len(r) else np.nan,
        "win_rate": float((r > 0).mean()) if len(r) else np.nan,
        "profit_factor": _profit_factor(r),
        "top5_winner_share": _top5_winner_share(r),
    }


def build_candidate_event_summary(events: pd.DataFrame, *, primary_horizon: int, min_count: int) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    ret_col = f"ret_h{int(primary_horizon)}_d0_cost1_net"
    fee2_col = f"ret_h{int(primary_horizon)}_d0_cost2_net"
    delay1_col = f"ret_h{int(primary_horizon)}_d1_cost1_net"
    rows: list[dict[str, object]] = []
    for name, part in events.groupby("event_name", observed=False):
        s = summarize_return(part[ret_col])
        yearly = part.assign(year=pd.to_datetime(part["signal_time"]).dt.year).groupby("year", observed=False)[ret_col].mean()
        rows.append(
            {
                "event_name": name,
                **s,
                "eligible": bool(s["count"] >= int(min_count)),
                "events_per_month": _events_per_month(part["signal_time"]),
                "max_days_without_event": _max_days_without_event(part["signal_time"]),
                "fee_2x_mean_net": float(pd.to_numeric(part[fee2_col], errors="coerce").mean()) if fee2_col in part else np.nan,
                "delay_1bar_mean_net": float(pd.to_numeric(part[delay1_col], errors="coerce").mean()) if delay1_col in part else np.nan,
                "positive_years": int((yearly > 0).sum()),
                "year_count": int(yearly.notna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_net", ascending=False, na_position="last")


def build_yearly_by_signal(events: pd.DataFrame, *, primary_horizon: int) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    ret_col = f"ret_h{int(primary_horizon)}_d0_cost1_net"
    rows: list[dict[str, object]] = []
    tmp = events.copy()
    tmp["year"] = pd.to_datetime(tmp["signal_time"]).dt.year
    for (name, year), part in tmp.groupby(["event_name", "year"], observed=False):
        rows.append({"event_name": name, "year": int(year), **summarize_return(part[ret_col])})
    return pd.DataFrame(rows)


def build_overlap_exact(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    sets = {name: set(part["event_bar_pos"].astype(int).tolist()) for name, part in events.groupby("event_name", observed=False)}
    rows: list[dict[str, object]] = []
    for a in CANDIDATE_NAMES:
        set_a = sets.get(a, set())
        for b in CANDIDATE_NAMES:
            set_b = sets.get(b, set())
            inter = set_a & set_b
            union = set_a | set_b
            rows.append(
                {
                    "signal_a": a,
                    "signal_b": b,
                    "count_a": len(set_a),
                    "count_b": len(set_b),
                    "exact_overlap_count": len(inter),
                    "jaccard": float(len(inter) / len(union)) if union else np.nan,
                    "pct_a_inside_b": float(len(inter) / len(set_a)) if set_a else np.nan,
                    "pct_b_inside_a": float(len(inter) / len(set_b)) if set_b else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _count_near(a: np.ndarray, b: np.ndarray, window: int) -> int:
    if len(a) == 0 or len(b) == 0:
        return 0
    b_sorted = np.sort(b.astype(int))
    n = 0
    for x in np.sort(a.astype(int)):
        left = np.searchsorted(b_sorted, int(x) - int(window), side="left")
        right = np.searchsorted(b_sorted, int(x) + int(window), side="right")
        if right > left:
            n += 1
    return int(n)


def build_near_overlap(events: pd.DataFrame, *, windows: tuple[int, ...]) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    arrs = {name: part["event_bar_pos"].astype(int).to_numpy() for name, part in events.groupby("event_name", observed=False)}
    rows: list[dict[str, object]] = []
    for w in windows:
        for a in CANDIDATE_NAMES:
            aa = arrs.get(a, np.array([], dtype=int))
            for b in CANDIDATE_NAMES:
                bb = arrs.get(b, np.array([], dtype=int))
                near = _count_near(aa, bb, int(w))
                rows.append(
                    {
                        "window_bars": int(w),
                        "signal_a": a,
                        "signal_b": b,
                        "count_a": int(len(aa)),
                        "count_b": int(len(bb)),
                        "a_events_with_b_near": int(near),
                        "pct_a_with_b_near": float(near / len(aa)) if len(aa) else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def build_membership_buckets(events: pd.DataFrame, *, primary_horizon: int) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    ret_col = f"ret_h{int(primary_horizon)}_d0_cost1_net"
    wide = events.pivot_table(index="event_bar_pos", columns="event_name", values=ret_col, aggfunc="first")
    # Because all four entries are next-open on the same event bar, returns are equal
    # for overlapping rows. Use row mean to be robust to duplicates.
    wide["bucket_ret"] = wide[list(c for c in CANDIDATE_NAMES if c in wide.columns)].mean(axis=1)
    rows: list[dict[str, object]] = []
    for _, row in wide.iterrows():
        flags = {name: bool(pd.notna(row[name])) if name in wide.columns else False for name in CANDIDATE_NAMES}
        key = "+".join([k.replace("panic_", "") for k, v in flags.items() if v]) or "none"
        rows.append({**flags, "membership_key": key, "bucket_ret": row["bucket_ret"]})
    tmp = pd.DataFrame(rows)
    out: list[dict[str, object]] = []
    for key, part in tmp.groupby("membership_key", observed=False):
        row: dict[str, object] = {"membership_key": key, **summarize_return(part["bucket_ret"])}
        for name in CANDIDATE_NAMES:
            row[name] = bool(part[name].iloc[0])
        out.append(row)
    return pd.DataFrame(out).sort_values(["mean_net", "count"], ascending=[False, False], na_position="last")


def _make_policy_events(events: pd.DataFrame, policy: str) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    if policy.startswith("single:"):
        name = policy.split(":", 1)[1]
        out = events.loc[events["event_name"] == name].copy()
        out["policy"] = policy
        out["source_event_names"] = out["event_name"]
        return out
    if policy == "union_all_exact_dedup":
        p = events.copy()
        p["priority"] = p["event_name"].map({name: i for i, name in enumerate(STRICT_PRIORITY)}).fillna(999).astype(int)
        out = p.sort_values(["event_bar_pos", "priority"]).drop_duplicates("event_bar_pos", keep="first").copy()
        grouped = p.groupby("event_bar_pos", observed=False)["event_name"].agg(lambda x: "+".join(sorted(set(x))))
        out["source_event_names"] = out["event_bar_pos"].map(grouped)
        out["policy"] = policy
        return out.drop(columns=["priority"])
    if policy == "strict_only_downtrend_neg_delta":
        out = events.loc[events["event_name"] == "panic_downtrend_neg_delta_absorption_long"].copy()
        out["policy"] = policy
        out["source_event_names"] = out["event_name"]
        return out
    if policy == "reclaim_parent_minus_strict_children":
        child_pos = set(events.loc[events["event_name"].isin(["panic_downtrend_lower_reclaim_long", "panic_neg_delta_absorption_long", "panic_downtrend_neg_delta_absorption_long"]), "event_bar_pos"].astype(int))
        out = events.loc[(events["event_name"] == "panic_lower_reclaim_long") & (~events["event_bar_pos"].astype(int).isin(child_pos))].copy()
        out["policy"] = policy
        out["source_event_names"] = out["event_name"]
        return out
    raise ValueError(f"Unknown policy: {policy}")


def build_policy_event_summary(events: pd.DataFrame, *, primary_horizon: int, min_count: int) -> pd.DataFrame:
    policies = [*(f"single:{x}" for x in CANDIDATE_NAMES), "union_all_exact_dedup", "strict_only_downtrend_neg_delta", "reclaim_parent_minus_strict_children"]
    rows: list[dict[str, object]] = []
    ret_col = f"ret_h{int(primary_horizon)}_d0_cost1_net"
    fee2_col = f"ret_h{int(primary_horizon)}_d0_cost2_net"
    delay1_col = f"ret_h{int(primary_horizon)}_d1_cost1_net"
    for policy in policies:
        part = _make_policy_events(events, policy)
        if part.empty:
            rows.append({"policy": policy, "count": 0, "eligible": False})
            continue
        s = summarize_return(part[ret_col])
        yearly = part.assign(year=pd.to_datetime(part["signal_time"]).dt.year).groupby("year", observed=False)[ret_col].mean()
        rows.append(
            {
                "policy": policy,
                **s,
                "eligible": bool(s["count"] >= int(min_count)),
                "events_per_month": _events_per_month(part["signal_time"]),
                "max_days_without_event": _max_days_without_event(part["signal_time"]),
                "fee_2x_mean_net": float(pd.to_numeric(part[fee2_col], errors="coerce").mean()) if fee2_col in part else np.nan,
                "delay_1bar_mean_net": float(pd.to_numeric(part[delay1_col], errors="coerce").mean()) if delay1_col in part else np.nan,
                "positive_years": int((yearly > 0).sum()),
                "year_count": int(yearly.notna().sum()),
            }
        )
    df = pd.DataFrame(rows)
    if "mean_net" in df.columns:
        df = df.sort_values("mean_net", ascending=False, na_position="last")
    return df


def simulate_path(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    policy: str,
    exit_mode: str,
    time_stop_bars: int,
    round_trip_cost_pct: float,
    stop_buffer_pct: float,
) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    frame = bars.sort_index()
    open_arr = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
    close_arr = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    low_arr = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    idx = frame.index
    sig = signals.sort_values(["entry_bar_pos", "event_name"]).copy()
    trades: list[dict[str, object]] = []
    next_free_pos = -1
    progress = ProgressReporter(label=f"[path] {policy} {exit_mode}{time_stop_bars}", total=len(sig), every=max(1, min(5000, len(sig))))
    for k, row in enumerate(sig.itertuples(index=False), start=1):
        entry_pos = int(getattr(row, "entry_bar_pos"))
        if entry_pos <= next_free_pos:
            progress.update(k)
            continue
        if entry_pos < 0 or entry_pos >= len(frame):
            progress.update(k)
            continue
        entry_price = float(open_arr[entry_pos])
        if not np.isfinite(entry_price) or entry_price <= 0:
            progress.update(k)
            continue
        max_exit_pos = min(len(frame) - 1, entry_pos + int(time_stop_bars) - 1)
        exit_pos = max_exit_pos
        exit_price = float(close_arr[max_exit_pos])
        exit_reason = f"time_stop_{int(time_stop_bars)}"
        stop_price = float(getattr(row, "event_low")) * (1.0 - float(stop_buffer_pct))
        if "event_low_stop" in exit_mode and np.isfinite(stop_price) and stop_price > 0:
            path_lows = low_arr[entry_pos : max_exit_pos + 1]
            hit = np.flatnonzero(path_lows <= stop_price)
            if len(hit):
                exit_pos = entry_pos + int(hit[0])
                exit_price = stop_price
                exit_reason = "event_low_stop"
        if not np.isfinite(exit_price) or exit_price <= 0:
            progress.update(k)
            continue
        gross = exit_price / entry_price - 1.0
        net = gross - float(round_trip_cost_pct)
        trades.append(
            {
                "policy": policy,
                "exit_mode": exit_mode,
                "time_stop_bars": int(time_stop_bars),
                "source_event_name": getattr(row, "event_name"),
                "source_event_names": getattr(row, "source_event_names", getattr(row, "event_name")),
                "event_bar_time": getattr(row, "event_bar_time"),
                "signal_time": getattr(row, "signal_time"),
                "entry_time": idx[entry_pos],
                "entry_pos": entry_pos,
                "entry_price": entry_price,
                "exit_time": idx[exit_pos],
                "exit_pos": exit_pos,
                "exit_price": exit_price,
                "exit_reason": exit_reason,
                "gross_return": gross,
                "net_return": net,
                "holding_bars": int(exit_pos - entry_pos + 1),
                "event_low": getattr(row, "event_low"),
            }
        )
        next_free_pos = exit_pos
        progress.update(k)
    progress.close()
    return pd.DataFrame(trades)


def build_path_probe(bars: pd.DataFrame, events: pd.DataFrame, *, time_stops: tuple[int, ...], round_trip_cost_pct: float, stop_buffer_pct: float) -> pd.DataFrame:
    print("[path] simulating fixed no-overlap path probes", flush=True)
    policies = [*(f"single:{x}" for x in CANDIDATE_NAMES), "union_all_exact_dedup"]
    frames: list[pd.DataFrame] = []
    for policy in policies:
        policy_events = _make_policy_events(events, policy)
        for t in time_stops:
            for exit_mode in ("time_stop", "event_low_stop_or_time"):
                frames.append(
                    simulate_path(
                        bars,
                        policy_events,
                        policy=policy,
                        exit_mode=exit_mode,
                        time_stop_bars=int(t),
                        round_trip_cost_pct=float(round_trip_cost_pct),
                        stop_buffer_pct=float(stop_buffer_pct),
                    )
                )
    non_empty = [x for x in frames if not x.empty]
    return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()


def build_path_summary(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, part in trades.groupby(["policy", "exit_mode", "time_stop_bars"], observed=False):
        policy, exit_mode, time_stop = keys
        yearly = part.assign(year=pd.to_datetime(part["entry_time"]).dt.year).groupby("year", observed=False)["net_return"].mean()
        rows.append(
            {
                "policy": policy,
                "exit_mode": exit_mode,
                "time_stop_bars": int(time_stop),
                **summarize_return(part["net_return"]),
                "events_per_month": _events_per_month(part["entry_time"]),
                "max_days_without_trade": _max_days_without_event(part["entry_time"]),
                "positive_years": int((yearly > 0).sum()),
                "year_count": int(yearly.notna().sum()),
                "avg_holding_bars": float(pd.to_numeric(part["holding_bars"], errors="coerce").mean()),
                "stop_exit_share": float((part["exit_reason"] == "event_low_stop").mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_net", ascending=False, na_position="last")


def build_path_yearly(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    tmp = trades.copy()
    tmp["year"] = pd.to_datetime(tmp["entry_time"]).dt.year
    for keys, part in tmp.groupby(["policy", "exit_mode", "time_stop_bars", "year"], observed=False):
        policy, exit_mode, time_stop, year = keys
        rows.append({"policy": policy, "exit_mode": exit_mode, "time_stop_bars": int(time_stop), "year": int(year), **summarize_return(part["net_return"])})
    return pd.DataFrame(rows)


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
    out["audit_notes"] = "closed 1m event bar -> next open; no MTF context"
    return out


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")


def write_reports(
    *,
    out_dir: Path,
    args: argparse.Namespace,
    thresholds: PanicThresholds,
    features: pd.DataFrame,
    events: pd.DataFrame,
    trades: pd.DataFrame,
    horizons: tuple[int, ...],
    cost_multipliers: tuple[float, ...],
    delay_bars_list: tuple[int, ...],
    near_windows: tuple[int, ...],
    time_stops: tuple[int, ...],
) -> None:
    print("[write] writing overlap/path probe reports", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    primary_h = int(args.primary_horizon)
    event_summary = build_candidate_event_summary(events, primary_horizon=primary_h, min_count=int(args.min_count))
    yearly = build_yearly_by_signal(events, primary_horizon=primary_h)
    exact_overlap = build_overlap_exact(events)
    near_overlap = build_near_overlap(events, windows=near_windows)
    membership = build_membership_buckets(events, primary_horizon=primary_h)
    policy_event_summary = build_policy_event_summary(events, primary_horizon=primary_h, min_count=int(args.min_count))
    path_summary = build_path_summary(trades)
    path_yearly = build_path_yearly(trades)
    causal = build_causal_audit(events)
    sample_cols = [
        "event_id",
        "event_name",
        "family",
        "event_bar_time",
        "signal_time",
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
        f"ret_h{primary_h}_d0_cost1_net",
    ]
    event_sample = events[[c for c in sample_cols if c in events.columns]].head(int(args.event_sample_size)).copy() if not events.empty else pd.DataFrame()
    trade_sample = trades.head(int(args.event_sample_size)).copy() if not trades.empty else pd.DataFrame()
    if not bool(args.write_full_trades):
        full_trades = pd.DataFrame()
    else:
        full_trades = trades

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
        "candidate_names": list(CANDIDATE_NAMES),
        "horizons": list(horizons),
        "primary_horizon": primary_h,
        "round_trip_cost_pct": float(args.round_trip_cost_pct),
        "cost_multipliers": list(cost_multipliers),
        "delay_bars_list": list(delay_bars_list),
        "near_overlap_bars": list(near_windows),
        "time_stop_bars": list(time_stops),
        "input_rows": int(len(features)),
        "event_count": int(len(events)),
        "path_trade_count": int(len(trades)),
        "thresholds": thresholds.__dict__,
        "causal_policy": "closed 1m event bar; signal available at bar close; entry next 1m open; no MTF context; path probe skips overlapping positions",
        "anti_overfit_policy": "Only four V2 shortlisted structural candidates plus fixed overlap/path diagnostics; no wick/volume/ATR/horizon parameter grid search",
        "created_at": pd.Timestamp.now("UTC").isoformat(),
    }
    (out_dir / "00_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_csv(event_summary, out_dir / "01_candidate_event_summary.csv")
    _write_csv(yearly, out_dir / "02_candidate_yearly.csv")
    _write_csv(exact_overlap, out_dir / "03_exact_overlap_matrix.csv")
    _write_csv(near_overlap, out_dir / "04_near_overlap_directional.csv")
    _write_csv(membership, out_dir / "05_membership_buckets.csv")
    _write_csv(policy_event_summary, out_dir / "06_policy_event_summary.csv")
    _write_csv(path_summary, out_dir / "07_path_probe_summary.csv")
    _write_csv(path_yearly, out_dir / "08_path_probe_yearly.csv")
    _write_csv(causal, out_dir / "09_causal_audit.csv")
    _write_csv(event_sample, out_dir / "10_event_sample.csv")
    _write_csv(trade_sample, out_dir / "11_path_trades_sample.csv")
    if not full_trades.empty:
        _write_csv(full_trades, out_dir / "12_full_path_trades.csv")

    notes = f"""# {TITLE}

This is a research-only overlap/path probe. It does **not** create a tradable edge.

## Four candidates studied
- `panic_lower_reclaim_long`
- `panic_downtrend_lower_reclaim_long`
- `panic_neg_delta_absorption_long`
- `panic_downtrend_neg_delta_absorption_long`

## Main questions
1. Are these four signals mostly the same trades, parent/child subsets, or partly independent?
2. Does `panic_lower_reclaim_long` add value after removing the stricter downtrend/flow children?
3. Does the strict union survive when duplicated signals are deduplicated and overlapping positions are skipped?
4. Does a fixed event-low stop improve or destroy the candidate path?

## What is deliberately not done
- No large parameter grid.
- No multi-timeframe context.
- No TP/SL optimization grid.
- No live/portfolio promotion.

## Files to inspect first
- `03_exact_overlap_matrix.csv`
- `04_near_overlap_directional.csv`
- `05_membership_buckets.csv`
- `06_policy_event_summary.csv`
- `07_path_probe_summary.csv`
- `08_path_probe_yearly.csv`
- `09_causal_audit.csv`
"""
    (out_dir / "README_RESEARCH.md").write_text(notes, encoding="utf-8")
    print("[review-pack] finalizing GPT review pack", flush=True)
    finalize_research_report(out_dir, experiment_id=EXPERIMENT_ID, edge_id=EDGE_ID, title=TITLE)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    horizons = _parse_csv_ints(args.horizons)
    cost_multipliers = _parse_csv_floats(args.cost_multipliers)
    delay_bars_list = _parse_csv_ints(args.delay_bars_list)
    near_windows = _parse_csv_ints(args.near_overlap_bars)
    time_stops = _parse_csv_ints(args.time_stop_bars)
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
    print("[scope] four V2 candidates; overlap, membership and fixed path probe; no parameter-grid search", flush=True)

    bars = load_trade_bars(args)
    features = build_features(bars, thresholds)
    research_mask = _research_window_mask(features.index, args.start_date, args.end_date)
    research_features = features.loc[research_mask.to_numpy(dtype=bool)].copy()
    events = build_four_candidate_events(research_features, thresholds)
    events = attach_forward_returns(
        bars=research_features,
        events=events,
        horizons=horizons,
        cost_multipliers=cost_multipliers,
        delay_bars_list=delay_bars_list,
        round_trip_cost_pct=float(args.round_trip_cost_pct),
        progress_every=int(args.progress_every),
    )
    causal = build_causal_audit(events)
    if not causal.empty and bool(causal["lookahead_flag"].any()):
        print(f"[causal] WARNING lookahead flags={int(causal['lookahead_flag'].sum())}", flush=True)
    else:
        print("[causal] no lookahead flags in event-specific next-open audit", flush=True)

    trades = build_path_probe(
        research_features,
        events,
        time_stops=time_stops,
        round_trip_cost_pct=float(args.round_trip_cost_pct),
        stop_buffer_pct=float(thresholds.stop_buffer_pct),
    )
    write_reports(
        out_dir=Path(args.out_dir),
        args=args,
        thresholds=thresholds,
        features=research_features,
        events=events,
        trades=trades,
        horizons=horizons,
        cost_multipliers=cost_multipliers,
        delay_bars_list=delay_bars_list,
        near_windows=near_windows,
        time_stops=time_stops,
    )
    print("[done] overlap/path probe completed; inspect gpt_review_pack.zip before deciding backtest promotion", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
