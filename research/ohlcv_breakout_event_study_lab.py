#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OHLCV Breakout Event Study Lab
==============================

Research-only smoke lab for the reusable event-study module.

Purpose
-------
Run a causal, closed-bar / next-open event study on simple ETH OHLCV events:
    1. prior-high / prior-low close breakout
    2. volume-confirmed breakout
    3. volatility-compression breakout
    4. higher-timeframe trend-aligned breakout
    5. sweep/rejection events around prior range extremes

Important boundaries
--------------------
1. Data must be loaded through src.data_feed.OKXDataLoader only. This script
   does not read SQLite files, raw CSVs, or ZIPs directly.
2. All breakout thresholds use prior rolling windows via shift(1). The signal
   bar itself is closed before the event is evaluated.
3. Higher-timeframe context is aligned by available_time through
   causal_align_context(), not by bar_start_time ffill.
4. Future returns, MFE/MAE, and first-touch labels are evaluation labels only.
   They are never used to generate event conditions.
5. This is not a final strategy. Any positive phenomenon must still pass full
   strategy backtest, fee/slippage/delay stress, parameter-neighbourhood checks,
   walk-forward, and live execution review before it can be promoted.

Example
-------
python research/ohlcv_breakout_event_study_lab.py --symbol ETH-USDT-SWAP --timeframe 5m --context-timeframe 15m --start-date 2023-01-01 --end-date 2026-06-30 --warmup-start-date 2022-01-01 --out-dir data/reports/research/ohlcv_breakout_event_study_lab
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.research_common.event_study import (  # noqa: E402
    CostConfig,
    EventStudyConfig,
    causal_align_context,
    condition_contrast,
    first_touch_outcome,
    fixed_threshold_labels,
    profit_factor,
    qcut_labels,
    run_event_study,
    summarize_many,
    top_winner_dependency,
)

SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1H", "4H", "1D"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causal OHLCV breakout event study using src.data_feed.OKXDataLoader.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", default="5m", choices=sorted(SUPPORTED_TIMEFRAMES))
    p.add_argument("--context-timeframe", default="15m", choices=sorted(SUPPORTED_TIMEFRAMES))
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--out-dir", default="data/reports/research/ohlcv_breakout_event_study_lab")

    p.add_argument("--lookbacks", default="24,48,96,192", help="Comma-separated prior range windows in primary bars.")
    p.add_argument("--horizons", default="1,3,6,12,24", help="Comma-separated forward horizons in primary bars.")
    p.add_argument("--mfe-mae-horizon", type=int, default=24)
    p.add_argument("--atr-window", type=int, default=42)
    p.add_argument("--volume-window", type=int, default=120)
    p.add_argument("--ema-fast", type=int, default=20)
    p.add_argument("--ema-slow", type=int, default=50)

    p.add_argument("--volume-spike-threshold", type=float, default=1.50)
    p.add_argument("--compression-atr-rel-threshold", type=float, default=0.80)
    p.add_argument("--sweep-min-wick-frac", type=float, default=0.45)
    p.add_argument("--min-count", type=int, default=80)
    p.add_argument("--candidate-horizon", type=int, default=12)
    p.add_argument("--max-top5-winner-share", type=float, default=0.50)

    p.add_argument("--entry-fee-rate", type=float, default=0.00055)
    p.add_argument("--exit-fee-rate", type=float, default=0.00055)
    p.add_argument("--entry-slippage-pct", type=float, default=0.00020)
    p.add_argument("--exit-slippage-pct", type=float, default=0.00020)
    p.add_argument("--touch-target-pct", type=float, default=0.0040)
    p.add_argument("--touch-stop-pct", type=float, default=0.0040)
    p.add_argument("--save-feature-sample", type=int, default=5000, help="Rows to save from feature tail. 0 disables.")
    return p.parse_args(argv)


def _parse_int_list(text: str, *, name: str) -> list[int]:
    values: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"{name} must contain positive integers")
        values.append(value)
    if not values:
        raise ValueError(f"{name} must not be empty")
    return sorted(set(values))


def _timeframe_delta(timeframe: str) -> pd.Timedelta:
    mapping = {
        "1m": pd.Timedelta(minutes=1),
        "5m": pd.Timedelta(minutes=5),
        "15m": pd.Timedelta(minutes=15),
        "30m": pd.Timedelta(minutes=30),
        "1H": pd.Timedelta(hours=1),
        "4H": pd.Timedelta(hours=4),
        "1D": pd.Timedelta(days=1),
    }
    if timeframe not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return mapping[timeframe]


def _load_ohlcv_via_data_feed(symbol: str, timeframe: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load OHLCV exclusively through the project's src.data_feed interface."""
    loader = OKXDataLoader(symbol=symbol, timeframe=timeframe)
    df = loader.fetch_data_by_date_range(start_date, end_date)
    if df.empty:
        raise RuntimeError(f"No data loaded for {symbol} {timeframe} {start_date} -> {end_date}")
    df = df.sort_index().copy()
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Loaded data missing columns: {missing}")
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)
    if df.index.has_duplicates:
        df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _true_range(frame: pd.DataFrame) -> pd.Series:
    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return pd.to_numeric(tr, errors="coerce")


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    out = pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce").replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def _side_name(side: int) -> str:
    return "LONG" if int(side) == 1 else "SHORT" if int(side) == -1 else "FLAT"


def build_primary_features(bars: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Build causal primary-timeframe features from closed and prior bars."""
    df = bars.copy().sort_index()
    df["ret_1"] = df["close"].pct_change()
    df["tr"] = _true_range(df)
    df["atr"] = df["tr"].rolling(int(args.atr_window), min_periods=int(args.atr_window)).mean()
    df["atr_pct"] = _safe_divide(df["atr"], df["close"])
    df["atr_pct_prior_median"] = df["atr_pct"].shift(1).rolling(int(args.volume_window), min_periods=max(10, int(args.volume_window) // 3)).median()
    df["atr_rel_to_prior_median"] = _safe_divide(df["atr_pct"].shift(1), df["atr_pct_prior_median"])

    vol_median = df["volume"].shift(1).rolling(int(args.volume_window), min_periods=max(10, int(args.volume_window) // 3)).median()
    df["volume_ratio"] = _safe_divide(df["volume"], vol_median)
    df["volume_spike"] = df["volume_ratio"] >= float(args.volume_spike_threshold)
    df["compression"] = df["atr_rel_to_prior_median"] <= float(args.compression_atr_rel_threshold)

    df["ema_fast"] = df["close"].ewm(span=int(args.ema_fast), adjust=False, min_periods=int(args.ema_fast)).mean()
    df["ema_slow"] = df["close"].ewm(span=int(args.ema_slow), adjust=False, min_periods=int(args.ema_slow)).mean()
    df["primary_trend_side"] = np.select(
        [
            (df["ema_fast"] > df["ema_slow"]) & (df["close"] > df["ema_fast"]),
            (df["ema_fast"] < df["ema_slow"]) & (df["close"] < df["ema_fast"]),
        ],
        [1, -1],
        default=0,
    ).astype(int)

    bar_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    df["close_pos"] = _safe_divide(df["close"] - df["low"], bar_range)
    df["upper_wick_frac"] = _safe_divide(df["high"] - df[["open", "close"]].max(axis=1), bar_range)
    df["lower_wick_frac"] = _safe_divide(df[["open", "close"]].min(axis=1) - df["low"], bar_range)
    return df


def build_context_features(context_bars: pd.DataFrame, args: argparse.Namespace, context_timeframe: str) -> pd.DataFrame:
    """Build higher-timeframe context with explicit available_time columns."""
    ctx = context_bars.copy().sort_index()
    delta = _timeframe_delta(context_timeframe)
    ctx["ctx_bar_start_time"] = ctx.index
    ctx["ctx_available_time"] = ctx.index + delta
    ctx["ctx_ema_fast"] = ctx["close"].ewm(span=int(args.ema_fast), adjust=False, min_periods=int(args.ema_fast)).mean()
    ctx["ctx_ema_slow"] = ctx["close"].ewm(span=int(args.ema_slow), adjust=False, min_periods=int(args.ema_slow)).mean()
    ctx["ctx_trend_side"] = np.select(
        [
            (ctx["ctx_ema_fast"] > ctx["ctx_ema_slow"]) & (ctx["close"] > ctx["ctx_ema_fast"]),
            (ctx["ctx_ema_fast"] < ctx["ctx_ema_slow"]) & (ctx["close"] < ctx["ctx_ema_fast"]),
        ],
        [1, -1],
        default=0,
    ).astype(int)
    ctx_tr = _true_range(ctx)
    ctx["ctx_atr_pct"] = _safe_divide(ctx_tr.rolling(int(args.atr_window), min_periods=int(args.atr_window)).mean(), ctx["close"])
    return ctx[["ctx_bar_start_time", "ctx_available_time", "ctx_trend_side", "ctx_atr_pct"]]


def align_context(primary: pd.DataFrame, context_features: pd.DataFrame, context_timeframe: str) -> pd.DataFrame:
    """Causally align context by available_time through the event-study helper."""
    return causal_align_context(
        primary,
        context_features,
        timeframe=_timeframe_delta(context_timeframe),
        suffix="_ctxdup",
    )


def build_events(features: pd.DataFrame, args: argparse.Namespace, lookbacks: Sequence[int]) -> pd.DataFrame:
    """Build a multi-family event table using only closed/prior information."""
    rows: list[pd.DataFrame] = []
    ctx_side = pd.to_numeric(features.get("ctx_trend_side", 0), errors="coerce").fillna(0).astype(int)
    primary_side = pd.to_numeric(features.get("primary_trend_side", 0), errors="coerce").fillna(0).astype(int)

    for lookback in lookbacks:
        prev_high = features["high"].shift(1).rolling(int(lookback), min_periods=int(lookback)).max()
        prev_low = features["low"].shift(1).rolling(int(lookback), min_periods=int(lookback)).min()
        prior_range_pct = _safe_divide(prev_high - prev_low, features["close"])

        close_break_up = features["close"] > prev_high
        close_break_down = features["close"] < prev_low
        sweep_high_reject = (features["high"] > prev_high) & (features["close"] < prev_high) & (features["upper_wick_frac"] >= float(args.sweep_min_wick_frac))
        sweep_low_reclaim = (features["low"] < prev_low) & (features["close"] > prev_low) & (features["lower_wick_frac"] >= float(args.sweep_min_wick_frac))

        definitions: list[tuple[str, pd.Series, int | pd.Series]] = [
            (f"lb{lookback}_close_breakout", close_break_up, 1),
            (f"lb{lookback}_close_breakdown", close_break_down, -1),
            (f"lb{lookback}_volume_breakout", close_break_up & features["volume_spike"].fillna(False), 1),
            (f"lb{lookback}_volume_breakdown", close_break_down & features["volume_spike"].fillna(False), -1),
            (f"lb{lookback}_compression_breakout", close_break_up & features["compression"].fillna(False), 1),
            (f"lb{lookback}_compression_breakdown", close_break_down & features["compression"].fillna(False), -1),
            (f"lb{lookback}_primary_trend_breakout", close_break_up & (primary_side == 1), 1),
            (f"lb{lookback}_primary_trend_breakdown", close_break_down & (primary_side == -1), -1),
            (f"lb{lookback}_ctx_trend_breakout", close_break_up & (ctx_side == 1), 1),
            (f"lb{lookback}_ctx_trend_breakdown", close_break_down & (ctx_side == -1), -1),
            (
                f"lb{lookback}_vol_comp_ctx_breakout",
                close_break_up & features["volume_spike"].fillna(False) & features["compression"].fillna(False) & (ctx_side == 1),
                1,
            ),
            (
                f"lb{lookback}_vol_comp_ctx_breakdown",
                close_break_down & features["volume_spike"].fillna(False) & features["compression"].fillna(False) & (ctx_side == -1),
                -1,
            ),
            (f"lb{lookback}_sweep_high_reject", sweep_high_reject, -1),
            (f"lb{lookback}_sweep_low_reclaim", sweep_low_reclaim, 1),
        ]

        common_cols = pd.DataFrame(
            {
                "signal_time": features.index,
                "lookback": int(lookback),
                "prior_range_pct": prior_range_pct,
                "volume_ratio": features["volume_ratio"],
                "atr_rel_to_prior_median": features["atr_rel_to_prior_median"],
                "close_pos": features["close_pos"],
                "primary_trend_side": primary_side,
                "ctx_trend_side": ctx_side,
                "ctx_bar_start_time": features.get("ctx_bar_start_time", pd.NaT),
                "ctx_available_time": features.get("ctx_available_time", pd.NaT),
                "volume_spike": features["volume_spike"].fillna(False).astype(bool),
                "compression": features["compression"].fillna(False).astype(bool),
            },
            index=features.index,
        )

        for event_name, mask, side in definitions:
            mask = mask.fillna(False).astype(bool)
            if not mask.any():
                continue
            part = common_cols.loc[mask].copy()
            part["event_name"] = event_name
            part["event_family"] = event_name.split(f"lb{lookback}_", 1)[-1]
            part["side"] = int(side) if not isinstance(side, pd.Series) else side.loc[mask].astype(int)
            part["side_name"] = part["side"].map(_side_name)
            rows.append(part)

    if not rows:
        return pd.DataFrame(columns=["signal_time", "event_name", "side"])
    events = pd.concat(rows, axis=0, ignore_index=True).sort_values(["signal_time", "event_name"]).reset_index(drop=True)
    events = events[pd.to_datetime(events["signal_time"]) >= pd.Timestamp(args.start_date)].copy()
    events = events[pd.to_datetime(events["signal_time"]) <= pd.Timestamp(args.end_date)].copy()
    return events.reset_index(drop=True)


def attach_first_touch_labels(bars: pd.DataFrame, events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Use first_touch_outcome() and merge target/stop labels back onto event rows."""
    if events.empty:
        return events.copy()
    parts: list[pd.DataFrame] = []
    for side_value in (1, -1):
        times = pd.to_datetime(events.loc[events["side"].astype(int) == side_value, "signal_time"]).drop_duplicates()
        if times.empty:
            continue
        side_series = pd.Series(0, index=bars.index, dtype="int64")
        aligned_times = bars.index.intersection(pd.DatetimeIndex(times))
        side_series.loc[aligned_times] = side_value
        touch = first_touch_outcome(
            bars,
            side_series,
            target_pct=float(args.touch_target_pct),
            stop_pct=float(args.touch_stop_pct),
            horizon=int(args.mfe_mae_horizon),
            entry_delay_bars=1,
            same_bar_policy="conservative",
        )
        touch = touch.loc[aligned_times].copy()
        touch["signal_time"] = touch.index
        touch["side"] = side_value
        parts.append(touch.reset_index(drop=True))
    if not parts:
        return events.copy()
    touch_events = pd.concat(parts, ignore_index=True)
    out = events.merge(touch_events, on=["signal_time", "side"], how="left")
    return out


def add_event_bins(events: pd.DataFrame) -> pd.DataFrame:
    """Exercise event-study binning helpers on event-level causal features."""
    out = events.copy()
    if out.empty:
        return out
    out["volume_ratio_q"] = qcut_labels(out["volume_ratio"], q=4, prefix="VOL_Q")
    out["prior_range_pct_q"] = qcut_labels(out["prior_range_pct"], q=4, prefix="RANGE_Q")
    out["atr_rel_bucket"] = fixed_threshold_labels(
        out["atr_rel_to_prior_median"],
        thresholds=(0.70, 0.90, 1.10, 1.30),
        labels=("ATR_REL_VERY_LOW", "ATR_REL_LOW", "ATR_REL_MID", "ATR_REL_HIGH", "ATR_REL_VERY_HIGH"),
    )
    out["close_pos_bucket"] = fixed_threshold_labels(
        out["close_pos"],
        thresholds=(0.25, 0.50, 0.75),
        labels=("CLOSE_LOW", "CLOSE_MID_LOW", "CLOSE_MID_HIGH", "CLOSE_HIGH"),
    )
    return out


def _clean_metric_value(value: object) -> float:
    if isinstance(value, str) and value == "inf":
        return float("inf")
    try:
        x = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")
    return x


def build_candidate_rank(
    events: pd.DataFrame,
    event_name_stats: pd.DataFrame,
    event_yearly_stats: pd.DataFrame,
    *,
    primary_return_col: str,
    min_count: int,
    max_top5_winner_share: float,
) -> pd.DataFrame:
    """Build a conservative discovery shortlist. It is not a live-trading pass."""
    if event_name_stats.empty:
        return pd.DataFrame()
    stats = event_name_stats[event_name_stats["metric"] == primary_return_col].copy()
    if stats.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, row in stats.iterrows():
        event_name = row.get("event_name")
        sub = events.loc[events["event_name"] == event_name, primary_return_col]
        yearly = event_yearly_stats[
            (event_yearly_stats.get("event_name") == event_name) & (event_yearly_stats.get("metric") == primary_return_col)
        ].copy()
        eligible_yearly = yearly[yearly["eligible"].astype(bool)] if not yearly.empty and "eligible" in yearly.columns else yearly
        positive_years = int((pd.to_numeric(eligible_yearly.get("mean", pd.Series(dtype=float)), errors="coerce") > 0).sum()) if not eligible_yearly.empty else 0
        tested_years = int(len(eligible_yearly))
        yearly_positive_rate = positive_years / tested_years if tested_years > 0 else np.nan
        top5_share = top_winner_dependency(sub, top_n=5)
        pf = _clean_metric_value(row.get("profit_factor"))
        mean = _clean_metric_value(row.get("mean"))
        median = _clean_metric_value(row.get("median"))
        win_rate = _clean_metric_value(row.get("win_rate"))
        count = int(row.get("count", 0) or 0)
        candidate_flag = bool(
            count >= int(min_count)
            and np.isfinite(mean)
            and mean > 0
            and np.isfinite(median)
            and median > -0.001
            and np.isfinite(pf)
            and pf > 1.05
            and (not np.isfinite(top5_share) or top5_share <= float(max_top5_winner_share))
            and (not np.isfinite(yearly_positive_rate) or yearly_positive_rate >= 0.50)
        )
        rows.append(
            {
                "event_name": event_name,
                "metric": primary_return_col,
                "count": count,
                "mean": mean,
                "median": median,
                "win_rate": win_rate,
                "profit_factor": pf,
                "payoff_ratio": _clean_metric_value(row.get("payoff_ratio")),
                "top5_winner_share_recalc": top5_share,
                "tested_years": tested_years,
                "positive_years": positive_years,
                "yearly_positive_rate": yearly_positive_rate,
                "candidate_flag": candidate_flag,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["rank_score"] = (
        pd.to_numeric(out["mean"], errors="coerce").fillna(-999.0) * 1000.0
        + (pd.to_numeric(out["profit_factor"], errors="coerce").replace(np.inf, 10.0).fillna(0.0) - 1.0)
        + (pd.to_numeric(out["win_rate"], errors="coerce").fillna(0.0) - 0.5)
    )
    return out.sort_values(["candidate_flag", "rank_score", "count"], ascending=[False, False, False])


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def run_lab(args: argparse.Namespace) -> dict[str, object]:
    lookbacks = _parse_int_list(args.lookbacks, name="lookbacks")
    horizons = _parse_int_list(args.horizons, name="horizons")
    if int(args.candidate_horizon) not in horizons:
        horizons = sorted(set([*horizons, int(args.candidate_horizon)]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[1/6] Loading primary OHLCV via src.data_feed: {args.symbol} {args.timeframe} "
        f"{args.warmup_start_date} -> {args.end_date}",
        flush=True,
    )
    primary_bars = _load_ohlcv_via_data_feed(args.symbol, args.timeframe, args.warmup_start_date, args.end_date)
    print(f"      primary rows={len(primary_bars):,} range={primary_bars.index[0]} -> {primary_bars.index[-1]}", flush=True)

    print(
        f"[2/6] Loading context OHLCV via src.data_feed: {args.symbol} {args.context_timeframe} "
        f"{args.warmup_start_date} -> {args.end_date}",
        flush=True,
    )
    context_bars = _load_ohlcv_via_data_feed(args.symbol, args.context_timeframe, args.warmup_start_date, args.end_date)
    print(f"      context rows={len(context_bars):,} range={context_bars.index[0]} -> {context_bars.index[-1]}", flush=True)

    print("[3/6] Building past-only features and causal context alignment", flush=True)
    features = build_primary_features(primary_bars, args)
    context_features = build_context_features(context_bars, args, args.context_timeframe)
    features = align_context(features, context_features, args.context_timeframe)

    # Use only the trade window for event generation/evaluation, but keep warmup-built features.
    eval_bars = primary_bars.loc[pd.Timestamp(args.start_date): pd.Timestamp(args.end_date)].copy()
    eval_features = features.loc[pd.Timestamp(args.start_date): pd.Timestamp(args.end_date)].copy()
    if eval_bars.empty or eval_features.empty:
        raise RuntimeError("No rows after start/end date slicing.")

    print("[4/6] Building event table", flush=True)
    events = build_events(eval_features, args, lookbacks)
    if events.empty:
        raise RuntimeError("No events were generated. Try smaller lookbacks or a longer date range.")
    events = add_event_bins(events)
    print(f"      events={len(events):,} unique_event_names={events['event_name'].nunique():,}", flush=True)

    print("[5/6] Running event-study core: returns, MFE/MAE, cost adjustment, causal audit", flush=True)
    config = EventStudyConfig(
        horizons=tuple(horizons),
        mfe_mae_horizon=int(args.mfe_mae_horizon),
        cost=CostConfig(
            entry_fee_rate=float(args.entry_fee_rate),
            exit_fee_rate=float(args.exit_fee_rate),
            entry_slippage_pct=float(args.entry_slippage_pct),
            exit_slippage_pct=float(args.exit_slippage_pct),
        ),
        context_available_time_cols=("ctx_available_time",),
        min_count=int(args.min_count),
    )
    result = run_event_study(eval_bars, events, config)
    result_events = attach_first_touch_labels(eval_bars, result.events, args)
    result = result.__class__(
        events=result_events,
        overview=result.overview,
        yearly=result.yearly,
        side_stats=result.side_stats,
        horizon_stats=result.horizon_stats,
        causal_audit=result.causal_audit,
        meta=result.meta,
    )
    result.write(out_dir)

    print("[6/6] Writing extra grouped stats, contrasts, bins, first-touch stats, and candidate rank", flush=True)
    return_cols = [f"next_open_ret_h{int(h)}_net" for h in horizons]
    primary_return_col = f"next_open_ret_h{int(args.candidate_horizon)}_net"
    event_name_stats = summarize_many(result.events, return_cols, group_cols=["event_name"], min_count=int(args.min_count))
    event_family_stats = summarize_many(result.events, return_cols, group_cols=["event_family"], min_count=int(args.min_count))
    event_yearly_stats = summarize_many(result.events, return_cols, group_cols=["event_name", "year"], min_count=max(10, int(args.min_count) // 3))
    bin_stats = summarize_many(
        result.events,
        [primary_return_col],
        group_cols=["event_family", "volume_ratio_q", "atr_rel_bucket", "close_pos_bucket"],
        min_count=max(10, int(args.min_count) // 3),
    )

    contrasts = []
    for condition in ["volume_spike", "compression"]:
        try:
            c = condition_contrast(result.events, condition_col=condition, return_col=primary_return_col, min_count=max(10, int(args.min_count) // 3))
            contrasts.append(c)
        except KeyError:
            pass
    contrast_df = pd.concat(contrasts, ignore_index=True) if contrasts else pd.DataFrame()

    touch_stats = pd.DataFrame()
    if "touch_result" in result.events.columns:
        touch_stats = result.events.groupby(["event_name", "touch_result"], dropna=False).size().reset_index(name="count")
        totals = touch_stats.groupby("event_name")["count"].transform("sum")
        touch_stats["share"] = touch_stats["count"] / totals.replace(0, np.nan)

    candidate_rank = build_candidate_rank(
        result.events,
        event_name_stats,
        event_yearly_stats,
        primary_return_col=primary_return_col,
        min_count=int(args.min_count),
        max_top5_winner_share=float(args.max_top5_winner_share),
    )

    event_name_stats.to_csv(out_dir / "06_event_name_stats.csv", index=False)
    event_family_stats.to_csv(out_dir / "07_event_family_stats.csv", index=False)
    event_yearly_stats.to_csv(out_dir / "09_event_name_yearly_stats.csv", index=False)
    contrast_df.to_csv(out_dir / "11_condition_contrast.csv", index=False)
    bin_stats.to_csv(out_dir / "12_feature_bin_stats.csv", index=False)
    touch_stats.to_csv(out_dir / "13_first_touch_stats.csv", index=False)
    candidate_rank.to_csv(out_dir / "14_candidate_rank.csv", index=False)

    if int(args.save_feature_sample) > 0:
        eval_features.tail(int(args.save_feature_sample)).to_csv(out_dir / "15_feature_tail_sample.csv")

    best = candidate_rank.head(20).copy() if not candidate_rank.empty else pd.DataFrame()
    meta = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "context_timeframe": args.context_timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "lookbacks": lookbacks,
        "horizons": horizons,
        "candidate_horizon": int(args.candidate_horizon),
        "event_count": int(len(result.events)),
        "unique_event_names": int(result.events["event_name"].nunique()) if not result.events.empty else 0,
        "causal_fail_count": int(result.causal_audit["causal_fail_flag"].sum()) if not result.causal_audit.empty else 0,
        "round_trip_cost_pct": float(config.cost.round_trip_cost_pct),
        "best_candidates": best.to_dict(orient="records"),
        "notes": "Discovery-only event study. Positive candidates are phenomena, not validated strategies.",
    }
    _write_json(out_dir / "16_lab_meta.json", meta)

    if not best.empty:
        print("\nTop candidate rows:", flush=True)
        print(
            best[["event_name", "count", "mean", "median", "win_rate", "profit_factor", "candidate_flag"]]
            .head(10)
            .to_string(index=False),
            flush=True,
        )
    else:
        print("No eligible candidate rows were ranked.", flush=True)
    print(f"\nReport written to: {out_dir}", flush=True)
    return meta


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_lab(args)


if __name__ == "__main__":
    main()
