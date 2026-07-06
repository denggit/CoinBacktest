#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Sweep Reversal Anti-Martingale Event Lab
=======================================

Research-only lab for the hypothesis:
    After an ultra-short spike sweeps a recently confirmed swing high/low,
    ETH may reverse into a wave. A pyramiding/anti-martingale structure may add
    only after price continues in the reversal direction, while one structural
    stop protects the whole position.

Boundaries:
    - Data is loaded only through src.data_feed loaders.
    - Swing highs/lows are confirmed with right-side bars, then shifted before
      they can be used by a signal. No unconfirmed pivot is used.
    - Signals are generated on a closed bar and evaluated/executed on the next
      bar open by the reusable event-study module.
    - The pyramiding simulator is a research diagnostic, not production logic.
    - Positive output is a phenomenon only; it still needs full backtest,
      parameter neighbourhood, fee/slippage/delay stress and live execution
      review before promotion.

Examples:
    python research/sweep_reversal_antimartingale_event_lab.py --data-source ohlcv --timeframe 1m --start-date 2023-01-01 --end-date 2026-06-30 --warmup-start-date 2022-01-01
    python research/sweep_reversal_antimartingale_event_lab.py --data-source trade_bar --timeframe 1m --start-date 2023-01-01 --end-date 2026-06-30 --warmup-start-date 2022-01-01
    python research/sweep_reversal_antimartingale_event_lab.py --data-source range_bar --range-pct 0.0015 --start-date 2023-01-01 --end-date 2026-06-30 --warmup-start-date 2022-01-01
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_feed.okx_loader import OKXDataLoader  # noqa: E402
from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader  # noqa: E402
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.event_study import (  # noqa: E402
    CostConfig,
    EventStudyConfig,
    condition_contrast,
    first_touch_outcome,
    fixed_threshold_labels,
    qcut_labels,
    run_event_study,
    summarize_many,
    top_winner_dependency,
)
from src.research_common.progress import ProgressReporter  # noqa: E402

SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1H", "4H", "1D"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causal swing-sweep reversal event study + research pyramiding simulator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--data-source", choices=["ohlcv", "trade_bar", "range_bar"], default="trade_bar")
    p.add_argument("--timeframe", choices=sorted(SUPPORTED_TIMEFRAMES), default="1m", help="Used by ohlcv/trade_bar sources.")
    p.add_argument("--range-pct", type=float, default=0.0015, help="Used by range_bar source.")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--out-dir", default="data/reports/research/sweep_reversal_antimartingale_event_lab")

    p.add_argument("--pivot-left", type=int, default=6)
    p.add_argument("--pivot-right", type=int, default=3)
    p.add_argument("--min-swing-age", type=int, default=3)
    p.add_argument("--max-swing-ages", default="12,24,48")
    p.add_argument("--min-swing-prominence-pcts", default="0.0015,0.0030")
    p.add_argument("--spike-pcts", default="0.0030,0.0050,0.0080")
    p.add_argument("--breakout-pcts", default="0.0000,0.0005")
    p.add_argument("--wick-min-frac", type=float, default=0.45)
    p.add_argument("--volume-window", type=int, default=120)
    p.add_argument("--atr-window", type=int, default=42)
    p.add_argument("--volume-spike-threshold", type=float, default=1.50)
    p.add_argument("--horizons", default="1,3,6,12,24,48")
    p.add_argument("--candidate-horizon", type=int, default=12)
    p.add_argument("--mfe-mae-horizon", type=int, default=48)
    p.add_argument("--min-count", type=int, default=80)
    p.add_argument("--max-top5-winner-share", type=float, default=0.50)
    p.add_argument("--dedupe-events", action="store_true", help="Keep one event per signal_time/side/event_name.")

    p.add_argument("--entry-fee-rate", type=float, default=0.00055)
    p.add_argument("--exit-fee-rate", type=float, default=0.00055)
    p.add_argument("--entry-slippage-pct", type=float, default=0.00020)
    p.add_argument("--exit-slippage-pct", type=float, default=0.00020)
    p.add_argument("--touch-target-pct", type=float, default=0.0040)
    p.add_argument("--touch-stop-pct", type=float, default=0.0040)

    p.add_argument("--run-pyramiding-sim", action="store_true", default=True)
    p.add_argument("--no-pyramiding-sim", dest="run_pyramiding_sim", action="store_false")
    p.add_argument("--sim-top-event-names", type=int, default=30)
    p.add_argument("--sim-min-event-count", type=int, default=100, help="Minimum event-study count required before an event_name is sent to the pyramiding simulator.")
    p.add_argument("--sim-min-tested-years", type=int, default=3, help="Minimum eligible yearly buckets required before pyramiding simulation.")
    p.add_argument("--sim-include-noncandidates", action="store_true", help="Allow pyramiding simulation on non-candidate event names. By default only candidate_flag=True rows are simulated.")
    p.add_argument("--max-sim-events-per-name", type=int, default=5000)
    p.add_argument("--max-hold-bars", type=int, default=48)
    p.add_argument("--max-adds", type=int, default=3)
    p.add_argument("--add-lookback", type=int, default=6)
    p.add_argument("--add-breakout-pct", type=float, default=0.0005)
    p.add_argument("--add-size-mult", type=float, default=1.0)
    p.add_argument("--trail-lookback", type=int, default=8)
    p.add_argument("--stop-buffer-pct", type=float, default=0.0005)
    p.add_argument("--min-initial-stop-pct", type=float, default=0.0015)
    p.add_argument("--max-initial-stop-pct", type=float, default=0.0200)

    p.add_argument("--progress-every", type=int, default=25000)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--save-feature-sample", type=int, default=5000)
    return p.parse_args(argv)


def _parse_number_list(text: str, *, cast=float, name: str = "values") -> list:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = cast(part)
        if value <= 0 and name not in {"breakout_pcts"}:
            raise ValueError(f"{name} must contain positive values")
        if value < 0:
            raise ValueError(f"{name} must not contain negative values")
        out.append(value)
    if not out:
        raise ValueError(f"{name} must not be empty")
    return sorted(set(out))


def _safe_divide(a: pd.Series, b: pd.Series) -> pd.Series:
    out = pd.to_numeric(a, errors="coerce") / pd.to_numeric(b, errors="coerce").replace(0.0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


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


def _load_bars_via_data_feed(args: argparse.Namespace) -> pd.DataFrame:
    if args.data_source == "ohlcv":
        print(f"[load] OKXDataLoader {args.symbol} {args.timeframe} {args.warmup_start_date}->{args.end_date}", flush=True)
        df = OKXDataLoader(symbol=args.symbol, timeframe=args.timeframe).fetch_data_by_date_range(args.warmup_start_date, args.end_date)
    elif args.data_source == "trade_bar":
        print(f"[load] OKXTradeBarLoader {args.symbol} {args.timeframe} {args.warmup_start_date}->{args.end_date}", flush=True)
        df = OKXTradeBarLoader(symbol=args.symbol, timeframe=args.timeframe).fetch_data_by_date_range(args.warmup_start_date, args.end_date)
    elif args.data_source == "range_bar":
        print(f"[load] OKXRangeBarLoader {args.symbol} range_pct={args.range_pct} {args.warmup_start_date}->{args.end_date}", flush=True)
        df = OKXRangeBarLoader(symbol=args.symbol, range_pct=float(args.range_pct)).fetch_data_by_date_range(args.warmup_start_date, args.end_date)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported data_source: {args.data_source}")

    if df.empty:
        raise RuntimeError(f"No data loaded for {args.symbol} source={args.data_source}")
    out = df.copy().sort_index()
    if "end_ts" in out.columns and args.data_source == "range_bar":
        out.index = pd.to_datetime(out["end_ts"], errors="coerce")
        out = out.sort_index()
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise RuntimeError(f"Loaded data missing required columns: {missing}")
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=required)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    print(f"       rows={len(out):,} range={out.index[0]} -> {out.index[-1]}", flush=True)
    return out


def _confirmed_swings(df: pd.DataFrame, *, left: int, right: int) -> pd.DataFrame:
    """Past-only confirmed swing levels and their age/prominence.

    A pivot centered at bar j is only confirmed at j + right. We then shift one
    more bar before using it, so a signal bar never uses a pivot confirmed by
    itself. This is intentionally conservative.
    """
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    pos = pd.Series(np.arange(len(df), dtype=float), index=df.index)
    left_high = high.shift(1).rolling(left, min_periods=left).max()
    right_high = high.iloc[::-1].shift(1).rolling(right, min_periods=right).max().iloc[::-1]
    left_low = low.shift(1).rolling(left, min_periods=left).min()
    right_low = low.iloc[::-1].shift(1).rolling(right, min_periods=right).min().iloc[::-1]
    pivot_high = (high > left_high) & (high >= right_high)
    pivot_low = (low < left_low) & (low <= right_low)

    window = left + right + 1
    local_low = low.rolling(window, center=True, min_periods=window).min()
    local_high = high.rolling(window, center=True, min_periods=window).max()
    high_prom = _safe_divide(high, local_low) - 1.0
    low_prom = _safe_divide(local_high, low) - 1.0

    confirmed_high = high.where(pivot_high).shift(right).shift(1).ffill()
    confirmed_low = low.where(pivot_low).shift(right).shift(1).ffill()
    confirmed_high_pos = pos.where(pivot_high).shift(right).shift(1).ffill()
    confirmed_low_pos = pos.where(pivot_low).shift(right).shift(1).ffill()
    confirmed_high_prom = high_prom.where(pivot_high).shift(right).shift(1).ffill()
    confirmed_low_prom = low_prom.where(pivot_low).shift(right).shift(1).ffill()

    out = pd.DataFrame(index=df.index)
    out["swing_high"] = confirmed_high
    out["swing_low"] = confirmed_low
    out["swing_high_age"] = pos - confirmed_high_pos
    out["swing_low_age"] = pos - confirmed_low_pos
    out["swing_high_prominence_pct"] = confirmed_high_prom
    out["swing_low_prominence_pct"] = confirmed_low_prom
    return out


def build_features(bars: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = bars.copy().sort_index()
    df["prev_close"] = df["close"].shift(1)
    df["ret_close_to_close"] = df["close"].pct_change()
    df["bar_body_ret"] = df["close"] / df["open"] - 1.0
    df["up_spike_pct"] = df["high"] / df["prev_close"] - 1.0
    df["down_spike_pct"] = df["prev_close"] / df["low"] - 1.0
    df["tr"] = _true_range(df)
    df["atr"] = df["tr"].rolling(int(args.atr_window), min_periods=int(args.atr_window)).mean()
    df["atr_pct"] = _safe_divide(df["atr"], df["close"])
    vol_base = df["volume"].shift(1).rolling(int(args.volume_window), min_periods=max(10, int(args.volume_window) // 3)).median()
    df["volume_ratio"] = _safe_divide(df["volume"], vol_base)
    df["volume_spike"] = df["volume_ratio"] >= float(args.volume_spike_threshold)
    bar_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    df["upper_wick_frac"] = _safe_divide(df["high"] - df[["open", "close"]].max(axis=1), bar_range)
    df["lower_wick_frac"] = _safe_divide(df[["open", "close"]].min(axis=1) - df["low"], bar_range)
    swings = _confirmed_swings(df, left=int(args.pivot_left), right=int(args.pivot_right))
    return pd.concat([df, swings], axis=1)


def _event_row(ts: pd.Timestamp, side: int, event_name: str, family: str, row: pd.Series, extra: dict[str, object]) -> dict[str, object]:
    is_short = int(side) < 0
    swing_level = row["swing_high"] if is_short else row["swing_low"]
    stop_extreme = row["high"] if is_short else row["low"]
    stop_level = max(float(stop_extreme), float(swing_level)) if is_short else min(float(stop_extreme), float(swing_level))
    return {
        "signal_time": ts,
        "side": int(side),
        "event_name": event_name,
        "event_family": family,
        "swing_level": float(swing_level),
        "sweep_extreme": float(stop_extreme),
        "structural_stop_level": float(stop_level),
        "swing_age": float(row["swing_high_age"] if is_short else row["swing_low_age"]),
        "swing_prominence_pct": float(row["swing_high_prominence_pct"] if is_short else row["swing_low_prominence_pct"]),
        "up_spike_pct": float(row.get("up_spike_pct", np.nan)),
        "down_spike_pct": float(row.get("down_spike_pct", np.nan)),
        "volume_ratio": float(row.get("volume_ratio", np.nan)),
        "volume_spike": bool(row.get("volume_spike", False)),
        "atr_pct": float(row.get("atr_pct", np.nan)),
        "upper_wick_frac": float(row.get("upper_wick_frac", np.nan)),
        "lower_wick_frac": float(row.get("lower_wick_frac", np.nan)),
        "close": float(row["close"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        **extra,
    }


def build_events(features: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    spike_pcts = _parse_number_list(args.spike_pcts, cast=float, name="spike_pcts")
    breakout_pcts = _parse_number_list(args.breakout_pcts, cast=float, name="breakout_pcts")
    max_ages = _parse_number_list(args.max_swing_ages, cast=int, name="max_swing_ages")
    min_proms = _parse_number_list(args.min_swing_prominence_pcts, cast=float, name="min_swing_prominence_pcts")

    rows: list[dict[str, object]] = []
    total = len(spike_pcts) * len(breakout_pcts) * len(max_ages) * len(min_proms) * 3 * 2
    progress = ProgressReporter(
        label="[events] sweep variants",
        total=total,
        every=1,
        enabled=not bool(args.no_progress),
    )
    done = 0
    for spike_pct in spike_pcts:
        for breakout_pct in breakout_pcts:
            for max_age in max_ages:
                for min_prom in min_proms:
                    high_base = (
                        features["swing_high"].notna()
                        & features["swing_high_age"].between(int(args.min_swing_age), int(max_age), inclusive="both")
                        & (features["swing_high_prominence_pct"] >= float(min_prom))
                        & (features["up_spike_pct"] >= float(spike_pct))
                        & (features["high"] >= features["swing_high"] * (1.0 + float(breakout_pct)))
                    )
                    low_base = (
                        features["swing_low"].notna()
                        & features["swing_low_age"].between(int(args.min_swing_age), int(max_age), inclusive="both")
                        & (features["swing_low_prominence_pct"] >= float(min_prom))
                        & (features["down_spike_pct"] >= float(spike_pct))
                        & (features["low"] <= features["swing_low"] * (1.0 - float(breakout_pct)))
                    )
                    variants = [
                        ("reject", high_base & (features["close"] < features["swing_high"]), low_base & (features["close"] > features["swing_low"])),
                        ("wick", high_base & (features["upper_wick_frac"] >= float(args.wick_min_frac)), low_base & (features["lower_wick_frac"] >= float(args.wick_min_frac))),
                        ("fade_close_through", high_base & (features["close"] >= features["swing_high"]), low_base & (features["close"] <= features["swing_low"])),
                    ]
                    for variant, high_mask, low_mask in variants:
                        family = f"sweep_reversal_{variant}"
                        name_suffix = f"sp{int(spike_pct*10000):04d}_br{int(breakout_pct*10000):04d}_age{max_age}_prom{int(min_prom*10000):04d}"
                        high_events = features.loc[high_mask]
                        for ts, row in high_events.iterrows():
                            rows.append(_event_row(ts, -1, f"high_{variant}_{name_suffix}", family, row, {
                                "spike_threshold_pct": spike_pct,
                                "breakout_threshold_pct": breakout_pct,
                                "max_swing_age": max_age,
                                "min_swing_prominence_pct": min_prom,
                            }))
                        done += 1
                        progress.update(done)
                        low_events = features.loc[low_mask]
                        for ts, row in low_events.iterrows():
                            rows.append(_event_row(ts, 1, f"low_{variant}_{name_suffix}", family, row, {
                                "spike_threshold_pct": spike_pct,
                                "breakout_threshold_pct": breakout_pct,
                                "max_swing_age": max_age,
                                "min_swing_prominence_pct": min_prom,
                            }))
                        done += 1
                        progress.update(done)
    progress.close()
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events = events.sort_values(["signal_time", "side", "event_name"]).reset_index(drop=True)
    if args.dedupe_events:
        events = events.drop_duplicates(["signal_time", "side", "event_name"], keep="first").reset_index(drop=True)
    return events


def add_event_bins(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["volume_ratio_q"] = qcut_labels(out["volume_ratio"], q=4, prefix="VOL_Q")
    out["atr_pct_q"] = qcut_labels(out["atr_pct"], q=4, prefix="ATR_Q")
    out["swing_prominence_q"] = qcut_labels(out["swing_prominence_pct"], q=4, prefix="PROM_Q")
    out["swing_age_bucket"] = fixed_threshold_labels(
        out["swing_age"],
        thresholds=[6, 12, 24, 48],
        labels=["AGE_0_6", "AGE_7_12", "AGE_13_24", "AGE_25_48", "AGE_GT_48"],
    )
    out["spike_bucket"] = fixed_threshold_labels(
        out[["up_spike_pct", "down_spike_pct"]].max(axis=1),
        thresholds=[0.003, 0.005, 0.008, 0.012],
        labels=["SPIKE_LT_30", "SPIKE_30_50", "SPIKE_50_80", "SPIKE_80_120", "SPIKE_GT_120"],
    )
    return out


def attach_first_touch_labels(bars: pd.DataFrame, events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    side_by_time = pd.Series(0, index=bars.index, dtype=int)
    valid = events[events["signal_time"].isin(bars.index)].drop_duplicates("signal_time", keep="first")
    if not valid.empty:
        side_by_time.loc[pd.to_datetime(valid["signal_time"])] = valid["side"].astype(int).to_numpy()
    touch = first_touch_outcome(
        bars,
        side_by_time,
        target_pct=float(args.touch_target_pct),
        stop_pct=float(args.touch_stop_pct),
        horizon=int(args.mfe_mae_horizon),
        entry_delay_bars=1,
        same_bar_policy="conservative",
    )
    # first_touch_outcome preserves the bars index name. On local data this is often
    # named ``timestamp``/``datetime`` rather than None, so reset_index() does not
    # necessarily create an ``index`` column. Always rename the first reset column
    # to signal_time to keep this helper index-name agnostic.
    touch = touch.reset_index()
    touch = touch.rename(columns={touch.columns[0]: "signal_time"})
    touch["signal_time"] = pd.to_datetime(touch["signal_time"])
    return events.merge(touch, on="signal_time", how="left")


def _metric_float(value: object) -> float:
    if isinstance(value, str) and value.lower() == "inf":
        return float("inf")
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return out


def build_candidate_rank(events: pd.DataFrame, stats: pd.DataFrame, yearly: pd.DataFrame, *, return_col: str, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    metric_stats = stats[stats["metric"] == return_col] if not stats.empty else pd.DataFrame()
    for _, row in metric_stats.iterrows():
        name = row.get("event_name")
        sub = pd.to_numeric(events.loc[events["event_name"] == name, return_col], errors="coerce").dropna()
        y = yearly[(yearly["event_name"] == name) & (yearly["metric"] == return_col)] if not yearly.empty else pd.DataFrame()
        eligible_y = y[y["eligible"].astype(bool)] if not y.empty and "eligible" in y.columns else y
        pos_years = int((pd.to_numeric(eligible_y.get("mean", pd.Series(dtype=float)), errors="coerce") > 0).sum()) if not eligible_y.empty else 0
        tested_years = int(len(eligible_y))
        yearly_positive_rate = pos_years / tested_years if tested_years else np.nan
        mean = _metric_float(row.get("mean"))
        median = _metric_float(row.get("median"))
        pf = _metric_float(row.get("profit_factor"))
        win_rate = _metric_float(row.get("win_rate"))
        top5 = top_winner_dependency(sub, top_n=5)
        count = int(row.get("count", 0) or 0)
        flag = bool(
            count >= int(args.min_count)
            and np.isfinite(mean) and mean > 0
            and np.isfinite(median) and median > -0.001
            and np.isfinite(pf) and pf > 1.05
            and (not np.isfinite(top5) or top5 <= float(args.max_top5_winner_share))
            and (not np.isfinite(yearly_positive_rate) or yearly_positive_rate >= 0.50)
        )
        rows.append({
            "event_name": name,
            "metric": return_col,
            "count": count,
            "mean": mean,
            "median": median,
            "win_rate": win_rate,
            "profit_factor": pf,
            "payoff_ratio": _metric_float(row.get("payoff_ratio")),
            "top5_winner_share_recalc": top5,
            "tested_years": tested_years,
            "positive_years": pos_years,
            "yearly_positive_rate": yearly_positive_rate,
            "candidate_flag": flag,
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    pf_score = pd.to_numeric(out["profit_factor"], errors="coerce").replace(np.inf, 10.0).fillna(0.0)
    out["rank_score"] = pd.to_numeric(out["mean"], errors="coerce").fillna(-999.0) * 1000.0 + (pf_score - 1.0) + (pd.to_numeric(out["win_rate"], errors="coerce").fillna(0.0) - 0.5)
    return out.sort_values(["candidate_flag", "rank_score", "count"], ascending=[False, False, False])


def _fill_entry_price(raw_open: float, side: int, slippage_pct: float) -> float:
    return raw_open * (1.0 + slippage_pct) if side > 0 else raw_open * (1.0 - slippage_pct)


def _fill_exit_price(raw_price: float, side: int, slippage_pct: float) -> float:
    return raw_price * (1.0 - slippage_pct) if side > 0 else raw_price * (1.0 + slippage_pct)


def select_pyramiding_event_names(candidate_rank: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Select event names worth sending into the slower pyramiding simulator.

    The simulator used to run on the top ranked rows even when candidate_flag=False.
    That made range-bar reports misleading: tiny 3-8 sample event names could rank high
    by luck and then produce one-digit trade counts in 18_pyramiding_summary.csv.
    This selector keeps the simulator aligned with the candidate gate by default.
    """
    if candidate_rank.empty:
        return pd.DataFrame(columns=["event_name", "sim_selected", "sim_skip_reason"])

    out = candidate_rank.copy()
    min_count = max(int(args.min_count), int(args.sim_min_event_count))
    min_years = int(args.sim_min_tested_years)

    reasons: list[str] = []
    selected: list[bool] = []
    for _, row in out.iterrows():
        reason_parts = []
        count = int(row.get("count", 0) or 0)
        tested_years = int(row.get("tested_years", 0) or 0)
        candidate_flag = bool(row.get("candidate_flag", False))
        if count < min_count:
            reason_parts.append(f"count<{min_count}")
        if tested_years < min_years:
            reason_parts.append(f"tested_years<{min_years}")
        if (not bool(args.sim_include_noncandidates)) and not candidate_flag:
            reason_parts.append("candidate_flag_false")
        ok = not reason_parts
        selected.append(ok)
        reasons.append("" if ok else ";".join(reason_parts))

    out["sim_selected"] = selected
    out["sim_skip_reason"] = reasons
    selected_out = out[out["sim_selected"].astype(bool)].head(int(args.sim_top_event_names)).copy()
    return selected_out


def simulate_pyramiding_group(bars: pd.DataFrame, group_events: pd.DataFrame, args: argparse.Namespace, *, label: str) -> pd.DataFrame:
    if group_events.empty:
        return pd.DataFrame()
    idx = bars.index
    pos_map = pd.Series(np.arange(len(idx)), index=idx)
    opens = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    prior_high_break = pd.to_numeric(bars["high"], errors="coerce").shift(1).rolling(int(args.add_lookback), min_periods=int(args.add_lookback)).max() * (1.0 + float(args.add_breakout_pct))
    prior_low_break = pd.to_numeric(bars["low"], errors="coerce").shift(1).rolling(int(args.add_lookback), min_periods=int(args.add_lookback)).min() * (1.0 - float(args.add_breakout_pct))
    trail_high = pd.to_numeric(bars["high"], errors="coerce").shift(1).rolling(int(args.trail_lookback), min_periods=int(args.trail_lookback)).max() * (1.0 + float(args.stop_buffer_pct))
    trail_low = pd.to_numeric(bars["low"], errors="coerce").shift(1).rolling(int(args.trail_lookback), min_periods=int(args.trail_lookback)).min() * (1.0 - float(args.stop_buffer_pct))
    add_long_level = prior_high_break.to_numpy(dtype=float)
    add_short_level = prior_low_break.to_numpy(dtype=float)
    trail_high_arr = trail_high.to_numpy(dtype=float)
    trail_low_arr = trail_low.to_numpy(dtype=float)

    rows: list[dict[str, object]] = []
    events = group_events.drop_duplicates(["signal_time", "side"], keep="first").head(int(args.max_sim_events_per_name)).copy()
    progress = ProgressReporter(label=f"[pyramid] {label[:32]}", total=len(events), every=max(1, min(int(args.progress_every), max(1, len(events)//5))), enabled=not bool(args.no_progress))
    for done, ev in enumerate(events.itertuples(index=False), start=1):
        signal_time = pd.Timestamp(getattr(ev, "signal_time"))
        if signal_time not in pos_map.index:
            progress.update(done)
            continue
        signal_pos = int(pos_map.loc[signal_time])
        side = int(getattr(ev, "side"))
        entry_pos = signal_pos + 1
        if entry_pos >= len(idx):
            progress.update(done)
            continue
        base_stop = float(getattr(ev, "structural_stop_level"))
        entry_raw = opens[entry_pos]
        if not np.isfinite(entry_raw) or entry_raw <= 0 or not np.isfinite(base_stop) or base_stop <= 0:
            progress.update(done)
            continue
        entry_price = _fill_entry_price(entry_raw, side, float(args.entry_slippage_pct))
        initial_stop_pct = (base_stop - entry_price) / entry_price if side < 0 else (entry_price - base_stop) / entry_price
        if not np.isfinite(initial_stop_pct) or initial_stop_pct < float(args.min_initial_stop_pct) or initial_stop_pct > float(args.max_initial_stop_pct):
            progress.update(done)
            continue
        stop = base_stop * (1.0 + float(args.stop_buffer_pct)) if side < 0 else base_stop * (1.0 - float(args.stop_buffer_pct))
        leg_entries = [entry_price]
        leg_sizes = [1.0]
        adds = 0
        pending_add = False
        exit_pos = min(len(idx) - 1, entry_pos + int(args.max_hold_bars))
        exit_reason = "TIME"
        exit_raw = closes[exit_pos]
        stop_hit = False
        for pos in range(entry_pos, min(len(idx), entry_pos + int(args.max_hold_bars) + 1)):
            if pending_add:
                if (side > 0 and opens[pos] <= stop) or (side < 0 and opens[pos] >= stop):
                    exit_pos = pos
                    exit_raw = opens[pos]
                    exit_reason = "GAP_STOP_BEFORE_ADD"
                    stop_hit = True
                    break
                add_price = _fill_entry_price(opens[pos], side, float(args.entry_slippage_pct))
                leg_entries.append(add_price)
                leg_sizes.append(float(args.add_size_mult))
                adds += 1
                pending_add = False
            if side > 0 and lows[pos] <= stop:
                exit_pos = pos
                exit_raw = min(opens[pos], stop) if opens[pos] < stop else stop
                exit_reason = "STOP"
                stop_hit = True
                break
            if side < 0 and highs[pos] >= stop:
                exit_pos = pos
                exit_raw = max(opens[pos], stop) if opens[pos] > stop else stop
                exit_reason = "STOP"
                stop_hit = True
                break
            if side > 0 and np.isfinite(trail_low_arr[pos]):
                stop = max(stop, float(trail_low_arr[pos]))
            elif side < 0 and np.isfinite(trail_high_arr[pos]):
                stop = min(stop, float(trail_high_arr[pos]))
            if adds < int(args.max_adds):
                if side > 0 and np.isfinite(add_long_level[pos]) and closes[pos] > add_long_level[pos]:
                    pending_add = True
                elif side < 0 and np.isfinite(add_short_level[pos]) and closes[pos] < add_short_level[pos]:
                    pending_add = True
        exit_price = _fill_exit_price(float(exit_raw), side, float(args.exit_slippage_pct))
        gross = sum(size * side * (exit_price / entry - 1.0) for entry, size in zip(leg_entries, leg_sizes, strict=False))
        total_size = float(sum(leg_sizes))
        fees = total_size * (float(args.entry_fee_rate) + float(args.exit_fee_rate))
        net = gross - fees
        rows.append({
            "event_name": getattr(ev, "event_name"),
            "signal_time": signal_time,
            "entry_time": idx[entry_pos],
            "exit_time": idx[exit_pos],
            "side": side,
            "entry_price_initial": entry_price,
            "exit_price": exit_price,
            "initial_stop_level": base_stop,
            "initial_stop_pct": initial_stop_pct,
            "adds": adds,
            "max_gross_exposure": total_size,
            "gross_return_per_initial_notional": gross,
            "net_return_per_initial_notional": net,
            "net_r_initial_risk": net / initial_stop_pct if initial_stop_pct > 0 else np.nan,
            "hold_bars": int(exit_pos - entry_pos),
            "exit_reason": exit_reason,
            "stop_hit_flag": stop_hit,
        })
        progress.update(done)
    progress.close()
    return pd.DataFrame(rows)


def summarize_pyramiding(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    rows = []
    for name, part in trades.groupby("event_name", dropna=False):
        x = pd.to_numeric(part["net_return_per_initial_notional"], errors="coerce").dropna()
        r = pd.to_numeric(part["net_r_initial_risk"], errors="coerce").dropna()
        rows.append({
            "event_name": name,
            "trades": int(len(x)),
            "mean_net": float(x.mean()) if len(x) else np.nan,
            "median_net": float(x.median()) if len(x) else np.nan,
            "win_rate": float((x > 0).mean()) if len(x) else np.nan,
            "profit_factor": float(x[x > 0].sum() / (-x[x <= 0].sum())) if len(x) and float(-x[x <= 0].sum()) > 0 else np.nan,
            "mean_r": float(r.mean()) if len(r) else np.nan,
            "median_r": float(r.median()) if len(r) else np.nan,
            "avg_adds": float(pd.to_numeric(part["adds"], errors="coerce").mean()),
            "stop_hit_rate": float(part["stop_hit_flag"].astype(bool).mean()),
            "avg_hold_bars": float(pd.to_numeric(part["hold_bars"], errors="coerce").mean()),
        })
    out = pd.DataFrame(rows)
    return out.sort_values(["mean_net", "profit_factor", "trades"], ascending=[False, False, False])


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)


def run_lab(args: argparse.Namespace) -> dict[str, object]:
    horizons = _parse_number_list(args.horizons, cast=int, name="horizons")
    if int(args.candidate_horizon) not in horizons:
        horizons = sorted(set([*horizons, int(args.candidate_horizon)]))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[1/7] Loading bars through src.data_feed", flush=True)
    bars_all = _load_bars_via_data_feed(args)
    print("[2/7] Building past-only swing/spike features", flush=True)
    features_all = build_features(bars_all, args)
    eval_start = pd.Timestamp(args.start_date)
    eval_end = pd.Timestamp(args.end_date)
    bars = bars_all.loc[eval_start:eval_end].copy()
    features = features_all.loc[eval_start:eval_end].copy()
    if bars.empty or features.empty:
        raise RuntimeError("No rows after evaluation date slicing.")

    print("[3/7] Building sweep reversal events", flush=True)
    events = build_events(features, args)
    if events.empty:
        raise RuntimeError("No sweep events generated. Try lower thresholds or a longer range.")
    events = add_event_bins(events)
    print(f"       events={len(events):,} unique_event_names={events['event_name'].nunique():,}", flush=True)

    print("[4/7] Running event-study core", flush=True)
    config = EventStudyConfig(
        horizons=tuple(int(h) for h in horizons),
        mfe_mae_horizon=int(args.mfe_mae_horizon),
        cost=CostConfig(
            entry_fee_rate=float(args.entry_fee_rate),
            exit_fee_rate=float(args.exit_fee_rate),
            entry_slippage_pct=float(args.entry_slippage_pct),
            exit_slippage_pct=float(args.exit_slippage_pct),
        ),
        min_count=int(args.min_count),
        progress_every=0 if args.no_progress else int(args.progress_every),
    )
    result = run_event_study(bars, events, config)
    result_events = attach_first_touch_labels(bars, result.events, args)
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

    print("[5/7] Writing grouped event-study diagnostics", flush=True)
    return_cols = [f"next_open_ret_h{int(h)}_net" for h in horizons]
    primary_return_col = f"next_open_ret_h{int(args.candidate_horizon)}_net"
    event_name_stats = summarize_many(result.events, return_cols, group_cols=["event_name"], min_count=int(args.min_count))
    event_family_stats = summarize_many(result.events, return_cols, group_cols=["event_family"], min_count=int(args.min_count))
    event_yearly_stats = summarize_many(result.events, return_cols, group_cols=["event_name", "year"], min_count=max(10, int(args.min_count) // 3))
    bin_stats = summarize_many(result.events, [primary_return_col], group_cols=["event_family", "volume_ratio_q", "atr_pct_q", "swing_prominence_q", "swing_age_bucket", "spike_bucket"], min_count=max(10, int(args.min_count) // 3))
    contrasts = []
    for condition in ["volume_spike"]:
        try:
            contrasts.append(condition_contrast(result.events, condition_col=condition, return_col=primary_return_col, min_count=max(10, int(args.min_count)//3)))
        except KeyError:
            pass
    contrast_df = pd.concat(contrasts, ignore_index=True) if contrasts else pd.DataFrame()
    touch_stats = result.events.groupby(["event_name", "touch_result"], dropna=False).size().reset_index(name="count") if "touch_result" in result.events.columns else pd.DataFrame()
    if not touch_stats.empty:
        touch_stats["share"] = touch_stats["count"] / touch_stats.groupby("event_name")["count"].transform("sum").replace(0, np.nan)
    candidate_rank = build_candidate_rank(result.events, event_name_stats, event_yearly_stats, return_col=primary_return_col, args=args)

    event_name_stats.to_csv(out_dir / "06_event_name_stats.csv", index=False)
    event_family_stats.to_csv(out_dir / "07_event_family_stats.csv", index=False)
    event_yearly_stats.to_csv(out_dir / "09_event_name_yearly_stats.csv", index=False)
    contrast_df.to_csv(out_dir / "11_condition_contrast.csv", index=False)
    bin_stats.to_csv(out_dir / "12_feature_bin_stats.csv", index=False)
    touch_stats.to_csv(out_dir / "13_first_touch_stats.csv", index=False)
    candidate_rank.to_csv(out_dir / "14_candidate_rank.csv", index=False)
    if int(args.save_feature_sample) > 0:
        features.tail(int(args.save_feature_sample)).to_csv(out_dir / "15_feature_tail_sample.csv")

    print("[6/7] Running research pyramiding simulator", flush=True)
    pyramid_trades = pd.DataFrame()
    pyramid_summary = pd.DataFrame()
    pyramid_selection = pd.DataFrame()
    if bool(args.run_pyramiding_sim) and not candidate_rank.empty:
        pyramid_selection = select_pyramiding_event_names(candidate_rank, args)
        top_names = pyramid_selection["event_name"].dropna().astype(str).tolist() if not pyramid_selection.empty else []
        if not top_names:
            print("       skipped: no event_name passed the pyramiding selection gate", flush=True)
        sim_parts = []
        for name in top_names:
            part_events = result.events[result.events["event_name"] == name].copy()
            sim = simulate_pyramiding_group(bars, part_events, args, label=name)
            if not sim.empty:
                sim_parts.append(sim)
        pyramid_trades = pd.concat(sim_parts, ignore_index=True) if sim_parts else pd.DataFrame()
        pyramid_summary = summarize_pyramiding(pyramid_trades)
    pyramid_selection.to_csv(out_dir / "16_pyramiding_selection.csv", index=False)
    pyramid_trades.to_csv(out_dir / "17_pyramiding_trades.csv", index=False)
    pyramid_summary.to_csv(out_dir / "18_pyramiding_summary.csv", index=False)

    print("[7/7] Writing lab meta", flush=True)
    best = candidate_rank.head(20).copy() if not candidate_rank.empty else pd.DataFrame()
    best_pyr = pyramid_summary.head(20).copy() if not pyramid_summary.empty else pd.DataFrame()
    meta = {
        "symbol": args.symbol,
        "data_source": args.data_source,
        "timeframe": args.timeframe,
        "range_pct": float(args.range_pct),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "event_count": int(len(result.events)),
        "unique_event_names": int(result.events["event_name"].nunique()),
        "causal_fail_count": int(result.causal_audit["causal_fail_flag"].sum()) if not result.causal_audit.empty else 0,
        "round_trip_cost_pct": float(config.cost.round_trip_cost_pct),
        "candidate_horizon": int(args.candidate_horizon),
        "best_event_candidates": best.to_dict(orient="records"),
        "pyramiding_selected_event_names": int(len(pyramid_selection)) if not pyramid_selection.empty else 0,
        "best_pyramiding_candidates": best_pyr.to_dict(orient="records"),
        "notes": "Research-only. Event-study and pyramiding outputs are phenomena diagnostics, not validated strategies. Pyramiding simulation now only uses event names that pass candidate/sample/year gates unless --sim-include-noncandidates is set.",
    }
    _write_json(out_dir / "19_lab_meta.json", meta)

    if not best.empty:
        print("\nTop event-study rows:", flush=True)
        cols = ["event_name", "count", "mean", "median", "win_rate", "profit_factor", "candidate_flag"]
        print(best[cols].head(10).to_string(index=False), flush=True)
    if not best_pyr.empty:
        print("\nTop pyramiding rows:", flush=True)
        cols = ["event_name", "trades", "mean_net", "median_net", "win_rate", "profit_factor", "avg_adds", "stop_hit_rate"]
        print(best_pyr[cols].head(10).to_string(index=False), flush=True)
    print(f"\nReport written to: {out_dir}", flush=True)
    return meta


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_lab(args)


if __name__ == "__main__":
    main()
