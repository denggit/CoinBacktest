#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Focused causal event study for ETH low-sweep panic reversal.

This lab intentionally studies the event itself, not pyramiding / anti-martingale
position management. It is narrower than ``sweep_reversal_antimartingale_event_lab``:

- data source: OKX trade bars only, loaded via ``src.data_feed.OKXTradeBarLoader``;
- direction: LONG only after a confirmed swing-low sweep;
- default focus: ``low_fade_close_through`` where the bar closes through the
  swing low instead of reclaiming immediately;
- outputs: event-name stats, canonical de-duplicated candidate-union stats,
  session/regime splits, delay stress, cost stress, and first-touch grids.

No direct SQLite/CSV/ZIP reading is performed here. Historical data access must
remain inside ``src.data_feed`` loaders.
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

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader  # noqa: E402
from src.research_common.event_study import (  # noqa: E402
    CostConfig,
    EventStudyConfig,
    first_touch_outcome,
    fixed_threshold_labels,
    qcut_labels,
    run_event_study,
    summarize_many,
    top_winner_dependency,
)
from src.research_common.progress import ProgressReporter  # noqa: E402

SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1H", "4H", "1D"}


# ---------------------------------------------------------------------------
# Args / parsing
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Focused low-sweep reversal event study on OKX trade bars.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", choices=sorted(SUPPORTED_TIMEFRAMES), default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--out-dir", default="data/reports/research/focused_low_sweep_reversal_event_lab")

    # Swing / event definition.
    p.add_argument("--pivot-left", type=int, default=6)
    p.add_argument("--pivot-right", type=int, default=3)
    p.add_argument("--min-swing-age", type=int, default=3)
    p.add_argument("--max-swing-ages", default="12,24,48")
    p.add_argument("--min-swing-prominence-pcts", default="0.0015,0.0030")
    p.add_argument("--spike-pcts", default="0.0060,0.0080,0.0100,0.0120")
    p.add_argument("--breakout-pcts", default="0.0000,0.0005")
    p.add_argument("--variants", default="fade_close_through,reject,wick")
    p.add_argument("--wick-min-frac", type=float, default=0.45)
    p.add_argument("--close-through-buffer-pct", type=float, default=0.0)

    # Feature windows.
    p.add_argument("--volume-window", type=int, default=120)
    p.add_argument("--atr-window", type=int, default=42)
    p.add_argument("--cvd-window", type=int, default=60)
    p.add_argument("--volume-spike-threshold", type=float, default=1.50)
    p.add_argument("--delta-capitulation-quantile", type=float, default=0.20)

    # Event-study labels / gates.
    p.add_argument("--horizons", default="3,6,12,24,48,96")
    p.add_argument("--candidate-horizon", type=int, default=48)
    p.add_argument("--mfe-mae-horizon", type=int, default=96)
    p.add_argument("--min-count", type=int, default=100)
    p.add_argument("--min-positive-years", type=int, default=3)
    p.add_argument("--min-profit-factor", type=float, default=1.15)
    p.add_argument("--min-win-rate", type=float, default=0.52)
    p.add_argument("--max-top5-winner-share", type=float, default=0.45)

    # Default base cost: OKX round-trip fee convention plus conservative slippage.
    p.add_argument("--entry-fee-rate", type=float, default=0.00055)
    p.add_argument("--exit-fee-rate", type=float, default=0.00055)
    p.add_argument("--entry-slippage-pct", type=float, default=0.00020)
    p.add_argument("--exit-slippage-pct", type=float, default=0.00020)

    # Stress / TP-SL grids.
    p.add_argument("--delay-bars-list", default="1,2,3")
    p.add_argument("--cost-multipliers", default="1.0,1.5,2.0")
    p.add_argument("--touch-target-pcts", default="0.0020,0.0030,0.0040,0.0060,0.0080")
    p.add_argument("--touch-stop-pcts", default="0.0020,0.0030,0.0040,0.0060,0.0080")
    p.add_argument("--touch-horizon", type=int, default=48)

    p.add_argument("--progress-every", type=int, default=25000)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--save-feature-sample", type=int, default=5000)
    return p.parse_args(argv)


def _parse_number_list(text: str, *, cast=float, name: str = "values", allow_zero: bool = False) -> list:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        value = cast(part)
        if allow_zero:
            if value < 0:
                raise ValueError(f"{name} must not contain negative values")
        elif value <= 0:
            raise ValueError(f"{name} must contain positive values")
        out.append(value)
    if not out:
        raise ValueError(f"{name} must not be empty")
    return sorted(set(out))


def _parse_variant_list(text: str) -> list[str]:
    valid = {"fade_close_through", "reject", "wick"}
    variants = [part.strip() for part in str(text).split(",") if part.strip()]
    bad = sorted(set(variants) - valid)
    if bad:
        raise ValueError(f"Unsupported variants={bad}; supported={sorted(valid)}")
    if not variants:
        raise ValueError("variants must not be empty")
    return variants


# ---------------------------------------------------------------------------
# Data / features
# ---------------------------------------------------------------------------


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


def load_trade_bars(args: argparse.Namespace) -> pd.DataFrame:
    """Load bars only via src.data_feed; no direct DB/file access in research."""
    print(
        f"[load] OKXTradeBarLoader {args.symbol} {args.timeframe} "
        f"{args.warmup_start_date}->{args.end_date}",
        flush=True,
    )
    df = OKXTradeBarLoader(symbol=args.symbol, timeframe=args.timeframe).fetch_data_by_date_range(
        args.warmup_start_date,
        args.end_date,
    )
    if df.empty:
        raise RuntimeError(f"No trade_bar data loaded for {args.symbol} {args.timeframe}")
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


def confirmed_swing_lows(df: pd.DataFrame, *, left: int, right: int) -> pd.DataFrame:
    """Past-only confirmed swing low level, age, and prominence.

    A pivot low centered at bar j is only confirmed after right future bars have
    closed. We then shift one extra bar before using it, so the signal bar cannot
    use a pivot that is only confirmed by itself.
    """
    low = pd.to_numeric(df["low"], errors="coerce")
    high = pd.to_numeric(df["high"], errors="coerce")
    pos = pd.Series(np.arange(len(df), dtype=float), index=df.index)

    left_low = low.shift(1).rolling(left, min_periods=left).min()
    right_low = low.iloc[::-1].shift(1).rolling(right, min_periods=right).min().iloc[::-1]
    pivot_low = (low < left_low) & (low <= right_low)

    window = left + right + 1
    local_high = high.rolling(window, center=True, min_periods=window).max()
    low_prom = _safe_divide(local_high, low) - 1.0

    swing_low = low.where(pivot_low).shift(right).shift(1).ffill()
    swing_low_pos = pos.where(pivot_low).shift(right).shift(1).ffill()
    swing_low_prom = low_prom.where(pivot_low).shift(right).shift(1).ffill()

    return pd.DataFrame(
        {
            "swing_low": swing_low,
            "swing_low_pos": swing_low_pos,
            "swing_low_age": pos - swing_low_pos,
            "swing_low_prominence_pct": swing_low_prom,
        },
        index=df.index,
    )


def build_features(bars: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    df = bars.copy().sort_index()
    df["prev_close"] = df["close"].shift(1)
    df["ret_close_to_close"] = df["close"].pct_change()
    df["bar_body_ret"] = df["close"] / df["open"] - 1.0
    df["down_spike_pct"] = df["prev_close"] / df["low"] - 1.0
    df["tr"] = _true_range(df)
    df["atr"] = df["tr"].rolling(int(args.atr_window), min_periods=int(args.atr_window)).mean()
    df["atr_pct"] = _safe_divide(df["atr"], df["close"])

    vol_min_periods = min(int(args.volume_window), max(10, int(args.volume_window) // 3))
    vol_base = df["volume"].shift(1).rolling(int(args.volume_window), min_periods=vol_min_periods).median()
    df["volume_ratio"] = _safe_divide(df["volume"], vol_base)
    df["volume_spike"] = df["volume_ratio"] >= float(args.volume_spike_threshold)

    if "trades_count" in df.columns:
        trade_min_periods = min(int(args.volume_window), max(10, int(args.volume_window) // 3))
        trade_base = pd.to_numeric(df["trades_count"], errors="coerce").shift(1).rolling(
            int(args.volume_window), min_periods=trade_min_periods
        ).median()
        df["trades_count_ratio"] = _safe_divide(pd.to_numeric(df["trades_count"], errors="coerce"), trade_base)
    else:
        df["trades_count_ratio"] = np.nan

    if "delta_notional" in df.columns:
        df["delta_notional"] = pd.to_numeric(df["delta_notional"], errors="coerce")
        notional_base = (pd.to_numeric(df.get("buy_notional", np.nan), errors="coerce") + pd.to_numeric(df.get("sell_notional", np.nan), errors="coerce")).replace(0.0, np.nan)
        df["delta_notional_ratio"] = df["delta_notional"] / notional_base
        cvd_min_periods = min(int(args.cvd_window), max(10, int(args.cvd_window) // 3))
        roll = df["delta_notional"].shift(1).rolling(int(args.cvd_window), min_periods=cvd_min_periods)
        df["delta_notional_z"] = (df["delta_notional"] - roll.mean()) / roll.std(ddof=0).replace(0.0, np.nan)
    else:
        df["delta_notional"] = np.nan
        df["delta_notional_ratio"] = np.nan
        df["delta_notional_z"] = np.nan

    if "cvd_notional" in df.columns:
        cvd = pd.to_numeric(df["cvd_notional"], errors="coerce")
        df["cvd_notional_change"] = cvd.diff(int(args.cvd_window))
    else:
        df["cvd_notional_change"] = np.nan

    if "taker_buy_ratio" in df.columns:
        df["taker_buy_ratio"] = pd.to_numeric(df["taker_buy_ratio"], errors="coerce")
    else:
        df["taker_buy_ratio"] = np.nan

    if "large_delta_notional" in df.columns:
        df["large_delta_notional"] = pd.to_numeric(df["large_delta_notional"], errors="coerce")
    else:
        df["large_delta_notional"] = np.nan

    bar_range = (df["high"] - df["low"]).replace(0.0, np.nan)
    df["lower_wick_frac"] = _safe_divide(df[["open", "close"]].min(axis=1) - df["low"], bar_range)
    df["close_pos_in_bar"] = _safe_divide(df["close"] - df["low"], bar_range)

    swings = confirmed_swing_lows(df, left=int(args.pivot_left), right=int(args.pivot_right))
    out = pd.concat([df, swings], axis=1)
    out["session_hour"] = out.index.hour
    out["session_bucket"] = pd.cut(
        out["session_hour"],
        bins=[-1, 7, 15, 23],
        labels=["S0_00_07", "S1_08_15", "S2_16_23"],
    ).astype("object").fillna("NA")
    out["weekday"] = out.index.dayofweek
    return out


# ---------------------------------------------------------------------------
# Events / bins / candidate ranking
# ---------------------------------------------------------------------------


def _event_row(ts: pd.Timestamp, event_name: str, family: str, variant: str, row: pd.Series, extra: dict[str, object]) -> dict[str, object]:
    return {
        "signal_time": ts,
        "side": 1,
        "event_name": event_name,
        "event_family": family,
        "variant": variant,
        "swing_level": float(row["swing_low"]),
        "sweep_extreme": float(row["low"]),
        "structural_stop_level": float(min(float(row["low"]), float(row["swing_low"]))),
        "swing_age": float(row["swing_low_age"]),
        "swing_prominence_pct": float(row["swing_low_prominence_pct"]),
        "down_spike_pct": float(row.get("down_spike_pct", np.nan)),
        "volume_ratio": float(row.get("volume_ratio", np.nan)),
        "trades_count_ratio": float(row.get("trades_count_ratio", np.nan)),
        "volume_spike": bool(row.get("volume_spike", False)),
        "atr_pct": float(row.get("atr_pct", np.nan)),
        "delta_notional": float(row.get("delta_notional", np.nan)),
        "delta_notional_ratio": float(row.get("delta_notional_ratio", np.nan)),
        "delta_notional_z": float(row.get("delta_notional_z", np.nan)),
        "cvd_notional_change": float(row.get("cvd_notional_change", np.nan)),
        "taker_buy_ratio": float(row.get("taker_buy_ratio", np.nan)),
        "large_delta_notional": float(row.get("large_delta_notional", np.nan)),
        "lower_wick_frac": float(row.get("lower_wick_frac", np.nan)),
        "close_pos_in_bar": float(row.get("close_pos_in_bar", np.nan)),
        "session_hour": int(row.get("session_hour", -1)) if pd.notna(row.get("session_hour", np.nan)) else -1,
        "session_bucket": str(row.get("session_bucket", "NA")),
        "weekday": int(row.get("weekday", -1)) if pd.notna(row.get("weekday", np.nan)) else -1,
        "close": float(row["close"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        **extra,
    }


def build_low_sweep_events(features: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    spike_pcts = _parse_number_list(args.spike_pcts, cast=float, name="spike_pcts")
    breakout_pcts = _parse_number_list(args.breakout_pcts, cast=float, name="breakout_pcts", allow_zero=True)
    max_ages = _parse_number_list(args.max_swing_ages, cast=int, name="max_swing_ages")
    min_proms = _parse_number_list(args.min_swing_prominence_pcts, cast=float, name="min_swing_prominence_pcts")
    variants = _parse_variant_list(args.variants)

    rows: list[dict[str, object]] = []
    total = len(spike_pcts) * len(breakout_pcts) * len(max_ages) * len(min_proms) * len(variants)
    progress = ProgressReporter(
        label="[events] focused low sweep",
        total=total,
        every=1,
        enabled=not bool(args.no_progress),
    )
    done = 0
    for spike_pct in spike_pcts:
        for breakout_pct in breakout_pcts:
            for max_age in max_ages:
                for min_prom in min_proms:
                    base = (
                        features["swing_low"].notna()
                        & features["swing_low_age"].between(int(args.min_swing_age), int(max_age), inclusive="both")
                        & (features["swing_low_prominence_pct"] >= float(min_prom))
                        & (features["down_spike_pct"] >= float(spike_pct))
                        & (features["low"] <= features["swing_low"] * (1.0 - float(breakout_pct)))
                    )
                    variant_masks = {
                        "fade_close_through": base & (features["close"] <= features["swing_low"] * (1.0 - float(args.close_through_buffer_pct))),
                        "reject": base & (features["close"] > features["swing_low"]),
                        "wick": base & (features["lower_wick_frac"] >= float(args.wick_min_frac)),
                    }
                    for variant in variants:
                        mask = variant_masks[variant]
                        event_family = f"low_sweep_{variant}"
                        suffix = f"sp{int(spike_pct * 10000):04d}_br{int(breakout_pct * 10000):04d}_age{max_age}_prom{int(min_prom * 10000):04d}"
                        event_name = f"low_{variant}_{suffix}"
                        for ts, row in features.loc[mask].iterrows():
                            rows.append(
                                _event_row(
                                    ts,
                                    event_name,
                                    event_family,
                                    variant,
                                    row,
                                    {
                                        "spike_threshold_pct": float(spike_pct),
                                        "breakout_threshold_pct": float(breakout_pct),
                                        "max_swing_age": int(max_age),
                                        "min_swing_prominence_pct": float(min_prom),
                                    },
                                )
                            )
                        done += 1
                        progress.update(done)
    progress.close()
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events = events.sort_values(["signal_time", "event_name"]).reset_index(drop=True)
    return events


def add_event_bins(events: pd.DataFrame, *, delta_capitulation_quantile: float = 0.20) -> pd.DataFrame:
    out = events.copy()
    out["volume_ratio_q"] = qcut_labels(out["volume_ratio"], q=4, prefix="VOL_Q")
    out["trades_count_ratio_q"] = qcut_labels(out["trades_count_ratio"], q=4, prefix="TRADES_Q")
    out["atr_pct_q"] = qcut_labels(out["atr_pct"], q=4, prefix="ATR_Q")
    out["swing_prominence_q"] = qcut_labels(out["swing_prominence_pct"], q=4, prefix="PROM_Q")
    out["delta_notional_q"] = qcut_labels(out["delta_notional"], q=4, prefix="DELTA_Q")
    out["taker_buy_ratio_q"] = qcut_labels(out["taker_buy_ratio"], q=4, prefix="BUY_Q")
    out["spike_bucket"] = fixed_threshold_labels(
        out["down_spike_pct"],
        thresholds=[0.006, 0.008, 0.010, 0.012, 0.016],
        labels=["SPIKE_LT_60", "SPIKE_60_80", "SPIKE_80_100", "SPIKE_100_120", "SPIKE_120_160", "SPIKE_GT_160"],
    )
    out["age_bucket"] = fixed_threshold_labels(
        out["swing_age"],
        thresholds=[6, 12, 24, 48],
        labels=["AGE_0_6", "AGE_7_12", "AGE_13_24", "AGE_25_48", "AGE_GT_48"],
    )
    out["close_pos_bucket"] = fixed_threshold_labels(
        out["close_pos_in_bar"],
        thresholds=[0.2, 0.4, 0.6, 0.8],
        labels=["CLOSE_0_20", "CLOSE_20_40", "CLOSE_40_60", "CLOSE_60_80", "CLOSE_80_100"],
    )
    delta = pd.to_numeric(out["delta_notional"], errors="coerce")
    q = float(delta.quantile(float(delta_capitulation_quantile))) if delta.notna().any() else np.nan
    out["delta_capitulation"] = delta <= q if np.isfinite(q) else False
    out["strong_volume_spike"] = pd.to_numeric(out["volume_ratio"], errors="coerce") >= 2.0
    out["deep_close"] = pd.to_numeric(out["close_pos_in_bar"], errors="coerce") <= 0.30
    out["large_lower_wick"] = pd.to_numeric(out["lower_wick_frac"], errors="coerce") >= 0.45
    return out


def build_canonical_events(events: pd.DataFrame) -> pd.DataFrame:
    """One row per signal_time+side using the most specific overlapping event."""
    if events.empty:
        return events.copy()
    out = events.copy()
    variant_rank = {"fade_close_through": 0, "wick": 1, "reject": 2}
    out["variant_rank"] = out["variant"].map(variant_rank).fillna(9).astype(int)
    out["specificity_score"] = (
        pd.to_numeric(out["spike_threshold_pct"], errors="coerce").fillna(0) * 10_000
        + pd.to_numeric(out["min_swing_prominence_pct"], errors="coerce").fillna(0) * 10_000
        + pd.to_numeric(out["breakout_threshold_pct"], errors="coerce").fillna(0) * 10_000
        - pd.to_numeric(out["max_swing_age"], errors="coerce").fillna(999) * 0.01
        - out["variant_rank"] * 0.001
    )
    out = out.sort_values(["signal_time", "side", "specificity_score"], ascending=[True, True, False])
    return out.drop_duplicates(["signal_time", "side"], keep="first").reset_index(drop=True)


def _metric_float(value: object) -> float:
    if value == "inf":
        return float("inf")
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return float("nan")


def build_candidate_rank(events: pd.DataFrame, yearly: pd.DataFrame, *, return_col: str, args: argparse.Namespace) -> pd.DataFrame:
    if events.empty or return_col not in events.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for event_name, part in events.groupby("event_name", dropna=False):
        x = pd.to_numeric(part[return_col], errors="coerce").dropna()
        if x.empty:
            continue
        yr = yearly[(yearly.get("event_name") == event_name) & (yearly.get("metric") == return_col)].copy() if not yearly.empty else pd.DataFrame()
        yr["mean_float"] = yr["mean"].map(_metric_float) if "mean" in yr.columns else np.nan
        tested_years = int((yr.get("count", pd.Series(dtype=float)).fillna(0).astype(float) >= int(args.min_count)).sum()) if not yr.empty else 0
        positive_years = int(((yr.get("count", pd.Series(dtype=float)).fillna(0).astype(float) >= int(args.min_count)) & (yr["mean_float"] > 0)).sum()) if not yr.empty else 0
        pf = _profit_factor_raw(x)
        top5 = top_winner_dependency(x, top_n=5)
        count = int(len(x))
        mean = float(x.mean())
        median = float(x.median())
        win_rate = float((x > 0).mean())
        candidate = (
            count >= int(args.min_count)
            and mean > 0
            and median > 0
            and win_rate >= float(args.min_win_rate)
            and pf >= float(args.min_profit_factor)
            and positive_years >= int(args.min_positive_years)
            and (not np.isfinite(top5) or top5 <= float(args.max_top5_winner_share))
        )
        sample = part.iloc[0]
        rank_score = mean * 10_000 + median * 4_000 + (pf if np.isfinite(pf) else 3.0) * 10 + win_rate * 10 + positive_years
        rows.append(
            {
                "event_name": event_name,
                "event_family": sample.get("event_family", ""),
                "variant": sample.get("variant", ""),
                "count": count,
                "unique_signal_time": int(part["signal_time"].nunique()) if "signal_time" in part.columns else count,
                "mean": mean,
                "median": median,
                "win_rate": win_rate,
                "profit_factor": pf,
                "top5_winner_share": top5,
                "tested_years": tested_years,
                "positive_years": positive_years,
                "candidate_flag": bool(candidate),
                "rank_score": float(rank_score),
                "spike_threshold_pct": sample.get("spike_threshold_pct", np.nan),
                "breakout_threshold_pct": sample.get("breakout_threshold_pct", np.nan),
                "max_swing_age": sample.get("max_swing_age", np.nan),
                "min_swing_prominence_pct": sample.get("min_swing_prominence_pct", np.nan),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["candidate_flag", "rank_score", "count"], ascending=[False, False, False]).reset_index(drop=True)


def _profit_factor_raw(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    gp = float(vals[vals > 0].sum())
    gl = float(-vals[vals <= 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else float("nan")
    return gp / gl


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _side_series_for_events(bars: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    side = pd.Series(0, index=bars.index, dtype="int64")
    if events.empty:
        return side
    idx = bars.index.get_indexer(pd.DatetimeIndex(pd.to_datetime(events["signal_time"])))
    valid = idx >= 0
    if valid.any():
        side.iloc[idx[valid]] = 1
    return side


def attach_first_touch_grid(bars: pd.DataFrame, events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    target_pcts = _parse_number_list(args.touch_target_pcts, cast=float, name="touch_target_pcts")
    stop_pcts = _parse_number_list(args.touch_stop_pcts, cast=float, name="touch_stop_pcts")
    rows: list[dict[str, object]] = []
    side = _side_series_for_events(bars, events)
    event_times = pd.DatetimeIndex(pd.to_datetime(events["signal_time"]))
    for target in target_pcts:
        for stop in stop_pcts:
            touch = first_touch_outcome(
                bars,
                side,
                target_pct=float(target),
                stop_pct=float(stop),
                horizon=int(args.touch_horizon),
                entry_delay_bars=1,
                same_bar_policy="conservative",
            )
            selected = touch.loc[event_times]
            counts = selected["touch_result"].value_counts(dropna=False)
            total = int(len(selected))
            rows.append(
                {
                    "target_pct": float(target),
                    "stop_pct": float(stop),
                    "horizon": int(args.touch_horizon),
                    "count": total,
                    "target_share": float(counts.get("TARGET", 0) / total) if total else np.nan,
                    "stop_share": float(counts.get("STOP", 0) / total) if total else np.nan,
                    "timeout_share": float(counts.get("TIMEOUT", 0) / total) if total else np.nan,
                    "both_unknown_share": float(counts.get("BOTH_UNKNOWN", 0) / total) if total else np.nan,
                    "same_bar_both_hit_share": float(selected["same_bar_both_hit_flag"].mean()) if total else np.nan,
                    "avg_touch_bars": float(pd.to_numeric(selected["touch_bars"], errors="coerce").mean()) if total else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["target_share", "stop_share"], ascending=[False, True]).reset_index(drop=True)


def summarize_candidate_union(events: pd.DataFrame, candidate_rank: pd.DataFrame, *, return_cols: Iterable[str], min_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if candidate_rank.empty:
        return pd.DataFrame(), pd.DataFrame()
    names = set(candidate_rank.loc[candidate_rank["candidate_flag"].astype(bool), "event_name"].astype(str))
    if not names:
        return pd.DataFrame(), pd.DataFrame()
    union_raw = events[events["event_name"].astype(str).isin(names)].copy()
    if union_raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    union_canonical = build_canonical_events(union_raw)
    rows = []
    for scope, frame in (("candidate_raw", union_raw), ("candidate_canonical", union_canonical)):
        summary = summarize_many(frame, return_cols, min_count=min_count)
        if not summary.empty:
            summary.insert(0, "scope", scope)
            rows.append(summary)
    return union_canonical, pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def run_event_study_for(events: pd.DataFrame, bars: pd.DataFrame, args: argparse.Namespace, *, entry_delay_bars: int, cost_mult: float, progress_label: str) -> pd.DataFrame:
    base_cost = CostConfig(
        entry_fee_rate=float(args.entry_fee_rate) * float(cost_mult),
        exit_fee_rate=float(args.exit_fee_rate) * float(cost_mult),
        entry_slippage_pct=float(args.entry_slippage_pct) * float(cost_mult),
        exit_slippage_pct=float(args.exit_slippage_pct) * float(cost_mult),
    )
    cfg = EventStudyConfig(
        horizons=tuple(_parse_number_list(args.horizons, cast=int, name="horizons")),
        mfe_mae_horizon=int(args.mfe_mae_horizon),
        entry_delay_bars=int(entry_delay_bars),
        cost=base_cost,
        min_count=int(args.min_count),
        progress_every=0,  # stress loops should stay quiet; the main run shows progress.
    )
    result = run_event_study(bars, events, cfg)
    return_cols = [f"next_open_ret_h{int(h)}_net" for h in cfg.horizons]
    summary = summarize_many(result.events, return_cols, min_count=int(args.min_count))
    summary.insert(0, "scope", progress_label)
    summary.insert(1, "entry_delay_bars", int(entry_delay_bars))
    summary.insert(2, "cost_mult", float(cost_mult))
    summary.insert(3, "round_trip_cost_pct", float(base_cost.round_trip_cost_pct))
    return summary


# ---------------------------------------------------------------------------
# Main lab
# ---------------------------------------------------------------------------


def run_lab(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bars_all = load_trade_bars(args)
    features_all = build_features(bars_all, args)
    start_ts = pd.Timestamp(args.start_date)
    end_ts = pd.Timestamp(args.end_date)
    bars = bars_all.loc[(bars_all.index >= start_ts) & (bars_all.index <= end_ts)].copy()
    features = features_all.loc[(features_all.index >= start_ts) & (features_all.index <= end_ts)].copy()
    if bars.empty or features.empty:
        raise RuntimeError(f"No bars/features in research window {start_ts}->{end_ts}")

    print(f"[features] research rows={len(features):,} range={features.index[0]} -> {features.index[-1]}", flush=True)
    events_raw = build_low_sweep_events(features, args)
    if events_raw.empty:
        raise RuntimeError("No low-sweep events generated. Try lower spike/pivot/prominence thresholds.")
    events_raw = add_event_bins(events_raw, delta_capitulation_quantile=float(args.delta_capitulation_quantile))
    events_canonical = build_canonical_events(events_raw)

    print(
        f"[events] raw={len(events_raw):,} canonical={len(events_canonical):,} "
        f"unique_signal_time={events_raw['signal_time'].nunique():,}",
        flush=True,
    )

    base_cost = CostConfig(
        entry_fee_rate=float(args.entry_fee_rate),
        exit_fee_rate=float(args.exit_fee_rate),
        entry_slippage_pct=float(args.entry_slippage_pct),
        exit_slippage_pct=float(args.exit_slippage_pct),
    )
    horizons = tuple(_parse_number_list(args.horizons, cast=int, name="horizons"))
    cfg = EventStudyConfig(
        horizons=horizons,
        mfe_mae_horizon=int(args.mfe_mae_horizon),
        entry_delay_bars=1,
        cost=base_cost,
        min_count=int(args.min_count),
        progress_every=0 if bool(args.no_progress) else int(args.progress_every),
    )

    print("[study] running raw event study", flush=True)
    raw_result = run_event_study(bars, events_raw, cfg)
    print("[study] running canonical event study", flush=True)
    canonical_result = run_event_study(bars, events_canonical, cfg)

    # Core stats.
    return_cols = [f"next_open_ret_h{int(h)}_net" for h in horizons]
    candidate_return_col = f"next_open_ret_h{int(args.candidate_horizon)}_net"
    if candidate_return_col not in raw_result.events.columns:
        raise RuntimeError(f"candidate horizon {args.candidate_horizon} is not in horizons={horizons}")

    event_name_stats = summarize_many(raw_result.events, return_cols, group_cols=["event_name"], min_count=int(args.min_count))
    event_family_stats = summarize_many(raw_result.events, return_cols, group_cols=["event_family"], min_count=int(args.min_count))
    variant_stats = summarize_many(raw_result.events, return_cols, group_cols=["variant"], min_count=int(args.min_count))
    yearly_stats = summarize_many(raw_result.events, return_cols, group_cols=["event_name", "year"], min_count=int(args.min_count))
    candidate_rank = build_candidate_rank(raw_result.events, yearly_stats, return_col=candidate_return_col, args=args)
    candidate_union_events, candidate_union_summary = summarize_candidate_union(
        raw_result.events,
        candidate_rank,
        return_cols=return_cols,
        min_count=int(args.min_count),
    )

    canonical_summary = summarize_many(canonical_result.events, return_cols, min_count=int(args.min_count))
    canonical_yearly = summarize_many(canonical_result.events, return_cols, group_cols=["year"], min_count=int(args.min_count))
    canonical_session = summarize_many(canonical_result.events, [candidate_return_col], group_cols=["session_bucket"], min_count=max(20, int(args.min_count) // 2))

    # Feature / condition splits on candidate union if available, otherwise canonical all events.
    split_base = candidate_union_events if not candidate_union_events.empty else canonical_result.events
    split_scope = "candidate_union" if not candidate_union_events.empty else "canonical_all"
    print(f"[splits] base={split_scope} rows={len(split_base):,}", flush=True)
    split_return_col = candidate_return_col
    grouped_stats: list[pd.DataFrame] = []
    for group_cols in (
        ["session_bucket"],
        ["variant"],
        ["spike_bucket"],
        ["atr_pct_q"],
        ["volume_ratio_q"],
        ["trades_count_ratio_q"],
        ["delta_notional_q"],
        ["taker_buy_ratio_q"],
        ["close_pos_bucket"],
        ["age_bucket"],
        ["session_bucket", "atr_pct_q"],
        ["session_bucket", "volume_ratio_q"],
        ["delta_notional_q", "taker_buy_ratio_q"],
    ):
        cols = [c for c in group_cols if c in split_base.columns]
        if len(cols) != len(group_cols):
            continue
        s = summarize_many(split_base, [split_return_col], group_cols=cols, min_count=max(20, int(args.min_count) // 2))
        if not s.empty:
            s.insert(0, "scope", split_scope)
            s.insert(1, "grouping", "+".join(cols))
            grouped_stats.append(s)
    feature_group_stats = pd.concat(grouped_stats, ignore_index=True) if grouped_stats else pd.DataFrame()

    condition_rows: list[pd.DataFrame] = []
    for cond in ["volume_spike", "strong_volume_spike", "delta_capitulation", "deep_close", "large_lower_wick"]:
        if cond not in split_base.columns:
            continue
        for label_value, mask in (("true", split_base[cond].astype(bool)), ("false", ~split_base[cond].astype(bool))):
            part = split_base.loc[mask]
            s = summarize_many(part, [split_return_col], min_count=max(20, int(args.min_count) // 2))
            if not s.empty:
                s.insert(0, "scope", split_scope)
                s.insert(1, "condition", cond)
                s.insert(2, "condition_value", label_value)
                condition_rows.append(s)
    condition_stats = pd.concat(condition_rows, ignore_index=True) if condition_rows else pd.DataFrame()

    first_touch_grid = attach_first_touch_grid(bars, split_base, args)

    # Delay / cost stress only on candidate union when available, else canonical all.
    stress_events = candidate_union_events if not candidate_union_events.empty else events_canonical
    delay_rows: list[pd.DataFrame] = []
    for delay in _parse_number_list(args.delay_bars_list, cast=int, name="delay_bars_list"):
        delay_rows.append(run_event_study_for(stress_events, bars, args, entry_delay_bars=int(delay), cost_mult=1.0, progress_label=split_scope))
    delay_stress = pd.concat(delay_rows, ignore_index=True) if delay_rows else pd.DataFrame()

    cost_rows: list[pd.DataFrame] = []
    for mult in _parse_number_list(args.cost_multipliers, cast=float, name="cost_multipliers"):
        cost_rows.append(run_event_study_for(stress_events, bars, args, entry_delay_bars=1, cost_mult=float(mult), progress_label=split_scope))
    cost_stress = pd.concat(cost_rows, ignore_index=True) if cost_rows else pd.DataFrame()

    # Files.
    _safe_to_csv(events_raw, out_dir / "01_events_raw.csv")
    _safe_to_csv(events_canonical, out_dir / "02_events_canonical.csv")
    _safe_to_csv(raw_result.overview, out_dir / "03_raw_overview.csv")
    _safe_to_csv(canonical_summary, out_dir / "04_canonical_overview.csv")
    _safe_to_csv(event_name_stats, out_dir / "05_event_name_stats.csv")
    _safe_to_csv(event_family_stats, out_dir / "06_event_family_stats.csv")
    _safe_to_csv(variant_stats, out_dir / "07_variant_stats.csv")
    _safe_to_csv(yearly_stats, out_dir / "08_event_name_yearly_stats.csv")
    _safe_to_csv(raw_result.causal_audit, out_dir / "09_causal_audit_raw.csv")
    _safe_to_csv(canonical_result.causal_audit, out_dir / "10_causal_audit_canonical.csv")
    _safe_to_csv(candidate_rank, out_dir / "11_candidate_rank.csv")
    _safe_to_csv(candidate_union_events, out_dir / "12_candidate_union_events_canonical.csv")
    _safe_to_csv(candidate_union_summary, out_dir / "13_candidate_union_summary.csv")
    _safe_to_csv(canonical_yearly, out_dir / "14_canonical_yearly.csv")
    _safe_to_csv(canonical_session, out_dir / "15_canonical_session.csv")
    _safe_to_csv(feature_group_stats, out_dir / "16_feature_group_stats.csv")
    _safe_to_csv(condition_stats, out_dir / "17_condition_stats.csv")
    _safe_to_csv(first_touch_grid, out_dir / "18_first_touch_grid.csv")
    _safe_to_csv(delay_stress, out_dir / "19_delay_stress.csv")
    _safe_to_csv(cost_stress, out_dir / "20_cost_stress.csv")

    if int(args.save_feature_sample) > 0:
        _safe_to_csv(features.tail(int(args.save_feature_sample)).reset_index().rename(columns={"index": "timestamp"}), out_dir / "21_feature_tail_sample.csv")

    meta = {
        "script": "focused_low_sweep_reversal_event_lab.py",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "rows": int(len(bars)),
        "raw_events": int(len(events_raw)),
        "canonical_events": int(len(events_canonical)),
        "candidate_event_names": int(candidate_rank["candidate_flag"].sum()) if not candidate_rank.empty else 0,
        "candidate_union_canonical_events": int(len(candidate_union_events)),
        "candidate_horizon": int(args.candidate_horizon),
        "horizons": [int(h) for h in horizons],
        "round_trip_cost_pct": float(base_cost.round_trip_cost_pct),
        "raw_causal_fail_count": int(raw_result.meta.get("causal_fail_count", 0)),
        "canonical_causal_fail_count": int(canonical_result.meta.get("causal_fail_count", 0)),
        "split_scope": split_scope,
        "params": vars(args),
    }
    _write_json(out_dir / "22_lab_meta.json", meta)
    print(f"[done] wrote reports -> {out_dir}", flush=True)
    return meta


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_lab(args)


if __name__ == "__main__":
    main()
