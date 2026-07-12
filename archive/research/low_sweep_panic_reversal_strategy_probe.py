#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Filter-matrix strategy probe for ETH low-sweep panic reversal.

This lab is intentionally downstream of ``focused_low_sweep_reversal_event_lab``.
It keeps the event definition narrow, then tries many *environment/order-flow*
filters on top of canonical low-sweep events:

- session / ATR / spike / swing structure filters;
- trade-bar order-flow filters: taker buy ratio, buy/sell pressure, delta;
- CVD filters: fixed thresholds, strict historical rolling quantiles, dump-then-turn patterns;

Important leakage rule:
- No in-sample/global quantile labels are allowed in filter ranking.
- Rolling quantile filters use ``feature.shift(1).rolling(...).quantile(...)`` so
  the current bar and future bars never set their own thresholds.
- fixed time exits and first-touch TP/SL probes;
- delay and cost stress on the best filter combinations.

No direct SQLite/CSV/ZIP reading is performed here. Data access stays inside
``src.data_feed.OKXTradeBarLoader``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from research.focused_low_sweep_reversal_event_lab import (  # noqa: E402
    _parse_number_list,
    _safe_divide,
    add_event_bins,
    build_canonical_events,
    build_features as build_base_features,
    build_low_sweep_events,
    load_trade_bars,
)
from src.research_common.event_study import CostConfig, EventStudyConfig, first_touch_outcome, fixed_threshold_labels, run_event_study, summarize_many, top_winner_dependency  # noqa: E402
from src.research_common.progress import ProgressReporter  # noqa: E402

SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m", "30m", "1H", "4H", "1D"}


@dataclass(frozen=True)
class FilterSpec:
    name: str
    family: str
    description: str
    mask_builder: Callable[[pd.DataFrame], pd.Series]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Low-sweep panic reversal filter-matrix strategy probe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="ETH-USDT-SWAP")
    p.add_argument("--timeframe", choices=sorted(SUPPORTED_TIMEFRAMES), default="1m")
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default="2026-06-30")
    p.add_argument("--warmup-start-date", default="2022-01-01")
    p.add_argument("--out-dir", default="data/reports/research/low_sweep_panic_reversal_strategy_probe")

    # Same event definition inputs as the focused event lab.
    p.add_argument("--pivot-left", type=int, default=6)
    p.add_argument("--pivot-right", type=int, default=3)
    p.add_argument("--min-swing-age", type=int, default=3)
    p.add_argument("--max-swing-ages", default="12,24,48")
    p.add_argument("--min-swing-prominence-pcts", default="0.0015,0.0030")
    p.add_argument("--spike-pcts", default="0.0060,0.0080,0.0100,0.0120")
    p.add_argument("--breakout-pcts", default="0.0000,0.0005")
    p.add_argument("--variants", default="fade_close_through")
    p.add_argument("--wick-min-frac", type=float, default=0.45)
    p.add_argument("--close-through-buffer-pct", type=float, default=0.0)

    # Feature windows.
    p.add_argument("--volume-window", type=int, default=120)
    p.add_argument("--atr-window", type=int, default=42)
    p.add_argument("--cvd-window", type=int, default=60)
    p.add_argument("--cvd-windows", default="5,15,30,60,120")
    p.add_argument("--volume-spike-threshold", type=float, default=1.50)
    p.add_argument("--delta-capitulation-quantile", type=float, default=0.20, help="Deprecated research-only setting; not used by causal filters.")

    # Leakage-safe filter thresholds. Fixed thresholds are directly tradable.
    # Rolling quantiles are historical only: shift(1).rolling(...).quantile(q).
    p.add_argument("--buy-ratio-thresholds", default="0.60,0.65,0.70")
    p.add_argument("--buy-pressure-thresholds", default="0.60,0.65,0.70")
    p.add_argument("--delta-pressure-thresholds", default="0.00,0.10,0.20")
    p.add_argument("--cvd-pressure-thresholds", default="0.00,0.05,0.10")
    p.add_argument("--rolling-quantile-days", default="30,90")
    p.add_argument("--rolling-quantiles", default="0.75,0.80")

    # Event study / strategy probe.
    p.add_argument("--horizons", default="12,24,48,96")
    p.add_argument("--candidate-horizon", type=int, default=48)
    p.add_argument("--mfe-mae-horizon", type=int, default=96)
    p.add_argument("--min-count", type=int, default=80)
    p.add_argument("--min-positive-years", type=int, default=3)
    p.add_argument("--min-profit-factor", type=float, default=1.10)
    p.add_argument("--min-win-rate", type=float, default=0.52)
    p.add_argument("--max-top5-winner-share", type=float, default=0.45)

    # Cost convention: default 0.11% fee + 0.04% slippage round trip.
    p.add_argument("--entry-fee-rate", type=float, default=0.00055)
    p.add_argument("--exit-fee-rate", type=float, default=0.00055)
    p.add_argument("--entry-slippage-pct", type=float, default=0.00020)
    p.add_argument("--exit-slippage-pct", type=float, default=0.00020)

    # Filter matrix / stress.
    p.add_argument("--max-filter-depth", type=int, default=3)
    p.add_argument("--max-family-repeat", type=int, default=1, help="Max filters from the same family in one combo; 1 avoids contradictory overfit combos.")
    p.add_argument("--top-filter-count", type=int, default=50)
    p.add_argument("--delay-bars-list", default="1,2,3")
    p.add_argument("--cost-multipliers", default="1.0,1.5,2.0")
    p.add_argument("--touch-target-pcts", default="0.0030,0.0040,0.0060,0.0080,0.0100")
    p.add_argument("--touch-stop-pcts", default="0.0040,0.0060,0.0080,0.0100,0.0120")
    p.add_argument("--touch-horizon", type=int, default=48)
    p.add_argument("--progress-every", type=int, default=25000)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--save-event-sample", type=int, default=5000)
    return p.parse_args(argv)


def _round_trip_cost(args: argparse.Namespace, mult: float = 1.0) -> float:
    return float(args.entry_fee_rate + args.exit_fee_rate + args.entry_slippage_pct + args.exit_slippage_pct) * float(mult)


def _to_float_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce")



def _timeframe_to_minutes(timeframe: str) -> int:
    tf = str(timeframe).strip()
    if tf.endswith("m"):
        return int(tf[:-1])
    if tf.endswith("H"):
        return int(tf[:-1]) * 60
    if tf.endswith("D"):
        return int(tf[:-1]) * 1440
    raise ValueError(f"Unsupported timeframe for rolling-day conversion: {timeframe}")


def _rolling_window_bars(args: argparse.Namespace, days: float) -> int:
    minutes = _timeframe_to_minutes(str(args.timeframe))
    return max(1, int(round(float(days) * 1440.0 / minutes)))


def _tag_number(value: float, *, scale: int = 100) -> str:
    """Stable filename/filter-name tag for decimals: 0.75 -> 75, 0.008 -> 0080."""
    return f"{int(round(float(value) * scale)):0{4 if scale == 10000 else 2}d}"


def _add_fixed_threshold_flags(out: pd.DataFrame, args: argparse.Namespace) -> None:
    """Add directly tradable fixed-threshold booleans."""
    threshold_specs = [
        ("taker_buy_ratio", "buy_ratio", _parse_number_list(args.buy_ratio_thresholds, cast=float, name="buy_ratio_thresholds"), 100),
        ("buy_pressure", "buy_pressure", _parse_number_list(args.buy_pressure_thresholds, cast=float, name="buy_pressure_thresholds"), 100),
        ("delta_pressure", "delta_pressure", _parse_number_list(args.delta_pressure_thresholds, cast=float, name="delta_pressure_thresholds", allow_zero=True), 100),
    ]
    for col, prefix, thresholds, scale in threshold_specs:
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        for threshold in thresholds:
            out[f"{prefix}_ge_{_tag_number(float(threshold), scale=scale)}"] = series >= float(threshold)

    for w in _parse_number_list(args.cvd_windows, cast=int, name="cvd_windows"):
        col = f"cvd_pressure_{int(w)}"
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        for threshold in _parse_number_list(args.cvd_pressure_thresholds, cast=float, name="cvd_pressure_thresholds", allow_zero=True):
            # Scale by 10000 to preserve e.g. 0.05 -> 0500 in the filter name.
            out[f"{col}_ge_{_tag_number(float(threshold), scale=10000)}"] = series >= float(threshold)


def _add_historical_rolling_quantile_flags(out: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Return frame with leakage-safe rolling-quantile booleans.

    The threshold for bar t is computed from bars strictly before t:
    ``series.shift(1).rolling(window).quantile(q)``. This is intentionally
    conservative and suitable for later strategy simulation.
    """
    days_list = _parse_number_list(args.rolling_quantile_days, cast=float, name="rolling_quantile_days")
    quantiles = _parse_number_list(args.rolling_quantiles, cast=float, name="rolling_quantiles")
    source_cols = [
        ("atr_pct", "atr"),
        ("volume_ratio", "volume"),
        ("trades_count_ratio", "trades"),
        ("taker_buy_ratio", "buy_ratio"),
        ("buy_pressure", "buy_pressure"),
        ("delta_pressure", "delta_pressure"),
        ("large_trade_share", "large_share"),
        ("large_delta_pressure", "large_delta"),
        ("cvd_pressure_15", "cvdp15"),
        ("cvd_pressure_60", "cvdp60"),
    ]
    new_cols: dict[str, pd.Series] = {}
    for col, prefix in source_cols:
        if col not in out.columns:
            continue
        series = pd.to_numeric(out[col], errors="coerce")
        for days in days_list:
            window = _rolling_window_bars(args, float(days))
            min_periods = min(window, max(100, window // 3))
            if window < 10:
                min_periods = max(3, window)
            for q in quantiles:
                q = float(q)
                if not 0.0 < q < 1.0:
                    continue
                threshold = series.shift(1).rolling(window, min_periods=min_periods).quantile(q)
                day_tag = str(int(days)) if float(days).is_integer() else str(days).replace(".", "p")
                q_tag = _tag_number(q, scale=100)
                base = f"{prefix}_rq{q_tag}_{day_tag}d"
                new_cols[f"{base}_threshold"] = threshold
                new_cols[base] = series >= threshold
    if not new_cols:
        return out
    return pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)


def build_enriched_features(bars: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Build base swing/event features plus richer trade-bar order-flow features."""
    out = build_base_features(bars, args).copy()

    buy_notional = _to_float_series(out, "buy_notional")
    sell_notional = _to_float_series(out, "sell_notional")
    total_notional = (buy_notional + sell_notional).replace(0.0, np.nan)
    out["total_notional"] = total_notional
    out["buy_pressure"] = buy_notional / total_notional
    out["sell_pressure"] = sell_notional / total_notional
    out["notional_imbalance"] = (buy_notional - sell_notional) / total_notional

    buy_volume = _to_float_series(out, "buy_volume")
    sell_volume = _to_float_series(out, "sell_volume")
    total_volume = (buy_volume + sell_volume).replace(0.0, np.nan)
    out["buy_volume_pressure"] = buy_volume / total_volume
    out["volume_imbalance"] = (buy_volume - sell_volume) / total_volume

    buy_trades = _to_float_series(out, "buy_trades_count")
    sell_trades = _to_float_series(out, "sell_trades_count")
    total_trades = (buy_trades + sell_trades).replace(0.0, np.nan)
    out["buy_trade_count_pressure"] = buy_trades / total_trades
    out["trade_count_imbalance"] = (buy_trades - sell_trades) / total_trades

    large_buy = _to_float_series(out, "large_buy_notional", 0.0)
    large_sell = _to_float_series(out, "large_sell_notional", 0.0)
    large_delta = _to_float_series(out, "large_delta_notional", 0.0)
    out["large_trade_share"] = (large_buy + large_sell) / total_notional
    out["large_delta_pressure"] = large_delta / total_notional

    delta_notional = _to_float_series(out, "delta_notional")
    out["delta_pressure"] = delta_notional / total_notional

    cvd = _to_float_series(out, "cvd_notional")
    cvd_windows = _parse_number_list(args.cvd_windows, cast=int, name="cvd_windows")
    for w in cvd_windows:
        w = int(w)
        delta_col = f"cvd_delta_{w}"
        pressure_col = f"cvd_pressure_{w}"
        z_col = f"cvd_delta_z_{w}"
        out[delta_col] = cvd - cvd.shift(w)
        notional_sum = total_notional.rolling(w, min_periods=max(3, min(w, max(3, w // 3)))).sum().replace(0.0, np.nan)
        out[pressure_col] = out[delta_col] / notional_sum
        hist = out[delta_col].shift(1).rolling(max(int(args.cvd_window), w), min_periods=max(10, min(int(args.cvd_window), w))).agg(["mean", "std"])
        out[z_col] = (out[delta_col] - hist["mean"]) / hist["std"].replace(0.0, np.nan)

    # Panic-then-turn proxy: medium-window CVD is still negative, but near-term CVD turns up.
    if "cvd_delta_5" in out.columns and "cvd_delta_30" in out.columns:
        out["cvd_short_turn_up_after_dump"] = (out["cvd_delta_30"] < 0) & (out["cvd_delta_5"] > 0)
    elif "cvd_delta_15" in out.columns and "cvd_delta_60" in out.columns:
        out["cvd_short_turn_up_after_dump"] = (out["cvd_delta_60"] < 0) & (out["cvd_delta_15"] > 0)
    else:
        out["cvd_short_turn_up_after_dump"] = False

    # Existing focused lab feature names plus stronger order-flow booleans.
    out["buy_pressure_high"] = out["buy_pressure"] >= out["buy_pressure"].quantile(0.75)
    out["sell_pressure_high"] = out["sell_pressure"] >= out["sell_pressure"].quantile(0.75)
    out["large_trade_share_high"] = out["large_trade_share"] >= out["large_trade_share"].quantile(0.75)
    out["large_buy_absorption"] = (out["large_delta_pressure"] > 0) & (out["down_spike_pct"] >= 0.008)

    _add_fixed_threshold_flags(out, args)
    out = _add_historical_rolling_quantile_flags(out, args)
    return out


def attach_extra_features_to_events(events: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events.copy()
    extra_cols = [
        "total_notional",
        "buy_pressure",
        "sell_pressure",
        "notional_imbalance",
        "buy_volume_pressure",
        "volume_imbalance",
        "buy_trade_count_pressure",
        "trade_count_imbalance",
        "large_trade_share",
        "large_delta_pressure",
        "delta_pressure",
        "cvd_short_turn_up_after_dump",
        "buy_pressure_high",
        "sell_pressure_high",
        "large_trade_share_high",
        "large_buy_absorption",
    ]
    for col in features.columns:
        if (
            col.startswith("cvd_delta_")
            or col.startswith("cvd_pressure_")
            or col.startswith("cvd_delta_z_")
            or col.startswith("buy_ratio_ge_")
            or col.startswith("buy_pressure_ge_")
            or col.startswith("delta_pressure_ge_")
            or col.startswith("atr_rq")
            or col.startswith("volume_rq")
            or col.startswith("trades_rq")
            or col.startswith("buy_ratio_rq")
            or col.startswith("buy_pressure_rq")
            or col.startswith("delta_pressure_rq")
            or col.startswith("large_share_rq")
            or col.startswith("large_delta_rq")
            or col.startswith("cvdp15_rq")
            or col.startswith("cvdp60_rq")
        ):
            extra_cols.append(col)
    extra_cols = [c for c in dict.fromkeys(extra_cols) if c in features.columns]
    if not extra_cols:
        return events.copy()
    right = features[extra_cols].copy().reset_index()
    right = right.rename(columns={right.columns[0]: "signal_time"})
    out = events.copy()
    out["signal_time"] = pd.to_datetime(out["signal_time"])
    return out.merge(right, on="signal_time", how="left", suffixes=("", "_extra"))


def add_filter_bins(events: pd.DataFrame) -> pd.DataFrame:
    """Add only fixed-bin labels; do not add in-sample/global qcut labels.

    Previous probe versions created BUY_Q4 / ATR_Q4 / CVDP15_Q4 with full-sample
    qcut. Those labels are useful for exploration but leak future distribution
    information into early bars. This strategy-probe version intentionally omits
    them from both event output and filter selection.
    """
    out = events.copy()
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
    out["delta_positive"] = pd.to_numeric(out.get("delta_notional", np.nan), errors="coerce") > 0
    out["delta_pressure_positive"] = pd.to_numeric(out.get("delta_pressure", np.nan), errors="coerce") > 0
    out["strong_volume_spike"] = pd.to_numeric(out["volume_ratio"], errors="coerce") >= 2.0
    out["deep_close"] = pd.to_numeric(out["close_pos_in_bar"], errors="coerce") <= 0.30
    out["large_lower_wick"] = pd.to_numeric(out["lower_wick_frac"], errors="coerce") >= 0.45
    return out


def build_filter_specs(events: pd.DataFrame) -> list[FilterSpec]:
    """Define atomic filters. Combos are generated from these atoms."""

    def col_eq(col: str, value: object) -> Callable[[pd.DataFrame], pd.Series]:
        return lambda df: df[col].astype("object") == value if col in df.columns else pd.Series(False, index=df.index)

    def ge(col: str, value: float) -> Callable[[pd.DataFrame], pd.Series]:
        return lambda df: pd.to_numeric(df.get(col, np.nan), errors="coerce") >= float(value)

    def le(col: str, value: float) -> Callable[[pd.DataFrame], pd.Series]:
        return lambda df: pd.to_numeric(df.get(col, np.nan), errors="coerce") <= float(value)

    def bool_col(col: str) -> Callable[[pd.DataFrame], pd.Series]:
        return lambda df: df[col].astype(bool) if col in df.columns else pd.Series(False, index=df.index)

    specs: list[FilterSpec] = [
        FilterSpec("variant_close_through", "variant", "close remains below swept swing low", col_eq("variant", "fade_close_through")),
        FilterSpec("session_00_07", "session", "UTC 00-07", col_eq("session_bucket", "S0_00_07")),
        FilterSpec("session_08_15", "session", "UTC 08-15", col_eq("session_bucket", "S1_08_15")),
        FilterSpec("session_16_23", "session", "UTC 16-23", col_eq("session_bucket", "S2_16_23")),
        FilterSpec("spike_ge_0080", "spike", "down spike >= 0.80%", ge("down_spike_pct", 0.0080)),
        FilterSpec("spike_ge_0100", "spike", "down spike >= 1.00%", ge("down_spike_pct", 0.0100)),
        FilterSpec("spike_ge_0120", "spike", "down spike >= 1.20%", ge("down_spike_pct", 0.0120)),
        FilterSpec("swing_age_le_24", "structure", "confirmed swing age <= 24 bars", le("swing_age", 24)),
        FilterSpec("swing_age_le_48", "structure", "confirmed swing age <= 48 bars", le("swing_age", 48)),
        FilterSpec("prom_ge_0030", "structure", "swing prominence >= 0.30%", ge("swing_prominence_pct", 0.0030)),
        FilterSpec("close_pos_le_030", "bar_shape", "close in lower 30% of bar", le("close_pos_in_bar", 0.30)),
        FilterSpec("large_lower_wick", "bar_shape", "large lower wick", bool_col("large_lower_wick")),
        FilterSpec("volume_ge_150", "volume", "volume >= 1.5x rolling median", ge("volume_ratio", 1.50)),
        FilterSpec("volume_ge_200", "volume", "volume >= 2.0x rolling median", ge("volume_ratio", 2.00)),
        FilterSpec("delta_positive", "delta", "delta notional > 0", bool_col("delta_positive")),
        FilterSpec("delta_pressure_positive", "delta", "delta pressure > 0", bool_col("delta_pressure_positive")),
        FilterSpec("large_buy_absorption", "large_flow", "large buy delta while sweeping low", bool_col("large_buy_absorption")),
        FilterSpec("cvd_turn_up_after_dump", "cvd", "medium CVD dump + short CVD turn-up", bool_col("cvd_short_turn_up_after_dump")),
    ]

    # Directly tradable fixed thresholds created in build_enriched_features().
    fixed_prefix_family = {
        "buy_ratio_ge_": "buy_sell_fixed",
        "buy_pressure_ge_": "buy_sell_fixed",
        "delta_pressure_ge_": "delta_fixed",
        "cvd_pressure_": "cvd_fixed",
    }
    for col in sorted(events.columns):
        if col.startswith("buy_ratio_ge_"):
            specs.append(FilterSpec(col, "buy_sell_fixed", col.replace("_", " "), bool_col(col)))
        elif col.startswith("buy_pressure_ge_"):
            specs.append(FilterSpec(col, "buy_sell_fixed", col.replace("_", " "), bool_col(col)))
        elif col.startswith("delta_pressure_ge_"):
            specs.append(FilterSpec(col, "delta_fixed", col.replace("_", " "), bool_col(col)))
        elif col.startswith("cvd_pressure_") and "_ge_" in col:
            specs.append(FilterSpec(col, "cvd_fixed", col.replace("_", " "), bool_col(col)))

    # Leakage-safe rolling historical quantile flags. These are created from
    # feature.shift(1).rolling(window).quantile(q), so the threshold is known
    # before the signal bar closes.
    rolling_family_prefixes = {
        "atr_rq": "volatility_rolling",
        "volume_rq": "volume_rolling",
        "trades_rq": "activity_rolling",
        "buy_ratio_rq": "buy_sell_rolling",
        "buy_pressure_rq": "buy_sell_rolling",
        "delta_pressure_rq": "delta_rolling",
        "large_share_rq": "large_flow_rolling",
        "large_delta_rq": "large_flow_rolling",
        "cvdp15_rq": "cvd_rolling",
        "cvdp60_rq": "cvd_rolling",
    }
    for col in sorted(events.columns):
        if col.endswith("_threshold"):
            continue
        for prefix, family in rolling_family_prefixes.items():
            if col.startswith(prefix):
                specs.append(FilterSpec(col, family, f"historical rolling quantile {col}", bool_col(col)))
                break

    # Keep only specs whose mask is not completely empty. This allows the script
    # to run on older trade-bar DBs that may lack some order-flow columns.
    out: list[FilterSpec] = []
    for spec in specs:
        try:
            mask = spec.mask_builder(events).fillna(False).astype(bool)
        except Exception:
            continue
        if int(mask.sum()) > 0:
            out.append(spec)
    return out


def _families_ok(combo: tuple[FilterSpec, ...], max_family_repeat: int) -> bool:
    counts: dict[str, int] = {}
    for spec in combo:
        counts[spec.family] = counts.get(spec.family, 0) + 1
    return all(v <= int(max_family_repeat) for v in counts.values())


def _profit_factor_raw(x: pd.Series) -> float:
    vals = pd.to_numeric(x, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    gp = float(vals[vals > 0].sum())
    gl = float(-vals[vals <= 0].sum())
    if gl <= 0:
        return float("inf") if gp > 0 else float("nan")
    return gp / gl


def _max_drawdown_simple(returns: pd.Series) -> float:
    vals = pd.to_numeric(returns, errors="coerce").dropna().to_numpy(dtype=float)
    if vals.size == 0:
        return float("nan")
    equity = np.cumprod(1.0 + vals)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(np.nanmin(dd))


def _year_stats(part: pd.DataFrame, return_col: str, min_count: int) -> tuple[int, int, str]:
    if part.empty or return_col not in part.columns:
        return 0, 0, ""
    rows = []
    tested = 0
    positive = 0
    for year, grp in part.groupby("year"):
        x = pd.to_numeric(grp[return_col], errors="coerce").dropna()
        if len(x) < int(min_count):
            continue
        tested += 1
        m = float(x.mean())
        if m > 0:
            positive += 1
        rows.append(f"{int(year)}:{len(x)}:{m:.6f}:{float((x>0).mean()):.3f}")
    return tested, positive, "|".join(rows)


def _combo_mask(events: pd.DataFrame, combo: tuple[FilterSpec, ...]) -> pd.Series:
    mask = pd.Series(True, index=events.index)
    for spec in combo:
        mask &= spec.mask_builder(events).fillna(False).astype(bool)
    return mask


def run_filter_matrix(events: pd.DataFrame, args: argparse.Namespace, return_cols: list[str]) -> pd.DataFrame:
    specs = build_filter_specs(events)
    rows: list[dict[str, object]] = []
    combos: list[tuple[FilterSpec, ...]] = [tuple()]
    for depth in range(1, int(args.max_filter_depth) + 1):
        for combo in itertools.combinations(specs, depth):
            if _families_ok(combo, int(args.max_family_repeat)):
                combos.append(combo)

    progress = ProgressReporter(
        label="[filters] matrix",
        total=len(combos),
        every=max(1, min(500, len(combos) // 20 if len(combos) > 20 else 1)),
        enabled=not bool(args.no_progress),
    )
    min_count = int(args.min_count)
    candidate_return_col = f"next_open_ret_h{int(args.candidate_horizon)}_net"
    for done, combo in enumerate(combos, start=1):
        mask = _combo_mask(events, combo) if combo else pd.Series(True, index=events.index)
        part = events.loc[mask]
        count = int(len(part))
        if count >= min_count:
            row: dict[str, object] = {
                "filter_name": "ALL" if not combo else "&".join(spec.name for spec in combo),
                "filter_depth": len(combo),
                "filter_families": "|".join(spec.family for spec in combo),
                "filter_descriptions": " | ".join(spec.description for spec in combo),
                "count": count,
                "unique_signal_time": int(part["signal_time"].nunique()) if "signal_time" in part.columns else count,
            }
            tested, positive, detail = _year_stats(part, candidate_return_col, max(20, min_count // 2))
            row["tested_years"] = tested
            row["positive_years"] = positive
            row["year_detail"] = detail
            for col in return_cols:
                x = pd.to_numeric(part[col], errors="coerce").dropna() if col in part.columns else pd.Series(dtype=float)
                prefix = col.replace("next_open_ret_", "")
                if x.empty:
                    row[f"{prefix}_mean"] = np.nan
                    row[f"{prefix}_median"] = np.nan
                    row[f"{prefix}_win_rate"] = np.nan
                    row[f"{prefix}_pf"] = np.nan
                    continue
                row[f"{prefix}_mean"] = float(x.mean())
                row[f"{prefix}_median"] = float(x.median())
                row[f"{prefix}_win_rate"] = float((x > 0).mean())
                row[f"{prefix}_pf"] = _profit_factor_raw(x)
                row[f"{prefix}_top5_share"] = top_winner_dependency(x, top_n=5)
                row[f"{prefix}_max_dd_simple"] = _max_drawdown_simple(x)
            cand_x = pd.to_numeric(part[candidate_return_col], errors="coerce").dropna() if candidate_return_col in part.columns else pd.Series(dtype=float)
            if not cand_x.empty:
                mean = float(cand_x.mean())
                median = float(cand_x.median())
                wr = float((cand_x > 0).mean())
                pf = _profit_factor_raw(cand_x)
                top5 = top_winner_dependency(cand_x, top_n=5)
                candidate_flag = (
                    mean > 0
                    and median > 0
                    and wr >= float(args.min_win_rate)
                    and pf >= float(args.min_profit_factor)
                    and positive >= int(args.min_positive_years)
                    and (not np.isfinite(top5) or top5 <= float(args.max_top5_winner_share))
                )
                row["candidate_flag"] = bool(candidate_flag)
                row["rank_score"] = mean * 10_000 + median * 4_000 + wr * 10 + (pf if np.isfinite(pf) else 3.0) * 10 + positive * 5 + math.log(count)
            else:
                row["candidate_flag"] = False
                row["rank_score"] = np.nan
            rows.append(row)
        progress.update(done)
    progress.close()
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["candidate_flag", "rank_score", "count"], ascending=[False, False, False]).reset_index(drop=True)


def select_top_filters(matrix: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if matrix.empty:
        return matrix
    candidates = matrix[matrix["candidate_flag"].astype(bool)].copy()
    if candidates.empty:
        candidates = matrix.copy()
    return candidates.sort_values(["rank_score", "count"], ascending=[False, False]).head(int(args.top_filter_count)).reset_index(drop=True)


def summarize_top_filter_yearly(events: pd.DataFrame, top_filters: pd.DataFrame, args: argparse.Namespace, return_cols: list[str]) -> pd.DataFrame:
    specs = {s.name: s for s in build_filter_specs(events)}
    rows: list[pd.DataFrame] = []
    for _, row in top_filters.iterrows():
        name = str(row["filter_name"])
        if name == "ALL":
            part = events
        else:
            combo = tuple(specs[n] for n in name.split("&") if n in specs)
            part = events.loc[_combo_mask(events, combo)] if combo else events.iloc[0:0]
        s = summarize_many(part, return_cols, group_cols=["year"], min_count=max(20, int(args.min_count) // 2))
        if not s.empty:
            s.insert(0, "filter_name", name)
            rows.append(s)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _side_series_for_events(bars: pd.DataFrame, events: pd.DataFrame) -> pd.Series:
    side = pd.Series(0, index=bars.index, dtype="int64")
    if events.empty:
        return side
    idx = bars.index.get_indexer(pd.DatetimeIndex(pd.to_datetime(events["signal_time"])))
    valid = idx >= 0
    if valid.any():
        side.iloc[idx[valid]] = 1
    return side


def first_touch_strategy_rank(bars: pd.DataFrame, events: pd.DataFrame, top_filters: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if events.empty or top_filters.empty:
        return pd.DataFrame()
    specs = {s.name: s for s in build_filter_specs(events)}
    targets = _parse_number_list(args.touch_target_pcts, cast=float, name="touch_target_pcts")
    stops = _parse_number_list(args.touch_stop_pcts, cast=float, name="touch_stop_pcts")
    horizon = int(args.touch_horizon)
    horizon_col = f"next_open_ret_h{horizon}_net"
    cost = _round_trip_cost(args, 1.0)
    rows: list[dict[str, object]] = []

    # Precompute touch result for every target/stop on the full axis.
    touch_cache: dict[tuple[float, float], pd.DataFrame] = {}
    all_side = _side_series_for_events(bars, events)
    for target in targets:
        for stop in stops:
            touch_cache[(float(target), float(stop))] = first_touch_outcome(
                bars,
                all_side,
                target_pct=float(target),
                stop_pct=float(stop),
                horizon=horizon,
                entry_delay_bars=1,
                same_bar_policy="conservative",
            )

    for _, filter_row in top_filters.iterrows():
        name = str(filter_row["filter_name"])
        if name == "ALL":
            part = events.copy()
        else:
            combo = tuple(specs[n] for n in name.split("&") if n in specs)
            part = events.loc[_combo_mask(events, combo)].copy() if combo else events.iloc[0:0].copy()
        if len(part) < int(args.min_count):
            continue
        event_times = pd.DatetimeIndex(pd.to_datetime(part["signal_time"]))
        timeout_ret = pd.to_numeric(part.get(horizon_col, np.nan), errors="coerce")
        for target in targets:
            for stop in stops:
                selected = touch_cache[(float(target), float(stop))].loc[event_times]
                result = selected["touch_result"].reset_index(drop=True)
                # Conservative realized return proxy: TP/SL return minus round-trip cost;
                # timeout uses the fixed-horizon net return from the event study.
                realized = pd.Series(np.nan, index=part.index, dtype="float64")
                realized.loc[result.values == "TARGET"] = float(target) - cost
                realized.loc[result.values == "STOP"] = -float(stop) - cost
                realized.loc[result.values == "BOTH_UNKNOWN"] = -float(stop) - cost
                timeout_mask = result.values == "TIMEOUT"
                if timeout_mask.any():
                    realized.iloc[np.where(timeout_mask)[0]] = timeout_ret.iloc[np.where(timeout_mask)[0]].to_numpy(dtype=float)
                realized = realized.dropna()
                if len(realized) < int(args.min_count):
                    continue
                counts = result.value_counts(dropna=False)
                tested, positive, detail = _year_stats(pd.concat([part.reset_index(drop=True), realized.reset_index(drop=True).rename("_ret")], axis=1), "_ret", max(20, int(args.min_count) // 2))
                pf = _profit_factor_raw(realized)
                mean = float(realized.mean())
                median = float(realized.median())
                wr = float((realized > 0).mean())
                rows.append(
                    {
                        "filter_name": name,
                        "target_pct": float(target),
                        "stop_pct": float(stop),
                        "horizon": horizon,
                        "count": int(len(realized)),
                        "mean_net": mean,
                        "median_net": median,
                        "win_rate": wr,
                        "profit_factor": pf,
                        "max_dd_simple": _max_drawdown_simple(realized),
                        "top5_winner_share": top_winner_dependency(realized, top_n=5),
                        "target_share": float(counts.get("TARGET", 0) / len(part)),
                        "stop_share": float(counts.get("STOP", 0) / len(part)),
                        "timeout_share": float(counts.get("TIMEOUT", 0) / len(part)),
                        "both_unknown_share": float(counts.get("BOTH_UNKNOWN", 0) / len(part)),
                        "same_bar_both_hit_share": float(selected["same_bar_both_hit_flag"].mean()),
                        "avg_touch_bars": float(pd.to_numeric(selected["touch_bars"], errors="coerce").mean()),
                        "tested_years": tested,
                        "positive_years": positive,
                        "year_detail": detail,
                        "rank_score": mean * 10_000 + median * 4_000 + wr * 10 + (pf if np.isfinite(pf) else 3.0) * 10 + positive * 5 + math.log(len(realized)),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["rank_score", "count"], ascending=[False, False]).reset_index(drop=True)


def run_stress_for_top_filters(bars: pd.DataFrame, events: pd.DataFrame, top_filters: pd.DataFrame, args: argparse.Namespace, *, stress_type: str) -> pd.DataFrame:
    if events.empty or top_filters.empty:
        return pd.DataFrame()
    specs = {s.name: s for s in build_filter_specs(events)}
    horizons = tuple(_parse_number_list(args.horizons, cast=int, name="horizons"))
    return_cols = [f"next_open_ret_h{int(h)}_net" for h in horizons]
    rows: list[pd.DataFrame] = []
    if stress_type == "delay":
        variants = [(int(x), 1.0) for x in _parse_number_list(args.delay_bars_list, cast=int, name="delay_bars_list")]
    elif stress_type == "cost":
        variants = [(1, float(x)) for x in _parse_number_list(args.cost_multipliers, cast=float, name="cost_multipliers")]
    else:
        raise ValueError("stress_type must be delay or cost")

    for _, filter_row in top_filters.iterrows():
        name = str(filter_row["filter_name"])
        if name == "ALL":
            base_part = events.copy()
        else:
            combo = tuple(specs[n] for n in name.split("&") if n in specs)
            base_part = events.loc[_combo_mask(events, combo)].copy() if combo else events.iloc[0:0].copy()
        if len(base_part) < int(args.min_count):
            continue
        slim = base_part[["signal_time", "side", "event_name"]].copy()
        for delay, cost_mult in variants:
            cfg = EventStudyConfig(
                horizons=horizons,
                mfe_mae_horizon=int(args.mfe_mae_horizon),
                entry_delay_bars=int(delay),
                cost=CostConfig(
                    entry_fee_rate=float(args.entry_fee_rate) * float(cost_mult),
                    exit_fee_rate=float(args.exit_fee_rate) * float(cost_mult),
                    entry_slippage_pct=float(args.entry_slippage_pct) * float(cost_mult),
                    exit_slippage_pct=float(args.exit_slippage_pct) * float(cost_mult),
                ),
                min_count=int(args.min_count),
                progress_every=0,
            )
            result = run_event_study(bars, slim, cfg)
            s = summarize_many(result.events, return_cols, min_count=int(args.min_count))
            if not s.empty:
                s.insert(0, "filter_name", name)
                s.insert(1, "stress_type", stress_type)
                s.insert(2, "entry_delay_bars", int(delay))
                s.insert(3, "cost_mult", float(cost_mult))
                s.insert(4, "round_trip_cost_pct", float(cfg.cost.round_trip_cost_pct))
                rows.append(s)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _filter_mask_by_names(events: pd.DataFrame, filter_names: Sequence[str]) -> pd.Series:
    """Build a mask from atomic filter names; missing atoms make the mask empty."""
    specs = {s.name: s for s in build_filter_specs(events)}
    mask = pd.Series(True, index=events.index)
    for name in filter_names:
        spec = specs.get(str(name))
        if spec is None:
            return pd.Series(False, index=events.index)
        mask &= spec.mask_builder(events).fillna(False).astype(bool)
    return mask


def build_fixed_candidate_masks(events: pd.DataFrame) -> dict[str, dict[str, object]]:
    """Fixed no-leakage candidate families for follow-up validation.

    These are intentionally small and explicit. They are not optimized from this
    run; they are the current human-selected research hypotheses to track across
    reports and, later, across strategy probes.
    """
    definitions: dict[str, dict[str, object]] = {
        "A_spike_close_large_share": {
            "description": "spike>=0.8%, close in lower 30% of bar, large-trade share above historical 90d q80",
            "filters": ("spike_ge_0080", "close_pos_le_030", "large_share_rq80_90d"),
            "edge_hypothesis": "panic flush with heavy large-trade participation",
        },
        "B_session_spike_atr": {
            "description": "UTC 00-07, spike>=0.8%, ATR above historical 90d q80",
            "filters": ("session_00_07", "spike_ge_0080", "atr_rq80_90d"),
            "edge_hypothesis": "high-volatility low sweep during the strongest reversal session",
        },
        "C_session_extreme_spike": {
            "description": "UTC 00-07 and spike>=1.2%",
            "filters": ("session_00_07", "spike_ge_0120"),
            "edge_hypothesis": "extreme downside stop-run during the strongest reversal session",
        },
    }
    base_masks = {name: _filter_mask_by_names(events, meta["filters"]) for name, meta in definitions.items()}
    out: dict[str, dict[str, object]] = {}
    for name, meta in definitions.items():
        out[name] = {**meta, "mask": base_masks[name], "bucket_type": "candidate"}

    a = base_masks["A_spike_close_large_share"]
    b = base_masks["B_session_spike_atr"]
    c = base_masks["C_session_extreme_spike"]
    union = a | b | c
    bucket_masks = {
        "ABC_union": union,
        "A_only": a & ~b & ~c,
        "B_only": b & ~a & ~c,
        "C_only": c & ~a & ~b,
        "AB_overlap_only": a & b & ~c,
        "AC_overlap_only": a & c & ~b,
        "BC_overlap_only": b & c & ~a,
        "ABC_overlap": a & b & c,
    }
    bucket_desc = {
        "ABC_union": "A or B or C, de-duplicated by canonical signal rows",
        "A_only": "A events that do not overlap B or C",
        "B_only": "B events that do not overlap A or C",
        "C_only": "C events that do not overlap A or B",
        "AB_overlap_only": "A and B overlap, excluding C",
        "AC_overlap_only": "A and C overlap, excluding B",
        "BC_overlap_only": "B and C overlap, excluding A",
        "ABC_overlap": "A, B, and C overlap",
    }
    for name, mask in bucket_masks.items():
        out[name] = {
            "description": bucket_desc[name],
            "filters": (),
            "edge_hypothesis": "candidate bucket / overlap validation",
            "mask": mask,
            "bucket_type": "union_bucket" if name == "ABC_union" else "overlap_bucket",
        }
    return out


def _summarize_return_frame(part: pd.DataFrame, return_cols: Sequence[str], args: argparse.Namespace) -> dict[str, object]:
    rec: dict[str, object] = {
        "count": int(len(part)),
        "unique_signal_time": int(part["signal_time"].nunique()) if "signal_time" in part.columns and not part.empty else 0,
    }
    for col in return_cols:
        x = pd.to_numeric(part[col], errors="coerce").dropna() if col in part.columns else pd.Series(dtype=float)
        prefix = col.replace("next_open_ret_", "")
        if x.empty:
            rec[f"{prefix}_mean"] = np.nan
            rec[f"{prefix}_median"] = np.nan
            rec[f"{prefix}_win_rate"] = np.nan
            rec[f"{prefix}_pf"] = np.nan
            rec[f"{prefix}_top5_share"] = np.nan
            rec[f"{prefix}_max_dd_simple"] = np.nan
            continue
        rec[f"{prefix}_mean"] = float(x.mean())
        rec[f"{prefix}_median"] = float(x.median())
        rec[f"{prefix}_win_rate"] = float((x > 0).mean())
        rec[f"{prefix}_pf"] = _profit_factor_raw(x)
        rec[f"{prefix}_top5_share"] = top_winner_dependency(x, top_n=5)
        rec[f"{prefix}_max_dd_simple"] = _max_drawdown_simple(x)
    return rec


def summarize_fixed_candidate_groups(events: pd.DataFrame, args: argparse.Namespace, return_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, meta in build_fixed_candidate_masks(events).items():
        mask = meta["mask"].fillna(False).astype(bool)
        part = events.loc[mask].copy()
        rec = _summarize_return_frame(part, return_cols, args)
        rec.update(
            {
                "candidate_name": name,
                "bucket_type": meta.get("bucket_type", "candidate"),
                "description": meta.get("description", ""),
                "filters": "&".join(meta.get("filters", ())),
                "edge_hypothesis": meta.get("edge_hypothesis", ""),
            }
        )
        for horizon in _parse_number_list(args.horizons, cast=int, name="horizons"):
            col = f"next_open_ret_h{int(horizon)}_net"
            tested, positive, detail = _year_stats(part, col, max(20, int(args.min_count) // 2))
            rec[f"h{int(horizon)}_tested_years"] = tested
            rec[f"h{int(horizon)}_positive_years"] = positive
            rec[f"h{int(horizon)}_year_detail"] = detail
        rows.append(rec)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # Prefer candidate families before overlap buckets, then strongest h48/h96.
    sort_cols = [c for c in ["h48_net_mean", "h96_net_mean", "count"] if c in out.columns]
    return out.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)


def summarize_fixed_candidate_yearly(events: pd.DataFrame, args: argparse.Namespace, return_cols: Sequence[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for name, meta in build_fixed_candidate_masks(events).items():
        mask = meta["mask"].fillna(False).astype(bool)
        part = events.loc[mask].copy()
        if part.empty:
            continue
        s = summarize_many(part, return_cols, group_cols=["year"], min_count=max(10, int(args.min_count) // 4))
        if not s.empty:
            s.insert(0, "candidate_name", name)
            s.insert(1, "bucket_type", meta.get("bucket_type", "candidate"))
            rows.append(s)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fixed_candidate_overlap_matrix(events: pd.DataFrame) -> pd.DataFrame:
    candidates = build_fixed_candidate_masks(events)
    base_names = ["A_spike_close_large_share", "B_session_spike_atr", "C_session_extreme_spike"]
    rows: list[dict[str, object]] = []
    masks = {name: candidates[name]["mask"].fillna(False).astype(bool) for name in base_names}
    for left in base_names:
        for right in base_names:
            inter = masks[left] & masks[right]
            union = masks[left] | masks[right]
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "left_count": int(masks[left].sum()),
                    "right_count": int(masks[right].sum()),
                    "intersection_count": int(inter.sum()),
                    "union_count": int(union.sum()),
                    "jaccard": float(inter.sum() / union.sum()) if int(union.sum()) > 0 else np.nan,
                    "left_overlap_share": float(inter.sum() / masks[left].sum()) if int(masks[left].sum()) > 0 else np.nan,
                    "right_overlap_share": float(inter.sum() / masks[right].sum()) if int(masks[right].sum()) > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _path_matrix_for_events(bars: pd.DataFrame, events: pd.DataFrame, *, horizon: int, entry_delay_bars: int = 1) -> tuple[pd.DataFrame, np.ndarray]:
    """Return valid events and gross forward return matrix using next-open entry.

    Column k-1 is the gross return from entry open to signal+k close, matching
    the reusable event-study convention. This is path analysis only; no future
    labels are used to form signals or filters.
    """
    if events.empty:
        return events.copy(), np.empty((0, int(horizon)), dtype=float)
    frame = bars.sort_index()
    event_times = pd.DatetimeIndex(pd.to_datetime(events["signal_time"]))
    signal_pos = frame.index.get_indexer(event_times)
    horizon = int(horizon)
    delay = int(entry_delay_bars)
    max_needed = signal_pos + horizon
    valid_mask = (signal_pos >= 0) & ((signal_pos + delay) < len(frame)) & (max_needed < len(frame))
    valid_events = events.loc[valid_mask].copy().reset_index(drop=True)
    if valid_events.empty:
        return valid_events, np.empty((0, horizon), dtype=float)
    valid_signal_pos = signal_pos[valid_mask]
    side = pd.to_numeric(valid_events.get("side", 1), errors="coerce").fillna(1).to_numpy(dtype=float)
    opens = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=float)
    entry = opens[valid_signal_pos + delay]
    arr = np.full((len(valid_events), horizon), np.nan, dtype=float)
    for k in range(1, horizon + 1):
        exit_close = closes[valid_signal_pos + k]
        gross = (exit_close / entry - 1.0) * side
        arr[:, k - 1] = gross
    return valid_events, arr


def path_forward_curve(bars: pd.DataFrame, events: pd.DataFrame, args: argparse.Namespace, candidate_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    horizon = int(args.mfe_mae_horizon)
    cost = _round_trip_cost(args, 1.0)
    candidates = build_fixed_candidate_masks(events)
    for name, meta in candidates.items():
        if name not in set(candidate_summary.get("candidate_name", pd.Series(dtype=str)).astype(str)):
            continue
        mask = meta["mask"].fillna(False).astype(bool)
        part = events.loc[mask].copy()
        if len(part) < max(1, int(args.min_count) // 4):
            continue
        _, arr = _path_matrix_for_events(bars, part, horizon=horizon, entry_delay_bars=1)
        if arr.size == 0:
            continue
        for k in range(1, horizon + 1):
            x = pd.Series(arr[:, k - 1] - cost).dropna()
            if len(x) == 0:
                continue
            rows.append(
                {
                    "candidate_name": name,
                    "bucket_type": meta.get("bucket_type", "candidate"),
                    "bar_forward": k,
                    "count": int(len(x)),
                    "mean_net": float(x.mean()),
                    "median_net": float(x.median()),
                    "win_rate": float((x > 0).mean()),
                    "profit_factor": _profit_factor_raw(x),
                    "p10_net": float(x.quantile(0.10)),
                    "p25_net": float(x.quantile(0.25)),
                    "p75_net": float(x.quantile(0.75)),
                    "p90_net": float(x.quantile(0.90)),
                }
            )
    return pd.DataFrame(rows)


def path_timing_stats(bars: pd.DataFrame, events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    horizon = int(args.mfe_mae_horizon)
    candidates = build_fixed_candidate_masks(events)
    for name, meta in candidates.items():
        mask = meta["mask"].fillna(False).astype(bool)
        part = events.loc[mask].copy()
        if len(part) < max(1, int(args.min_count) // 4):
            continue
        _, arr = _path_matrix_for_events(bars, part, horizon=horizon, entry_delay_bars=1)
        if arr.size == 0:
            continue
        with np.errstate(all="ignore"):
            mae = np.nanmin(arr, axis=1)
            mfe = np.nanmax(arr, axis=1)
            time_to_mae = np.nanargmin(arr, axis=1) + 1
            time_to_mfe = np.nanargmax(arr, axis=1) + 1
        first_positive = []
        for row in arr:
            idx = np.where(row > 0)[0]
            first_positive.append(float(idx[0] + 1) if idx.size else np.nan)
        h12 = arr[:, min(12, horizon) - 1] if horizon >= 12 else arr[:, -1]
        h48 = arr[:, min(48, horizon) - 1] if horizon >= 48 else arr[:, -1]
        h96 = arr[:, min(96, horizon) - 1] if horizon >= 96 else arr[:, -1]
        rows.append(
            {
                "candidate_name": name,
                "bucket_type": meta.get("bucket_type", "candidate"),
                "count": int(arr.shape[0]),
                "mae_mean": float(np.nanmean(mae)),
                "mae_median": float(np.nanmedian(mae)),
                "mae_p10": float(np.nanquantile(mae, 0.10)),
                "mae_p25": float(np.nanquantile(mae, 0.25)),
                "mfe_mean": float(np.nanmean(mfe)),
                "mfe_median": float(np.nanmedian(mfe)),
                "mfe_p75": float(np.nanquantile(mfe, 0.75)),
                "mfe_p90": float(np.nanquantile(mfe, 0.90)),
                "median_time_to_mae": float(np.nanmedian(time_to_mae)),
                "median_time_to_mfe": float(np.nanmedian(time_to_mfe)),
                "mae_before_mfe_share": float(np.nanmean(time_to_mae < time_to_mfe)),
                "first_positive_median_bar": float(np.nanmedian(np.asarray(first_positive, dtype=float))),
                "first_positive_within_12_share": float(np.nanmean(np.asarray(first_positive, dtype=float) <= 12)),
                "h12_nonpos_h48_pos_share": float(np.nanmean((h12 <= 0) & (h48 > 0))),
                "h12_nonpos_h96_pos_share": float(np.nanmean((h12 <= 0) & (h96 > 0))),
                "drawdown_04_then_h48_pos_share": float(np.nanmean((mae <= -0.004) & (h48 > 0))),
                "drawdown_08_then_h48_pos_share": float(np.nanmean((mae <= -0.008) & (h48 > 0))),
            }
        )
    return pd.DataFrame(rows)


def path_stop_sensitivity(bars: pd.DataFrame, events: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    candidates = build_fixed_candidate_masks(events)
    stops = _parse_number_list(args.touch_stop_pcts, cast=float, name="touch_stop_pcts")
    horizons = [h for h in (48, 96) if h <= int(args.mfe_mae_horizon)]
    if not horizons:
        horizons = [int(args.mfe_mae_horizon)]
    max_h = max(horizons)
    cost = _round_trip_cost(args, 1.0)
    for name, meta in candidates.items():
        mask = meta["mask"].fillna(False).astype(bool)
        part = events.loc[mask].copy()
        if len(part) < max(1, int(args.min_count) // 4):
            continue
        _, arr = _path_matrix_for_events(bars, part, horizon=max_h, entry_delay_bars=1)
        if arr.size == 0:
            continue
        with np.errstate(all="ignore"):
            running_min = np.minimum.accumulate(arr, axis=1)
        for h in horizons:
            fixed_net = arr[:, h - 1] - cost
            for stop in stops:
                stop = float(stop)
                stopped = running_min[:, h - 1] <= -stop
                realized = fixed_net.copy()
                realized[stopped] = -stop - cost
                x = pd.Series(realized).dropna()
                if len(x) == 0:
                    continue
                rows.append(
                    {
                        "candidate_name": name,
                        "bucket_type": meta.get("bucket_type", "candidate"),
                        "time_exit_horizon": int(h),
                        "stop_pct": stop,
                        "count": int(len(x)),
                        "stop_breach_share": float(np.nanmean(stopped)),
                        "mean_net_with_stop": float(x.mean()),
                        "median_net_with_stop": float(x.median()),
                        "win_rate_with_stop": float((x > 0).mean()),
                        "profit_factor_with_stop": _profit_factor_raw(x),
                    }
                )
    return pd.DataFrame(rows)


def build_edge_registry(candidate_summary: pd.DataFrame, candidate_yearly: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    """Standardized edge-candidate ledger for later live-strategy review."""
    if candidate_summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for _, row in candidate_summary.iterrows():
        name = str(row.get("candidate_name", ""))
        if not name or str(row.get("bucket_type", "")) not in {"candidate", "union_bucket"}:
            continue
        h48_mean = float(row.get("h48_net_mean", np.nan))
        h48_median = float(row.get("h48_net_median", np.nan))
        h48_pf = float(row.get("h48_net_pf", np.nan))
        h48_wr = float(row.get("h48_net_win_rate", np.nan))
        h48_pos_years = int(row.get("h48_positive_years", 0) or 0)
        h96_mean = float(row.get("h96_net_mean", np.nan))
        h96_median = float(row.get("h96_net_median", np.nan))
        h96_pf = float(row.get("h96_net_pf", np.nan))
        h96_wr = float(row.get("h96_net_win_rate", np.nan))
        h96_pos_years = int(row.get("h96_positive_years", 0) or 0)
        count = int(row.get("count", 0) or 0)
        passes_h48 = count >= int(args.min_count) and h48_mean > 0 and h48_median > 0 and h48_pf >= float(args.min_profit_factor) and h48_wr >= float(args.min_win_rate) and h48_pos_years >= int(args.min_positive_years)
        passes_h96 = count >= int(args.min_count) and h96_mean > 0 and h96_median > 0 and h96_pf >= float(args.min_profit_factor) and h96_wr >= float(args.min_win_rate) and h96_pos_years >= int(args.min_positive_years)
        if passes_h48 and passes_h96:
            status = "edge_candidate_h48_h96"
        elif passes_h48:
            status = "edge_candidate_h48_only"
        elif passes_h96:
            status = "edge_candidate_h96_only"
        else:
            status = "watchlist_not_passed"
        rows.append(
            {
                "edge_name": name,
                "source_script": "low_sweep_panic_reversal_strategy_probe.py",
                "status": status,
                "direction": "LONG",
                "data_source": "OKX trade_bar",
                "timeframe": args.timeframe,
                "logic": row.get("description", ""),
                "filters": row.get("filters", ""),
                "edge_hypothesis": row.get("edge_hypothesis", ""),
                "count": count,
                "h48_mean": h48_mean,
                "h48_median": h48_median,
                "h48_win_rate": h48_wr,
                "h48_pf": h48_pf,
                "h48_positive_years": h48_pos_years,
                "h96_mean": h96_mean,
                "h96_median": h96_median,
                "h96_win_rate": h96_wr,
                "h96_pf": h96_pf,
                "h96_positive_years": h96_pos_years,
                "leakage_status": "no full-sample qcut; fixed thresholds or shift(1) rolling quantiles only",
                "strategy_readiness": "research_edge_only_not_live_ready",
                "next_step": "validate as independent strategy probes with time-exit 48/96, delay/cost stress, and conflict resolver",
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["status", "h48_mean", "count"], ascending=[True, False, False]).reset_index(drop=True)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def run_lab(args: argparse.Namespace) -> dict[str, object]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bars_all = load_trade_bars(args)
    features_all = build_enriched_features(bars_all, args)
    start_ts = pd.Timestamp(args.start_date)
    end_ts = pd.Timestamp(args.end_date)
    bars = bars_all.loc[(bars_all.index >= start_ts) & (bars_all.index <= end_ts)].copy()
    features = features_all.loc[(features_all.index >= start_ts) & (features_all.index <= end_ts)].copy()
    if bars.empty or features.empty:
        raise RuntimeError(f"No bars/features in research window {start_ts}->{end_ts}")

    print(f"[features] rows={len(features):,} range={features.index[0]} -> {features.index[-1]}", flush=True)
    events_raw = build_low_sweep_events(features, args)
    events_raw = attach_extra_features_to_events(events_raw, features)
    events_raw = add_filter_bins(events_raw)
    events_canonical = build_canonical_events(events_raw)
    events_canonical = add_filter_bins(events_canonical)
    print(f"[events] raw={len(events_raw):,} canonical={len(events_canonical):,}", flush=True)

    horizons = tuple(_parse_number_list(args.horizons, cast=int, name="horizons"))
    cfg = EventStudyConfig(
        horizons=horizons,
        mfe_mae_horizon=int(args.mfe_mae_horizon),
        entry_delay_bars=1,
        cost=CostConfig(
            entry_fee_rate=float(args.entry_fee_rate),
            exit_fee_rate=float(args.exit_fee_rate),
            entry_slippage_pct=float(args.entry_slippage_pct),
            exit_slippage_pct=float(args.exit_slippage_pct),
        ),
        min_count=int(args.min_count),
        progress_every=0 if bool(args.no_progress) else int(args.progress_every),
    )

    print("[study] canonical event study", flush=True)
    canonical_result = run_event_study(bars, events_canonical, cfg)
    studied = add_filter_bins(canonical_result.events)
    return_cols = [f"next_open_ret_h{int(h)}_net" for h in horizons]
    candidate_return_col = f"next_open_ret_h{int(args.candidate_horizon)}_net"
    if candidate_return_col not in studied.columns:
        raise RuntimeError(f"candidate horizon {args.candidate_horizon} not in horizons={horizons}")

    overview = summarize_many(studied, return_cols, min_count=int(args.min_count))
    yearly = summarize_many(studied, return_cols, group_cols=["year"], min_count=max(20, int(args.min_count) // 2))
    event_name_stats = summarize_many(studied, return_cols, group_cols=["event_name"], min_count=int(args.min_count))

    matrix = run_filter_matrix(studied, args, return_cols)
    top_filters = select_top_filters(matrix, args)
    top_yearly = summarize_top_filter_yearly(studied, top_filters, args, return_cols)
    tp_sl_rank = first_touch_strategy_rank(bars, studied, top_filters, args)
    delay_stress = run_stress_for_top_filters(bars, studied, top_filters.head(min(20, int(args.top_filter_count))), args, stress_type="delay")
    cost_stress = run_stress_for_top_filters(bars, studied, top_filters.head(min(20, int(args.top_filter_count))), args, stress_type="cost")

    fixed_candidate_summary = summarize_fixed_candidate_groups(studied, args, return_cols)
    fixed_candidate_yearly = summarize_fixed_candidate_yearly(studied, args, return_cols)
    fixed_candidate_overlap = fixed_candidate_overlap_matrix(studied)
    path_curve = path_forward_curve(bars, studied, args, fixed_candidate_summary)
    path_timing = path_timing_stats(bars, studied, args)
    path_stop = path_stop_sensitivity(bars, studied, args)
    edge_registry = build_edge_registry(fixed_candidate_summary, fixed_candidate_yearly, args)

    atomic_specs = build_filter_specs(studied)
    atomic_rows = []
    for spec in atomic_specs:
        mask = spec.mask_builder(studied).fillna(False).astype(bool)
        part = studied.loc[mask]
        if len(part) < max(20, int(args.min_count) // 2):
            continue
        s = summarize_many(part, [candidate_return_col], min_count=max(20, int(args.min_count) // 2))
        if not s.empty:
            rec = s.iloc[0].to_dict()
            rec.update({"filter_name": spec.name, "family": spec.family, "description": spec.description, "count_events": int(len(part))})
            atomic_rows.append(rec)
    atomic_filter_stats = pd.DataFrame(atomic_rows).sort_values(["mean", "count_events"], ascending=[False, False]) if atomic_rows else pd.DataFrame()

    write_csv(events_raw.head(int(args.save_event_sample)) if int(args.save_event_sample) > 0 else events_raw.iloc[0:0], out_dir / "01_events_raw_sample.csv")
    write_csv(studied, out_dir / "02_events_canonical_studied.csv")
    write_csv(overview, out_dir / "03_canonical_overview.csv")
    write_csv(yearly, out_dir / "04_canonical_yearly.csv")
    write_csv(event_name_stats, out_dir / "05_event_name_stats.csv")
    write_csv(canonical_result.causal_audit, out_dir / "06_causal_audit.csv")
    write_csv(atomic_filter_stats, out_dir / "07_atomic_filter_stats.csv")
    write_csv(matrix, out_dir / "08_filter_matrix_summary.csv")
    write_csv(top_filters, out_dir / "09_top_filters.csv")
    write_csv(top_yearly, out_dir / "10_top_filter_yearly.csv")
    write_csv(tp_sl_rank, out_dir / "11_first_touch_strategy_rank.csv")
    write_csv(delay_stress, out_dir / "12_delay_stress_top_filters.csv")
    write_csv(cost_stress, out_dir / "13_cost_stress_top_filters.csv")
    write_csv(fixed_candidate_summary, out_dir / "15_fixed_candidate_summary.csv")
    write_csv(fixed_candidate_yearly, out_dir / "16_fixed_candidate_yearly.csv")
    write_csv(fixed_candidate_overlap, out_dir / "17_fixed_candidate_overlap.csv")
    write_csv(path_curve, out_dir / "18_path_forward_curve.csv")
    write_csv(path_timing, out_dir / "19_path_timing_stats.csv")
    write_csv(path_stop, out_dir / "20_path_stop_sensitivity.csv")
    write_csv(edge_registry, out_dir / "21_edge_registry.csv")

    meta = {
        "script": "low_sweep_panic_reversal_strategy_probe.py",
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "warmup_start_date": args.warmup_start_date,
        "rows": int(len(bars)),
        "raw_events": int(len(events_raw)),
        "canonical_events": int(len(events_canonical)),
        "studied_events": int(len(studied)),
        "filter_specs": int(len(atomic_specs)),
        "filter_matrix_rows": int(len(matrix)),
        "leakage_guard": "Filter specs exclude full-sample qcut labels. Rolling quantile filters use feature.shift(1).rolling(...).quantile(...).",
        "train_threshold_mode": "not_implemented_in_this_patch",
        "top_filters": int(len(top_filters)),
        "fixed_candidate_rows": int(len(fixed_candidate_summary)),
        "edge_registry_rows": int(len(edge_registry)),
        "path_analysis_horizon": int(args.mfe_mae_horizon),
        "fixed_candidates": [
            "A_spike_close_large_share",
            "B_session_spike_atr",
            "C_session_extreme_spike",
            "ABC_union",
        ],
        "horizons": [int(h) for h in horizons],
        "candidate_horizon": int(args.candidate_horizon),
        "round_trip_cost_pct": _round_trip_cost(args, 1.0),
        "causal_fail_count": int(canonical_result.meta.get("causal_fail_count", 0)),
        "params": vars(args),
    }
    write_json(out_dir / "14_lab_meta.json", meta)
    print(f"[done] wrote reports -> {out_dir}", flush=True)
    return meta


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    run_lab(args)


if __name__ == "__main__":
    main()
