#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal Range-Bar, Books and Footprint feature construction."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from .models import LiquidityHuntConfig

EPS = 1e-12


def _numeric(frame: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[name], errors="coerce")


def _safe_divide(num: pd.Series | np.ndarray, den: pd.Series | np.ndarray) -> np.ndarray:
    n = np.asarray(num, dtype=float)
    d = np.asarray(den, dtype=float)
    out = np.full(np.broadcast_shapes(n.shape, d.shape), np.nan, dtype=float)
    np.divide(n, d, out=out, where=np.isfinite(d) & (np.abs(d) > EPS))
    return out


def _ensure_range_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "bar_id",
        "start_ts",
        "end_ts",
        "open",
        "high",
        "low",
        "close",
        "direction",
        "notional",
        "buy_notional",
        "sell_notional",
        "taker_buy_ratio",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"range bars missing required columns: {missing}")
    # OKXRangeBarLoader intentionally returns ``end_ts`` both as the
    # index and as a column.  Treat loader indexes as access conveniences only;
    # otherwise pandas operations on the explicit ``end_ts`` column are
    # ambiguous.  Normalize every external range frame to a plain RangeIndex.
    out = frame.reset_index(drop=True).copy()
    out["start_ts"] = pd.to_datetime(out["start_ts"], errors="coerce")
    out["end_ts"] = pd.to_datetime(out["end_ts"], errors="coerce")
    out = out.dropna(subset=["start_ts", "end_ts"]).sort_values(["end_ts", "bar_id"]).reset_index(drop=True)
    if out["end_ts"].duplicated().any():
        out = out.sort_values(["end_ts", "bar_id"], kind="stable").reset_index(drop=True)
    return out


def build_range_features(frame: pd.DataFrame, cfg: LiquidityHuntConfig) -> pd.DataFrame:
    """Build causal range-bar features from completed bars only."""

    cfg.validate()
    out = _ensure_range_frame(frame)
    for col in (
        "open",
        "high",
        "low",
        "close",
        "direction",
        "notional",
        "buy_notional",
        "sell_notional",
        "taker_buy_ratio",
        "duration_seconds",
        "volume",
    ):
        out[col] = _numeric(out, col)

    out["signal_time"] = out["end_ts"]
    out["buy_ratio"] = out["taker_buy_ratio"].clip(0.0, 1.0)
    out["sell_ratio"] = 1.0 - out["buy_ratio"]
    out["bar_return"] = _safe_divide(out["close"], out["open"]) - 1.0

    prior_notional = out["notional"].shift(1).rolling(
        int(cfg.notional_median_bars),
        min_periods=int(cfg.notional_min_periods),
    ).median()
    out["prior_notional_median"] = prior_notional
    out["notional_multiple"] = _safe_divide(out["notional"], prior_notional)

    lookback = int(cfg.support_lookback_bars)
    out["prior_support_low"] = out["low"].shift(1).rolling(lookback, min_periods=lookback).min()
    out["prior_resistance_high"] = out["high"].shift(1).rolling(lookback, min_periods=lookback).max()

    # Previous completed range bar.  The pair i-1 -> i is the causal event
    # sequence; no values from i+1 are used in the signal.
    previous_columns = [
        "bar_id",
        "start_ts",
        "end_ts",
        "open",
        "high",
        "low",
        "close",
        "direction",
        "notional",
        "notional_multiple",
        "buy_ratio",
        "sell_ratio",
        "prior_support_low",
        "prior_resistance_high",
        "volume",
    ]
    for col in previous_columns:
        out[f"prev_{col}"] = out[col].shift(1)

    out["reclaim_to_attack_notional_ratio"] = _safe_divide(out["notional"], out["prev_notional"])
    out["reclaim_to_attack_volume_ratio"] = _safe_divide(out["volume"], out["prev_volume"])
    return out


def prepare_book_features(frame: pd.DataFrame, cfg: LiquidityHuntConfig) -> pd.DataFrame:
    """Build trailing 1-second liquidity features with causal windows."""

    if frame is None or frame.empty:
        return pd.DataFrame()
    if "available_time" not in frame.columns:
        raise ValueError("liquidity features require available_time")
    # The causal key is the explicit ``available_time`` column.  Rebuild the
    # index below after validation instead of trusting a loader-provided index.
    out = frame.reset_index(drop=True).copy()
    out["available_time"] = pd.to_datetime(out["available_time"], errors="coerce")
    out = out.dropna(subset=["available_time"]).sort_values("available_time").drop_duplicates(
        "available_time", keep="last"
    )
    out = out.set_index("available_time", drop=False)

    bid5 = _numeric(out, "bid_depth_5bps_base", 0.0).fillna(0.0)
    ask5 = _numeric(out, "ask_depth_5bps_base", 0.0).fillna(0.0)
    bid25 = _numeric(out, "bid_depth_25bps_base", 0.0).fillna(0.0)
    ask25 = _numeric(out, "ask_depth_25bps_base", 0.0).fillna(0.0)
    out["obi_5bps"] = _safe_divide(bid5 - ask5, bid5 + ask5)
    out["obi_25bps"] = _safe_divide(bid25 - ask25, bid25 + ask25)

    window = f"{int(cfg.flow_window_seconds)}s"
    min_periods = max(1, int(cfg.flow_window_seconds) // 2)
    out["obi_5s"] = out["obi_5bps"].rolling(window, min_periods=min_periods).mean()
    out["obi_25s"] = out["obi_25bps"].rolling(window, min_periods=min_periods).mean()
    out["obi_5s_min"] = out["obi_5s"].rolling(window, min_periods=1).min()
    out["obi_5s_max"] = out["obi_5s"].rolling(window, min_periods=1).max()
    out["obi_5s_jump"] = out["obi_5s"] - out["obi_5s"].shift(int(cfg.flow_window_seconds))

    ref_window = f"{int(cfg.book_reference_minutes)}min"
    # Shift first so the baseline never contains the current second.
    ask_ref = ask25.shift(1).rolling(ref_window, min_periods=max(10, cfg.book_reference_minutes * 6)).median()
    bid_ref = bid25.shift(1).rolling(ref_window, min_periods=max(10, cfg.book_reference_minutes * 6)).median()
    out["ask_depth_25bps_ref_ratio"] = _safe_divide(ask25, ask_ref)
    out["bid_depth_25bps_ref_ratio"] = _safe_divide(bid25, bid_ref)
    out["ask_to_bid_depth_25bps"] = _safe_divide(ask25, bid25)
    out["bid_to_ask_depth_25bps"] = _safe_divide(bid25, ask25)

    flow_columns = (
        "aggressive_buy_base",
        "aggressive_sell_base",
        "book_added_bid_base",
        "book_added_ask_base",
        "book_removed_bid_base",
        "book_removed_ask_base",
        "estimated_bid_cancel_base",
        "estimated_ask_cancel_base",
        "estimated_bid_consumed_base",
        "estimated_ask_consumed_base",
        "estimated_bid_replenished_base",
        "estimated_ask_replenished_base",
    )
    for col in flow_columns:
        values = _numeric(out, col, 0.0).fillna(0.0)
        out[f"{col}_{cfg.flow_window_seconds}s"] = values.rolling(window, min_periods=1).sum()

    bid_consume = _numeric(out, f"estimated_bid_consumed_base_{cfg.flow_window_seconds}s", 0.0)
    bid_replenish = _numeric(out, f"estimated_bid_replenished_base_{cfg.flow_window_seconds}s", 0.0)
    ask_consume = _numeric(out, f"estimated_ask_consumed_base_{cfg.flow_window_seconds}s", 0.0)
    ask_replenish = _numeric(out, f"estimated_ask_replenished_base_{cfg.flow_window_seconds}s", 0.0)
    out["bid_replenish_to_consume"] = _safe_divide(bid_replenish, bid_consume)
    out["ask_replenish_to_consume"] = _safe_divide(ask_replenish, ask_consume)

    out["book_valid"] = _numeric(out, "book_valid", 0.0).fillna(0.0)
    out["trade_attribution_valid"] = _numeric(out, "trade_attribution_valid", 0.0).fillna(0.0)
    return out


BOOK_CONTEXT_COLUMNS = (
    "available_time",
    "book_valid",
    "trade_attribution_valid",
    "best_bid",
    "best_ask",
    "mid_price",
    "spread_bps",
    "bid_depth_5bps_base",
    "ask_depth_5bps_base",
    "bid_depth_25bps_base",
    "ask_depth_25bps_base",
    "obi_5bps",
    "obi_25bps",
    "obi_5s",
    "obi_25s",
    "obi_5s_min",
    "obi_5s_max",
    "obi_5s_jump",
    "ask_depth_25bps_ref_ratio",
    "bid_depth_25bps_ref_ratio",
    "ask_to_bid_depth_25bps",
    "bid_to_ask_depth_25bps",
    "nearest_large_bid_price",
    "nearest_large_ask_price",
    "nearest_large_bid_depth_base",
    "nearest_large_ask_depth_base",
    "top_bid_wall_price",
    "top_ask_wall_price",
    "top_bid_wall_depth_base",
    "top_ask_wall_depth_base",
    "book_added_bid_base_5s",
    "book_added_ask_base_5s",
    "book_removed_bid_base_5s",
    "book_removed_ask_base_5s",
    "estimated_bid_cancel_base_5s",
    "estimated_ask_cancel_base_5s",
    "estimated_bid_consumed_base_5s",
    "estimated_ask_consumed_base_5s",
    "estimated_bid_replenished_base_5s",
    "estimated_ask_replenished_base_5s",
    "bid_replenish_to_consume",
    "ask_replenish_to_consume",
)


def datetime_index_to_ns_int64(
    values: Sequence[pd.Timestamp] | pd.Series | pd.DatetimeIndex,
) -> np.ndarray:
    """Return datetime values as nanosecond int64 across pandas resolutions.

    Pandas 2+ may preserve ``datetime64[us]`` values returned by SQLite.
    ``DatetimeIndex.view("int64")`` then yields microseconds, while
    ``Timestamp.value`` and ``Timedelta.value`` are nanoseconds.  Mixing those
    units silently breaks ``searchsorted``.  Normalize explicitly to ns first.
    """

    index = pd.DatetimeIndex(pd.to_datetime(values, errors="coerce"))
    return index.to_numpy(dtype="datetime64[ns]").view("int64")


def align_book_features_to_times(
    times: Sequence[pd.Timestamp] | pd.Series | pd.DatetimeIndex,
    prepared_book: pd.DataFrame,
    *,
    tolerance: pd.Timedelta = pd.Timedelta(seconds=10),
) -> pd.DataFrame:
    """Align the latest available liquidity row to decision timestamps."""

    query = pd.DatetimeIndex(pd.to_datetime(times))
    result = pd.DataFrame(index=np.arange(len(query)))
    if len(query) == 0:
        return result
    if prepared_book is None or prepared_book.empty:
        for col in BOOK_CONTEXT_COLUMNS:
            if col == "available_time":
                result[f"book_{col}"] = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")
            else:
                result[f"book_{col}"] = np.nan
        result["book_context_missing_flag"] = True
        result["book_available_after_signal_flag"] = False
        return result

    source = prepared_book.sort_index()
    source_ns = datetime_index_to_ns_int64(source.index)
    query_ns = datetime_index_to_ns_int64(query)
    positions = np.searchsorted(source_ns, query_ns, side="right") - 1
    valid = positions >= 0
    clipped = np.clip(positions, 0, max(len(source_ns) - 1, 0))
    tolerance_ns = int(pd.Timedelta(tolerance).value)
    valid &= (query_ns - source_ns[clipped]) <= tolerance_ns

    for col in BOOK_CONTEXT_COLUMNS:
        if col == "available_time":
            values = pd.Series(pd.NaT, index=result.index, dtype="datetime64[ns]")
            if col in source.columns and valid.any():
                selected = pd.to_datetime(source.iloc[clipped[valid]][col], errors="coerce").to_numpy()
                values.iloc[np.flatnonzero(valid)] = selected
            result[f"book_{col}"] = values
            continue
        values = np.full(len(query), np.nan, dtype=float)
        if col in source.columns and valid.any():
            selected = pd.to_numeric(source.iloc[clipped[valid]][col], errors="coerce").to_numpy(dtype=float)
            values[valid] = selected
        result[f"book_{col}"] = values
    result["book_context_missing_flag"] = ~valid
    result["book_available_after_signal_flag"] = False
    if "book_available_time" in result:
        available = pd.to_datetime(result["book_available_time"], errors="coerce")
        result["book_available_after_signal_flag"] = available > pd.Series(query)
    return result


def attach_book_context(
    range_bars: pd.DataFrame,
    prepared_book: pd.DataFrame,
    *,
    tolerance: pd.Timedelta = pd.Timedelta(seconds=10),
) -> pd.DataFrame:
    out = range_bars.reset_index(drop=True).copy()
    aligned = align_book_features_to_times(out["end_ts"], prepared_book, tolerance=tolerance)
    aligned.index = out.index
    for col in aligned.columns:
        out[col] = aligned[col]
    return out


def aggregate_footprint_features(footprints: pd.DataFrame) -> pd.DataFrame:
    """Aggregate price-bucket footprints to one compact row per range bar."""

    if footprints is None or footprints.empty:
        return pd.DataFrame(columns=["bar_id"])
    required = {"bar_id", "price_bucket", "notional", "delta_notional"}
    missing = sorted(required - set(footprints.columns))
    if missing:
        raise ValueError(f"footprint missing required columns: {missing}")
    # Normalize at the input boundary so a future loader index named
    # ``bar_id`` cannot become ambiguous during groupby/merge.
    fp = footprints.reset_index(drop=True).copy()
    for col in (
        "price_bucket",
        "notional",
        "delta_notional",
        "buy_notional",
        "sell_notional",
        "large_delta_notional",
        "max_trade_notional",
    ):
        fp[col] = _numeric(fp, col, 0.0).fillna(0.0)
    fp["bar_id"] = pd.to_numeric(fp["bar_id"], errors="coerce").fillna(-1).astype("int64")
    grouped = fp.groupby("bar_id", sort=False, observed=True)
    low = grouped["price_bucket"].transform("min")
    high = grouped["price_bucket"].transform("max")
    span = (high - low).replace(0.0, np.nan)
    fp["bucket_pos"] = ((fp["price_bucket"] - low) / span).fillna(0.5)
    fp["low_zone"] = fp["bucket_pos"] <= 0.25
    fp["high_zone"] = fp["bucket_pos"] >= 0.75

    def zone_sum(mask: pd.Series, column: str) -> pd.Series:
        return fp.loc[mask].groupby("bar_id", observed=True)[column].sum()

    summary = grouped.agg(
        fp_total_notional=("notional", "sum"),
        fp_delta_notional=("delta_notional", "sum"),
        fp_bucket_count=("price_bucket", "size"),
        fp_max_trade_notional=("max_trade_notional", "max"),
        fp_large_delta_notional=("large_delta_notional", "sum"),
    )
    low_notional = zone_sum(fp["low_zone"], "notional")
    low_delta = zone_sum(fp["low_zone"], "delta_notional")
    high_notional = zone_sum(fp["high_zone"], "notional")
    high_delta = zone_sum(fp["high_zone"], "delta_notional")
    summary["fp_low_zone_delta_ratio"] = _safe_divide(
        low_delta.reindex(summary.index).fillna(0.0),
        low_notional.reindex(summary.index).fillna(0.0),
    )
    summary["fp_high_zone_delta_ratio"] = _safe_divide(
        high_delta.reindex(summary.index).fillna(0.0),
        high_notional.reindex(summary.index).fillna(0.0),
    )
    summary["fp_delta_ratio"] = _safe_divide(summary["fp_delta_notional"], summary["fp_total_notional"])
    summary = summary.reset_index()
    return summary


def attach_footprint_features(range_bars: pd.DataFrame, footprints: pd.DataFrame) -> pd.DataFrame:
    out = range_bars.copy()
    compact = aggregate_footprint_features(footprints)
    if compact.empty:
        for col in (
            "fp_total_notional",
            "fp_delta_notional",
            "fp_bucket_count",
            "fp_max_trade_notional",
            "fp_large_delta_notional",
            "fp_low_zone_delta_ratio",
            "fp_high_zone_delta_ratio",
            "fp_delta_ratio",
        ):
            out[col] = np.nan
        out["footprint_missing_flag"] = True
        return out
    out = out.merge(compact, on="bar_id", how="left", validate="one_to_one")
    out["footprint_missing_flag"] = out["fp_total_notional"].isna()
    return out
