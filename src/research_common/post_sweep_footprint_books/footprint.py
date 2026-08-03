#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal Range-Footprint extraction for R07.

Only completed range bars whose ``end_ts`` is no later than an attempt's
``checkpoint_available_time`` are attached.  A footprint from a range bar that
still contains future trades is never used as a strategy-facing feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.data_feed.okx_range_bar_loader import OKXRangeBarLoader
from src.data_feed.okx_range_footprint_loader import OKXRangeFootprintLoader
from src.research_common.progress import ProgressReporter

from .config import PostSweepFootprintBooksConfig


RANGE_BAR_COLUMNS = (
    "bar_id",
    "start_ts",
    "end_ts",
    "duration_seconds",
    "open",
    "high",
    "low",
    "close",
    "direction",
    "notional",
    "buy_notional",
    "sell_notional",
    "delta_notional",
    "large_buy_notional",
    "large_sell_notional",
    "large_delta_notional",
    "max_trade_notional",
)

FOOTPRINT_COLUMNS = (
    "bar_id",
    "price_bucket",
    "notional",
    "trades_count",
    "buy_notional",
    "sell_notional",
    "delta_notional",
    "large_buy_notional",
    "large_sell_notional",
    "large_delta_notional",
    "max_trade_notional",
)

FOOTPRINT_FEATURE_COLUMNS = (
    "fp_bar_id",
    "fp_start_ts",
    "fp_end_ts",
    "fp_age_seconds",
    "fp_duration_seconds",
    "fp_direction",
    "fp_open",
    "fp_high",
    "fp_low",
    "fp_close",
    "fp_close_off_low_bp",
    "fp_total_notional",
    "fp_sell_notional",
    "fp_delta_ratio",
    "fp_sell_share",
    "fp_poc_off_low_bp",
    "fp_sell_poc_off_low_bp",
    "fp_bucket_count",
    "fp_low1_notional_share",
    "fp_low1_sell_share",
    "fp_low1_delta_ratio",
    "fp_low3_notional_share",
    "fp_low3_sell_share",
    "fp_low3_delta_ratio",
    "fp_low3_large_sell_share",
    "fp_low3_stacked_sell_bins",
    "fp_low3_max_trade_notional",
    "fp_low5_notional_share",
    "fp_low5_sell_share",
    "fp_low5_delta_ratio",
    "fp_new_low_extension_bp_vs_prev",
    "fp_downside_bp_per_sell_million",
    "fp_low3_downside_bp_per_sell_million",
    "fp_impact_ratio_vs_prev_down",
    "fp_low3_sell_vs_prev_down_ratio",
    "fp_low3_delta_improvement_vs_prev_down",
    "fp_poc_shift_vs_prev_down_bp",
    "fp_close_off_low_improvement_vs_prev_down_bp",
    "fp_prev_down_bar_id",
    "fp_lag1_bar_id",
    "fp_lag1_end_ts",
    "fp_lag1_direction",
    "fp_lag1_low3_sell_share",
    "fp_lag1_low3_delta_ratio",
    "fp_lag1_downside_bp_per_sell_million",
    "fp_lag2_bar_id",
    "fp_lag2_end_ts",
    "fp_causal_valid",
)


@dataclass(frozen=True)
class FootprintBuildResult:
    context: pd.DataFrame
    audit: pd.DataFrame


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = pd.to_numeric(denominator, errors="coerce")
    num = pd.to_numeric(numerator, errors="coerce")
    return num / den.where(den.abs() > 1e-12)


def _bp(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    return _safe_ratio(numerator, denominator) * 10_000.0


def _group_sums(frame: pd.DataFrame, columns: Iterable[str], prefix: str) -> pd.DataFrame:
    selected = [name for name in columns if name in frame.columns]
    if frame.empty:
        return pd.DataFrame(columns=[f"{prefix}{name}" for name in selected])
    out = frame.groupby("bar_id", sort=False, observed=True)[selected].sum(min_count=1)
    return out.rename(columns={name: f"{prefix}{name}" for name in selected})


def aggregate_footprint_bars(
    range_bars: pd.DataFrame,
    footprint_rows: pd.DataFrame,
    config: PostSweepFootprintBooksConfig,
) -> pd.DataFrame:
    """Aggregate price-bucket footprint rows once per completed range bar."""

    cfg = config.validate()
    if range_bars is None or range_bars.empty or footprint_rows is None or footprint_rows.empty:
        return pd.DataFrame()
    bars = range_bars.reset_index(drop=True).copy()
    for name in ("start_ts", "end_ts"):
        bars[name] = pd.to_datetime(bars[name], errors="coerce")
    bars["bar_id"] = pd.to_numeric(bars["bar_id"], errors="coerce").astype("Int64")
    bars = bars.dropna(subset=["bar_id", "end_ts", "low"]).copy()
    bars["bar_id"] = bars["bar_id"].astype("int64")
    bars = bars.sort_values(["end_ts", "bar_id"], kind="mergesort").drop_duplicates("bar_id", keep="last")

    fp = footprint_rows.copy()
    fp["bar_id"] = pd.to_numeric(fp["bar_id"], errors="coerce").astype("Int64")
    fp["price_bucket"] = pd.to_numeric(fp["price_bucket"], errors="coerce")
    numeric = [name for name in FOOTPRINT_COLUMNS if name not in {"bar_id", "price_bucket"} and name in fp.columns]
    for name in numeric:
        fp[name] = pd.to_numeric(fp[name], errors="coerce").fillna(0.0)
    fp = fp.dropna(subset=["bar_id", "price_bucket"]).copy()
    fp["bar_id"] = fp["bar_id"].astype("int64")
    fp = fp.merge(bars[["bar_id", "low"]], on="bar_id", how="inner", validate="many_to_one")
    if fp.empty:
        return pd.DataFrame()
    fp["bin_from_low"] = np.rint((fp["price_bucket"] - fp["low"]) / cfg.footprint_price_step).astype("int64")
    fp["bin_from_low"] = fp["bin_from_low"].clip(lower=0)

    total_cols = (
        "notional",
        "trades_count",
        "buy_notional",
        "sell_notional",
        "delta_notional",
        "large_buy_notional",
        "large_sell_notional",
        "large_delta_notional",
    )
    total = _group_sums(fp, total_cols, "fp_total_")
    total["fp_bucket_count"] = fp.groupby("bar_id", sort=False, observed=True).size().astype(float)
    total["fp_max_trade_notional"] = fp.groupby("bar_id", sort=False, observed=True)["max_trade_notional"].max()

    aggregates = [total]
    for bins in cfg.low_zone_bins:
        subset = fp.loc[fp["bin_from_low"] < int(bins)].copy()
        agg = _group_sums(subset, total_cols, f"fp_low{bins}_")
        agg[f"fp_low{bins}_max_trade_notional"] = subset.groupby("bar_id", sort=False, observed=True)["max_trade_notional"].max()
        stacked = subset.loc[
            (subset["sell_notional"] > 0)
            & (subset["sell_notional"] >= cfg.stacked_sell_ratio * subset["buy_notional"].clip(lower=1e-12))
        ]
        agg[f"fp_low{bins}_stacked_sell_bins"] = stacked.groupby("bar_id", sort=False, observed=True).size().astype(float)
        aggregates.append(agg)

    poc = (
        fp.sort_values(["bar_id", "notional", "price_bucket"], ascending=[True, False, True], kind="mergesort")
        .drop_duplicates("bar_id", keep="first")
        .set_index("bar_id")["price_bucket"]
        .rename("fp_poc_price")
    )
    sell_poc = (
        fp.sort_values(["bar_id", "sell_notional", "price_bucket"], ascending=[True, False, True], kind="mergesort")
        .drop_duplicates("bar_id", keep="first")
        .set_index("bar_id")["price_bucket"]
        .rename("fp_sell_poc_price")
    )
    features = bars.set_index("bar_id").join([*aggregates, poc, sell_poc], how="inner").reset_index()

    features = features.rename(
        columns={
            "bar_id": "fp_bar_id",
            "start_ts": "fp_start_ts",
            "end_ts": "fp_end_ts",
            "duration_seconds": "fp_duration_seconds",
            "direction": "fp_direction",
            "open": "fp_open",
            "high": "fp_high",
            "low": "fp_low",
            "close": "fp_close",
            "notional": "range_total_notional",
            "sell_notional": "range_sell_notional",
            "delta_notional": "range_delta_notional",
        }
    )
    features["fp_total_notional"] = pd.to_numeric(features["fp_total_notional"], errors="coerce")
    features["fp_sell_notional"] = pd.to_numeric(features["fp_total_sell_notional"], errors="coerce")
    features["fp_delta_ratio"] = _safe_ratio(features["fp_total_delta_notional"], features["fp_total_notional"])
    features["fp_sell_share"] = _safe_ratio(features["fp_sell_notional"], features["fp_total_notional"])
    features["fp_close_off_low_bp"] = _bp(features["fp_close"] - features["fp_low"], features["fp_low"])
    features["fp_poc_off_low_bp"] = _bp(features["fp_poc_price"] - features["fp_low"], features["fp_low"])
    features["fp_sell_poc_off_low_bp"] = _bp(features["fp_sell_poc_price"] - features["fp_low"], features["fp_low"])

    for bins in cfg.low_zone_bins:
        prefix = f"fp_low{bins}_"
        features[f"{prefix}notional_share"] = _safe_ratio(features[f"{prefix}notional"], features["fp_total_notional"])
        features[f"{prefix}sell_share"] = _safe_ratio(features[f"{prefix}sell_notional"], features[f"{prefix}notional"])
        features[f"{prefix}delta_ratio"] = _safe_ratio(features[f"{prefix}delta_notional"], features[f"{prefix}notional"])
        features[f"{prefix}large_sell_share"] = _safe_ratio(
            features[f"{prefix}large_sell_notional"], features[f"{prefix}sell_notional"]
        )
        features[f"{prefix}stacked_sell_bins"] = pd.to_numeric(
            features.get(f"{prefix}stacked_sell_bins"), errors="coerce"
        ).fillna(0.0)

    features = features.sort_values(["fp_end_ts", "fp_bar_id"], kind="mergesort").reset_index(drop=True)
    previous_low = pd.to_numeric(features["fp_low"], errors="coerce").shift(1)
    features["fp_new_low_extension_bp_vs_prev"] = _bp(
        (previous_low - features["fp_low"]).clip(lower=0.0), previous_low
    )
    features["fp_downside_bp_per_sell_million"] = _safe_ratio(
        features["fp_new_low_extension_bp_vs_prev"], features["fp_sell_notional"] / 1_000_000.0
    )
    features["fp_low3_downside_bp_per_sell_million"] = _safe_ratio(
        features["fp_new_low_extension_bp_vs_prev"], features["fp_low3_sell_notional"] / 1_000_000.0
    )

    down = pd.to_numeric(features["fp_direction"], errors="coerce") < 0
    down_source = features[[
        "fp_bar_id",
        "fp_downside_bp_per_sell_million",
        "fp_low3_sell_notional",
        "fp_low3_delta_ratio",
        "fp_poc_off_low_bp",
        "fp_close_off_low_bp",
    ]].where(down)
    prev_down = down_source.shift(1).ffill()
    features["fp_prev_down_bar_id"] = pd.to_numeric(prev_down["fp_bar_id"], errors="coerce")
    features["fp_impact_ratio_vs_prev_down"] = _safe_ratio(
        features["fp_downside_bp_per_sell_million"], prev_down["fp_downside_bp_per_sell_million"]
    )
    features["fp_low3_sell_vs_prev_down_ratio"] = _safe_ratio(
        features["fp_low3_sell_notional"], prev_down["fp_low3_sell_notional"]
    )
    features["fp_low3_delta_improvement_vs_prev_down"] = (
        pd.to_numeric(features["fp_low3_delta_ratio"], errors="coerce")
        - pd.to_numeric(prev_down["fp_low3_delta_ratio"], errors="coerce")
    )
    features["fp_poc_shift_vs_prev_down_bp"] = (
        pd.to_numeric(features["fp_poc_off_low_bp"], errors="coerce")
        - pd.to_numeric(prev_down["fp_poc_off_low_bp"], errors="coerce")
    )
    features["fp_close_off_low_improvement_vs_prev_down_bp"] = (
        pd.to_numeric(features["fp_close_off_low_bp"], errors="coerce")
        - pd.to_numeric(prev_down["fp_close_off_low_bp"], errors="coerce")
    )

    lag_source = (
        "fp_bar_id",
        "fp_end_ts",
        "fp_direction",
        "fp_low3_sell_share",
        "fp_low3_delta_ratio",
        "fp_downside_bp_per_sell_million",
    )
    for lag in range(1, cfg.footprint_lag_bars):
        for name in lag_source:
            features[f"fp_lag{lag}_{name.removeprefix('fp_')}"] = features[name].shift(lag)

    # Trim bulky intermediate columns.  Keep price-level sums needed for audits
    # in the on-disk full table, but expose a compact stable feature contract.
    desired = [name for name in FOOTPRINT_FEATURE_COLUMNS if name in features.columns]
    return features[desired].copy()


def attach_footprint_context(events: pd.DataFrame, bar_features: pd.DataFrame) -> pd.DataFrame:
    """Backward as-of attach of the latest completed footprint bar."""

    if events.empty:
        return events.copy()
    base = events.copy()
    base["checkpoint_available_time"] = pd.to_datetime(base["checkpoint_available_time"], errors="coerce")
    base["_event_order"] = np.arange(len(base), dtype=np.int64)
    if bar_features.empty:
        for name in FOOTPRINT_FEATURE_COLUMNS:
            if name not in base.columns:
                base[name] = np.nan
        base["fp_causal_valid"] = False
        return base.sort_values("_event_order").drop(columns="_event_order")
    right = bar_features.copy()
    right["fp_end_ts"] = pd.to_datetime(right["fp_end_ts"], errors="coerce")
    left = base.dropna(subset=["checkpoint_available_time"]).sort_values("checkpoint_available_time", kind="mergesort")
    right = right.dropna(subset=["fp_end_ts"]).sort_values("fp_end_ts", kind="mergesort")
    merged = pd.merge_asof(
        left,
        right,
        left_on="checkpoint_available_time",
        right_on="fp_end_ts",
        direction="backward",
        allow_exact_matches=True,
    )
    merged["fp_age_seconds"] = (
        merged["checkpoint_available_time"] - merged["fp_end_ts"]
    ).dt.total_seconds()
    merged["fp_causal_valid"] = (
        merged["fp_end_ts"].notna()
        & (merged["fp_end_ts"] <= merged["checkpoint_available_time"])
        & (merged["fp_age_seconds"] >= 0)
    )
    missing = base.loc[base["checkpoint_available_time"].isna()].copy()
    for name in right.columns:
        if name not in missing.columns:
            missing[name] = np.nan
    missing["fp_causal_valid"] = False
    out = merged if missing.empty else pd.concat([merged, missing], ignore_index=True, sort=False)
    return out.sort_values("_event_order", kind="mergesort").drop(columns="_event_order").reset_index(drop=True)


def _date_chunks(start: pd.Timestamp, end: pd.Timestamp, days: int):
    cursor = start.normalize()
    while cursor <= end:
        chunk_end = min(end, cursor + pd.Timedelta(days=days) - pd.Timedelta(microseconds=1))
        yield cursor, chunk_end
        cursor = chunk_end.normalize() + pd.Timedelta(days=1)


def build_footprint_context(
    events: pd.DataFrame,
    *,
    range_loader: OKXRangeBarLoader,
    footprint_loader: OKXRangeFootprintLoader,
    config: PostSweepFootprintBooksConfig,
    progress: bool = True,
) -> FootprintBuildResult:
    """Build causal footprint context in chronological chunks.

    The algorithm reads each range/footprint interval once, aggregates all price
    buckets once per bar, then uses vectorized as-of joins for every event in the
    interval.  No event-by-event SQLite query or Pandas resample is performed.
    """

    cfg = config.validate()
    if events.empty:
        return FootprintBuildResult(pd.DataFrame(), pd.DataFrame())
    source = events[["checkpoint_id", "checkpoint_available_time"]].drop_duplicates("checkpoint_id").copy()
    source["checkpoint_available_time"] = pd.to_datetime(source["checkpoint_available_time"], errors="coerce")
    source = source.dropna(subset=["checkpoint_available_time"]).sort_values("checkpoint_available_time", kind="mergesort")
    start = source["checkpoint_available_time"].min()
    end = source["checkpoint_available_time"].max()
    chunks = list(_date_chunks(start, end, cfg.footprint_chunk_days))
    reporter = ProgressReporter(
        label="[footprint] causal chunks",
        total=len(chunks),
        every=1,
        enabled=progress,
    )
    contexts: list[pd.DataFrame] = []
    audits: list[dict[str, object]] = []
    for index, (chunk_start, chunk_end) in enumerate(chunks, start=1):
        event_mask = source["checkpoint_available_time"].between(chunk_start, chunk_end, inclusive="both")
        chunk_events = source.loc[event_mask].copy()
        if chunk_events.empty:
            reporter.update(index)
            continue
        # Several completed bars of lead-in are needed for previous-down and lag
        # features.  Seven calendar days is far more than three ETH 0.20% bars
        # while keeping memory bounded.
        load_start = chunk_start - pd.Timedelta(days=7)
        load_end = chunk_end
        bars = range_loader.load_local_data(
            load_start,
            load_end,
            columns=RANGE_BAR_COLUMNS,
        )
        if bars.empty:
            attached = attach_footprint_context(chunk_events, pd.DataFrame())
            contexts.append(attached)
            audits.append(
                {
                    "chunk_start": chunk_start,
                    "chunk_end": chunk_end,
                    "events": len(chunk_events),
                    "range_bars": 0,
                    "footprint_rows": 0,
                    "aggregated_bars": 0,
                    "causal_attached": 0,
                    "status": "missing_range_bars",
                }
            )
            reporter.update(index)
            continue
        bar_id_min = int(pd.to_numeric(bars["bar_id"], errors="coerce").min())
        bar_id_max = int(pd.to_numeric(bars["bar_id"], errors="coerce").max())
        footprints = footprint_loader.load_local_data(
            bar_id_min=bar_id_min,
            bar_id_max=bar_id_max,
            columns=FOOTPRINT_COLUMNS,
        )
        bar_features = aggregate_footprint_bars(bars, footprints, cfg)
        attached = attach_footprint_context(chunk_events, bar_features)
        contexts.append(attached)
        audits.append(
            {
                "chunk_start": chunk_start,
                "chunk_end": chunk_end,
                "events": len(chunk_events),
                "range_bars": len(bars),
                "footprint_rows": len(footprints),
                "aggregated_bars": len(bar_features),
                "causal_attached": int(attached.get("fp_causal_valid", False).fillna(False).astype(bool).sum()),
                "bar_id_min": bar_id_min,
                "bar_id_max": bar_id_max,
                "status": "complete" if not footprints.empty else "missing_footprints",
            }
        )
        reporter.update(index)
    reporter.close()
    context = pd.concat(contexts, ignore_index=True) if contexts else pd.DataFrame()
    # The 7-day chunk overlap can return the same checkpoint only once because
    # event intervals are non-overlapping; assert rather than silently dedupe.
    if not context.empty and context["checkpoint_id"].duplicated().any():
        dup = context.loc[context["checkpoint_id"].duplicated(), "checkpoint_id"].astype(str).head().tolist()
        raise RuntimeError(f"duplicate footprint context checkpoints: {dup}")
    return FootprintBuildResult(context=context, audit=pd.DataFrame(audits))
