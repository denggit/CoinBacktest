#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal last-snapshot liquidity-wall detector.

The detector deliberately works on the same matrix a live operator sees:

* one column per chart bar;
* each historical column is the final valid order-book snapshot before the bar
  closes;
* the current live column is the latest snapshot and is overwritten as new
  books arrive;
* one row per price bin;
* cell value is depth divided by the causal rolling 24-hour robust high-depth
  reference.

A wall is therefore not cut independently from every 5-second snapshot.  At
bar ``t`` the detector looks only at the completed/latest columns up to ``t``
and asks whether a fixed price area has remained visibly deep across a rolling
history window.  A narrow persistent line becomes ``POINT``; several nearby
persistent deep rows, with limited light gaps, become ``MAIN``.

No later column can alter an earlier decision or earlier wall boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .aggregation import infer_heatmap_seconds
from .depth_scale import CausalDepthScaleConfig, attach_causal_depth_scale


@dataclass(frozen=True)
class PersistentWallConfig:
    """Parameters for the last-snapshot rolling-matrix detector."""

    reference_window_hours: float = 24.0
    reference_snapshot_quantile: float = 0.99

    # Colour/depth semantics shared with the heatmap.
    strong_depth_ratio: float = 0.50
    zone_depth_ratio: float = 0.30
    support_depth_ratio: float = 0.15
    isolated_point_ratio: float = 0.50

    # Rolling history used to decide whether rows remain visibly deep.
    lookback_hours: float = 4.0
    minimum_history_bars: int = 4
    minimum_support_time_coverage: float = 0.70
    minimum_zone_time_coverage: float = 0.55
    minimum_core_time_coverage: float = 0.20
    minimum_average_depth_ratio: float = 0.16
    minimum_current_depth_ratio: float = 0.12

    # A wall rectangle must be visibly filled across both time and price.  These
    # are matrix occupancies, not per-row maxima, so a box full of white holes
    # cannot pass merely because a few rows are intermittently deep.
    minimum_rectangle_support_occupancy: float = 0.65
    minimum_rectangle_zone_occupancy: float = 0.30
    minimum_current_support_occupancy: float = 0.55
    rectangle_price_persistence: float = 0.60

    # Spatial structure.
    minimum_zone_band_points: int = 3
    minimum_zone_support_points: int = 4
    minimum_zone_density_mass: float = 0.90
    minimum_zone_price_coverage: float = 0.70
    minimum_zone_strong_points: int = 1
    strongless_zone_min_band_points: int = 4
    point_minimum_zone_coverage: float = 0.55
    point_minimum_core_coverage: float = 0.20
    maximum_missing_price_bins: int = 1
    maximum_cluster_span_bins: int = 18
    # A visually stable wall may move by a few $1 bins between bar-end
    # snapshots. Historical persistence is evaluated on this local price
    # neighbourhood instead of requiring one exact price row forever.
    history_price_tolerance_bins: int = 2

    minimum_absolute_depth: float = 0.0
    minimum_zone_total_depth: float = 0.0
    maximum_distance_bps: float = 500.0
    minimum_market_clearance_bins: int = 1

    # Lifecycle is measured in chart bars, not 5-second frames.
    minimum_confirm_bars: int = 2
    persistent_after_minutes: int = 60
    major_after_minutes: int = 240
    maximum_missing_bars: int = 1
    minimum_match_overlap: float = 0.35
    maximum_center_drift_bins: int = 2
    boundary_smoothing_bars: int = 4

    minimum_strength_score: float = 35.0
    maximum_walls: int = 300

    # Compatibility arguments accepted from V2.4 and earlier. They are ignored
    # unless explicitly mapped by the plugin.
    minimum_confirm_seconds: int | None = None
    minimum_time_coverage: float | None = None
    maximum_missing_frames: int | None = None
    maximum_fade_minutes: float | None = None
    ghost_approach_distance_bps: float | None = None
    snapshot_depth_quantile: float | None = None
    minimum_side_max_ratio: float | None = None
    isolated_side_max_ratio: float | None = None
    minimum_cluster_points: int | None = None
    snapshot_depth_multiplier: float | None = None
    isolated_core_multiplier: float | None = None
    local_contrast_multiplier: float | None = None
    local_peak_radius_bins: int | None = None
    local_mean_radius_bins: int | None = None
    history_hours: float | None = None
    depth_percentile: float | None = None
    minimum_history_minutes: int | None = None
    minimum_duration_minutes: int | None = None
    minimum_band_local_ratio: float | None = None
    minimum_spatial_contrast: float | None = None
    band_reference_quantile: float | None = None
    weak_threshold_ratio: float | None = None
    strong_threshold_ratio: float | None = None
    minimum_segment_price_coverage: float | None = None
    minimum_strong_observation_ratio: float | None = None
    minimum_average_threshold_ratio: float | None = None
    robust_bounds_quantile: float | None = None
    main_wall_min_span_bins: int | None = None
    main_wall_min_unique_bins: int | None = None

    def validate(self) -> None:
        if self.reference_window_hours <= 0 or self.lookback_hours <= 0:
            raise ValueError("reference_window_hours and lookback_hours must be > 0")
        if not 0.5 <= self.reference_snapshot_quantile <= 1.0:
            raise ValueError("reference_snapshot_quantile must be in [0.5, 1.0]")
        if not 0 < self.support_depth_ratio <= self.zone_depth_ratio <= self.strong_depth_ratio <= 1:
            raise ValueError("wall ratios must satisfy 0 < support <= zone <= strong <= 1")
        if not self.strong_depth_ratio <= self.isolated_point_ratio <= 1:
            raise ValueError("isolated_point_ratio must be in [strong_depth_ratio, 1]")
        if self.minimum_history_bars < 2:
            raise ValueError("minimum_history_bars must be >= 2")
        for name, value in (
            ("minimum_support_time_coverage", self.minimum_support_time_coverage),
            ("minimum_zone_time_coverage", self.minimum_zone_time_coverage),
            ("minimum_core_time_coverage", self.minimum_core_time_coverage),
            ("minimum_zone_price_coverage", self.minimum_zone_price_coverage),
            ("point_minimum_zone_coverage", self.point_minimum_zone_coverage),
            ("point_minimum_core_coverage", self.point_minimum_core_coverage),
            ("minimum_match_overlap", self.minimum_match_overlap),
            ("minimum_rectangle_support_occupancy", self.minimum_rectangle_support_occupancy),
            ("minimum_rectangle_zone_occupancy", self.minimum_rectangle_zone_occupancy),
            ("minimum_current_support_occupancy", self.minimum_current_support_occupancy),
            ("rectangle_price_persistence", self.rectangle_price_persistence),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.minimum_average_depth_ratio < 0 or self.minimum_current_depth_ratio < 0:
            raise ValueError("depth-ratio floors must be >= 0")
        if self.minimum_zone_band_points < 2:
            raise ValueError("minimum_zone_band_points must be >= 2")
        if self.minimum_zone_support_points < self.minimum_zone_band_points:
            raise ValueError("minimum_zone_support_points must be >= minimum_zone_band_points")
        if self.maximum_missing_price_bins < 0 or self.maximum_cluster_span_bins <= 0:
            raise ValueError("price clustering parameters are invalid")
        if self.history_price_tolerance_bins < 0:
            raise ValueError("history_price_tolerance_bins must be >= 0")
        if self.maximum_distance_bps <= 0:
            raise ValueError("maximum_distance_bps must be > 0")
        if self.minimum_market_clearance_bins < 0:
            raise ValueError("minimum_market_clearance_bins must be >= 0")
        if self.minimum_confirm_bars <= 0 or self.maximum_missing_bars < 0:
            raise ValueError("bar lifecycle parameters are invalid")
        if self.maximum_center_drift_bins < 0 or self.boundary_smoothing_bars <= 0:
            raise ValueError("wall tracking parameters are invalid")
        if not 0 <= self.minimum_strength_score <= 100:
            raise ValueError("minimum_strength_score must be in [0, 100]")
        if self.maximum_walls <= 0:
            raise ValueError("maximum_walls must be > 0")


@dataclass(frozen=True)
class PersistentLiquidityWall:
    wall_id: int
    side: str
    wall_type: str
    first_seen_ms: int
    confirmed_at_ms: int
    last_seen_ms: int
    end_ms: int
    price_low: float
    price_high: float
    center_price: float
    duration_minutes: float
    confirmed_duration_minutes: float
    time_coverage: float
    price_coverage: float
    unique_price_bins: int
    span_price_bins: int
    average_depth: float
    peak_depth: float
    average_threshold_ratio: float
    peak_threshold_ratio: float
    maximum_observed_fade_minutes: float
    observations: int
    strength_score: float
    active_at_end: bool
    fields: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _Region:
    start_ms: int
    end_ms: int
    side: str
    wall_type: str
    price_low: float
    price_high: float
    center_price: float
    depth_sum: float
    peak_depth: float
    reference_depth: float
    peak_ratio: float
    mean_ratio: float
    density_mass: float
    price_coverage: float
    support_time_coverage: float
    zone_time_coverage: float
    core_time_coverage: float
    snapshot_mid: float
    distance_bps: float
    price_indices: tuple[int, ...]
    band_indices: tuple[int, ...]
    strong_indices: tuple[int, ...]


@dataclass(frozen=True)
class _Observation:
    start_ms: int
    end_ms: int
    wall_type: str
    price_low: float
    price_high: float
    center_price: float
    depth_sum: float
    peak_depth: float
    reference_depth: float
    peak_ratio: float
    mean_ratio: float
    density_mass: float
    price_coverage: float
    support_time_coverage: float
    zone_time_coverage: float
    core_time_coverage: float
    snapshot_mid: float
    distance_bps: float
    price_indices: tuple[int, ...]
    band_indices: tuple[int, ...]
    strong_indices: tuple[int, ...]


@dataclass
class _ActiveWall:
    wall_id: int
    side: str
    first_seen_ms: int
    last_seen_ms: int
    last_end_ms: int
    price_step: float
    bar_seconds: int
    observations: list[_Observation] = field(default_factory=list)
    missed_bars: int = 0
    maximum_missed_bars: int = 0
    confirmed_at_ms: int | None = None

    @classmethod
    def from_region(cls, wall_id: int, region: _Region, *, price_step: float, bar_seconds: int) -> "_ActiveWall":
        item = cls(
            wall_id=wall_id,
            side=region.side,
            first_seen_ms=region.start_ms,
            last_seen_ms=region.start_ms,
            last_end_ms=region.end_ms,
            price_step=price_step,
            bar_seconds=bar_seconds,
        )
        item.update(region, smoothing_bars=1)
        return item

    def update(self, region: _Region, *, smoothing_bars: int) -> None:
        # Causal boundary smoothing: only current and previous observations are
        # used.  This prevents one bar's edge noise from splitting a visual wall
        # while never backfilling a later range into an earlier bar.
        recent = self.observations[-max(0, smoothing_bars - 1):]
        lows = [item.price_low for item in recent] + [region.price_low]
        highs = [item.price_high for item in recent] + [region.price_high]
        centers = [item.center_price for item in recent] + [region.center_price]
        low = math.floor(float(np.quantile(lows, 0.25)) / self.price_step + 1e-12) * self.price_step
        high = math.ceil(float(np.quantile(highs, 0.75)) / self.price_step - 1e-12) * self.price_step
        if high <= low:
            high = low + self.price_step
        center = float(np.median(centers))
        self.last_seen_ms = region.start_ms
        self.last_end_ms = region.end_ms
        self.missed_bars = 0
        self.observations.append(
            _Observation(
                start_ms=region.start_ms,
                end_ms=region.end_ms,
                wall_type=region.wall_type,
                price_low=low,
                price_high=high,
                center_price=center,
                depth_sum=region.depth_sum,
                peak_depth=region.peak_depth,
                reference_depth=region.reference_depth,
                peak_ratio=region.peak_ratio,
                mean_ratio=region.mean_ratio,
                density_mass=region.density_mass,
                price_coverage=region.price_coverage,
                support_time_coverage=region.support_time_coverage,
                zone_time_coverage=region.zone_time_coverage,
                core_time_coverage=region.core_time_coverage,
                snapshot_mid=region.snapshot_mid,
                distance_bps=region.distance_bps,
                price_indices=region.price_indices,
                band_indices=region.band_indices,
                strong_indices=region.strong_indices,
            )
        )

    @property
    def observed_bars(self) -> int:
        return len(self.observations)

    @property
    def duration_minutes(self) -> float:
        return max(0.0, (self.last_end_ms - self.first_seen_ms) / 60_000.0)

    @property
    def time_coverage(self) -> float:
        expected = max(1, int(round((self.last_end_ms - self.first_seen_ms) / max(self.bar_seconds * 1000, 1))))
        return min(1.0, self.observed_bars / expected)

    def anchor(self, bars: int = 6) -> tuple[float, float, float]:
        recent = self.observations[-max(1, bars):]
        return (
            float(np.median([item.price_low for item in recent])),
            float(np.median([item.price_high for item in recent])),
            float(np.median([item.center_price for item in recent])),
        )


_REQUIRED = {"bucket_start_ms", "bucket_end_ms", "price_index", "side_code", "price_low", "price_high"}


def _prepare_frame(
    frame: pd.DataFrame,
    *,
    depth_column: str,
    config: PersistentWallConfig,
) -> tuple[pd.DataFrame, int, float]:
    missing = sorted(_REQUIRED.difference(frame.columns))
    if missing:
        raise ValueError(f"wall detector missing fields: {missing}")
    if depth_column not in frame.columns:
        raise ValueError(f"wall detector missing depth field: {depth_column}")

    out = frame.copy()
    for name in ("bucket_start_ms", "bucket_end_ms", "price_index", "side_code"):
        out[name] = pd.to_numeric(out[name], errors="coerce")
    for name in ("price_low", "price_high", depth_column):
        out[name] = pd.to_numeric(out[name], errors="coerce")
    out = out.dropna(subset=["bucket_start_ms", "bucket_end_ms", "price_index", "side_code", "price_low", "price_high"])
    out["bucket_start_ms"] = out["bucket_start_ms"].astype("int64")
    out["bucket_end_ms"] = out["bucket_end_ms"].astype("int64")
    out["price_index"] = out["price_index"].astype("int64")
    out["side_code"] = out["side_code"].astype("int8")
    out["side"] = out["side_code"].map({1: "bid", -1: "ask"}).fillna("unknown")
    out["depth"] = pd.to_numeric(out[depth_column], errors="coerce").fillna(0.0).clip(lower=0.0)
    out = out.loc[out["side"].isin(["bid", "ask"]) & (out["depth"] > 0)].copy()
    bar_seconds = infer_heatmap_seconds(frame, fallback=900)
    price_step = float(frame.attrs.get("price_step", 1.0))
    if out.empty:
        return out, bar_seconds, price_step

    # Analyze Tool may already have calculated the exact same causal colour
    # scale. Reuse it instead of normalising hundreds of thousands of cells a
    # second time. Standalone research/backtest callers still get the full
    # causal calculation here.
    has_precomputed = {"depth_ratio", "reference_depth"}.issubset(out.columns)
    if has_precomputed:
        out["depth_ratio"] = pd.to_numeric(out["depth_ratio"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        out["reference_depth"] = pd.to_numeric(out["reference_depth"], errors="coerce").fillna(1e-12).clip(lower=1e-12)
        if "snapshot_high_depth" not in out.columns:
            out["snapshot_high_depth"] = out["reference_depth"]
    else:
        out = attach_causal_depth_scale(
            out,
            depth_column="depth",
            config=CausalDepthScaleConfig(
                window_hours=config.reference_window_hours,
                snapshot_reference_quantile=config.reference_snapshot_quantile,
            ),
            ratio_column="depth_ratio",
            reference_column="reference_depth",
            snapshot_max_column="snapshot_high_depth",
        )
    best_bid = out.loc[out["side"] == "bid"].groupby("bucket_start_ms", observed=True)["price_high"].max()
    best_ask = out.loc[out["side"] == "ask"].groupby("bucket_start_ms", observed=True)["price_low"].min()
    midpoint = ((best_bid + best_ask) / 2.0).rename("snapshot_mid")
    out = out.drop(columns=["snapshot_mid"], errors="ignore")
    out = out.merge(midpoint, left_on="bucket_start_ms", right_index=True, how="left", validate="many_to_one")
    clearance = price_step * float(config.minimum_market_clearance_bins)
    proper_side = np.where(
        out["side"] == "bid",
        out["price_high"] <= out["snapshot_mid"] - clearance + 1e-12,
        out["price_low"] >= out["snapshot_mid"] + clearance - 1e-12,
    )
    out = out.loc[(out["snapshot_mid"] > 0) & proper_side].copy()
    out["distance_bps"] = np.where(
        out["side"] == "bid",
        (out["snapshot_mid"] - out["price_high"]) / out["snapshot_mid"] * 10_000.0,
        (out["price_low"] - out["snapshot_mid"]) / out["snapshot_mid"] * 10_000.0,
    )
    out = out.loc[
        (out["distance_bps"] >= 0.0)
        & (out["distance_bps"] <= config.maximum_distance_bps + 1e-12)
    ].copy()
    out = out.sort_values(["bucket_start_ms", "side_code", "price_index"]).reset_index(drop=True)
    return out, bar_seconds, price_step


def _split_indices(indices: Iterable[int], *, max_gap_bins: int, max_span_bins: int) -> list[list[int]]:
    ordered = sorted({int(value) for value in indices})
    if not ordered:
        return []
    groups: list[list[int]] = []
    current = [ordered[0]]
    for value in ordered[1:]:
        if value - current[-1] > max_gap_bins + 1 or value - current[0] + 1 > max_span_bins:
            groups.append(current)
            current = [value]
        else:
            current.append(value)
    groups.append(current)
    return groups


def _regions_for_side(
    side_frame: pd.DataFrame,
    *,
    all_times: np.ndarray,
    bar_seconds: int,
    price_step: float,
    config: PersistentWallConfig,
) -> dict[int, list[_Region]]:
    if side_frame.empty:
        return {}
    observed_prices = np.array(sorted(side_frame["price_index"].unique()), dtype=np.int64)
    if observed_prices.size == 0:
        return {}
    # Use a real contiguous price axis. Missing rows are zero depth, not an
    # invitation to collapse two distant areas into adjacent matrix columns.
    prices = np.arange(int(observed_prices.min()), int(observed_prices.max()) + 1, dtype=np.int64)
    ratio = (
        side_frame.pivot_table(index="bucket_start_ms", columns="price_index", values="depth_ratio", aggfunc="last")
        .reindex(index=all_times, columns=prices, fill_value=0.0)
        .fillna(0.0)
        .astype("float32")
    )
    depth = (
        side_frame.pivot_table(index="bucket_start_ms", columns="price_index", values="depth", aggfunc="last")
        .reindex(index=all_times, columns=prices, fill_value=0.0)
        .fillna(0.0)
        .astype("float32")
    )
    reference = (
        side_frame.groupby("bucket_start_ms", observed=True)["reference_depth"].median()
        .reindex(all_times)
        .ffill()
        .fillna(1e-12)
    )
    midpoint = (
        side_frame.groupby("bucket_start_ms", observed=True)["snapshot_mid"].median()
        .reindex(all_times)
        .ffill()
    )
    bucket_end = (
        side_frame.groupby("bucket_start_ms", observed=True)["bucket_end_ms"].max()
        .reindex(all_times)
        .fillna(pd.Series(all_times, index=all_times) + bar_seconds * 1000)
        .astype("int64")
    )

    lookback_bars = max(
        config.minimum_history_bars,
        int(math.ceil(config.lookback_hours * 3600.0 / max(bar_seconds, 1))),
    )
    min_periods = min(config.minimum_history_bars, lookback_bars)
    ratio_values = ratio.to_numpy(dtype=np.float32, copy=False)
    tolerance = int(config.history_price_tolerance_bins)
    if tolerance > 0:
        neighbourhood = np.zeros_like(ratio_values, dtype=np.float32)
        width = ratio_values.shape[1]
        for shift in range(-tolerance, tolerance + 1):
            source_low = max(0, -shift)
            source_high = min(width, width - shift)
            target_low = max(0, shift)
            target_high = min(width, width + shift)
            if source_high > source_low and target_high > target_low:
                np.maximum(
                    neighbourhood[:, target_low:target_high],
                    ratio_values[:, source_low:source_high],
                    out=neighbourhood[:, target_low:target_high],
                )
    else:
        neighbourhood = ratio_values
    neighbourhood_ratio = pd.DataFrame(neighbourhood, index=ratio.index, columns=ratio.columns)
    support_cov = (neighbourhood_ratio >= config.support_depth_ratio).rolling(lookback_bars, min_periods=min_periods).mean()
    zone_cov = (neighbourhood_ratio >= config.zone_depth_ratio).rolling(lookback_bars, min_periods=min_periods).mean()
    core_cov = (neighbourhood_ratio >= config.strong_depth_ratio).rolling(lookback_bars, min_periods=min_periods).mean()
    mean_ratio = neighbourhood_ratio.rolling(lookback_bars, min_periods=min_periods).mean()

    regions_by_time: dict[int, list[_Region]] = {}
    side = str(side_frame["side"].iloc[0])
    for row_number, timestamp in enumerate(all_times):
        if row_number + 1 < min_periods:
            continue
        current = ratio.iloc[row_number].to_numpy(dtype=float)
        current_neighbourhood = neighbourhood[row_number].astype(float, copy=False)
        historical_support = support_cov.iloc[row_number].to_numpy(dtype=float)
        historical_zone = zone_cov.iloc[row_number].to_numpy(dtype=float)
        historical_core = core_cov.iloc[row_number].to_numpy(dtype=float)
        historical_mean = mean_ratio.iloc[row_number].to_numpy(dtype=float)
        if not np.isfinite(historical_mean).any():
            continue

        active_support = (
            (historical_support + 1e-12 >= config.minimum_support_time_coverage)
            & (historical_mean + 1e-12 >= config.minimum_average_depth_ratio * 0.65)
            & (current + 1e-12 >= config.minimum_current_depth_ratio)
        )
        groups = _split_indices(
            prices[active_support],
            max_gap_bins=config.maximum_missing_price_bins,
            max_span_bins=config.maximum_cluster_span_bins,
        )
        row_regions: list[_Region] = []
        occupied: set[int] = set()
        price_to_pos = {int(value): index for index, value in enumerate(prices)}
        for group in groups:
            positions = np.array([price_to_pos[value] for value in group], dtype=int)
            low_index, high_index = min(group), max(group)
            span_positions = np.array(
                [price_to_pos[value] for value in prices if low_index <= int(value) <= high_index],
                dtype=int,
            )
            if span_positions.size == 0:
                continue
            zone_positions = positions[historical_zone[positions] + 1e-12 >= config.minimum_zone_time_coverage]
            core_positions = positions[historical_core[positions] + 1e-12 >= config.minimum_core_time_coverage]
            span_bins = high_index - low_index + 1
            price_coverage = len(group) / max(span_bins, 1)
            rectangle_support_occupancy = float(np.mean(historical_support[span_positions]))
            rectangle_zone_occupancy = float(np.mean(historical_zone[span_positions]))
            current_support_occupancy = float(
                np.mean(current[span_positions] + 1e-12 >= config.support_depth_ratio)
            )
            density_mass = float(np.sum(historical_mean[positions]))
            current_depth_sum = float(np.sum(depth.iloc[row_number].to_numpy(dtype=float)[positions]))
            peak_depth = float(np.max(depth.iloc[row_number].to_numpy(dtype=float)[positions]))
            peak_ratio = float(np.max(current[positions]))
            average_ratio = float(np.mean(historical_mean[positions]))
            support_time = float(np.mean(historical_support[positions]))
            zone_time = float(np.mean(historical_zone[positions]))
            core_time = float(np.mean(historical_core[positions]))

            is_point = (
                span_bins <= 2
                and len(zone_positions) >= 1
                and float(np.max(historical_zone[positions])) + 1e-12 >= config.point_minimum_zone_coverage
                and (
                    float(np.max(historical_core[positions])) + 1e-12 >= config.point_minimum_core_coverage
                    or peak_ratio + 1e-12 >= config.isolated_point_ratio
                )
            )
            is_main = (
                len(zone_positions) >= config.minimum_zone_band_points
                and len(group) >= config.minimum_zone_support_points
                and density_mass + 1e-12 >= config.minimum_zone_density_mass
                and price_coverage + 1e-12 >= config.minimum_zone_price_coverage
                and rectangle_support_occupancy + 1e-12 >= config.minimum_rectangle_support_occupancy
                and rectangle_zone_occupancy + 1e-12 >= config.minimum_rectangle_zone_occupancy
                and current_support_occupancy + 1e-12 >= config.minimum_current_support_occupancy
                and (
                    len(core_positions) >= config.minimum_zone_strong_points
                    or len(zone_positions) >= config.strongless_zone_min_band_points
                )
            )
            if current_depth_sum + 1e-12 < config.minimum_zone_total_depth:
                continue
            if peak_depth + 1e-12 < config.minimum_absolute_depth:
                continue
            wall_type = "MAIN" if is_main else "POINT" if is_point else ""
            if not wall_type:
                continue

            weights = np.maximum(historical_mean[positions], 1e-9)
            centers = prices[positions].astype(float) * price_step + price_step / 2.0
            center_price = float(np.average(centers, weights=weights))
            price_low = low_index * price_step
            price_high = (high_index + 1) * price_step
            mid = float(midpoint.loc[timestamp]) if pd.notna(midpoint.loc[timestamp]) else center_price
            clearance = price_step * float(config.minimum_market_clearance_bins)
            if side == "bid" and price_high > mid - clearance + 1e-12:
                continue
            if side == "ask" and price_low < mid + clearance - 1e-12:
                continue
            distance_bps = (
                (mid - price_high) / mid * 10_000.0
                if side == "bid" and mid > 0
                else (price_low - mid) / mid * 10_000.0
                if side == "ask" and mid > 0
                else math.inf
            )
            row_regions.append(
                _Region(
                    start_ms=int(timestamp),
                    end_ms=int(bucket_end.loc[timestamp]),
                    side=side,
                    wall_type=wall_type,
                    price_low=float(price_low),
                    price_high=float(price_high),
                    center_price=center_price,
                    depth_sum=current_depth_sum,
                    peak_depth=peak_depth,
                    reference_depth=float(reference.loc[timestamp]),
                    peak_ratio=peak_ratio,
                    mean_ratio=average_ratio,
                    density_mass=density_mass,
                    price_coverage=price_coverage,
                    support_time_coverage=support_time,
                    zone_time_coverage=zone_time,
                    core_time_coverage=core_time,
                    snapshot_mid=mid,
                    distance_bps=float(distance_bps),
                    price_indices=tuple(int(value) for value in group),
                    band_indices=tuple(int(prices[pos]) for pos in zone_positions),
                    strong_indices=tuple(int(prices[pos]) for pos in core_positions),
                )
            )
            occupied.update(group)

        # Persistent narrow lines can be hidden inside a larger support group
        # that failed the MAIN density test. Recover only truly persistent
        # individual rows and never duplicate a row already claimed by MAIN.
        point_mask = (
            (historical_zone + 1e-12 >= config.point_minimum_zone_coverage)
            & (historical_core + 1e-12 >= config.point_minimum_core_coverage)
            & (current + 1e-12 >= config.minimum_current_depth_ratio)
        )
        for position in np.flatnonzero(point_mask):
            price_index = int(prices[position])
            if price_index in occupied:
                continue
            current_depth = float(depth.iloc[row_number, position])
            if current_depth + 1e-12 < config.minimum_absolute_depth:
                continue
            price_low = price_index * price_step
            price_high = price_low + price_step
            mid = float(midpoint.loc[timestamp]) if pd.notna(midpoint.loc[timestamp]) else price_low + price_step / 2.0
            clearance = price_step * float(config.minimum_market_clearance_bins)
            if side == "bid" and price_high > mid - clearance + 1e-12:
                continue
            if side == "ask" and price_low < mid + clearance - 1e-12:
                continue
            distance_bps = (
                (mid - price_high) / mid * 10_000.0
                if side == "bid" and mid > 0
                else (price_low - mid) / mid * 10_000.0
                if side == "ask" and mid > 0
                else math.inf
            )
            row_regions.append(
                _Region(
                    start_ms=int(timestamp),
                    end_ms=int(bucket_end.loc[timestamp]),
                    side=side,
                    wall_type="POINT",
                    price_low=price_low,
                    price_high=price_high,
                    center_price=price_low + price_step / 2.0,
                    depth_sum=current_depth,
                    peak_depth=current_depth,
                    reference_depth=float(reference.loc[timestamp]),
                    peak_ratio=float(current[position]),
                    mean_ratio=float(historical_mean[position]),
                    density_mass=float(historical_mean[position]),
                    price_coverage=1.0,
                    support_time_coverage=float(historical_support[position]),
                    zone_time_coverage=float(historical_zone[position]),
                    core_time_coverage=float(historical_core[position]),
                    snapshot_mid=mid,
                    distance_bps=float(distance_bps),
                    price_indices=(price_index,),
                    band_indices=(price_index,),
                    strong_indices=(price_index,),
                )
            )
        if row_regions:
            regions_by_time[int(timestamp)] = sorted(row_regions, key=lambda item: (item.price_low, item.price_high))
    return regions_by_time


def _overlap_ratio(a_low: float, a_high: float, b_low: float, b_high: float) -> float:
    overlap = max(0.0, min(a_high, b_high) - max(a_low, b_low))
    denominator = max(min(a_high - a_low, b_high - b_low), 1e-12)
    return overlap / denominator


def _match_score(item: _ActiveWall, region: _Region, config: PersistentWallConfig) -> tuple[float, float] | None:
    if item.side != region.side or not item.observations:
        return None
    anchor_low, anchor_high, anchor_center = item.anchor()
    overlap = _overlap_ratio(anchor_low, anchor_high, region.price_low, region.price_high)
    center_bins = abs(anchor_center - region.center_price) / max(item.price_step, 1e-12)
    gap = max(0.0, max(anchor_low, region.price_low) - min(anchor_high, region.price_high)) / max(item.price_step, 1e-12)
    if overlap + 1e-12 < config.minimum_match_overlap and (
        center_bins > config.maximum_center_drift_bins + 1e-12
        or gap > config.maximum_missing_price_bins + 1e-12
    ):
        return None
    return (-overlap, center_bins + 0.25 * gap)


def _stage(item: _ActiveWall, observation: _Observation, config: PersistentWallConfig) -> str:
    if item.confirmed_at_ms is None:
        return "FORMING"
    age_minutes = max(0.0, (observation.end_ms - item.confirmed_at_ms) / 60_000.0)
    if age_minutes >= config.major_after_minutes:
        return "MAJOR"
    if age_minutes >= config.persistent_after_minutes:
        return "PERSISTENT"
    return "STABLE"


def _timeline(item: _ActiveWall, config: PersistentWallConfig) -> list[dict[str, Any]]:
    slices: list[dict[str, Any]] = []
    previous: _Observation | None = None
    for observation in item.observations:
        if previous is not None and observation.start_ms > previous.end_ms:
            slices.append({
                "start_ms": int(previous.end_ms),
                "end_ms": int(observation.start_ms),
                "status": "FADING",
                "wall_type": previous.wall_type,
                "price_low": float(previous.price_low),
                "price_high": float(previous.price_high),
                "center_price": float(previous.center_price),
                "depth_sum": 0.0,
                "peak_depth": 0.0,
                "reference_depth": float(previous.reference_depth),
                "peak_amount_ratio": 0.0,
                "mean_amount_ratio": 0.0,
                "density_mass": 0.0,
                "price_coverage": 0.0,
                "support_time_coverage": float(previous.support_time_coverage),
                "zone_time_coverage": float(previous.zone_time_coverage),
                "core_time_coverage": float(previous.core_time_coverage),
                "snapshot_mid": float(previous.snapshot_mid),
                "distance_bps": float(previous.distance_bps),
                "price_indices": [int(value) for value in previous.price_indices],
                "band_indices": [],
                "strong_indices": [],
                "median_depth": float(previous.reference_depth),
                "peak_snapshot_ratio": 0.0,
                "zone_snapshot_ratio": 0.0,
                "peak_local_contrast": 0.0,
                "core_indices": [],
            })
        payload = {
            "start_ms": int(observation.start_ms),
            "end_ms": int(observation.end_ms),
            "status": _stage(item, observation, config),
            "wall_type": observation.wall_type,
            "price_low": float(observation.price_low),
            "price_high": float(observation.price_high),
            "center_price": float(observation.center_price),
            "depth_sum": float(observation.depth_sum),
            "peak_depth": float(observation.peak_depth),
            "reference_depth": float(observation.reference_depth),
            "peak_amount_ratio": float(observation.peak_ratio),
            "mean_amount_ratio": float(observation.mean_ratio),
            "density_mass": float(observation.density_mass),
            "price_coverage": float(observation.price_coverage),
            "support_time_coverage": float(observation.support_time_coverage),
            "zone_time_coverage": float(observation.zone_time_coverage),
            "core_time_coverage": float(observation.core_time_coverage),
            "snapshot_mid": float(observation.snapshot_mid),
            "distance_bps": float(observation.distance_bps),
            "price_indices": [int(value) for value in observation.price_indices],
            "band_indices": [int(value) for value in observation.band_indices],
            "strong_indices": [int(value) for value in observation.strong_indices],
            "median_depth": float(observation.reference_depth),
            "peak_snapshot_ratio": float(observation.peak_ratio),
            "zone_snapshot_ratio": float(observation.density_mass),
            "peak_local_contrast": float(observation.mean_ratio),
            "core_indices": [int(value) for value in observation.band_indices],
        }
        if slices:
            last = slices[-1]
            same = (
                last["status"] == payload["status"]
                and last["wall_type"] == payload["wall_type"]
                and abs(float(last["price_low"]) - payload["price_low"]) <= item.price_step * 0.01
                and abs(float(last["price_high"]) - payload["price_high"]) <= item.price_step * 0.01
                and payload["start_ms"] <= int(last["end_ms"]) + item.bar_seconds * 1000
            )
            if same:
                last["end_ms"] = payload["end_ms"]
                last["depth_sum"] = max(float(last["depth_sum"]), payload["depth_sum"])
                last["peak_depth"] = max(float(last["peak_depth"]), payload["peak_depth"])
                last["peak_amount_ratio"] = max(float(last["peak_amount_ratio"]), payload["peak_amount_ratio"])
                last["density_mass"] = max(float(last["density_mass"]), payload["density_mass"])
                previous = observation
                continue
        slices.append(payload)
        previous = observation
    return slices


def _stable_rectangle_bounds(
    item: _ActiveWall,
    config: PersistentWallConfig,
) -> tuple[float, float, float, tuple[int, ...]]:
    """Return one fixed, persistent price rectangle for a wall lifecycle.

    Bounds are based only on observations already available in the lifecycle.
    A bin must recur in a configurable fraction of observations; this extracts
    the stable blocking core instead of drawing the union of every noisy edge.
    """

    observations = [
        observation
        for observation in item.observations
        if item.confirmed_at_ms is None or observation.end_ms >= item.confirmed_at_ms
    ] or list(item.observations)
    counts: dict[int, int] = {}
    for observation in observations:
        for value in set(observation.price_indices):
            counts[int(value)] = counts.get(int(value), 0) + 1
    required = max(1, int(math.ceil(len(observations) * config.rectangle_price_persistence - 1e-12)))
    persistent = sorted(value for value, count in counts.items() if count >= required)
    groups = _split_indices(
        persistent,
        max_gap_bins=min(1, config.maximum_missing_price_bins),
        max_span_bins=config.maximum_cluster_span_bins,
    )
    if groups:
        anchor_center = float(np.median([observation.center_price for observation in observations]))
        def group_score(group: list[int]) -> tuple[float, float, int]:
            mass = float(sum(counts.get(value, 0) for value in group))
            center = (min(group) + max(group) + 1) * item.price_step / 2.0
            return (mass, -abs(center - anchor_center), len(group))
        selected = max(groups, key=group_score)
        low = min(selected) * item.price_step
        high = (max(selected) + 1) * item.price_step
        indices = tuple(int(value) for value in selected)
    else:
        lows = [observation.price_low for observation in observations]
        highs = [observation.price_high for observation in observations]
        low = math.floor(float(np.median(lows)) / item.price_step + 1e-12) * item.price_step
        high = math.ceil(float(np.median(highs)) / item.price_step - 1e-12) * item.price_step
        if high <= low:
            high = low + item.price_step
        indices = tuple(range(int(round(low / item.price_step)), int(round(high / item.price_step))))

    clearance = item.price_step * float(config.minimum_market_clearance_bins)
    mids = [observation.snapshot_mid for observation in observations if observation.snapshot_mid > 0]
    if mids:
        if item.side == "bid":
            maximum_high = math.floor((min(mids) - clearance) / item.price_step + 1e-12) * item.price_step
            high = min(high, maximum_high)
        else:
            minimum_low = math.ceil((max(mids) + clearance) / item.price_step - 1e-12) * item.price_step
            low = max(low, minimum_low)
    if high <= low:
        current = observations[-1]
        low, high = current.price_low, current.price_high
    center = (low + high) / 2.0
    return float(low), float(high), float(center), indices


def _finalize(item: _ActiveWall, config: PersistentWallConfig, query_end_ms: int) -> PersistentLiquidityWall | None:
    if item.confirmed_at_ms is None or not item.observations:
        return None
    current = item.observations[-1]
    rectangle_low, rectangle_high, rectangle_center, rectangle_indices = _stable_rectangle_bounds(item, config)
    all_indices = {value for observation in item.observations for value in observation.price_indices}
    average_depth = float(np.mean([observation.depth_sum for observation in item.observations]))
    peak_depth = float(max(observation.peak_depth for observation in item.observations))
    average_ratio = float(np.mean([observation.mean_ratio for observation in item.observations]))
    peak_ratio = float(max(observation.peak_ratio for observation in item.observations))
    price_coverage = float(np.mean([observation.price_coverage for observation in item.observations]))
    support_coverage = float(np.mean([observation.support_time_coverage for observation in item.observations]))
    zone_coverage = float(np.mean([observation.zone_time_coverage for observation in item.observations]))
    duration_component = min(1.0, item.duration_minutes / max(config.major_after_minutes, 1))
    ratio_component = min(1.0, peak_ratio / max(config.strong_depth_ratio, 1e-12))
    continuity_component = min(1.0, support_coverage / max(config.minimum_support_time_coverage, 1e-12))
    zone_component = min(1.0, zone_coverage / max(config.minimum_zone_time_coverage, 1e-12))
    spatial_component = 0.35 if current.wall_type == "POINT" else min(
        1.0, price_coverage / max(config.minimum_zone_price_coverage, 1e-12)
    )
    strength = 100.0 * (
        0.25 * ratio_component
        + 0.30 * continuity_component
        + 0.20 * zone_component
        + 0.15 * spatial_component
        + 0.10 * duration_component
    )
    if strength + 1e-12 < config.minimum_strength_score:
        return None
    active_at_end = current.end_ms >= query_end_ms - (config.maximum_missing_bars + 1) * item.bar_seconds * 1000
    return PersistentLiquidityWall(
        wall_id=item.wall_id,
        side=item.side,
        wall_type=current.wall_type,
        first_seen_ms=item.first_seen_ms,
        confirmed_at_ms=int(item.confirmed_at_ms),
        last_seen_ms=item.last_seen_ms,
        end_ms=item.last_end_ms,
        price_low=rectangle_low,
        price_high=rectangle_high,
        center_price=rectangle_center,
        duration_minutes=item.duration_minutes,
        confirmed_duration_minutes=max(0.0, (item.last_end_ms - item.confirmed_at_ms) / 60_000.0),
        time_coverage=float(item.time_coverage),
        price_coverage=price_coverage,
        unique_price_bins=len(all_indices),
        span_price_bins=max(1, int(round((rectangle_high - rectangle_low) / item.price_step))),
        average_depth=average_depth,
        peak_depth=peak_depth,
        average_threshold_ratio=average_ratio,
        peak_threshold_ratio=peak_ratio,
        maximum_observed_fade_minutes=item.maximum_missed_bars * item.bar_seconds / 60.0,
        observations=item.observed_bars,
        strength_score=float(strength),
        active_at_end=active_at_end,
        fields={
            "detector_version": "v2_5_4_strict_rectangular_market_side",
            "input_semantics": "one final/latest order-book snapshot per chart bar",
            "lookback_hours": float(config.lookback_hours),
            "reference_window_hours": float(config.reference_window_hours),
            "ratio_semantics": "depth / causal rolling robust high-depth reference",
            "history_price_tolerance_bins": int(config.history_price_tolerance_bins),
            "rectangle_price_low": rectangle_low,
            "rectangle_price_high": rectangle_high,
            "rectangle_price_indices": [int(value) for value in rectangle_indices],
            "rectangle_price_persistence": float(config.rectangle_price_persistence),
            "minimum_market_clearance_bins": int(config.minimum_market_clearance_bins),
            "minimum_rectangle_support_occupancy": float(config.minimum_rectangle_support_occupancy),
            "minimum_rectangle_zone_occupancy": float(config.minimum_rectangle_zone_occupancy),
            "minimum_current_support_occupancy": float(config.minimum_current_support_occupancy),
            "lifecycle_stage": _stage(item, current, config),
            "lifecycle_status": "ACTIVE" if active_at_end else "ENDED",
            "timeline": _timeline(item, config),
        },
    )


def detect_persistent_liquidity_walls(
    frame: pd.DataFrame,
    *,
    depth_column: str = "end_depth_base",
    config: PersistentWallConfig | None = None,
) -> list[PersistentLiquidityWall]:
    """Detect persistent point/main walls from one latest snapshot per bar."""

    cfg = config or PersistentWallConfig()
    cfg.validate()
    if frame is None or frame.empty:
        return []
    prepared, bar_seconds, price_step = _prepare_frame(frame, depth_column=depth_column, config=cfg)
    if prepared.empty:
        return []

    all_times = np.array(sorted(prepared["bucket_start_ms"].unique()), dtype=np.int64)
    regions_by_time: dict[int, list[_Region]] = {int(value): [] for value in all_times}
    for side in ("bid", "ask"):
        side_regions = _regions_for_side(
            prepared.loc[prepared["side"] == side].copy(),
            all_times=all_times,
            bar_seconds=bar_seconds,
            price_step=price_step,
            config=cfg,
        )
        for timestamp, regions in side_regions.items():
            regions_by_time.setdefault(timestamp, []).extend(regions)

    active: list[_ActiveWall] = []
    completed: list[_ActiveWall] = []
    next_wall_id = 1
    end_by_time = prepared.groupby("bucket_start_ms", observed=True)["bucket_end_ms"].max().to_dict()
    midpoint_by_time = (
        prepared.groupby("bucket_start_ms", observed=True)["snapshot_mid"].median().to_dict()
    )
    for timestamp in all_times:
        current_mid = float(midpoint_by_time.get(int(timestamp), math.nan))
        clearance = price_step * float(cfg.minimum_market_clearance_bins)
        still_active: list[_ActiveWall] = []
        for item in active:
            anchor_low, anchor_high, _ = item.anchor()
            market_entered = bool(
                math.isfinite(current_mid)
                and (
                    (item.side == "bid" and current_mid < anchor_high + clearance - 1e-12)
                    or (item.side == "ask" and current_mid > anchor_low - clearance + 1e-12)
                )
            )
            if market_entered:
                completed.append(item)
            else:
                still_active.append(item)
        active = still_active

        regions = regions_by_time.get(int(timestamp), [])
        candidates: list[tuple[tuple[float, float], int, int]] = []
        for active_index, item in enumerate(active):
            for region_index, region in enumerate(regions):
                score = _match_score(item, region, cfg)
                if score is not None:
                    candidates.append((score, active_index, region_index))
        candidates.sort(key=lambda item: item[0])
        matched_active: set[int] = set()
        matched_regions: set[int] = set()
        for _, active_index, region_index in candidates:
            if active_index in matched_active or region_index in matched_regions:
                continue
            item = active[active_index]
            item.update(regions[region_index], smoothing_bars=cfg.boundary_smoothing_bars)
            if item.confirmed_at_ms is None and item.observed_bars >= cfg.minimum_confirm_bars:
                item.confirmed_at_ms = item.last_end_ms
            matched_active.add(active_index)
            matched_regions.add(region_index)

        survivors: list[_ActiveWall] = []
        for active_index, item in enumerate(active):
            if active_index not in matched_active:
                item.missed_bars += 1
                item.maximum_missed_bars = max(item.maximum_missed_bars, item.missed_bars)
            if item.missed_bars > cfg.maximum_missing_bars:
                completed.append(item)
            else:
                survivors.append(item)
        active = survivors

        for region_index, region in enumerate(regions):
            if region_index in matched_regions:
                continue
            item = _ActiveWall.from_region(next_wall_id, region, price_step=price_step, bar_seconds=bar_seconds)
            if item.observed_bars >= cfg.minimum_confirm_bars:
                item.confirmed_at_ms = item.last_end_ms
            active.append(item)
            next_wall_id += 1

    completed.extend(active)
    query_end_ms = int(max(end_by_time.values()))
    walls = [wall for item in completed if (wall := _finalize(item, cfg, query_end_ms)) is not None]
    walls.sort(key=lambda wall: (wall.confirmed_at_ms, wall.side, wall.price_low, wall.wall_id))
    if len(walls) > cfg.maximum_walls:
        strongest = sorted(walls, key=lambda wall: (wall.strength_score, wall.duration_minutes), reverse=True)[: cfg.maximum_walls]
        walls = sorted(strongest, key=lambda wall: (wall.confirmed_at_ms, wall.wall_id))
    return walls


__all__ = ["PersistentWallConfig", "PersistentLiquidityWall", "detect_persistent_liquidity_walls"]
