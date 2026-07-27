#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal liquidity-wall discovery and touch-environment event research.

This module deliberately does **not** treat the current Analyze Tool wall
rectangles as ground truth.  It starts from the canonical order-book heatmap
snapshots and produces a broad, multi-scale candidate vocabulary.  Future price
movement is attached only after candidates and lifecycles are frozen.

Causal contract
---------------
* ``end_depth_base`` is the exact reconstructed book state at a source bucket's
  end and becomes available at ``bucket_end_ms``.
* A wall state available at time ``t`` may first be touched by the execution bar
  whose start time is ``>= t``.  A state from the same bar after the touch is
  never used to explain that touch.
* Candidate extraction and lifecycle tracking use current/past book data only.
* Bounce/break labels are outcomes, never inputs to wall discovery.
* Train quantile cut points are reused unchanged on holdout.

The output is intentionally continuous rather than an early wall/not-wall
binary.  Point concentration, wide-band mass, persistence, drift, fading,
withdrawal, consumption and replenishment remain separate features so later
research can learn which combinations truly behave like support/resistance.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from src.data_feed.okx_liquidity_primitives import LiquidityPrimitiveSnapshot

try:
    from config.loader import TIMEZONE
except ImportError:  # pragma: no cover
    TIMEZONE = "+8"


def _project_timezone_offset() -> pd.Timedelta:
    text = str(TIMEZONE).strip()
    if text.startswith("+"):
        return pd.Timedelta(hours=float(text[1:] or 0))
    if text.startswith("-"):
        return -pd.Timedelta(hours=float(text[1:] or 0))
    return pd.Timedelta(0)


def _index_to_utc_ms(index: pd.DatetimeIndex) -> np.ndarray:
    if index.tz is not None:
        utc = index.tz_convert("UTC")
        return (utc.view("int64") // 1_000_000).astype("int64")
    utc_naive = index - _project_timezone_offset()
    return (utc_naive.view("int64") // 1_000_000).astype("int64")


@dataclass(frozen=True)
class WallDiscoveryConfig:
    price_step: float = 1.0
    reference_window_hours: float = 24.0
    snapshot_reference_quantile: float = 0.99
    side_baseline_quantile: float = 0.50
    candidate_widths: tuple[int, ...] = (1, 3, 5, 8, 13)
    support_multiple: float = 1.50
    thick_multiple: float = 3.00
    point_multiple: float = 5.00
    band_average_multiple: float = 2.20
    band_minimum_thick_bins: int = 2
    band_minimum_occupancy: float = 0.55
    band_minimum_contrast: float = 1.15
    point_minimum_global_ratio: float = 0.06
    band_minimum_global_mass: float = 0.20
    maximum_distance_bps: float = 600.0
    maximum_candidates_per_side: int = 12
    nms_overlap_fraction: float = 0.65

    minimum_track_observations: int = 2
    maximum_missing_frames: int = 2
    minimum_match_overlap: float = 0.25
    maximum_center_drift_bins: float = 3.0
    ghost_drift_widths_per_minute: float = 0.35
    recent_fade_observations: int = 12
    minimum_touch_age_seconds: float = 10.0
    minimum_touch_retention: float = 0.20
    event_cooldown_seconds: int = 300

    bounce_bps: tuple[int, ...] = (20, 30, 50)
    break_buffer_bps: float = 12.0
    break_buffer_bins: int = 1
    outcome_horizons_minutes: tuple[int, ...] = (15, 30, 60)
    breakout_volume_multiple: float = 1.50
    breakout_sell_imbalance: float = 0.58
    train_fraction: float = 0.75
    minimum_bin_events: int = 20

    def validate(self) -> None:
        if self.price_step <= 0:
            raise ValueError("price_step must be > 0")
        if self.reference_window_hours <= 0:
            raise ValueError("reference_window_hours must be > 0")
        if not 0.5 <= self.snapshot_reference_quantile <= 1.0:
            raise ValueError("snapshot_reference_quantile must be in [0.5, 1]")
        if not 0 < self.side_baseline_quantile <= 1:
            raise ValueError("side_baseline_quantile must be in (0, 1]")
        widths = tuple(sorted({int(value) for value in self.candidate_widths}))
        if not widths or widths[0] <= 0:
            raise ValueError("candidate_widths must contain positive integers")
        if self.point_multiple < self.thick_multiple:
            raise ValueError("point_multiple must be >= thick_multiple")
        if self.maximum_candidates_per_side <= 0:
            raise ValueError("maximum_candidates_per_side must be > 0")
        if self.maximum_missing_frames < 0:
            raise ValueError("maximum_missing_frames must be >= 0")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1)")
        if not self.bounce_bps or not self.outcome_horizons_minutes:
            raise ValueError("bounce and outcome horizon grids must not be empty")


@dataclass(frozen=True)
class WallCandidate:
    available_time_ms: int
    bucket_start_ms: int
    bucket_end_ms: int
    side: str
    morphology: str
    price_low: float
    price_high: float
    center_price: float
    low_bin: int
    high_bin: int
    width_bins: int
    midpoint: float
    distance_bps: float
    side_baseline_depth: float
    causal_reference_depth: float
    total_depth_base: float
    mean_depth_base: float
    peak_depth_base: float
    average_local_multiple: float
    peak_local_multiple: float
    mean_global_ratio: float
    peak_global_ratio: float
    support_occupancy: float
    thick_occupancy: float
    nonzero_occupancy: float
    hole_ratio: float
    spatial_contrast: float
    shape_score: float
    added_base: float
    removed_base: float
    executed_base: float
    cancelled_base: float
    consumed_base: float
    replenished_base: float
    flow_valid: int

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Track:
    wall_id: int
    side: str
    source_seconds: int
    ghost_threshold: float
    first_seen_ms: int
    last_seen_ms: int
    last_available_ms: int
    observations: list[dict[str, Any]] = field(default_factory=list)
    missed_frames: int = 0
    maximum_missed_frames: int = 0
    peak_total_depth: float = 0.0
    peak_shape_score: float = 0.0
    touch_count: int = 0
    last_touch_ms: int | None = None
    last_probe_depth: float = 0.0
    last_probe_cancelled: float = 0.0
    last_probe_consumed: float = 0.0
    last_probe_replenished: float = 0.0
    death_reason: str | None = None

    def update(self, candidate: WallCandidate) -> dict[str, Any]:
        self.last_seen_ms = candidate.bucket_start_ms
        self.last_available_ms = candidate.available_time_ms
        self.missed_frames = 0
        self.peak_total_depth = max(self.peak_total_depth, candidate.total_depth_base)
        self.peak_shape_score = max(self.peak_shape_score, candidate.shape_score)
        record = candidate.to_record()
        self.observations.append(record)
        return self.state_record(record)

    def state_record(self, record: dict[str, Any]) -> dict[str, Any]:
        centers = np.asarray([item["center_price"] for item in self.observations], dtype=float)
        widths = np.asarray([item["width_bins"] for item in self.observations], dtype=float)
        depths = np.asarray([item["total_depth_base"] for item in self.observations], dtype=float)
        scores = np.asarray([item["shape_score"] for item in self.observations], dtype=float)
        elapsed_minutes = max((record["available_time_ms"] - self.first_seen_ms) / 60_000.0, 1e-9)
        center_span_bins = (float(np.nanmax(centers)) - float(np.nanmin(centers))) / max(
            float(record["price_high"] - record["price_low"]) / max(float(record["width_bins"]), 1.0),
            1e-12,
        )
        average_width = max(float(np.nanmean(widths)), 1.0)
        drift_widths_per_minute = center_span_bins / average_width / elapsed_minutes
        recent_n = min(len(depths), 12)
        fade_slope = _normalized_slope(depths[-recent_n:]) if recent_n >= 3 else 0.0
        retention = float(record["total_depth_base"]) / max(self.peak_total_depth, 1e-12)
        observation_count = len(self.observations)
        expected_frames = max(
            1,
            int(round((record["available_time_ms"] - self.first_seen_ms) / max(self.source_seconds * 1000, 1))) + 1,
        )
        morphology_counts = Counter(item["morphology"] for item in self.observations)
        dominant_morphology = morphology_counts.most_common(1)[0][0]
        out = dict(record)
        out.update(
            {
                "wall_id": self.wall_id,
                "wall_age_seconds": max(0.0, (record["available_time_ms"] - self.first_seen_ms) / 1000.0),
                "wall_observations": observation_count,
                "wall_time_coverage": min(1.0, observation_count / expected_frames),
                "wall_peak_total_depth": self.peak_total_depth,
                "wall_current_retention": retention,
                "wall_mean_shape_score": float(np.nanmean(scores)),
                "wall_peak_shape_score": self.peak_shape_score,
                "wall_center_span_bins": center_span_bins,
                "wall_drift_widths_per_minute": drift_widths_per_minute,
                "wall_recent_fade_slope": fade_slope,
                "wall_dominant_morphology": dominant_morphology,
                "wall_point_share": morphology_counts.get("POINT", 0) / observation_count,
                "wall_is_ghost": int(drift_widths_per_minute > self.ghost_threshold),
                "wall_touch_count_before": self.touch_count,
            }
        )
        return out

    def summary(self) -> dict[str, Any]:
        if not self.observations:
            return {}
        states = [self.state_record(item) for item in self.observations]
        latest = states[-1]
        lows = np.asarray([item["price_low"] for item in self.observations], dtype=float)
        highs = np.asarray([item["price_high"] for item in self.observations], dtype=float)
        latest.update(
            {
                "first_seen_ms": self.first_seen_ms,
                "last_seen_ms": self.last_seen_ms,
                "last_available_ms": self.last_available_ms,
                "duration_seconds": max(0.0, (self.last_available_ms - self.first_seen_ms) / 1000.0),
                "stable_price_low": float(np.quantile(lows, 0.25)),
                "stable_price_high": float(np.quantile(highs, 0.75)),
                "maximum_missed_frames": self.maximum_missed_frames,
                "touch_count": self.touch_count,
                "active_at_end": int(self.death_reason in {None, "ACTIVE_AT_END"}),
                "death_reason": self.death_reason or "ACTIVE_AT_END",
                "last_probe_depth": self.last_probe_depth,
                "last_probe_cancelled": self.last_probe_cancelled,
                "last_probe_consumed": self.last_probe_consumed,
                "last_probe_replenished": self.last_probe_replenished,
            }
        )
        return latest


class CausalDepthReference:
    """Monotonic rolling maximum of each snapshot's robust high depth."""

    def __init__(self, *, window_hours: float, quantile: float, minimum: float = 1e-12) -> None:
        self.window_ms = max(1, int(round(float(window_hours) * 3_600_000.0)))
        self.quantile = float(quantile)
        self.minimum = float(minimum)
        self._queue: deque[tuple[int, float]] = deque()

    def update(self, timestamp_ms: int, depths: np.ndarray) -> tuple[float, float]:
        clean = np.asarray(depths, dtype=np.float64)
        clean = clean[np.isfinite(clean) & (clean > self.minimum)]
        snapshot_high = float(np.quantile(clean, self.quantile)) if len(clean) else self.minimum
        cutoff = int(timestamp_ms) - self.window_ms
        while self._queue and self._queue[0][0] < cutoff:
            self._queue.popleft()
        while self._queue and self._queue[-1][1] <= snapshot_high:
            self._queue.pop()
        self._queue.append((int(timestamp_ms), max(snapshot_high, self.minimum)))
        return max(float(self._queue[0][1]), self.minimum), snapshot_high


class WallDiscoveryEngine:
    """Stateful causal candidate extractor and cross-snapshot tracker."""

    def __init__(self, config: WallDiscoveryConfig, *, source_seconds: int = 5) -> None:
        config.validate()
        self.config = config
        self.source_seconds = max(1, int(source_seconds))
        self.reference = CausalDepthReference(
            window_hours=config.reference_window_hours,
            quantile=config.snapshot_reference_quantile,
        )
        self._active: dict[int, _Track] = {}
        self._closed: list[_Track] = []
        self._next_wall_id = 1

    def process_snapshot(self, snapshot: pd.DataFrame) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        candidates = extract_snapshot_candidates(snapshot, self.config, self.reference)
        states = self._update_tracks(candidates, snapshot)
        return [item.to_record() for item in candidates], states

    def process_primitive_snapshot(
        self, snapshot: LiquidityPrimitiveSnapshot
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Process one cached primitive snapshot without constructing a DataFrame."""

        candidates = extract_primitive_snapshot_candidates(snapshot, self.config)
        states = self._update_tracks(candidates, snapshot)
        return [item.to_record() for item in candidates], states

    def drain_closed_summaries(self) -> list[dict[str, Any]]:
        """Return newly closed tracks and release them from memory."""

        rows = [item.summary() for item in self._closed if item.observations]
        self._closed.clear()
        return rows

    def finish(self) -> list[dict[str, Any]]:
        for track in list(self._active.values()):
            track.death_reason = track.death_reason or "ACTIVE_AT_END"
            self._closed.append(track)
        self._active.clear()
        return self.drain_closed_summaries()

    def mark_touch(self, wall_id: int, timestamp_ms: int) -> None:
        track = self._active.get(int(wall_id))
        if track is None:
            return
        track.touch_count += 1
        track.last_touch_ms = int(timestamp_ms)

    def _update_tracks(
        self,
        candidates: Sequence[WallCandidate],
        snapshot: pd.DataFrame | LiquidityPrimitiveSnapshot,
    ) -> list[dict[str, Any]]:
        cfg = self.config
        used_tracks: set[int] = set()
        states: list[dict[str, Any]] = []
        for candidate in sorted(candidates, key=lambda item: item.shape_score, reverse=True):
            best_id = None
            best_score = -np.inf
            for wall_id, track in self._active.items():
                if wall_id in used_tracks or track.side != candidate.side or not track.observations:
                    continue
                previous = track.observations[-1]
                overlap = _price_overlap_fraction(
                    previous["price_low"], previous["price_high"], candidate.price_low, candidate.price_high
                )
                step = max(self.config.price_step, 1e-12)
                drift_bins = abs(float(previous["center_price"]) - candidate.center_price) / step
                if overlap < cfg.minimum_match_overlap and drift_bins > cfg.maximum_center_drift_bins:
                    continue
                score = 3.0 * overlap - 0.12 * drift_bins + 0.01 * candidate.shape_score
                if score > best_score:
                    best_score = score
                    best_id = wall_id
            if best_id is None:
                wall_id = self._next_wall_id
                self._next_wall_id += 1
                track = _Track(
                    wall_id=wall_id,
                    side=candidate.side,
                    source_seconds=self.source_seconds,
                    ghost_threshold=cfg.ghost_drift_widths_per_minute,
                    first_seen_ms=candidate.available_time_ms,
                    last_seen_ms=candidate.bucket_start_ms,
                    last_available_ms=candidate.available_time_ms,
                )
                self._active[wall_id] = track
            else:
                wall_id = int(best_id)
                track = self._active[wall_id]
            used_tracks.add(wall_id)
            state = track.update(candidate)
            state["wall_is_ghost"] = int(
                float(state["wall_drift_widths_per_minute"]) > cfg.ghost_drift_widths_per_minute
            )
            states.append(state)

        closed_ids: list[int] = []
        for wall_id, track in self._active.items():
            if wall_id in used_tracks:
                continue
            track.missed_frames += 1
            track.maximum_missed_frames = max(track.maximum_missed_frames, track.missed_frames)
            probe = (
                _probe_track_band_primitive(snapshot, track, self.config.price_step)
                if isinstance(snapshot, LiquidityPrimitiveSnapshot)
                else _probe_track_band(snapshot, track, self.config.price_step)
            )
            track.last_probe_depth = probe["depth"]
            track.last_probe_cancelled = probe["cancelled"]
            track.last_probe_consumed = probe["consumed"]
            track.last_probe_replenished = probe["replenished"]
            if track.missed_frames > cfg.maximum_missing_frames:
                if probe["cancelled"] > max(probe["consumed"] * 1.25, 1e-12):
                    track.death_reason = "WITHDRAWN"
                elif probe["consumed"] > 0 and probe["replenished"] < probe["consumed"] * 0.50:
                    track.death_reason = "CONSUMED_NOT_REPLENISHED"
                elif probe["depth"] <= max(track.peak_total_depth * 0.10, 1e-12):
                    track.death_reason = "FADED_OR_REMOVED"
                else:
                    track.death_reason = "TRACK_LOST"
                self._closed.append(track)
                closed_ids.append(wall_id)
        for wall_id in closed_ids:
            self._active.pop(wall_id, None)
        return states


_REQUIRED_SNAPSHOT_COLUMNS = {
    "bucket_start_ms",
    "bucket_end_ms",
    "price_index",
    "side_code",
}


def extract_snapshot_candidates(
    snapshot: pd.DataFrame,
    config: WallDiscoveryConfig,
    reference: CausalDepthReference,
) -> list[WallCandidate]:
    """Extract broad multi-scale candidates from one completed snapshot."""

    missing = sorted(_REQUIRED_SNAPSHOT_COLUMNS.difference(snapshot.columns))
    if missing:
        raise ValueError(f"snapshot missing fields: {missing}")
    if snapshot.empty:
        return []
    depth_column = "end_depth_base" if "end_depth_base" in snapshot.columns else "depth_base"
    if depth_column not in snapshot.columns:
        raise ValueError("snapshot needs end_depth_base or depth_base")
    work = snapshot.copy()
    for column in ("bucket_start_ms", "bucket_end_ms", "price_index", "side_code"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    for column in (
        depth_column,
        "added_base",
        "removed_base",
        "executed_base",
        "cancelled_base",
        "consumed_base",
        "replenished_base",
        "flow_valid",
    ):
        if column not in work.columns:
            work[column] = 0.0
        work[column] = pd.to_numeric(work[column], errors="coerce").fillna(0.0)
    work = work.dropna(subset=["bucket_start_ms", "bucket_end_ms", "price_index", "side_code"])
    work["depth"] = work[depth_column].clip(lower=0.0)
    work = work.loc[work["side_code"].isin([1, -1]) & (work["depth"] > 0)].copy()
    if work.empty:
        return []
    bucket_start_ms = int(work["bucket_start_ms"].max())
    bucket_end_ms = int(work["bucket_end_ms"].max())
    all_depths = work["depth"].to_numpy(dtype=float)
    reference_depth, _ = reference.update(bucket_end_ms, all_depths)
    best_bid_bin = work.loc[work["side_code"] == 1, "price_index"].max()
    best_ask_bin = work.loc[work["side_code"] == -1, "price_index"].min()
    if pd.isna(best_bid_bin) or pd.isna(best_ask_bin):
        return []
    midpoint = (float(best_bid_bin + 1) + float(best_ask_bin)) * config.price_step / 2.0
    if midpoint <= 0:
        return []

    candidates: list[WallCandidate] = []
    for side_code, side_name in ((1, "bid"), (-1, "ask")):
        side = work.loc[work["side_code"] == side_code].copy()
        if side.empty:
            continue
        side_candidates = _extract_side_candidates(
            side,
            side_name=side_name,
            bucket_start_ms=bucket_start_ms,
            bucket_end_ms=bucket_end_ms,
            midpoint=midpoint,
            reference_depth=reference_depth,
            config=config,
        )
        candidates.extend(side_candidates)
    return candidates


def _extract_side_candidates(
    side: pd.DataFrame,
    *,
    side_name: str,
    bucket_start_ms: int,
    bucket_end_ms: int,
    midpoint: float,
    reference_depth: float,
    config: WallDiscoveryConfig,
) -> list[WallCandidate]:
    step = float(config.price_step)
    side = side.sort_values("price_index").drop_duplicates("price_index", keep="last")
    low_idx = int(side["price_index"].min())
    high_idx = int(side["price_index"].max())
    indices = np.arange(low_idx, high_idx + 1, dtype=np.int64)
    width = len(indices)
    depth = np.zeros(width, dtype=np.float64)
    fields = {
        name: np.zeros(width, dtype=np.float64)
        for name in (
            "added_base",
            "removed_base",
            "executed_base",
            "cancelled_base",
            "consumed_base",
            "replenished_base",
            "flow_valid",
        )
    }
    positions = side["price_index"].to_numpy(dtype=np.int64) - low_idx
    depth[positions] = side["depth"].to_numpy(dtype=float)
    for name in fields:
        fields[name][positions] = side[name].to_numpy(dtype=float)
    positive = depth[depth > 0]
    if len(positive) == 0:
        return []
    baseline = max(float(np.quantile(positive, config.side_baseline_quantile)), 1e-12)
    local_multiple = depth / baseline
    global_ratio = depth / max(reference_depth, 1e-12)
    support_mask = local_multiple >= config.support_multiple
    thick_mask = local_multiple >= config.thick_multiple
    nonzero = depth > 0

    raw: list[WallCandidate] = []
    prefix_depth = np.concatenate(([0.0], np.cumsum(depth)))
    prefix_nonzero = np.concatenate(([0], np.cumsum(nonzero.astype(np.int64))))
    for candidate_width in sorted({int(value) for value in config.candidate_widths}):
        if candidate_width > width:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(depth, candidate_width)
        depth_sum = windows.sum(axis=1)
        peak_depths = windows.max(axis=1)
        support_count = np.lib.stride_tricks.sliding_window_view(
            support_mask.astype(np.int8), candidate_width
        ).sum(axis=1)
        thick_count = np.lib.stride_tricks.sliding_window_view(
            thick_mask.astype(np.int8), candidate_width
        ).sum(axis=1)
        nonzero_count = np.lib.stride_tricks.sliding_window_view(
            nonzero.astype(np.int8), candidate_width
        ).sum(axis=1)
        global_windows = np.lib.stride_tricks.sliding_window_view(global_ratio, candidate_width)
        global_mass = global_windows.sum(axis=1)
        peak_global = global_windows.max(axis=1)
        starts = np.arange(len(depth_sum), dtype=np.int64)
        ends = starts + candidate_width
        low_bins = indices[starts]
        high_bins = indices[ends - 1]
        price_lows = low_bins.astype(float) * step
        price_highs = (high_bins.astype(float) + 1.0) * step
        centers = (price_lows + price_highs) / 2.0
        if side_name == "bid":
            distances = (midpoint - price_highs) / midpoint * 10_000.0
        else:
            distances = (price_lows - midpoint) / midpoint * 10_000.0
        average_multiple = depth_sum / baseline / candidate_width
        peak_multiple = peak_depths / baseline
        support_occupancy = support_count / candidate_width
        thick_occupancy = thick_count / candidate_width
        nonzero_occupancy = nonzero_count / candidate_width
        hole_ratio = 1.0 - nonzero_occupancy
        mean_global_ratio = global_mass / candidate_width

        left_starts = np.maximum(0, starts - candidate_width)
        left_ends = starts
        right_starts = ends
        right_ends = np.minimum(width, ends + candidate_width)
        neighbor_sum = (prefix_depth[left_ends] - prefix_depth[left_starts]) + (
            prefix_depth[right_ends] - prefix_depth[right_starts]
        )
        neighbor_count = (prefix_nonzero[left_ends] - prefix_nonzero[left_starts]) + (
            prefix_nonzero[right_ends] - prefix_nonzero[right_starts]
        )
        inside_mean = depth_sum / candidate_width
        outside_mean = np.where(
            neighbor_count > 0, neighbor_sum / np.maximum(neighbor_count, 1), inside_mean * 0.05
        )
        contrast = inside_mean / np.maximum(outside_mean, 1e-12)
        distance_ok = (distances >= -1e-9) & (distances <= config.maximum_distance_bps)
        if candidate_width == 1:
            passes = (
                (peak_multiple >= config.point_multiple)
                & (peak_global >= config.point_minimum_global_ratio)
                & distance_ok
                & (depth_sum > 0)
            )
        else:
            minimum_thick = min(config.band_minimum_thick_bins, candidate_width)
            passes = (
                (average_multiple >= config.band_average_multiple)
                & (thick_count >= minimum_thick)
                & (nonzero_occupancy >= config.band_minimum_occupancy)
                & (contrast >= config.band_minimum_contrast)
                & (global_mass >= config.band_minimum_global_mass)
                & distance_ok
                & (depth_sum > 0)
            )
        flow_sums = {
            name: np.lib.stride_tricks.sliding_window_view(values, candidate_width).sum(axis=1)
            for name, values in fields.items()
        }
        for start_pos in np.flatnonzero(passes):
            end_pos = int(start_pos + candidate_width)
            morphology = (
                "POINT"
                if candidate_width == 1
                else (
                    "COMPOSITE"
                    if hole_ratio[start_pos] > 0 and support_occupancy[start_pos] >= 0.50
                    else "BAND"
                )
            )
            score = (
                22.0 * math.log1p(max(float(average_multiple[start_pos]) - 1.0, 0.0))
                + 12.0 * math.log1p(max(float(peak_multiple[start_pos]) - 1.0, 0.0))
                + 8.0 * math.log1p(candidate_width)
                + 12.0 * float(thick_occupancy[start_pos])
                + 8.0 * float(support_occupancy[start_pos])
                + 8.0 * min(max(float(contrast[start_pos]) - 1.0, 0.0), 2.0)
                - 16.0 * float(hole_ratio[start_pos])
            )
            flow = {name: float(values[start_pos]) for name, values in flow_sums.items()}
            raw.append(
                WallCandidate(
                    available_time_ms=bucket_end_ms,
                    bucket_start_ms=bucket_start_ms,
                    bucket_end_ms=bucket_end_ms,
                    side=side_name,
                    morphology=morphology,
                    price_low=float(price_lows[start_pos]),
                    price_high=float(price_highs[start_pos]),
                    center_price=float(centers[start_pos]),
                    low_bin=int(low_bins[start_pos]),
                    high_bin=int(high_bins[start_pos]),
                    width_bins=candidate_width,
                    midpoint=midpoint,
                    distance_bps=float(distances[start_pos]),
                    side_baseline_depth=baseline,
                    causal_reference_depth=float(reference_depth),
                    total_depth_base=float(depth_sum[start_pos]),
                    mean_depth_base=float(inside_mean[start_pos]),
                    peak_depth_base=float(peak_depths[start_pos]),
                    average_local_multiple=float(average_multiple[start_pos]),
                    peak_local_multiple=float(peak_multiple[start_pos]),
                    mean_global_ratio=float(mean_global_ratio[start_pos]),
                    peak_global_ratio=float(peak_global[start_pos]),
                    support_occupancy=float(support_occupancy[start_pos]),
                    thick_occupancy=float(thick_occupancy[start_pos]),
                    nonzero_occupancy=float(nonzero_occupancy[start_pos]),
                    hole_ratio=float(hole_ratio[start_pos]),
                    spatial_contrast=float(contrast[start_pos]),
                    shape_score=float(score),
                    added_base=flow["added_base"],
                    removed_base=flow["removed_base"],
                    executed_base=flow["executed_base"],
                    cancelled_base=flow["cancelled_base"],
                    consumed_base=flow["consumed_base"],
                    replenished_base=flow["replenished_base"],
                    flow_valid=int(flow["flow_valid"] >= candidate_width - 1e-9),
                )
            )
    return _non_maximum_suppression(raw, config)



def extract_primitive_snapshot_candidates(
    snapshot: LiquidityPrimitiveSnapshot,
    config: WallDiscoveryConfig,
) -> list[WallCandidate]:
    """Extract candidates from low-semantic cached arrays.

    The cache supplies only completed-book state and neutral relative-depth
    summaries. Candidate widths and all wall thresholds remain live research
    parameters here.
    """

    if snapshot.midpoint <= 0 or not np.isfinite(snapshot.midpoint):
        return []
    reference_quantile = float(config.snapshot_reference_quantile)
    if abs(reference_quantile - 0.95) <= 1e-12:
        reference_depth = snapshot.causal_q95
    elif abs(reference_quantile - 0.99) <= 1e-12:
        reference_depth = snapshot.causal_q99
    else:
        raise ValueError(
            "primitive cache supports snapshot_reference_quantile 0.95 or 0.99; "
            "rebuild a new primitive schema to study another rolling quantile"
        )
    candidates: list[WallCandidate] = []
    for code, side_name in ((1, "bid"), (-1, "ask")):
        mask = snapshot.side_code == code
        if not np.any(mask):
            continue
        candidates.extend(
            _extract_side_candidate_arrays(
                price_index=np.asarray(snapshot.price_index[mask], dtype=np.int64),
                depth=np.asarray(snapshot.depth_base[mask], dtype=np.float64),
                fields={
                    name: np.asarray(getattr(snapshot, name)[mask], dtype=np.float64)
                    for name in (
                        "added_base",
                        "removed_base",
                        "executed_base",
                        "cancelled_base",
                        "consumed_base",
                        "replenished_base",
                        "flow_valid",
                    )
                },
                side_name=side_name,
                bucket_start_ms=snapshot.bucket_start_ms,
                bucket_end_ms=snapshot.bucket_end_ms,
                midpoint=snapshot.midpoint,
                reference_depth=max(float(reference_depth), 1e-12),
                baseline=snapshot.baseline(side_name, config.side_baseline_quantile),
                config=config,
            )
        )
    return candidates


def _extract_side_candidate_arrays(
    *,
    price_index: np.ndarray,
    depth: np.ndarray,
    fields: dict[str, np.ndarray],
    side_name: str,
    bucket_start_ms: int,
    bucket_end_ms: int,
    midpoint: float,
    reference_depth: float,
    baseline: float,
    config: WallDiscoveryConfig,
) -> list[WallCandidate]:
    """NumPy-only candidate extraction used by the primitive-cache path."""

    if len(price_index) == 0:
        return []
    order = np.argsort(price_index, kind="stable")
    prices_sorted = price_index[order]
    depth_sorted = depth[order]
    keep = np.ones(len(prices_sorted), dtype=bool)
    if len(keep) > 1:
        keep[:-1] = prices_sorted[:-1] != prices_sorted[1:]
    prices_sorted = prices_sorted[keep]
    depth_sorted = depth_sorted[keep]
    field_sorted = {name: values[order][keep] for name, values in fields.items()}

    low_idx = int(prices_sorted[0])
    high_idx = int(prices_sorted[-1])
    indices = np.arange(low_idx, high_idx + 1, dtype=np.int64)
    width = len(indices)
    dense_depth = np.zeros(width, dtype=np.float64)
    dense_fields = {name: np.zeros(width, dtype=np.float64) for name in field_sorted}
    positions = prices_sorted - low_idx
    dense_depth[positions] = depth_sorted
    for name, values in field_sorted.items():
        dense_fields[name][positions] = values

    baseline = max(float(baseline), 1e-12)
    local_multiple = dense_depth / baseline
    global_ratio = dense_depth / max(float(reference_depth), 1e-12)
    support_mask = local_multiple >= config.support_multiple
    thick_mask = local_multiple >= config.thick_multiple
    nonzero = dense_depth > 0
    prefix_depth = np.concatenate(([0.0], np.cumsum(dense_depth)))
    prefix_nonzero = np.concatenate(([0], np.cumsum(nonzero.astype(np.int64))))
    raw: list[WallCandidate] = []

    for candidate_width in sorted({int(value) for value in config.candidate_widths}):
        if candidate_width > width:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(dense_depth, candidate_width)
        depth_sum = windows.sum(axis=1)
        peak_depths = windows.max(axis=1)
        support_count = np.lib.stride_tricks.sliding_window_view(
            support_mask.astype(np.int8), candidate_width
        ).sum(axis=1)
        thick_count = np.lib.stride_tricks.sliding_window_view(
            thick_mask.astype(np.int8), candidate_width
        ).sum(axis=1)
        nonzero_count = np.lib.stride_tricks.sliding_window_view(
            nonzero.astype(np.int8), candidate_width
        ).sum(axis=1)
        global_windows = np.lib.stride_tricks.sliding_window_view(global_ratio, candidate_width)
        global_mass = global_windows.sum(axis=1)
        peak_global = global_windows.max(axis=1)
        starts = np.arange(len(depth_sum), dtype=np.int64)
        ends = starts + candidate_width
        low_bins = indices[starts]
        high_bins = indices[ends - 1]
        price_lows = low_bins.astype(float) * config.price_step
        price_highs = (high_bins.astype(float) + 1.0) * config.price_step
        centers = (price_lows + price_highs) / 2.0
        distances = (
            (midpoint - price_highs) / midpoint * 10_000.0
            if side_name == "bid"
            else (price_lows - midpoint) / midpoint * 10_000.0
        )
        average_multiple = depth_sum / baseline / candidate_width
        peak_multiple = peak_depths / baseline
        support_occupancy = support_count / candidate_width
        thick_occupancy = thick_count / candidate_width
        nonzero_occupancy = nonzero_count / candidate_width
        hole_ratio = 1.0 - nonzero_occupancy
        mean_global_ratio = global_mass / candidate_width
        left_starts = np.maximum(0, starts - candidate_width)
        left_ends = starts
        right_starts = ends
        right_ends = np.minimum(width, ends + candidate_width)
        neighbor_sum = (prefix_depth[left_ends] - prefix_depth[left_starts]) + (
            prefix_depth[right_ends] - prefix_depth[right_starts]
        )
        neighbor_count = (prefix_nonzero[left_ends] - prefix_nonzero[left_starts]) + (
            prefix_nonzero[right_ends] - prefix_nonzero[right_starts]
        )
        inside_mean = depth_sum / candidate_width
        outside_mean = np.where(
            neighbor_count > 0,
            neighbor_sum / np.maximum(neighbor_count, 1),
            inside_mean * 0.05,
        )
        contrast = inside_mean / np.maximum(outside_mean, 1e-12)
        distance_ok = (distances >= -1e-9) & (distances <= config.maximum_distance_bps)
        if candidate_width == 1:
            passes = (
                (peak_multiple >= config.point_multiple)
                & (peak_global >= config.point_minimum_global_ratio)
                & distance_ok
                & (depth_sum > 0)
            )
        else:
            minimum_thick = min(config.band_minimum_thick_bins, candidate_width)
            passes = (
                (average_multiple >= config.band_average_multiple)
                & (thick_count >= minimum_thick)
                & (nonzero_occupancy >= config.band_minimum_occupancy)
                & (contrast >= config.band_minimum_contrast)
                & (global_mass >= config.band_minimum_global_mass)
                & distance_ok
                & (depth_sum > 0)
            )
        flow_sums = {
            name: np.lib.stride_tricks.sliding_window_view(values, candidate_width).sum(axis=1)
            for name, values in dense_fields.items()
        }
        for start_pos in np.flatnonzero(passes):
            morphology = (
                "POINT"
                if candidate_width == 1
                else (
                    "COMPOSITE"
                    if hole_ratio[start_pos] > 0 and support_occupancy[start_pos] >= 0.50
                    else "BAND"
                )
            )
            score = (
                22.0 * math.log1p(max(float(average_multiple[start_pos]) - 1.0, 0.0))
                + 12.0 * math.log1p(max(float(peak_multiple[start_pos]) - 1.0, 0.0))
                + 8.0 * math.log1p(candidate_width)
                + 12.0 * float(thick_occupancy[start_pos])
                + 8.0 * float(support_occupancy[start_pos])
                + 8.0 * min(max(float(contrast[start_pos]) - 1.0, 0.0), 2.0)
                - 16.0 * float(hole_ratio[start_pos])
            )
            flow = {name: float(values[start_pos]) for name, values in flow_sums.items()}
            raw.append(
                WallCandidate(
                    available_time_ms=int(bucket_end_ms),
                    bucket_start_ms=int(bucket_start_ms),
                    bucket_end_ms=int(bucket_end_ms),
                    side=side_name,
                    morphology=morphology,
                    price_low=float(price_lows[start_pos]),
                    price_high=float(price_highs[start_pos]),
                    center_price=float(centers[start_pos]),
                    low_bin=int(low_bins[start_pos]),
                    high_bin=int(high_bins[start_pos]),
                    width_bins=candidate_width,
                    midpoint=float(midpoint),
                    distance_bps=float(distances[start_pos]),
                    side_baseline_depth=float(baseline),
                    causal_reference_depth=float(reference_depth),
                    total_depth_base=float(depth_sum[start_pos]),
                    mean_depth_base=float(inside_mean[start_pos]),
                    peak_depth_base=float(peak_depths[start_pos]),
                    average_local_multiple=float(average_multiple[start_pos]),
                    peak_local_multiple=float(peak_multiple[start_pos]),
                    mean_global_ratio=float(mean_global_ratio[start_pos]),
                    peak_global_ratio=float(peak_global[start_pos]),
                    support_occupancy=float(support_occupancy[start_pos]),
                    thick_occupancy=float(thick_occupancy[start_pos]),
                    nonzero_occupancy=float(nonzero_occupancy[start_pos]),
                    hole_ratio=float(hole_ratio[start_pos]),
                    spatial_contrast=float(contrast[start_pos]),
                    shape_score=float(score),
                    added_base=flow["added_base"],
                    removed_base=flow["removed_base"],
                    executed_base=flow["executed_base"],
                    cancelled_base=flow["cancelled_base"],
                    consumed_base=flow["consumed_base"],
                    replenished_base=flow["replenished_base"],
                    flow_valid=int(flow["flow_valid"] >= candidate_width - 1e-9),
                )
            )
    return _non_maximum_suppression(raw, config)


def _probe_track_band_primitive(
    snapshot: LiquidityPrimitiveSnapshot,
    track: _Track,
    price_step: float,
) -> dict[str, float]:
    if not track.observations:
        return {"depth": 0.0, "cancelled": 0.0, "consumed": 0.0, "replenished": 0.0}
    previous = track.observations[-1]
    side_code = 1 if track.side == "bid" else -1
    low_bin = int(math.floor(float(previous["price_low"]) / price_step + 1e-12))
    high_bin = int(math.ceil(float(previous["price_high"]) / price_step - 1e-12)) - 1
    mask = (
        (snapshot.side_code == side_code)
        & (snapshot.price_index >= low_bin)
        & (snapshot.price_index <= high_bin)
    )
    return {
        "depth": float(np.sum(snapshot.depth_base[mask])),
        "cancelled": float(np.sum(snapshot.cancelled_base[mask])),
        "consumed": float(np.sum(snapshot.consumed_base[mask])),
        "replenished": float(np.sum(snapshot.replenished_base[mask])),
    }

def _non_maximum_suppression(
    candidates: Sequence[WallCandidate], config: WallDiscoveryConfig
) -> list[WallCandidate]:
    kept: list[WallCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.shape_score, reverse=True):
        duplicate = False
        for existing in kept:
            overlap = _price_overlap_fraction(
                candidate.price_low,
                candidate.price_high,
                existing.price_low,
                existing.price_high,
            )
            if overlap >= config.nms_overlap_fraction:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(candidate)
        if len(kept) >= config.maximum_candidates_per_side:
            break
    return sorted(kept, key=lambda item: (item.side, item.price_low))


def _window_contrast(depth: np.ndarray, start: int, end: int) -> float:
    width = max(1, end - start)
    inside = float(np.mean(depth[start:end]))
    left = depth[max(0, start - width):start]
    right = depth[end:min(len(depth), end + width)]
    neighbors = np.concatenate([left, right]) if len(left) or len(right) else np.empty(0)
    positive = neighbors[neighbors > 0]
    outside = float(np.mean(positive)) if len(positive) else max(inside * 0.05, 1e-12)
    return inside / max(outside, 1e-12)


def _probe_track_band(snapshot: pd.DataFrame, track: _Track, price_step: float) -> dict[str, float]:
    if snapshot.empty or not track.observations:
        return {name: 0.0 for name in ("depth", "cancelled", "consumed", "replenished")}
    last = track.observations[-1]
    side_code = 1 if track.side == "bid" else -1
    low_bin = int(math.floor(float(last["price_low"]) / price_step + 1e-12))
    high_bin = int(math.ceil(float(last["price_high"]) / price_step - 1e-12)) - 1
    work = snapshot.loc[
        (pd.to_numeric(snapshot["side_code"], errors="coerce") == side_code)
        & (pd.to_numeric(snapshot["price_index"], errors="coerce") >= low_bin)
        & (pd.to_numeric(snapshot["price_index"], errors="coerce") <= high_bin)
    ]
    depth_column = "end_depth_base" if "end_depth_base" in work.columns else "depth_base"
    out = {}
    for output, column in (
        ("depth", depth_column),
        ("cancelled", "cancelled_base"),
        ("consumed", "consumed_base"),
        ("replenished", "replenished_base"),
    ):
        if column not in work.columns:
            out[output] = 0.0
        else:
            out[output] = float(pd.to_numeric(work[column], errors="coerce").fillna(0.0).sum())
    return out


def discover_wall_states(
    heatmap: pd.DataFrame,
    config: WallDiscoveryConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience in-memory discovery path used by tests and bounded audits."""

    if heatmap.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    durations = pd.to_numeric(heatmap["bucket_end_ms"], errors="coerce") - pd.to_numeric(
        heatmap["bucket_start_ms"], errors="coerce"
    )
    source_seconds = max(1, int(round(float(durations.dropna().median()) / 1000.0)))
    engine = WallDiscoveryEngine(config, source_seconds=source_seconds)
    candidate_rows: list[dict[str, Any]] = []
    state_rows: list[dict[str, Any]] = []
    for _, snapshot in heatmap.groupby("bucket_start_ms", sort=True, observed=True):
        candidates, states = engine.process_snapshot(snapshot)
        candidate_rows.extend(candidates)
        state_rows.extend(states)
    summaries = engine.finish()
    return pd.DataFrame(candidate_rows), pd.DataFrame(state_rows), pd.DataFrame(summaries)


def prepare_execution_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()
    out = bars.copy().sort_index()
    if not isinstance(out.index, pd.DatetimeIndex):
        raise ValueError("execution bars must use a DatetimeIndex")
    numeric = (
        "open",
        "high",
        "low",
        "close",
        "notional",
        "buy_notional",
        "sell_notional",
        "large_buy_notional",
        "large_sell_notional",
        "delta_notional",
        "volume",
    )
    for column in numeric:
        if column not in out.columns:
            out[column] = 0.0
        out[column] = pd.to_numeric(out[column], errors="coerce").fillna(0.0)
    utc_ms = _index_to_utc_ms(out.index)
    out["bar_start_ms"] = utc_ms
    if len(out.index) >= 2:
        step_ms = int(np.median(np.diff(utc_ms)))
    else:
        step_ms = 5_000
    out["bar_end_ms"] = out["bar_start_ms"] + max(step_ms, 1)
    out.attrs["bar_start_ms_array"] = out["bar_start_ms"].to_numpy(dtype=np.int64)
    total_notional = out["buy_notional"] + out["sell_notional"]
    out["sell_imbalance"] = np.where(total_notional > 0, out["sell_notional"] / total_notional, 0.5)
    out["large_sell_share"] = np.where(
        total_notional > 0, out["large_sell_notional"] / total_notional, 0.0
    )
    pre_close = out["close"].shift(1)
    for seconds in (30, 120, 300, 900):
        periods = max(1, int(round(seconds * 1000 / max(step_ms, 1))))
        out[f"ret_{seconds}s"] = pre_close.pct_change(periods)
    one_minute_periods = max(1, int(round(60_000 / max(step_ms, 1))))
    five_minute_periods = max(2, int(round(300_000 / max(step_ms, 1))))
    returns = out["close"].pct_change().shift(1)
    out["realized_vol_5m"] = returns.rolling(five_minute_periods, min_periods=2).std()
    out["notional_baseline_5m"] = out["notional"].shift(1).rolling(five_minute_periods, min_periods=2).median()
    out["sell_imbalance_pre_1m"] = out["sell_imbalance"].shift(1).rolling(one_minute_periods, min_periods=1).mean()
    out["large_sell_share_pre_1m"] = out["large_sell_share"].shift(1).rolling(one_minute_periods, min_periods=1).mean()
    out["sell_notional_pre_1m"] = out["sell_notional"].shift(1).rolling(one_minute_periods, min_periods=1).sum()
    return out


def touch_event_from_state(
    record: dict[str, Any],
    prepared_bars: pd.DataFrame,
    config: WallDiscoveryConfig,
) -> dict[str, Any] | None:
    """Return a causal touch event for one state, or ``None`` if not touched next."""

    if prepared_bars is None or prepared_bars.empty:
        return None
    if int(record.get("wall_observations", 0)) < config.minimum_track_observations:
        return None
    if float(record.get("wall_age_seconds", 0.0)) < config.minimum_touch_age_seconds:
        return None
    if float(record.get("wall_current_retention", 0.0)) < config.minimum_touch_retention:
        return None
    starts = prepared_bars.attrs.get("bar_start_ms_array")
    if starts is None:
        starts = prepared_bars["bar_start_ms"].to_numpy(dtype=np.int64)
    available_ms = int(record["available_time_ms"])
    bar_pos = int(np.searchsorted(starts, available_ms, side="left"))
    if bar_pos >= len(prepared_bars):
        return None
    bar = prepared_bars.iloc[bar_pos]
    if available_ms > int(bar["bar_start_ms"]):
        return None
    if not (
        float(bar["low"]) <= float(record["price_high"])
        and float(bar["high"]) >= float(record["price_low"])
    ):
        return None
    side = str(record["side"])
    event = dict(record)
    event.update(
        {
            "touch_time": prepared_bars.index[bar_pos],
            "touch_time_ms": int(bar["bar_start_ms"]),
            "touch_bar_end_ms": int(bar["bar_end_ms"]),
            "touch_open": float(bar["open"]),
            "touch_high": float(bar["high"]),
            "touch_low": float(bar["low"]),
            "touch_close": float(bar["close"]),
            "touch_notional": float(bar["notional"]),
            "touch_sell_imbalance": float(bar["sell_imbalance"]),
            "touch_large_sell_share": float(bar["large_sell_share"]),
            "pre_ret_30s": float(bar.get("ret_30s", np.nan)),
            "pre_ret_2m": float(bar.get("ret_120s", np.nan)),
            "pre_ret_5m": float(bar.get("ret_300s", np.nan)),
            "pre_ret_15m": float(bar.get("ret_900s", np.nan)),
            "pre_realized_vol_5m": float(bar.get("realized_vol_5m", np.nan)),
            "pre_sell_imbalance_1m": float(bar.get("sell_imbalance_pre_1m", np.nan)),
            "pre_large_sell_share_1m": float(bar.get("large_sell_share_pre_1m", np.nan)),
            "pre_sell_notional_1m": float(bar.get("sell_notional_pre_1m", np.nan)),
            "touch_notional_multiple": _safe_ratio(
                float(bar["notional"]), float(bar.get("notional_baseline_5m", np.nan))
            ),
            "causal_state_flag": int(available_ms <= int(bar["bar_start_ms"])),
        }
    )
    if side == "bid":
        penetration = max(0.0, float(record["price_high"]) - float(bar["low"]))
        event["touch_penetration_fraction"] = penetration / max(
            float(record["price_high"]) - float(record["price_low"]), config.price_step
        )
        event["touch_close_position"] = (float(bar["close"]) - float(record["price_low"])) / max(
            float(record["price_high"]) - float(record["price_low"]), config.price_step
        )
    else:
        penetration = max(0.0, float(bar["high"]) - float(record["price_low"]))
        event["touch_penetration_fraction"] = penetration / max(
            float(record["price_high"]) - float(record["price_low"]), config.price_step
        )
        event["touch_close_position"] = (float(record["price_high"]) - float(bar["close"])) / max(
            float(record["price_high"]) - float(record["price_low"]), config.price_step
        )
    return event


def build_wall_touch_events(
    wall_states: pd.DataFrame,
    execution_bars: pd.DataFrame,
    config: WallDiscoveryConfig,
) -> pd.DataFrame:
    """Build first-touch events using only wall state available before each bar."""

    if wall_states is None or wall_states.empty or execution_bars is None or execution_bars.empty:
        return pd.DataFrame()
    bars = prepare_execution_bars(execution_bars)
    rows: list[dict[str, Any]] = []
    last_touch_by_wall: dict[int, int] = {}
    states = wall_states.sort_values(["available_time_ms", "wall_id"])
    for record in states.to_dict("records"):
        event = touch_event_from_state(record, bars, config)
        if event is None:
            continue
        wall_id = int(event["wall_id"])
        touch_ms = int(event["touch_time_ms"])
        previous_touch = last_touch_by_wall.get(wall_id)
        if previous_touch is not None and touch_ms - previous_touch < config.event_cooldown_seconds * 1000:
            continue
        last_touch_by_wall[wall_id] = touch_ms
        rows.append(event)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["touch_time_ms", "wall_id"]).drop_duplicates(
        ["wall_id", "touch_time_ms"], keep="first"
    ).reset_index(drop=True)


def attach_touch_outcomes(
    events: pd.DataFrame,
    execution_bars: pd.DataFrame,
    config: WallDiscoveryConfig,
) -> pd.DataFrame:
    """Attach symmetric bounce/break first-passage and path labels."""

    if events is None or events.empty:
        return pd.DataFrame() if events is None else events.copy()
    bars = prepare_execution_bars(execution_bars)
    starts = bars["bar_start_ms"].to_numpy(dtype=np.int64)
    out_rows: list[dict[str, Any]] = []
    max_horizon_ms = max(config.outcome_horizons_minutes) * 60_000
    for event in events.to_dict("records"):
        touch_ms = int(event["touch_time_ms"])
        start_pos = int(np.searchsorted(starts, int(event["touch_bar_end_ms"]), side="left"))
        end_pos = int(np.searchsorted(starts, touch_ms + max_horizon_ms, side="right"))
        path = bars.iloc[start_pos:end_pos]
        row = dict(event)
        side = str(event["side"])
        wall_low = float(event["price_low"])
        wall_high = float(event["price_high"])
        break_buffer = max(
            config.break_buffer_bins * config.price_step,
            ((wall_low + wall_high) / 2.0) * config.break_buffer_bps / 10_000.0,
        )
        if side == "bid":
            break_price = wall_low - break_buffer
            anchor = wall_high
        else:
            break_price = wall_high + break_buffer
            anchor = wall_low
        row["break_price"] = break_price
        first_passages: dict[int, str] = {}
        for bounce_bps in config.bounce_bps:
            if side == "bid":
                bounce_price = anchor * (1.0 + bounce_bps / 10_000.0)
            else:
                bounce_price = anchor * (1.0 - bounce_bps / 10_000.0)
            label, label_ms, label_pos = _first_passage(path, side, bounce_price, break_price)
            row[f"outcome_{bounce_bps}bps"] = label
            row[f"outcome_{bounce_bps}bps_time_ms"] = label_ms
            first_passages[bounce_bps] = label
            if bounce_bps == min(config.bounce_bps):
                row["primary_outcome"] = label
                if label == "BREAK" and label_pos is not None:
                    break_bar = path.iloc[label_pos]
                    volume_multiple = _safe_ratio(
                        float(break_bar["notional"]), float(break_bar.get("notional_baseline_5m", np.nan))
                    )
                    sell_imbalance = float(break_bar["sell_imbalance"])
                    continuation = _break_continuation(path.iloc[label_pos:], side, break_price)
                    row["break_volume_multiple"] = volume_multiple
                    row["break_sell_imbalance"] = sell_imbalance
                    row["break_continuation_bps"] = continuation
                    row["volume_confirmed_break"] = int(
                        volume_multiple >= config.breakout_volume_multiple
                        and (
                            sell_imbalance >= config.breakout_sell_imbalance
                            if side == "bid"
                            else sell_imbalance <= 1.0 - config.breakout_sell_imbalance
                        )
                    )
                else:
                    row["break_volume_multiple"] = np.nan
                    row["break_sell_imbalance"] = np.nan
                    row["break_continuation_bps"] = np.nan
                    row["volume_confirmed_break"] = 0
        for horizon in config.outcome_horizons_minutes:
            horizon_end = touch_ms + horizon * 60_000
            horizon_path = path.loc[path["bar_start_ms"] < horizon_end]
            if horizon_path.empty:
                row[f"close_return_{horizon}m"] = np.nan
                row[f"mfe_{horizon}m_bps"] = np.nan
                row[f"mae_{horizon}m_bps"] = np.nan
                continue
            final_close = float(horizon_path.iloc[-1]["close"])
            if side == "bid":
                row[f"close_return_{horizon}m"] = final_close / anchor - 1.0
                row[f"mfe_{horizon}m_bps"] = (float(horizon_path["high"].max()) / anchor - 1.0) * 10_000
                row[f"mae_{horizon}m_bps"] = (float(horizon_path["low"].min()) / anchor - 1.0) * 10_000
            else:
                row[f"close_return_{horizon}m"] = anchor / final_close - 1.0
                row[f"mfe_{horizon}m_bps"] = (anchor / float(horizon_path["low"].min()) - 1.0) * 10_000
                row[f"mae_{horizon}m_bps"] = (anchor / float(horizon_path["high"].max()) - 1.0) * 10_000
        out_rows.append(row)
    return pd.DataFrame(out_rows)


def chronological_split(events: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame, int | None]:
    if events is None or events.empty:
        empty = pd.DataFrame() if events is None else events.copy()
        return empty, empty, None
    ordered = events.sort_values("touch_time_ms").reset_index(drop=True)
    split_pos = min(max(1, int(math.floor(len(ordered) * train_fraction))), len(ordered) - 1)
    split_ms = int(ordered.iloc[split_pos]["touch_time_ms"])
    return ordered.iloc[:split_pos].copy(), ordered.iloc[split_pos:].copy(), split_ms


def environment_feature_uplift(
    events: pd.DataFrame,
    *,
    features: Sequence[str],
    train_fraction: float = 0.75,
    quantiles: int = 4,
    minimum_bin_events: int = 20,
) -> pd.DataFrame:
    """Train-defined quantile bins evaluated separately on train and holdout."""

    if events is None or events.empty:
        return pd.DataFrame()
    train, holdout, split_ms = chronological_split(events, train_fraction)
    rows: list[dict[str, Any]] = []
    for feature in features:
        if feature not in events.columns:
            continue
        train_values = pd.to_numeric(train[feature], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if train_values.nunique() < 2:
            continue
        edges = np.unique(np.quantile(train_values, np.linspace(0.0, 1.0, quantiles + 1)))
        if len(edges) < 3:
            continue
        edges[0] = -np.inf
        edges[-1] = np.inf
        for sample_name, sample in (("train", train), ("holdout", holdout)):
            values = pd.to_numeric(sample[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
            bins = pd.cut(values, bins=edges, include_lowest=True, duplicates="drop")
            for interval, subset in sample.groupby(bins, observed=True):
                count = len(subset)
                if count < minimum_bin_events:
                    continue
                primary = subset["primary_outcome"].astype(str)
                rows.append(
                    {
                        "feature": feature,
                        "sample": sample_name,
                        "split_time_ms": split_ms,
                        "bin": str(interval),
                        "bin_low": float(interval.left),
                        "bin_high": float(interval.right),
                        "events": count,
                        "bounce_rate": float((primary == "BOUNCE").mean()),
                        "break_rate": float((primary == "BREAK").mean()),
                        "ambiguous_rate": float((primary == "AMBIGUOUS").mean()),
                        "volume_confirmed_break_rate": float(
                            pd.to_numeric(subset.get("volume_confirmed_break", 0), errors="coerce")
                            .fillna(0)
                            .mean()
                        ),
                        "mean_close_return_15m": float(
                            pd.to_numeric(subset.get("close_return_15m"), errors="coerce").mean()
                        ),
                        "mean_mfe_15m_bps": float(
                            pd.to_numeric(subset.get("mfe_15m_bps"), errors="coerce").mean()
                        ),
                        "mean_mae_15m_bps": float(
                            pd.to_numeric(subset.get("mae_15m_bps"), errors="coerce").mean()
                        ),
                    }
                )
    return pd.DataFrame(rows)


def grouped_wall_outcomes(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    group_fields = [
        field
        for field in ("side", "morphology", "wall_dominant_morphology", "wall_is_ghost")
        if field in events.columns
    ]
    rows: list[dict[str, Any]] = []
    for fields in ([field] for field in group_fields):
        for key, subset in events.groupby(fields, observed=True, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            primary = subset["primary_outcome"].astype(str)
            row = {"group_field": fields[0], "group_value": str(key[0]), "events": len(subset)}
            row.update(
                {
                    "bounce_rate": float((primary == "BOUNCE").mean()),
                    "break_rate": float((primary == "BREAK").mean()),
                    "volume_confirmed_break_rate": float(
                        pd.to_numeric(subset.get("volume_confirmed_break", 0), errors="coerce").fillna(0).mean()
                    ),
                    "mean_close_return_15m": float(
                        pd.to_numeric(subset.get("close_return_15m"), errors="coerce").mean()
                    ),
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def causal_replay_audit(events: pd.DataFrame) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()
    out = events[
        [
            column
            for column in (
                "wall_id",
                "side",
                "available_time_ms",
                "touch_time_ms",
                "touch_bar_end_ms",
                "wall_age_seconds",
                "price_low",
                "price_high",
                "primary_outcome",
            )
            if column in events.columns
        ]
    ].copy()
    out["state_available_before_touch_bar_flag"] = (
        pd.to_numeric(events["available_time_ms"], errors="coerce")
        <= pd.to_numeric(events["touch_time_ms"], errors="coerce")
    ).astype("int8")
    out["wall_existed_before_touch_flag"] = (
        pd.to_numeric(events["wall_age_seconds"], errors="coerce") >= 0
    ).astype("int8")
    out["causal_audit_pass"] = (
        out["state_available_before_touch_bar_flag"].astype(bool)
        & out["wall_existed_before_touch_flag"].astype(bool)
    ).astype("int8")
    return out


def _first_passage(
    path: pd.DataFrame,
    side: str,
    bounce_price: float,
    break_price: float,
) -> tuple[str, int | None, int | None]:
    if path.empty:
        return "NO_DATA", None, None
    for pos, (_, bar) in enumerate(path.iterrows()):
        if side == "bid":
            bounce_hit = float(bar["high"]) >= bounce_price
            break_hit = float(bar["low"]) <= break_price
        else:
            bounce_hit = float(bar["low"]) <= bounce_price
            break_hit = float(bar["high"]) >= break_price
        if bounce_hit and break_hit:
            return "AMBIGUOUS", int(bar["bar_start_ms"]), pos
        if break_hit:
            return "BREAK", int(bar["bar_start_ms"]), pos
        if bounce_hit:
            return "BOUNCE", int(bar["bar_start_ms"]), pos
    return "NEITHER", None, None


def _break_continuation(path: pd.DataFrame, side: str, break_price: float) -> float:
    if path.empty or break_price <= 0:
        return np.nan
    if side == "bid":
        minimum = float(path["low"].min())
        return (break_price / minimum - 1.0) * 10_000.0 if minimum > 0 else np.nan
    return (float(path["high"].max()) / break_price - 1.0) * 10_000.0


def _price_overlap_fraction(a_low: float, a_high: float, b_low: float, b_high: float) -> float:
    intersection = max(0.0, min(a_high, b_high) - max(a_low, b_low))
    denominator = max(min(a_high - a_low, b_high - b_low), 1e-12)
    return intersection / denominator


def _normalized_slope(values: np.ndarray) -> float:
    clean = np.asarray(values, dtype=float)
    if len(clean) < 2 or not np.isfinite(clean).any():
        return 0.0
    y = np.nan_to_num(clean, nan=float(np.nanmedian(clean)))
    x = np.arange(len(y), dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    return slope / max(float(np.nanmean(np.abs(y))), 1e-12)


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator <= 1e-12:
        return np.nan
    return numerator / denominator


__all__ = [
    "CausalDepthReference",
    "WallCandidate",
    "WallDiscoveryConfig",
    "WallDiscoveryEngine",
    "attach_touch_outcomes",
    "build_wall_touch_events",
    "causal_replay_audit",
    "chronological_split",
    "discover_wall_states",
    "environment_feature_uplift",
    "extract_snapshot_candidates",
    "grouped_wall_outcomes",
    "prepare_execution_bars",
    "touch_event_from_state",
]
