#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal reversal-zone process helpers for research 06.

A zone is not a retrospective best window.  It starts when a frozen 2023
anchor model enters a high-score state and is updated only by later *closed*
1m bars that are already visible.  Every state can therefore be replayed
online.  Future closes are used only by the shared research labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from src.research_common.progress import ProgressReporter
except Exception:  # pragma: no cover
    ProgressReporter = None  # type: ignore[assignment]

from research.market_structure.swing_low_typology.common.reversal_opportunity import (
    opportunity_event_metrics,
)

EPS = 1e-12
ZONE_PROCESS_GROUP = "Z1_process"


@dataclass(frozen=True)
class ZoneBuildResult:
    frame: pd.DataFrame
    dictionary: pd.DataFrame
    summary: pd.DataFrame


def attach_zone_split(
    frame: pd.DataFrame,
    *,
    zone_fit_end: pd.Timestamp,
    zone_validation_end: pd.Timestamp,
) -> pd.DataFrame:
    out = frame.copy()
    ts = pd.to_datetime(out["extreme_time"])
    out["zone_split"] = np.where(
        ts <= zone_fit_end,
        "zone_fit",
        np.where(ts <= zone_validation_end, "zone_validation", "holdout"),
    )
    out["zone_year"] = ts.dt.year
    return out


def purge_zone_label_overlap(
    frame: pd.DataFrame,
    *,
    zone_fit_end: pd.Timestamp,
    zone_validation_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = frame.copy()
    label_end = pd.to_datetime(out["label_end_time"])
    split = out["zone_split"].astype(str)
    crossed_fit = split.eq("zone_fit") & (label_end > zone_fit_end)
    crossed_validation = split.eq("zone_validation") & (label_end > zone_validation_end)
    remove = crossed_fit | crossed_validation
    summary = pd.DataFrame(
        [
            {"zone_split": "zone_fit", "removed_cross_boundary": int(crossed_fit.sum())},
            {"zone_split": "zone_validation", "removed_cross_boundary": int(crossed_validation.sum())},
            {"zone_split": "holdout", "removed_cross_boundary": 0},
            {"zone_split": "ALL", "removed_cross_boundary": int(remove.sum())},
        ]
    )
    return out.loc[~remove].reset_index(drop=True), summary


def _numeric_array(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.zeros(len(frame), dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float, copy=False)


def _ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(denominator) or abs(denominator) <= EPS:
        return np.nan
    return float(numerator / denominator)


def _slope(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return 0.0
    y = values[finite].astype(float)
    x = np.arange(len(values), dtype=float)[finite]
    x -= x.mean()
    denominator = float(np.dot(x, x))
    if denominator <= EPS:
        return 0.0
    return float(np.dot(x, y - y.mean()) / denominator)


def _segment_ratio(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    if not len(values):
        return np.nan
    if weights is None:
        return float(np.nanmean(values))
    denominator = float(np.nansum(weights))
    return _ratio(float(np.nansum(values)), denominator)


def _impact_ratio(close_segment: np.ndarray, delta_segment: np.ndarray, notional_segment: np.ndarray) -> float:
    if len(close_segment) < 2 or close_segment[0] <= EPS:
        return np.nan
    negative_flow = max(0.0, -_ratio(float(np.nansum(delta_segment)), float(np.nansum(notional_segment))))
    negative_return = max(0.0, 1.0 - float(close_segment[-1] / close_segment[0]))
    if not np.isfinite(negative_flow) or negative_flow <= EPS:
        return 0.0 if negative_return <= EPS else np.nan
    return float(negative_return / negative_flow)


def _process_dictionary() -> pd.DataFrame:
    descriptions = {
        "zone_age_bars": "bars elapsed since the first causal high-score zone state",
        "zone_observation_number": "ordinal high-score observation inside the active zone",
        "zone_observation_density": "high-score observations divided by elapsed bars",
        "zone_score_current": "current frozen anchor score percentile",
        "zone_score_first": "first anchor score percentile in this zone",
        "zone_score_mean": "mean anchor score percentile observed so far",
        "zone_score_max": "maximum anchor score percentile observed so far",
        "zone_score_change_from_first": "current minus first anchor score percentile",
        "zone_score_change_from_previous": "current minus previous zone observation score",
        "zone_score_slope": "causal slope of anchor score across observed zone states",
        "zone_return_from_start": "current close return from zone start close",
        "zone_low_return_from_start": "lowest low so far relative to zone start close",
        "zone_rebound_from_low": "current close rebound from the lowest low seen so far",
        "zone_range_position": "current close position inside the zone high-low range so far",
        "zone_price_slope": "normalized close-path slope from zone start through current bar",
        "zone_new_low_count": "number of causal running-low updates inside the zone",
        "zone_support_test_count": "bars testing within tolerance of the lowest low known now",
        "zone_support_test_density": "support-test count divided by zone length",
        "zone_bars_since_low": "bars elapsed since the lowest low seen so far",
        "zone_cumulative_delta_ratio": "cumulative aggressive delta divided by zone notional",
        "zone_cumulative_large_delta_ratio": "cumulative large-trade delta divided by zone notional",
        "zone_negative_delta_share": "share of zone bars with negative delta",
        "zone_recent_delta_ratio": "recent-half cumulative delta ratio",
        "zone_early_delta_ratio": "early-half cumulative delta ratio",
        "zone_delta_improvement": "recent minus early cumulative delta ratio",
        "zone_recent_large_delta_ratio": "recent-half large delta ratio",
        "zone_early_large_delta_ratio": "early-half large delta ratio",
        "zone_large_delta_improvement": "recent minus early large delta ratio",
        "zone_early_sell_impact": "early price decline per unit negative delta",
        "zone_recent_sell_impact": "recent price decline per unit negative delta",
        "zone_absorption_improvement": "early minus recent sell-impact ratio",
        "zone_absorption_level": "negative cumulative delta with limited price decline proxy",
        "zone_notional_recent_vs_early": "recent-half mean notional divided by early-half mean",
        "zone_trades_recent_vs_early": "recent-half mean trade count divided by early-half mean",
        "zone_range_recent_vs_early": "recent-half mean bar range divided by early-half mean",
        "zone_return_vol_recent_vs_early": "recent-half return volatility divided by early-half volatility",
        "zone_rebound_strengthening": "recent-half maximum rebound minus early-half maximum rebound",
        "zone_low_progression_slope": "slope of running low relative to zone start price",
        "zone_close_above_previous": "whether current close is above prior close",
        "zone_reclaim_10bp": "whether current close has rebounded at least 10bp from zone low",
        "zone_reclaim_20bp": "whether current close has rebounded at least 20bp from zone low",
    }
    return pd.DataFrame(
        [
            {
                "feature": name,
                "feature_group": ZONE_PROCESS_GROUP,
                "description": description,
                "source": "closed 1m bars from causal zone start through current closed bar",
                "available_rule": "all source bars <= current extreme_time",
            }
            for name, description in descriptions.items()
        ]
    )


def build_causal_zone_states(
    bars: pd.DataFrame,
    scored_candidates: pd.DataFrame,
    *,
    activation_percentile: float = 95.0,
    max_gap_bars: int = 5,
    max_zone_bars: int = 120,
    support_tolerance_bp: float = 25.0,
    score_column: str = "anchor_score_0_100",
    show_progress: bool = True,
) -> ZoneBuildResult:
    """Build variable-length online zone states from frozen anchor scores.

    Only candidates already at or above ``activation_percentile`` are zone
    observations.  A zone ends when the next observation is too far away or
    the maximum causal duration is exceeded.  The current row never depends on
    a later observation or on the eventual zone end.
    """

    if not 0.0 < float(activation_percentile) < 100.0:
        raise ValueError("activation_percentile must be between 0 and 100")
    if max_gap_bars < 1 or max_zone_bars < max_gap_bars:
        raise ValueError("invalid zone gap/duration")
    required = {"event_id", "extreme_pos", "extreme_time", "zone_split", score_column}
    missing = sorted(required - set(scored_candidates.columns))
    if missing:
        raise RuntimeError(f"zone builder missing candidate columns: {missing}")

    selected = scored_candidates[
        pd.to_numeric(scored_candidates[score_column], errors="coerce") >= float(activation_percentile)
    ].copy()
    if selected.empty:
        return ZoneBuildResult(pd.DataFrame(), _process_dictionary(), pd.DataFrame())
    selected = selected.sort_values(["zone_split", "extreme_pos", "event_id"]).reset_index(drop=True)

    zone_ids: list[str] = []
    zone_starts: list[int] = []
    observation_numbers: list[int] = []
    zone_counter: dict[str, int] = {}
    current_split = ""
    current_zone = ""
    current_start = -1
    previous_pos = -10**18
    observation = 0
    for row in selected.itertuples(index=False):
        split_name = str(getattr(row, "zone_split"))
        position = int(getattr(row, "extreme_pos"))
        new_zone = (
            split_name != current_split
            or position - previous_pos > int(max_gap_bars)
            or (current_start >= 0 and position - current_start > int(max_zone_bars))
        )
        if new_zone:
            zone_counter[split_name] = zone_counter.get(split_name, 0) + 1
            current_zone = f"Z_{split_name}_{zone_counter[split_name]:07d}"
            current_start = position
            observation = 1
        else:
            observation += 1
        zone_ids.append(current_zone)
        zone_starts.append(current_start)
        observation_numbers.append(observation)
        current_split = split_name
        previous_pos = position

    selected["zone_id"] = zone_ids
    selected["zone_start_pos"] = np.asarray(zone_starts, dtype=np.int64)
    selected["zone_observation_number"] = np.asarray(observation_numbers, dtype=np.int32)

    open_values = _numeric_array(bars, "open")
    high_values = _numeric_array(bars, "high")
    low_values = _numeric_array(bars, "low")
    close_values = _numeric_array(bars, "close")
    notional_values = _numeric_array(bars, "notional")
    trades_values = _numeric_array(bars, "trades_count")
    delta_values = _numeric_array(bars, "delta_notional")
    large_delta_values = _numeric_array(bars, "large_delta_notional")
    score_values = pd.to_numeric(selected[score_column], errors="coerce").to_numpy(dtype=float)
    tolerance = float(support_tolerance_bp) / 10_000.0

    feature_names = _process_dictionary()["feature"].tolist()
    # zone_observation_number is causal metadata already materialized above;
    # keep it in the feature dictionary without creating a duplicate column.
    columns: dict[str, list[float]] = {name: [] for name in feature_names if name not in selected.columns}
    reporter = ProgressReporter("[zones] causal process states", total=len(selected), every=max(1, min(10_000, len(selected)))) if ProgressReporter and show_progress else None
    processed = 0

    for _, group in selected.groupby("zone_id", sort=False):
        group_indices = group.index.to_numpy(dtype=np.int64)
        start_pos = int(group.iloc[0]["zone_start_pos"])
        observed_scores: list[float] = []
        previous_score = np.nan
        for local_number, row_index in enumerate(group_indices, start=1):
            current_pos = int(selected.at[row_index, "extreme_pos"])
            current_score = float(score_values[row_index])
            observed_scores.append(current_score)
            start = max(0, start_pos)
            end = min(len(bars), current_pos + 1)
            close_seg = close_values[start:end]
            high_seg = high_values[start:end]
            low_seg = low_values[start:end]
            notional_seg = notional_values[start:end]
            trades_seg = trades_values[start:end]
            delta_seg = delta_values[start:end]
            large_delta_seg = large_delta_values[start:end]
            length = len(close_seg)
            split_at = max(1, length // 2)
            early_slice = slice(0, split_at)
            recent_slice = slice(split_at, length) if split_at < length else slice(max(0, length - 1), length)

            start_close = float(close_seg[0]) if length else np.nan
            current_close = float(close_seg[-1]) if length else np.nan
            zone_low = float(np.nanmin(low_seg)) if length else np.nan
            zone_high = float(np.nanmax(high_seg)) if length else np.nan
            running_low = np.minimum.accumulate(low_seg) if length else np.asarray([], dtype=float)
            new_low_count = int(np.sum(np.r_[True, np.diff(running_low) < -EPS])) if length else 0
            low_index = int(np.nanargmin(low_seg)) if length else 0
            support_count = int(np.sum(low_seg <= zone_low * (1.0 + tolerance))) if length and zone_low > EPS else 0
            returns = np.r_[0.0, np.diff(close_seg) / np.maximum(close_seg[:-1], EPS)] if length else np.asarray([], dtype=float)
            ranges = (high_seg - low_seg) / np.maximum(close_seg, EPS) if length else np.asarray([], dtype=float)
            normalized_close = close_seg / start_close - 1.0 if length and start_close > EPS else np.zeros(length)
            normalized_running_low = running_low / start_close - 1.0 if length and start_close > EPS else np.zeros(length)

            early_delta_ratio = _ratio(float(np.nansum(delta_seg[early_slice])), float(np.nansum(notional_seg[early_slice])))
            recent_delta_ratio = _ratio(float(np.nansum(delta_seg[recent_slice])), float(np.nansum(notional_seg[recent_slice])))
            early_large_ratio = _ratio(float(np.nansum(large_delta_seg[early_slice])), float(np.nansum(notional_seg[early_slice])))
            recent_large_ratio = _ratio(float(np.nansum(large_delta_seg[recent_slice])), float(np.nansum(notional_seg[recent_slice])))
            cumulative_delta_ratio = _ratio(float(np.nansum(delta_seg)), float(np.nansum(notional_seg)))
            cumulative_large_ratio = _ratio(float(np.nansum(large_delta_seg)), float(np.nansum(notional_seg)))
            early_impact = _impact_ratio(close_seg[early_slice], delta_seg[early_slice], notional_seg[early_slice])
            recent_impact = _impact_ratio(close_seg[recent_slice], delta_seg[recent_slice], notional_seg[recent_slice])

            early_running_low = np.minimum.accumulate(low_seg[early_slice]) if len(low_seg[early_slice]) else np.asarray([], dtype=float)
            recent_running_low = np.minimum.accumulate(low_seg[recent_slice]) if len(low_seg[recent_slice]) else np.asarray([], dtype=float)
            early_rebound = float(np.nanmax(close_seg[early_slice] / np.maximum(early_running_low, EPS) - 1.0)) if len(early_running_low) else 0.0
            recent_rebound = float(np.nanmax(close_seg[recent_slice] / np.maximum(recent_running_low, EPS) - 1.0)) if len(recent_running_low) else 0.0

            score_array = np.asarray(observed_scores, dtype=float)
            columns["zone_age_bars"].append(float(current_pos - start_pos))
            if "zone_observation_number" in columns:
                columns["zone_observation_number"].append(float(local_number))
            columns["zone_observation_density"].append(float(local_number / max(1, current_pos - start_pos + 1)))
            columns["zone_score_current"].append(current_score)
            columns["zone_score_first"].append(float(score_array[0]))
            columns["zone_score_mean"].append(float(np.nanmean(score_array)))
            columns["zone_score_max"].append(float(np.nanmax(score_array)))
            columns["zone_score_change_from_first"].append(float(current_score - score_array[0]))
            columns["zone_score_change_from_previous"].append(0.0 if not np.isfinite(previous_score) else float(current_score - previous_score))
            columns["zone_score_slope"].append(_slope(score_array))
            columns["zone_return_from_start"].append(_ratio(current_close, start_close) - 1.0 if start_close > EPS else np.nan)
            columns["zone_low_return_from_start"].append(_ratio(zone_low, start_close) - 1.0 if start_close > EPS else np.nan)
            columns["zone_rebound_from_low"].append(_ratio(current_close, zone_low) - 1.0 if zone_low > EPS else np.nan)
            columns["zone_range_position"].append(_ratio(current_close - zone_low, zone_high - zone_low))
            columns["zone_price_slope"].append(_slope(normalized_close))
            columns["zone_new_low_count"].append(float(new_low_count))
            columns["zone_support_test_count"].append(float(support_count))
            columns["zone_support_test_density"].append(float(support_count / max(1, length)))
            columns["zone_bars_since_low"].append(float(length - 1 - low_index))
            columns["zone_cumulative_delta_ratio"].append(cumulative_delta_ratio)
            columns["zone_cumulative_large_delta_ratio"].append(cumulative_large_ratio)
            columns["zone_negative_delta_share"].append(float(np.mean(delta_seg < 0.0)) if length else np.nan)
            columns["zone_recent_delta_ratio"].append(recent_delta_ratio)
            columns["zone_early_delta_ratio"].append(early_delta_ratio)
            columns["zone_delta_improvement"].append(recent_delta_ratio - early_delta_ratio if np.isfinite(recent_delta_ratio) and np.isfinite(early_delta_ratio) else np.nan)
            columns["zone_recent_large_delta_ratio"].append(recent_large_ratio)
            columns["zone_early_large_delta_ratio"].append(early_large_ratio)
            columns["zone_large_delta_improvement"].append(recent_large_ratio - early_large_ratio if np.isfinite(recent_large_ratio) and np.isfinite(early_large_ratio) else np.nan)
            columns["zone_early_sell_impact"].append(early_impact)
            columns["zone_recent_sell_impact"].append(recent_impact)
            columns["zone_absorption_improvement"].append(early_impact - recent_impact if np.isfinite(early_impact) and np.isfinite(recent_impact) else np.nan)
            negative_flow = max(0.0, -cumulative_delta_ratio) if np.isfinite(cumulative_delta_ratio) else np.nan
            price_decline = max(0.0, 1.0 - _ratio(current_close, start_close)) if start_close > EPS else np.nan
            columns["zone_absorption_level"].append(negative_flow - price_decline if np.isfinite(negative_flow) and np.isfinite(price_decline) else np.nan)
            columns["zone_notional_recent_vs_early"].append(_ratio(float(np.nanmean(notional_seg[recent_slice])), float(np.nanmean(notional_seg[early_slice]))))
            columns["zone_trades_recent_vs_early"].append(_ratio(float(np.nanmean(trades_seg[recent_slice])), float(np.nanmean(trades_seg[early_slice]))))
            columns["zone_range_recent_vs_early"].append(_ratio(float(np.nanmean(ranges[recent_slice])), float(np.nanmean(ranges[early_slice]))))
            columns["zone_return_vol_recent_vs_early"].append(_ratio(float(np.nanstd(returns[recent_slice])), float(np.nanstd(returns[early_slice]))))
            columns["zone_rebound_strengthening"].append(float(recent_rebound - early_rebound))
            columns["zone_low_progression_slope"].append(_slope(normalized_running_low))
            columns["zone_close_above_previous"].append(float(length >= 2 and close_seg[-1] > close_seg[-2]))
            rebound = _ratio(current_close, zone_low) - 1.0 if zone_low > EPS else np.nan
            columns["zone_reclaim_10bp"].append(float(np.isfinite(rebound) and rebound >= 0.0010))
            columns["zone_reclaim_20bp"].append(float(np.isfinite(rebound) and rebound >= 0.0020))
            previous_score = current_score
            processed += 1
            if reporter is not None and processed < len(selected):
                reporter.update(processed)
    if reporter is not None:
        reporter.close()

    process_frame = pd.DataFrame({name: np.asarray(values, dtype=np.float32) for name, values in columns.items()})
    result = pd.concat([selected.reset_index(drop=True), process_frame], axis=1)
    result["zone_end_known_at_state"] = False
    result["zone_state_count"] = result.groupby("zone_id")["event_id"].transform("size").astype(np.int32)
    # Training duplicate control must itself be causal.  Do not weight an early
    # state by the eventual zone size (which is unknown then); discount later
    # repeated observations only by the count already observed at that state.
    result["episode_weight"] = (1.0 / result["zone_observation_number"].clip(lower=1)).astype(np.float32)
    summary_rows: list[dict[str, object]] = []
    for split_name, split_frame in result.groupby("zone_split", sort=False):
        zone_sizes = split_frame.groupby("zone_id").size()
        zone_durations = split_frame.groupby("zone_id").agg(start=("extreme_pos", "min"), end=("extreme_pos", "max"))
        duration = zone_durations["end"] - zone_durations["start"]
        summary_rows.append(
            {
                "zone_split": split_name,
                "activation_percentile": float(activation_percentile),
                "state_count": int(len(split_frame)),
                "zone_count": int(split_frame["zone_id"].nunique()),
                "median_observations_per_zone": float(zone_sizes.median()),
                "p90_observations_per_zone": float(zone_sizes.quantile(0.90)),
                "median_zone_duration_bars": float(duration.median()),
                "p90_zone_duration_bars": float(duration.quantile(0.90)),
            }
        )
    return ZoneBuildResult(result, _process_dictionary(), pd.DataFrame(summary_rows))


def zone_feature_groups(snapshot_features: Sequence[str], process_features: Sequence[str]) -> dict[str, tuple[str, ...]]:
    snapshot = tuple(dict.fromkeys(["anchor_score_0_100", "anchor_raw_score", *snapshot_features]))
    process = tuple(dict.fromkeys(process_features))
    return {
        "Z0_snapshot": snapshot,
        "Z1_process": process,
        "Z2_hybrid": tuple(dict.fromkeys([*snapshot, *process])),
    }


def select_first_zone_signal(
    frame: pd.DataFrame,
    *,
    score_column: str,
    threshold: float,
    minimum_observations: int,
    cooldown_bars: int = 0,
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    eligible = frame[
        (pd.to_numeric(frame[score_column], errors="coerce") >= float(threshold))
        & (pd.to_numeric(frame["zone_observation_number"], errors="coerce") >= int(minimum_observations))
    ].sort_values(["extreme_pos", "zone_id"])
    first = eligible.groupby("zone_id", sort=False, as_index=False).head(1).sort_values("extreme_pos")
    if cooldown_bars > 0 and not first.empty:
        kept: list[int] = []
        last_position = -10**18
        positions = pd.to_numeric(first["extreme_pos"], errors="raise").to_numpy(dtype=np.int64)
        for i, position in enumerate(positions):
            if int(position) - last_position < int(cooldown_bars):
                continue
            kept.append(i)
            last_position = int(position)
        first = first.iloc[kept]
    out = first.copy().reset_index(drop=True)
    out["zone_signal_threshold"] = float(threshold)
    out["minimum_zone_observations"] = int(minimum_observations)
    out["zone_signal_cooldown_bars"] = int(cooldown_bars)
    return out


def zone_trigger_grid(
    reference: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    score_column: str,
    fractions: Sequence[float],
    minimum_observations_grid: Sequence[int],
    cooldowns: Sequence[int],
    threshold_source: str,
) -> tuple[pd.DataFrame, dict[tuple[float, int, int], pd.DataFrame]]:
    reference_scores = pd.to_numeric(reference[score_column], errors="coerce").dropna()
    if reference_scores.empty:
        raise RuntimeError("zone trigger reference scores are empty")
    rows: list[dict[str, object]] = []
    selected: dict[tuple[float, int, int], pd.DataFrame] = {}
    for fraction_raw in fractions:
        fraction = float(fraction_raw)
        threshold = float(reference_scores.quantile(1.0 - fraction))
        for observations_raw in minimum_observations_grid:
            observations = int(observations_raw)
            for cooldown_raw in cooldowns:
                cooldown = int(cooldown_raw)
                events = select_first_zone_signal(
                    evaluation,
                    score_column=score_column,
                    threshold=threshold,
                    minimum_observations=observations,
                    cooldown_bars=cooldown,
                )
                metrics = opportunity_event_metrics(events)
                rows.append(
                    {
                        "threshold_source": threshold_source,
                        "top_fraction": fraction,
                        "score_threshold": threshold,
                        "minimum_zone_observations": observations,
                        "cooldown_bars": cooldown,
                        **metrics,
                    }
                )
                selected[(fraction, observations, cooldown)] = events
    return pd.DataFrame(rows), selected


def choose_zone_trigger_spec(grid: pd.DataFrame, *, minimum_events: int = 20) -> pd.Series:
    if grid.empty:
        raise RuntimeError("zone trigger grid is empty")
    data = grid.copy()
    eligible = data[pd.to_numeric(data["event_count"], errors="coerce") >= int(minimum_events)].copy()
    if eligible.empty:
        eligible = data.copy()
    eligible["selection_objective"] = (
        0.55 * pd.to_numeric(eligible["clean_0p25_wilson_lower"], errors="coerce").fillna(-1.0)
        + 0.35 * pd.to_numeric(eligible["tp_wilson_lower"], errors="coerce").fillna(-1.0)
        - 0.10 * (pd.to_numeric(eligible["median_mae_before_tp_pct"], errors="coerce").fillna(2.0) / 2.0)
    )
    return eligible.sort_values(
        [
            "selection_objective",
            "clean_0p25_rate",
            "tp_rate",
            "median_mae_before_tp_pct",
            "event_count",
            "minimum_zone_observations",
            "top_fraction",
        ],
        ascending=[False, False, False, True, False, True, True],
    ).iloc[0]


def observation_timing_metrics(frame: pd.DataFrame, observations: Sequence[int] = (1, 2, 3, 4)) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for observation in observations:
        events = frame[pd.to_numeric(frame["zone_observation_number"], errors="coerce") == int(observation)].copy()
        metrics = opportunity_event_metrics(events)
        rows.append({"zone_observation_number": int(observation), **metrics})
    return pd.DataFrame(rows)
