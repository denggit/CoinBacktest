#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Walk-forward multi-objective reversal research helpers.

The helpers in this module deliberately avoid retrospective Swing-Low cluster
IDs.  They provide three causal feature layers:

* current 1m snapshot features built by ``reversal_opportunity``;
* train-fitted soft shock/trend/base mechanism scores;
* broad candidate-region process features whose state ends at the current
  closed bar.

Future closes are labels only.  No function in this module uses a future price,
future region end, or eventual region size as a model feature.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

try:
    from src.research_common.progress import ProgressReporter
except Exception:  # pragma: no cover
    ProgressReporter = None  # type: ignore[assignment]

EPS = 1e-12
REGION_FEATURE_GROUP = "U2_region_process"
MECHANISM_FEATURE_GROUP = "U1_soft_mechanism"


@dataclass(frozen=True)
class RegionBuildResult:
    frame: pd.DataFrame
    dictionary: pd.DataFrame
    summary: pd.DataFrame


@dataclass(frozen=True)
class SoftMechanismTransformer:
    component_center: Mapping[str, float]
    component_scale: Mapping[str, float]
    train_raw_scores: Mapping[str, np.ndarray]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        raw = _mechanism_raw_scores(frame, self.component_center, self.component_scale)
        out = pd.DataFrame(index=frame.index)
        for name in ("shock", "trend", "base"):
            reference = np.asarray(self.train_raw_scores[name], dtype=float)
            values = pd.to_numeric(raw[f"mechanism_{name}_raw"], errors="coerce").to_numpy(dtype=float)
            finite_reference = np.sort(reference[np.isfinite(reference)])
            percentile = np.full(len(values), np.nan, dtype=float)
            finite = np.isfinite(values)
            if finite_reference.size:
                percentile[finite] = (
                    np.searchsorted(finite_reference, values[finite], side="right")
                    / float(finite_reference.size)
                    * 100.0
                )
            out[f"mechanism_{name}_score"] = percentile.astype(np.float32)
        score_matrix = out[["mechanism_shock_score", "mechanism_trend_score", "mechanism_base_score"]].to_numpy(dtype=float)
        ordered = np.sort(score_matrix, axis=1)
        out["mechanism_top_margin"] = (ordered[:, -1] - ordered[:, -2]).astype(np.float32)
        probability = np.clip(score_matrix / 100.0, 1e-6, 1.0)
        probability = probability / np.maximum(probability.sum(axis=1, keepdims=True), EPS)
        entropy = -(probability * np.log(probability)).sum(axis=1) / np.log(3.0)
        out["mechanism_entropy"] = entropy.astype(np.float32)
        dominant_index = np.nanargmax(np.where(np.isfinite(score_matrix), score_matrix, -np.inf), axis=1)
        names = np.asarray(["shock", "trend", "base"], dtype=object)
        out["mechanism_dominant"] = names[dominant_index]
        return out


@dataclass(frozen=True)
class QuantileRiskModel:
    feature_columns: tuple[str, ...]
    medians: pd.Series
    quantile: float
    model: object | None
    constant_value: float

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(frame), self.constant_value, dtype=float)
        x = (
            frame.reindex(columns=self.feature_columns)
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(self.medians)
        )
        prediction = np.asarray(self.model.predict(x), dtype=float)
        return np.clip(prediction, 0.0, 5.0)


def _numeric_array(frame: pd.DataFrame, column: str, *, default: float = 0.0) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), float(default), dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(float(default)).to_numpy(dtype=float, copy=False)


def _safe_divide(numerator: np.ndarray | float, denominator: np.ndarray | float) -> np.ndarray:
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    return np.divide(num, den, out=np.zeros(np.broadcast_shapes(num.shape, den.shape), dtype=float), where=np.abs(den) > EPS)


def _prefix_sum(values: np.ndarray) -> np.ndarray:
    clean = np.where(np.isfinite(values), values, 0.0)
    return np.r_[0.0, np.cumsum(clean, dtype=float)]


def _interval_sum(prefix: np.ndarray, start: np.ndarray, end_inclusive: np.ndarray) -> np.ndarray:
    return prefix[end_inclusive + 1] - prefix[start]


def _feature_dictionary() -> pd.DataFrame:
    descriptions = {
        "region_age_bars": "bars since the causal broad candidate region began",
        "region_observation_number": "current candidate observation number inside the region",
        "region_candidate_density": "candidate observations divided by elapsed region bars",
        "region_return_from_start": "current close return from region-start close",
        "region_low_progression": "current running region low versus region-start low",
        "region_rebound_from_low": "current close rebound from running region low",
        "region_drawdown_from_high": "current close distance from running region high",
        "region_new_low_count": "number of causal running-low updates through current bar",
        "region_candidate_retest_count": "candidate lows within tolerance of the running candidate low",
        "region_bars_since_low": "bars since the latest running region low",
        "region_cumulative_delta_ratio": "cumulative delta divided by cumulative notional",
        "region_cumulative_large_delta_ratio": "cumulative large delta divided by cumulative notional",
        "region_recent_delta_ratio": "recent-half cumulative delta ratio",
        "region_early_delta_ratio": "early-half cumulative delta ratio",
        "region_delta_improvement": "recent minus early cumulative delta ratio",
        "region_recent_large_delta_ratio": "recent-half large delta ratio",
        "region_early_large_delta_ratio": "early-half large delta ratio",
        "region_large_delta_improvement": "recent minus early large delta ratio",
        "region_absorption_improvement": "early minus recent negative-flow price-impact proxy",
        "region_notional_recent_vs_early": "recent-half mean notional divided by early-half mean",
        "region_trades_recent_vs_early": "recent-half mean trade count divided by early-half mean",
        "region_range_recent_vs_early": "recent-half mean bar range divided by early-half mean",
        "region_vol_recent_vs_early": "recent-half close-return volatility divided by early-half volatility",
        "region_close_above_previous": "current close is above the previous closed bar",
        "region_reclaim_10bp": "current close is at least 10bp above running region low",
        "region_reclaim_20bp": "current close is at least 20bp above running region low",
    }
    return pd.DataFrame(
        [
            {
                "feature": name,
                "feature_group": REGION_FEATURE_GROUP,
                "description": description,
                "source": "closed 1m bars from causal candidate-region start through current closed bar",
                "available_rule": "all source bars <= current extreme_time; region end and eventual size unused",
            }
            for name, description in descriptions.items()
        ]
    )


def build_broad_candidate_regions(
    bars: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    max_gap_bars: int = 2,
    max_region_bars: int = 120,
    retest_tolerance_bp: float = 25.0,
    show_progress: bool = True,
) -> RegionBuildResult:
    """Attach causal broad-region process features to every candidate row.

    Region membership depends only on candidate timestamps seen so far.  A new
    region starts after a gap larger than ``max_gap_bars`` or when the current
    region has already lasted ``max_region_bars``.  Every feature for a row is
    calculated from the region start through that row's closed bar.
    """

    if candidates.empty:
        return RegionBuildResult(pd.DataFrame(), _feature_dictionary(), pd.DataFrame())
    if max_gap_bars < 1 or max_region_bars < max_gap_bars:
        raise ValueError("invalid broad-region gap/duration")
    required = {"event_id", "extreme_pos", "extreme_time"}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise RuntimeError(f"broad region builder missing columns: {missing}")

    data = candidates.sort_values(["extreme_pos", "event_id"]).reset_index(drop=True).copy()
    positions = pd.to_numeric(data["extreme_pos"], errors="raise").to_numpy(dtype=np.int64)
    if positions.min(initial=0) < 0 or positions.max(initial=0) >= len(bars):
        raise RuntimeError("candidate position is outside loaded bars")

    n = len(data)
    region_number = np.empty(n, dtype=np.int64)
    region_start = np.empty(n, dtype=np.int64)
    observation = np.empty(n, dtype=np.int32)
    current_region = 0
    current_start = int(positions[0])
    previous = -10**18
    obs = 0
    for i, position in enumerate(positions):
        new_region = i == 0 or position - previous > int(max_gap_bars) or position - current_start > int(max_region_bars)
        if new_region:
            current_region += 1
            current_start = int(position)
            obs = 1
        else:
            obs += 1
        region_number[i] = current_region
        region_start[i] = current_start
        observation[i] = obs
        previous = int(position)

    data["causal_region_number"] = region_number
    data["causal_region_id"] = np.asarray([f"CR_{value:08d}" for value in region_number], dtype=object)
    data["causal_region_start_pos"] = region_start
    data["region_observation_number"] = observation

    open_values = _numeric_array(bars, "open")
    high_values = _numeric_array(bars, "high")
    low_values = _numeric_array(bars, "low")
    close_values = _numeric_array(bars, "close")
    notional_values = _numeric_array(bars, "notional")
    trades_values = _numeric_array(bars, "trades_count")
    delta_values = _numeric_array(bars, "delta_notional")
    large_delta_values = _numeric_array(bars, "large_delta_notional")
    bar_range_values = _safe_divide(high_values - low_values, np.maximum(close_values, EPS))
    close_return_values = np.r_[0.0, np.diff(close_values) / np.maximum(close_values[:-1], EPS)]

    feature_names = _feature_dictionary()["feature"].astype(str).tolist()
    arrays: dict[str, np.ndarray] = {name: np.full(n, np.nan, dtype=np.float32) for name in feature_names}
    reporter = (
        ProgressReporter("[regions] broad causal states", total=n, every=max(1, min(50_000, n)))
        if ProgressReporter and show_progress
        else None
    )
    boundaries = np.r_[0, np.flatnonzero(np.diff(region_number) != 0) + 1, n]
    processed = 0
    tolerance = float(retest_tolerance_bp) / 10_000.0

    for left, right in zip(boundaries[:-1], boundaries[1:]):
        group_positions = positions[left:right]
        start_pos = int(group_positions[0])
        last_pos = int(group_positions[-1])
        segment_slice = slice(start_pos, last_pos + 1)
        low_seg = low_values[segment_slice]
        high_seg = high_values[segment_slice]
        close_seg = close_values[segment_slice]
        notional_seg = notional_values[segment_slice]
        trades_seg = trades_values[segment_slice]
        delta_seg = delta_values[segment_slice]
        large_delta_seg = large_delta_values[segment_slice]
        range_seg = bar_range_values[segment_slice]
        ret_seg = close_return_values[segment_slice]
        offsets = group_positions - start_pos
        rows = np.arange(left, right, dtype=np.int64)
        ages = offsets.astype(float)

        running_low = np.minimum.accumulate(low_seg)
        running_high = np.maximum.accumulate(high_seg)
        low_update = np.r_[True, np.diff(running_low) < -EPS]
        new_low_cum = np.cumsum(low_update)
        last_low_index = np.maximum.accumulate(np.where(low_update, np.arange(len(low_seg)), 0))

        candidate_lows = low_values[group_positions]
        running_candidate_low = np.minimum.accumulate(candidate_lows)
        retest = candidate_lows <= running_candidate_low * (1.0 + tolerance)
        retest_cum = np.cumsum(retest)

        prefix_notional = _prefix_sum(notional_seg)
        prefix_trades = _prefix_sum(trades_seg)
        prefix_delta = _prefix_sum(delta_seg)
        prefix_large = _prefix_sum(large_delta_seg)
        prefix_range = _prefix_sum(range_seg)
        prefix_ret = _prefix_sum(ret_seg)
        prefix_ret_sq = _prefix_sum(ret_seg * ret_seg)

        start_index = np.zeros(len(offsets), dtype=np.int64)
        midpoint = offsets // 2
        total_count = offsets + 1
        early_count = midpoint + 1
        recent_count = total_count - early_count
        recent_start = midpoint + 1

        total_notional = _interval_sum(prefix_notional, start_index, offsets)
        total_delta = _interval_sum(prefix_delta, start_index, offsets)
        total_large = _interval_sum(prefix_large, start_index, offsets)
        early_notional = _interval_sum(prefix_notional, start_index, midpoint)
        early_delta = _interval_sum(prefix_delta, start_index, midpoint)
        early_large = _interval_sum(prefix_large, start_index, midpoint)
        early_trades = _interval_sum(prefix_trades, start_index, midpoint)
        early_range = _interval_sum(prefix_range, start_index, midpoint)
        early_ret_sum = _interval_sum(prefix_ret, start_index, midpoint)
        early_ret_sq = _interval_sum(prefix_ret_sq, start_index, midpoint)

        recent_notional = np.where(
            recent_count > 0,
            _interval_sum(prefix_notional, np.minimum(recent_start, offsets), offsets),
            early_notional,
        )
        recent_delta = np.where(
            recent_count > 0,
            _interval_sum(prefix_delta, np.minimum(recent_start, offsets), offsets),
            early_delta,
        )
        recent_large = np.where(
            recent_count > 0,
            _interval_sum(prefix_large, np.minimum(recent_start, offsets), offsets),
            early_large,
        )
        recent_trades = np.where(
            recent_count > 0,
            _interval_sum(prefix_trades, np.minimum(recent_start, offsets), offsets),
            early_trades,
        )
        recent_range = np.where(
            recent_count > 0,
            _interval_sum(prefix_range, np.minimum(recent_start, offsets), offsets),
            early_range,
        )
        recent_ret_sum = np.where(
            recent_count > 0,
            _interval_sum(prefix_ret, np.minimum(recent_start, offsets), offsets),
            early_ret_sum,
        )
        recent_ret_sq = np.where(
            recent_count > 0,
            _interval_sum(prefix_ret_sq, np.minimum(recent_start, offsets), offsets),
            early_ret_sq,
        )

        early_delta_ratio = _safe_divide(early_delta, early_notional)
        recent_delta_ratio = _safe_divide(recent_delta, recent_notional)
        early_large_ratio = _safe_divide(early_large, early_notional)
        recent_large_ratio = _safe_divide(recent_large, recent_notional)
        early_close = close_seg[midpoint]
        recent_start_close = close_seg[np.minimum(recent_start, offsets)]
        current_close = close_seg[offsets]
        early_price_return = _safe_divide(early_close, np.maximum(close_seg[0], EPS)) - 1.0
        recent_price_return = _safe_divide(current_close, np.maximum(recent_start_close, EPS)) - 1.0
        early_impact = _safe_divide(np.maximum(-early_price_return, 0.0), np.maximum(-early_delta_ratio, 0.0) + EPS)
        recent_impact = _safe_divide(np.maximum(-recent_price_return, 0.0), np.maximum(-recent_delta_ratio, 0.0) + EPS)
        early_ret_mean = _safe_divide(early_ret_sum, early_count)
        recent_count_safe = np.maximum(recent_count, 1)
        recent_ret_mean = _safe_divide(recent_ret_sum, recent_count_safe)
        early_var = np.maximum(_safe_divide(early_ret_sq, early_count) - early_ret_mean * early_ret_mean, 0.0)
        recent_var = np.maximum(_safe_divide(recent_ret_sq, recent_count_safe) - recent_ret_mean * recent_ret_mean, 0.0)

        arrays["region_age_bars"][rows] = ages
        arrays["region_observation_number"][rows] = observation[left:right]
        arrays["region_candidate_density"][rows] = _safe_divide(observation[left:right], ages + 1.0)
        arrays["region_return_from_start"][rows] = _safe_divide(current_close, np.maximum(close_seg[0], EPS)) - 1.0
        arrays["region_low_progression"][rows] = _safe_divide(running_low[offsets], np.maximum(low_seg[0], EPS)) - 1.0
        arrays["region_rebound_from_low"][rows] = _safe_divide(current_close, np.maximum(running_low[offsets], EPS)) - 1.0
        arrays["region_drawdown_from_high"][rows] = _safe_divide(current_close, np.maximum(running_high[offsets], EPS)) - 1.0
        arrays["region_new_low_count"][rows] = new_low_cum[offsets]
        arrays["region_candidate_retest_count"][rows] = retest_cum
        arrays["region_bars_since_low"][rows] = offsets - last_low_index[offsets]
        arrays["region_cumulative_delta_ratio"][rows] = _safe_divide(total_delta, total_notional)
        arrays["region_cumulative_large_delta_ratio"][rows] = _safe_divide(total_large, total_notional)
        arrays["region_recent_delta_ratio"][rows] = recent_delta_ratio
        arrays["region_early_delta_ratio"][rows] = early_delta_ratio
        arrays["region_delta_improvement"][rows] = recent_delta_ratio - early_delta_ratio
        arrays["region_recent_large_delta_ratio"][rows] = recent_large_ratio
        arrays["region_early_large_delta_ratio"][rows] = early_large_ratio
        arrays["region_large_delta_improvement"][rows] = recent_large_ratio - early_large_ratio
        arrays["region_absorption_improvement"][rows] = early_impact - recent_impact
        arrays["region_notional_recent_vs_early"][rows] = _safe_divide(
            _safe_divide(recent_notional, recent_count_safe),
            _safe_divide(early_notional, early_count) + EPS,
        )
        arrays["region_trades_recent_vs_early"][rows] = _safe_divide(
            _safe_divide(recent_trades, recent_count_safe),
            _safe_divide(early_trades, early_count) + EPS,
        )
        arrays["region_range_recent_vs_early"][rows] = _safe_divide(
            _safe_divide(recent_range, recent_count_safe),
            _safe_divide(early_range, early_count) + EPS,
        )
        arrays["region_vol_recent_vs_early"][rows] = _safe_divide(np.sqrt(recent_var), np.sqrt(early_var) + EPS)
        previous_close = close_values[np.maximum(group_positions - 1, 0)]
        arrays["region_close_above_previous"][rows] = (close_values[group_positions] > previous_close).astype(np.float32)
        rebound = _safe_divide(current_close, np.maximum(running_low[offsets], EPS)) - 1.0
        arrays["region_reclaim_10bp"][rows] = (rebound >= 0.001).astype(np.float32)
        arrays["region_reclaim_20bp"][rows] = (rebound >= 0.002).astype(np.float32)

        processed += right - left
        if reporter is not None and processed < n:
            reporter.update(processed)
    if reporter is not None:
        reporter.close()

    for name, values in arrays.items():
        data[name] = values
    region_sizes = data.groupby("causal_region_id", sort=False).size()
    summary = pd.DataFrame(
        [
            {"metric": "candidate_state_count", "value": int(len(data))},
            {"metric": "causal_region_count", "value": int(region_sizes.size)},
            {"metric": "median_candidate_states_per_region", "value": float(region_sizes.median())},
            {"metric": "p90_candidate_states_per_region", "value": float(region_sizes.quantile(0.90))},
            {"metric": "median_region_age_bars", "value": float(pd.to_numeric(data["region_age_bars"], errors="coerce").median())},
            {"metric": "p90_region_age_bars", "value": float(pd.to_numeric(data["region_age_bars"], errors="coerce").quantile(0.90))},
        ]
    )
    return RegionBuildResult(data, _feature_dictionary(), summary)


_MECHANISM_COMPONENTS: dict[str, tuple[tuple[str, float], ...]] = {
    "shock": (
        ("current_range_pct", 1.0),
        ("notional_intensity_30", 1.0),
        ("trades_intensity_30", 1.0),
        ("return_acceleration_5_30", -1.0),
        ("current_delta_ratio", -0.75),
        ("current_large_delta_ratio", -0.50),
    ),
    "trend": (
        ("price_return_60", -1.0),
        ("price_return_120", -1.0),
        ("price_return_240", -1.0),
        ("down_bar_share_60", 0.75),
        ("down_bar_share_120", 0.75),
        ("path_efficiency_60", 0.50),
        ("delta_ratio_60", -0.75),
        ("delta_ratio_120", -0.75),
    ),
    "base": (
        ("support_test_density_60", 1.0),
        ("support_test_density_120", 1.0),
        ("support_test_density_240", 0.75),
        ("range_position_120", -0.75),
        ("range_position_240", -0.75),
        ("sell_pressure_absorption_30", 0.75),
        ("sell_pressure_absorption_60", 1.0),
        ("vol_compression_10_60", -0.50),
    ),
}


def fit_soft_mechanism_transformer(train: pd.DataFrame) -> SoftMechanismTransformer:
    component_center: dict[str, float] = {}
    component_scale: dict[str, float] = {}
    all_components = sorted({column for components in _MECHANISM_COMPONENTS.values() for column, _ in components})
    for column in all_components:
        values = pd.to_numeric(train.get(column, pd.Series(dtype=float)), errors="coerce").replace([np.inf, -np.inf], np.nan)
        median = float(values.median()) if not values.empty else 0.0
        q10 = float(values.quantile(0.10)) if not values.empty else -1.0
        q90 = float(values.quantile(0.90)) if not values.empty else 1.0
        scale = q90 - q10
        if not np.isfinite(scale) or scale <= EPS:
            scale = float(values.std()) if not values.empty else 1.0
        if not np.isfinite(scale) or scale <= EPS:
            scale = 1.0
        component_center[column] = median if np.isfinite(median) else 0.0
        component_scale[column] = scale
    raw = _mechanism_raw_scores(train, component_center, component_scale)
    references = {
        name: pd.to_numeric(raw[f"mechanism_{name}_raw"], errors="coerce").to_numpy(dtype=float)
        for name in ("shock", "trend", "base")
    }
    return SoftMechanismTransformer(component_center, component_scale, references)


def _mechanism_raw_scores(
    frame: pd.DataFrame,
    center: Mapping[str, float],
    scale: Mapping[str, float],
) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for mechanism, components in _MECHANISM_COMPONENTS.items():
        pieces: list[np.ndarray] = []
        weights: list[float] = []
        for column, direction in components:
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
            z = np.clip((values - float(center[column])) / float(scale[column]), -5.0, 5.0)
            pieces.append(z * float(direction))
            weights.append(abs(float(direction)))
        if not pieces:
            output[f"mechanism_{mechanism}_raw"] = np.nan
        else:
            matrix = np.vstack(pieces)
            weight_array = np.asarray(weights, dtype=float)[:, None]
            finite = np.isfinite(matrix)
            numerator = np.nansum(matrix * weight_array, axis=0)
            denominator = np.sum(finite * weight_array, axis=0)
            output[f"mechanism_{mechanism}_raw"] = np.divide(
                numerator,
                denominator,
                out=np.full(matrix.shape[1], np.nan, dtype=float),
                where=denominator > EPS,
            )
    return output


def mechanism_feature_dictionary() -> pd.DataFrame:
    descriptions = {
        "mechanism_shock_score": "train-percentile score for abrupt concentrated sell shock",
        "mechanism_trend_score": "train-percentile score for persistent directional decline",
        "mechanism_base_score": "train-percentile score for repeated low testing/absorption/compression",
        "mechanism_top_margin": "gap between the highest and second-highest mechanism scores",
        "mechanism_entropy": "normalized uncertainty across the three soft mechanism scores",
    }
    return pd.DataFrame(
        [
            {
                "feature": name,
                "feature_group": MECHANISM_FEATURE_GROUP,
                "description": description,
                "source": "train-fitted robust transforms of causal M0 snapshot features",
                "available_rule": "current closed 1m bar or older; no cluster ID or future path",
            }
            for name, description in descriptions.items()
        ]
    )


def attach_positive_opportunity_episodes(
    frame: pd.DataFrame,
    *,
    max_gap_bars: int = 2,
    target_column: str = "tp_hit_1pct",
) -> pd.DataFrame:
    """Attach retrospective positive opportunity episodes for evaluation/weights.

    The episode ID is label metadata only and must never be included in model
    features.  Consecutive TP-positive candidate rows within ``max_gap_bars``
    are treated as one underlying opportunity episode.
    """

    data = frame.sort_values(["extreme_pos", "event_id"]).reset_index(drop=True).copy()
    positive = data[target_column].astype(bool).to_numpy()
    positions = pd.to_numeric(data["extreme_pos"], errors="raise").to_numpy(dtype=np.int64)
    episode_number = np.full(len(data), -1, dtype=np.int64)
    current = 0
    previous_positive_position = -10**18
    for i, (is_positive, position) in enumerate(zip(positive, positions)):
        if not is_positive:
            continue
        if position - previous_positive_position > int(max_gap_bars):
            current += 1
        episode_number[i] = current
        previous_positive_position = int(position)
    data["positive_episode_number"] = episode_number
    data["positive_episode_id"] = np.where(
        episode_number >= 0,
        np.asarray([f"PE_{value:08d}" if value >= 0 else "" for value in episode_number], dtype=object),
        "",
    )
    sizes = data.loc[positive].groupby("positive_episode_id").size()
    data["positive_episode_size"] = data["positive_episode_id"].map(sizes).fillna(0).astype(np.int32)
    return data


def attach_episode_balanced_weight(frame: pd.DataFrame) -> pd.DataFrame:
    """Give each positive episode and negative causal region comparable mass."""

    data = frame.copy()
    positive = data["tp_hit_1pct"].astype(bool)
    positive_group = data["positive_episode_id"].astype(str)
    negative_group = data["causal_region_id"].astype(str)
    group_key = np.where(positive, "P_" + positive_group, "N_" + negative_group)
    counts = pd.Series(group_key).value_counts()
    weight = pd.Series(group_key).map(1.0 / counts).to_numpy(dtype=float)
    if np.isfinite(weight).any() and np.nanmean(weight) > EPS:
        weight = weight / np.nanmean(weight)
    data["episode_weight"] = weight.astype(np.float32)
    return data


def select_first_region_signal(
    frame: pd.DataFrame,
    *,
    score_column: str,
    threshold: float,
    cooldown_bars: int = 0,
) -> pd.DataFrame:
    """Select the first threshold crossing per causal broad candidate region."""

    if frame.empty:
        return frame.copy()
    required = {"causal_region_id", "extreme_pos", score_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"region signal selection missing columns: {missing}")
    data = frame.sort_values(["extreme_pos", "event_id"]).copy()
    score = pd.to_numeric(data[score_column], errors="coerce")
    eligible = data[score >= float(threshold)].copy()
    if eligible.empty:
        return eligible
    eligible = eligible.sort_values(["causal_region_id", "extreme_pos"]).groupby("causal_region_id", sort=False).head(1)
    eligible = eligible.sort_values("extreme_pos")
    if int(cooldown_bars) > 0:
        chosen: list[int] = []
        last_position = -10**18
        for row_index, position in zip(eligible.index, pd.to_numeric(eligible["extreme_pos"], errors="raise")):
            if int(position) - last_position < int(cooldown_bars):
                continue
            chosen.append(int(row_index))
            last_position = int(position)
        eligible = eligible.loc[chosen]
    result = eligible.reset_index(drop=True)
    result["signal_threshold"] = float(threshold)
    result["cooldown_bars"] = int(cooldown_bars)
    return result


def fit_quantile_risk_model(
    train: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    quantile: float,
    target_column: str = "mae_before_tp_pct",
    weight_column: str = "episode_weight",
    random_state: int = 42,
    min_samples_leaf: int = 100,
) -> QuantileRiskModel:
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must be between 0 and 1")
    source = train[train["tp_hit_1pct"].astype(bool)].copy()
    target = pd.to_numeric(source[target_column], errors="coerce")
    valid = target.notna() & np.isfinite(target)
    source = source.loc[valid].copy()
    target = target.loc[valid].clip(0.0, 5.0)
    constant = float(target.quantile(float(quantile))) if len(target) else 1.0
    if not np.isfinite(constant):
        constant = 1.0
    if len(source) < max(100, int(min_samples_leaf) * 2) or target.nunique() < 5:
        return QuantileRiskModel(tuple(feature_columns), pd.Series(0.0, index=feature_columns), float(quantile), None, constant)
    x_raw = (
        source.reindex(columns=feature_columns)
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    medians = x_raw.median().fillna(0.0)
    x = x_raw.fillna(medians)
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=float(quantile),
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=15,
        max_depth=3,
        min_samples_leaf=max(20, int(min_samples_leaf)),
        l2_regularization=2.0,
        early_stopping=True,
        validation_fraction=0.12,
        random_state=int(random_state),
    )
    weight = (
        pd.to_numeric(source[weight_column], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        if weight_column in source.columns
        else np.ones(len(source), dtype=float)
    )
    model.fit(x, target.to_numpy(dtype=float), sample_weight=weight)
    return QuantileRiskModel(tuple(feature_columns), medians, float(quantile), model, constant)


def fixed_multiobjective_score(
    p_tp60: Sequence[float],
    p_clean50: Sequence[float],
    p_fast15: Sequence[float],
    mae_q90_pct: Sequence[float],
) -> np.ndarray:
    """A pre-declared diagnostic score; never tuned on a walk-forward test fold."""

    tp = np.clip(np.asarray(p_tp60, dtype=float), 0.0, 1.0)
    clean = np.clip(np.asarray(p_clean50, dtype=float), 0.0, 1.0)
    fast = np.clip(np.asarray(p_fast15, dtype=float), 0.0, 1.0)
    mae = np.clip(np.asarray(mae_q90_pct, dtype=float), 0.0, 5.0)
    benefit = 0.50 * tp + 0.30 * clean + 0.20 * fast
    risk_quality = 1.0 / (1.0 + mae / 0.50)
    return np.clip(benefit * risk_quality, 0.0, 1.0)


def positive_episode_coverage(events: pd.DataFrame, population: pd.DataFrame) -> dict[str, float]:
    positive_population = population[population["tp_hit_1pct"].astype(bool)]
    total_ids = set(positive_population["positive_episode_id"].astype(str)) - {""}
    captured_ids = set(events.loc[events["tp_hit_1pct"].astype(bool), "positive_episode_id"].astype(str)) - {""}
    return {
        "positive_episode_count": float(len(total_ids)),
        "captured_positive_episode_count": float(len(captured_ids)),
        "positive_episode_coverage": float(len(captured_ids) / len(total_ids)) if total_ids else np.nan,
    }


def concentration_metrics(events: pd.DataFrame) -> dict[str, float]:
    if events.empty:
        return {
            "active_day_count": 0.0,
            "top5_day_event_share": np.nan,
            "top10_day_event_share": np.nan,
            "top5_day_tp_share": np.nan,
            "top10_day_tp_share": np.nan,
        }
    data = events.copy()
    data["event_date"] = pd.to_datetime(data["extreme_time"]).dt.date.astype(str)
    by_day = data.groupby("event_date").agg(event_count=("event_id", "size"), tp_count=("tp_hit_1pct", "sum"))
    event_total = max(1, int(by_day["event_count"].sum()))
    tp_total = max(1, int(by_day["tp_count"].sum()))
    return {
        "active_day_count": float(len(by_day)),
        "top5_day_event_share": float(by_day["event_count"].nlargest(5).sum() / event_total),
        "top10_day_event_share": float(by_day["event_count"].nlargest(10).sum() / event_total),
        "top5_day_tp_share": float(by_day["tp_count"].nlargest(5).sum() / tp_total),
        "top10_day_tp_share": float(by_day["tp_count"].nlargest(10).sum() / tp_total),
    }


def remove_strongest_days(events: pd.DataFrame, day_count: int) -> pd.DataFrame:
    if events.empty or day_count <= 0:
        return events.copy()
    data = events.copy()
    data["_event_date"] = pd.to_datetime(data["extreme_time"]).dt.date.astype(str)
    ranking = (
        data.groupby("_event_date")
        .agg(tp_count=("tp_hit_1pct", "sum"), event_count=("event_id", "size"))
        .sort_values(["tp_count", "event_count"], ascending=False)
    )
    remove = set(ranking.head(int(day_count)).index)
    return data[~data["_event_date"].isin(remove)].drop(columns="_event_date").reset_index(drop=True)
