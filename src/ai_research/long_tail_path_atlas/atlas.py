#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Per-event 1m path extraction, semantic flags and discovery-only clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

from src.ai_research.long_tail_exit_audit.data import MinutePathData
from src.ai_research.long_tail_exit_audit.simulator import EventCandidate, ScoreTimeline

from .config import LongTailPathAtlasConfig

_ONE_MINUTE_NS = int(pd.Timedelta(minutes=1).value)


@dataclass(frozen=True)
class EventPathExtraction:
    summary: dict[str, object]
    points: pd.DataFrame


@dataclass
class PathClusterModel:
    columns: tuple[str, ...]
    medians: np.ndarray
    scaler: RobustScaler
    model: KMeans
    cluster_names: dict[int, str]
    discovery_rows: int
    silhouette: float

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrix = frame.loc[:, self.columns].to_numpy(dtype=float)
        invalid = ~np.isfinite(matrix)
        if invalid.any():
            matrix[invalid] = np.take(self.medians, np.where(invalid)[1])
        scaled = self.scaler.transform(matrix)
        labels = self.model.predict(scaled)
        distances = self.model.transform(scaled).min(axis=1)
        return labels.astype(int), distances.astype(float)


CLUSTER_FEATURES: tuple[str, ...] = (
    "ret_30m",
    "ret_60m",
    "ret_180m",
    "ret_360m",
    "ret_720m",
    "ret_1440m",
    "mfe_60m",
    "mae_60m",
    "mfe_360m",
    "mae_360m",
    "time_to_mfe_360m",
    "peak_giveback_360m",
    "underwater_fraction_360m",
    "post6_mfe_increment_1440m",
    "score_percentile_change_360m",
    "q90_reconfirmations_360m",
)


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float)[np.isfinite(reference)])
    out = np.full(len(values), np.nan, dtype=float)
    valid = np.isfinite(values)
    if len(ref):
        out[valid] = np.searchsorted(ref, np.asarray(values, dtype=float)[valid], side="right") / len(ref)
    return out


def _score_asof(
    minute_times_ns: np.ndarray,
    timeline: ScoreTimeline,
    calibration_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(timeline.decision_times_ns, minute_times_ns, side="right") - 1
    score = np.full(len(minute_times_ns), np.nan, dtype=float)
    valid = positions >= 0
    score[valid] = np.asarray(timeline.scores, dtype=float)[positions[valid]]
    return score, empirical_percentile(calibration_scores, score)


def _exact_event_slice(
    path: MinutePathData,
    event: EventCandidate,
    config: LongTailPathAtlasConfig,
) -> tuple[int, int] | None:
    entry_ns = event.decision_time_ns + config.entry_delay_minutes * _ONE_MINUTE_NS
    start = int(np.searchsorted(path.timestamps_ns, entry_ns, side="left"))
    rows = config.analysis_horizon_hours * 60
    stop = start + rows
    if start >= len(path.timestamps_ns) or stop > len(path.timestamps_ns):
        return None
    expected = entry_ns + np.arange(rows, dtype=np.int64) * _ONE_MINUTE_NS
    actual = np.asarray(path.timestamps_ns[start:stop], dtype=np.int64)
    if len(actual) != rows or not np.array_equal(actual, expected):
        return None
    return start, stop


def _first_position(mask: np.ndarray) -> float:
    positions = np.flatnonzero(mask)
    return float(positions[0]) if len(positions) else np.nan


def _longest_true_run(mask: np.ndarray) -> int:
    values = np.asarray(mask, dtype=bool)
    if not values.any():
        return 0
    padded = np.concatenate([[False], values, [False]]).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return int(np.max(ends - starts))


def _directional_efficiency(close: np.ndarray) -> float:
    if len(close) < 2 or close[0] <= 0:
        return np.nan
    increments = np.diff(close) / close[:-1]
    distance = float(np.abs(increments).sum())
    return float((close[-1] / close[0] - 1.0) / distance) if distance > 0 else 0.0


def _horizon_features(
    high_ret: np.ndarray,
    low_drawdown: np.ndarray,
    close_ret: np.ndarray,
    close: np.ndarray,
    minute: int,
) -> dict[str, float]:
    end = minute
    highs = high_ret[:end]
    lows = low_drawdown[:end]
    closes = close_ret[:end]
    return {
        f"ret_{minute}m": float(closes[-1]),
        f"mfe_{minute}m": float(np.max(highs)),
        f"mae_{minute}m": float(np.max(lows)),
        f"time_to_mfe_{minute}m": float(np.argmax(highs)),
        f"time_to_mae_{minute}m": float(np.argmax(lows)),
        f"peak_giveback_{minute}m": float(np.max(highs) - closes[-1]),
        f"underwater_fraction_{minute}m": float(np.mean(closes < 0)),
        f"longest_underwater_{minute}m": float(_longest_true_run(closes < 0)),
        f"directional_efficiency_{minute}m": _directional_efficiency(close[:end]),
    }


def _score_window_features(
    event: EventCandidate,
    timeline: ScoreTimeline,
    calibration_scores: np.ndarray,
    minutes: int,
) -> dict[str, float]:
    start_ns = event.decision_time_ns
    end_ns = start_ns + minutes * _ONE_MINUTE_NS
    left = int(np.searchsorted(timeline.decision_times_ns, start_ns, side="left"))
    right = int(np.searchsorted(timeline.decision_times_ns, end_ns, side="right"))
    raw = np.asarray(timeline.scores[left:right], dtype=float)
    times = np.asarray(timeline.decision_times_ns[left:right], dtype=np.int64)
    pct = empirical_percentile(calibration_scores, raw)
    if not len(raw):
        return {
            f"score_percentile_end_{minutes}m": np.nan,
            f"score_percentile_min_{minutes}m": np.nan,
            f"score_percentile_max_{minutes}m": np.nan,
            f"score_percentile_change_{minutes}m": np.nan,
            f"q90_reconfirmations_{minutes}m": 0.0,
            f"q95_reconfirmations_{minutes}m": 0.0,
            f"first_below_q70_{minutes}m": np.nan,
            f"first_below_q50_{minutes}m": np.nan,
            f"longest_below_q50_decisions_{minutes}m": 0.0,
        }
    relative_minutes = (times - start_ns) / _ONE_MINUTE_NS
    start_pct = float(pct[0])
    return {
        f"score_percentile_end_{minutes}m": float(pct[-1]),
        f"score_percentile_min_{minutes}m": float(np.nanmin(pct)),
        f"score_percentile_max_{minutes}m": float(np.nanmax(pct)),
        f"score_percentile_change_{minutes}m": float(pct[-1] - start_pct),
        f"q90_reconfirmations_{minutes}m": float(max(0, int(np.sum(pct >= 0.90)) - int(pct[0] >= 0.90))),
        f"q95_reconfirmations_{minutes}m": float(max(0, int(np.sum(pct >= 0.95)) - int(pct[0] >= 0.95))),
        f"first_below_q70_{minutes}m": _first_position(pct < 0.70) * 15.0 if np.any(pct < 0.70) else np.nan,
        f"first_below_q50_{minutes}m": _first_position(pct < 0.50) * 15.0 if np.any(pct < 0.50) else np.nan,
        f"longest_below_q50_decisions_{minutes}m": float(_longest_true_run(pct < 0.50)),
    }


def extract_event_path(
    *,
    event: EventCandidate,
    fold_id: str,
    phase: str,
    path: MinutePathData,
    timeline: ScoreTimeline,
    calibration_scores: np.ndarray,
    config: LongTailPathAtlasConfig,
) -> EventPathExtraction | None:
    positions = _exact_event_slice(path, event, config)
    if positions is None:
        return None
    start, stop = positions
    timestamps_ns = np.asarray(path.timestamps_ns[start:stop], dtype=np.int64)
    open_price = np.asarray(path.open[start:stop], dtype=float)
    high = np.asarray(path.high[start:stop], dtype=float)
    low = np.asarray(path.low[start:stop], dtype=float)
    close = np.asarray(path.close[start:stop], dtype=float)
    entry_price = float(open_price[0])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None
    high_ret = high / entry_price - 1.0
    low_drawdown = 1.0 - low / entry_price
    close_ret = close / entry_price - 1.0
    running_mfe = np.maximum.accumulate(high_ret)
    running_mae = np.maximum.accumulate(low_drawdown)
    drawdown_from_peak = running_mfe - close_ret
    score, score_percentile = _score_asof(timestamps_ns, timeline, calibration_scores)
    canonical_event_id = f"{fold_id}_{phase}_{event.event_id}"

    points = pd.DataFrame(
        {
            "event_id": canonical_event_id,
            "fold_id": fold_id,
            "phase": phase,
            "signal_quantile": event.signal_quantile,
            "decision_time": pd.Timestamp(event.decision_time_ns, unit="ns"),
            "entry_time": pd.Timestamp(timestamps_ns[0], unit="ns"),
            "timestamp": pd.to_datetime(timestamps_ns, unit="ns"),
            "minute_since_entry": np.arange(len(timestamps_ns), dtype=np.int32),
            "open_return": open_price / entry_price - 1.0,
            "high_return": high_ret,
            "low_return": -low_drawdown,
            "close_return": close_ret,
            "running_mfe": running_mfe,
            "running_mae": running_mae,
            "drawdown_from_peak": drawdown_from_peak,
            "base_score": score,
            "score_percentile": score_percentile,
        }
    )

    summary: dict[str, object] = {
        "event_id": canonical_event_id,
        "source_event_id": event.event_id,
        "fold_id": fold_id,
        "phase": phase,
        "decision_time": pd.Timestamp(event.decision_time_ns, unit="ns"),
        "entry_time": pd.Timestamp(timestamps_ns[0], unit="ns"),
        "entry_price": entry_price,
        "signal_quantile": float(event.signal_quantile),
        "event_score": float(event.score),
        "event_score_percentile": float(empirical_percentile(calibration_scores, np.array([event.score]))[0]),
        "is_q95": bool(event.score >= float(np.nanquantile(calibration_scores, config.quality_control_quantile))),
        "path_rows": int(len(points)),
    }
    for minute in config.checkpoint_minutes:
        summary.update(_horizon_features(high_ret, low_drawdown, close_ret, close, minute))
    for minute in (60, 180, 360, 720, 1440, 2880):
        summary.update(_score_window_features(event, timeline, calibration_scores, minute))
    for level in config.upside_levels:
        label = f"{level * 100:.1f}".replace(".", "p")
        hit = np.flatnonzero(high_ret >= level)
        summary[f"time_to_up_{label}pct"] = float(hit[0]) if len(hit) else np.nan
        summary[f"hit_up_{label}pct_6h"] = bool(len(hit) and int(hit[0]) < 360)
        summary[f"hit_up_{label}pct_24h"] = bool(len(hit) and int(hit[0]) < 1440)
    for level in config.downside_levels:
        label = f"{level * 100:.1f}".replace(".", "p")
        hit = np.flatnonzero(low_drawdown >= level)
        summary[f"time_to_down_{label}pct"] = float(hit[0]) if len(hit) else np.nan
        summary[f"hit_down_{label}pct_6h"] = bool(len(hit) and int(hit[0]) < 360)
        summary[f"hit_down_{label}pct_24h"] = bool(len(hit) and int(hit[0]) < 1440)

    target_hit = np.flatnonzero(high_ret >= config.immediate_target_pct)
    if len(target_hit):
        target_pos = int(target_hit[0])
        summary["mae_before_first_1pct"] = float(np.max(low_drawdown[: target_pos + 1]))
    else:
        summary["mae_before_first_1pct"] = np.nan
    summary["first_positive_close_minute"] = _first_position(close_ret > 0)
    summary["post6_mfe_increment_1440m"] = float(np.max(high_ret[:1440]) - np.max(high_ret[:360]))
    summary["post6_mfe_increment_2880m"] = float(np.max(high_ret) - np.max(high_ret[:360]))
    summary["post6_close_increment_1440m"] = float(close_ret[1439] - close_ret[359])
    summary["post6_close_increment_2880m"] = float(close_ret[-1] - close_ret[359])
    summary["fixed6h_gross_return"] = float(close_ret[359])
    summary["fixed6h_net_1x"] = float(close_ret[359] - config.base_round_trip_cost)
    summary["fixed6h_positive_expectancy_event"] = bool(summary["fixed6h_net_1x"] > 0)
    best_horizons = (60, 180, 360, 720, 1440, 2880)
    close_values = np.array([summary[f"ret_{minute}m"] for minute in best_horizons], dtype=float)
    best_index = int(np.nanargmax(close_values))
    summary["oracle_best_close_horizon_minutes"] = int(best_horizons[best_index])
    summary["oracle_best_close_return"] = float(close_values[best_index])
    summary["fixed6h_capture_of_best_close"] = (
        float(summary["ret_360m"] / close_values[best_index]) if close_values[best_index] > 0 else np.nan
    )
    summary["oracle_peak_mfe_48h"] = float(np.max(high_ret))
    summary["oracle_peak_mfe_minute"] = float(np.argmax(high_ret))
    summary.update(semantic_path_labels(summary, config))
    return EventPathExtraction(summary=summary, points=points)


def semantic_path_labels(row: dict[str, object] | pd.Series, config: LongTailPathAtlasConfig) -> dict[str, object]:
    get = row.get  # type: ignore[attr-defined]
    winner = bool(get("fixed6h_positive_expectancy_event", False))
    t1 = float(get("time_to_up_1p0pct", np.nan))
    mae_before = float(get("mae_before_first_1pct", np.nan))
    mae6 = float(get("mae_360m", np.nan))
    underwater6 = float(get("longest_underwater_360m", np.nan))
    mfe3 = float(get("mfe_180m", np.nan))
    mfe6 = float(get("mfe_360m", np.nan))
    giveback6 = float(get("peak_giveback_360m", np.nan))
    t_mfe6 = float(get("time_to_mfe_360m", np.nan))
    ret24 = float(get("ret_1440m", np.nan))
    mfe24 = float(get("mfe_1440m", np.nan))
    post6 = float(get("post6_mfe_increment_1440m", np.nan))

    immediate = winner and np.isfinite(t1) and t1 <= config.immediate_target_minutes and np.isfinite(mae_before) and mae_before <= config.immediate_max_mae_before_target
    spike_giveback = np.isfinite(mfe3) and mfe3 >= config.early_spike_mfe and np.isfinite(giveback6) and giveback6 >= config.early_spike_giveback
    delayed = winner and not immediate and ((np.isfinite(mae6) and mae6 >= config.delayed_recovery_mae) or (np.isfinite(underwater6) and underwater6 >= config.delayed_recovery_underwater_minutes))
    slow = winner and not immediate and not delayed and not spike_giveback and np.isfinite(mfe6) and mfe6 >= config.slow_grind_min_mfe and np.isfinite(t_mfe6) and t_mfe6 >= config.slow_grind_peak_after_minutes
    late_rescue = (not winner) and np.isfinite(ret24) and ret24 - config.base_round_trip_cost > 0 and np.isfinite(mfe24) and mfe24 >= config.slow_grind_min_mfe
    persistent_failure = (not winner) and np.isfinite(mfe24) and mfe24 < config.persistent_failure_max_24h_mfe and np.isfinite(ret24) and ret24 <= 0
    volatile_failure = (not winner) and np.isfinite(mfe6) and mfe6 >= config.early_spike_mfe and np.isfinite(giveback6) and giveback6 >= config.early_spike_giveback

    if immediate:
        primary = "immediate_clean_winner"
    elif winner and spike_giveback:
        primary = "early_spike_giveback_winner"
    elif delayed:
        primary = "delayed_recovery_winner"
    elif slow:
        primary = "slow_grind_winner"
    elif winner:
        primary = "other_6h_winner"
    elif late_rescue:
        primary = "late_rescue_after_6h"
    elif persistent_failure:
        primary = "persistent_failure"
    elif volatile_failure:
        primary = "volatile_giveback_failure"
    else:
        primary = "other_6h_failure"

    return {
        "semantic_path_type": primary,
        "flag_immediate_clean": immediate,
        "flag_early_spike_giveback": spike_giveback,
        "flag_delayed_recovery": delayed,
        "flag_slow_grind": slow,
        "flag_late_rescue": late_rescue,
        "flag_persistent_failure": persistent_failure,
        "flag_post6_continuation": bool(np.isfinite(post6) and post6 >= config.post6_continuation_increment),
        "flag_deep_6h_mae": bool(np.isfinite(mae6) and mae6 >= 0.010),
        "flag_score_reconfirmed_6h": bool(float(get("q90_reconfirmations_360m", 0.0)) >= 2.0),
        "flag_score_decayed_below_median_6h": bool(float(get("score_percentile_min_360m", 1.0)) < 0.50),
    }


def _cluster_name(centroid: pd.Series, cluster_id: int) -> str:
    ret1 = float(centroid.get("ret_60m", np.nan))
    ret6 = float(centroid.get("ret_360m", np.nan))
    ret24 = float(centroid.get("ret_1440m", np.nan))
    mfe3 = float(centroid.get("mfe_180m", np.nan))
    mfe6 = float(centroid.get("mfe_360m", np.nan))
    mae1 = float(centroid.get("mae_60m", np.nan))
    mae6 = float(centroid.get("mae_360m", np.nan))
    giveback = float(centroid.get("peak_giveback_360m", np.nan))
    if ret6 <= 0 and ret24 > 0.003:
        base = "late_rescue"
    elif ret6 <= 0 and mfe6 < 0.010:
        base = "persistent_failure"
    elif ret1 > 0.005 and mae1 < 0.005:
        base = "immediate_expansion"
    elif mfe3 >= 0.015 and giveback >= 0.0075:
        base = "early_spike_giveback"
    elif ret6 > 0 and mae6 >= 0.0075:
        base = "delayed_recovery"
    elif ret6 > 0:
        base = "slow_or_clean_positive"
    else:
        base = "volatile_mixed"
    return f"{base}_c{cluster_id}"


def fit_path_cluster_model(
    discovery: pd.DataFrame,
    config: LongTailPathAtlasConfig,
) -> PathClusterModel | None:
    if len(discovery) < config.minimum_discovery_events:
        return None
    matrix = discovery.loc[:, CLUSTER_FEATURES].to_numpy(dtype=float)
    medians = np.nanmedian(matrix, axis=0)
    medians[~np.isfinite(medians)] = 0.0
    invalid = ~np.isfinite(matrix)
    if invalid.any():
        matrix[invalid] = np.take(medians, np.where(invalid)[1])
    scaler = RobustScaler(quantile_range=(10.0, 90.0))
    scaled = scaler.fit_transform(matrix)
    cluster_count = min(config.cluster_count, max(2, len(discovery) // 6))
    model = KMeans(n_clusters=cluster_count, n_init=20, random_state=config.random_state)
    labels = model.fit_predict(scaled)
    silhouette = float(silhouette_score(scaled, labels)) if len(set(labels)) > 1 and len(discovery) > cluster_count else np.nan
    original_centers = scaler.inverse_transform(model.cluster_centers_)
    centroid_frame = pd.DataFrame(original_centers, columns=CLUSTER_FEATURES)
    names = {cluster_id: _cluster_name(centroid_frame.iloc[cluster_id], cluster_id) for cluster_id in range(cluster_count)}
    return PathClusterModel(
        columns=CLUSTER_FEATURES,
        medians=medians,
        scaler=scaler,
        model=model,
        cluster_names=names,
        discovery_rows=len(discovery),
        silhouette=silhouette,
    )


def cluster_centroid_frame(cluster: PathClusterModel, fold_id: str) -> pd.DataFrame:
    centers = cluster.scaler.inverse_transform(cluster.model.cluster_centers_)
    frame = pd.DataFrame(centers, columns=cluster.columns)
    frame.insert(0, "cluster_id", np.arange(len(frame), dtype=int))
    frame.insert(0, "cluster_name", [cluster.cluster_names[int(value)] for value in frame["cluster_id"]])
    frame.insert(0, "fold_id", fold_id)
    frame["discovery_rows"] = cluster.discovery_rows
    frame["discovery_silhouette"] = cluster.silhouette
    return frame


def assign_clusters(frame: pd.DataFrame, cluster: PathClusterModel | None) -> pd.DataFrame:
    output = frame.copy()
    if cluster is None or output.empty:
        output["cluster_id"] = np.nan
        output["cluster_name"] = "UNAVAILABLE_INSUFFICIENT_DISCOVERY"
        output["cluster_distance"] = np.nan
        return output
    labels, distances = cluster.transform(output)
    output["cluster_id"] = labels
    output["cluster_name"] = [cluster.cluster_names[int(label)] for label in labels]
    output["cluster_distance"] = distances
    return output


def representative_events(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_columns = ["fold_id", "semantic_path_type"]
    for keys, group in frame.groupby(group_columns, sort=False):
        fold_id, path_type = keys
        candidates: list[tuple[str, pd.Series]] = []
        candidates.append(("best_6h", group.loc[group["fixed6h_net_1x"].idxmax()]))
        candidates.append(("worst_6h", group.loc[group["fixed6h_net_1x"].idxmin()]))
        candidates.append(("largest_mfe_48h", group.loc[group["mfe_2880m"].idxmax()]))
        numeric = group.loc[:, ["ret_360m", "mfe_360m", "mae_360m", "ret_1440m"]].astype(float)
        center = numeric.median(axis=0)
        scale = numeric.mad(axis=0) if hasattr(numeric, "mad") else (numeric - center).abs().median(axis=0)
        scale = scale.replace(0, 1.0).fillna(1.0)
        distance = ((numeric - center) / scale).pow(2).sum(axis=1)
        candidates.append(("typical", group.loc[distance.idxmin()]))
        seen: set[str] = set()
        for role, item in candidates:
            event_id = str(item["event_id"])
            if event_id in seen:
                continue
            seen.add(event_id)
            rows.append(
                {
                    "fold_id": fold_id,
                    "semantic_path_type": path_type,
                    "role": role,
                    "event_id": event_id,
                    "decision_time": item["decision_time"],
                    "entry_time": item["entry_time"],
                    "fixed6h_net_1x": item["fixed6h_net_1x"],
                    "mfe_360m": item["mfe_360m"],
                    "mae_360m": item["mae_360m"],
                    "ret_1440m": item["ret_1440m"],
                    "oracle_best_close_horizon_minutes": item["oracle_best_close_horizon_minutes"],
                }
            )
    return pd.DataFrame(rows)


def selected_feature_contrast(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    selected = (
        "ret_60m",
        "mfe_60m",
        "mae_60m",
        "mfe_360m",
        "mae_360m",
        "time_to_mfe_360m",
        "longest_underwater_360m",
        "peak_giveback_360m",
        "post6_mfe_increment_1440m",
        "score_percentile_change_360m",
        "q90_reconfirmations_360m",
    )
    rows: list[dict[str, object]] = []
    for fold_id, fold in frame.groupby("fold_id", sort=False):
        winners = fold.loc[fold["fixed6h_positive_expectancy_event"]]
        losers = fold.loc[~fold["fixed6h_positive_expectancy_event"]]
        for feature in selected:
            left = winners[feature].astype(float)
            right = losers[feature].astype(float)
            pooled = float(pd.concat([left, right]).std(ddof=0))
            rows.append(
                {
                    "fold_id": fold_id,
                    "feature": feature,
                    "winner_rows": int(left.notna().sum()),
                    "loser_rows": int(right.notna().sum()),
                    "winner_median": float(left.median()),
                    "loser_median": float(right.median()),
                    "median_difference": float(left.median() - right.median()),
                    "standardized_median_difference": float((left.median() - right.median()) / pooled) if pooled > 0 else np.nan,
                }
            )
    return pd.DataFrame(rows)


def summarize_path_types(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []
    target_rows: list[dict[str, object]] = []
    quarter_frame = frame.copy()
    quarter_frame["quarter"] = pd.to_datetime(quarter_frame["entry_time"]).dt.to_period("Q").astype(str)
    quarter_totals = quarter_frame.groupby(["fold_id", "quarter"]).size().to_dict()
    for keys, group in quarter_frame.groupby(["fold_id", "semantic_path_type"], sort=False):
        fold_id, path_type = keys
        rows.append(
            {
                "fold_id": fold_id,
                "semantic_path_type": path_type,
                "events": int(len(group)),
                "share": float(len(group) / len(frame.loc[frame["fold_id"] == fold_id])),
                "fixed6h_win_rate_1x": float(group["fixed6h_positive_expectancy_event"].mean()),
                "mean_fixed6h_net_1x": float(group["fixed6h_net_1x"].mean()),
                "median_fixed6h_net_1x": float(group["fixed6h_net_1x"].median()),
                "mean_mfe_360m": float(group["mfe_360m"].mean()),
                "mean_mae_360m": float(group["mae_360m"].mean()),
                "median_time_to_mfe_360m": float(group["time_to_mfe_360m"].median()),
                "mean_peak_giveback_360m": float(group["peak_giveback_360m"].mean()),
                "post6_continuation_rate": float(group["flag_post6_continuation"].mean()),
                "late_rescue_rate": float(group["flag_late_rescue"].mean()),
                "mean_ret_1440m": float(group["ret_1440m"].mean()),
                "mean_oracle_best_close_return": float(group["oracle_best_close_return"].mean()),
                "median_oracle_best_horizon_minutes": float(group["oracle_best_close_horizon_minutes"].median()),
            }
        )
        work = group.copy()
        for quarter, quarter_group in work.groupby("quarter", sort=True):
            period_rows.append(
                {
                    "fold_id": fold_id,
                    "semantic_path_type": path_type,
                    "quarter": quarter,
                    "events": int(len(quarter_group)),
                    "share_within_quarter": float(len(quarter_group) / quarter_totals[(fold_id, quarter)]),
                    "mean_fixed6h_net_1x": float(quarter_group["fixed6h_net_1x"].mean()),
                    "win_rate_1x": float(quarter_group["fixed6h_positive_expectancy_event"].mean()),
                }
            )
    for fold_id, group in frame.groupby("fold_id", sort=False):
        for level in (0.005, 0.010, 0.015, 0.020, 0.030):
            label = f"{level * 100:.1f}".replace(".", "p")
            column = f"time_to_up_{label}pct"
            hit = group[column].notna()
            target_rows.append(
                {
                    "fold_id": fold_id,
                    "direction": "up",
                    "level": level,
                    "hit_rate_6h": float(group[f"hit_up_{label}pct_6h"].mean()),
                    "hit_rate_24h": float(group[f"hit_up_{label}pct_24h"].mean()),
                    "median_hit_minute_if_hit": float(group.loc[hit, column].median()) if hit.any() else np.nan,
                }
            )
        for level in (0.005, 0.010, 0.015, 0.020, 0.030):
            label = f"{level * 100:.1f}".replace(".", "p")
            column = f"time_to_down_{label}pct"
            hit = group[column].notna()
            target_rows.append(
                {
                    "fold_id": fold_id,
                    "direction": "down",
                    "level": level,
                    "hit_rate_6h": float(group[f"hit_down_{label}pct_6h"].mean()),
                    "hit_rate_24h": float(group[f"hit_down_{label}pct_24h"].mean()),
                    "median_hit_minute_if_hit": float(group.loc[hit, column].median()) if hit.any() else np.nan,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(period_rows), pd.DataFrame(target_rows)


def oracle_exit_summary(frame: pd.DataFrame, *, base_round_trip_cost: float = 0.0013) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for fold_id, group in frame.groupby("fold_id", sort=False):
        for horizon in (60, 180, 360, 720, 1440, 2880):
            values = group[f"ret_{horizon}m"].astype(float) - base_round_trip_cost
            rows.append(
                {
                    "fold_id": fold_id,
                    "exit_view": f"fixed_close_{horizon}m",
                    "events": int(len(group)),
                    "mean_net_1x": float(values.mean()),
                    "median_net_1x": float(values.median()),
                    "win_rate_1x": float((values > 0).mean()),
                }
            )
        rows.append(
            {
                "fold_id": fold_id,
                "exit_view": "oracle_best_fixed_close_1h_to_48h",
                "events": int(len(group)),
                "mean_net_1x": float((group["oracle_best_close_return"] - base_round_trip_cost).mean()),
                "median_net_1x": float((group["oracle_best_close_return"] - base_round_trip_cost).median()),
                "win_rate_1x": float(((group["oracle_best_close_return"] - base_round_trip_cost) > 0).mean()),
            }
        )
    return pd.DataFrame(rows)
