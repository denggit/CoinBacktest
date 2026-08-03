#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Supervised continuity models and diagnostics for R03.3.3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from src.ai_research.swing_baseline.dataset import load_year_shard

from .config import MarketStateContinuityConfig
from .state_cache import StateYearShard, load_state_year_shard, ns_to_datetime

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None  # type: ignore[assignment]


@dataclass(frozen=True)
class ContinuityFold:
    fold_id: str
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict[str, object]:
        return {
            key: str(value) if isinstance(value, pd.Timestamp) else value
            for key, value in asdict(self).items()
        }


def default_continuity_folds(config: MarketStateContinuityConfig) -> tuple[ContinuityFold, ...]:
    embargo = pd.Timedelta(hours=config.maximum_target_horizon_hours + 24)
    return (
        ContinuityFold(
            "WF_2024",
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2024-01-01") - embargo,
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-12-31 23:59:59"),
        ),
        ContinuityFold(
            "WF_2025",
            pd.Timestamp("2021-01-01"),
            pd.Timestamp("2025-01-01") - embargo,
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-12-31 23:59:59"),
        ),
    )


@dataclass(frozen=True)
class ContinuityPeriodData:
    timestamps_ns: np.ndarray
    x: np.ndarray
    y: np.ndarray
    feature_columns: tuple[str, ...]
    time_to_change_hours: np.ndarray | None = None

    @property
    def index(self) -> pd.DatetimeIndex:
        return ns_to_datetime(self.timestamps_ns)


def _year_from_state_path(path: Path) -> int:
    shard = load_state_year_shard(path)
    return int(shard.year)


def _year_from_trade_path(path: Path) -> int:
    shard = load_year_shard(path)
    return int(pd.to_datetime(np.asarray(shard.decision_times_ns[:1], dtype=np.int64), unit="ns")[0].year)


def _state_derived_columns(shard: StateYearShard) -> tuple[str, ...]:
    prefixes = (
        "strategic_",
        "tactical_",
        "entry_",
        "activity_",
        "all_direction_",
        "long_pullback_",
        "short_pullback_",
        "trend_momentum_",
    )
    return tuple(column for column in shard.feature_columns if column.startswith(prefixes))


def collect_continuity_period_data(
    state_paths: list[Path],
    trade_paths: list[Path],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    target: str,
    architecture: str,
) -> ContinuityPeriodData:
    state_map = {_year_from_state_path(path): path for path in state_paths}
    trade_map = {_year_from_trade_path(path): path for path in trade_paths}
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    change_time_parts: list[np.ndarray] = []
    expected_columns: tuple[str, ...] | None = None

    for year in sorted(state_map):
        shard = load_state_year_shard(state_map[year])
        times = np.asarray(shard.decision_times_ns, dtype=np.int64)
        left = int(np.searchsorted(times, int(pd.Timestamp(start).value), side="left"))
        right = int(np.searchsorted(times, int(pd.Timestamp(end).value), side="right"))
        if right <= left or target not in shard.target_index:
            continue
        state_times = times[left:right]
        state_target = np.asarray(shard.targets[left:right, shard.target_index[target]], dtype=float)
        change_column = f"{target}_time_to_change_hours"
        if change_column in shard.target_index:
            state_change_hours = np.asarray(
                shard.targets[left:right, shard.target_index[change_column]],
                dtype=float,
            )
        else:
            state_change_hours = np.full(len(state_target), np.nan, dtype=float)

        if architecture == "universal_ohlcv_lightgbm":
            matrix = np.asarray(shard.features[left:right], dtype=np.float32)
            columns = shard.feature_columns
        elif architecture == "trade_enhanced_lightgbm":
            if year not in trade_map:
                continue
            trade = load_year_shard(trade_map[year])
            trade_times = np.asarray(trade.decision_times_ns, dtype=np.int64)
            common, state_positions, trade_positions = np.intersect1d(
                state_times,
                trade_times,
                assume_unique=True,
                return_indices=True,
            )
            if not len(common):
                continue
            derived_columns = _state_derived_columns(shard)
            derived_index = [shard.feature_columns.index(column) for column in derived_columns]
            state_matrix = np.asarray(shard.features[left:right], dtype=np.float32)[state_positions][:, derived_index]
            trade_matrix = np.asarray(trade.features, dtype=np.float32)[trade_positions]
            matrix = np.concatenate([trade_matrix, state_matrix], axis=1)
            columns = tuple(trade.full_feature_columns) + tuple(f"state::{column}" for column in derived_columns)
            state_target = state_target[state_positions]
            state_change_hours = state_change_hours[state_positions]
            state_times = common
        else:
            raise ValueError(f"unsupported continuity architecture: {architecture}")

        if expected_columns is None:
            expected_columns = tuple(columns)
        elif tuple(columns) != expected_columns:
            raise RuntimeError(f"R03.3.3 feature schema drift in {architecture} year={year}")
        valid = np.isfinite(state_target)
        if not np.any(valid):
            continue
        x_parts.append(matrix[valid])
        y_parts.append(state_target[valid].astype(np.int8))
        time_parts.append(state_times[valid])
        change_time_parts.append(state_change_hours[valid])

    if not x_parts or expected_columns is None:
        raise RuntimeError(
            f"R03.3.3 no aligned data architecture={architecture} target={target} range={start}->{end}"
        )
    return ContinuityPeriodData(
        timestamps_ns=np.concatenate(time_parts),
        x=np.concatenate(x_parts, axis=0),
        y=np.concatenate(y_parts, axis=0),
        feature_columns=expected_columns,
        time_to_change_hours=np.concatenate(change_time_parts, axis=0),
    )


def validate_continuity_dependencies() -> None:
    if LGBMClassifier is None:
        raise RuntimeError("R03.3.3 requires lightgbm; install it before building caches or fitting models")


def _sample_rows(data: ContinuityPeriodData, max_rows: int) -> ContinuityPeriodData:
    if len(data.y) <= max_rows:
        return data
    positions = np.linspace(0, len(data.y) - 1, max_rows, dtype=np.int64)
    return ContinuityPeriodData(
        timestamps_ns=data.timestamps_ns[positions],
        x=data.x[positions],
        y=data.y[positions],
        feature_columns=data.feature_columns,
        time_to_change_hours=(
            data.time_to_change_hours[positions]
            if data.time_to_change_hours is not None
            else None
        ),
    )


def fit_continuity_model(
    data: ContinuityPeriodData,
    config: MarketStateContinuityConfig,
):
    validate_continuity_dependencies()
    sampled = _sample_rows(data, config.maximum_rows_per_fit)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=450,
        learning_rate=0.035,
        num_leaves=31,
        min_child_samples=250,
        subsample=0.85,
        colsample_bytree=0.80,
        reg_alpha=0.5,
        reg_lambda=2.0,
        random_state=config.random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(sampled.x, sampled.y)
    return model


def subset_period_data(
    data: ContinuityPeriodData,
    columns: tuple[str, ...],
) -> ContinuityPeriodData:
    missing = [column for column in columns if column not in data.feature_columns]
    if missing:
        raise RuntimeError(f"R03.3.3.1 mechanical baseline columns missing: {missing}")
    positions = [data.feature_columns.index(column) for column in columns]
    return ContinuityPeriodData(
        timestamps_ns=data.timestamps_ns,
        x=np.asarray(data.x[:, positions], dtype=np.float32),
        y=data.y,
        feature_columns=columns,
        time_to_change_hours=data.time_to_change_hours,
    )


def mechanical_feature_sets(target: str) -> dict[str, tuple[str, ...]]:
    layer = target.split("_persist_", 1)[0]
    age = f"{layer}_age_bars"
    margin = f"{layer}_boundary_margin"
    state = f"{layer}_state"
    return {
        "mechanical_age_only": (age,),
        "mechanical_margin_only": (margin,),
        "mechanical_age_margin_state": (age, margin, state),
    }


def fit_mechanical_baseline(
    data: ContinuityPeriodData,
    config: MarketStateContinuityConfig,
):
    validate_continuity_dependencies()
    sampled = _sample_rows(data, config.mechanical_baseline_max_rows)
    model = LGBMClassifier(
        objective="binary",
        n_estimators=120,
        learning_rate=0.05,
        num_leaves=7,
        min_child_samples=500,
        subsample=0.90,
        colsample_bytree=1.0,
        reg_alpha=1.0,
        reg_lambda=4.0,
        random_state=config.random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(sampled.x, sampled.y)
    return model


def build_mechanical_increment_audit(
    full_metrics: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    config: MarketStateContinuityConfig,
) -> pd.DataFrame:
    if full_metrics.empty or baseline_metrics.empty:
        return pd.DataFrame()
    full = full_metrics.loc[
        full_metrics["architecture"] == "universal_ohlcv_lightgbm"
    ].copy()
    if full.empty:
        return pd.DataFrame()
    ranked = baseline_metrics.sort_values(
        ["fold_id", "target", "auc", "brier_skill"],
        ascending=[True, True, False, False],
        kind="stable",
    )
    best = ranked.groupby(["fold_id", "target"], as_index=False).first()
    merged = full.merge(best, on=["fold_id", "target"], suffixes=("_full", "_mechanical"))
    if merged.empty:
        return merged
    merged["delta_auc_vs_mechanical"] = merged["auc_full"] - merged["auc_mechanical"]
    merged["delta_brier_skill_vs_mechanical"] = (
        merged["brier_skill_full"] - merged["brier_skill_mechanical"]
    )
    merged["delta_transition_lift_vs_mechanical"] = (
        merged["bottom_decile_transition_lift_full"]
        - merged["bottom_decile_transition_lift_mechanical"]
    )
    merged["incremental_auc_passed"] = (
        merged["delta_auc_vs_mechanical"] >= config.minimum_auc_increment_vs_mechanical
    )
    return merged[
        [
            "fold_id",
            "target",
            "architecture_full",
            "architecture_mechanical",
            "auc_full",
            "auc_mechanical",
            "delta_auc_vs_mechanical",
            "brier_skill_full",
            "brier_skill_mechanical",
            "delta_brier_skill_vs_mechanical",
            "bottom_decile_transition_lift_full",
            "bottom_decile_transition_lift_mechanical",
            "delta_transition_lift_vs_mechanical",
            "incremental_auc_passed",
        ]
    ]


def transition_alert_episode_audit(
    model,
    fit: ContinuityPeriodData,
    test: ContinuityPeriodData,
    *,
    fold_id: str,
    target: str,
    config: MarketStateContinuityConfig,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Merge low-persistence predictions into independent transition-warning episodes."""
    fit_prediction = np.asarray(model.predict_proba(fit.x)[:, 1], dtype=float)
    test_prediction = np.asarray(model.predict_proba(test.x)[:, 1], dtype=float)
    threshold = float(np.quantile(fit_prediction, config.transition_alert_train_quantile))
    times = test.index
    risk_positions = np.flatnonzero(test_prediction <= threshold)
    horizon_hours = next(spec.horizon_hours for spec in config.targets if spec.target_id == target)
    if not len(risk_positions):
        return (
            {
                "fold_id": fold_id,
                "target": target,
                "train_alert_quantile": config.transition_alert_train_quantile,
                "alert_threshold": threshold,
                "episodes": 0,
                "episodes_per_month": 0.0,
                "success_rate": np.nan,
                "false_alert_rate": np.nan,
                "median_lead_hours": np.nan,
                "transition_event_coverage": 0.0,
                "median_episode_duration_hours": np.nan,
            },
            pd.DataFrame(),
        )

    max_gap = pd.Timedelta(minutes=config.transition_alert_merge_gap_minutes)
    groups: list[list[int]] = []
    current_group = [int(risk_positions[0])]
    for position in risk_positions[1:]:
        position = int(position)
        if times[position] - times[current_group[-1]] <= max_gap:
            current_group.append(position)
        else:
            groups.append(current_group)
            current_group = [position]
    groups.append(current_group)

    change_hours = (
        np.asarray(test.time_to_change_hours, dtype=float)
        if test.time_to_change_hours is not None
        else np.full(len(test.y), np.nan, dtype=float)
    )
    episode_rows: list[dict[str, object]] = []
    covered_events: list[pd.Timestamp] = []
    for episode_id, positions in enumerate(groups, start=1):
        first = positions[0]
        last = positions[-1]
        lead = float(change_hours[first]) if np.isfinite(change_hours[first]) else np.nan
        success = bool(np.isfinite(lead) and 0 < lead <= horizon_hours)
        event_time = times[first] + pd.Timedelta(hours=lead) if success else pd.NaT
        if success:
            covered_events.append(pd.Timestamp(event_time))
        episode_rows.append(
            {
                "fold_id": fold_id,
                "target": target,
                "episode_id": episode_id,
                "episode_start": times[first],
                "episode_end": times[last],
                "episode_duration_hours": float(
                    (times[last] - times[first]) / pd.Timedelta(hours=1)
                    + config.decision_interval_minutes / 60.0
                ),
                "first_prediction": float(test_prediction[first]),
                "minimum_prediction": float(np.min(test_prediction[positions])),
                "alert_threshold": threshold,
                "actual_transition_within_horizon": success,
                "lead_hours": lead if success else np.nan,
                "transition_time": event_time,
                "points_in_episode": len(positions),
            }
        )
    episodes = pd.DataFrame(episode_rows)

    possible_events: set[pd.Timestamp] = set()
    for timestamp, lead in zip(times, change_hours):
        if np.isfinite(lead) and 0 < lead <= horizon_hours:
            possible_events.add(pd.Timestamp(timestamp + pd.Timedelta(hours=float(lead))))
    covered = set(covered_events)
    successes = int(episodes["actual_transition_within_horizon"].sum())
    months = max((times[-1] - times[0]) / pd.Timedelta(days=30.4375), 1.0)
    metrics = {
        "fold_id": fold_id,
        "target": target,
        "train_alert_quantile": config.transition_alert_train_quantile,
        "alert_threshold": threshold,
        "episodes": int(len(episodes)),
        "episodes_per_month": float(len(episodes) / months),
        "successful_episodes": successes,
        "success_rate": float(successes / len(episodes)) if len(episodes) else np.nan,
        "false_alert_rate": float(1.0 - successes / len(episodes)) if len(episodes) else np.nan,
        "median_lead_hours": float(
            episodes.loc[episodes["actual_transition_within_horizon"], "lead_hours"].median()
        )
        if successes
        else np.nan,
        "p25_lead_hours": float(
            episodes.loc[episodes["actual_transition_within_horizon"], "lead_hours"].quantile(0.25)
        )
        if successes
        else np.nan,
        "transition_events": int(len(possible_events)),
        "covered_transition_events": int(len(covered & possible_events)),
        "transition_event_coverage": float(len(covered & possible_events) / len(possible_events))
        if possible_events
        else np.nan,
        "median_episode_duration_hours": float(episodes["episode_duration_hours"].median()),
    }
    return metrics, episodes


def _safe_auc(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(roc_auc_score(y, prediction)) if len(np.unique(y)) > 1 else np.nan


def evaluate_continuity_model(
    model,
    data: ContinuityPeriodData,
    *,
    fold_id: str,
    architecture: str,
    target: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    prediction = np.asarray(model.predict_proba(data.x)[:, 1], dtype=float)
    y = np.asarray(data.y, dtype=int)
    base_rate = float(np.mean(y))
    brier = float(brier_score_loss(y, prediction))
    baseline_brier = float(brier_score_loss(y, np.full(len(y), base_rate)))
    order = np.argsort(prediction, kind="stable")
    bucket = np.empty(len(order), dtype=np.int8)
    bucket[order] = np.minimum((np.arange(len(order)) * 10 // max(len(order), 1)) + 1, 10)
    rows: list[dict[str, object]] = []
    for decile in range(1, 11):
        mask = bucket == decile
        if not np.any(mask):
            continue
        rows.append(
            {
                "fold_id": fold_id,
                "architecture": architecture,
                "target": target,
                "decile": decile,
                "rows": int(np.sum(mask)),
                "mean_prediction": float(np.mean(prediction[mask])),
                "actual_persistence_rate": float(np.mean(y[mask])),
                "actual_transition_rate": float(np.mean(1 - y[mask])),
            }
        )
    curve = pd.DataFrame(rows)
    top_mask = bucket == 10
    bottom_mask = bucket == 1
    transition_rate = float(np.mean(1 - y))
    bottom_transition_rate = float(np.mean(1 - y[bottom_mask])) if np.any(bottom_mask) else np.nan
    top_persist_rate = float(np.mean(y[top_mask])) if np.any(top_mask) else np.nan
    transition_total = int(np.sum(1 - y))
    transition_capture = (
        float(np.sum((1 - y[bottom_mask])) / transition_total)
        if np.any(bottom_mask) and transition_total > 0
        else np.nan
    )
    metrics = {
        "fold_id": fold_id,
        "architecture": architecture,
        "target": target,
        "rows": int(len(y)),
        "persistence_rate": base_rate,
        "transition_rate": transition_rate,
        "auc": _safe_auc(y, prediction),
        "average_precision": float(average_precision_score(y, prediction)) if len(np.unique(y)) > 1 else np.nan,
        "brier": brier,
        "baseline_brier": baseline_brier,
        "brier_skill": float(1.0 - brier / baseline_brier) if baseline_brier > 0 else np.nan,
        "top_decile_persistence_rate": top_persist_rate,
        "top_decile_persistence_lift": float(top_persist_rate / base_rate) if base_rate > 0 else np.nan,
        "bottom_decile_transition_rate": bottom_transition_rate,
        "bottom_decile_transition_lift": (
            float(bottom_transition_rate / transition_rate) if transition_rate > 0 else np.nan
        ),
        "transition_capture_bottom_decile": transition_capture,
        "prediction_mean": float(np.mean(prediction)),
        "prediction_std": float(np.std(prediction)),
    }
    samples = pd.DataFrame(
        {
            "decision_time": data.index,
            "fold_id": fold_id,
            "architecture": architecture,
            "target": target,
            "actual_persist": y,
            "prediction": prediction,
            "decile": bucket,
        }
    )
    return metrics, curve


def prediction_samples(
    model,
    data: ContinuityPeriodData,
    *,
    fold_id: str,
    architecture: str,
    target: str,
    max_rows: int = 5_000,
) -> pd.DataFrame:
    prediction = np.asarray(model.predict_proba(data.x)[:, 1], dtype=float)
    if len(prediction) > max_rows:
        positions = np.linspace(0, len(prediction) - 1, max_rows, dtype=np.int64)
    else:
        positions = np.arange(len(prediction))
    return pd.DataFrame(
        {
            "decision_time": data.index[positions],
            "fold_id": fold_id,
            "architecture": architecture,
            "target": target,
            "actual_persist": data.y[positions],
            "prediction": prediction[positions],
        }
    )


def feature_importance_frame(
    model,
    columns: tuple[str, ...],
    *,
    fold_id: str,
    architecture: str,
    target: str,
) -> pd.DataFrame:
    importance = np.asarray(getattr(model, "feature_importances_", np.zeros(len(columns))), dtype=float)
    frame = pd.DataFrame(
        {
            "fold_id": fold_id,
            "architecture": architecture,
            "target": target,
            "feature": columns,
            "importance": importance,
        }
    )
    return frame.sort_values("importance", ascending=False, kind="stable").head(150)


def select_stable_candidates(
    metrics: pd.DataFrame,
    config: MarketStateContinuityConfig,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (architecture, target), group in metrics.groupby(["architecture", "target"], sort=False):
        by_fold = {str(row["fold_id"]): row for row in group.to_dict("records")}
        if "WF_2024" not in by_fold or "WF_2025" not in by_fold:
            continue
        first = by_fold["WF_2024"]
        second = by_fold["WF_2025"]
        passed = (
            float(first["auc"]) >= config.minimum_auc
            and float(second["auc"]) >= config.minimum_auc
            and float(first["brier_skill"]) >= config.minimum_brier_skill
            and float(second["brier_skill"]) >= config.minimum_brier_skill
            and float(first["bottom_decile_transition_lift"]) >= config.minimum_transition_lift
            and float(second["bottom_decile_transition_lift"]) >= config.minimum_transition_lift
        )
        rows.append(
            {
                "architecture": architecture,
                "target": target,
                "WF_2024_auc": first["auc"],
                "WF_2025_auc": second["auc"],
                "WF_2024_brier_skill": first["brier_skill"],
                "WF_2025_brier_skill": second["brier_skill"],
                "WF_2024_transition_lift": first["bottom_decile_transition_lift"],
                "WF_2025_transition_lift": second["bottom_decile_transition_lift"],
                "WF_2024_transition_capture": first["transition_capture_bottom_decile"],
                "WF_2025_transition_capture": second["transition_capture_bottom_decile"],
                "minimum_auc": min(float(first["auc"]), float(second["auc"])),
                "minimum_brier_skill": min(float(first["brier_skill"]), float(second["brier_skill"])),
                "minimum_transition_lift": min(
                    float(first["bottom_decile_transition_lift"]),
                    float(second["bottom_decile_transition_lift"]),
                ),
                "passed": bool(passed),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["stability_score"] = (
        frame["minimum_auc"]
        + 0.5 * frame["minimum_brier_skill"]
        + 0.1 * frame["minimum_transition_lift"]
    )
    return frame.sort_values(["passed", "stability_score"], ascending=[False, False], kind="stable")


def attribution_specs() -> tuple[tuple[str, str, str], ...]:
    return (
        ("train_2023_test_2024", "2023-01-01", "2024-12-31 23:59:59"),
        ("train_2021_2023_test_2024", "2021-01-01", "2024-12-31 23:59:59"),
        ("train_2023_test_2025", "2023-01-01", "2025-12-31 23:59:59"),
        ("train_2024_test_2025", "2024-01-01", "2025-12-31 23:59:59"),
        ("train_2023_2024_test_2025", "2023-01-01", "2025-12-31 23:59:59"),
        ("train_2021_2024_test_2025", "2021-01-01", "2025-12-31 23:59:59"),
    )


def run_training_attribution(
    state_paths: list[Path],
    *,
    target: str,
    config: MarketStateContinuityConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    embargo = pd.Timedelta(hours=config.maximum_target_horizon_hours + 24)
    definitions = (
        ("train_2023_test_2024", "2023-01-01", "2023-12-31 23:59:59", "2024-01-01", "2024-12-31 23:59:59"),
        ("train_2021_2023_test_2024", "2021-01-01", "2023-12-31 23:59:59", "2024-01-01", "2024-12-31 23:59:59"),
        ("train_2023_test_2025", "2023-01-01", "2023-12-31 23:59:59", "2025-01-01", "2025-12-31 23:59:59"),
        ("train_2024_test_2025", "2024-01-01", "2024-12-31 23:59:59", "2025-01-01", "2025-12-31 23:59:59"),
        ("train_2023_2024_test_2025", "2023-01-01", "2024-12-31 23:59:59", "2025-01-01", "2025-12-31 23:59:59"),
        ("train_2021_2024_test_2025", "2021-01-01", "2024-12-31 23:59:59", "2025-01-01", "2025-12-31 23:59:59"),
    )
    for label, train_start, train_end, test_start, test_end in definitions:
        fit = collect_continuity_period_data(
            state_paths,
            [],
            start=pd.Timestamp(train_start),
            end=pd.Timestamp(train_end) - embargo,
            target=target,
            architecture="universal_ohlcv_lightgbm",
        )
        test = collect_continuity_period_data(
            state_paths,
            [],
            start=pd.Timestamp(test_start),
            end=pd.Timestamp(test_end),
            target=target,
            architecture="universal_ohlcv_lightgbm",
        )
        model = fit_continuity_model(fit, config)
        metrics, _ = evaluate_continuity_model(
            model,
            test,
            fold_id=label,
            architecture="universal_ohlcv_lightgbm",
            target=target,
        )
        metrics["train_start"] = train_start
        metrics["train_end"] = str(pd.Timestamp(train_end) - embargo)
        metrics["test_start"] = test_start
        metrics["test_end"] = test_end
        rows.append(metrics)
    return pd.DataFrame(rows)
