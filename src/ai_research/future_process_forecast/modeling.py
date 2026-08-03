#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Chronological multi-label models and anti-tail-car evaluation for R03.3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None  # type: ignore[assignment]

from src.ai_research.swing_baseline.dataset import load_year_shard

from .config import FutureProcessForecastConfig, PROCESS_TYPES
from .events import load_event_year_shard
from .micro_features import load_micro_year_shard


class BinaryClassifier(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray): ...
    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class ForecastFold:
    fold_id: str
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        return {key: str(value) if isinstance(value, pd.Timestamp) else value for key, value in payload.items()}


def default_folds(config: FutureProcessForecastConfig) -> tuple[ForecastFold, ...]:
    # A label may look 24h to the next start and the event detector may inspect a
    # further 24h path.  The 72h embargo is deliberately conservative.
    embargo = pd.Timedelta(hours=72)
    return (
        ForecastFold(
            "WF_2024",
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-10-01") - embargo,
            pd.Timestamp("2023-10-01"),
            pd.Timestamp("2023-12-31 23:59:59"),
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-12-31 23:59:59"),
        ),
        ForecastFold(
            "WF_2025",
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2024-10-01") - embargo,
            pd.Timestamp("2024-10-01"),
            pd.Timestamp("2024-12-31 23:59:59"),
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-12-31 23:59:59"),
        ),
    )


@dataclass(frozen=True)
class ForecastPeriodData:
    timestamps_ns: np.ndarray
    macro_x: np.ndarray
    full_x: np.ndarray
    micro_x: np.ndarray
    combined_x: np.ndarray
    labels: dict[str, np.ndarray]
    diagnostics: dict[str, np.ndarray]
    macro_columns: tuple[str, ...]
    full_columns: tuple[str, ...]
    micro_columns: tuple[str, ...]

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.to_datetime(self.timestamps_ns)


def _year_map(paths: list[Path], kind: str) -> dict[int, Path]:
    output: dict[int, Path] = {}
    for path in paths:
        if kind == "base":
            shard = load_year_shard(path)
            times = shard.decision_times_ns
        elif kind == "event":
            shard = load_event_year_shard(path)
            times = shard.decision_times_ns
        else:
            shard = load_micro_year_shard(path)
            times = shard.decision_times_ns
        year = int(pd.to_datetime(np.asarray(times[:1], dtype=np.int64))[0].year)
        output[year] = path
    return output


def collect_period_data(
    base_paths: list[Path],
    event_paths: list[Path],
    micro_paths: list[Path],
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: FutureProcessForecastConfig,
) -> ForecastPeriodData:
    base_map = _year_map(base_paths, "base")
    event_map = _year_map(event_paths, "event")
    micro_map = _year_map(micro_paths, "micro")
    macro_parts: list[np.ndarray] = []
    full_parts: list[np.ndarray] = []
    micro_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    label_names = [f"{process}_start_h{h}" for process in PROCESS_TYPES for h in config.forecast_horizons_hours]
    diagnostic_names = [
        f"{process}_{suffix}"
        for process in PROCESS_TYPES[:-1]
        for suffix in ("next_lead_hours", "ongoing", "progress", "next_event_id")
    ]
    label_parts: dict[str, list[np.ndarray]] = {name: [] for name in label_names}
    diagnostic_parts: dict[str, list[np.ndarray]] = {name: [] for name in diagnostic_names}
    expected_macro: tuple[str, ...] | None = None
    expected_full: tuple[str, ...] | None = None
    expected_micro: tuple[str, ...] | None = None
    for year in sorted(base_map):
        if year not in event_map or year not in micro_map:
            continue
        base = load_year_shard(base_map[year])
        event = load_event_year_shard(event_map[year])
        micro = load_micro_year_shard(micro_map[year])
        if not (
            np.array_equal(base.decision_times_ns, event.decision_times_ns)
            and np.array_equal(base.decision_times_ns, micro.decision_times_ns)
        ):
            raise RuntimeError(f"R03.3 decision-axis mismatch in {year}")
        times = np.asarray(base.decision_times_ns, dtype=np.int64)
        left = int(np.searchsorted(times, int(start.value), side="left"))
        right = int(np.searchsorted(times, int(end.value), side="right"))
        if right <= left:
            continue
        if expected_macro is None:
            expected_macro = base.high_feature_columns
            expected_full = base.full_feature_columns
            expected_micro = micro.feature_columns
        if base.high_feature_columns != expected_macro or base.full_feature_columns != expected_full:
            raise RuntimeError(f"R03.3 base feature schema mismatch in {year}")
        if micro.feature_columns != expected_micro:
            raise RuntimeError(f"R03.3 micro feature schema mismatch in {year}")
        high_count = len(base.high_feature_columns)
        macro_parts.append(np.asarray(base.features[left:right, :high_count], dtype=np.float32))
        full_parts.append(np.asarray(base.features[left:right], dtype=np.float32))
        micro_parts.append(np.asarray(micro.features[left:right], dtype=np.float32))
        time_parts.append(times[left:right])
        label_map = event.label_index
        for name in label_names:
            label_parts[name].append(np.asarray(event.labels[left:right, label_map[name]], dtype=np.float32))
        for name in diagnostic_names:
            diagnostic_parts[name].append(np.asarray(event.labels[left:right, label_map[name]], dtype=np.float32))
    if not time_parts:
        raise RuntimeError(f"no R03.3 rows for {start} -> {end}")
    timestamps = np.concatenate(time_parts)
    order = np.argsort(timestamps, kind="stable")
    return ForecastPeriodData(
        timestamps_ns=timestamps[order],
        macro_x=np.concatenate(macro_parts)[order],
        full_x=np.concatenate(full_parts)[order],
        micro_x=np.concatenate(micro_parts)[order],
        combined_x=np.concatenate([np.concatenate(full_parts)[order], np.concatenate(micro_parts)[order]], axis=1),
        labels={name: np.concatenate(parts)[order] for name, parts in label_parts.items()},
        diagnostics={name: np.concatenate(parts)[order] for name, parts in diagnostic_parts.items()},
        macro_columns=expected_macro or (),
        full_columns=expected_full or (),
        micro_columns=expected_micro or (),
    )


def validate_model_dependencies(config: FutureProcessForecastConfig) -> None:
    if any("lightgbm" in name for name in config.architectures) and LGBMClassifier is None:
        raise RuntimeError(
            "R03.3 dependency preflight failed: LightGBM is not installed. "
            "Install it before cache work with: python -m pip install lightgbm"
        )


def architecture_matrix(
    architecture: str,
    data: ForecastPeriodData,
) -> tuple[np.ndarray, tuple[str, ...], str]:
    if architecture == "macro_lightgbm":
        return data.macro_x, data.macro_columns, "lightgbm"
    if architecture == "multiframe_lightgbm":
        return data.full_x, data.full_columns, "lightgbm"
    combined = data.combined_x
    columns = (*data.full_columns, *data.micro_columns)
    if architecture == "multiframe_micro_lightgbm":
        return combined, columns, "lightgbm"
    if architecture == "multiframe_micro_logistic":
        return combined, columns, "logistic"
    raise ValueError(f"unknown R03.3 architecture: {architecture}")


def _create_model(kind: str, config: FutureProcessForecastConfig) -> BinaryClassifier:
    if kind == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=config.logistic_c,
                        class_weight="balanced",
                        max_iter=config.logistic_max_iter,
                        solver="lbfgs",
                        random_state=config.random_seed,
                    ),
                ),
            ]
        )
    if kind == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("LightGBM is required")
        return LGBMClassifier(
            objective="binary",
            n_estimators=config.lightgbm_n_estimators,
            learning_rate=config.lightgbm_learning_rate,
            num_leaves=config.lightgbm_num_leaves,
            min_child_samples=config.lightgbm_min_child_samples,
            colsample_bytree=config.lightgbm_feature_fraction,
            subsample=0.85,
            subsample_freq=1,
            reg_alpha=0.35,
            reg_lambda=1.5,
            class_weight="balanced",
            random_state=config.random_seed,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(kind)


def _raw_probability(model: BinaryClassifier, x: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


@dataclass
class PlattCalibrator:
    model: LogisticRegression | None = None

    def fit(self, raw_probability: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        valid = np.isfinite(raw_probability) & np.isfinite(y)
        p = np.clip(raw_probability[valid], 1e-6, 1.0 - 1e-6)
        labels = y[valid].astype(int)
        if len(p) < 20 or len(np.unique(labels)) < 2:
            self.model = None
            return self
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        model = LogisticRegression(C=100.0, max_iter=1000, solver="lbfgs")
        model.fit(logits, labels)
        self.model = model
        return self

    def predict(self, raw_probability: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6)
        if self.model is None:
            return p
        logits = np.log(p / (1.0 - p)).reshape(-1, 1)
        return np.asarray(self.model.predict_proba(logits)[:, 1], dtype=float)


def _training_positions(
    data: ForecastPeriodData,
    x: np.ndarray,
    y: np.ndarray,
    config: FutureProcessForecastConfig,
) -> np.ndarray:
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    positions = np.flatnonzero(valid)[:: config.sample_stride_decisions]
    if len(positions) == 0:
        raise RuntimeError("R03.3 training period has zero valid strided rows")
    if len(positions) > config.train_sample_cap:
        rng = np.random.default_rng(config.random_seed)
        positions = np.sort(rng.choice(positions, size=config.train_sample_cap, replace=False))
    if len(np.unique(y[positions].astype(int))) < 2:
        raise RuntimeError("R03.3 training label has only one class")
    return positions


def _event_balanced_weights(y: np.ndarray) -> np.ndarray:
    """Give each contiguous positive forecast window approximately equal mass."""
    labels = np.asarray(y, dtype=int)
    weights = np.ones(len(labels), dtype=float)
    positive = labels == 1
    if not np.any(positive):
        return weights
    starts = np.flatnonzero(positive & ~np.r_[False, positive[:-1]])
    ends = np.flatnonzero(positive & ~np.r_[positive[1:], False]) + 1
    for left, right in zip(starts, ends, strict=True):
        weights[left:right] = 1.0 / max(right - left, 1)
    mean = float(np.mean(weights))
    return weights / mean if mean > 0 else weights


def fit_one(
    architecture: str,
    label_name: str,
    fit_data: ForecastPeriodData,
    calibration_data: ForecastPeriodData,
    config: FutureProcessForecastConfig,
) -> tuple[BinaryClassifier, PlattCalibrator, np.ndarray, tuple[str, ...], dict[str, object]]:
    fit_x, columns, kind = architecture_matrix(architecture, fit_data)
    calibration_x, _, _ = architecture_matrix(architecture, calibration_data)
    fit_y = fit_data.labels[label_name]
    calibration_y = calibration_data.labels[label_name]
    positions = _training_positions(fit_data, fit_x, fit_y, config)
    model = _create_model(kind, config)
    train_y = fit_y[positions].astype(int)
    weights = _event_balanced_weights(train_y)
    if isinstance(model, Pipeline):
        model.fit(fit_x[positions], train_y, model__sample_weight=weights)
    else:
        model.fit(fit_x[positions], train_y, sample_weight=weights)
    raw_calibration = _raw_probability(model, calibration_x)
    calibrator = PlattCalibrator().fit(raw_calibration, calibration_y)
    calibrated = calibrator.predict(raw_calibration)
    metadata = {
        "architecture": architecture,
        "label": label_name,
        "train_rows": int(len(positions)),
        "train_positive_rate": float(np.mean(train_y)),
        "event_balanced_sample_weight": True,
        "calibration_rows": int(len(calibration_y)),
        "calibration_positive_rate": float(np.mean(calibration_y)),
        "calibrator": "platt" if calibrator.model is not None else "identity",
    }
    return model, calibrator, calibrated, columns, metadata


def _base_metrics(y: np.ndarray, score: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(y) & np.isfinite(score)
    labels = y[valid].astype(int)
    probability = np.clip(score[valid], 1e-7, 1.0 - 1e-7)
    if len(labels) == 0:
        return {"rows": 0, "positive_rate": np.nan, "roc_auc": np.nan, "average_precision": np.nan, "brier": np.nan, "baseline_brier": np.nan}
    rate = float(np.mean(labels))
    return {
        "rows": int(len(labels)),
        "positive_rate": rate,
        "roc_auc": float(roc_auc_score(labels, probability)) if len(np.unique(labels)) == 2 else np.nan,
        "average_precision": float(average_precision_score(labels, probability)) if np.any(labels == 1) else np.nan,
        "brier": float(brier_score_loss(labels, probability)),
        "baseline_brier": rate * (1.0 - rate),
    }


def evaluate_one(
    *,
    fold_id: str,
    architecture: str,
    process: str,
    horizon: int,
    calibration_score: np.ndarray,
    test_score: np.ndarray,
    test_data: ForecastPeriodData,
    config: FutureProcessForecastConfig,
) -> tuple[dict[str, object], list[dict[str, object]], pd.DataFrame]:
    label_name = f"{process}_start_h{horizon}"
    y = test_data.labels[label_name]
    base = {
        "fold_id": fold_id,
        "architecture": architecture,
        "process": process,
        "horizon_hours": horizon,
        **_base_metrics(y, test_score),
    }
    quantile_rows: list[dict[str, object]] = []
    sample_rows: list[pd.DataFrame] = []
    for quantile in config.signal_quantiles:
        threshold = float(np.quantile(calibration_score[np.isfinite(calibration_score)], quantile))
        signal = np.isfinite(test_score) & (test_score >= threshold)
        signals = int(signal.sum())
        positives = int(np.sum(y[signal] > 0.5)) if signals else 0
        precision = positives / signals if signals else np.nan
        baseline = float(np.nanmean(y))
        lift = precision / baseline if signals and baseline > 0 else np.nan
        row: dict[str, object] = {
            "fold_id": fold_id,
            "architecture": architecture,
            "process": process,
            "horizon_hours": horizon,
            "quantile": quantile,
            "threshold": threshold,
            "signals": signals,
            "positives": positives,
            "precision": precision,
            "baseline_rate": baseline,
            "lift": lift,
            "median_true_lead_hours": np.nan,
            "lead_ge_1h_rate": np.nan,
            "lead_ge_3h_rate": np.nan,
            "lead_ge_6h_rate": np.nan,
            "ongoing_rate": np.nan,
            "tail_car_rate_progress30": np.nan,
            "median_progress_if_ongoing": np.nan,
        }
        if process != "low_opportunity" and signals:
            lead = test_data.diagnostics[f"{process}_next_lead_hours"][signal]
            ongoing = test_data.diagnostics[f"{process}_ongoing"][signal] > 0.5
            progress = test_data.diagnostics[f"{process}_progress"][signal]
            true_lead = lead[(y[signal] > 0.5) & np.isfinite(lead)]
            row["median_true_lead_hours"] = float(np.median(true_lead)) if len(true_lead) else np.nan
            row["lead_ge_1h_rate"] = float(np.mean(true_lead >= 1.0)) if len(true_lead) else np.nan
            row["lead_ge_3h_rate"] = float(np.mean(true_lead >= 3.0)) if len(true_lead) else np.nan
            row["lead_ge_6h_rate"] = float(np.mean(true_lead >= 6.0)) if len(true_lead) else np.nan
            row["ongoing_rate"] = float(np.mean(ongoing))
            row["tail_car_rate_progress30"] = float(np.mean(ongoing & (progress >= 0.30)))
            row["median_progress_if_ongoing"] = float(np.median(progress[ongoing])) if np.any(ongoing) else np.nan
        quantile_rows.append(row)
        positions = np.flatnonzero(signal)
        if len(positions):
            sample = pd.DataFrame(
                {
                    "fold_id": fold_id,
                    "architecture": architecture,
                    "process": process,
                    "horizon_hours": horizon,
                    "quantile": quantile,
                    "decision_time": pd.to_datetime(test_data.timestamps_ns[positions]),
                    "score": test_score[positions],
                    "label": y[positions],
                }
            )
            if process != "low_opportunity":
                sample["next_lead_hours"] = test_data.diagnostics[f"{process}_next_lead_hours"][positions]
                sample["ongoing"] = test_data.diagnostics[f"{process}_ongoing"][positions]
                sample["progress"] = test_data.diagnostics[f"{process}_progress"][positions]
            sample_rows.append(sample)
    return base, quantile_rows, pd.concat(sample_rows, ignore_index=True) if sample_rows else pd.DataFrame()


def feature_importance(
    model: BinaryClassifier,
    columns: tuple[str, ...],
    *,
    fold_id: str,
    architecture: str,
    process: str,
    horizon: int,
) -> list[dict[str, object]]:
    raw = model.named_steps["model"] if isinstance(model, Pipeline) else model
    if hasattr(raw, "feature_importances_"):
        values = np.asarray(raw.feature_importances_, dtype=float)
    elif hasattr(raw, "coef_"):
        values = np.abs(np.asarray(raw.coef_[0], dtype=float))
    else:
        return []
    return [
        {
            "fold_id": fold_id,
            "architecture": architecture,
            "process": process,
            "horizon_hours": horizon,
            "feature": feature,
            "importance": float(value),
        }
        for feature, value in zip(columns, values, strict=True)
    ]


def select_stable_candidates(
    probability_metrics: pd.DataFrame,
    quantile_metrics: pd.DataFrame,
) -> pd.DataFrame:
    if probability_metrics.empty or quantile_metrics.empty:
        return pd.DataFrame()
    q95 = quantile_metrics.loc[np.isclose(quantile_metrics["quantile"], 0.95)].copy()
    keys = ["architecture", "process", "horizon_hours"]
    fold_frames: list[pd.DataFrame] = []
    for fold in ("WF_2024", "WF_2025"):
        p = probability_metrics.loc[probability_metrics["fold_id"] == fold, [*keys, "roc_auc", "average_precision", "positive_rate", "brier", "baseline_brier"]].copy()
        q = q95.loc[q95["fold_id"] == fold, [*keys, "signals", "precision", "lift", "median_true_lead_hours", "tail_car_rate_progress30"]].copy()
        merged = p.merge(q, on=keys, how="inner")
        merged = merged.rename(columns={column: f"{fold}_{column}" for column in merged.columns if column not in keys})
        fold_frames.append(merged)
    if len(fold_frames) != 2:
        return pd.DataFrame()
    candidates = fold_frames[0].merge(fold_frames[1], on=keys, how="inner")
    lead_floor = np.where(candidates["horizon_hours"] <= 6, 1.0, 3.0)
    process_mask = candidates["process"] != "low_opportunity"
    stable = (
        (candidates["WF_2024_signals"] >= 20)
        & (candidates["WF_2025_signals"] >= 20)
        & (candidates["WF_2024_roc_auc"] >= 0.56)
        & (candidates["WF_2025_roc_auc"] >= 0.56)
        & (candidates["WF_2024_lift"] >= 1.50)
        & (candidates["WF_2025_lift"] >= 1.50)
        & (candidates["WF_2024_brier"] <= candidates["WF_2024_baseline_brier"])
        & (candidates["WF_2025_brier"] <= candidates["WF_2025_baseline_brier"])
    )
    stable &= (~process_mask) | (
        (candidates["WF_2024_median_true_lead_hours"] >= lead_floor)
        & (candidates["WF_2025_median_true_lead_hours"] >= lead_floor)
        & (candidates["WF_2024_tail_car_rate_progress30"] <= 0.35)
        & (candidates["WF_2025_tail_car_rate_progress30"] <= 0.35)
    )
    candidates["passes"] = stable
    candidates["robust_score"] = (
        candidates[["WF_2024_roc_auc", "WF_2025_roc_auc"]].min(axis=1)
        + 0.25 * candidates[["WF_2024_lift", "WF_2025_lift"]].min(axis=1).clip(upper=4.0)
        - 0.25 * candidates[["WF_2024_tail_car_rate_progress30", "WF_2025_tail_car_rate_progress30"]].fillna(0.0).max(axis=1)
    )
    return candidates.sort_values(["passes", "robust_score"], ascending=[False, False], kind="stable")
