#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Chronological model fitting for the R03 high-context and entry challengers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol

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

from .config import SwingBaselineConfig, SwingTargetSpec
from .dataset import SwingYearShard, load_year_shard


ARCHITECTURES = (
    "high_logistic",
    "high_lightgbm",
    "full_lightgbm",
    "hierarchical_lightgbm",
)


class BinaryClassifier(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray): ...
    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    locked_holdout: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, pd.Timestamp):
                payload[key] = str(value)
        return payload


def default_folds(config: SwingBaselineConfig) -> tuple[WalkForwardFold, ...]:
    embargo = pd.Timedelta(hours=config.max_horizon_hours + 1)

    def fit_end(before: str) -> pd.Timestamp:
        return pd.Timestamp(before) - embargo

    return (
        WalkForwardFold(
            "WF_2024",
            pd.Timestamp("2023-01-01"),
            fit_end("2023-10-01"),
            pd.Timestamp("2023-10-01"),
            pd.Timestamp("2023-12-31 23:59:59"),
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-12-31 23:59:59"),
        ),
        WalkForwardFold(
            "WF_2025",
            pd.Timestamp("2023-01-01"),
            fit_end("2024-10-01"),
            pd.Timestamp("2024-10-01"),
            pd.Timestamp("2024-12-31 23:59:59"),
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-12-31 23:59:59"),
        ),
        WalkForwardFold(
            "WF_2026",
            pd.Timestamp("2023-01-01"),
            fit_end("2025-10-01"),
            pd.Timestamp("2025-10-01"),
            pd.Timestamp("2025-12-31 23:59:59"),
            pd.Timestamp("2026-01-01"),
            pd.Timestamp(config.research_end),
            locked_holdout=True,
        ),
    )


@dataclass(frozen=True)
class PeriodData:
    timestamps_ns: np.ndarray
    high_x: np.ndarray
    full_x: np.ndarray
    labels: dict[str, np.ndarray]
    context: np.ndarray
    context_columns: tuple[str, ...]
    entry_times_ns: np.ndarray
    entry_prices: np.ndarray

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.to_datetime(self.timestamps_ns)


@dataclass(frozen=True)
class ProbabilityMetrics:
    rows: int
    positive_rate: float
    roc_auc: float
    average_precision: float
    brier: float
    score_mean: float
    score_std: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _period_parts(
    paths: Iterable[Path],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[tuple[SwingYearShard, slice]]:
    parts: list[tuple[SwingYearShard, slice]] = []
    for path in paths:
        shard = load_year_shard(path)
        positions = shard.decision_positions(start, end)
        if int(positions.stop or 0) > int(positions.start or 0):
            parts.append((shard, positions))
    return parts


def collect_period_data(
    paths: Iterable[Path],
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    label_names: Iterable[str],
) -> PeriodData:
    parts = _period_parts(paths, start, end)
    if not parts:
        raise RuntimeError(f"no R03 samples for {start} -> {end}")
    high_parts: list[np.ndarray] = []
    full_parts: list[np.ndarray] = []
    context_parts: list[np.ndarray] = []
    timestamp_parts: list[np.ndarray] = []
    entry_time_parts: list[np.ndarray] = []
    entry_price_parts: list[np.ndarray] = []
    label_parts: dict[str, list[np.ndarray]] = {name: [] for name in label_names}
    expected_high: tuple[str, ...] | None = None
    expected_full: tuple[str, ...] | None = None
    expected_context: tuple[str, ...] | None = None
    for shard, positions in parts:
        if expected_high is None:
            expected_high = shard.high_feature_columns
            expected_full = shard.full_feature_columns
            expected_context = shard.context_columns
        if shard.high_feature_columns != expected_high or shard.full_feature_columns != expected_full:
            raise RuntimeError(f"R03 feature schema mismatch in {shard.path}")
        if shard.context_columns != expected_context:
            raise RuntimeError(f"R03 context schema mismatch in {shard.path}")
        high_count = len(shard.high_feature_columns)
        high_parts.append(np.asarray(shard.features[positions, :high_count], dtype=np.float32))
        full_parts.append(np.asarray(shard.features[positions], dtype=np.float32))
        context_parts.append(np.asarray(shard.context[positions], dtype=np.float64))
        timestamp_parts.append(np.asarray(shard.decision_times_ns[positions], dtype=np.int64))
        entry_time_parts.append(np.asarray(shard.entry_times_ns[positions], dtype=np.int64))
        entry_price_parts.append(np.asarray(shard.entry_prices[positions], dtype=np.float64))
        label_map = shard.label_index
        for name in label_parts:
            if name not in label_map:
                raise RuntimeError(f"missing R03 label {name} in {shard.path}")
            label_parts[name].append(np.asarray(shard.labels[positions, label_map[name]], dtype=np.float32))
    timestamps = np.concatenate(timestamp_parts)
    order = np.argsort(timestamps, kind="stable")
    return PeriodData(
        timestamps_ns=timestamps[order],
        high_x=np.concatenate(high_parts)[order],
        full_x=np.concatenate(full_parts)[order],
        labels={name: np.concatenate(values)[order] for name, values in label_parts.items()},
        context=np.concatenate(context_parts)[order],
        context_columns=expected_context or (),
        entry_times_ns=np.concatenate(entry_time_parts)[order],
        entry_prices=np.concatenate(entry_price_parts)[order],
    )


def _mix_u64(values: np.ndarray, seed: int) -> np.ndarray:
    x = values.astype(np.uint64, copy=False) + np.uint64(seed)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


def _sample_training_rows(data: PeriodData, labels: list[np.ndarray], cap: int, seed: int) -> np.ndarray:
    valid = np.isfinite(data.high_x).all(axis=1) & np.isfinite(data.full_x).all(axis=1)
    for label in labels:
        valid &= np.isfinite(label)
    positions = np.flatnonzero(valid)
    if len(positions) == 0:
        raise RuntimeError("R03 training period has zero valid rows")
    if len(positions) > cap:
        hashes = _mix_u64(data.timestamps_ns[positions].astype(np.uint64), seed)
        positions = positions[np.argsort(hashes, kind="stable")[:cap]]
    return positions


def validate_model_dependencies(architectures: Iterable[str]) -> dict[str, str]:
    requested = tuple(dict.fromkeys(architectures))
    unknown = sorted(set(requested) - set(ARCHITECTURES))
    if unknown:
        raise ValueError(f"unknown R03 architectures: {unknown}")
    needs_lightgbm = any("lightgbm" in name for name in requested)
    if needs_lightgbm and LGBMClassifier is None:
        raise RuntimeError(
            "R03 startup dependency check failed: LightGBM is not installed.\n"
            "Install it before any cache work with:\n"
            "  python -m pip install lightgbm\n"
            "Then rerun the same command without --force-rebuild-cache."
        )
    return {name: "available" for name in requested}


def _create_classifier(kind: str, config: SwingBaselineConfig) -> BinaryClassifier:
    if kind == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=500,
                        solver="lbfgs",
                        random_state=config.random_seed,
                    ),
                ),
            ]
        )
    if kind == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("lightgbm is required for the R03 tree models")
        return LGBMClassifier(
            objective="binary",
            n_estimators=config.lightgbm_n_estimators,
            learning_rate=config.lightgbm_learning_rate,
            num_leaves=config.lightgbm_num_leaves,
            min_child_samples=config.lightgbm_min_child_samples,
            colsample_bytree=config.lightgbm_feature_fraction,
            subsample=0.85,
            subsample_freq=1,
            reg_alpha=0.2,
            reg_lambda=1.0,
            class_weight="balanced",
            random_state=config.random_seed,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"unknown classifier kind: {kind}")


def _predict_probability(model: BinaryClassifier, x: np.ndarray) -> np.ndarray:
    booster = getattr(model, "booster_", None)
    if booster is not None:
        return np.asarray(booster.predict(x), dtype=float)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=float)


@dataclass
class ModelPair:
    long_model: BinaryClassifier
    short_model: BinaryClassifier
    feature_columns: tuple[str, ...]


@dataclass
class SwingModelBundle:
    architecture: str
    target_id: str
    high_pair: ModelPair | None
    full_pair: ModelPair | None

    def predict(self, high_x: np.ndarray, full_x: np.ndarray) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        if self.high_pair is not None:
            output["high_long"] = _predict_probability(self.high_pair.long_model, high_x)
            output["high_short"] = _predict_probability(self.high_pair.short_model, high_x)
        if self.full_pair is not None:
            output["full_long"] = _predict_probability(self.full_pair.long_model, full_x)
            output["full_short"] = _predict_probability(self.full_pair.short_model, full_x)
        if self.architecture.startswith("high_"):
            output["score_long"] = output["high_long"]
            output["score_short"] = output["high_short"]
        elif self.architecture == "full_lightgbm":
            output["score_long"] = output["full_long"]
            output["score_short"] = output["full_short"]
        elif self.architecture == "hierarchical_lightgbm":
            output["score_long"] = np.minimum(output["high_long"], output["full_long"])
            output["score_short"] = np.minimum(output["high_short"], output["full_short"])
        else:
            raise ValueError(f"unsupported architecture: {self.architecture}")
        return output


def _fit_pair(
    kind: str,
    x: np.ndarray,
    long_y: np.ndarray,
    short_y: np.ndarray,
    feature_columns: tuple[str, ...],
    config: SwingBaselineConfig,
) -> ModelPair:
    for label_name, y in (("long", long_y), ("short", short_y)):
        classes = np.unique(y.astype(int))
        if len(classes) < 2:
            raise RuntimeError(f"R03 {label_name} label has only one class in the training period")
    long_model = _create_classifier(kind, config)
    short_model = _create_classifier(kind, config)
    long_model.fit(x, long_y.astype(int))
    short_model.fit(x, short_y.astype(int))
    return ModelPair(long_model=long_model, short_model=short_model, feature_columns=feature_columns)


def fit_model_bundle_from_period(
    architecture: str,
    target: SwingTargetSpec,
    data: PeriodData,
    *,
    high_columns: tuple[str, ...],
    full_columns: tuple[str, ...],
    config: SwingBaselineConfig,
    metadata_context: dict[str, object] | None = None,
) -> tuple[SwingModelBundle, dict[str, object]]:
    """Fit one R03 model bundle from an already assembled chronological period.

    R03.1 reuses the frozen feature cache while replacing only the training labels
    with exact first-hit outcomes. Keeping the fitting path here prevents a second
    model implementation from drifting away from the original R03 baseline.
    """
    long_label = f"{target.target_id}_long_quality"
    short_label = f"{target.target_id}_short_quality"
    missing = sorted({long_label, short_label} - set(data.labels))
    if missing:
        raise RuntimeError(f"R03 period missing model labels: {missing}")
    positions = _sample_training_rows(
        data,
        [data.labels[long_label], data.labels[short_label]],
        config.train_sample_cap,
        config.random_seed + target.horizon_hours,
    )
    high_pair: ModelPair | None = None
    full_pair: ModelPair | None = None
    kind = "logistic" if architecture == "high_logistic" else "lightgbm"
    if architecture in {"high_logistic", "high_lightgbm", "hierarchical_lightgbm"}:
        high_pair = _fit_pair(
            kind,
            data.high_x[positions],
            data.labels[long_label][positions],
            data.labels[short_label][positions],
            high_columns,
            config,
        )
    if architecture in {"full_lightgbm", "hierarchical_lightgbm"}:
        full_pair = _fit_pair(
            "lightgbm",
            data.full_x[positions],
            data.labels[long_label][positions],
            data.labels[short_label][positions],
            full_columns,
            config,
        )
    bundle = SwingModelBundle(architecture, target.target_id, high_pair, full_pair)
    metadata = {
        "architecture": architecture,
        "target": target.to_dict(),
        "train_rows": int(len(positions)),
        "long_positive_rate": float(np.mean(data.labels[long_label][positions])),
        "short_positive_rate": float(np.mean(data.labels[short_label][positions])),
        "train_start_actual": str(pd.to_datetime(data.timestamps_ns[positions].min())),
        "train_end_actual": str(pd.to_datetime(data.timestamps_ns[positions].max())),
    }
    if metadata_context:
        metadata.update(metadata_context)
    return bundle, metadata


def fit_model_bundle(
    architecture: str,
    target: SwingTargetSpec,
    paths: Iterable[Path],
    fold: WalkForwardFold,
    config: SwingBaselineConfig,
) -> tuple[SwingModelBundle, dict[str, object]]:
    paths = list(paths)
    if not paths:
        raise RuntimeError("no R03 cache shards supplied")
    long_label = f"{target.target_id}_long_quality"
    short_label = f"{target.target_id}_short_quality"
    data = collect_period_data(paths, fold.fit_start, fold.fit_end, label_names=(long_label, short_label))
    first_shard = load_year_shard(paths[0])
    return fit_model_bundle_from_period(
        architecture,
        target,
        data,
        high_columns=first_shard.high_feature_columns,
        full_columns=first_shard.full_feature_columns,
        config=config,
        metadata_context={"fold": fold.to_dict()},
    )


def probability_metrics(y_true: np.ndarray, score: np.ndarray) -> ProbabilityMetrics:
    valid = np.isfinite(y_true) & np.isfinite(score)
    y = y_true[valid].astype(int)
    p = np.clip(score[valid], 1e-7, 1.0 - 1e-7)
    if len(y) == 0:
        return ProbabilityMetrics(0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    roc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
    ap = float(average_precision_score(y, p)) if np.any(y == 1) else float("nan")
    return ProbabilityMetrics(
        rows=int(len(y)),
        positive_rate=float(np.mean(y)),
        roc_auc=roc,
        average_precision=ap,
        brier=float(brier_score_loss(y, p)),
        score_mean=float(np.mean(p)),
        score_std=float(np.std(p)),
    )


def feature_importance_rows(bundle: SwingModelBundle) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for layer_name, pair in (("high", bundle.high_pair), ("full", bundle.full_pair)):
        if pair is None:
            continue
        for direction, model in (("long", pair.long_model), ("short", pair.short_model)):
            raw_model = model.named_steps["model"] if isinstance(model, Pipeline) else model
            if hasattr(raw_model, "feature_importances_"):
                values = np.asarray(raw_model.feature_importances_, dtype=float)
            elif hasattr(raw_model, "coef_"):
                values = np.abs(np.asarray(raw_model.coef_[0], dtype=float))
            else:
                continue
            for feature, value in zip(pair.feature_columns, values, strict=True):
                rows.append(
                    {
                        "architecture": bundle.architecture,
                        "target_id": bundle.target_id,
                        "layer": layer_name,
                        "direction": direction,
                        "feature": feature,
                        "importance": float(value),
                    }
                )
    return rows
