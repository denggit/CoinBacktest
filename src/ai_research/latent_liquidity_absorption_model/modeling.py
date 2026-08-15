#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fixed R01.3 multi-task LightGBM models and causal period evaluation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMClassifier = None  # type: ignore[assignment]
    LGBMRegressor = None  # type: ignore[assignment]

from .config import AbsorptionModelConfig

_META_COLUMNS = {
    "event_id", "release_episode_id", "event_time", "decision_time", "entry_time",
    "event_side", "period", "event_reference_price", "known_extreme_price", "current_close",
    "feature_available_time", "entry_price", "barrier_result",
}
_LABEL_PREFIXES = ("future_", "tradeable_", "barrier_result_", "absorption_complete_target")
_RAW_PRICE_SUFFIXES = ("_price",)


@dataclass
class FittedModels:
    feature_columns: tuple[str, ...]
    tradeable_model: Pipeline
    absorption_model: Pipeline
    extension_model: Pipeline
    mfe_model: Pipeline
    baseline_model: Pipeline


def _validate_dependencies() -> None:
    if LGBMClassifier is None or LGBMRegressor is None:
        raise RuntimeError("R01.3 requires lightgbm. Install with: python -m pip install lightgbm")


def prepare_design(frame: pd.DataFrame, feature_columns: tuple[str, ...] | None = None) -> tuple[pd.DataFrame, tuple[str, ...]]:
    work = frame.copy()
    work["side_is_down"] = work["event_side"].astype(str).eq("DOWN").astype(np.int8)
    for cluster in sorted(pd.to_numeric(work["path_cluster"], errors="coerce").dropna().astype(int).unique()):
        work[f"cluster_{cluster}"] = pd.to_numeric(work["path_cluster"], errors="coerce").eq(cluster).astype(np.int8)
    if feature_columns is None:
        columns: list[str] = []
        for name in work.columns:
            if name in _META_COLUMNS or name == "path_cluster":
                continue
            if name.startswith(_LABEL_PREFIXES):
                continue
            if name.startswith("entry_price_d") or name.startswith("structural_stop_distance_bp_d"):
                continue
            if name.endswith(_RAW_PRICE_SUFFIXES):
                continue
            if pd.api.types.is_numeric_dtype(work[name]):
                columns.append(name)
        feature_columns = tuple(columns)
    missing = [name for name in feature_columns if name not in work]
    if missing:
        raise RuntimeError(f"R01.3 feature schema mismatch: {missing[:10]}")
    return work.loc[:, feature_columns].apply(pd.to_numeric, errors="coerce"), feature_columns


def baseline_columns(all_columns: tuple[str, ...]) -> tuple[str, ...]:
    allowed = {
        "decision_offset_seconds", "side_is_down", "cluster_distance",
        "extension_from_reference_bp", "reclaim_from_known_extreme_bp", "seconds_since_known_extreme",
    }
    return tuple(name for name in all_columns if name in allowed or name.startswith("cluster_"))


def _classifier(config: AbsorptionModelConfig) -> Pipeline:
    _validate_dependencies()
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                LGBMClassifier(
                    objective="binary",
                    n_estimators=config.model_n_estimators,
                    learning_rate=config.model_learning_rate,
                    num_leaves=config.model_num_leaves,
                    min_child_samples=config.model_min_child_samples,
                    subsample=0.85,
                    colsample_bytree=0.80,
                    reg_alpha=1.0,
                    reg_lambda=4.0,
                    class_weight="balanced",
                    random_state=config.random_state,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    ).set_output(transform="pandas")


def _regressor(config: AbsorptionModelConfig) -> Pipeline:
    _validate_dependencies()
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                LGBMRegressor(
                    objective="huber",
                    n_estimators=config.model_n_estimators,
                    learning_rate=config.model_learning_rate,
                    num_leaves=config.model_num_leaves,
                    min_child_samples=config.model_min_child_samples,
                    subsample=0.85,
                    colsample_bytree=0.80,
                    reg_alpha=1.0,
                    reg_lambda=4.0,
                    random_state=config.random_state,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    ).set_output(transform="pandas")


def _episode_weights(frame: pd.DataFrame) -> np.ndarray:
    counts = frame.groupby("event_id")["event_id"].transform("count").to_numpy(dtype=float)
    weights = 1.0 / np.maximum(counts, 1.0)
    return weights / max(float(np.mean(weights)), 1e-12)


def _validate_binary(y: np.ndarray, config: AbsorptionModelConfig, task: str) -> None:
    if len(y) < config.minimum_train_rows:
        raise RuntimeError(f"R01.3 {task} insufficient train rows: {len(y)}")
    counts = np.bincount(y.astype(np.int8), minlength=2)
    if int(counts.min()) < config.minimum_class_rows:
        raise RuntimeError(f"R01.3 {task} insufficient class rows: {counts.tolist()}")


def fit_models(frame: pd.DataFrame, config: AbsorptionModelConfig) -> FittedModels:
    train = frame.loc[frame["period"].astype(str).eq(config.train_period)].copy()
    x, columns = prepare_design(train)
    trade_y = train["tradeable_before_stop_target"].astype(bool).to_numpy(dtype=np.int8)
    absorption_y = train["absorption_complete_target"].astype(bool).to_numpy(dtype=np.int8)
    _validate_binary(trade_y, config, "tradeable")
    _validate_binary(absorption_y, config, "absorption")
    weights = _episode_weights(train)
    trade_model = _classifier(config)
    absorption_model = _classifier(config)
    extension_model = _regressor(config)
    mfe_model = _regressor(config)
    trade_model.fit(x, trade_y, model__sample_weight=weights)
    absorption_model.fit(x, absorption_y, model__sample_weight=weights)
    extension_model.fit(x, train["future_additional_extension_bp"].to_numpy(dtype=float), model__sample_weight=weights)
    mfe_model.fit(x, train["future_favorable_mfe_bp"].to_numpy(dtype=float), model__sample_weight=weights)
    base_cols = baseline_columns(columns)
    if not base_cols:
        raise RuntimeError("R01.3 baseline feature set is empty")
    baseline = _classifier(config)
    baseline.fit(x.loc[:, base_cols], trade_y, model__sample_weight=weights)
    return FittedModels(columns, trade_model, absorption_model, extension_model, mfe_model, baseline)


def predict(frame: pd.DataFrame, models: FittedModels) -> pd.DataFrame:
    x, _ = prepare_design(frame, models.feature_columns)
    base_cols = baseline_columns(models.feature_columns)
    out = frame.copy()
    out["p_tradeable"] = np.asarray(models.tradeable_model.predict_proba(x)[:, 1], dtype=float)
    out["p_absorption_complete"] = np.asarray(models.absorption_model.predict_proba(x)[:, 1], dtype=float)
    out["pred_additional_extension_bp"] = np.maximum(0.0, np.asarray(models.extension_model.predict(x), dtype=float))
    out["pred_remaining_mfe_bp"] = np.maximum(0.0, np.asarray(models.mfe_model.predict(x), dtype=float))
    out["p_tradeable_baseline"] = np.asarray(models.baseline_model.predict_proba(x.loc[:, base_cols])[:, 1], dtype=float)
    out["trade_score"] = np.sqrt(np.clip(out["p_tradeable"], 0.0, 1.0) * np.clip(out["p_absorption_complete"], 0.0, 1.0))
    out["predicted_net_room_bp"] = out["pred_remaining_mfe_bp"] - out["pred_additional_extension_bp"]
    return out


def _binary_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(p)
    y = y[valid].astype(np.int8)
    p = np.clip(p[valid], 1e-6, 1.0 - 1e-6)
    if len(y) == 0:
        return {"rows": 0, "positive_rate": np.nan, "roc_auc": np.nan, "average_precision": np.nan, "ap_lift": np.nan, "brier": np.nan, "brier_skill": np.nan}
    rate = float(np.mean(y))
    base_brier = float(brier_score_loss(y, np.full(len(y), rate))) if 0 < rate < 1 else np.nan
    brier = float(brier_score_loss(y, p))
    return {
        "rows": int(len(y)),
        "positive_rate": rate,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "average_precision": float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "ap_lift": float(average_precision_score(y, p) / rate) if len(np.unique(y)) == 2 and rate > 0 else np.nan,
        "brier": brier,
        "brier_skill": float(1.0 - brier / base_brier) if np.isfinite(base_brier) and base_brier > 0 else np.nan,
    }


def metric_table(predictions: pd.DataFrame, config: AbsorptionModelConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, subset in predictions.groupby("period", sort=True):
        for side, side_frame in subset.groupby("event_side", sort=True):
            for task, target, score in (
                ("TRADEABLE_FULL", "tradeable_before_stop_target", "p_tradeable"),
                ("TRADEABLE_BASELINE", "tradeable_before_stop_target", "p_tradeable_baseline"),
                ("ABSORPTION_COMPLETE", "absorption_complete_target", "p_absorption_complete"),
            ):
                metrics = _binary_metrics(side_frame[target].astype(bool).to_numpy(dtype=np.int8), side_frame[score].to_numpy(dtype=float))
                rows.append({"period": period, "event_side": side, "task": task, **metrics})
            for task, target, score in (
                ("ADDITIONAL_EXTENSION", "future_additional_extension_bp", "pred_additional_extension_bp"),
                ("REMAINING_MFE", "future_favorable_mfe_bp", "pred_remaining_mfe_bp"),
            ):
                y = side_frame[target].to_numpy(dtype=float)
                p = side_frame[score].to_numpy(dtype=float)
                valid = np.isfinite(y) & np.isfinite(p)
                rho = spearmanr(y[valid], p[valid]).statistic if int(valid.sum()) >= 20 else np.nan
                rows.append({
                    "period": period,
                    "event_side": side,
                    "task": task,
                    "rows": int(valid.sum()),
                    "mae_bp": float(mean_absolute_error(y[valid], p[valid])) if valid.any() else np.nan,
                    "spearman": float(rho) if np.isfinite(rho) else np.nan,
                })
    return pd.DataFrame(rows)


def feature_importance(models: FittedModels) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task, pipeline in (
        ("TRADEABLE", models.tradeable_model),
        ("ABSORPTION", models.absorption_model),
        ("ADDITIONAL_EXTENSION", models.extension_model),
        ("REMAINING_MFE", models.mfe_model),
    ):
        estimator = pipeline.named_steps["model"]
        values = np.asarray(getattr(estimator, "feature_importances_", np.zeros(len(models.feature_columns))), dtype=float)
        # Missing indicators appended by SimpleImputer are omitted from the
        # human-readable feature table; only original causal features are shown.
        values = values[: len(models.feature_columns)]
        denom = max(float(values.sum()), 1e-12)
        for name, value in zip(models.feature_columns, values, strict=True):
            rows.append({"task": task, "feature": name, "importance": float(value), "importance_share": float(value / denom)})
    return pd.DataFrame(rows).sort_values(["task", "importance"], ascending=[True, False]).reset_index(drop=True)
