#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression models and causal validation for incremental holding value."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ai_research.long_tail_multistage_decision.features import FeatureSet

from .config import IncrementalHoldConfig
from .features import assert_no_future_features

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None  # type: ignore[assignment]


@dataclass
class FittedHoldModel:
    name: str
    columns: tuple[str, ...]
    model: object

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame.loc[:, self.columns].to_numpy(dtype=float)
        return np.asarray(self.model.predict(x), dtype=float)


@dataclass(frozen=True)
class OOFRegressionResult:
    predictions: np.ndarray
    folds_used: int
    fold_audit: pd.DataFrame


def _rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(values, dtype=float)).rank(method="average").to_numpy(dtype=float)


def rank_ic(actual: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(prediction, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    if valid.sum() < 3:
        return np.nan
    yr = _rank(y[valid])
    pr = _rank(p[valid])
    if np.std(yr) <= 1e-12 or np.std(pr) <= 1e-12:
        return np.nan
    return float(np.corrcoef(yr, pr)[0, 1])


def _make_model(feature_set: FeatureSet, config: IncrementalHoldConfig) -> object:
    assert_no_future_features(feature_set.columns)
    if feature_set.name.endswith("ridge"):
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        )
    if "lightgbm" in feature_set.name:
        if LGBMRegressor is None:
            raise RuntimeError("LightGBM is required for R03.4.2.6")
        return LGBMRegressor(
            objective="regression_l1",
            n_estimators=config.regression_n_estimators,
            learning_rate=config.regression_learning_rate,
            num_leaves=config.regression_num_leaves,
            min_child_samples=config.regression_min_child_samples,
            colsample_bytree=0.80,
            subsample=0.85,
            subsample_freq=1,
            reg_lambda=2.0,
            random_state=config.random_state,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"unknown feature set {feature_set.name}")


def fit_model(
    frame: pd.DataFrame,
    target: np.ndarray,
    *,
    feature_set: FeatureSet,
    config: IncrementalHoldConfig,
) -> FittedHoldModel:
    if len(frame) != len(target):
        raise ValueError("target length mismatch")
    if not feature_set.columns:
        raise ValueError("empty feature set")
    y = np.asarray(target, dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < config.minimum_train_rows:
        raise RuntimeError(f"insufficient regression rows={int(valid.sum())}")
    x = frame.loc[valid, feature_set.columns].to_numpy(dtype=float)
    model = _make_model(feature_set, config)
    model.fit(x, y[valid])
    return FittedHoldModel(feature_set.name, feature_set.columns, model)


def _time_splits(frame: pd.DataFrame, config: IncrementalHoldConfig) -> list[tuple[np.ndarray, np.ndarray]]:
    times = pd.to_datetime(frame["checkpoint_time"]).to_numpy(dtype="datetime64[ns]")
    order = np.argsort(times, kind="stable")
    blocks = [chunk for chunk in np.array_split(order, config.holding_oof_splits + 1) if len(chunk)]
    embargo = np.timedelta64(config.holding_oof_embargo_hours, "h")
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for index in range(1, len(blocks)):
        test = np.asarray(blocks[index], dtype=np.int64)
        test_start = times[test].min()
        train = np.concatenate(blocks[:index])
        train = train[times[train] <= test_start - embargo]
        if len(train) < config.minimum_train_rows or len(test) < max(15, config.minimum_test_rows // 2):
            continue
        splits.append((np.asarray(train, dtype=np.int64), test))
    return splits


def causal_oof(
    frame: pd.DataFrame,
    target: np.ndarray,
    *,
    feature_set: FeatureSet,
    config: IncrementalHoldConfig,
) -> OOFRegressionResult:
    prediction = np.full(len(frame), np.nan, dtype=float)
    audits: list[dict[str, object]] = []
    splits = _time_splits(frame, config)
    for fold_id, (train, test) in enumerate(splits, start=1):
        model = fit_model(frame.iloc[train], np.asarray(target)[train], feature_set=feature_set, config=config)
        prediction[test] = model.predict(frame.iloc[test])
        audits.append(
            {
                "oof_fold": fold_id,
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_end": str(pd.to_datetime(frame.iloc[train]["checkpoint_time"]).max()),
                "test_start": str(pd.to_datetime(frame.iloc[test]["checkpoint_time"]).min()),
                "embargo_hours": config.holding_oof_embargo_hours,
            }
        )
    if len(audits) < config.minimum_oof_folds:
        raise RuntimeError(f"insufficient causal OOF folds={len(audits)}")
    return OOFRegressionResult(prediction, len(audits), pd.DataFrame(audits))


def _bucket_ids(prediction: np.ndarray, buckets: int) -> np.ndarray:
    p = np.asarray(prediction, dtype=float)
    order = np.argsort(p, kind="stable")
    ids = np.empty(len(p), dtype=np.int16)
    ids[order] = np.minimum((np.arange(len(p)) * buckets) // max(len(p), 1), buckets - 1)
    return ids


def evaluate(
    actual: np.ndarray,
    prediction: np.ndarray,
    *,
    fold_id: str,
    checkpoint_minutes: int,
    target_name: str,
    feature_set: str,
    scope: str,
    config: IncrementalHoldConfig,
) -> tuple[dict[str, object], pd.DataFrame]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(prediction, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]
    if len(y) < config.minimum_test_rows:
        raise RuntimeError(f"insufficient evaluation rows={len(y)}")
    baseline = np.full(len(y), float(np.median(y)), dtype=float)
    ids = _bucket_ids(p, config.rank_bucket_count)
    bucket_rows: list[dict[str, object]] = []
    bucket_means: list[float] = []
    for bucket in range(config.rank_bucket_count):
        mask = ids == bucket
        values = y[mask]
        bucket_mean = float(np.mean(values)) if len(values) else np.nan
        bucket_means.append(bucket_mean)
        bucket_rows.append(
            {
                "fold_id": fold_id,
                "checkpoint_minutes": checkpoint_minutes,
                "target": target_name,
                "feature_set": feature_set,
                "scope": scope,
                "decile": bucket + 1,
                "rows": int(mask.sum()),
                "prediction_mean": float(np.mean(p[mask])) if mask.any() else np.nan,
                "actual_mean": bucket_mean,
                "actual_median": float(np.median(values)) if len(values) else np.nan,
                "positive_utility_rate": float(np.mean(values > config.positive_utility_buffer)) if len(values) else np.nan,
            }
        )
    decile_monotonicity = rank_ic(np.arange(1, config.rank_bucket_count + 1), np.asarray(bucket_means))
    sign = y > config.positive_utility_buffer
    sign_prediction = p > 0
    sign_accuracy = float(np.mean(sign_prediction == sign))
    auc = np.nan
    if len(np.unique(sign)) == 2:
        auc = float(roc_auc_score(sign.astype(int), p))
    top = y[ids >= config.rank_bucket_count - 2]
    bottom = y[ids <= 1]
    metrics = {
        "fold_id": fold_id,
        "checkpoint_minutes": int(checkpoint_minutes),
        "target": target_name,
        "feature_set": feature_set,
        "scope": scope,
        "rows": int(len(y)),
        "rank_ic": rank_ic(y, p),
        "mae": float(mean_absolute_error(y, p)),
        "baseline_mae": float(mean_absolute_error(y, baseline)),
        "mae_skill": float(1.0 - mean_absolute_error(y, p) / max(mean_absolute_error(y, baseline), 1e-12)),
        "sign_auc": auc,
        "sign_accuracy": sign_accuracy,
        "actual_mean": float(np.mean(y)),
        "actual_positive_rate": float(np.mean(sign)),
        "top_quintile_actual_mean": float(np.mean(top)) if len(top) else np.nan,
        "bottom_quintile_actual_mean": float(np.mean(bottom)) if len(bottom) else np.nan,
        "top_bottom_spread": float(np.mean(top) - np.mean(bottom)) if len(top) and len(bottom) else np.nan,
        "decile_monotonicity": decile_monotonicity,
    }
    return metrics, pd.DataFrame(bucket_rows)


def oof_score(actual: np.ndarray, oof_prediction: np.ndarray) -> float:
    valid = np.isfinite(actual) & np.isfinite(oof_prediction)
    if valid.sum() < 30:
        return -np.inf
    ic = rank_ic(np.asarray(actual)[valid], np.asarray(oof_prediction)[valid])
    mae = mean_absolute_error(np.asarray(actual)[valid], np.asarray(oof_prediction)[valid])
    baseline = mean_absolute_error(np.asarray(actual)[valid], np.full(valid.sum(), np.median(np.asarray(actual)[valid])))
    skill = 1.0 - mae / max(baseline, 1e-12)
    return float((ic if np.isfinite(ic) else -1.0) + 0.20 * skill)


def choose_feature_set(
    candidates: list[tuple[FeatureSet, OOFRegressionResult]],
    target: np.ndarray,
) -> tuple[FeatureSet, OOFRegressionResult, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    complexity = {
        "mechanical_ridge": 0,
        "path_structure_ridge": 1,
        "path_structure_lightgbm": 2,
        "score_only_lightgbm": 3,
        "path_plus_score_lightgbm": 4,
    }
    scored: list[tuple[float, int, FeatureSet, OOFRegressionResult]] = []
    for feature_set, result in candidates:
        valid = np.isfinite(result.predictions) & np.isfinite(target)
        score = oof_score(np.asarray(target)[valid], result.predictions[valid])
        ic = rank_ic(np.asarray(target)[valid], result.predictions[valid])
        rows.append(
            {
                "feature_set": feature_set.name,
                "oof_rows": int(valid.sum()),
                "oof_folds": result.folds_used,
                "oof_rank_ic": ic,
                "selection_score": score,
            }
        )
        scored.append((score, -complexity.get(feature_set.name, 99), feature_set, result))
    if not scored:
        raise RuntimeError("no OOF candidates")
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, selected, selected_result = scored[0]
    audit = pd.DataFrame(rows)
    audit["selected"] = audit["feature_set"] == selected.name
    return selected, selected_result, audit


def feature_importance(model: FittedHoldModel) -> pd.DataFrame:
    raw = model.model
    if isinstance(raw, Pipeline):
        estimator = raw.named_steps.get("model")
        if estimator is not None and hasattr(estimator, "coef_"):
            values = np.abs(np.asarray(estimator.coef_, dtype=float)).reshape(-1)
            return pd.DataFrame({"feature": model.columns, "importance": values}).sort_values("importance", ascending=False)
        return pd.DataFrame()
    if hasattr(raw, "feature_importances_"):
        values = np.asarray(raw.feature_importances_, dtype=float)
        return pd.DataFrame({"feature": model.columns, "importance": values}).sort_values("importance", ascending=False)
    return pd.DataFrame()


def stable_candidates(metrics: pd.DataFrame, config: IncrementalHoldConfig) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    keys = ["checkpoint_minutes", "target", "feature_set", "scope"]
    rows: list[dict[str, object]] = []
    for key, group in metrics.groupby(keys, sort=False):
        checkpoint, target, feature_set, scope = key
        row: dict[str, object] = {
            "checkpoint_minutes": int(checkpoint),
            "target": target,
            "feature_set": feature_set,
            "scope": scope,
        }
        passed = True
        for fold in ("WF_2024", "WF_2025"):
            part = group.loc[group["fold_id"] == fold]
            if part.empty:
                passed = False
                continue
            item = part.iloc[0]
            for column in (
                "rows",
                "rank_ic",
                "mae_skill",
                "sign_auc",
                "sign_accuracy",
                "top_quintile_actual_mean",
                "bottom_quintile_actual_mean",
                "top_bottom_spread",
                "decile_monotonicity",
            ):
                row[f"{fold}_{column}"] = item[column]
            fold_pass = bool(
                int(item["rows"]) >= config.minimum_test_rows
                and float(item["rank_ic"]) >= config.minimum_rank_ic
                and float(item["top_bottom_spread"]) >= config.minimum_top_bottom_spread
                and float(item["decile_monotonicity"]) >= config.minimum_decile_monotonicity
                and float(item["sign_accuracy"]) >= config.minimum_sign_accuracy
                and float(item["top_quintile_actual_mean"]) > 0
            )
            row[f"{fold}_passes"] = fold_pass
            passed = passed and fold_pass
        row["passes_cross_year"] = bool(passed)
        rank_values = [float(row.get(f"{fold}_rank_ic", np.nan)) for fold in ("WF_2024", "WF_2025")]
        spreads = [float(row.get(f"{fold}_top_bottom_spread", np.nan)) for fold in ("WF_2024", "WF_2025")]
        row["stability_score"] = float(np.nanmin(rank_values) + 5.0 * np.nanmin(spreads))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["passes_cross_year", "stability_score"], ascending=[False, False], kind="stable"
    ).reset_index(drop=True)
