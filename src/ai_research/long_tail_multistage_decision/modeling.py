#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Holding-path classifiers, causal OOF thresholds and model selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import LongTailMultistageConfig
from .features import FeatureSet

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None  # type: ignore[assignment]


@dataclass
class BinaryModel:
    task: str
    feature_set: FeatureSet
    model: object

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame.loc[:, self.feature_set.columns].to_numpy(dtype=float)
        return np.clip(np.asarray(self.model.predict_proba(matrix)[:, 1], dtype=float), 1e-6, 1.0 - 1e-6)


@dataclass(frozen=True)
class OOFResult:
    probabilities: np.ndarray
    folds_used: int


def _model(feature_set: FeatureSet, config: LongTailMultistageConfig):
    if feature_set.name.endswith("logistic"):
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.35,
                        class_weight="balanced",
                        max_iter=2500,
                        solver="lbfgs",
                        random_state=config.random_state,
                    ),
                ),
            ]
        )
    if feature_set.name.endswith("lightgbm"):
        if LGBMClassifier is None:
            raise RuntimeError("R03.4.2.3 LightGBM models require lightgbm")
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    LGBMClassifier(
                        objective="binary",
                        n_estimators=config.classifier_n_estimators,
                        learning_rate=config.classifier_learning_rate,
                        num_leaves=config.classifier_num_leaves,
                        min_child_samples=config.classifier_min_child_samples,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_alpha=1.0,
                        reg_lambda=4.0,
                        class_weight="balanced",
                        random_state=config.random_state,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    raise ValueError(feature_set.name)


def _validate_target(target: np.ndarray, config: LongTailMultistageConfig) -> None:
    y = np.asarray(target, dtype=np.int8)
    if len(y) < config.minimum_train_rows:
        raise RuntimeError(f"insufficient train rows {len(y)}")
    counts = np.bincount(y, minlength=2)
    if int(counts.min()) < config.minimum_class_rows:
        raise RuntimeError(f"insufficient class rows {counts.tolist()}")


def fit_model(
    frame: pd.DataFrame,
    target: np.ndarray,
    *,
    task: str,
    feature_set: FeatureSet,
    config: LongTailMultistageConfig,
) -> BinaryModel:
    _validate_target(target, config)
    matrix = frame.loc[:, feature_set.columns].to_numpy(dtype=float)
    model = _model(feature_set, config)
    model.fit(matrix, np.asarray(target, dtype=np.int8))
    return BinaryModel(task=task, feature_set=feature_set, model=model)


def causal_oof(
    frame: pd.DataFrame,
    target: np.ndarray,
    *,
    task: str,
    feature_set: FeatureSet,
    config: LongTailMultistageConfig,
) -> OOFResult:
    order = np.argsort(pd.to_datetime(frame["decision_time"]).to_numpy(dtype="datetime64[ns]"))
    ordered = frame.iloc[order].reset_index(drop=True)
    y = np.asarray(target, dtype=np.int8)[order]
    times = pd.to_datetime(ordered["decision_time"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    probabilities = np.full(len(ordered), np.nan, dtype=float)
    blocks = [block for block in np.array_split(np.arange(len(ordered)), config.holding_oof_splits + 1) if len(block)]
    embargo_ns = int(pd.Timedelta(hours=config.holding_oof_embargo_hours).value)
    folds_used = 0
    for block in blocks[1:]:
        validation_start = int(times[int(block[0])])
        train_pos = np.flatnonzero(times < validation_start - embargo_ns)
        if len(train_pos) < config.minimum_train_rows:
            continue
        counts = np.bincount(y[train_pos], minlength=2)
        if int(counts.min()) < config.minimum_class_rows:
            continue
        model = fit_model(
            ordered.iloc[train_pos],
            y[train_pos],
            task=task,
            feature_set=feature_set,
            config=config,
        )
        probabilities[block] = model.predict(ordered.iloc[block])
        folds_used += 1
    restored = np.full(len(frame), np.nan, dtype=float)
    restored[order] = probabilities
    return OOFResult(probabilities=restored, folds_used=folds_used)


def metric_row(
    target: np.ndarray,
    probability: np.ndarray,
    *,
    fold_id: str,
    task: str,
    checkpoint_minutes: int,
    feature_set: str,
    scope: str,
) -> dict[str, object]:
    y = np.asarray(target, dtype=np.int8)
    p = np.asarray(probability, dtype=float)
    valid = np.isfinite(p)
    y, p = y[valid], p[valid]
    if len(y) < 10 or len(np.unique(y)) < 2:
        raise RuntimeError("metric needs both classes")
    base = float(y.mean())
    auc = float(roc_auc_score(y, p))
    ap = float(average_precision_score(y, p))
    brier = float(brier_score_loss(y, p))
    base_brier = float(brier_score_loss(y, np.full(len(y), base)))
    cut = float(np.quantile(p, 0.90))
    top = y[p >= cut]
    top_rate = float(top.mean()) if len(top) else np.nan
    return {
        "fold_id": fold_id,
        "task": task,
        "checkpoint_minutes": int(checkpoint_minutes),
        "feature_set": feature_set,
        "scope": scope,
        "rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "positive_rate": base,
        "roc_auc": auc,
        "average_precision": ap,
        "average_precision_lift": float(ap / base) if base > 0 else np.nan,
        "brier_score": brier,
        "brier_skill": float(1.0 - brier / base_brier) if base_brier > 0 else np.nan,
        "top_decile_positive_rate": top_rate,
        "top_decile_lift": float(top_rate / base) if base > 0 and np.isfinite(top_rate) else np.nan,
    }


def oof_metrics(target: np.ndarray, probability: np.ndarray) -> tuple[float, float, float]:
    y = np.asarray(target, dtype=np.int8)
    p = np.asarray(probability, dtype=float)
    valid = np.isfinite(p)
    y, p = y[valid], p[valid]
    if len(y) < 30 or len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    base = float(y.mean())
    auc = float(roc_auc_score(y, p))
    ap_lift = float(average_precision_score(y, p) / base) if base > 0 else np.nan
    brier = float(brier_score_loss(y, p))
    base_brier = float(brier_score_loss(y, np.full(len(y), base)))
    brier_skill = float(1.0 - brier / base_brier) if base_brier > 0 else np.nan
    return auc, ap_lift, brier_skill


def choose_feature_set(
    candidates: list[tuple[FeatureSet, OOFResult]],
    target: np.ndarray,
) -> tuple[FeatureSet, OOFResult, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for feature_set, result in candidates:
        auc, ap_lift, brier_skill = oof_metrics(target, result.probabilities)
        rows.append(
            {
                "feature_set": feature_set.name,
                "oof_auc": auc,
                "oof_ap_lift": ap_lift,
                "oof_brier_skill": brier_skill,
                "oof_rows": int(np.isfinite(result.probabilities).sum()),
                "oof_folds": result.folds_used,
                "uses_entry_score": bool(any(column.startswith("x_score__") for column in feature_set.columns)),
                "complexity_rank": 0 if feature_set.name == "mechanical_logistic" else 1 if feature_set.name == "path_structure_logistic" else 2 if feature_set.name == "path_structure_lightgbm" else 3,
            }
        )
    audit = pd.DataFrame(rows)
    valid = audit.loc[
        np.isfinite(audit["oof_auc"])
        & (audit["oof_folds"] >= 2)
        & (audit["oof_rows"] >= 30)
    ].copy()
    if valid.empty:
        raise RuntimeError("no candidate holding model produced causal OOF predictions")
    # Prefer materially better OOF AUC, then simpler/path-only models on near ties.
    best_auc = float(valid["oof_auc"].max())
    shortlist = valid.loc[valid["oof_auc"] >= best_auc - 0.01].sort_values(
        ["uses_entry_score", "complexity_rank", "oof_brier_skill"],
        ascending=[True, True, False],
    )
    chosen_name = str(shortlist.iloc[0]["feature_set"])
    audit["selected"] = audit["feature_set"] == chosen_name
    for feature_set, result in candidates:
        if feature_set.name == chosen_name:
            return feature_set, result, audit
    raise AssertionError(chosen_name)


def threshold(probabilities: np.ndarray, quantile: float) -> float:
    values = np.asarray(probabilities, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 30:
        raise RuntimeError("insufficient causal OOF probabilities for threshold")
    return float(np.quantile(values, quantile))


def feature_importance(bundle: BinaryModel) -> pd.DataFrame:
    estimator = bundle.model.named_steps["model"] if isinstance(bundle.model, Pipeline) else bundle.model
    if hasattr(estimator, "coef_"):
        signed = np.asarray(estimator.coef_[0], dtype=float)
        values = np.abs(signed)
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
        signed = values.copy()
    else:
        return pd.DataFrame()
    order = np.argsort(values)[::-1]
    return pd.DataFrame(
        {
            "task": bundle.task,
            "feature_set": bundle.feature_set.name,
            "feature": np.asarray(bundle.feature_set.columns, dtype=object)[order],
            "importance": values[order],
            "signed_effect": signed[order],
            "rank": np.arange(1, len(order) + 1),
        }
    )
