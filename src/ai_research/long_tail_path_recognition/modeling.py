#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal binary models, OOF thresholds and diagnostics for R03.4.2.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import LongTailPathRecognitionConfig
from .features import FeatureSet, task_target

try:
    from lightgbm import LGBMClassifier
except ImportError:  # pragma: no cover
    LGBMClassifier = None  # type: ignore[assignment]


@dataclass
class BinaryModelBundle:
    model: object
    feature_set: FeatureSet
    task: str

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = frame.loc[:, self.feature_set.columns].to_numpy(dtype=float)
        values = np.asarray(self.model.predict_proba(matrix)[:, 1], dtype=float)
        return np.clip(values, 1e-6, 1.0 - 1e-6)


@dataclass(frozen=True)
class OOFResult:
    probabilities: np.ndarray
    valid_mask: np.ndarray
    folds_used: int


def _validate_binary(y: np.ndarray, config: LongTailPathRecognitionConfig) -> None:
    target = np.asarray(y, dtype=np.int8)
    if len(target) < config.minimum_train_rows:
        raise RuntimeError(f"insufficient train rows: {len(target)}")
    counts = np.bincount(target, minlength=2)
    if int(counts.min()) < config.minimum_class_rows:
        raise RuntimeError(f"insufficient class rows: {counts.tolist()}")


def _make_model(name: str, config: LongTailPathRecognitionConfig):
    if name.endswith("logistic"):
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.5,
                        class_weight="balanced",
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=config.random_state,
                    ),
                ),
            ]
        )
    if name.endswith("lightgbm"):
        if LGBMClassifier is None:
            raise RuntimeError("R03.4.2.2 LightGBM variant requires lightgbm")
        return Pipeline(
            steps=[
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
                        colsample_bytree=0.80,
                        reg_alpha=1.0,
                        reg_lambda=3.0,
                        class_weight="balanced",
                        random_state=config.random_state,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
    raise ValueError(f"unsupported model variant: {name}")


def fit_binary_model(
    frame: pd.DataFrame,
    *,
    target: np.ndarray,
    task: str,
    feature_set: FeatureSet,
    config: LongTailPathRecognitionConfig,
) -> BinaryModelBundle:
    _validate_binary(target, config)
    matrix = frame.loc[:, feature_set.columns].to_numpy(dtype=float)
    model = _make_model(feature_set.name, config)
    model.fit(matrix, target)
    return BinaryModelBundle(model=model, feature_set=feature_set, task=task)


def causal_oof_probabilities(
    frame: pd.DataFrame,
    *,
    target: np.ndarray,
    task: str,
    feature_set: FeatureSet,
    config: LongTailPathRecognitionConfig,
) -> OOFResult:
    """Expanding-window OOF predictions with a 48h embargo.

    Rows before the first validation block intentionally remain NaN. OOF
    predictions are used only to freeze probability thresholds, never to report
    in-sample model quality.
    """

    order = np.argsort(pd.to_datetime(frame["decision_time"]).to_numpy(dtype="datetime64[ns]"))
    ordered = frame.iloc[order].reset_index(drop=True)
    y = np.asarray(target, dtype=np.int8)[order]
    times = pd.to_datetime(ordered["decision_time"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    n = len(ordered)
    probabilities = np.full(n, np.nan, dtype=float)
    blocks = np.array_split(np.arange(n), config.oof_splits + 1)
    folds_used = 0
    embargo_ns = int(pd.Timedelta(hours=config.oof_embargo_hours).value)

    for block in blocks[1:]:
        if not len(block):
            continue
        validation_start_time = int(times[int(block[0])])
        train_positions = np.flatnonzero(times < validation_start_time - embargo_ns)
        if len(train_positions) < config.minimum_train_rows:
            continue
        train_target = y[train_positions]
        counts = np.bincount(train_target, minlength=2)
        if int(counts.min()) < config.minimum_class_rows:
            continue
        model = fit_binary_model(
            ordered.iloc[train_positions],
            target=train_target,
            task=task,
            feature_set=feature_set,
            config=config,
        )
        probabilities[block] = model.predict(ordered.iloc[block])
        folds_used += 1

    restored = np.full(n, np.nan, dtype=float)
    restored[order] = probabilities
    return OOFResult(probabilities=restored, valid_mask=np.isfinite(restored), folds_used=folds_used)


def probability_threshold(
    probabilities: np.ndarray,
    *,
    quantile: float,
) -> float:
    values = np.asarray(probabilities, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 30:
        raise RuntimeError("insufficient causal OOF probabilities for threshold")
    return float(np.quantile(values, quantile))


def classification_metrics(
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
    y = y[valid]
    p = p[valid]
    if len(y) == 0 or len(np.unique(y)) < 2:
        raise RuntimeError("classification metric requires both classes")
    base_rate = float(y.mean())
    auc = float(roc_auc_score(y, p))
    ap = float(average_precision_score(y, p))
    brier = float(brier_score_loss(y, p))
    base_brier = float(brier_score_loss(y, np.full(len(y), base_rate)))
    top_cut = float(np.quantile(p, 0.90))
    top = y[p >= top_cut]
    top_rate = float(top.mean()) if len(top) else np.nan
    return {
        "fold_id": fold_id,
        "task": task,
        "checkpoint_minutes": int(checkpoint_minutes),
        "feature_set": feature_set,
        "scope": scope,
        "rows": int(len(y)),
        "positive_rows": int(y.sum()),
        "positive_rate": base_rate,
        "roc_auc": auc,
        "average_precision": ap,
        "average_precision_lift": float(ap / base_rate) if base_rate > 0 else np.nan,
        "brier_score": brier,
        "brier_skill": float(1.0 - brier / base_brier) if base_brier > 0 else np.nan,
        "top_probability_decile_positive_rate": top_rate,
        "top_probability_decile_lift": float(top_rate / base_rate) if base_rate > 0 and np.isfinite(top_rate) else np.nan,
    }


def prediction_deciles(
    frame: pd.DataFrame,
    *,
    target: np.ndarray,
    probability: np.ndarray,
    fold_id: str,
    task: str,
    checkpoint_minutes: int,
    feature_set: str,
    scope: str,
) -> pd.DataFrame:
    work = frame.copy()
    work["target"] = np.asarray(target, dtype=np.int8)
    work["probability"] = np.asarray(probability, dtype=float)
    work = work.loc[np.isfinite(work["probability"])].copy()
    if work.empty:
        return pd.DataFrame()
    try:
        work["probability_decile"] = pd.qcut(work["probability"], q=10, labels=False, duplicates="drop")
    except ValueError:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for decile, group in work.groupby("probability_decile", sort=True):
        rows.append(
            {
                "fold_id": fold_id,
                "task": task,
                "checkpoint_minutes": int(checkpoint_minutes),
                "feature_set": feature_set,
                "scope": scope,
                "probability_decile": int(decile) + 1,
                "rows": int(len(group)),
                "mean_probability": float(group["probability"].mean()),
                "positive_rate": float(group["target"].mean()),
                "mean_fixed6h_net_1x": float(group["fixed6h_net_1x"].mean()),
                "mean_net_24h_1x": float(group["net_24h_1x"].mean()),
                "mean_net_48h_1x": float(group["net_48h_1x"].mean()),
                "mean_checkpoint_net_exit_1x": float(group["checkpoint_net_exit_1x"].mean()),
            }
        )
    return pd.DataFrame(rows)


def feature_importance(bundle: BinaryModelBundle) -> pd.DataFrame:
    model = bundle.model
    if isinstance(model, Pipeline):
        estimator = model.named_steps["model"]
    else:
        estimator = model
    if hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_[0], dtype=float))
        signed = np.asarray(estimator.coef_[0], dtype=float)
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


def stable_signal_candidates(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    primary = metrics.loc[metrics["scope"] == "primary_q90"]
    for keys, group in primary.groupby(["task", "checkpoint_minutes", "feature_set"], sort=False):
        task, checkpoint, feature_set = keys
        by_fold = group.set_index("fold_id")
        if not {"WF_2024", "WF_2025"}.issubset(by_fold.index):
            continue
        aucs = [float(by_fold.loc[fold, "roc_auc"]) for fold in ("WF_2024", "WF_2025")]
        lifts = [float(by_fold.loc[fold, "top_probability_decile_lift"]) for fold in ("WF_2024", "WF_2025")]
        ap_lifts = [float(by_fold.loc[fold, "average_precision_lift"]) for fold in ("WF_2024", "WF_2025")]
        if task in {"persistent_failure", "recovery_from_underwater"}:
            pass_gate = min(aucs) >= 0.65 and min(lifts) >= 1.40 and min(ap_lifts) >= 1.15
        else:
            pass_gate = min(aucs) >= 0.60 and min(lifts) >= 1.25 and min(ap_lifts) >= 1.10
        rows.append(
            {
                "task": task,
                "checkpoint_minutes": int(checkpoint),
                "feature_set": feature_set,
                "auc_2024": aucs[0],
                "auc_2025": aucs[1],
                "minimum_auc": min(aucs),
                "minimum_top_decile_lift": min(lifts),
                "minimum_ap_lift": min(ap_lifts),
                "stable_signal": bool(pass_gate),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["stable_signal", "minimum_auc", "minimum_top_decile_lift"],
        ascending=[False, False, False],
    ) if rows else pd.DataFrame()
