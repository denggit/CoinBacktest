#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Models for R02.1 pool strength, independent from arrival probability."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None

from src.ai_research.latent_liquidity_pool_forecast.modeling import feature_columns as r02_feature_columns
from .config import LatentLiquidityPoolStrengthConfig

_LABEL_EXACT = {
    "high_strength_label", "release_episode_count", "release_density_sum", "release_density_max",
    "release_episode_size_sum", "release_score_sum", "favorable_episode_count",
    "continuation_episode_count", "favorable_density_sum", "continuation_density_sum",
    "sweep_depth_weighted_bp", "reversal_room_weighted_bp", "first_release_minutes",
    "release_density_log", "release_count_log", "release_size_log", "release_peak_log",
}
_PRED_PREFIX = ("p_strength_", "pred_density_", "p_favorable_", "p_continuation_", "pred_sweep_")


def feature_columns(frame: pd.DataFrame, *, include_swing: bool) -> tuple[str, ...]:
    cols = []
    for name in r02_feature_columns(frame, include_swing=include_swing):
        if name in _LABEL_EXACT or name.startswith(_PRED_PREFIX) or name.startswith("high_strength"):
            continue
        cols.append(name)
    return tuple(cols)


def baseline_columns(path_columns: tuple[str, ...]) -> tuple[str, ...]:
    # Strict arrival-independent spatial baseline: distance + side only.
    return tuple(name for name in path_columns if name in {"zone_distance_bp", "side_is_down"})


def _classifier(config: LatentLiquidityPoolStrengthConfig) -> Pipeline:
    if LGBMClassifier is None:
        raise RuntimeError("lightgbm is required for R02.1")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", LGBMClassifier(
            objective="binary", n_estimators=config.model_n_estimators,
            learning_rate=config.model_learning_rate, num_leaves=config.model_num_leaves,
            min_child_samples=config.model_min_child_samples, subsample=0.85,
            colsample_bytree=0.80, reg_alpha=1.0, reg_lambda=4.0,
            class_weight="balanced", random_state=config.random_state,
            n_jobs=-1, verbosity=-1,
        )),
    ]).set_output(transform="pandas")


def _regressor(config: LatentLiquidityPoolStrengthConfig) -> Pipeline:
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is required for R02.1")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", LGBMRegressor(
            objective="huber", n_estimators=config.model_n_estimators,
            learning_rate=config.model_learning_rate, num_leaves=config.model_num_leaves,
            min_child_samples=config.model_min_child_samples, subsample=0.85,
            colsample_bytree=0.80, reg_alpha=1.0, reg_lambda=4.0,
            random_state=config.random_state, n_jobs=-1, verbosity=-1,
        )),
    ]).set_output(transform="pandas")


def _cap(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame
    h = pd.util.hash_pandas_object(frame["zone_id"].astype(str), index=False).to_numpy(dtype=np.uint64)
    idx = np.argpartition(h, cap - 1)[:cap]
    return frame.iloc[np.sort(idx)].copy()


def _check_binary(frame: pd.DataFrame, target: str, config: LatentLiquidityPoolStrengthConfig) -> None:
    y = frame[target].astype(bool).to_numpy(dtype=np.int8)
    if len(y) < config.minimum_train_rows:
        raise RuntimeError(f"R02.1 {target} insufficient rows: {len(y)}")
    counts = np.bincount(y, minlength=2)
    if int(counts.min()) < config.minimum_class_rows:
        raise RuntimeError(f"R02.1 {target} insufficient class rows: {counts.tolist()}")


@dataclass
class StrengthModelBundle:
    full_columns: tuple[str, ...]
    path_columns: tuple[str, ...]
    baseline_columns: tuple[str, ...]
    strength_full: Pipeline
    strength_path: Pipeline
    strength_baseline: Pipeline
    density_full: Pipeline
    density_path: Pipeline
    density_baseline: Pipeline
    favorable_path: Pipeline
    favorable_full: Pipeline
    continuation_path: Pipeline
    continuation_full: Pipeline
    sweep_depth_path: Pipeline


def fit_models(frame: pd.DataFrame, config: LatentLiquidityPoolStrengthConfig) -> StrengthModelBundle:
    model_rows = frame.loc[frame.get("model_sample_keep", True).astype(bool)] if "model_sample_keep" in frame else frame
    train = model_rows.loc[
        model_rows["period"].astype(str).eq(config.train_period)
        & model_rows["touch_720m"].astype(bool)
    ].copy()
    train = _cap(train, config.model_train_cap_rows)
    full = feature_columns(train, include_swing=True)
    path = feature_columns(train, include_swing=False)
    base = baseline_columns(path)
    if not full or not path or not base:
        raise RuntimeError("R02.1 feature schema is empty")
    weights = pd.to_numeric(train.get("sample_weight", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    _check_binary(train, "high_strength_label", config)

    strength_full = _classifier(config); strength_full.fit(train.loc[:, full], train["high_strength_label"].astype(bool), model__sample_weight=weights)
    strength_path = _classifier(config); strength_path.fit(train.loc[:, path], train["high_strength_label"].astype(bool), model__sample_weight=weights)
    strength_base = _classifier(config); strength_base.fit(train.loc[:, base], train["high_strength_label"].astype(bool), model__sample_weight=weights)

    density_full = _regressor(config); density_full.fit(train.loc[:, full], train["release_density_log"].astype(float), model__sample_weight=weights)
    density_path = _regressor(config); density_path.fit(train.loc[:, path], train["release_density_log"].astype(float), model__sample_weight=weights)
    density_base = _regressor(config); density_base.fit(train.loc[:, base], train["release_density_log"].astype(float), model__sample_weight=weights)

    released = train.loc[train["release_episode_count"].gt(0)].copy()
    wr = pd.to_numeric(released.get("sample_weight", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=float)
    released["favorable_any"] = released["favorable_episode_count"].gt(0)
    released["continuation_any"] = released["continuation_episode_count"].gt(0)
    _check_binary(released, "favorable_any", config)
    _check_binary(released, "continuation_any", config)
    fav_path = _classifier(config); fav_path.fit(released.loc[:, path], released["favorable_any"], model__sample_weight=wr)
    fav_full = _classifier(config); fav_full.fit(released.loc[:, full], released["favorable_any"], model__sample_weight=wr)
    cont_path = _classifier(config); cont_path.fit(released.loc[:, path], released["continuation_any"], model__sample_weight=wr)
    cont_full = _classifier(config); cont_full.fit(released.loc[:, full], released["continuation_any"], model__sample_weight=wr)
    sweep = _regressor(config); sweep.fit(released.loc[:, path], released["sweep_depth_weighted_bp"].astype(float), model__sample_weight=wr)

    return StrengthModelBundle(full, path, base, strength_full, strength_path, strength_base, density_full, density_path, density_base, fav_path, fav_full, cont_path, cont_full, sweep)


def predict(frame: pd.DataFrame, models: StrengthModelBundle) -> pd.DataFrame:
    out = frame.copy()
    xf = out.loc[:, models.full_columns]
    xp = out.loc[:, models.path_columns]
    xb = out.loc[:, models.baseline_columns]
    out["p_strength_full"] = models.strength_full.predict_proba(xf)[:, 1]
    out["p_strength_path"] = models.strength_path.predict_proba(xp)[:, 1]
    out["p_strength_baseline"] = models.strength_baseline.predict_proba(xb)[:, 1]
    out["pred_density_full"] = np.maximum(0.0, models.density_full.predict(xf))
    out["pred_density_path"] = np.maximum(0.0, models.density_path.predict(xp))
    out["pred_density_baseline"] = np.maximum(0.0, models.density_baseline.predict(xb))
    out["p_favorable_path"] = models.favorable_path.predict_proba(xp)[:, 1]
    out["p_favorable_full"] = models.favorable_full.predict_proba(xf)[:, 1]
    out["p_continuation_path"] = models.continuation_path.predict_proba(xp)[:, 1]
    out["p_continuation_full"] = models.continuation_full.predict_proba(xf)[:, 1]
    out["pred_sweep_depth_path_bp"] = np.maximum(0.0, models.sweep_depth_path.predict(xp))
    # PRIMARY latent-pool score deliberately excludes both Touch/Arrival and Swing.
    out["pool_strength_score"] = out["p_strength_path"]
    return out


def _binary_metrics(y: np.ndarray, p: np.ndarray, w: np.ndarray | None) -> dict[str, float | int]:
    valid = np.isfinite(p)
    y = y[valid].astype(np.int8); p = p[valid]
    ww = w[valid] if w is not None else None
    if not len(y):
        return {"rows": 0, "positive_rate": np.nan, "roc_auc": np.nan, "average_precision": np.nan}
    return {
        "rows": int(len(y)),
        "positive_rate": float(np.average(y, weights=ww)),
        "roc_auc": float(roc_auc_score(y, p, sample_weight=ww)) if len(np.unique(y)) == 2 else np.nan,
        "average_precision": float(average_precision_score(y, p, sample_weight=ww)) if len(np.unique(y)) == 2 else np.nan,
    }


def _reg_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    valid = np.isfinite(y) & np.isfinite(p)
    yv, pv = y[valid], p[valid]
    if len(yv) < 20:
        return {"rows": int(len(yv)), "mae": np.nan, "spearman": np.nan}
    rho = spearmanr(yv, pv).statistic if np.nanstd(yv) > 1e-12 and np.nanstd(pv) > 1e-12 else np.nan
    return {"rows": int(len(yv)), "mae": float(mean_absolute_error(yv, pv)), "spearman": float(rho) if np.isfinite(rho) else np.nan}


def metric_table(pred: pd.DataFrame, config: LatentLiquidityPoolStrengthConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for period, pf in pred.groupby("period", sort=True):
        for side, sf0 in pf.groupby("zone_side", sort=True):
            sf = sf0.loc[sf0["touch_720m"].astype(bool)].copy()
            if sf.empty:
                continue
            w = pd.to_numeric(sf.get("sample_weight", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=float)
            for task, score in (
                ("HIGH_STRENGTH_PATH_NO_SWING", "p_strength_path"),
                ("HIGH_STRENGTH_FULL_WITH_SWING", "p_strength_full"),
                ("HIGH_STRENGTH_DISTANCE_BASELINE", "p_strength_baseline"),
            ):
                rows.append({"period": period, "zone_side": side, "task": task, **_binary_metrics(sf["high_strength_label"].to_numpy(dtype=np.int8), sf[score].to_numpy(dtype=float), w)})
            for task, score in (
                ("DENSITY_PATH_NO_SWING", "pred_density_path"),
                ("DENSITY_FULL_WITH_SWING", "pred_density_full"),
                ("DENSITY_DISTANCE_BASELINE", "pred_density_baseline"),
            ):
                rows.append({"period": period, "zone_side": side, "task": task, **_reg_metrics(sf["release_density_log"].to_numpy(dtype=float), sf[score].to_numpy(dtype=float))})
            released = sf.loc[sf["release_episode_count"].gt(0)].copy()
            if released.empty:
                continue
            wr = pd.to_numeric(released.get("sample_weight", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=float)
            favorable = released["favorable_episode_count"].gt(0).to_numpy(dtype=np.int8)
            continuation = released["continuation_episode_count"].gt(0).to_numpy(dtype=np.int8)
            for task, target, score in (
                ("FAVORABLE_PATH_NO_SWING", favorable, "p_favorable_path"),
                ("FAVORABLE_FULL_WITH_SWING", favorable, "p_favorable_full"),
                ("CONTINUATION_PATH_NO_SWING", continuation, "p_continuation_path"),
                ("CONTINUATION_FULL_WITH_SWING", continuation, "p_continuation_full"),
            ):
                rows.append({"period": period, "zone_side": side, "task": task, **_binary_metrics(target, released[score].to_numpy(dtype=float), wr)})
            rows.append({"period": period, "zone_side": side, "task": "SWEEP_DEPTH_PATH_NO_SWING", **_reg_metrics(released["sweep_depth_weighted_bp"].to_numpy(dtype=float), released["pred_sweep_depth_path_bp"].to_numpy(dtype=float))})
    return pd.DataFrame(rows)


def feature_importance(models: StrengthModelBundle) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    tasks = (
        ("HIGH_STRENGTH_FULL_WITH_SWING", models.strength_full, models.full_columns),
        ("DENSITY_FULL_WITH_SWING", models.density_full, models.full_columns),
        ("FAVORABLE_FULL_WITH_SWING", models.favorable_full, models.full_columns),
    )
    for task, pipe, cols in tasks:
        raw = np.asarray(pipe.named_steps["model"].feature_importances_, dtype=float)[:len(cols)]
        denom = max(float(raw.sum()), 1e-12)
        for name, value in zip(cols, raw, strict=True):
            rows.append({"task": task, "feature": name, "feature_family": "SWING_SUPPLEMENT" if name.startswith("swing_") else "LIQUIDITY_PATH", "importance": float(value), "importance_share": float(value / denom)})
    return pd.DataFrame(rows).sort_values(["task", "importance"], ascending=[True, False]).reset_index(drop=True)
