#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""LightGBM models and diagnostics for R02 spatial liquidity forecasting."""
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

from .config import LatentLiquidityPoolForecastConfig

_LABEL_PREFIXES = (
    "touch_", "release_", "favorable_", "continuation_", "time_to_release_",
    "sweep_depth_", "reversal_after_", "time_to_extreme_",
)
_META = {
    "zone_id", "decision_time", "feature_available_time", "period", "zone_side",
    "zone_price", "zone_near_price", "zone_far_price", "zone_near_distance_bp", "zone_far_distance_bp", "current_price", "release_outcome_type", "sample_weight",
    "primary_touch_label_complete", "model_sample_keep", "full_lattice_audit_group",
}


def feature_columns(frame: pd.DataFrame, *, include_swing: bool) -> tuple[str, ...]:
    cols: list[str] = []
    for name in frame.columns:
        if name in _META or name.startswith(_LABEL_PREFIXES):
            continue
        if name in {"release_within_horizon", "release_on_touch", "favorable_on_release", "release_path_cluster"}:
            continue
        if not include_swing and name.startswith("swing_"):
            continue
        if name.startswith("macro_notional_") and not name.startswith("macro_notional_intensity_"):
            continue
        if name.startswith(("micro_path_notional_", "micro_path_delta_", "micro_path_trades_", "micro_path_max_trade_", "micro_path_large_delta_", "micro_path_turnover_per_range_")) and not name.startswith(("micro_path_notional_intensity_", "micro_path_delta_share_", "micro_path_trades_intensity_", "micro_path_max_trade_ratio_", "micro_path_turnover_per_range_intensity_")):
            continue
        if pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name]):
            cols.append(name)
    return tuple(cols)


def baseline_columns(all_columns: tuple[str, ...]) -> tuple[str, ...]:
    allowed = {"zone_distance_bp", "side_is_down", "zone_boundary_nesting_count", "zone_untouched_window_count"}
    return tuple(name for name in all_columns if name in allowed)


def _classifier(config: LatentLiquidityPoolForecastConfig) -> Pipeline:
    if LGBMClassifier is None:
        raise RuntimeError("lightgbm is required for R02")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", LGBMClassifier(
            objective="binary", n_estimators=config.model_n_estimators,
            learning_rate=config.model_learning_rate, num_leaves=config.model_num_leaves,
            min_child_samples=config.model_min_child_samples, subsample=0.85,
            colsample_bytree=0.80, reg_alpha=1.0, reg_lambda=4.0,
            class_weight="balanced", random_state=config.random_state, n_jobs=-1, verbosity=-1,
        )),
    ]).set_output(transform="pandas")


def _regressor(config: LatentLiquidityPoolForecastConfig) -> Pipeline:
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is required for R02")
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


def _cap(frame: pd.DataFrame, cap: int, seed_col: str = "zone_id") -> pd.DataFrame:
    if len(frame) <= cap:
        return frame
    priority = pd.util.hash_pandas_object(frame[seed_col].astype(str), index=False).to_numpy(dtype=np.uint64)
    order = np.argpartition(priority, cap - 1)[:cap]
    return frame.iloc[np.sort(order)].copy()


def _check_binary(frame: pd.DataFrame, target: str, config: LatentLiquidityPoolForecastConfig) -> None:
    y = frame[target].astype(bool).to_numpy(dtype=np.int8)
    if len(y) < config.minimum_train_rows:
        raise RuntimeError(f"R02 {target} insufficient rows: {len(y)}")
    counts = np.bincount(y, minlength=2)
    if int(counts.min()) < config.minimum_class_rows:
        raise RuntimeError(f"R02 {target} insufficient class rows: {counts.tolist()}")


@dataclass
class ModelBundle:
    full_columns: tuple[str, ...]
    path_columns: tuple[str, ...]
    baseline_columns: tuple[str, ...]
    touch_full: Pipeline
    release_full: Pipeline
    release_path: Pipeline
    release_baseline: Pipeline
    favorable_full: Pipeline
    favorable_path: Pipeline
    sweep_depth: Pipeline
    reversal_room: Pipeline


def fit_models(frame: pd.DataFrame, config: LatentLiquidityPoolForecastConfig) -> ModelBundle:
    train = frame.loc[frame["period"].astype(str).eq(config.train_period)].copy()
    train = _cap(train, config.model_train_cap_rows)
    full_cols = feature_columns(train, include_swing=True)
    path_cols = feature_columns(train, include_swing=False)
    base_cols = baseline_columns(path_cols)
    if not full_cols or not path_cols or not base_cols:
        raise RuntimeError("R02 feature schema is empty")
    xfull = train.loc[:, full_cols]
    xpath = train.loc[:, path_cols]
    xbase = train.loc[:, base_cols]
    weights = pd.to_numeric(train["sample_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    primary_touch = f"touch_{config.primary_horizon_minutes}m"
    _check_binary(train, primary_touch, config)
    touch = _classifier(config); touch.fit(xfull, train[primary_touch].astype(bool), model__sample_weight=weights)

    touched = train.loc[train[primary_touch].astype(bool)].copy()
    _check_binary(touched, "release_within_horizon", config)
    wt = pd.to_numeric(touched["sample_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    rel_full = _classifier(config); rel_full.fit(touched.loc[:, full_cols], touched["release_within_horizon"].astype(bool), model__sample_weight=wt)
    rel_path = _classifier(config); rel_path.fit(touched.loc[:, path_cols], touched["release_within_horizon"].astype(bool), model__sample_weight=wt)
    rel_base = _classifier(config); rel_base.fit(touched.loc[:, base_cols], touched["release_within_horizon"].astype(bool), model__sample_weight=wt)

    released = train.loc[train["release_within_horizon"].astype(bool)].copy()
    _check_binary(released, "favorable_release", config)
    wr = pd.to_numeric(released["sample_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    fav_full = _classifier(config); fav_full.fit(released.loc[:, full_cols], released["favorable_release"].astype(bool), model__sample_weight=wr)
    fav_path = _classifier(config); fav_path.fit(released.loc[:, path_cols], released["favorable_release"].astype(bool), model__sample_weight=wr)
    depth = _regressor(config); depth.fit(released.loc[:, full_cols], released["sweep_depth_bp"].astype(float), model__sample_weight=wr)
    room = _regressor(config); room.fit(released.loc[:, full_cols], released["reversal_after_extreme_bp"].astype(float), model__sample_weight=wr)
    return ModelBundle(full_cols, path_cols, base_cols, touch, rel_full, rel_path, rel_base, fav_full, fav_path, depth, room)


def predict(frame: pd.DataFrame, models: ModelBundle) -> pd.DataFrame:
    out = frame.copy()
    xf = out.loc[:, models.full_columns]
    xp = out.loc[:, models.path_columns]
    xb = out.loc[:, models.baseline_columns]
    out["p_touch"] = models.touch_full.predict_proba(xf)[:, 1]
    out["p_release_full"] = models.release_full.predict_proba(xf)[:, 1]
    out["p_release_path"] = models.release_path.predict_proba(xp)[:, 1]
    out["p_release_baseline"] = models.release_baseline.predict_proba(xb)[:, 1]
    out["p_favorable_full"] = models.favorable_full.predict_proba(xf)[:, 1]
    out["p_favorable_path"] = models.favorable_path.predict_proba(xp)[:, 1]
    out["pred_sweep_depth_bp"] = np.maximum(0.0, models.sweep_depth.predict(xf))
    out["pred_reversal_room_bp"] = np.maximum(0.0, models.reversal_room.predict(xf))
    out["pool_score"] = np.cbrt(np.clip(out["p_touch"], 0, 1) * np.clip(out["p_release_full"], 0, 1) * np.clip(out["p_favorable_full"], 0, 1))
    out["pred_room_after_sweep_bp"] = out["pred_reversal_room_bp"] - out["pred_sweep_depth_bp"]
    return out


def _bin_metrics(y: np.ndarray, p: np.ndarray, weights: np.ndarray | None = None) -> dict[str, float | int]:
    valid = np.isfinite(p)
    y = y[valid].astype(np.int8); p = p[valid]
    if len(y) == 0:
        return {"rows": 0, "positive_rate": np.nan, "roc_auc": np.nan, "average_precision": np.nan}
    return {
        "rows": int(len(y)),
        "positive_rate": float(np.average(y, weights=weights[valid] if weights is not None else None)),
        "roc_auc": float(roc_auc_score(y, p, sample_weight=weights[valid] if weights is not None else None)) if len(np.unique(y)) == 2 else np.nan,
        "average_precision": float(average_precision_score(y, p, sample_weight=weights[valid] if weights is not None else None)) if len(np.unique(y)) == 2 else np.nan,
    }


def metric_table(pred: pd.DataFrame, config: LatentLiquidityPoolForecastConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    primary_touch = f"touch_{config.primary_horizon_minutes}m"
    for period, pf in pred.groupby("period", sort=True):
        for side, sf in pf.groupby("zone_side", sort=True):
            weights = pd.to_numeric(sf["sample_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
            for task, target, score in (
                ("TOUCH_FULL", primary_touch, "p_touch"),
                ("RELEASE_FULL", "release_within_horizon", "p_release_full"),
                ("RELEASE_PATH_NO_SWING", "release_within_horizon", "p_release_path"),
                ("RELEASE_DISTANCE_BASELINE", "release_within_horizon", "p_release_baseline"),
                ("FAVORABLE_FULL", "favorable_release", "p_favorable_full"),
                ("FAVORABLE_PATH_NO_SWING", "favorable_release", "p_favorable_path"),
            ):
                subset = sf
                if task.startswith("RELEASE_"):
                    subset = sf.loc[sf[primary_touch].astype(bool)]
                if task.startswith("FAVORABLE_"):
                    subset = sf.loc[sf["release_within_horizon"].astype(bool)]
                if subset.empty:
                    continue
                w = pd.to_numeric(subset["sample_weight"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
                metrics = _bin_metrics(subset[target].astype(bool).to_numpy(dtype=np.int8), subset[score].to_numpy(dtype=float), w)
                rows.append({"period": period, "zone_side": side, "task": task, **metrics})
            release = sf.loc[sf["release_within_horizon"].astype(bool)]
            if not release.empty:
                for task, target, score in (
                    ("SWEEP_DEPTH", "sweep_depth_bp", "pred_sweep_depth_bp"),
                    ("REVERSAL_ROOM", "reversal_after_extreme_bp", "pred_reversal_room_bp"),
                ):
                    y = pd.to_numeric(release[target], errors="coerce").to_numpy(dtype=float)
                    p = pd.to_numeric(release[score], errors="coerce").to_numpy(dtype=float)
                    valid = np.isfinite(y) & np.isfinite(p)
                    if int(valid.sum()) >= 20 and np.nanstd(y[valid]) > 1e-12 and np.nanstd(p[valid]) > 1e-12:
                        rho = spearmanr(y[valid], p[valid]).statistic
                    else:
                        rho = np.nan
                    rows.append({"period": period, "zone_side": side, "task": task, "rows": int(valid.sum()), "mae_bp": float(mean_absolute_error(y[valid], p[valid])) if valid.any() else np.nan, "spearman": float(rho) if np.isfinite(rho) else np.nan})
    return pd.DataFrame(rows)


def feature_importance(models: ModelBundle) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for task, pipe in (("RELEASE_FULL", models.release_full), ("FAVORABLE_FULL", models.favorable_full), ("SWEEP_DEPTH", models.sweep_depth), ("REVERSAL_ROOM", models.reversal_room)):
        values = np.asarray(pipe.named_steps["model"].feature_importances_, dtype=float)[: len(models.full_columns)]
        denom = max(float(values.sum()), 1e-12)
        for name, value in zip(models.full_columns, values, strict=True):
            rows.append({"task": task, "feature": name, "feature_family": "SWING_SUPPLEMENT" if name.startswith("swing_") else "LIQUIDITY_PATH", "importance": float(value), "importance_share": float(value / denom)})
    return pd.DataFrame(rows).sort_values(["task", "importance"], ascending=[True, False]).reset_index(drop=True)
