#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Past-only hurdle nuisance models aligned to E[log1p(release_density) | X].

R02.3.1 mixed two scales: it modeled positive log-density, converted that estimate
back to raw density, multiplied by release probability, and only then applied log1p.
R02.3.1b keeps the old proxy for diagnosis but defines its primary expectation on the
same scale as the residual target:

    E[Z | X] = P(Y > 0 | X) * E[log1p(Y) | Y > 0, X],  Z = log1p(Y)

The conditional log-magnitude mean uses L2 regression. A parallel Huber fit is kept
only to separate the nonlinear transform mismatch from the old robust-location
objective mismatch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline

from src.research_common.progress import ProgressReporter
from src.ai_research.latent_liquidity_hurdle_residualization.nuisance import (
    nuisance_feature_audit,
    nuisance_feature_columns,
    prepare_nuisance_frame,
)

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None

from .config import TargetConsistencyConfig


@dataclass
class FittedTargetConsistencyModels:
    release: Pipeline
    positive_log_huber: Pipeline
    positive_log_mean: Pipeline
    huber_smearing_factor: float


@dataclass
class TargetConsistencyNuisanceResult:
    frame: pd.DataFrame
    feature_audit: pd.DataFrame
    fold_audit: pd.DataFrame


def _classifier(config: TargetConsistencyConfig) -> Pipeline:
    if LGBMClassifier is None:
        raise RuntimeError("lightgbm is required for R02.3.1b")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", LGBMClassifier(
            objective="binary",
            n_estimators=config.nuisance_model_n_estimators,
            learning_rate=config.nuisance_model_learning_rate,
            num_leaves=config.nuisance_model_num_leaves,
            max_depth=config.nuisance_model_max_depth,
            min_child_samples=config.nuisance_model_min_child_samples,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_alpha=2.0,
            reg_lambda=8.0,
            random_state=config.random_state,
            n_jobs=-1,
            verbosity=-1,
        )),
    ]).set_output(transform="pandas")


def _regressor(config: TargetConsistencyConfig, *, objective: str) -> Pipeline:
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is required for R02.3.1b")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", LGBMRegressor(
            objective=objective,
            n_estimators=config.nuisance_model_n_estimators,
            learning_rate=config.nuisance_model_learning_rate,
            num_leaves=config.nuisance_model_num_leaves,
            max_depth=config.nuisance_model_max_depth,
            min_child_samples=config.nuisance_model_min_child_samples,
            subsample=0.90,
            colsample_bytree=0.90,
            reg_alpha=2.0,
            reg_lambda=8.0,
            random_state=config.random_state,
            n_jobs=-1,
            verbosity=-1,
        )),
    ]).set_output(transform="pandas")


def _cap(frame: pd.DataFrame, cap: int) -> pd.DataFrame:
    if len(frame) <= cap:
        return frame
    key = frame.get("zone_id", frame.index.to_series()).astype(str)
    priority = pd.util.hash_pandas_object(key, index=False).to_numpy(dtype=np.uint64)
    keep = np.argpartition(priority, cap - 1)[:cap]
    return frame.iloc[np.sort(keep)].copy()


def construct_target_consistency_columns(
    frame: pd.DataFrame,
    *,
    p_release: np.ndarray | pd.Series,
    huber_log_if_release: np.ndarray | pd.Series,
    mean_log_if_release: np.ndarray | pd.Series,
    huber_smearing_factor: float,
) -> pd.DataFrame:
    """Attach old-scale and same-scale hurdle expectations without model fitting."""
    out = frame.copy()
    p = np.clip(np.asarray(p_release, dtype=float), 1e-6, 1.0 - 1e-6)
    huber_log = np.maximum(0.0, np.asarray(huber_log_if_release, dtype=float))
    mean_log = np.maximum(0.0, np.asarray(mean_log_if_release, dtype=float))
    raw_log = pd.to_numeric(out["raw_log_release_density"], errors="coerce").to_numpy(dtype=float)

    legacy_positive_density = np.maximum(0.0, np.exp(huber_log) * float(huber_smearing_factor) - 1.0)
    legacy_expected_density = p * legacy_positive_density
    legacy_expected_log_proxy = np.log1p(legacy_expected_density)
    formula_only_expected_log = p * huber_log
    mean_aligned_expected_log = p * mean_log

    out["nuisance_p_release"] = p
    out["nuisance_huber_log_density_if_release"] = huber_log
    out["nuisance_mean_log_density_if_release"] = mean_log
    out["legacy_nuisance_pred_density_if_release"] = legacy_positive_density
    out["legacy_nuisance_expected_density"] = legacy_expected_density
    out["legacy_expected_log_proxy"] = legacy_expected_log_proxy
    out["formula_only_expected_log_density"] = formula_only_expected_log
    out["mean_aligned_expected_log_density"] = mean_aligned_expected_log
    out["legacy_excess_residual"] = raw_log - legacy_expected_log_proxy
    out["formula_only_excess_residual"] = raw_log - formula_only_expected_log
    out["target_consistent_excess_residual"] = raw_log - mean_aligned_expected_log
    out["transform_gap_formula_only_minus_legacy"] = formula_only_expected_log - legacy_expected_log_proxy
    out["objective_gap_mean_minus_huber"] = mean_aligned_expected_log - formula_only_expected_log
    out["total_expected_log_correction"] = mean_aligned_expected_log - legacy_expected_log_proxy
    return out


def _fit_models(train: pd.DataFrame, cols: tuple[str, ...], config: TargetConsistencyConfig) -> FittedTargetConsistencyModels:
    work = train.loc[train["r02_3_1b_upstream_eligible"].astype(bool)].copy()
    if len(work) < config.nuisance_min_rows:
        raise RuntimeError(f"R02.3.1b nuisance train rows too small: {len(work)}")
    work = _cap(work, config.nuisance_train_cap_rows_per_side)
    y = work["release_observed_180s"].astype(bool).to_numpy(dtype=np.int8)
    counts = np.bincount(y, minlength=2)
    if int(counts.min()) < config.nuisance_min_class_rows:
        raise RuntimeError(f"R02.3.1b nuisance release classes too small: {counts.tolist()}")

    release = _classifier(config)
    release.fit(work.loc[:, cols], y)

    positive = work.loc[work["release_observed_180s"].astype(bool)].copy()
    if len(positive) < config.nuisance_min_positive_rows:
        raise RuntimeError(f"R02.3.1b nuisance positive rows too small: {len(positive)}")
    y_log = positive["raw_log_release_density"].to_numpy(dtype=float)

    huber = _regressor(config, objective="huber")
    huber.fit(positive.loc[:, cols], y_log)
    train_huber = np.maximum(0.0, huber.predict(positive.loc[:, cols]))
    residual = y_log - train_huber
    finite_residual = residual[np.isfinite(residual)]
    smearing = float(np.mean(np.exp(np.clip(finite_residual, -4.0, 4.0)))) if len(finite_residual) else 1.0
    smearing = max(smearing, 1e-6)

    mean_model = _regressor(config, objective="regression_l2")
    mean_model.fit(positive.loc[:, cols], y_log)
    return FittedTargetConsistencyModels(
        release=release,
        positive_log_huber=huber,
        positive_log_mean=mean_model,
        huber_smearing_factor=smearing,
    )


def _predict_into(
    out: pd.DataFrame,
    idx: pd.Index,
    models: FittedTargetConsistencyModels,
    cols: tuple[str, ...],
    source: str,
) -> None:
    if len(idx) == 0:
        return
    x = out.loc[idx, cols]
    p = models.release.predict_proba(x)[:, 1]
    huber_log = models.positive_log_huber.predict(x)
    mean_log = models.positive_log_mean.predict(x)
    scored = construct_target_consistency_columns(
        out.loc[idx],
        p_release=p,
        huber_log_if_release=huber_log,
        mean_log_if_release=mean_log,
        huber_smearing_factor=models.huber_smearing_factor,
    )
    cols_to_copy = (
        "nuisance_p_release",
        "nuisance_huber_log_density_if_release",
        "nuisance_mean_log_density_if_release",
        "legacy_nuisance_pred_density_if_release",
        "legacy_nuisance_expected_density",
        "legacy_expected_log_proxy",
        "formula_only_expected_log_density",
        "mean_aligned_expected_log_density",
        "legacy_excess_residual",
        "formula_only_excess_residual",
        "target_consistent_excess_residual",
        "transform_gap_formula_only_minus_legacy",
        "objective_gap_mean_minus_huber",
        "total_expected_log_correction",
    )
    for name in cols_to_copy:
        out.loc[idx, name] = scored[name].to_numpy()
    out.loc[idx, "nuisance_prediction_source"] = source


def _month_start(values: pd.Series) -> pd.Series:
    return values.dt.to_period("M").dt.to_timestamp()


def attach_past_only_target_consistency_predictions(
    frame: pd.DataFrame,
    config: TargetConsistencyConfig,
    *,
    progress: bool = False,
) -> TargetConsistencyNuisanceResult:
    """Create expanding OOS TRAIN and full-TRAIN-frozen future nuisance estimates."""
    w = config.primary_label_window_seconds
    out = prepare_nuisance_frame(frame)
    split_purge = pd.Timedelta(hours=config.split_boundary_purge_hours)
    decision_time = pd.to_datetime(out["decision_time"], errors="coerce")
    out["split_purge_eligible"] = True
    out.loc[
        out["period"].astype(str).eq(config.train_period)
        & decision_time.ge(pd.Timestamp("2025-01-01") - split_purge),
        "split_purge_eligible",
    ] = False
    out.loc[
        out["period"].astype(str).eq(config.calibration_period)
        & decision_time.ge(pd.Timestamp("2025-10-01") - split_purge),
        "split_purge_eligible",
    ] = False

    density = pd.to_numeric(out[f"ft_release_density_sum_{w}s"], errors="coerce").fillna(0.0).clip(lower=0.0)
    release_count = pd.to_numeric(out[f"ft_release_episode_count_{w}s"], errors="coerce").fillna(0.0)
    out["release_observed_180s"] = release_count.gt(0)
    out["raw_release_density"] = density
    out["raw_log_release_density"] = np.log1p(density)
    out["nuisance_prediction_source"] = "UNSCORED"
    for name in (
        "nuisance_p_release",
        "nuisance_huber_log_density_if_release",
        "nuisance_mean_log_density_if_release",
        "legacy_nuisance_pred_density_if_release",
        "legacy_nuisance_expected_density",
        "legacy_expected_log_proxy",
        "formula_only_expected_log_density",
        "mean_aligned_expected_log_density",
        "legacy_excess_residual",
        "formula_only_excess_residual",
        "target_consistent_excess_residual",
        "transform_gap_formula_only_minus_legacy",
        "objective_gap_mean_minus_huber",
        "total_expected_log_correction",
    ):
        out[name] = np.nan

    cols = nuisance_feature_columns(out)
    feature_audit = nuisance_feature_audit(out, cols)
    failed_features = feature_audit.loc[feature_audit["status"].eq("FAIL"), "feature"].tolist()
    if failed_features:
        raise RuntimeError(f"R02.3.1b nuisance features are not group-level constants: {failed_features}")

    fold_rows: list[dict[str, object]] = []
    purge = pd.Timedelta(hours=config.nuisance_purge_hours)
    total_fits = 0
    for side in ("DOWN", "UP"):
        mask = out["zone_side"].astype(str).eq(side) & out["period"].astype(str).eq(config.train_period)
        months = pd.Index(sorted(_month_start(out.loc[mask, "decision_time"]).dropna().unique()))
        if len(months) > config.nuisance_initial_train_months:
            total_fits += int(np.ceil((len(months) - config.nuisance_initial_train_months) / config.nuisance_forward_block_months)) + 1
    reporter = ProgressReporter(label="[r02.3.1b] nuisance fits", total=total_fits, every=1, enabled=progress)
    done = 0

    for side in ("DOWN", "UP"):
        side_mask = out["zone_side"].astype(str).eq(side)
        train_mask = side_mask & out["period"].astype(str).eq(config.train_period)
        train_labels = train_mask & out["r02_3_1b_upstream_eligible"].astype(bool) & out["split_purge_eligible"].astype(bool)
        months = pd.Index(sorted(_month_start(out.loc[train_mask, "decision_time"]).dropna().unique()))
        if len(months) <= config.nuisance_initial_train_months:
            raise RuntimeError(f"R02.3.1b needs more TRAIN months: side={side} months={len(months)}")

        start = config.nuisance_initial_train_months
        while start < len(months):
            block_months = months[start:start + config.nuisance_forward_block_months]
            block_start = pd.Timestamp(block_months[0])
            block_end = pd.Timestamp(block_months[-1]) + pd.offsets.MonthBegin(1)
            fit_end = block_start - purge
            fit_mask = train_labels & out["decision_time"].lt(fit_end)
            pred_mask = (
                side_mask
                & out["period"].astype(str).eq(config.train_period)
                & out["decision_time"].ge(block_start)
                & out["decision_time"].lt(block_end)
            )
            models = _fit_models(out.loc[fit_mask].copy(), cols, config)
            _predict_into(out, out.index[pred_mask], models, cols, "TRAIN_EXPANDING_OOS")
            fold_rows.append({
                "zone_side": side,
                "prediction_source": "TRAIN_EXPANDING_OOS",
                "block_start": block_start,
                "block_end_exclusive": block_end,
                "fit_end_exclusive": fit_end,
                "fit_rows": int(fit_mask.sum()),
                "fit_release_rows": int(out.loc[fit_mask, "release_observed_180s"].sum()),
                "pred_rows": int(pred_mask.sum()),
                "huber_smearing_factor": float(models.huber_smearing_factor),
                "positive_log_primary_objective": "regression_l2",
                "causal_fit_before_prediction": bool(fit_end < block_start),
            })
            done += 1
            reporter.update(done)
            start += config.nuisance_forward_block_months

        final_models = _fit_models(out.loc[train_labels].copy(), cols, config)
        future_mask = side_mask & out["period"].astype(str).isin([config.calibration_period, config.holdout_period])
        _predict_into(out, out.index[future_mask], final_models, cols, "TRAIN_FULL_FROZEN")
        fold_rows.append({
            "zone_side": side,
            "prediction_source": "TRAIN_FULL_FROZEN",
            "block_start": pd.NaT,
            "block_end_exclusive": pd.NaT,
            "fit_end_exclusive": out.loc[train_labels, "decision_time"].max(),
            "fit_rows": int(train_labels.sum()),
            "fit_release_rows": int(out.loc[train_labels, "release_observed_180s"].sum()),
            "pred_rows": int(future_mask.sum()),
            "huber_smearing_factor": float(final_models.huber_smearing_factor),
            "positive_log_primary_objective": "regression_l2",
            "causal_fit_before_prediction": True,
        })
        done += 1
        reporter.update(done)
    reporter.close()

    prediction_ok = out["nuisance_prediction_source"].isin(["TRAIN_EXPANDING_OOS", "TRAIN_FULL_FROZEN"])
    upstream = out["r02_3_1b_upstream_eligible"].astype(bool)
    out["r02_3_1b_source_eligible"] = (
        upstream
        & prediction_ok
        & out["split_purge_eligible"].astype(bool)
        & out["mean_aligned_expected_log_density"].notna()
    )
    out["release_probability_surprise"] = (
        out["release_observed_180s"].astype(float) - pd.to_numeric(out["nuisance_p_release"], errors="coerce")
    )
    out["positive_log_density_mean_residual"] = np.where(
        out["release_observed_180s"].astype(bool),
        out["raw_log_release_density"] - pd.to_numeric(out["nuisance_mean_log_density_if_release"], errors="coerce"),
        np.nan,
    )
    return TargetConsistencyNuisanceResult(
        frame=out,
        feature_audit=feature_audit,
        fold_audit=pd.DataFrame(fold_rows),
    )


def nuisance_metric_table(frame: pd.DataFrame, config: TargetConsistencyConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (period, side), sf in frame.groupby(["period", "zone_side"], sort=True):
        work = sf.loc[sf["r02_3_1b_source_eligible"].astype(bool)].copy()
        if work.empty:
            continue
        y = work["release_observed_180s"].astype(int).to_numpy(dtype=int)
        p = pd.to_numeric(work["nuisance_p_release"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(p)
        rows.append({
            "period": period,
            "zone_side": side,
            "task": "RELEASE_HURDLE",
            "rows": int(valid.sum()),
            "actual_mean": float(np.mean(y[valid])) if valid.any() else np.nan,
            "predicted_mean": float(np.mean(p[valid])) if valid.any() else np.nan,
            "roc_auc": float(roc_auc_score(y[valid], p[valid])) if valid.any() and len(np.unique(y[valid])) == 2 else np.nan,
            "average_precision": float(average_precision_score(y[valid], p[valid])) if valid.any() and len(np.unique(y[valid])) == 2 else np.nan,
            "brier": float(brier_score_loss(y[valid], p[valid])) if valid.any() else np.nan,
        })

        pos = work.loc[work["release_observed_180s"].astype(bool)].copy()
        actual_pos = pd.to_numeric(pos["raw_log_release_density"], errors="coerce").to_numpy(dtype=float)
        for task, col in (
            ("POSITIVE_LOG_HUBER_DIAGNOSTIC", "nuisance_huber_log_density_if_release"),
            ("POSITIVE_LOG_MEAN_PRIMARY", "nuisance_mean_log_density_if_release"),
        ):
            pred = pd.to_numeric(pos[col], errors="coerce").to_numpy(dtype=float)
            good = np.isfinite(actual_pos) & np.isfinite(pred)
            rho = np.nan
            if int(good.sum()) >= 20 and np.nanstd(actual_pos[good]) > 1e-12 and np.nanstd(pred[good]) > 1e-12:
                value = spearmanr(actual_pos[good], pred[good]).statistic
                rho = float(value) if np.isfinite(value) else np.nan
            rows.append({
                "period": period,
                "zone_side": side,
                "task": task,
                "rows": int(good.sum()),
                "actual_mean": float(np.mean(actual_pos[good])) if good.any() else np.nan,
                "predicted_mean": float(np.mean(pred[good])) if good.any() else np.nan,
                "mae": float(mean_absolute_error(actual_pos[good], pred[good])) if good.any() else np.nan,
                "spearman": rho,
            })

        actual_log = pd.to_numeric(work["raw_log_release_density"], errors="coerce").to_numpy(dtype=float)
        for task, col in (
            ("LEGACY_EXPECTED_LOG_PROXY", "legacy_expected_log_proxy"),
            ("FORMULA_ONLY_EXPECTED_LOG", "formula_only_expected_log_density"),
            ("MEAN_ALIGNED_EXPECTED_LOG_PRIMARY", "mean_aligned_expected_log_density"),
        ):
            pred_log = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
            good = np.isfinite(actual_log) & np.isfinite(pred_log)
            rows.append({
                "period": period,
                "zone_side": side,
                "task": task,
                "rows": int(good.sum()),
                "actual_mean": float(np.mean(actual_log[good])) if good.any() else np.nan,
                "predicted_mean": float(np.mean(pred_log[good])) if good.any() else np.nan,
                "actual_to_predicted_ratio": (
                    float(np.mean(actual_log[good]) / max(np.mean(pred_log[good]), 1e-12)) if good.any() else np.nan
                ),
                "mae": float(mean_absolute_error(actual_log[good], pred_log[good])) if good.any() else np.nan,
            })
    return pd.DataFrame(rows)
