#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Past-only hurdle nuisance models for R02.3.1.

The nuisance family is deliberately narrow: raw zone distance, calendar/session,
and broad activity/volatility features that are constant across all zones inside a
decision_time x side ranking group. It may explain mechanical exposure/activity,
but it cannot use zone-specific liquidity-path structure or Swing.
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

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMClassifier = None
    LGBMRegressor = None

from .config import HurdleResidualizationConfig

_ACTIVITY_PREFIXES = (
    "macro_notional_intensity_",
    "macro_trades_intensity_",
    "macro_realized_vol_",
    "macro_range_bp_",
    "micro_path_notional_intensity_",
    "micro_path_trades_intensity_",
    "micro_path_realized_vol_",
    "micro_path_range_bp_",
)
_DERIVED_NUISANCE = (
    "nuisance_hour_sin",
    "nuisance_hour_cos",
    "nuisance_dow_sin",
    "nuisance_dow_cos",
)


@dataclass
class FittedNuisanceModels:
    release: Pipeline
    positive_log_density: Pipeline
    reversal_quality: Pipeline
    density_smearing_factor: float


@dataclass
class NuisanceResult:
    frame: pd.DataFrame
    feature_audit: pd.DataFrame
    fold_audit: pd.DataFrame


def prepare_nuisance_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["decision_time"] = pd.to_datetime(out["decision_time"], errors="coerce")
    hour = out["decision_time"].dt.hour.fillna(0).to_numpy(dtype=float) + out["decision_time"].dt.minute.fillna(0).to_numpy(dtype=float) / 60.0
    dow = out["decision_time"].dt.dayofweek.fillna(0).to_numpy(dtype=float)
    out["nuisance_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0).astype(np.float32)
    out["nuisance_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0).astype(np.float32)
    out["nuisance_dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0).astype(np.float32)
    out["nuisance_dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0).astype(np.float32)
    out["ranking_group"] = out["decision_time"].astype(str) + "|" + out["zone_side"].astype(str)
    return out


def nuisance_feature_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    cols = ["zone_distance_bp"]
    for name in frame.columns:
        if name in _DERIVED_NUISANCE or name.startswith(_ACTIVITY_PREFIXES):
            if pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name]):
                cols.append(name)
    # deterministic order; zone_distance always first
    return tuple(dict.fromkeys(cols))


def nuisance_activity_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(c for c in nuisance_feature_columns(frame) if c != "zone_distance_bp")


def nuisance_feature_audit(frame: pd.DataFrame, cols: tuple[str, ...]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group = frame["ranking_group"].astype(str)
    for name in cols:
        if name == "zone_distance_bp":
            rows.append({
                "feature": name,
                "role": "SPATIAL_DISTANCE_NUISANCE",
                "max_unique_within_ranking_group": int(frame.groupby(group, sort=False)[name].nunique(dropna=False).max()),
                "group_level_constant_required": False,
                "status": "PASS",
            })
            continue
        max_unique = int(frame.groupby(group, sort=False)[name].nunique(dropna=False).max()) if len(frame) else 0
        rows.append({
            "feature": name,
            "role": "GROUP_LEVEL_ACTIVITY_NUISANCE",
            "max_unique_within_ranking_group": max_unique,
            "group_level_constant_required": True,
            "status": "PASS" if max_unique <= 1 else "FAIL",
        })
    return pd.DataFrame(rows)


def _classifier(config: HurdleResidualizationConfig) -> Pipeline:
    if LGBMClassifier is None:
        raise RuntimeError("lightgbm is required for R02.3.1")
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


def _regressor(config: HurdleResidualizationConfig) -> Pipeline:
    if LGBMRegressor is None:
        raise RuntimeError("lightgbm is required for R02.3.1")
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", LGBMRegressor(
            objective="huber",
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


def _fit_models(train: pd.DataFrame, cols: tuple[str, ...], config: HurdleResidualizationConfig) -> FittedNuisanceModels:
    work = train.loc[train["r02_3_1_upstream_eligible"].astype(bool)].copy()
    if len(work) < config.nuisance_min_rows:
        raise RuntimeError(f"R02.3.1 nuisance train rows too small: {len(work)}")
    work = _cap(work, config.nuisance_train_cap_rows_per_side)
    y = work["release_observed_180s"].astype(bool).to_numpy(dtype=np.int8)
    counts = np.bincount(y, minlength=2)
    if int(counts.min()) < config.nuisance_min_class_rows:
        raise RuntimeError(f"R02.3.1 nuisance release classes too small: {counts.tolist()}")
    release = _classifier(config)
    release.fit(work.loc[:, cols], y)

    positive = work.loc[work["release_observed_180s"].astype(bool)].copy()
    if len(positive) < config.nuisance_min_positive_rows:
        raise RuntimeError(f"R02.3.1 nuisance positive rows too small: {len(positive)}")
    mag = _regressor(config)
    mag.fit(positive.loc[:, cols], positive["raw_log_release_density"].to_numpy(dtype=float))
    train_log = positive["raw_log_release_density"].to_numpy(dtype=float)
    train_pred_log = np.maximum(0.0, mag.predict(positive.loc[:, cols]))
    residual = train_log - train_pred_log
    finite_residual = residual[np.isfinite(residual)]
    smearing = float(np.mean(np.exp(np.clip(finite_residual, -4.0, 4.0)))) if len(finite_residual) else 1.0
    smearing = max(smearing, 1e-6)
    reversal = _regressor(config)
    reversal.fit(positive.loc[:, cols], positive["raw_reversal_quality"].to_numpy(dtype=float))
    return FittedNuisanceModels(
        release=release,
        positive_log_density=mag,
        reversal_quality=reversal,
        density_smearing_factor=smearing,
    )


def _predict_into(out: pd.DataFrame, idx: pd.Index, models: FittedNuisanceModels, cols: tuple[str, ...], source: str) -> None:
    if len(idx) == 0:
        return
    x = out.loc[idx, cols]
    p = np.clip(models.release.predict_proba(x)[:, 1], 1e-6, 1.0 - 1e-6)
    log_mag = np.maximum(0.0, models.positive_log_density.predict(x))
    positive_density = np.maximum(0.0, np.exp(log_mag) * float(models.density_smearing_factor) - 1.0)
    rev = models.reversal_quality.predict(x)
    out.loc[idx, "nuisance_p_release"] = p
    out.loc[idx, "nuisance_pred_log_density_if_release"] = log_mag
    out.loc[idx, "nuisance_pred_density_if_release"] = positive_density
    out.loc[idx, "nuisance_expected_density"] = p * positive_density
    out.loc[idx, "nuisance_expected_reversal_quality"] = rev
    out.loc[idx, "nuisance_prediction_source"] = source


def _month_start(values: pd.Series) -> pd.Series:
    return values.dt.to_period("M").dt.to_timestamp()


def attach_past_only_nuisance_predictions(
    frame: pd.DataFrame,
    config: HurdleResidualizationConfig,
    *,
    progress: bool = False,
) -> NuisanceResult:
    """Create expanding OOS TRAIN predictions and full-TRAIN frozen future predictions."""
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
    fav = pd.to_numeric(out[f"ft_favorable_density_sum_{w}s"], errors="coerce").fillna(0.0).clip(lower=0.0)
    cont = pd.to_numeric(out[f"ft_continuation_density_sum_{w}s"], errors="coerce").fillna(0.0).clip(lower=0.0)
    release_count = pd.to_numeric(out[f"ft_release_episode_count_{w}s"], errors="coerce").fillna(0.0)
    out["release_observed_180s"] = release_count.gt(0)
    out["raw_release_density"] = density
    out["raw_log_release_density"] = np.log1p(density)
    out["raw_reversal_quality"] = np.log1p(fav) - np.log1p(cont)
    out["sweep_depth_target_bp"] = pd.to_numeric(out[f"ft_sweep_depth_weighted_bp_{w}s"], errors="coerce")
    out["reversal_room_target_bp"] = pd.to_numeric(out[f"ft_reversal_room_weighted_bp_{w}s"], errors="coerce")
    out["nuisance_p_release"] = np.nan
    out["nuisance_pred_log_density_if_release"] = np.nan
    out["nuisance_pred_density_if_release"] = np.nan
    out["nuisance_expected_density"] = np.nan
    out["nuisance_expected_reversal_quality"] = np.nan
    out["nuisance_prediction_source"] = "UNSCORED"

    cols = nuisance_feature_columns(out)
    feature_audit = nuisance_feature_audit(out, cols)
    failed_features = feature_audit.loc[feature_audit["status"].eq("FAIL"), "feature"].tolist()
    if failed_features:
        raise RuntimeError(f"R02.3.1 nuisance activity features are not group-level constants: {failed_features}")

    fold_rows: list[dict[str, object]] = []
    purge = pd.Timedelta(hours=config.nuisance_purge_hours)
    total_fits = 0
    for _side in ("DOWN", "UP"):
        _mask = out["zone_side"].astype(str).eq(_side) & out["period"].astype(str).eq(config.train_period)
        _months = pd.Index(sorted(_month_start(out.loc[_mask, "decision_time"]).dropna().unique()))
        if len(_months) > config.nuisance_initial_train_months:
            total_fits += int(np.ceil((len(_months) - config.nuisance_initial_train_months) / config.nuisance_forward_block_months)) + 1
    reporter = ProgressReporter(label="[r02.3.1] nuisance fits", total=total_fits, every=1, enabled=progress)
    done_fits = 0
    for side in ("DOWN", "UP"):
        side_mask = out["zone_side"].astype(str).eq(side)
        train_mask = side_mask & out["period"].astype(str).eq(config.train_period)
        train_labels = train_mask & out["r02_3_1_upstream_eligible"].astype(bool) & out["split_purge_eligible"].astype(bool)
        months = pd.Index(sorted(_month_start(out.loc[train_mask, "decision_time"]).dropna().unique()))
        if len(months) <= config.nuisance_initial_train_months:
            raise RuntimeError(f"R02.3.1 needs more TRAIN months for expanding nuisance OOS: side={side} months={len(months)}")
        start = config.nuisance_initial_train_months
        while start < len(months):
            block_months = months[start:start + config.nuisance_forward_block_months]
            block_start = pd.Timestamp(block_months[0])
            block_end = pd.Timestamp(block_months[-1]) + pd.offsets.MonthBegin(1)
            fit_end = block_start - purge
            fit_mask = train_labels & out["decision_time"].lt(fit_end)
            pred_mask = side_mask & out["period"].astype(str).eq(config.train_period) & out["decision_time"].ge(block_start) & out["decision_time"].lt(block_end)
            fit_frame = out.loc[fit_mask].copy()
            models = _fit_models(fit_frame, cols, config)
            idx = out.index[pred_mask]
            _predict_into(out, idx, models, cols, "TRAIN_EXPANDING_OOS")
            fold_rows.append({
                "zone_side": side,
                "prediction_source": "TRAIN_EXPANDING_OOS",
                "block_start": block_start,
                "block_end_exclusive": block_end,
                "fit_end_exclusive": fit_end,
                "fit_rows": int(fit_mask.sum()),
                "fit_release_rows": int(out.loc[fit_mask, "release_observed_180s"].sum()),
                "pred_rows": int(pred_mask.sum()),
                "density_smearing_factor": float(models.density_smearing_factor),
                "causal_fit_before_prediction": bool(fit_end < block_start),
            })
            done_fits += 1
            reporter.update(done_fits)
            start += config.nuisance_forward_block_months

        # Freeze one nuisance family on all TRAIN labels for Validation/Holdout.
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
            "density_smearing_factor": float(final_models.density_smearing_factor),
            "causal_fit_before_prediction": True,
        })
        done_fits += 1
        reporter.update(done_fits)
    reporter.close()

    # Only exact-touch rows with a genuinely out-of-sample nuisance estimate may
    # define residual targets. Warm-up TRAIN rows remain visible for scoring but
    # cannot train/evaluate the residual ranker.
    prediction_ok = out["nuisance_prediction_source"].isin(["TRAIN_EXPANDING_OOS", "TRAIN_FULL_FROZEN"])
    upstream = out["r02_3_1_upstream_eligible"].astype(bool)
    out["r02_3_1_source_eligible"] = upstream & prediction_ok & out["split_purge_eligible"].astype(bool) & out["nuisance_expected_density"].notna()
    expected = pd.to_numeric(out["nuisance_expected_density"], errors="coerce").clip(lower=0.0)
    out["excess_liquidity_residual"] = np.log1p(out["raw_release_density"]) - np.log1p(expected)
    out["density_vs_nuisance_expected_ratio"] = (1.0 + out["raw_release_density"]) / (1.0 + expected)
    out["release_probability_surprise"] = out["release_observed_180s"].astype(float) - pd.to_numeric(out["nuisance_p_release"], errors="coerce")
    out["positive_log_density_residual"] = np.where(
        out["release_observed_180s"].astype(bool),
        out["raw_log_release_density"] - pd.to_numeric(out["nuisance_pred_log_density_if_release"], errors="coerce"),
        np.nan,
    )
    out["reversal_quality_residual"] = out["raw_reversal_quality"] - pd.to_numeric(out["nuisance_expected_reversal_quality"], errors="coerce")
    return NuisanceResult(frame=out, feature_audit=feature_audit, fold_audit=pd.DataFrame(fold_rows))


def nuisance_metric_table(frame: pd.DataFrame, config: HurdleResidualizationConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (period, side), sf in frame.groupby(["period", "zone_side"], sort=True):
        eligible = sf["r02_3_1_source_eligible"].astype(bool)
        work = sf.loc[eligible].copy()
        if work.empty:
            continue
        y = work["release_observed_180s"].astype(int).to_numpy(dtype=int)
        p = pd.to_numeric(work["nuisance_p_release"], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(p)
        row: dict[str, object] = {
            "period": period,
            "zone_side": side,
            "task": "RELEASE_HURDLE",
            "rows": int(valid.sum()),
            "actual_mean": float(np.mean(y[valid])) if valid.any() else np.nan,
            "predicted_mean": float(np.mean(p[valid])) if valid.any() else np.nan,
            "roc_auc": float(roc_auc_score(y[valid], p[valid])) if valid.any() and len(np.unique(y[valid])) == 2 else np.nan,
            "average_precision": float(average_precision_score(y[valid], p[valid])) if valid.any() and len(np.unique(y[valid])) == 2 else np.nan,
            "brier": float(brier_score_loss(y[valid], p[valid])) if valid.any() else np.nan,
        }
        rows.append(row)

        pos = work.loc[work["release_observed_180s"].astype(bool)].copy()
        if not pos.empty:
            actual = pd.to_numeric(pos["raw_log_release_density"], errors="coerce").to_numpy(dtype=float)
            pred = pd.to_numeric(pos["nuisance_pred_log_density_if_release"], errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(actual) & np.isfinite(pred)
            rho = spearmanr(actual[valid], pred[valid]).statistic if int(valid.sum()) >= 20 and np.nanstd(actual[valid]) > 1e-12 and np.nanstd(pred[valid]) > 1e-12 else np.nan
            rows.append({
                "period": period, "zone_side": side, "task": "POSITIVE_LOG_DENSITY",
                "rows": int(valid.sum()), "mae": float(mean_absolute_error(actual[valid], pred[valid])) if valid.any() else np.nan,
                "spearman": float(rho) if np.isfinite(rho) else np.nan,
            })
            actual_r = pd.to_numeric(pos["raw_reversal_quality"], errors="coerce").to_numpy(dtype=float)
            pred_r = pd.to_numeric(pos["nuisance_expected_reversal_quality"], errors="coerce").to_numpy(dtype=float)
            valid_r = np.isfinite(actual_r) & np.isfinite(pred_r)
            rho_r = spearmanr(actual_r[valid_r], pred_r[valid_r]).statistic if int(valid_r.sum()) >= 20 and np.nanstd(actual_r[valid_r]) > 1e-12 and np.nanstd(pred_r[valid_r]) > 1e-12 else np.nan
            rows.append({
                "period": period, "zone_side": side, "task": "REVERSAL_NUISANCE",
                "rows": int(valid_r.sum()), "mae": float(mean_absolute_error(actual_r[valid_r], pred_r[valid_r])) if valid_r.any() else np.nan,
                "spearman": float(rho_r) if np.isfinite(rho_r) else np.nan,
            })
        actual_density = pd.to_numeric(work["raw_release_density"], errors="coerce").to_numpy(dtype=float)
        expected_density = pd.to_numeric(work["nuisance_expected_density"], errors="coerce").to_numpy(dtype=float)
        valid_d = np.isfinite(actual_density) & np.isfinite(expected_density)
        rows.append({
            "period": period, "zone_side": side, "task": "EXPECTED_DENSITY_CALIBRATION",
            "rows": int(valid_d.sum()),
            "actual_mean": float(np.mean(actual_density[valid_d])) if valid_d.any() else np.nan,
            "predicted_mean": float(np.mean(expected_density[valid_d])) if valid_d.any() else np.nan,
            "actual_to_predicted_ratio": float(np.mean(actual_density[valid_d]) / max(np.mean(expected_density[valid_d]), 1e-12)) if valid_d.any() else np.nan,
        })
    return pd.DataFrame(rows)
