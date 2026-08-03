#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression models and ranking diagnostics for R03.3.2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from src.ai_research.swing_baseline.dataset import load_year_shard

from .intensity_config import FutureIntensityConfig
from .intensity_targets import load_intensity_year_shard
from .micro_features import load_micro_year_shard

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover - dependency preflight reports this cleanly.
    LGBMRegressor = None  # type: ignore[assignment]


@dataclass(frozen=True)
class IntensityFold:
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


def default_intensity_folds(config: FutureIntensityConfig) -> tuple[IntensityFold, ...]:
    embargo = pd.Timedelta(hours=max(config.horizons_hours) + 12)
    return (
        IntensityFold(
            "WF_2024",
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-10-01") - embargo,
            pd.Timestamp("2023-10-01"),
            pd.Timestamp("2023-12-31 23:59:59"),
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-12-31 23:59:59"),
        ),
        IntensityFold(
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
class IntensityPeriodData:
    timestamps_ns: np.ndarray
    macro_x: np.ndarray
    full_x: np.ndarray
    micro_x: np.ndarray
    combined_x: np.ndarray
    targets: dict[str, np.ndarray]
    macro_columns: tuple[str, ...]
    full_columns: tuple[str, ...]
    micro_columns: tuple[str, ...]

    @property
    def index(self) -> pd.DatetimeIndex:
        return pd.to_datetime(self.timestamps_ns)


def _path_year(path: Path, kind: str) -> int:
    if kind == "base":
        times = load_year_shard(path).decision_times_ns
    elif kind == "target":
        times = load_intensity_year_shard(path).decision_times_ns
    elif kind == "micro":
        times = load_micro_year_shard(path).decision_times_ns
    else:
        raise ValueError(kind)
    return int(pd.to_datetime(np.asarray(times[:1], dtype=np.int64))[0].year)


def collect_intensity_period_data(
    base_paths: list[Path],
    target_paths: list[Path],
    micro_paths: list[Path],
    start: pd.Timestamp,
    end: pd.Timestamp,
    config: FutureIntensityConfig,
) -> IntensityPeriodData:
    base_map = {_path_year(path, "base"): path for path in base_paths}
    target_map = {_path_year(path, "target"): path for path in target_paths}
    micro_map = {_path_year(path, "micro"): path for path in micro_paths}

    macro_parts: list[np.ndarray] = []
    full_parts: list[np.ndarray] = []
    micro_parts: list[np.ndarray] = []
    time_parts: list[np.ndarray] = []
    target_parts: dict[str, list[np.ndarray]] = {name: [] for name in config.target_names()}
    expected_macro: tuple[str, ...] | None = None
    expected_full: tuple[str, ...] | None = None
    expected_micro: tuple[str, ...] | None = None

    for year in sorted(base_map):
        if year not in target_map or year not in micro_map:
            continue
        base = load_year_shard(base_map[year])
        target = load_intensity_year_shard(target_map[year])
        micro = load_micro_year_shard(micro_map[year])
        if not (
            np.array_equal(base.decision_times_ns, target.decision_times_ns)
            and np.array_equal(base.decision_times_ns, micro.decision_times_ns)
        ):
            raise RuntimeError(f"R03.3.2 decision-axis mismatch in {year}")
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
            raise RuntimeError(f"R03.3.2 base feature schema mismatch in {year}")
        if micro.feature_columns != expected_micro:
            raise RuntimeError(f"R03.3.2 micro feature schema mismatch in {year}")
        high_count = len(base.high_feature_columns)
        macro_parts.append(np.asarray(base.features[left:right, :high_count], dtype=np.float32))
        full_parts.append(np.asarray(base.features[left:right], dtype=np.float32))
        micro_parts.append(np.asarray(micro.features[left:right], dtype=np.float32))
        time_parts.append(times[left:right])
        index = target.target_index
        for name in config.target_names():
            target_parts[name].append(np.asarray(target.targets[left:right, index[name]], dtype=np.float32))

    if not time_parts:
        raise RuntimeError(f"no R03.3.2 rows for {start} -> {end}")
    timestamps = np.concatenate(time_parts)
    order = np.argsort(timestamps, kind="stable")
    macro = np.concatenate(macro_parts)[order]
    full = np.concatenate(full_parts)[order]
    micro = np.concatenate(micro_parts)[order]
    return IntensityPeriodData(
        timestamps_ns=timestamps[order],
        macro_x=macro,
        full_x=full,
        micro_x=micro,
        combined_x=np.concatenate([full, micro], axis=1),
        targets={name: np.concatenate(parts)[order] for name, parts in target_parts.items()},
        macro_columns=expected_macro or (),
        full_columns=expected_full or (),
        micro_columns=expected_micro or (),
    )


def validate_intensity_dependencies(config: FutureIntensityConfig) -> None:
    if LGBMRegressor is None:
        raise RuntimeError(
            "R03.3.2 dependency preflight failed: LightGBM is not installed. "
            "Install it before cache work with: python -m pip install lightgbm"
        )


def architecture_matrix(
    architecture: str,
    data: IntensityPeriodData,
) -> tuple[np.ndarray, tuple[str, ...]]:
    if architecture == "macro_lightgbm":
        return data.macro_x, data.macro_columns
    if architecture == "multiframe_lightgbm":
        return data.full_x, data.full_columns
    if architecture == "multiframe_micro_lightgbm":
        return data.combined_x, (*data.full_columns, *data.micro_columns)
    raise ValueError(f"unknown R03.3.2 architecture: {architecture}")


@dataclass
class FittedIntensityModel:
    model: object
    target_scale: float
    target_clip: float
    baseline_value: float

    def predict(self, x: np.ndarray) -> np.ndarray:
        raw = np.asarray(self.model.predict(x), dtype=float)
        transformed = np.maximum(raw, 0.0)
        prediction = self.target_scale * np.expm1(transformed)
        return np.clip(prediction, 0.0, self.target_clip)


def _training_positions(
    x: np.ndarray,
    y: np.ndarray,
    config: FutureIntensityConfig,
) -> np.ndarray:
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y) & (y >= 0)
    positions = np.flatnonzero(valid)[:: config.sample_stride_decisions]
    if len(positions) == 0:
        raise RuntimeError("R03.3.2 training period has zero valid strided rows")
    if len(positions) > config.train_sample_cap:
        rng = np.random.default_rng(config.base.random_seed)
        positions = np.sort(rng.choice(positions, size=config.train_sample_cap, replace=False))
    return positions


def fit_intensity_model(
    architecture: str,
    target_name: str,
    fit_data: IntensityPeriodData,
    config: FutureIntensityConfig,
) -> tuple[FittedIntensityModel, tuple[str, ...], dict[str, object]]:
    x, columns = architecture_matrix(architecture, fit_data)
    y = np.asarray(fit_data.targets[target_name], dtype=float)
    positions = _training_positions(x, y, config)
    train_y = y[positions]
    clip = float(np.nanquantile(train_y, config.target_clip_quantile))
    clipped = np.clip(train_y, 0.0, clip)
    positive = clipped[clipped > 0]
    scale = float(np.nanmedian(positive)) if len(positive) else max(float(np.nanmean(clipped)), 1e-4)
    scale = max(scale, 1e-6)
    transformed = np.log1p(clipped / scale)
    if LGBMRegressor is None:
        raise RuntimeError("LightGBM is required")
    model = LGBMRegressor(
        objective="regression_l1",
        n_estimators=config.lightgbm_n_estimators,
        learning_rate=config.lightgbm_learning_rate,
        num_leaves=config.lightgbm_num_leaves,
        min_child_samples=config.lightgbm_min_child_samples,
        colsample_bytree=config.lightgbm_feature_fraction,
        subsample=0.85,
        subsample_freq=1,
        reg_alpha=0.35,
        reg_lambda=1.5,
        random_state=config.base.random_seed,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x[positions], transformed)
    fitted = FittedIntensityModel(
        model=model,
        target_scale=scale,
        target_clip=clip,
        baseline_value=float(np.nanmedian(train_y)),
    )
    metadata = {
        "architecture": architecture,
        "target": target_name,
        "train_rows": int(len(positions)),
        "target_scale": scale,
        "target_clip": clip,
        "baseline_value": fitted.baseline_value,
    }
    return fitted, columns, metadata


def _rank(values: np.ndarray) -> np.ndarray:
    series = pd.Series(np.asarray(values, dtype=float))
    return series.rank(method="average").to_numpy(dtype=float)


def _correlation(left: np.ndarray, right: np.ndarray, *, rank: bool = False) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return np.nan
    x = _rank(left[valid]) if rank else np.asarray(left[valid], dtype=float)
    y = _rank(right[valid]) if rank else np.asarray(right[valid], dtype=float)
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def evaluate_intensity_model(
    *,
    fold_id: str,
    architecture: str,
    target_name: str,
    fitted: FittedIntensityModel,
    calibration_data: IntensityPeriodData,
    test_data: IntensityPeriodData,
    config: FutureIntensityConfig,
) -> tuple[dict[str, object], list[dict[str, object]], list[dict[str, object]], pd.DataFrame]:
    calibration_x, _ = architecture_matrix(architecture, calibration_data)
    test_x, _ = architecture_matrix(architecture, test_data)
    calibration_prediction = fitted.predict(calibration_x)
    test_prediction = fitted.predict(test_x)
    calibration_actual = np.asarray(calibration_data.targets[target_name], dtype=float)
    actual = np.asarray(test_data.targets[target_name], dtype=float)
    valid = np.isfinite(actual) & np.isfinite(test_prediction) & np.isfinite(test_x).all(axis=1)
    y = actual[valid]
    p = test_prediction[valid]
    timestamps = np.asarray(test_data.timestamps_ns, dtype=np.int64)[valid]
    if len(y) == 0:
        raise RuntimeError(f"R03.3.2 zero valid test rows for {fold_id}/{architecture}/{target_name}")

    baseline = np.full(len(y), fitted.baseline_value, dtype=float)
    rank_ic = _correlation(y, p, rank=True)
    pearson = _correlation(y, p, rank=False)

    order = np.argsort(p, kind="stable")
    bucket_id = np.empty(len(p), dtype=int)
    bucket_id[order] = np.minimum((np.arange(len(p)) * 10) // max(len(p), 1), 9)
    bucket_rows: list[dict[str, object]] = []
    bucket_means: list[float] = []
    for bucket in range(10):
        mask = bucket_id == bucket
        values = y[mask]
        predictions = p[mask]
        mean_actual = float(np.nanmean(values)) if len(values) else np.nan
        bucket_means.append(mean_actual)
        bucket_rows.append(
            {
                "fold_id": fold_id,
                "architecture": architecture,
                "target": target_name,
                "decile": bucket + 1,
                "rows": int(mask.sum()),
                "prediction_mean": float(np.nanmean(predictions)) if len(predictions) else np.nan,
                "actual_mean": mean_actual,
                "actual_median": float(np.nanmedian(values)) if len(values) else np.nan,
            }
        )
    monotonicity = _correlation(np.arange(1, 11, dtype=float), np.asarray(bucket_means), rank=True)
    overall_mean = float(np.nanmean(y))
    top_decile_mean = float(bucket_means[-1])
    top_decile_lift = top_decile_mean / overall_mean if overall_mean > 0 else np.nan

    metrics: dict[str, object] = {
        "fold_id": fold_id,
        "architecture": architecture,
        "target": target_name,
        "rows": int(len(y)),
        "actual_mean": overall_mean,
        "actual_median": float(np.nanmedian(y)),
        "prediction_mean": float(np.nanmean(p)),
        "prediction_median": float(np.nanmedian(p)),
        "mae": float(mean_absolute_error(y, p)),
        "baseline_mae": float(mean_absolute_error(y, baseline)),
        "mae_skill": float(1.0 - mean_absolute_error(y, p) / max(mean_absolute_error(y, baseline), 1e-12)),
        "rank_ic": rank_ic,
        "pearson": pearson,
        "decile_monotonicity": monotonicity,
        "top_decile_actual_mean": top_decile_mean,
        "top_decile_lift": top_decile_lift,
    }

    quantile_rows: list[dict[str, object]] = []
    valid_cal = np.isfinite(calibration_prediction) & np.isfinite(calibration_actual)
    cal_pred = calibration_prediction[valid_cal]
    cal_actual = calibration_actual[valid_cal]
    actual_q75 = float(np.nanquantile(cal_actual, 0.75))
    actual_q90 = float(np.nanquantile(cal_actual, 0.90))
    sample_parts: list[pd.DataFrame] = []
    for quantile in config.rank_quantiles:
        threshold = float(np.nanquantile(cal_pred, quantile))
        signal = p >= threshold
        values = y[signal]
        quantile_rows.append(
            {
                "fold_id": fold_id,
                "architecture": architecture,
                "target": target_name,
                "quantile": quantile,
                "threshold": threshold,
                "signals": int(signal.sum()),
                "signal_rate": float(np.mean(signal)),
                "actual_mean": float(np.nanmean(values)) if len(values) else np.nan,
                "actual_median": float(np.nanmedian(values)) if len(values) else np.nan,
                "mean_lift": float(np.nanmean(values) / overall_mean) if len(values) and overall_mean > 0 else np.nan,
                "actual_ge_cal_q75_rate": float(np.mean(values >= actual_q75)) if len(values) else np.nan,
                "actual_ge_cal_q90_rate": float(np.mean(values >= actual_q90)) if len(values) else np.nan,
            }
        )
        positions = np.flatnonzero(signal)
        if len(positions):
            sample_parts.append(
                pd.DataFrame(
                    {
                        "fold_id": fold_id,
                        "architecture": architecture,
                        "target": target_name,
                        "quantile": quantile,
                        "decision_time": pd.to_datetime(timestamps[positions]),
                        "prediction": p[positions],
                        "actual": y[positions],
                    }
                )
            )
    samples = pd.concat(sample_parts, ignore_index=True) if sample_parts else pd.DataFrame()
    return metrics, bucket_rows, quantile_rows, samples


def feature_importance(
    fitted: FittedIntensityModel,
    columns: tuple[str, ...],
    *,
    fold_id: str,
    architecture: str,
    target_name: str,
) -> list[dict[str, object]]:
    raw = fitted.model
    if not hasattr(raw, "feature_importances_"):
        return []
    values = np.asarray(raw.feature_importances_, dtype=float)
    return [
        {
            "fold_id": fold_id,
            "architecture": architecture,
            "target": target_name,
            "feature": feature,
            "importance": float(value),
        }
        for feature, value in zip(columns, values, strict=True)
    ]


def select_stable_intensity_candidates(
    metrics: pd.DataFrame,
    config: FutureIntensityConfig,
) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    keys = ["architecture", "target"]
    rows: list[dict[str, object]] = []
    for values, group in metrics.groupby(keys, sort=False):
        row: dict[str, object] = dict(zip(keys, values, strict=True))
        passed_folds = 0
        for fold_id in ("WF_2024", "WF_2025"):
            subset = group.loc[group["fold_id"] == fold_id]
            if subset.empty:
                continue
            item = subset.iloc[0]
            for column in (
                "rows",
                "rank_ic",
                "mae_skill",
                "decile_monotonicity",
                "top_decile_lift",
                "top_decile_actual_mean",
            ):
                row[f"{fold_id}_{column}"] = item[column]
            passed = (
                int(item["rows"]) >= config.minimum_test_rows
                and float(item["rank_ic"]) >= config.minimum_rank_ic
                and float(item["top_decile_lift"]) >= config.minimum_top_decile_lift
                and float(item["decile_monotonicity"]) >= config.minimum_decile_monotonicity
            )
            row[f"{fold_id}_passes"] = bool(passed)
            passed_folds += int(passed)
        row["passes"] = passed_folds == 2
        rank_values = [float(row.get(f"{fold}_rank_ic", np.nan)) for fold in ("WF_2024", "WF_2025")]
        lift_values = [float(row.get(f"{fold}_top_decile_lift", np.nan)) for fold in ("WF_2024", "WF_2025")]
        row["stability_score"] = float(np.nanmin(rank_values) + 0.20 * np.nanmin(lift_values))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["passes", "stability_score"],
        ascending=[False, False],
        kind="stable",
    ).reset_index(drop=True)
