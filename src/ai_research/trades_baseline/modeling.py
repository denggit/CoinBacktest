#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Walk-forward model fitting and prediction metrics for R01."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Protocol

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None  # type: ignore[assignment]

from .config import TradesBaselineConfig
from .dataset import feature_columns, load_month_shard


class Regressor(Protocol):
    def fit(self, x: np.ndarray, y: np.ndarray): ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: str
    fit_start: pd.Timestamp
    fit_end: pd.Timestamp
    calibration_start: pd.Timestamp
    calibration_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    sealed: bool = False

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, pd.Timestamp):
                payload[key] = str(value)
        return payload


@dataclass(frozen=True)
class PredictionMetrics:
    rows: int
    rmse: float
    mae: float
    pearson_ic: float
    directional_accuracy: float
    target_mean: float
    prediction_mean: float
    target_std: float
    prediction_std: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_folds(config: TradesBaselineConfig) -> tuple[WalkForwardFold, ...]:
    research_end = pd.Timestamp(config.research_end)
    embargo = pd.Timedelta(seconds=config.max_future_seconds)

    def safe_end(value: str) -> pd.Timestamp:
        return pd.Timestamp(value) - embargo

    return (
        WalkForwardFold(
            "WF_2024",
            pd.Timestamp("2023-01-01"),
            safe_end("2023-10-01"),
            pd.Timestamp("2023-10-01"),
            safe_end("2024-01-01"),
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-12-31 23:59:59"),
        ),
        WalkForwardFold(
            "WF_2025",
            pd.Timestamp("2023-01-01"),
            safe_end("2024-10-01"),
            pd.Timestamp("2024-10-01"),
            safe_end("2025-01-01"),
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-12-31 23:59:59"),
        ),
        WalkForwardFold(
            "WF_2026",
            pd.Timestamp("2023-01-01"),
            safe_end("2025-10-01"),
            pd.Timestamp("2025-10-01"),
            safe_end("2026-01-01"),
            pd.Timestamp("2026-01-01"),
            research_end,
            sealed=True,
        ),
    )


def _validate_shard_schema(shard, features: list[str], target_col: str) -> tuple[int, int]:
    if tuple(features) != shard.feature_names:
        raise RuntimeError(f"feature schema mismatch in {shard.path}")
    try:
        target_idx = shard.label_names.index(target_col)
    except ValueError as exc:
        raise RuntimeError(f"missing target {target_col} in {shard.path}") from exc
    return len(features), target_idx


def _count_rows(paths: Iterable[Path], start: pd.Timestamp, end: pd.Timestamp, target_col: str, features: list[str]) -> int:
    # Use the timestamp slice length for the sampling fraction. Invalid labels
    # are rare (mainly the very end of the dataset) and are filtered in the
    # actual collection pass. This avoids rereading a target column solely to
    # count rows for every fold/model/horizon.
    total = 0
    for path in paths:
        shard = load_month_shard(path)
        _validate_shard_schema(shard, features, target_col)
        pos = shard.positions(start, end)
        total += max(0, int(pos.stop or 0) - int(pos.start or 0))
    return total


def _mix_u64(values: np.ndarray, seed: int) -> np.ndarray:
    x = values.astype(np.uint64, copy=False) + np.uint64(seed)
    x ^= x >> np.uint64(30)
    x *= np.uint64(0xBF58476D1CE4E5B9)
    x ^= x >> np.uint64(27)
    x *= np.uint64(0x94D049BB133111EB)
    x ^= x >> np.uint64(31)
    return x


def collect_training_sample(
    paths: Iterable[Path],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    features: list[str],
    target_col: str,
    cap: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = list(paths)
    total = _count_rows(paths, start, end, target_col, features)
    if total <= 0:
        raise RuntimeError(f"no training rows for {start} -> {end} target={target_col}")
    fraction = min(1.0, (cap * 1.10) / total)
    threshold = np.uint64(min(2**64 - 1, int(fraction * (2**64 - 1))))
    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    t_parts: list[np.ndarray] = []
    for path in paths:
        shard = load_month_shard(path)
        _, target_idx = _validate_shard_schema(shard, features, target_col)
        pos = shard.positions(start, end)
        x_view = shard.features[pos]
        y_view = shard.labels[pos, target_idx]
        ts_view = shard.timestamps_ns[pos]
        valid = np.isfinite(y_view) & np.isfinite(x_view).all(axis=1)
        if not valid.any():
            continue
        x = np.asarray(x_view[valid], dtype=np.float32)
        y = np.asarray(y_view[valid], dtype=np.float32)
        ts = np.asarray(ts_view[valid], dtype=np.int64)
        if fraction < 1.0:
            keep = _mix_u64(ts.astype(np.uint64), seed) <= threshold
            x, y, ts = x[keep], y[keep], ts[keep]
        if len(y):
            x_parts.append(x)
            y_parts.append(y)
            t_parts.append(ts)
    if not x_parts:
        raise RuntimeError("deterministic sample produced zero rows")
    x = np.concatenate(x_parts)
    y = np.concatenate(y_parts)
    times = np.concatenate(t_parts)
    if len(y) > cap:
        order = np.argsort(_mix_u64(times.astype(np.uint64), seed + 17), kind="stable")[:cap]
        x, y, times = x[order], y[order], times[order]
    return x, y, times


def validate_model_dependencies(model_names: Iterable[str]) -> dict[str, str]:
    """Validate requested model backends before any data/cache work starts.

    R01 may spend minutes building reusable monthly shards. Optional model
    dependencies therefore must fail fast, before public-loader preflight or
    cache construction.
    """
    requested = tuple(dict.fromkeys(model_names))
    unknown = sorted(set(requested) - {"ridge", "lightgbm"})
    if unknown:
        raise ValueError(f"unknown R01 models: {unknown}")

    status = {"ridge": "available"}
    if "lightgbm" in requested:
        if LGBMRegressor is None:
            raise RuntimeError(
                "R01 startup dependency check failed: requested model 'lightgbm' "
                "is not installed in the active Python environment.\n"
                "Install it with:\n"
                "  python -m pip install lightgbm\n"
                "Then rerun the same R01 command. Existing monthly R01 cache "
                "is reusable automatically; do not pass --force-rebuild-cache.\n"
                "For a Ridge-only diagnostic run, use:\n"
                "  python research\\eth_ai_trading\\01_trades_only_supervised_baseline.py --models ridge"
            )
        status["lightgbm"] = "available"
    return {name: status[name] for name in requested}


def create_model(model_name: str, config: TradesBaselineConfig) -> Regressor:
    if model_name == "ridge":
        return Pipeline([("scale", StandardScaler()), ("ridge", Ridge(alpha=10.0))])
    if model_name == "lightgbm":
        if LGBMRegressor is None:
            raise RuntimeError("lightgbm is required for the R01 tree baseline")
        return LGBMRegressor(
            objective="regression_l1",
            n_estimators=config.model_n_estimators,
            learning_rate=config.model_learning_rate,
            num_leaves=config.model_num_leaves,
            min_child_samples=config.model_min_child_samples,
            colsample_bytree=config.model_feature_fraction,
            subsample=0.85,
            subsample_freq=1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=config.random_seed,
            n_jobs=-1,
            verbosity=-1,
        )
    raise ValueError(f"unknown model: {model_name}")


def predict_model(model: Regressor, x: np.ndarray) -> np.ndarray:
    """Predict without sklearn feature-name warning on LightGBM's wrapper."""
    booster = getattr(model, "booster_", None)
    if booster is not None:
        return np.asarray(booster.predict(x), dtype=float)
    return np.asarray(model.predict(x), dtype=float)


def fit_model(
    model_name: str,
    paths: Iterable[Path],
    fold: WalkForwardFold,
    horizon: int,
    config: TradesBaselineConfig,
) -> tuple[Regressor, dict[str, object]]:
    features = feature_columns(config)
    target_col = f"gross_ret_h{horizon}_lat{int(config.base_latency_seconds * 1000)}"
    cap = config.linear_sample_cap if model_name == "ridge" else config.train_sample_cap
    x, y, times = collect_training_sample(
        paths,
        start=fold.fit_start,
        end=fold.fit_end,
        features=features,
        target_col=target_col,
        cap=cap,
        seed=config.random_seed + horizon,
    )
    model = create_model(model_name, config)
    model.fit(x, y)
    metadata: dict[str, object] = {
        "model": model_name,
        "fold": fold.to_dict(),
        "horizon_seconds": horizon,
        "target_column": target_col,
        "feature_columns": features,
        "train_rows": int(len(y)),
        "train_start_actual": str(pd.to_datetime(times.min())),
        "train_end_actual": str(pd.to_datetime(times.max())),
        "target_mean": float(np.mean(y)),
        "target_std": float(np.std(y)),
    }
    return model, metadata


def calibration_thresholds(
    model: Regressor,
    paths: Iterable[Path],
    fold: WalkForwardFold,
    horizon: int,
    config: TradesBaselineConfig,
) -> dict[str, float]:
    features = feature_columns(config)
    target_col = f"gross_ret_h{horizon}_lat{int(config.base_latency_seconds * 1000)}"
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for path in paths:
        shard = load_month_shard(path)
        _, target_idx = _validate_shard_schema(shard, features, target_col)
        pos = shard.positions(fold.calibration_start, fold.calibration_end)
        x_view = shard.features[pos]
        y_view = shard.labels[pos, target_idx]
        valid = np.isfinite(y_view) & np.isfinite(x_view).all(axis=1)
        if not valid.any():
            continue
        x = np.asarray(x_view[valid], dtype=np.float32)
        y = np.asarray(y_view[valid], dtype=float)
        preds.append(predict_model(model, x))
        targets.append(y)
    if not preds:
        raise RuntimeError(f"no calibration predictions for {fold.fold_id}")
    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    thresholds: dict[str, float] = {}
    for q in config.signal_quantiles:
        long_threshold = float(np.quantile(pred, q))
        short_threshold = float(np.quantile(pred, 1.0 - q))
        long_mask = pred >= long_threshold
        short_mask = pred <= short_threshold
        thresholds[f"q{q:.3f}_long"] = long_threshold
        thresholds[f"q{q:.3f}_short"] = short_threshold
        thresholds[f"q{q:.3f}_long_expected_gross"] = float(np.mean(target[long_mask])) if long_mask.any() else float("nan")
        thresholds[f"q{q:.3f}_short_expected_gross"] = float(np.mean(-target[short_mask])) if short_mask.any() else float("nan")
        thresholds[f"q{q:.3f}_long_rows"] = float(long_mask.sum())
        thresholds[f"q{q:.3f}_short_rows"] = float(short_mask.sum())
    thresholds["prediction_mean"] = float(np.mean(pred))
    thresholds["prediction_std"] = float(np.std(pred))
    thresholds["target_mean"] = float(np.mean(target))
    thresholds["rows"] = float(len(pred))
    return thresholds


class OnlineRegressionMetrics:
    def __init__(self) -> None:
        self.n = 0
        self.sum_y = self.sum_p = self.sum_y2 = self.sum_p2 = self.sum_yp = 0.0
        self.sum_sq_err = self.sum_abs_err = 0.0
        self.sign_correct = 0

    def update(self, y: np.ndarray, p: np.ndarray) -> None:
        y = np.asarray(y, dtype=float)
        p = np.asarray(p, dtype=float)
        valid = np.isfinite(y) & np.isfinite(p)
        y, p = y[valid], p[valid]
        if len(y) == 0:
            return
        self.n += len(y)
        self.sum_y += float(y.sum())
        self.sum_p += float(p.sum())
        self.sum_y2 += float(np.dot(y, y))
        self.sum_p2 += float(np.dot(p, p))
        self.sum_yp += float(np.dot(y, p))
        err = p - y
        self.sum_sq_err += float(np.dot(err, err))
        self.sum_abs_err += float(np.abs(err).sum())
        self.sign_correct += int((np.sign(y) == np.sign(p)).sum())

    def finalize(self) -> PredictionMetrics:
        if self.n == 0:
            return PredictionMetrics(0, *(math.nan for _ in range(8)))
        n = float(self.n)
        cov = self.sum_yp - self.sum_y * self.sum_p / n
        var_y = self.sum_y2 - self.sum_y * self.sum_y / n
        var_p = self.sum_p2 - self.sum_p * self.sum_p / n
        corr = cov / math.sqrt(var_y * var_p) if var_y > 0 and var_p > 0 else 0.0
        return PredictionMetrics(
            rows=self.n,
            rmse=math.sqrt(self.sum_sq_err / n),
            mae=self.sum_abs_err / n,
            pearson_ic=corr,
            directional_accuracy=self.sign_correct / n,
            target_mean=self.sum_y / n,
            prediction_mean=self.sum_p / n,
            target_std=math.sqrt(max(0.0, var_y / n)),
            prediction_std=math.sqrt(max(0.0, var_p / n)),
        )


def save_model_bundle(model: Regressor, metadata: dict[str, object], thresholds: dict[str, float], target_dir: Path) -> dict[str, Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    model_path = target_dir / "model.joblib"
    metadata_path = target_dir / "metadata.json"
    joblib.dump(model, model_path)
    payload = dict(metadata)
    payload["thresholds"] = thresholds
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"model": model_path, "metadata": metadata_path}


def feature_importance_frame(model: Regressor, features: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(getattr(model, "feature_importances_"), dtype=float)
    elif isinstance(model, Pipeline) and hasattr(model[-1], "coef_"):
        values = np.abs(np.asarray(model[-1].coef_, dtype=float)).reshape(-1)
    else:
        values = np.zeros(len(features), dtype=float)
    return pd.DataFrame({"feature": features, "importance": values}).sort_values(
        "importance", ascending=False, kind="stable"
    )


def fit_model_set(
    model_names: tuple[str, ...],
    paths: Iterable[Path],
    fold: WalkForwardFold,
    horizon: int,
    config: TradesBaselineConfig,
) -> dict[str, tuple[Regressor, dict[str, object]]]:
    """Fit all requested simple baselines from one shared deterministic sample."""
    features = feature_columns(config)
    target_col = f"gross_ret_h{horizon}_lat{int(config.base_latency_seconds * 1000)}"
    requested_cap = max(
        config.linear_sample_cap if name == "ridge" else config.train_sample_cap
        for name in model_names
    )
    x, y, times = collect_training_sample(
        paths,
        start=fold.fit_start,
        end=fold.fit_end,
        features=features,
        target_col=target_col,
        cap=requested_cap,
        seed=config.random_seed + horizon,
    )
    outputs: dict[str, tuple[Regressor, dict[str, object]]] = {}
    for model_name in model_names:
        cap = config.linear_sample_cap if model_name == "ridge" else config.train_sample_cap
        if len(y) > cap:
            order = np.argsort(
                _mix_u64(times.astype(np.uint64), config.random_seed + horizon + 101),
                kind="stable",
            )[:cap]
            x_fit, y_fit, times_fit = x[order], y[order], times[order]
        else:
            x_fit, y_fit, times_fit = x, y, times
        model = create_model(model_name, config)
        model.fit(x_fit, y_fit)
        metadata: dict[str, object] = {
            "model": model_name,
            "fold": fold.to_dict(),
            "horizon_seconds": horizon,
            "target_column": target_col,
            "feature_columns": features,
            "train_rows": int(len(y_fit)),
            "shared_sample_rows": int(len(y)),
            "train_start_actual": str(pd.to_datetime(times_fit.min())),
            "train_end_actual": str(pd.to_datetime(times_fit.max())),
            "target_mean": float(np.mean(y_fit)),
            "target_std": float(np.std(y_fit)),
        }
        outputs[model_name] = (model, metadata)
    return outputs
