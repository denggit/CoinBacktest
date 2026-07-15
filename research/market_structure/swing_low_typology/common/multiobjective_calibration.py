#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Calibration, hierarchical policy and region diagnostics for research 08.

All helpers are research-only.  Calibration is fitted strictly inside an
expanding walk-forward training interval and no future path value is used as a
model feature.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from scipy.linalg import LinAlgWarning
from scipy.special import expit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import brier_score_loss, log_loss

from research.market_structure.swing_low_typology.common.online_recognizability import (
    prepare_feature_matrix,
)
from research.market_structure.swing_low_typology.common.reversal_opportunity import (
    opportunity_event_metrics,
)
from research.market_structure.swing_low_typology.common.walkforward_reversal import (
    concentration_metrics,
    positive_episode_coverage,
    remove_strongest_days,
)

EPS = 1e-12


@dataclass(frozen=True)
class BinaryProbabilityCalibrator:
    method: str
    model: object | None = None
    constant: float | None = None

    def transform(self, probability: Sequence[float]) -> np.ndarray:
        raw = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
        if self.method == "identity":
            return raw
        if self.method == "constant":
            value = float(self.constant if self.constant is not None else 0.0)
            return np.full(raw.shape, np.clip(value, 0.0, 1.0), dtype=float)
        if self.method == "sigmoid":
            logit = np.log(raw / (1.0 - raw)).reshape(-1, 1)
            return np.clip(np.asarray(self.model.predict_proba(logit)[:, 1], dtype=float), 0.0, 1.0)
        if self.method == "isotonic":
            return np.clip(np.asarray(self.model.predict(raw), dtype=float), 0.0, 1.0)
        raise ValueError(f"unknown calibration method: {self.method}")


@dataclass(frozen=True)
class BinaryScoreProbabilityCalibrator:
    """Map an unsquashed binary decision score to a probability."""

    method: str
    model: object | None = None
    constant: float | None = None

    def transform(self, score: Sequence[float]) -> np.ndarray:
        raw = np.asarray(score, dtype=float)
        if self.method == "identity":
            return np.clip(expit(raw), 0.0, 1.0)
        if self.method == "constant":
            value = float(self.constant if self.constant is not None else 0.0)
            return np.full(raw.shape, np.clip(value, 0.0, 1.0), dtype=float)
        if self.method == "sigmoid":
            return np.clip(np.asarray(self.model.predict_proba(raw.reshape(-1, 1))[:, 1], dtype=float), 0.0, 1.0)
        if self.method == "isotonic":
            return np.clip(np.asarray(self.model.predict(raw), dtype=float), 0.0, 1.0)
        raise ValueError(f"unknown score calibration method: {self.method}")


def fit_score_probability_calibrators(
    raw_score: Sequence[float],
    y_true: Sequence[bool | int],
) -> dict[str, BinaryScoreProbabilityCalibrator]:
    """Fit probability calibrators directly on decision scores.

    This preserves ranking information even when the classifier's sigmoid
    probability has saturated to exact 0/1.  All fitting data must come from
    the walk-forward development interval.
    """

    raw = np.asarray(raw_score, dtype=float)
    y = np.asarray(y_true, dtype=int)
    valid = np.isfinite(raw) & np.isfinite(y)
    raw = raw[valid]
    y = y[valid]
    if len(raw) == 0 or np.unique(y).size < 2 or np.unique(raw).size < 2:
        constant = float(y.mean()) if len(y) else 0.0
        return {
            "identity": BinaryScoreProbabilityCalibrator("identity"),
            "sigmoid": BinaryScoreProbabilityCalibrator("constant", constant=constant),
            "isotonic": BinaryScoreProbabilityCalibrator("constant", constant=constant),
        }

    sigmoid = Pipeline(
        [
            ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
            (
                "model",
                LogisticRegression(
                    C=1.0,
                    class_weight=None,
                    solver="lbfgs",
                    max_iter=1_000,
                    tol=1e-6,
                    random_state=42,
                ),
            ),
        ]
    )
    sigmoid.fit(raw.reshape(-1, 1), y)
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(raw, y)
    return {
        "identity": BinaryScoreProbabilityCalibrator("identity"),
        "sigmoid": BinaryScoreProbabilityCalibrator("sigmoid", model=sigmoid),
        "isotonic": BinaryScoreProbabilityCalibrator("isotonic", model=isotonic),
    }


@dataclass(frozen=True)
class QuantileRiskModel:
    feature_columns: tuple[str, ...]
    medians: pd.Series
    quantile: float
    target_column: str
    model: object | None
    constant: float
    success_only: bool

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(frame), self.constant, dtype=float)
        x = prepare_feature_matrix(frame, self.feature_columns, self.medians)
        return np.maximum(0.0, np.asarray(self.model.predict(x), dtype=float))




@dataclass(frozen=True)
class RiskPointModel:
    feature_columns: tuple[str, ...]
    medians: pd.Series
    target_column: str
    model: object | None
    constant: float
    success_only: bool
    fit_method: str = "constant"
    converged: bool = True
    iterations: int = 0

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            return np.full(len(frame), self.constant, dtype=float)
        x = prepare_feature_matrix(frame, self.feature_columns, self.medians)
        return np.maximum(0.0, np.asarray(self.model.predict(x), dtype=float))


@dataclass(frozen=True)
class ConformalQuantileAdjustment:
    quantile: float
    additive_shift: float
    calibration_count: int

    def apply(self, prediction: Sequence[float]) -> np.ndarray:
        values = np.asarray(prediction, dtype=float)
        return np.maximum(0.0, values + float(self.additive_shift))


def adaptive_ece(y_true: Sequence[bool | int], probability: Sequence[float], bins: int = 10) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]
    if y.size == 0:
        return np.nan
    order = np.argsort(p, kind="mergesort")
    chunks = np.array_split(order, min(max(1, int(bins)), len(order)))
    total = float(len(order))
    return float(
        sum((len(chunk) / total) * abs(float(y[chunk].mean()) - float(p[chunk].mean())) for chunk in chunks if len(chunk))
    )


def calibration_metrics(y_true: Sequence[bool | int], probability: Sequence[float]) -> dict[str, float | int]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    valid = np.isfinite(p)
    y = y[valid]
    p = p[valid]
    if y.size == 0:
        return {"count": 0, "positive_count": 0, "base_rate": np.nan, "brier": np.nan, "log_loss": np.nan, "ece": np.nan}
    return {
        "count": int(len(y)),
        "positive_count": int(y.sum()),
        "base_rate": float(y.mean()),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": adaptive_ece(y, p),
    }


def fit_probability_calibrators(
    raw_probability: Sequence[float],
    y_true: Sequence[bool | int],
) -> dict[str, BinaryProbabilityCalibrator]:
    raw = np.clip(np.asarray(raw_probability, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y_true, dtype=int)
    valid = np.isfinite(raw) & np.isfinite(y)
    raw = raw[valid]
    y = y[valid]
    if len(raw) == 0 or np.unique(y).size < 2:
        constant = float(y.mean()) if len(y) else 0.0
        return {
            "identity": BinaryProbabilityCalibrator("identity"),
            "sigmoid": BinaryProbabilityCalibrator("constant", constant=constant),
            "isotonic": BinaryProbabilityCalibrator("constant", constant=constant),
        }

    logit = np.log(raw / (1.0 - raw)).reshape(-1, 1)
    sigmoid = LogisticRegression(
        C=1.0,
        class_weight=None,
        solver="lbfgs",
        max_iter=1_000,
        tol=1e-6,
        random_state=42,
    )
    sigmoid.fit(logit, y)
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(raw, y)
    return {
        "identity": BinaryProbabilityCalibrator("identity"),
        "sigmoid": BinaryProbabilityCalibrator("sigmoid", model=sigmoid),
        "isotonic": BinaryProbabilityCalibrator("isotonic", model=isotonic),
    }


def choose_calibrator(
    calibrators: dict[str, BinaryProbabilityCalibrator],
    raw_probability: Sequence[float],
    y_true: Sequence[bool | int],
) -> tuple[str, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for method, calibrator in calibrators.items():
        prediction = calibrator.transform(raw_probability)
        metrics = calibration_metrics(y_true, prediction)
        rows.append({"method": method, **metrics})
    table = pd.DataFrame(rows)
    if table.empty:
        raise RuntimeError("no calibration method evaluated")
    # Brier is primary; ECE and log-loss are deterministic tie-breakers.
    ranking = table.sort_values(["brier", "ece", "log_loss", "method"], na_position="last")
    return str(ranking.iloc[0]["method"]), table


def fit_risk_point_model(
    train: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    success_only: bool,
    weight_column: str = "episode_weight",
) -> RiskPointModel:
    """Fast stable risk center model; q50/q90 come from split conformal residuals."""

    source = train[train["tp_hit_1pct"].astype(bool)].copy() if success_only else train.copy()
    target = pd.to_numeric(source[target_column], errors="coerce")
    valid = target.notna() & np.isfinite(target)
    source = source.loc[valid].copy()
    target = target.loc[valid].clip(0.0, 10.0)
    constant = float(target.median()) if len(target) else 1.0
    if not np.isfinite(constant):
        constant = 1.0
    medians = pd.Series(0.0, index=list(feature_columns), dtype=float)
    if len(source) < 100 or target.nunique() < 5:
        return RiskPointModel(tuple(feature_columns), medians, target_column, None, constant, success_only, fit_method="constant", converged=True, iterations=0)
    x_raw = source.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = x_raw.median().fillna(0.0)
    x = x_raw.fillna(medians)
    weight = (
        pd.to_numeric(source[weight_column], errors="coerce").fillna(1.0).clip(lower=0.0).to_numpy(dtype=float)
        if weight_column in source.columns
        else np.ones(len(source), dtype=float)
    )
    positive = np.isfinite(weight) & (weight > 0.0)
    if positive.any():
        weight = np.where(positive, weight / max(float(weight[positive].mean()), EPS), 0.0)
    else:
        weight = np.ones(len(source), dtype=float)

    # LSQR avoids the direct normal-equation solve that emits LinAlgWarning on
    # strongly collinear range/footprint features.  Formal research must never
    # accept a warning and silently continue, so warnings/non-finite predictions
    # trigger one deterministic shallow-tree fallback and are recorded.
    ridge = Pipeline(
        [
            ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
            (
                "model",
                Ridge(
                    alpha=24.0,
                    solver="lsqr",
                    max_iter=5_000,
                    tol=1e-5,
                ),
            ),
        ]
    )
    fit_method = "ridge_lsqr"
    converged = True
    iterations = 0
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        warnings.simplefilter("always", LinAlgWarning)
        ridge.fit(x, target.to_numpy(dtype=float), model__sample_weight=weight)
    unstable = [
        item for item in caught
        if issubclass(item.category, (ConvergenceWarning, LinAlgWarning))
    ]
    ridge_model = ridge.named_steps["model"]
    n_iter = getattr(ridge_model, "n_iter_", None)
    if n_iter is not None:
        values = np.atleast_1d(n_iter).astype(float)
        finite_values = values[np.isfinite(values)]
        iterations = int(finite_values.max()) if finite_values.size else 0
        if iterations >= 5_000:
            converged = False
    probe = np.asarray(ridge.predict(x.iloc[: min(512, len(x))]), dtype=float)
    if unstable or not np.isfinite(probe).all() or not converged:
        fallback = HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=100,
            max_leaf_nodes=15,
            max_depth=3,
            min_samples_leaf=100,
            l2_regularization=4.0,
            loss="squared_error",
            early_stopping=True,
            validation_fraction=0.12,
            random_state=42,
        )
        fallback.fit(x, target.to_numpy(dtype=float), sample_weight=weight)
        probe = np.asarray(fallback.predict(x.iloc[: min(512, len(x))]), dtype=float)
        if not np.isfinite(probe).all():
            raise RuntimeError(f"risk target {target_column} produced non-finite predictions after stable fallback")
        model: object = fallback
        fit_method = "hist_gbdt_risk_fallback"
        converged = True
        iterations = int(getattr(fallback, "n_iter_", 0) or 0)
    else:
        model = ridge

    return RiskPointModel(
        tuple(feature_columns),
        medians,
        target_column,
        model,
        constant,
        success_only,
        fit_method=fit_method,
        converged=converged,
        iterations=iterations,
    )


def fit_quantile_model(
    train: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    quantile: float,
    success_only: bool,
    random_state: int = 42,
    min_samples_leaf: int = 100,
    weight_column: str = "episode_weight",
) -> QuantileRiskModel:
    if not 0.0 < float(quantile) < 1.0:
        raise ValueError("quantile must be between zero and one")
    source = train[train["tp_hit_1pct"].astype(bool)].copy() if success_only else train.copy()
    target = pd.to_numeric(source[target_column], errors="coerce")
    valid = target.notna() & np.isfinite(target)
    source = source.loc[valid].copy()
    target = target.loc[valid].clip(0.0, 10.0)
    constant = float(target.quantile(float(quantile))) if len(target) else 1.0
    if not np.isfinite(constant):
        constant = 1.0
    medians = pd.Series(0.0, index=list(feature_columns), dtype=float)
    minimum_rows = max(100, int(min_samples_leaf) * 2)
    if len(source) < minimum_rows or target.nunique() < 5:
        return QuantileRiskModel(tuple(feature_columns), medians, float(quantile), target_column, None, constant, success_only)

    x_raw = source.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = x_raw.median().fillna(0.0)
    x = x_raw.fillna(medians)
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=float(quantile),
        learning_rate=0.05,
        max_iter=120,
        max_leaf_nodes=15,
        max_depth=3,
        min_samples_leaf=max(20, int(min_samples_leaf)),
        l2_regularization=2.0,
        early_stopping=True,
        validation_fraction=0.12,
        random_state=int(random_state),
    )
    weight = (
        pd.to_numeric(source[weight_column], errors="coerce").fillna(1.0).clip(lower=0.0).to_numpy(dtype=float)
        if weight_column in source.columns
        else np.ones(len(source), dtype=float)
    )
    positive = np.isfinite(weight) & (weight > 0.0)
    if positive.any():
        weight = np.where(positive, weight / max(float(weight[positive].mean()), EPS), 0.0)
    else:
        weight = np.ones(len(source), dtype=float)
    model.fit(x, target.to_numpy(dtype=float), sample_weight=weight)
    return QuantileRiskModel(tuple(feature_columns), medians, float(quantile), target_column, model, constant, success_only)


def fit_conformal_adjustment(
    actual: Sequence[float],
    prediction: Sequence[float],
    *,
    quantile: float,
) -> ConformalQuantileAdjustment:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(prediction, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    residual = y[valid] - p[valid]
    if residual.size == 0:
        return ConformalQuantileAdjustment(float(quantile), 0.0, 0)
    shift = float(np.quantile(residual, float(quantile), method="higher"))
    return ConformalQuantileAdjustment(float(quantile), shift, int(residual.size))


def quantile_metrics(actual: Sequence[float], prediction: Sequence[float]) -> dict[str, float | int]:
    y = np.asarray(actual, dtype=float)
    p = np.asarray(prediction, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]
    if y.size == 0:
        return {"count": 0, "mae": np.nan, "coverage": np.nan, "median_actual": np.nan, "median_prediction": np.nan}
    return {
        "count": int(len(y)),
        "mae": float(np.mean(np.abs(y - p))),
        "coverage": float(np.mean(y <= p)),
        "median_actual": float(np.median(y)),
        "median_prediction": float(np.median(p)),
    }


def hierarchical_policy_specs() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for tp_fraction in (0.02, 0.05, 0.10, 0.20):
        for fast_fraction in (None, 0.50, 0.25):
            for clean_fraction in (None, 0.50, 0.25):
                for risk_fraction in (None, 0.75, 0.50):
                    # A pure TP baseline and hierarchical combinations are kept;
                    # a policy with no secondary gate but only a risk percentile is
                    # still useful as a direct MAE-risk control.
                    rows.append(
                        {
                            "policy_id": f"TP{int(tp_fraction*100):02d}_F{('NA' if fast_fraction is None else int(fast_fraction*100))}_C{('NA' if clean_fraction is None else int(clean_fraction*100))}_R{('NA' if risk_fraction is None else int(risk_fraction*100))}",
                            "tp_top_fraction": tp_fraction,
                            "fast30_top_fraction": fast_fraction,
                            "clean50_top_fraction": clean_fraction,
                            "horizon_mae_q90_bottom_fraction": risk_fraction,
                        }
                    )
    return pd.DataFrame(rows).drop_duplicates("policy_id").reset_index(drop=True)


def policy_thresholds(reference: pd.DataFrame, spec: pd.Series) -> dict[str, float]:
    thresholds = {
        "tp_threshold": float(pd.to_numeric(reference["p_tp60_cal"], errors="coerce").quantile(1.0 - float(spec["tp_top_fraction"]))),
        "fast30_threshold": np.nan,
        "clean50_threshold": np.nan,
        "horizon_mae_q90_threshold": np.nan,
    }
    if pd.notna(spec["fast30_top_fraction"]):
        thresholds["fast30_threshold"] = float(pd.to_numeric(reference["p_fast30_cal"], errors="coerce").quantile(1.0 - float(spec["fast30_top_fraction"])))
    if pd.notna(spec["clean50_top_fraction"]):
        thresholds["clean50_threshold"] = float(pd.to_numeric(reference["p_clean50_cal"], errors="coerce").quantile(1.0 - float(spec["clean50_top_fraction"])))
    if pd.notna(spec["horizon_mae_q90_bottom_fraction"]):
        thresholds["horizon_mae_q90_threshold"] = float(pd.to_numeric(reference["mae_horizon_q90_cal"], errors="coerce").quantile(float(spec["horizon_mae_q90_bottom_fraction"])))
    return thresholds


def select_hierarchical_events(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    *,
    cooldown_bars: int,
) -> pd.DataFrame:
    eligible = frame[pd.to_numeric(frame["p_tp60_cal"], errors="coerce") >= float(thresholds["tp_threshold"])].copy()
    if np.isfinite(thresholds.get("fast30_threshold", np.nan)):
        eligible = eligible[pd.to_numeric(eligible["p_fast30_cal"], errors="coerce") >= float(thresholds["fast30_threshold"])]
    if np.isfinite(thresholds.get("clean50_threshold", np.nan)):
        eligible = eligible[pd.to_numeric(eligible["p_clean50_cal"], errors="coerce") >= float(thresholds["clean50_threshold"])]
    if np.isfinite(thresholds.get("horizon_mae_q90_threshold", np.nan)):
        eligible = eligible[pd.to_numeric(eligible["mae_horizon_q90_cal"], errors="coerce") <= float(thresholds["horizon_mae_q90_threshold"])]
    if eligible.empty:
        return eligible
    eligible = eligible.sort_values(["extreme_pos", "event_id"]).drop_duplicates("causal_region_id", keep="first")
    if int(cooldown_bars) > 0:
        chosen: list[int] = []
        last_position = -10**18
        for row_index, position in zip(eligible.index, pd.to_numeric(eligible["extreme_pos"], errors="raise").astype(int)):
            if int(position) - last_position < int(cooldown_bars):
                continue
            chosen.append(int(row_index))
            last_position = int(position)
        eligible = eligible.loc[chosen]
    return eligible.reset_index(drop=True)


def policy_metrics(
    events: pd.DataFrame,
    population: pd.DataFrame,
    *,
    months: int,
) -> dict[str, float | int]:
    base = opportunity_event_metrics(events)
    coverage = positive_episode_coverage(events, population)
    concentration = concentration_metrics(events)
    result: dict[str, float | int] = {
        "events_per_month": float(len(events) / max(1, int(months))),
        **base,
        **coverage,
        **concentration,
    }
    if events.empty:
        result.update(
            {
                "fast15_rate": np.nan,
                "fast30_rate": np.nan,
                "median_horizon_mae_pct": np.nan,
                "p90_horizon_mae_pct": np.nan,
                "median_success_mae_before_tp_pct": np.nan,
            }
        )
    else:
        result.update(
            {
                "fast15_rate": float(events["tp_within_15"].astype(bool).mean()),
                "fast30_rate": float(events["tp_within_30"].astype(bool).mean()),
                "median_horizon_mae_pct": float(pd.to_numeric(events["mae_horizon_pct"], errors="coerce").median()),
                "p90_horizon_mae_pct": float(pd.to_numeric(events["mae_horizon_pct"], errors="coerce").quantile(0.90)),
                "median_success_mae_before_tp_pct": float(pd.to_numeric(events.loc[events["tp_hit_1pct"].astype(bool), "mae_before_tp_pct"], errors="coerce").median()),
            }
        )
    return result


def pareto_policy_ids(
    policy_table: pd.DataFrame,
    *,
    minimum_events: int,
    maximum_policies: int = 20,
) -> tuple[str, ...]:
    source = policy_table[pd.to_numeric(policy_table["event_count"], errors="coerce") >= int(minimum_events)].copy()
    if source.empty:
        source = policy_table.nlargest(min(int(maximum_policies), len(policy_table)), "event_count").copy()
    metrics = source[["tp_rate", "clean_0p50_rate", "fast30_rate"]].apply(pd.to_numeric, errors="coerce").fillna(-np.inf).to_numpy(dtype=float)
    risk = pd.to_numeric(source["median_horizon_mae_pct"], errors="coerce").fillna(np.inf).to_numpy(dtype=float)
    keep = np.ones(len(source), dtype=bool)
    for i in range(len(source)):
        if not keep[i]:
            continue
        dominates = (
            (metrics[:, 0] >= metrics[i, 0])
            & (metrics[:, 1] >= metrics[i, 1])
            & (metrics[:, 2] >= metrics[i, 2])
            & (risk <= risk[i])
            & (
                (metrics[:, 0] > metrics[i, 0])
                | (metrics[:, 1] > metrics[i, 1])
                | (metrics[:, 2] > metrics[i, 2])
                | (risk < risk[i])
            )
        )
        dominates[i] = False
        if dominates.any():
            keep[i] = False
    frontier = source.loc[keep].copy()
    if len(frontier) > int(maximum_policies):
        # Only a deterministic reporting cap; no test data is involved.
        frontier["policy_rank_score"] = (
            frontier["tp_rate"].rank(pct=True)
            + frontier["clean_0p50_rate"].rank(pct=True)
            + frontier["fast30_rate"].rank(pct=True)
            + (-frontier["median_horizon_mae_pct"]).rank(pct=True)
        )
        frontier = frontier.nlargest(int(maximum_policies), "policy_rank_score")
    return tuple(frontier["policy_id"].astype(str))


def delete_day_stress(events: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for removed in (0, 5, 10):
        reduced = events if removed == 0 else remove_strongest_days(events, removed)
        rows.append({"removed_strongest_days": int(removed), **opportunity_event_metrics(reduced)})
    return pd.DataFrame(rows)


def build_region_geometry(bars: pd.DataFrame, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrospective region geometry for diagnostics only.

    The outputs must never be merged into model features. Primary price width
    and continuation risk use closed-bar closes; structural low/high width is
    reported separately for interpretation. Event diagnostics are filled in
    vectorized region blocks rather than one Python object per candidate.
    """

    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    open_ = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    time_index = bars.index
    data = frame.sort_values(["causal_region_id", "extreme_pos", "event_id"]).reset_index(drop=True)
    n = len(data)
    bars_to_low = np.full(n, 0, dtype=np.int32)
    low_after = np.zeros(n, dtype=bool)
    adverse_to_end = np.full(n, np.nan, dtype=np.float32)
    position_inside = np.full(n, np.nan, dtype=np.float32)
    region_rows: list[dict[str, object]] = []

    for region_id, group in data.groupby("causal_region_id", sort=False):
        row_index = group.index.to_numpy(dtype=np.int64)
        positions = pd.to_numeric(group["extreme_pos"], errors="raise").astype(int).to_numpy()
        start = int(positions.min())
        end = int(positions.max())
        segment = slice(start, end + 1)
        close_seg = close[segment]
        high_seg = high[segment]
        low_seg = low[segment]
        if not np.isfinite(close_seg).any():
            continue
        local_min = int(np.nanargmin(close_seg))
        min_pos = start + local_min
        start_close = max(abs(float(close[start])), EPS)
        close_width = (float(np.nanmax(close_seg)) - float(np.nanmin(close_seg))) / start_close * 100.0
        structural_width = (float(np.nanmax(high_seg)) - float(np.nanmin(low_seg))) / start_close * 100.0
        region_rows.append(
            {
                "causal_region_id": region_id,
                "region_start_time": time_index[start],
                "region_end_time": time_index[end],
                "region_start_pos": start,
                "region_end_pos": end,
                "region_duration_bars": end - start,
                "region_candidate_state_count": int(len(group)),
                "region_close_width_pct": close_width,
                "region_structural_width_pct": structural_width,
                "region_final_close_low_time": time_index[min_pos],
                "region_final_close_low_pos": min_pos,
                "region_end_rebound_from_close_low_pct": (float(close[end]) / max(float(close[min_pos]), EPS) - 1.0) * 100.0,
            }
        )

        # Suffix close minimum provides each candidate's adverse continuation
        # through the already-defined retrospective region end in O(region span).
        safe_close = np.where(np.isfinite(close_seg), close_seg, np.inf)
        suffix_min = np.minimum.accumulate(safe_close[::-1])[::-1]
        entry_positions = np.minimum(positions + 1, len(bars) - 1)
        entry_offsets = np.minimum(entry_positions - start, len(close_seg) - 1)
        entry_prices = open_[entry_positions]
        adverse = np.maximum(0.0, (entry_prices - suffix_min[entry_offsets]) / np.maximum(np.abs(entry_prices), EPS) * 100.0)
        bars_to_low[row_index] = (min_pos - positions).astype(np.int32)
        low_after[row_index] = min_pos > positions
        adverse_to_end[row_index] = adverse.astype(np.float32)
        position_inside[row_index] = ((positions - start) / max(1, end - start)).astype(np.float32)

    event_rows = pd.DataFrame(
        {
            "event_id": data["event_id"].to_numpy(),
            "causal_region_id": data["causal_region_id"].to_numpy(),
            "bars_from_signal_to_region_close_low": bars_to_low,
            "region_close_low_after_signal": low_after,
            "close_adverse_to_region_end_pct": adverse_to_end,
            "signal_position_inside_region": position_inside,
        }
    )
    return pd.DataFrame(region_rows), event_rows


def geometry_summary(events: pd.DataFrame, group_columns: Iterable[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    columns = list(group_columns)
    grouped = [((), events)] if not columns else events.groupby(columns, dropna=False)
    for key, group in grouped:
        key_values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(columns, key_values))
        row.update(
            {
                "event_count": int(len(group)),
                "tp_rate": float(group["tp_hit_1pct"].astype(bool).mean()) if len(group) else np.nan,
                "median_region_duration_bars": float(pd.to_numeric(group["region_duration_bars"], errors="coerce").median()),
                "p90_region_duration_bars": float(pd.to_numeric(group["region_duration_bars"], errors="coerce").quantile(0.90)),
                "median_region_close_width_pct": float(pd.to_numeric(group["region_close_width_pct"], errors="coerce").median()),
                "p90_region_close_width_pct": float(pd.to_numeric(group["region_close_width_pct"], errors="coerce").quantile(0.90)),
                "median_bars_to_region_close_low": float(pd.to_numeric(group["bars_from_signal_to_region_close_low"], errors="coerce").median()),
                "share_region_low_after_signal": float(group["region_close_low_after_signal"].astype(bool).mean()),
                "median_close_adverse_to_region_end_pct": float(pd.to_numeric(group["close_adverse_to_region_end_pct"], errors="coerce").median()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)
