#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal online-candidate labels and frozen recognizability models.

This module is research infrastructure only.  It does not place trades or run a
portfolio backtest.  Candidate features end at the current closed bar; forward
bars are used only to create supervised research labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Iterable, Mapping, Sequence

import os

# Avoid pathological OpenMP/BLAS oversubscription on high-core research hosts.
# Users can override these values explicitly before launching Python.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge, SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

try:
    from src.research_common.progress import ProgressReporter
except Exception:  # pragma: no cover
    ProgressReporter = None  # type: ignore[assignment]

EPS = 1e-12

CLEAR_MECHANISMS: tuple[str, ...] = (
    "T2_staged_acceleration",
    "T3_sync_persistent_selling",
    "T4_price_cvd_divergence",
    "B2_compression",
    "B3_spring_false_breakdown",
    "B4_repeated_support_test",
)

FUTURE_LABEL_COLUMNS: frozenset[str] = frozenset(
    {
        "entry_time",
        "entry_price",
        "label_end_time",
        "forward_horizon_bars",
        "mfe_pct",
        "mae_pct",
        "terminal_return_pct",
        "tp_hit_1pct",
        "adverse_hit_1pct",
        "tp_first_touch_bar",
        "adverse_first_touch_bar",
        "same_bar_tp_adverse_flag",
        "mae_before_tp_pct",
        "mfe_only_score",
        "mfe_minus_mae_score",
        "tp_priority_score",
        "first_touch_score",
        "reference_mechanism_type",
        "historical_clear_swing_low",
        "joint_swing_tp_success",
        "mechanism_joint_success",
    }
)


@dataclass(frozen=True)
class CandidateGateConfig:
    lookback: int = 240
    horizon: int = 60
    new_low_window: int = 5
    near_floor_window: int = 60
    position_window: int = 120
    near_floor_tolerance_bp: float = 20.0
    max_position_in_range: float = 0.55


@dataclass(frozen=True)
class FittedBinaryModel:
    family: str
    feature_columns: tuple[str, ...]
    medians: pd.Series
    model: object

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        x = prepare_feature_matrix(frame, self.feature_columns, self.medians)
        values = self.model.predict_proba(x)
        return np.asarray(values[:, 1], dtype=float)

    def predict_score(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the unsquashed binary ranking score when available.

        ``predict_proba`` can numerically saturate at exactly 0/1 for large
        linear margins, collapsing many distinct observations into only a few
        probability levels.  Ranking and empirical-CDF policies must therefore
        use the estimator's decision function.  Probability calibration remains
        a separate operation.
        """

        x = prepare_feature_matrix(frame, self.feature_columns, self.medians)
        decision = getattr(self.model, "decision_function", None)
        if callable(decision):
            values = np.asarray(decision(x), dtype=float)
            if values.ndim == 2:
                if values.shape[1] != 2:
                    raise RuntimeError(f"binary decision_function returned shape={values.shape}")
                values = values[:, 1]
            return values.reshape(-1)

        probability = np.clip(np.asarray(self.model.predict_proba(x)[:, 1], dtype=float), 1e-15, 1.0 - 1e-15)
        return np.log(probability / (1.0 - probability))


@dataclass(frozen=True)
class FittedScoreModel:
    family: str
    feature_columns: tuple[str, ...]
    medians: pd.Series
    model: object

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = prepare_feature_matrix(frame, self.feature_columns, self.medians)
        return np.clip(np.asarray(self.model.predict(x), dtype=float), -100.0, 100.0)


def _infer_bar_delta(index: pd.DatetimeIndex) -> pd.Timedelta:
    diffs = index.to_series().diff().dropna()
    positive = diffs[diffs > pd.Timedelta(0)]
    return positive.median() if not positive.empty else pd.Timedelta(minutes=1)


def _rolling_prior_min(values: pd.Series, window: int) -> pd.Series:
    return values.shift(1).rolling(int(window), min_periods=int(window)).min()


def _rolling_prior_max(values: pd.Series, window: int) -> pd.Series:
    return values.shift(1).rolling(int(window), min_periods=int(window)).max()


def build_online_candidate_events(
    bars: pd.DataFrame,
    *,
    research_start: pd.Timestamp,
    research_end_exclusive: pd.Timestamp,
    config: CandidateGateConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a broad, causal low-like candidate universe.

    The gate intentionally uses only the current closed bar and older bars.  It
    is broad enough to retain new lows, tests near an existing floor, and lows
    sitting in the lower half of a wider range.  Future labels never decide
    whether a row enters the universe.
    """

    if config.lookback < max(config.position_window, config.near_floor_window):
        raise ValueError("lookback must cover all candidate-gate windows")
    if config.horizon < 1:
        raise ValueError("horizon must be >= 1")
    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(bars.columns))
    if missing:
        raise RuntimeError(f"candidate gate missing OHLC fields: {missing}")

    index = pd.DatetimeIndex(bars.index)
    low = pd.to_numeric(bars["low"], errors="coerce")
    high = pd.to_numeric(bars["high"], errors="coerce")
    close = pd.to_numeric(bars["close"], errors="coerce")

    prior_new_low = _rolling_prior_min(low, config.new_low_window)
    prior_floor = _rolling_prior_min(low, config.near_floor_window)
    prior_range_low = _rolling_prior_min(low, config.position_window)
    prior_range_high = _rolling_prior_max(high, config.position_window)
    range_position = (close - prior_range_low) / (prior_range_high - prior_range_low).replace(0.0, np.nan)

    new_low = low <= prior_new_low
    near_floor = low <= prior_floor * (1.0 + float(config.near_floor_tolerance_bp) / 10_000.0)
    lower_range = range_position <= float(config.max_position_in_range)
    candidate = (near_floor | (new_low & lower_range)).fillna(False)

    positions = np.flatnonzero(candidate.to_numpy(dtype=bool))
    valid = (
        (positions >= int(config.lookback))
        & (positions + int(config.horizon) < len(bars))
        & (index[positions] >= research_start)
        & (index[positions] < research_end_exclusive)
    )
    positions = positions[valid]
    bar_delta = _infer_bar_delta(index)

    selected_time = index[positions]
    # Build columns directly.  A list of ~750k dictionaries consumes several
    # gigabytes before pandas creates its own arrays, while this columnar form
    # has only one allocation per field and is exactly equivalent.
    events = pd.DataFrame(
        {
            "event_id": [f"OC_{ts.strftime('%Y%m%d_%H%M%S')}_{int(pos)}" for ts, pos in zip(selected_time, positions)],
            "extreme_time": selected_time,
            "feature_available_time": selected_time + bar_delta,
            "extreme_pos": positions.astype(np.int64, copy=False),
            "extreme_price": low.to_numpy(dtype=float, copy=False)[positions],
            # Placeholders required by shared causal feature builders. They are
            # metadata only and are never model features in research 04.
            "confirmation_time": selected_time + bar_delta,
            "confirmation_available_time": selected_time + 2 * bar_delta,
            "completion_bars": np.zeros(len(positions), dtype=np.int16),
            "realized_confirmation_move_pct": np.full(len(positions), np.nan, dtype=float),
            "cluster_id": "ONLINE_CANDIDATE",
            "distance_to_train_centroid": np.nan,
            "parent_cluster_id": "ONLINE_CANDIDATE",
            "parent_distance_to_centroid": np.nan,
            "split": "",
            "candidate_new_low": new_low.to_numpy(dtype=bool, copy=False)[positions],
            "candidate_near_floor": near_floor.to_numpy(dtype=bool, copy=False)[positions],
            "candidate_range_position": range_position.to_numpy(dtype=float, copy=False)[positions],
        }
    )

    coverage = pd.DataFrame(
        [
            {"metric": "bars_total", "value": int(len(bars))},
            {"metric": "candidate_count", "value": int(len(events))},
            {"metric": "candidate_share", "value": float(len(events) / max(1, len(bars)))},
            {"metric": "new_low_candidate_count", "value": int(events.get("candidate_new_low", pd.Series(dtype=bool)).sum())},
            {"metric": "near_floor_candidate_count", "value": int(events.get("candidate_near_floor", pd.Series(dtype=bool)).sum())},
        ]
    )
    return events, coverage


def attach_temporal_split(
    events: pd.DataFrame,
    *,
    fit_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> pd.DataFrame:
    out = events.copy()
    ts = pd.to_datetime(out["extreme_time"])
    split = np.where(ts <= fit_end, "fit", np.where(ts <= validation_end, "validation", "holdout"))
    out["split"] = split
    out["year"] = ts.dt.year
    return out


def build_forward_path_labels(
    bars: pd.DataFrame,
    candidates: pd.DataFrame,
    *,
    horizon: int = 60,
    target_move_pct: float = 1.0,
    adverse_move_pct: float = 1.0,
    progress_every: int = 10_000,
    vectorized_chunk_size: int = 50_000,
) -> pd.DataFrame:
    """Label candidates from next-bar open using future closed-bar closes only.

    The executable reference is the next bar open. A target succeeds only when
    one of the following ``horizon`` bars has *closed* at or above the target.
    Future intrabar high/low touches are ignored. MFE, MAE, first-touch order,
    and MAE-before-TP are all calculated from the same future close path.
    """

    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    target = float(target_move_pct) / 100.0
    adverse = float(adverse_move_pct) / 100.0
    if target <= 0 or adverse <= 0:
        raise ValueError("target/adverse percentages must be positive")

    index = pd.DatetimeIndex(bars.index)
    open_ = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float, copy=False)
    close = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float, copy=False)

    if vectorized_chunk_size < 1:
        raise ValueError("vectorized_chunk_size must be >= 1")
    reporter = (
        ProgressReporter("[labels] close-only forward path", total=len(candidates), every=max(1, int(progress_every)))
        if ProgressReporter is not None
        else None
    )
    # sliding_window_view is a zero-copy view of close paths.  Candidate rows
    # are materialized only per bounded chunk, avoiding Python loops and a
    # giant list[dict].
    windows = np.lib.stride_tricks.sliding_window_view(close, int(horizon))
    outputs: list[pd.DataFrame] = []
    processed = 0
    for start in range(0, len(candidates), int(vectorized_chunk_size)):
        chunk = candidates.iloc[start : start + int(vectorized_chunk_size)]
        pos = pd.to_numeric(chunk["extreme_pos"], errors="coerce").to_numpy(dtype=np.int64)
        entry_pos = pos + 1
        valid = (entry_pos >= 0) & (entry_pos < len(windows))
        valid_index = np.flatnonzero(valid)
        if valid_index.size:
            valid_entries = entry_pos[valid_index]
            valid[valid_index] &= np.isfinite(open_[valid_entries]) & (open_[valid_entries] > EPS)
        if not valid.any():
            processed += len(chunk)
            if reporter is not None and processed < len(candidates):
                reporter.update(processed)
            continue
        chunk = chunk.iloc[np.flatnonzero(valid)].reset_index(drop=True)
        entry_pos = entry_pos[valid]
        entry = open_[entry_pos]
        path = windows[entry_pos]
        finite_any = np.isfinite(path).any(axis=1)
        if not finite_any.all():
            chunk = chunk.iloc[np.flatnonzero(finite_any)].reset_index(drop=True)
            entry_pos = entry_pos[finite_any]
            entry = entry[finite_any]
            path = path[finite_any]
        if not len(chunk):
            processed += len(candidates.iloc[start : start + int(vectorized_chunk_size)])
            continue

        max_close = np.nanmax(path, axis=1)
        min_close = np.nanmin(path, axis=1)
        mfe = np.maximum(0.0, max_close / entry - 1.0)
        mae = np.maximum(0.0, 1.0 - min_close / entry)
        terminal = path[:, -1] / entry - 1.0
        tp_mask = path >= entry[:, None] * (1.0 + target)
        adverse_mask = path <= entry[:, None] * (1.0 - adverse)
        tp_hit = tp_mask.any(axis=1)
        adverse_hit = adverse_mask.any(axis=1)
        tp_index = np.argmax(tp_mask, axis=1)
        adverse_index = np.argmax(adverse_mask, axis=1)
        tp_bar = np.where(tp_hit, tp_index + 1, np.nan)
        adverse_bar = np.where(adverse_hit, adverse_index + 1, np.nan)

        horizon_axis = np.arange(int(horizon))[None, :]
        before_tp = horizon_axis <= tp_index[:, None]
        before_tp_path = np.where(before_tp, path, np.nan)
        mae_before_tp = np.where(
            tp_hit,
            np.maximum(0.0, 1.0 - np.nanmin(before_tp_path, axis=1) / entry),
            np.nan,
        )
        mfe_only_score = np.clip(mfe / target * 100.0, 0.0, 100.0)
        diff_score = np.clip((mfe - mae) / target * 100.0, -100.0, 100.0)
        tp_priority_score = np.where(tp_hit, 100.0, np.minimum(99.999, diff_score))
        first_touch_score = np.where(
            tp_hit & ~adverse_hit,
            100.0,
            np.where(
                adverse_hit & ~tp_hit,
                -100.0,
                np.where(tp_hit & adverse_hit, np.where(tp_index < adverse_index, 100.0, -100.0), diff_score),
            ),
        )
        outputs.append(
            pd.DataFrame(
                {
                    "event_id": chunk["event_id"].to_numpy(),
                    "entry_time": index[entry_pos],
                    "entry_price": entry,
                    "label_end_time": index[entry_pos + int(horizon) - 1],
                    "forward_horizon_bars": np.full(len(chunk), int(horizon), dtype=np.int16),
                    "mfe_pct": mfe * 100.0,
                    "mae_pct": mae * 100.0,
                    "terminal_return_pct": terminal * 100.0,
                    "tp_hit_1pct": tp_hit,
                    "adverse_hit_1pct": adverse_hit,
                    "tp_first_touch_bar": tp_bar,
                    "adverse_first_touch_bar": adverse_bar,
                    "same_bar_tp_adverse_flag": np.zeros(len(chunk), dtype=bool),
                    "mae_before_tp_pct": mae_before_tp * 100.0,
                    "mfe_only_score": mfe_only_score,
                    "mfe_minus_mae_score": diff_score,
                    "tp_priority_score": tp_priority_score,
                    "first_touch_score": first_touch_score,
                    "entry_price_source": "next_bar_open",
                    "path_observation_source": "future_closed_bar_close",
                }
            )
        )
        processed += len(candidates.iloc[start : start + int(vectorized_chunk_size)])
        if reporter is not None and processed < len(candidates):
            reporter.update(processed)
    if reporter is not None:
        reporter.close()
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def purge_temporal_label_overlap(
    frame: pd.DataFrame,
    *,
    fit_end: pd.Timestamp,
    validation_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove rows whose forward label crosses a temporal split boundary."""

    out = frame.copy()
    label_end = pd.to_datetime(out["label_end_time"])
    split = out["split"].astype(str)
    crossed_fit = split.eq("fit") & (label_end > fit_end)
    crossed_validation = split.eq("validation") & (label_end > validation_end)
    remove = crossed_fit | crossed_validation
    summary = pd.DataFrame(
        [
            {"split": "fit", "removed_cross_boundary": int(crossed_fit.sum())},
            {"split": "validation", "removed_cross_boundary": int(crossed_validation.sum())},
            {"split": "holdout", "removed_cross_boundary": 0},
            {"split": "ALL", "removed_cross_boundary": int(remove.sum())},
        ]
    )
    return out.loc[~remove].reset_index(drop=True), summary


def attach_reference_swing_targets(
    candidates: pd.DataFrame,
    reference_events: pd.DataFrame,
    *,
    reference_type_column: str = "mechanism_type",
) -> pd.DataFrame:
    """Attach exact retrospective clear-Swing-Low identity as target metadata."""

    required = {"extreme_pos", reference_type_column}
    missing = sorted(required - set(reference_events.columns))
    if missing:
        raise RuntimeError(f"reference Swing Low events missing columns: {missing}")
    reference = reference_events[["extreme_pos", reference_type_column]].copy()
    reference["extreme_pos"] = pd.to_numeric(reference["extreme_pos"], errors="coerce")
    reference = reference.dropna(subset=["extreme_pos"]).copy()
    reference["extreme_pos"] = reference["extreme_pos"].astype(int)
    reference = reference.drop_duplicates("extreme_pos", keep="first").rename(
        columns={reference_type_column: "reference_mechanism_type"}
    )
    out = candidates.copy()
    out["extreme_pos"] = pd.to_numeric(out["extreme_pos"], errors="raise").astype(int)
    out = out.merge(reference, on="extreme_pos", how="left", validate="many_to_one")
    out["historical_clear_swing_low"] = out["reference_mechanism_type"].notna()
    return out


def build_candidate_episodes(candidates: pd.DataFrame, *, max_gap_bars: int = 5) -> pd.DataFrame:
    """Group adjacent candidates so evaluation can discount serial duplicates."""

    out = candidates.sort_values("extreme_pos").copy()
    gaps = pd.to_numeric(out["extreme_pos"], errors="coerce").diff().fillna(max_gap_bars + 1)
    out["episode_id"] = (gaps > int(max_gap_bars)).cumsum().astype(int)
    size = out.groupby("episode_id")["event_id"].transform("size")
    out["episode_size"] = size.astype(int)
    out["episode_weight"] = 1.0 / np.maximum(size.to_numpy(dtype=float), 1.0)
    return out


def mechanism_assignment_from_scores(
    score_frame: pd.DataFrame,
    *,
    clear_mechanisms: Sequence[str] = CLEAR_MECHANISMS,
    top_score_threshold: float = 0.55,
    margin_threshold: float = 0.05,
) -> pd.DataFrame:
    columns = [f"score_{label}" for label in clear_mechanisms]
    missing = [column for column in columns if column not in score_frame.columns]
    if missing:
        raise RuntimeError(f"mechanism score frame missing columns: {missing}")
    values = score_frame[columns].to_numpy(dtype=float)
    order = np.argsort(values, axis=1)
    best_idx = order[:, -1]
    second_idx = order[:, -2]
    labels = np.asarray(tuple(clear_mechanisms), dtype=object)
    best = labels[best_idx]
    second = labels[second_idx]
    top = values[np.arange(len(values)), best_idx]
    second_value = values[np.arange(len(values)), second_idx]
    margin = top - second_value
    clear = (top >= float(top_score_threshold)) & (margin >= float(margin_threshold))
    out = score_frame.copy()
    out["mechanism_type"] = np.where(clear, best, "unclear")
    out["secondary_mechanism_type"] = second
    out["mechanism_top_score"] = top
    out["mechanism_margin"] = margin
    out["mechanism_clear"] = clear
    return out


def fit_mechanism_clarity_thresholds(reference_scores: pd.DataFrame, *, quantile: float = 0.20) -> tuple[float, float]:
    top = pd.to_numeric(reference_scores["mechanism_top_score"], errors="coerce")
    margin = pd.to_numeric(reference_scores["mechanism_margin"], errors="coerce")
    return float(top.quantile(quantile)), float(margin.quantile(quantile))


def select_model_features(
    features: pd.DataFrame,
    *,
    metadata_columns: Iterable[str],
    max_missing_ratio: float = 0.35,
    max_features: int = 220,
) -> tuple[str, ...]:
    forbidden = set(metadata_columns) | set(FUTURE_LABEL_COLUMNS)
    candidates: list[str] = []
    for column in features.columns:
        if column in forbidden:
            continue
        lower = column.lower()
        if any(
            token in lower
            for token in (
                "future", "forward", "confirmation", "completion", "mfe", "mae",
                "target_score", "reference_", "historical_", "joint_swing", "mechanism_joint",
            )
        ):
            continue
        numeric = pd.to_numeric(features[column], errors="coerce")
        if float(numeric.isna().mean()) > float(max_missing_ratio):
            continue
        if int(numeric.nunique(dropna=True)) <= 1:
            continue
        candidates.append(column)
    # Prefer mechanism scores and compact current/path features before the many
    # phase bins.  The order is deterministic and does not use future labels.
    priority = sorted(
        candidates,
        key=lambda c: (
            0 if c.startswith("score_") else 1,
            0 if c.startswith("current_") else 1,
            0 if any(token in c for token in ("support_", "spring_", "compression", "divergence", "acceleration", "persistence", "decay")) else 1,
            1 if c.startswith("phase_") else 0,
            c,
        ),
    )
    return tuple(priority[: int(max_features)])


def prepare_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    medians: pd.Series | None = None,
) -> pd.DataFrame:
    x = frame.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    if medians is None:
        medians = x.median().fillna(0.0)
    return x.fillna(medians).astype(float)


def _binary_estimator(
    family: str,
    *,
    random_state: int,
    min_samples_leaf: int,
    logistic_solver: str = "newton-cholesky",
    logistic_max_iter: int = 300,
    logistic_tol: float = 1e-3,
) -> object:
    if family == "logistic":
        return Pipeline(
            [
                ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
                (
                    "model",
                    LogisticRegression(
                        C=0.35,
                        # Class balancing is already applied exactly once in
                        # _balanced_sample_weight together with episode weights.
                        # Setting class_weight="balanced" here would double-count
                        # rare positives and distort both convergence and scores.
                        class_weight=None,
                        max_iter=int(logistic_max_iter),
                        solver=str(logistic_solver),
                        tol=float(logistic_tol),
                        random_state=int(random_state),
                    ),
                ),
            ]
        )
    if family == "logistic_sgd":
        # Large-sample linear probability head.  Early stopping is deliberate:
        # sklearn's plain max-iteration path can emit a warning yet still return
        # coefficients, which is unacceptable for formal range-data ablation.
        # A held-out slice of the fit period decides convergence; no future test
        # rows are involved.
        return Pipeline(
            [
                ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
                (
                    "model",
                    SGDClassifier(
                        loss="log_loss",
                        penalty="l2",
                        alpha=2e-4,
                        max_iter=max(5_000, int(logistic_max_iter)),
                        tol=min(1e-4, float(logistic_tol)),
                        average=True,
                        learning_rate="optimal",
                        early_stopping=True,
                        validation_fraction=0.10,
                        n_iter_no_change=15,
                        class_weight=None,
                        random_state=int(random_state),
                    ),
                ),
            ]
        )
    if family == "hist_gbdt":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=120,
            max_leaf_nodes=15,
            max_depth=3,
            min_samples_leaf=max(20, int(min_samples_leaf)),
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.12,
            random_state=int(random_state),
        )
    raise ValueError(f"unknown binary model family: {family}")


def _score_estimator(family: str, *, random_state: int, min_samples_leaf: int) -> object:
    if family == "ridge":
        return Pipeline(
            [
                ("scale", RobustScaler(quantile_range=(10.0, 90.0))),
                ("model", Ridge(alpha=8.0, random_state=int(random_state))),
            ]
        )
    if family == "hist_gbdt":
        return HistGradientBoostingRegressor(
            learning_rate=0.05,
            max_iter=120,
            max_leaf_nodes=15,
            max_depth=3,
            min_samples_leaf=max(20, int(min_samples_leaf)),
            l2_regularization=2.0,
            loss="squared_error",
            early_stopping=True,
            validation_fraction=0.12,
            random_state=int(random_state),
        )
    raise ValueError(f"unknown score model family: {family}")


def _balanced_sample_weight(y: pd.Series, base_weight: pd.Series | None = None) -> np.ndarray:
    values = y.astype(int).to_numpy()
    n = len(values)
    positives = max(1, int(values.sum()))
    negatives = max(1, int(n - positives))
    class_weight = np.where(values == 1, n / (2.0 * positives), n / (2.0 * negatives))
    if base_weight is not None:
        class_weight = class_weight * pd.to_numeric(base_weight, errors="coerce").fillna(1.0).to_numpy(dtype=float)
    weight = np.asarray(class_weight, dtype=float)
    finite_positive = np.isfinite(weight) & (weight > 0.0)
    if not finite_positive.any():
        return np.ones(n, dtype=float)
    # Keep the average optimization mass at one.  This does not change the
    # relative class/episode weighting but makes regularization and solver
    # tolerances stable across targets with very different positive rates.
    mean_weight = float(np.mean(weight[finite_positive]))
    weight = np.where(finite_positive, weight / max(mean_weight, EPS), 0.0)
    return weight


def fit_binary_model(
    train: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    family: str,
    random_state: int = 42,
    min_samples_leaf: int = 100,
    weight_column: str | None = "episode_weight",
) -> FittedBinaryModel:
    x_raw = train.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = x_raw.median().fillna(0.0)
    x = x_raw.fillna(medians)
    y = train[target_column].astype(int)
    if y.nunique() < 2:
        raise RuntimeError(f"target {target_column} has only one class")
    base_weight = train[weight_column] if weight_column and weight_column in train.columns else None
    sample_weight = _balanced_sample_weight(y, base_weight)

    actual_family = family
    if family in {"logistic", "logistic_sgd"}:
        if family == "logistic_sgd":
            attempts = (
                ("logistic_sgd", "sgd", 5_000, 1e-4, "logistic_sgd"),
                ("logistic", "newton-cholesky", 500, 1e-3, "logistic_newton_cholesky_fallback"),
            )
        else:
            attempts = (
                ("logistic", "newton-cholesky", 300, 1e-3, "logistic_newton_cholesky"),
                ("logistic", "lbfgs", 2_000, 1e-3, "logistic_lbfgs"),
                ("logistic_sgd", "sgd", 5_000, 1e-4, "logistic_sgd"),
            )
        last_warning = ""
        estimator: object | None = None
        for attempt_family, solver, max_iter, tol, method_name in attempts:
            candidate = _binary_estimator(
                attempt_family,
                random_state=random_state,
                min_samples_leaf=min_samples_leaf,
                logistic_solver=solver,
                logistic_max_iter=max_iter,
                logistic_tol=tol,
            )
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                candidate.fit(x, y, model__sample_weight=sample_weight)
            convergence = [item for item in caught if issubclass(item.category, ConvergenceWarning)]
            probe = np.asarray(candidate.predict_proba(x.iloc[: min(512, len(x))]), dtype=float)
            finite_probability = bool(probe.ndim == 2 and probe.shape[1] == 2 and np.isfinite(probe).all())
            if not convergence and finite_probability:
                estimator = candidate
                actual_family = method_name
                break
            if convergence:
                last_warning = str(convergence[-1].message)
            elif not finite_probability:
                last_warning = f"{method_name} produced non-finite probabilities"

        if estimator is None:
            # A non-converged auxiliary head must not invalidate an hour-long
            # walk-forward run.  Fall back to the already-supported shallow
            # nonlinear classifier, record the actual family, and continue.
            candidate = _binary_estimator(
                "hist_gbdt",
                random_state=random_state,
                min_samples_leaf=min_samples_leaf,
            )
            candidate.fit(x, y, sample_weight=sample_weight)
            probe = np.asarray(candidate.predict_proba(x.iloc[: min(512, len(x))]), dtype=float)
            if probe.ndim != 2 or probe.shape[1] != 2 or not np.isfinite(probe).all():
                raise RuntimeError(
                    f"binary target {target_column} failed all stable solvers; last logistic warning: {last_warning}"
                )
            estimator = candidate
            actual_family = "hist_gbdt_convergence_fallback"
    else:
        estimator = _binary_estimator(family, random_state=random_state, min_samples_leaf=min_samples_leaf)
        if isinstance(estimator, Pipeline):
            estimator.fit(x, y, model__sample_weight=sample_weight)
        else:
            estimator.fit(x, y, sample_weight=sample_weight)
    return FittedBinaryModel(family=actual_family, feature_columns=tuple(feature_columns), medians=medians, model=estimator)


def fit_score_model(
    train: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    family: str,
    random_state: int = 42,
    min_samples_leaf: int = 100,
    weight_column: str | None = "episode_weight",
) -> FittedScoreModel:
    x_raw = train.reindex(columns=feature_columns).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = x_raw.median().fillna(0.0)
    x = x_raw.fillna(medians)
    y = pd.to_numeric(train[target_column], errors="coerce").clip(-100.0, 100.0)
    estimator = _score_estimator(family, random_state=random_state, min_samples_leaf=min_samples_leaf)
    sample_weight = (
        pd.to_numeric(train[weight_column], errors="coerce").fillna(1.0).to_numpy(dtype=float)
        if weight_column and weight_column in train.columns
        else np.ones(len(train), dtype=float)
    )
    if isinstance(estimator, Pipeline):
        estimator.fit(x, y, model__sample_weight=sample_weight)
    else:
        estimator.fit(x, y, sample_weight=sample_weight)
    return FittedScoreModel(family=family, feature_columns=tuple(feature_columns), medians=medians, model=estimator)


def binary_metrics(y_true: Sequence[bool | int], probability: Sequence[float]) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0)
    out = {
        "count": float(len(y)),
        "positive_count": float(y.sum()),
        "base_rate": float(y.mean()) if len(y) else np.nan,
        "pr_auc": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
        "brier": float(brier_score_loss(y, p)) if len(y) else np.nan,
    }
    for frac in (0.01, 0.05, 0.10, 0.20):
        n = max(1, int(np.ceil(len(y) * frac)))
        chosen = np.argsort(p)[-n:]
        out[f"precision_top_{int(frac * 100)}pct"] = float(y[chosen].mean())
        out[f"lift_top_{int(frac * 100)}pct"] = float(y[chosen].mean() / max(y.mean(), EPS))
    return out


def score_metrics(y_true: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(predicted, dtype=float)
    valid = np.isfinite(y) & np.isfinite(p)
    y = y[valid]
    p = p[valid]
    if not len(y):
        return {"count": 0.0, "mae": np.nan, "spearman": np.nan}
    return {
        "count": float(len(y)),
        "mae": float(mean_absolute_error(y, p)),
        "spearman": float(pd.Series(y).corr(pd.Series(p), method="spearman")),
    }


def choose_binary_family(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    families: Sequence[str] = ("logistic", "hist_gbdt"),
    random_state: int = 42,
    min_samples_leaf: int = 100,
) -> tuple[str, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for family in families:
        model = fit_binary_model(
            fit,
            feature_columns=feature_columns,
            target_column=target_column,
            family=family,
            random_state=random_state,
            min_samples_leaf=min_samples_leaf,
        )
        probability = model.predict_proba(validation)
        metrics = binary_metrics(validation[target_column], probability)
        rows.append({"family": family, **metrics})
    table = pd.DataFrame(rows).sort_values(["pr_auc", "brier"], ascending=[False, True]).reset_index(drop=True)
    return str(table.iloc[0]["family"]), table


def choose_score_family(
    fit: pd.DataFrame,
    validation: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    families: Sequence[str] = ("ridge", "hist_gbdt"),
    random_state: int = 42,
    min_samples_leaf: int = 100,
) -> tuple[str, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for family in families:
        model = fit_score_model(
            fit,
            feature_columns=feature_columns,
            target_column=target_column,
            family=family,
            random_state=random_state,
            min_samples_leaf=min_samples_leaf,
        )
        predicted = model.predict(validation)
        metrics = score_metrics(validation[target_column], predicted)
        rows.append({"family": family, **metrics})
    table = pd.DataFrame(rows).sort_values(["mae", "spearman"], ascending=[True, False]).reset_index(drop=True)
    return str(table.iloc[0]["family"]), table


def probability_bucket_table(
    frame: pd.DataFrame,
    *,
    probability_column: str,
    target_column: str = "joint_swing_tp_success",
    score_column: str = "tp_priority_score",
    buckets: int = 10,
) -> pd.DataFrame:
    data = frame.copy()
    p = pd.to_numeric(data[probability_column], errors="coerce")
    rank = p.rank(method="first", pct=True)
    data["probability_bucket"] = np.minimum((rank * int(buckets)).apply(np.ceil), int(buckets)).astype(int)
    aggregations: dict[str, tuple[str, str]] = {
        "count": ("event_id", "size"),
        "mean_probability": (probability_column, "mean"),
        "target_rate": (target_column, "mean"),
        "mean_path_score": (score_column, "mean"),
        "median_mfe_pct": ("mfe_pct", "median"),
        "median_mae_pct": ("mae_pct", "median"),
        "median_mae_before_tp_pct": ("mae_before_tp_pct", "median"),
    }
    if "tp_hit_1pct" in data.columns:
        aggregations["tp_rate"] = ("tp_hit_1pct", "mean")
    if "historical_clear_swing_low" in data.columns:
        aggregations["historical_swing_rate"] = ("historical_clear_swing_low", "mean")
    return (
        data.groupby("probability_bucket", as_index=False)
        .agg(**aggregations)
        .sort_values("probability_bucket")
    )


def build_model_stability(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    target_column: str,
    family: str,
    seeds: Sequence[int] = (41, 42, 43, 44, 45),
    min_samples_leaf: int = 100,
) -> pd.DataFrame:
    predictions: dict[int, np.ndarray] = {}
    for seed in seeds:
        model = fit_binary_model(
            train,
            feature_columns=feature_columns,
            target_column=target_column,
            family=family,
            random_state=int(seed),
            min_samples_leaf=min_samples_leaf,
        )
        predictions[int(seed)] = model.predict_proba(holdout)
    base_seed = int(seeds[0])
    base = predictions[base_seed]
    n_top = max(1, int(np.ceil(len(base) * 0.10)))
    base_top = set(np.argsort(base)[-n_top:].tolist())
    rows: list[dict[str, object]] = []
    for seed in seeds:
        pred = predictions[int(seed)]
        top = set(np.argsort(pred)[-n_top:].tolist())
        rows.append(
            {
                "seed": int(seed),
                "prediction_correlation_vs_first": float(np.corrcoef(base, pred)[0, 1]),
                "top_decile_overlap_vs_first": float(len(base_top & top) / max(1, len(base_top | top))),
                **binary_metrics(holdout[target_column], pred),
            }
        )
    return pd.DataFrame(rows)


def model_feature_importance(
    model: FittedBinaryModel,
    validation: pd.DataFrame,
    *,
    target_column: str,
    max_rows: int = 10_000,
    random_state: int = 42,
    top_n: int = 30,
) -> pd.DataFrame:
    sample = validation
    if len(sample) > int(max_rows):
        sample = sample.sample(int(max_rows), random_state=int(random_state))
    x = prepare_feature_matrix(sample, model.feature_columns, model.medians)
    y = sample[target_column].astype(int)
    if y.nunique() < 2:
        return pd.DataFrame()
    result = permutation_importance(
        model.model,
        x,
        y,
        scoring="average_precision",
        n_repeats=3,
        random_state=int(random_state),
        n_jobs=1,
    )
    rows = pd.DataFrame(
        {
            "feature": model.feature_columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return rows.sort_values("importance_mean", ascending=False).head(int(top_n)).reset_index(drop=True)


def representative_prediction_cases(
    frame: pd.DataFrame,
    *,
    probability_column: str,
    target_column: str = "joint_swing_tp_success",
    per_case: int = 20,
) -> pd.DataFrame:
    data = frame.copy()
    target = data[target_column].astype(bool)
    data["prediction_case"] = np.select(
        [
            target & (data[probability_column] >= data[probability_column].quantile(0.90)),
            ~target & (data[probability_column] >= data[probability_column].quantile(0.90)),
            target & (data[probability_column] <= data[probability_column].quantile(0.10)),
        ],
        ["high_score_true_positive", "high_score_false_positive", "low_score_false_negative"],
        default="other",
    )
    rows: list[pd.DataFrame] = []
    for case, group in data[data["prediction_case"] != "other"].groupby("prediction_case", sort=True):
        ascending = case == "low_score_false_negative"
        rows.append(group.sort_values(probability_column, ascending=ascending).head(int(per_case)))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def candidate_gate_recall(
    candidates: pd.DataFrame,
    reference_events: pd.DataFrame,
    *,
    type_column: str,
) -> pd.DataFrame:
    candidate_positions = set(pd.to_numeric(candidates["extreme_pos"], errors="coerce").dropna().astype(int))
    rows: list[dict[str, object]] = []
    for type_id, group in reference_events.groupby(type_column, sort=True):
        positions = pd.to_numeric(group["extreme_pos"], errors="coerce").dropna().astype(int)
        retained = int(positions.isin(candidate_positions).sum())
        rows.append(
            {
                type_column: type_id,
                "reference_count": int(len(positions)),
                "gate_retained_count": retained,
                "gate_recall": float(retained / max(1, len(positions))),
            }
        )
    return pd.DataFrame(rows)


def label_definition_table(target_move_pct: float, adverse_move_pct: float, horizon: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "label": "joint_swing_tp_success",
                "range": "0/1",
                "definition": "current bar is an exact clear 03 low-anchored Swing Low and a future closed-bar close reaches the next-bar-open +1% target",
                "purpose": "primary online recognizability target",
            },
            {
                "label": "historical_clear_swing_low",
                "range": "0/1",
                "definition": "current bar position matches a T2/T3/T4/B2/B3/B4 historical Swing Low",
                "purpose": "exact retrospective identity target",
            },
            {
                "label": "mechanism_joint_success",
                "range": "0/1",
                "definition": "joint Swing+TP success and online mechanism equals the frozen 03 mechanism",
                "purpose": "mechanism-specific specialist target",
            },
            {
                "label": "tp_hit_1pct",
                "range": "0/1",
                "definition": f"next-bar open reference; a future closed-bar close reaches +{target_move_pct:g}% within {horizon} bars",
                "purpose": "primary success target",
            },
            {
                "label": "mfe_only_score",
                "range": "0..100",
                "definition": f"min(100, close-based MFE/{target_move_pct:g}%*100)",
                "purpose": "matches 0.6% -> 60 when adverse path is ignored",
            },
            {
                "label": "mfe_minus_mae_score",
                "range": "-100..100",
                "definition": f"clip((close-based MFE-MAE)/{target_move_pct:g}%*100)",
                "purpose": "continuous path quality",
            },
            {
                "label": "tp_priority_score",
                "range": "-100..100",
                "definition": "100 if a closed-bar close reaches TP; otherwise close-based MFE-MAE score",
                "purpose": "user rule: no forced 60-bar holding after TP",
            },
            {
                "label": "first_touch_score",
                "range": "-100..100",
                "definition": f"+100 if a close reaches +{target_move_pct:g}% first; -100 if a close reaches -{adverse_move_pct:g}% first; otherwise path score",
                "purpose": "separates clean reversals from deep adverse-first paths",
            },
        ]
    )
