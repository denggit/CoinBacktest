#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal online expert-model helpers for frozen Swing Low typologies.

Research 17 uses the original Research 01-03 labels only as supervised targets.
The live-side object is always a current closed-bar candidate.  All model
features are causal and all ranking thresholds are fitted from a trailing
policy window without reading policy/test labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from research.market_structure.swing_low_typology.common.online_recognizability import (
    FittedBinaryModel,
    fit_binary_model,
)
from research.market_structure.swing_low_typology.common.original_typology_bridge import (
    BridgeFold,
    walkforward_folds,
)

EPS = 1e-12


class ExpertModelUnavailableError(RuntimeError):
    """Expected fold-level model unavailability, not a data-integrity failure."""


class ExpertFeatureUnavailableError(ExpertModelUnavailableError):
    pass


class ExpertTargetUnavailableError(ExpertModelUnavailableError):
    pass


class ExpertScoreResolutionError(ExpertModelUnavailableError):
    pass


class ExpertRankUnavailableError(ExpertModelUnavailableError):
    pass

# Fixed, compact causal feature set.  The order is deliberate and is also used
# for deterministic correlation pruning.  Research 17 must not search across
# arbitrary feature families after seeing frozen test outcomes.
COMPACT_EXPERT_FEATURES: tuple[str, ...] = (
    "current_return_1",
    "current_body_pct",
    "current_range_pct",
    "current_lower_wick_share",
    "current_close_in_bar",
    "current_delta_ratio",
    "current_large_delta_ratio",
    "current_buy_ratio",
    "price_return_5",
    "price_return_10",
    "price_return_30",
    "price_return_60",
    "drawdown_from_high_30",
    "rebound_from_low_5",
    "rebound_from_low_30",
    "range_position_30",
    "realized_vol_5",
    "realized_vol_30",
    "down_bar_share_10",
    "down_bar_share_30",
    "path_efficiency_10",
    "path_efficiency_30",
    "delta_ratio_5",
    "delta_ratio_30",
    "large_delta_ratio_5",
    "large_delta_ratio_30",
    "support_test_density_10",
    "support_test_density_30",
    "sell_pressure_absorption_30",
    "return_acceleration_5_30",
    "session_return_from_open",
    "session_range_position",
    "region_age_bars",
    "region_candidate_density",
    "region_return_from_start",
    "region_low_progression",
    "region_rebound_from_low",
    "region_new_low_count",
    "region_candidate_retest_count",
    "region_bars_since_low",
    "region_cumulative_delta_ratio",
    "region_recent_delta_ratio",
    "region_delta_improvement",
    "region_absorption_improvement",
    "region_range_recent_vs_early",
    "region_vol_recent_vs_early",
    "tf15m_return_3",
    "tf15m_range_position_3",
    "tf60m_return_3",
    "tf60m_range_position_3",
)

FORBIDDEN_FEATURE_TOKENS: tuple[str, ...] = (
    "future",
    "reference_",
    "tp_",
    "mfe",
    "mae",
    "label_",
    "completion",
    "confirmation",
    "realized_confirmation",
    "target_",
    "expert_",
    "raw_score",
    "rank",
    "probability",
)


@dataclass(frozen=True)
class ExpertSpec:
    expert_id: str
    display_name: str
    target_kind: str
    target_level: str | None
    target_type: str | None
    special_feature: str | None
    special_direction: float
    description: str


EXPERT_SPECS: tuple[ExpertSpec, ...] = (
    ExpertSpec(
        expert_id="E1_C3D_PRICE_RESPONSE",
        display_name="C3-D price-response failure",
        target_kind="future_original_type",
        target_level="stage2_type",
        target_type="C3-D",
        special_feature="sell_pressure_absorption_30",
        special_direction=1.0,
        description="Identify online candidates that lead into frozen C3-D Swing Lows; retain the predeclared price-response-failure condition.",
    ),
    ExpertSpec(
        expert_id="E2_C3E_EARLY_RECOVERY",
        display_name="C3-E early recovery",
        target_kind="future_original_type",
        target_level="stage2_type",
        target_type="C3-E",
        special_feature="region_rebound_from_low",
        special_direction=1.0,
        description="Identify online candidates that lead into frozen C3-E Swing Lows; retain the predeclared current-recovery condition.",
    ),
    ExpertSpec(
        expert_id="E3_FIRST_SWEEP_CONTROL",
        display_name="Macro First Sweep control",
        target_kind="causal_event",
        target_level=None,
        target_type=None,
        special_feature=None,
        special_direction=1.0,
        description="Known-at-signal First Sweep branch used as a low-frequency control, not as a gate for E1/E2.",
    ),
)


@dataclass(frozen=True)
class EmpiricalRankReference:
    sorted_values: np.ndarray

    @classmethod
    def fit(cls, values: Sequence[float] | np.ndarray) -> "EmpiricalRankReference":
        array = np.asarray(values, dtype=float).reshape(-1)
        finite = np.sort(array[np.isfinite(array)])
        if finite.size < 20 or np.unique(finite).size < 5:
            raise ExpertRankUnavailableError(
                "empirical rank reference is degenerate: "
                f"finite={finite.size} unique={np.unique(finite).size}"
            )
        return cls(sorted_values=finite)

    def transform(self, values: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float).reshape(-1)
        out = np.full(array.shape, np.nan, dtype=float)
        finite = np.isfinite(array)
        out[finite] = (
            np.searchsorted(self.sorted_values, array[finite], side="right")
            / float(len(self.sorted_values))
            * 100.0
        )
        return out


@dataclass(frozen=True)
class FittedResolvedBinary:
    model: FittedBinaryModel
    selected_features: tuple[str, ...]
    diagnostics: Mapping[str, object]


@dataclass(frozen=True)
class FoldSplit:
    fold: BridgeFold
    policy_start: pd.Timestamp
    model_fit_end: pd.Timestamp
    model_fit_mask: np.ndarray
    policy_mask: np.ndarray
    test_mask: np.ndarray


def expert_spec_table() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "expert_id": spec.expert_id,
                "display_name": spec.display_name,
                "target_kind": spec.target_kind,
                "target_level": spec.target_level,
                "target_type": spec.target_type,
                "special_feature": spec.special_feature,
                "special_direction": spec.special_direction,
                "description": spec.description,
            }
            for spec in EXPERT_SPECS
        ]
    )


def build_expert_targets(frame: pd.DataFrame, *, bridge_maximum_lead_bars: int = 15) -> pd.DataFrame:
    """Attach sparse expert labels without allowing them into model features."""

    out = frame.copy()
    matched = out["reference_swing_matched"].fillna(False).astype(bool)
    for spec in EXPERT_SPECS:
        column = f"target_{spec.expert_id}"
        if spec.target_kind == "future_original_type":
            reference_column = f"reference_{spec.target_level}"
            if reference_column not in out.columns:
                raise RuntimeError(f"expert target source missing {reference_column}")
            # ``reference_stage2_type`` is legitimately missing for matched C1/C2
            # Swing Lows and for unmatched online candidates.  E1/E2 are
            # one-vs-rest classifiers, so those rows are explicit negatives, not
            # unknown labels.  Pandas nullable-string equality otherwise yields
            # ``pd.NA`` and leaks a nullable boolean target into sklearn.
            reference_match = (
                out[reference_column]
                .astype("string")
                .eq(str(spec.target_type))
                .fillna(False)
                .astype(bool)
            )
            out[column] = (matched & reference_match).astype(bool)
        elif spec.target_kind == "causal_event":
            if "is_macro_first_sweep" not in out.columns:
                raise RuntimeError("First Sweep control requires is_macro_first_sweep")
            out[column] = out["is_macro_first_sweep"].fillna(False).astype(bool)
        else:  # pragma: no cover - guarded by frozen specs
            raise ValueError(spec.target_kind)
    extreme_time = pd.to_datetime(out["extreme_time"], errors="raise")
    out["type_label_end_time"] = extreme_time + pd.Timedelta(minutes=int(bridge_maximum_lead_bars))
    out["target_tp60"] = out["tp_1_h60"].fillna(False).astype(bool)
    return out


def validate_feature_names(feature_columns: Sequence[str]) -> None:
    bad = [
        str(column)
        for column in feature_columns
        if any(token in str(column).lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if bad:
        raise RuntimeError(f"forbidden future/label-derived expert features: {bad}")


def _float64_numeric_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    converted: dict[str, pd.Series] = {}
    for column in columns:
        if column not in frame.columns:
            converted[column] = pd.Series(np.nan, index=frame.index, dtype="float64")
            continue
        series = frame[column]
        if pd.api.types.is_bool_dtype(series.dtype):
            converted[column] = series.astype("Float64").astype("float64")
        else:
            converted[column] = pd.to_numeric(series, errors="coerce").astype("float64")
    return pd.DataFrame(converted, index=frame.index).replace([np.inf, -np.inf], np.nan)


def condition_feature_columns(
    train: pd.DataFrame,
    requested: Sequence[str],
    *,
    required_features: Sequence[str] = (),
    minimum_non_null_share: float = 0.70,
    minimum_robust_span: float = 1e-10,
    maximum_abs_correlation: float = 0.9975,
) -> tuple[tuple[str, ...], pd.DataFrame]:
    """Deterministically remove missing, constant and duplicate-like features.

    Feature order is frozen.  No target is used for selection.  Boolean and
    nullable-boolean columns are explicitly converted to float64 before any
    quantile arithmetic, preventing the NumPy boolean-subtract failure found in
    Research 16.
    """

    requested_unique = tuple(dict.fromkeys(str(column) for column in requested))
    validate_feature_names(requested_unique)
    required = set(str(column) for column in required_features)
    numeric = _float64_numeric_frame(train, requested_unique)
    diagnostics: list[dict[str, object]] = []
    candidates: list[str] = []
    for column in requested_unique:
        values = numeric[column]
        non_null_share = float(values.notna().mean())
        finite = values.dropna()
        unique = int(finite.nunique())
        robust_span = (
            float(abs(finite.quantile(0.90) - finite.quantile(0.10)))
            if len(finite)
            else np.nan
        )
        status = "selected_pre_correlation"
        if non_null_share < float(minimum_non_null_share):
            status = "dropped_missing"
        elif unique <= 1:
            status = "dropped_constant"
        elif not np.isfinite(robust_span) or robust_span <= float(minimum_robust_span):
            status = "dropped_near_constant"
        else:
            candidates.append(column)
        diagnostics.append(
            {
                "feature": column,
                "non_null_share": non_null_share,
                "unique_values": unique,
                "robust_span_90_10": robust_span,
                "status": status,
                "correlated_with": "",
            }
        )

    selected: list[str] = []
    for column in candidates:
        correlated_with = ""
        for existing in selected:
            pair = numeric[[existing, column]].dropna()
            if len(pair) < 20:
                continue
            correlation = float(pair[existing].corr(pair[column]))
            if np.isfinite(correlation) and abs(correlation) >= float(maximum_abs_correlation):
                correlated_with = existing
                break
        if correlated_with:
            for record in diagnostics:
                if record["feature"] == column:
                    record["status"] = "dropped_correlated"
                    record["correlated_with"] = correlated_with
                    break
        else:
            selected.append(column)

    missing_required = sorted(required.difference(selected))
    if missing_required:
        detail = {
            row["feature"]: row["status"]
            for row in diagnostics
            if row["feature"] in missing_required
        }
        raise ExpertFeatureUnavailableError(f"required expert features did not enter model: {detail}")
    if len(selected) < 8:
        raise ExpertFeatureUnavailableError(f"too few valid expert features after conditioning: {selected}")
    return tuple(selected), pd.DataFrame(diagnostics)


def predict_binary_score_chunked(
    model: FittedBinaryModel,
    frame: pd.DataFrame,
    *,
    chunk_size: int = 20_000,
) -> np.ndarray:
    parts = [
        np.asarray(model.predict_score(frame.iloc[start : start + int(chunk_size)]), dtype=float)
        for start in range(0, len(frame), int(chunk_size))
    ]
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def predict_binary_probability_chunked(
    model: FittedBinaryModel,
    frame: pd.DataFrame,
    *,
    chunk_size: int = 20_000,
) -> np.ndarray:
    parts = [
        np.asarray(model.predict_proba(frame.iloc[start : start + int(chunk_size)]), dtype=float)
        for start in range(0, len(frame), int(chunk_size))
    ]
    return np.concatenate(parts) if parts else np.asarray([], dtype=float)


def raw_score_resolution(values: Sequence[float] | np.ndarray) -> dict[str, object]:
    array = np.asarray(values, dtype=float).reshape(-1)
    finite = array[np.isfinite(array)]
    unique = int(np.unique(finite).size)
    required = max(10, min(100, int(np.ceil(max(1, finite.size) * 0.01))))
    return {
        "rows": int(array.size),
        "finite_share": float(np.isfinite(array).mean()) if array.size else np.nan,
        "unique_scores": unique,
        "required_unique_scores": required,
        "passed": bool(array.size and np.isfinite(array).all() and unique >= required),
    }


def fit_resolved_binary_model(
    train: pd.DataFrame,
    policy: pd.DataFrame,
    *,
    requested_features: Sequence[str],
    required_features: Sequence[str],
    target_column: str,
    random_state: int,
    min_samples_leaf: int,
    prediction_chunk_size: int = 20_000,
) -> tuple[FittedResolvedBinary, pd.DataFrame]:
    if train.empty or policy.empty:
        raise RuntimeError("binary model train/policy rows must be non-empty")
    if target_column not in train.columns:
        raise RuntimeError(f"binary model target column missing: {target_column}")
    target = train[target_column]
    missing_targets = int(target.isna().sum())
    if missing_targets:
        raise RuntimeError(
            f"target {target_column} contains {missing_targets} NA rows before model fit; "
            "expert targets must be resolved to explicit one-vs-rest booleans"
        )
    if target.astype(int).nunique() < 2:
        raise ExpertTargetUnavailableError(f"target {target_column} has only one class in model-fit rows")
    selected, feature_diagnostics = condition_feature_columns(
        train,
        requested_features,
        required_features=required_features,
    )
    attempts: list[dict[str, object]] = []
    seen_actual: set[str] = set()
    chosen: FittedBinaryModel | None = None
    chosen_resolution: dict[str, object] | None = None
    for requested_family in ("logistic", "hist_gbdt"):
        model = fit_binary_model(
            train,
            feature_columns=selected,
            target_column=target_column,
            family=requested_family,
            random_state=int(random_state),
            min_samples_leaf=int(min_samples_leaf),
            weight_column="episode_weight",
        )
        actual_family = str(model.family)
        if actual_family in seen_actual:
            continue
        seen_actual.add(actual_family)
        score = predict_binary_score_chunked(model, policy, chunk_size=prediction_chunk_size)
        resolution = raw_score_resolution(score)
        attempts.append(
            {
                "requested_family": requested_family,
                "actual_family": actual_family,
                **resolution,
            }
        )
        if bool(resolution["passed"]):
            chosen = model
            chosen_resolution = resolution
            break
    if chosen is None or chosen_resolution is None:
        raise ExpertScoreResolutionError(
            f"all model families lost raw-score resolution for {target_column}: {attempts}"
        )
    fitted = FittedResolvedBinary(
        model=chosen,
        selected_features=selected,
        diagnostics={
            "target_column": target_column,
            "actual_family": chosen.family,
            "feature_count": len(selected),
            "attempts": attempts,
            **chosen_resolution,
        },
    )
    return fitted, feature_diagnostics


def build_fold_split(
    frame: pd.DataFrame,
    fold: BridgeFold,
    *,
    policy_days: int = 90,
    fit_label_end_column: str,
) -> FoldSplit:
    times = pd.to_datetime(frame["extreme_time"], errors="raise")
    label_end = pd.to_datetime(frame[fit_label_end_column], errors="raise")
    policy_start = fold.train_end.normalize() - pd.Timedelta(days=int(policy_days) - 1)
    model_fit_end = policy_start - pd.Timedelta(nanoseconds=1)
    model_fit = (
        (times >= fold.train_start)
        & (times <= model_fit_end)
        & (label_end <= model_fit_end)
    )
    policy = (times >= policy_start) & (times <= fold.train_end)
    test = (
        (times >= fold.test_start)
        & (times <= fold.test_end)
        & (pd.to_datetime(frame["label_end_time"], errors="raise") <= fold.test_end)
        & (pd.to_datetime(frame["type_label_end_time"], errors="raise") <= fold.test_end)
    )
    return FoldSplit(
        fold=fold,
        policy_start=policy_start,
        model_fit_end=model_fit_end,
        model_fit_mask=model_fit.to_numpy(dtype=bool),
        policy_mask=policy.to_numpy(dtype=bool),
        test_mask=test.to_numpy(dtype=bool),
    )


def add_episode_weight(
    frame: pd.DataFrame,
    *,
    positive_target_column: str | None = None,
) -> pd.DataFrame:
    """Limit repeated candidate rows from the same region/future event."""

    out = frame.copy()
    region = out.get("causal_region_id", out["event_id"]).astype(str)
    region_count = region.map(region.value_counts()).astype(float)
    weight = 1.0 / np.maximum(region_count.to_numpy(dtype=float), 1.0)
    if positive_target_column is not None:
        positive = out[positive_target_column].fillna(False).astype(bool)
        reference = out.get("reference_swing_event_id", pd.Series(pd.NA, index=out.index)).astype("string")
        valid_positive = positive & reference.notna()
        if valid_positive.any():
            positive_counts = reference[valid_positive].map(reference[valid_positive].value_counts()).astype(float)
            weight[valid_positive.to_numpy()] = 1.0 / np.maximum(
                positive_counts.to_numpy(dtype=float), 1.0
            )
    out["episode_weight"] = weight
    return out


def binary_ranking_metrics(
    target: Sequence[bool] | pd.Series,
    score: Sequence[float] | np.ndarray,
) -> dict[str, float]:
    y = pd.Series(target).fillna(False).astype(int).to_numpy()
    s = np.asarray(score, dtype=float)
    finite = np.isfinite(s)
    if finite.sum() < 2 or np.unique(y[finite]).size < 2:
        return {"roc_auc": np.nan, "average_precision": np.nan, "positive_rate": float(y.mean())}
    return {
        "roc_auc": float(roc_auc_score(y[finite], s[finite])),
        "average_precision": float(average_precision_score(y[finite], s[finite])),
        "positive_rate": float(y[finite].mean()),
    }


def signed_feature_rank(
    policy: pd.Series,
    test: pd.Series,
    *,
    direction: float,
) -> tuple[np.ndarray, np.ndarray, EmpiricalRankReference]:
    policy_values = pd.to_numeric(policy, errors="coerce").to_numpy(dtype=float) * float(direction)
    test_values = pd.to_numeric(test, errors="coerce").to_numpy(dtype=float) * float(direction)
    reference = EmpiricalRankReference.fit(policy_values)
    return reference.transform(policy_values), reference.transform(test_values), reference


def combine_component_ranks(
    policy_components: Mapping[str, np.ndarray],
    test_components: Mapping[str, np.ndarray],
    *,
    weights: Mapping[str, float],
) -> tuple[np.ndarray, np.ndarray, EmpiricalRankReference]:
    if set(policy_components) != set(test_components) or set(weights) != set(policy_components):
        raise ValueError("rank component keys and weights must match")
    total_weight = float(sum(float(value) for value in weights.values()))
    if total_weight <= 0:
        raise ValueError("combined rank weights must be positive")
    policy_score = np.zeros(len(next(iter(policy_components.values()))), dtype=float)
    test_score = np.zeros(len(next(iter(test_components.values()))), dtype=float)
    for key, weight in weights.items():
        policy_score += np.asarray(policy_components[key], dtype=float) * float(weight) / total_weight
        test_score += np.asarray(test_components[key], dtype=float) * float(weight) / total_weight
    reference = EmpiricalRankReference.fit(policy_score)
    return reference.transform(policy_score), reference.transform(test_score), reference


def top_rank_mask(rank: Sequence[float] | np.ndarray, top_pct: int) -> np.ndarray:
    if int(top_pct) not in {20, 30, 40, 100}:
        raise ValueError("Research 17 only permits Top20/30/40 and explicit all-event controls")
    values = np.asarray(rank, dtype=float)
    if int(top_pct) == 100:
        return np.isfinite(values)
    return np.isfinite(values) & (values >= 100.0 - float(top_pct))


def months_in_fold(fold: BridgeFold) -> int:
    start = fold.test_start.to_period("M")
    end = fold.test_end.to_period("M")
    return int(end.ordinal - start.ordinal + 1)


def frequency_guard(
    *,
    raw_events: int,
    executable_trades: int,
    months: int,
    minimum_annualized_raw_events: float = 600.0,
    minimum_annualized_executable_trades: float = 240.0,
    minimum_raw_events: int = 100,
) -> dict[str, object]:
    annualized_raw = float(raw_events / max(1, months) * 12.0)
    annualized_executable = float(executable_trades / max(1, months) * 12.0)
    checks = {
        "minimum_raw_events_passed": bool(raw_events >= int(minimum_raw_events)),
        "annualized_raw_frequency_passed": bool(annualized_raw >= float(minimum_annualized_raw_events)),
        "annualized_executable_frequency_passed": bool(
            annualized_executable >= float(minimum_annualized_executable_trades)
        ),
    }
    return {
        "raw_events": int(raw_events),
        "executable_trades": int(executable_trades),
        "months": int(months),
        "annualized_raw_events": annualized_raw,
        "annualized_executable_trades": annualized_executable,
        **checks,
        "frequency_guard_passed": bool(all(checks.values())),
    }


def target_capture_metrics(
    selected: pd.DataFrame,
    test: pd.DataFrame,
    *,
    target_column: str,
) -> dict[str, object]:
    target_all = test[target_column].fillna(False).astype(bool)
    target_selected = selected[target_column].fillna(False).astype(bool)
    positive_rows = int(target_all.sum())
    selected_positive_rows = int(target_selected.sum())
    all_ids = set(
        test.loc[target_all, "reference_swing_event_id"].dropna().astype(str)
    )
    selected_ids = set(
        selected.loc[target_selected, "reference_swing_event_id"].dropna().astype(str)
    )
    return {
        "target_positive_candidate_rows": positive_rows,
        "selected_target_positive_candidate_rows": selected_positive_rows,
        "target_candidate_precision": float(target_selected.mean()) if len(selected) else np.nan,
        "target_candidate_recall": float(selected_positive_rows / positive_rows) if positive_rows else np.nan,
        "target_unique_swing_events": int(len(all_ids)),
        "selected_target_unique_swing_events": int(len(selected_ids)),
        "target_unique_swing_recall": float(len(selected_ids) / len(all_ids)) if all_ids else np.nan,
    }


def policy_path_metrics(
    selected: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    horizon: int = 60,
) -> dict[str, object]:
    if selected.empty:
        return {
            "tp60_rate": np.nan,
            "tp60_uplift_pp": np.nan,
            "mean_mae60_pct": np.nan,
            "mae60_change_pp": np.nan,
            "median_mfe60_pct": np.nan,
            "clean60_rate": np.nan,
            "permanent_failure60_rate": np.nan,
        }
    tp = float(selected[f"tp_1_h{horizon}"].astype(bool).mean())
    base_tp = float(baseline[f"tp_1_h{horizon}"].astype(bool).mean())
    mae = float(pd.to_numeric(selected[f"mae_h{horizon}_pct"], errors="coerce").mean())
    base_mae = float(pd.to_numeric(baseline[f"mae_h{horizon}_pct"], errors="coerce").mean())
    return {
        "tp60_rate": tp,
        "tp60_uplift_pp": (tp - base_tp) * 100.0,
        "mean_mae60_pct": mae,
        "mae60_change_pp": mae - base_mae,
        "median_mfe60_pct": float(
            pd.to_numeric(selected[f"mfe_h{horizon}_pct"], errors="coerce").median()
        ),
        "clean60_rate": float(selected[f"clean_0p5_h{horizon}"].astype(bool).mean()),
        "permanent_failure60_rate": float(
            selected[f"permanent_failure_h{horizon}"].astype(bool).mean()
        ),
    }


def crossfold_policy_summary(scorecard: pd.DataFrame) -> pd.DataFrame:
    if scorecard.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_columns = ["expert_id", "policy_id", "top_pct", "cost_multiplier"]
    for keys, group in scorecard.groupby(group_columns, sort=True, dropna=False):
        values = dict(zip(group_columns, keys))
        primary = group[group["entry_delay_bars"].eq(0)]
        if primary.empty:
            continue
        frequency_passes = int(primary["frequency_guard_passed"].fillna(False).astype(bool).sum())
        positive_expectancy = int((primary["net_expectancy_bps"] > 0.0).sum())
        positive_tp_uplift = int((primary["tp60_uplift_pp"] > 0.0).sum())
        nonworse_mae = int((primary["mae60_change_pp"] <= 0.05).sum())
        folds = int(primary["fold"].nunique())
        total_executable = int(primary["executable_trades"].sum())
        minimum_profit_factor = float(primary["profit_factor"].min())
        median_profit_factor = float(primary["profit_factor"].median())
        candidate = bool(
            folds == 3
            and frequency_passes == 3
            and positive_expectancy == 3
            and positive_tp_uplift == 3
            and nonworse_mae == 3
            and total_executable >= 600
            and float(primary["net_expectancy_bps"].median()) > 0.0
            and minimum_profit_factor > 1.0
            and median_profit_factor >= 1.05
        )
        throughput_series = pd.to_numeric(
            primary.get(
                "net_edge_throughput_bps_per_month",
                pd.Series(np.nan, index=primary.index, dtype=float),
            ),
            errors="coerce",
        )
        throughput_median = (
            float(throughput_series.dropna().median())
            if throughput_series.notna().any()
            else np.nan
        )
        rows.append(
            {
                **values,
                "folds": folds,
                "frequency_pass_folds": frequency_passes,
                "positive_net_expectancy_folds": positive_expectancy,
                "positive_tp_uplift_folds": positive_tp_uplift,
                "nonworse_mae_folds": nonworse_mae,
                "total_oos_raw_events": int(primary["raw_events"].sum()),
                "total_oos_executable_trades": total_executable,
                "minimum_annualized_raw_events": float(primary["annualized_raw_events"].min()),
                "minimum_annualized_executable_trades": float(
                    primary["annualized_executable_trades"].min()
                ),
                "median_tp60_uplift_pp": float(primary["tp60_uplift_pp"].median()),
                "median_mae60_change_pp": float(primary["mae60_change_pp"].median()),
                "median_net_expectancy_bps": float(primary["net_expectancy_bps"].median()),
                "minimum_profit_factor": minimum_profit_factor,
                "median_profit_factor": median_profit_factor,
                "median_net_edge_throughput_bps_per_month": throughput_median,
                "research_candidate_status": "candidate" if candidate else "not_supported",
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["research_candidate_status", "median_net_expectancy_bps", "total_oos_executable_trades"],
        ascending=[True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)



def policy_neighborhood_summary(crossfold: pd.DataFrame) -> pd.DataFrame:
    """Require broad Top20/30/40 neighborhood support, not a single tail."""

    if crossfold.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in crossfold.groupby(
        ["expert_id", "policy_id", "cost_multiplier"], sort=True, dropna=False
    ):
        expert_id, policy_id, cost_multiplier = keys
        ranked = group[group["top_pct"].isin([20, 30, 40])].copy()
        all_event = group[group["top_pct"].eq(100)].copy()
        if not ranked.empty:
            candidate_tops = sorted(
                ranked.loc[
                    ranked["research_candidate_status"].eq("candidate"), "top_pct"
                ].astype(int).tolist()
            )
            top30_supported = 30 in candidate_tops
            neighbor_supported = bool(20 in candidate_tops or 40 in candidate_tops)
            status = (
                "candidate"
                if top30_supported and neighbor_supported
                else "not_supported"
            )
            rows.append(
                {
                    "expert_id": expert_id,
                    "policy_id": policy_id,
                    "cost_multiplier": float(cost_multiplier),
                    "evaluated_top_pcts": "|".join(map(str, sorted(ranked["top_pct"].astype(int).unique()))),
                    "candidate_top_pcts": "|".join(map(str, candidate_tops)),
                    "top30_supported": top30_supported,
                    "adjacent_neighbor_supported": neighbor_supported,
                    "minimum_total_oos_executable_trades": int(
                        ranked["total_oos_executable_trades"].min()
                    ),
                    "neighborhood_status": status,
                }
            )
        for row in all_event.itertuples(index=False):
            rows.append(
                {
                    "expert_id": expert_id,
                    "policy_id": policy_id,
                    "cost_multiplier": float(cost_multiplier),
                    "evaluated_top_pcts": "100",
                    "candidate_top_pcts": (
                        "100" if row.research_candidate_status == "candidate" else ""
                    ),
                    "top30_supported": False,
                    "adjacent_neighbor_supported": False,
                    "minimum_total_oos_executable_trades": int(
                        row.total_oos_executable_trades
                    ),
                    "neighborhood_status": row.research_candidate_status,
                }
            )
    return pd.DataFrame(rows)

def folds_for_end_date(end_date: str) -> tuple[BridgeFold, ...]:
    return walkforward_folds(end_date)
