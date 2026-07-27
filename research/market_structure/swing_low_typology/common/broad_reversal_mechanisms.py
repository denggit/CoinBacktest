#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal mechanism rules and matched controls for research 16.

The module contains no future labels and no performance-based threshold search.
All mechanism scores are train-fitted empirical percentiles of current or older
closed-bar features.  A fixed score threshold and fixed semantic gates convert
those scores into overlapping mechanism memberships.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

EPS = 1e-12
MECHANISM_ORDER: tuple[str, ...] = (
    "G1_shock_macro_first_sweep",
    "G2_trend_exhaustion",
    "G3_base_stabilization",
    "G4_absorption_price_response_failure",
    "G5_fast_recovery_breakdown_failure",
    "G6_trend_pullback_restart",
)
UNRESOLVED_MECHANISM = "G7_unresolved_mixed"

# Higher transformed percentile always means "more of the named mechanism".
MECHANISM_COMPONENTS: Mapping[str, tuple[tuple[str, float], ...]] = {
    "G1_shock_macro_first_sweep": (
        ("current_range_pct", 1.0),
        ("notional_intensity_30", 1.0),
        ("trades_intensity_30", 1.0),
        ("price_return_15", -1.0),
        ("current_delta_ratio", -1.0),
        ("current_large_delta_ratio", -1.0),
    ),
    "G2_trend_exhaustion": (
        ("price_return_60", -1.0),
        ("price_return_120", -1.0),
        ("return_acceleration_5_30", 1.0),
        ("return_acceleration_10_60", 1.0),
        ("region_delta_improvement", 1.0),
        ("region_large_delta_improvement", 1.0),
        ("region_absorption_improvement", 1.0),
        ("region_bars_since_low", 1.0),
    ),
    "G3_base_stabilization": (
        ("support_test_density_60", 1.0),
        ("support_test_density_120", 1.0),
        ("region_candidate_retest_count", 1.0),
        ("region_age_bars", 1.0),
        ("region_candidate_density", 1.0),
        ("vol_compression_10_60", -1.0),
        ("range_compression_10_60", -1.0),
        ("region_range_recent_vs_early", -1.0),
    ),
    "G4_absorption_price_response_failure": (
        ("sell_pressure_absorption_30", 1.0),
        ("sell_pressure_absorption_60", 1.0),
        ("price_delta_divergence_30", 1.0),
        ("price_delta_divergence_60", 1.0),
        ("region_absorption_improvement", 1.0),
        ("region_delta_improvement", 1.0),
        ("delta_ratio_30", -1.0),
        ("delta_ratio_60", -1.0),
    ),
    "G5_fast_recovery_breakdown_failure": (
        ("current_return_1", 1.0),
        ("current_close_in_bar", 1.0),
        ("current_delta_ratio", 1.0),
        ("current_buy_ratio", 1.0),
        ("region_rebound_from_low", 1.0),
        ("region_reclaim_10bp", 1.0),
        ("region_reclaim_20bp", 1.0),
        ("return_acceleration_5_30", 1.0),
    ),
    "G6_trend_pullback_restart": (
        ("tf60m_return_3", 1.0),
        ("tf60m_return_6", 1.0),
        ("tf60m_range_position_6", 1.0),
        ("price_return_15", -1.0),
        ("price_return_30", -1.0),
        ("current_return_1", 1.0),
        ("current_delta_ratio", 1.0),
        ("region_delta_improvement", 1.0),
    ),
}

SIMPLE_FACTOR_SPECS: Mapping[str, tuple[str, float]] = {
    "D1_recent_decline": ("price_return_30", -1.0),
    "D2_selling_decay": ("region_delta_improvement", 1.0),
    "D3_price_response_failure": ("sell_pressure_absorption_30", 1.0),
    "D4_notional_intensity": ("notional_intensity_30", 1.0),
    "D5_realized_volatility": ("realized_vol_30", 1.0),
    "D6_current_recovery": ("region_rebound_from_low", 1.0),
}

_EXPERT_FEATURES: dict[str, tuple[str, ...]] = {
    mechanism: tuple(dict.fromkeys(column for column, _ in components))
    for mechanism, components in MECHANISM_COMPONENTS.items()
}
_EXPERT_FEATURES["G1_shock_macro_first_sweep"] = (
    *_EXPERT_FEATURES["G1_shock_macro_first_sweep"],
    "is_macro_first_sweep",
)
EXPERT_FEATURES: Mapping[str, tuple[str, ...]] = _EXPERT_FEATURES


@dataclass(frozen=True)
class EmpiricalReference:
    sorted_values: np.ndarray

    @classmethod
    def fit(cls, values: Sequence[float]) -> "EmpiricalReference":
        array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        finite = np.sort(array[np.isfinite(array)])
        if finite.size == 0:
            raise RuntimeError("cannot fit empirical reference without finite values")
        return cls(finite)

    def percentile(self, values: Sequence[float]) -> np.ndarray:
        array = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
        out = np.full(len(array), np.nan, dtype=float)
        finite = np.isfinite(array)
        out[finite] = np.searchsorted(self.sorted_values, array[finite], side="right") / float(
            len(self.sorted_values)
        ) * 100.0
        return out


@dataclass(frozen=True)
class FrozenMechanismScorer:
    references: Mapping[str, EmpiricalReference]
    train_medians: Mapping[str, float]
    minimum_score: float = 70.0

    @classmethod
    def fit(cls, train: pd.DataFrame, *, minimum_score: float = 70.0) -> "FrozenMechanismScorer":
        columns = sorted({column for specs in MECHANISM_COMPONENTS.values() for column, _ in specs})
        references: dict[str, EmpiricalReference] = {}
        medians: dict[str, float] = {}
        missing: list[str] = []
        for column in columns:
            if column not in train.columns:
                missing.append(column)
                continue
            values = pd.to_numeric(train[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            if values.notna().sum() < 20 or values.nunique(dropna=True) <= 1:
                missing.append(column)
                continue
            references[column] = EmpiricalReference.fit(values.to_numpy(dtype=float))
            median = float(values.median())
            medians[column] = median if np.isfinite(median) else 0.0
        required_by_mechanism = {
            name: [column for column, _ in specs if column in references]
            for name, specs in MECHANISM_COMPONENTS.items()
        }
        empty = [name for name, columns_ in required_by_mechanism.items() if len(columns_) < 4]
        if empty:
            raise RuntimeError(
                "mechanism feature coverage too low before expensive model fitting: "
                f"empty_or_thin={empty} missing={missing}"
            )
        return cls(references=references, train_medians=medians, minimum_score=float(minimum_score))

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        for mechanism in MECHANISM_ORDER:
            pieces: list[np.ndarray] = []
            for column, direction in MECHANISM_COMPONENTS[mechanism]:
                reference = self.references.get(column)
                if reference is None or column not in frame.columns:
                    continue
                percentile = reference.percentile(frame[column])
                if float(direction) < 0:
                    percentile = 100.0 - percentile
                pieces.append(percentile)
            if len(pieces) < 4:
                raise RuntimeError(f"{mechanism} lost actual feature coverage during transform")
            matrix = np.vstack(pieces)
            finite = np.isfinite(matrix)
            counts = finite.sum(axis=0)
            score = np.divide(
                np.where(finite, matrix, 0.0).sum(axis=0),
                counts,
                out=np.full(matrix.shape[1], np.nan, dtype=float),
                where=counts > 0,
            )
            if mechanism == "G1_shock_macro_first_sweep" and "is_macro_first_sweep" in frame.columns:
                macro = frame["is_macro_first_sweep"].fillna(False).astype(bool).to_numpy()
                score = np.where(macro, np.maximum(score, 100.0), score)
            column = f"mechanism_score__{mechanism}"
            out[column] = score.astype(np.float32)

        gates = _semantic_gates(frame, self.train_medians)
        for mechanism in MECHANISM_ORDER:
            score = pd.to_numeric(out[f"mechanism_score__{mechanism}"], errors="coerce").to_numpy(dtype=float)
            eligible = np.isfinite(score) & (score >= float(self.minimum_score)) & gates[mechanism]
            if mechanism == "G1_shock_macro_first_sweep" and "is_macro_first_sweep" in frame.columns:
                eligible |= frame["is_macro_first_sweep"].fillna(False).astype(bool).to_numpy()
            out[f"mechanism_eligible__{mechanism}"] = eligible

        eligible_matrix = np.column_stack(
            [out[f"mechanism_eligible__{mechanism}"].to_numpy(dtype=bool) for mechanism in MECHANISM_ORDER]
        )
        score_matrix = np.column_stack(
            [pd.to_numeric(out[f"mechanism_score__{mechanism}"], errors="coerce").to_numpy(dtype=float) for mechanism in MECHANISM_ORDER]
        )
        masked = np.where(eligible_matrix & np.isfinite(score_matrix), score_matrix, -np.inf)
        best = np.argmax(masked, axis=1)
        any_eligible = eligible_matrix.any(axis=1)
        names = np.asarray(MECHANISM_ORDER, dtype=object)
        out["primary_mechanism"] = np.where(any_eligible, names[best], UNRESOLVED_MECHANISM)
        out["mechanism_count"] = eligible_matrix.sum(axis=1).astype(np.int8)
        out["mechanism_max_score"] = np.where(any_eligible, np.max(masked, axis=1), np.nan).astype(np.float32)
        return out


def _numeric(frame: pd.DataFrame, column: str, default: float = np.nan) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), float(default), dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _semantic_gates(frame: pd.DataFrame, medians: Mapping[str, float]) -> Mapping[str, np.ndarray]:
    current_return = _numeric(frame, "current_return_1")
    close_in_bar = _numeric(frame, "current_close_in_bar")
    return15 = _numeric(frame, "price_return_15")
    return30 = _numeric(frame, "price_return_30")
    return60 = _numeric(frame, "price_return_60")
    return120 = _numeric(frame, "price_return_120")
    accel = _numeric(frame, "return_acceleration_5_30")
    delta30 = _numeric(frame, "delta_ratio_30")
    delta_improve = _numeric(frame, "region_delta_improvement")
    absorption = _numeric(frame, "sell_pressure_absorption_30")
    absorption_improve = _numeric(frame, "region_absorption_improvement")
    retests = _numeric(frame, "region_candidate_retest_count", 0.0)
    support = _numeric(frame, "support_test_density_60")
    vol_compression = _numeric(frame, "vol_compression_10_60")
    reclaim10 = _numeric(frame, "region_reclaim_10bp", 0.0)
    htf3 = _numeric(frame, "tf60m_return_3")
    htf6 = _numeric(frame, "tf60m_return_6")
    current_range = _numeric(frame, "current_range_pct")
    range_median = float(medians.get("current_range_pct", 0.0))
    support_median = float(medians.get("support_test_density_60", 0.0))
    accel_median = float(medians.get("return_acceleration_5_30", 0.0))
    absorption_median = float(medians.get("sell_pressure_absorption_30", 0.0))
    macro = (
        frame.get("is_macro_first_sweep", pd.Series(False, index=frame.index))
        .fillna(False)
        .astype(bool)
        .to_numpy()
    )
    return {
        "G1_shock_macro_first_sweep": macro | ((return30 < 0.0) & (current_range >= range_median)),
        "G2_trend_exhaustion": (
            (return60 < 0.0)
            & (return120 < 0.0)
            & (accel >= accel_median)
            & (delta_improve >= 0.0)
        ),
        "G3_base_stabilization": (
            ((retests >= 2.0) | (support >= support_median))
            & (vol_compression <= 1.25)
            & ~macro
        ),
        "G4_absorption_price_response_failure": (
            (delta30 < 0.0)
            & (absorption >= absorption_median)
            & (absorption_improve >= 0.0)
        ),
        "G5_fast_recovery_breakdown_failure": (
            (current_return > 0.0)
            & ((reclaim10 >= 1.0) | (close_in_bar >= 0.65))
            & (accel >= 0.0)
        ),
        "G6_trend_pullback_restart": (
            ((htf3 > 0.0) | (htf6 > 0.0))
            & ((return15 < 0.0) | (return30 < 0.0))
            & (current_return > 0.0)
        ),
    }



def merge_macro_first_sweep_candidates(
    bars: pd.DataFrame,
    candidates: pd.DataFrame,
    sweep_decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Union causal macro first-sweep bars into the broad candidate universe.

    A region's first low-like observation and a later respected-macro first sweep
    are distinct deployable decisions.  The sweep must therefore survive region
    de-duplication without becoming a gate for non-sweep candidates.  Multiple
    respected levels swept on the same closed bar collapse deterministically to
    one 1m event while retaining the swept-level count.
    """

    out = candidates.copy().reset_index(drop=True)
    out["is_macro_first_sweep"] = False
    out["force_include_event"] = False
    out["macro_first_sweep_count"] = 0
    if "level_price" not in out.columns:
        out["level_price"] = np.nan
    if "sweep_low" not in out.columns:
        out["sweep_low"] = np.nan
    if sweep_decisions.empty:
        return out

    sweeps = sweep_decisions.loc[sweep_decisions["decision_path"].eq("sweep")].copy()
    if sweeps.empty:
        return out
    required = {"event_id", "extreme_pos", "extreme_time", "feature_available_time"}
    missing = sorted(required.difference(sweeps.columns))
    if missing:
        raise RuntimeError(f"macro first-sweep candidate merge missing columns: {missing}")

    sweeps["extreme_pos"] = pd.to_numeric(sweeps["extreme_pos"], errors="raise").astype(np.int64)
    sort_columns = ["extreme_pos"]
    ascending = [True]
    if "fse_level_strength" in sweeps.columns:
        sort_columns.append("fse_level_strength")
        ascending.append(False)
    if "level_id" in sweeps.columns:
        sort_columns.append("level_id")
        ascending.append(True)
    sort_columns.append("event_id")
    ascending.append(True)
    sweeps = sweeps.sort_values(sort_columns, ascending=ascending, kind="mergesort")
    counts = sweeps.groupby("extreme_pos", sort=False).size().rename("macro_first_sweep_count")
    representative = sweeps.groupby("extreme_pos", sort=False).head(1).copy()
    representative = representative.merge(counts, on="extreme_pos", how="left", validate="one_to_one")

    positions = pd.to_numeric(out["extreme_pos"], errors="raise").astype(np.int64)
    representative_by_pos = representative.set_index("extreme_pos")
    existing_mask = positions.isin(representative_by_pos.index)
    if existing_mask.any():
        existing_positions = positions.loc[existing_mask]
        out.loc[existing_mask, "is_macro_first_sweep"] = True
        out.loc[existing_mask, "force_include_event"] = True
        out.loc[existing_mask, "macro_first_sweep_count"] = existing_positions.map(counts).to_numpy(dtype=np.int64)
        if "level_price" in representative_by_pos.columns:
            out.loc[existing_mask, "level_price"] = existing_positions.map(representative_by_pos["level_price"]).to_numpy(dtype=float)
        low_values = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float, copy=False)
        out.loc[existing_mask, "sweep_low"] = low_values[existing_positions.to_numpy(dtype=np.int64)]

    missing_representatives = representative.loc[~representative["extreme_pos"].isin(set(positions))].copy()
    if not missing_representatives.empty:
        low_values = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float, copy=False)
        appended = pd.DataFrame(index=np.arange(len(missing_representatives)))
        for column in out.columns:
            appended[column] = np.nan
        appended["event_id"] = [
            f"BROAD_MACRO_FIRST_SWEEP_{int(position)}_{pd.Timestamp(timestamp).strftime('%Y%m%d_%H%M%S')}"
            for position, timestamp in zip(
                missing_representatives["extreme_pos"],
                missing_representatives["extreme_time"],
                strict=True,
            )
        ]
        appended["extreme_pos"] = missing_representatives["extreme_pos"].to_numpy(dtype=np.int64)
        appended["extreme_time"] = pd.to_datetime(missing_representatives["extreme_time"]).to_numpy()
        appended["feature_available_time"] = pd.to_datetime(
            missing_representatives["feature_available_time"]
        ).to_numpy()
        appended["extreme_price"] = low_values[appended["extreme_pos"].to_numpy(dtype=np.int64)]
        appended["sweep_low"] = appended["extreme_price"].to_numpy(dtype=float)
        if "level_price" in missing_representatives.columns:
            appended["level_price"] = pd.to_numeric(
                missing_representatives["level_price"], errors="coerce"
            ).to_numpy(dtype=float)
        appended["is_macro_first_sweep"] = True
        appended["force_include_event"] = True
        appended["macro_first_sweep_count"] = missing_representatives[
            "macro_first_sweep_count"
        ].to_numpy(dtype=np.int64)
        if "confirmation_time" in appended.columns:
            appended["confirmation_time"] = appended["feature_available_time"]
        if "confirmation_available_time" in appended.columns:
            appended["confirmation_available_time"] = appended["feature_available_time"]
        if "completion_bars" in appended.columns:
            appended["completion_bars"] = 0
        if "cluster_id" in appended.columns:
            appended["cluster_id"] = "MACRO_FIRST_SWEEP"
        if "parent_cluster_id" in appended.columns:
            appended["parent_cluster_id"] = "MACRO_FIRST_SWEEP"
        if "split" in appended.columns:
            appended["split"] = ""
        if "candidate_new_low" in appended.columns:
            appended["candidate_new_low"] = True
        if "candidate_near_floor" in appended.columns:
            appended["candidate_near_floor"] = True
        out = pd.concat([out, appended], ignore_index=True, sort=False)

    out["is_macro_first_sweep"] = out["is_macro_first_sweep"].fillna(False).astype(bool)
    out["force_include_event"] = out["force_include_event"].fillna(False).astype(bool)
    out["macro_first_sweep_count"] = pd.to_numeric(
        out["macro_first_sweep_count"], errors="coerce"
    ).fillna(0).astype(np.int16)
    out = out.sort_values(["extreme_pos", "force_include_event", "event_id"], ascending=[True, False, True], kind="mergesort")
    if out["event_id"].duplicated().any():
        raise RuntimeError("macro first-sweep merge produced duplicate event_id")
    return out.reset_index(drop=True)

def mechanism_dictionary() -> pd.DataFrame:
    descriptions = {
        "G1_shock_macro_first_sweep": "abrupt sell shock or respected-macro first sweep",
        "G2_trend_exhaustion": "persistent decline whose recent price/flow pressure is decelerating",
        "G3_base_stabilization": "repeated low tests with compression and causal stabilization",
        "G4_absorption_price_response_failure": "negative aggressive flow with weak price continuation over multiple bars",
        "G5_fast_recovery_breakdown_failure": "failed breakdown followed by immediate causal recovery",
        "G6_trend_pullback_restart": "positive closed 1h context, local pullback, and restart response",
        UNRESOLVED_MECHANISM: "broad candidate that satisfies no frozen mechanism rule",
    }
    rows: list[dict[str, object]] = []
    for mechanism, description in descriptions.items():
        components = MECHANISM_COMPONENTS.get(mechanism, ())
        rows.append(
            {
                "mechanism": mechanism,
                "description": description,
                "components": "|".join(f"{column}:{direction:+.0f}" for column, direction in components),
                "minimum_score": 70.0 if mechanism != UNRESOLVED_MECHANISM else np.nan,
                "future_information_used": False,
                "multi_label_allowed": mechanism != UNRESOLVED_MECHANISM,
            }
        )
    return pd.DataFrame(rows)


def mechanism_overlap_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left in MECHANISM_ORDER:
        left_mask = frame[f"mechanism_eligible__{left}"].astype(bool)
        for right in MECHANISM_ORDER:
            right_mask = frame[f"mechanism_eligible__{right}"].astype(bool)
            intersection = int((left_mask & right_mask).sum())
            union = int((left_mask | right_mask).sum())
            rows.append(
                {
                    "mechanism_left": left,
                    "mechanism_right": right,
                    "intersection_events": intersection,
                    "union_events": union,
                    "jaccard": float(intersection / union) if union else np.nan,
                    "left_share_overlapped": float(intersection / int(left_mask.sum())) if int(left_mask.sum()) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def select_broad_region_events(
    frame: pd.DataFrame,
    *,
    cooldown_bars: int = 15,
) -> pd.DataFrame:
    """Deterministically thin causal region snapshots while preserving sweeps.

    Keeping only a region's first observation would make later stabilization,
    repeated tests, absorption, and recovery features structurally constant.
    This selector therefore keeps causal snapshots separated by a fixed global
    cooldown, plus every forced macro first-sweep event.  No future region end
    or future outcome participates in the decision.
    """

    required = {"causal_region_id", "extreme_pos", "event_id"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"broad event selection missing columns: {missing}")
    if int(cooldown_bars) < 1:
        raise ValueError("cooldown_bars must be >= 1")

    ordered = frame.copy()
    if "force_include_event" not in ordered.columns:
        ordered["force_include_event"] = False
    ordered["force_include_event"] = ordered["force_include_event"].fillna(False).astype(bool)
    ordered = ordered.sort_values(
        ["extreme_pos", "force_include_event", "event_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    if "region_observation_number" in ordered.columns:
        first_mask = pd.to_numeric(ordered["region_observation_number"], errors="coerce").eq(1)
    else:
        first_indices = set(
            ordered.sort_values(["causal_region_id", "extreme_pos", "event_id"], kind="mergesort")
            .groupby("causal_region_id", sort=False)
            .head(1)
            .index
        )
        first_mask = ordered.index.isin(first_indices)

    selected: list[int] = []
    rules: list[str] = []
    last_position = -10**18
    for index, row in ordered.iterrows():
        position = int(row["extreme_pos"])
        is_forced = bool(row["force_include_event"])
        if not is_forced and position - last_position < int(cooldown_bars):
            continue
        selected.append(int(index))
        if is_forced:
            rules.append("forced_macro_first_sweep")
        elif bool(first_mask.loc[index] if isinstance(first_mask, pd.Series) else first_mask[ordered.index.get_loc(index)]):
            rules.append("first_causal_region_observation")
        else:
            rules.append("spaced_causal_region_observation")
        last_position = max(last_position, position)

    out = ordered.loc[selected].reset_index(drop=True)
    out["broad_selection_rule"] = rules
    out["broad_cooldown_bars"] = int(cooldown_bars)
    return out


def fit_simple_factor_references(policy: pd.DataFrame) -> Mapping[str, EmpiricalReference]:
    references: dict[str, EmpiricalReference] = {}
    missing: list[str] = []
    for factor, (column, _) in SIMPLE_FACTOR_SPECS.items():
        if column not in policy.columns:
            missing.append(column)
            continue
        values = pd.to_numeric(policy[column], errors="coerce")
        if values.notna().sum() < 20 or values.nunique(dropna=True) <= 1:
            missing.append(column)
            continue
        references[factor] = EmpiricalReference.fit(values)
    if len(references) != len(SIMPLE_FACTOR_SPECS):
        raise RuntimeError(f"simple-factor baseline lost actual inputs: {missing}")
    return references


def attach_simple_factor_scores(
    frame: pd.DataFrame,
    references: Mapping[str, EmpiricalReference],
) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    for factor, (column, direction) in SIMPLE_FACTOR_SPECS.items():
        percentile = references[factor].percentile(frame[column])
        if direction < 0:
            percentile = 100.0 - percentile
        out[f"simple_factor_score__{factor}"] = percentile.astype(np.float32)
    return out


def _matching_fields(bars: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(bars["close"], errors="coerce")
    ret1 = close.pct_change(fill_method=None)
    return30 = close.pct_change(30, fill_method=None)
    return240 = close.pct_change(240, fill_method=None)
    vol30 = ret1.rolling(30, min_periods=20).std()
    index = pd.DatetimeIndex(bars.index)
    out = pd.DataFrame(index=index)
    out["match_month"] = index.to_period("M").astype(str)
    out["match_session"] = (index.hour // 4).astype(np.int8)
    out["match_decline_bin"] = pd.cut(
        return30,
        bins=[-np.inf, -0.020, -0.010, -0.005, -0.002, 0.0, 0.002, 0.005, 0.010, np.inf],
        labels=False,
        include_lowest=True,
    )
    out["match_vol_bin"] = pd.cut(
        vol30,
        bins=[-np.inf, 0.0005, 0.0008, 0.0012, 0.0018, 0.0025, 0.0035, 0.0050, np.inf],
        labels=False,
        include_lowest=True,
    )
    out["match_state_bin"] = pd.cut(
        return240,
        bins=[-np.inf, -0.030, -0.010, -0.003, 0.003, 0.010, 0.030, np.inf],
        labels=False,
        include_lowest=True,
    )
    out["extreme_pos"] = np.arange(len(out), dtype=np.int64)
    out["extreme_time"] = index
    return out.reset_index(drop=True)


def _key(row: pd.Series, columns: Sequence[str]) -> tuple[object, ...]:
    values: list[object] = []
    for column in columns:
        value = row[column]
        if pd.isna(value):
            values.append(None)
        elif isinstance(value, (np.integer, int)):
            values.append(int(value))
        else:
            values.append(str(value))
    return tuple(values)


def build_matched_random_samples(
    bars: pd.DataFrame,
    events: pd.DataFrame,
    *,
    excluded_positions: Sequence[int],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    maximum_horizon: int = 180,
    replicates: int = 50,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create deterministic matched 1m controls without future outcome matching.

    Matching starts with month, four-hour session, 30m decline, 30m volatility,
    and 240m market-state bins.  Sparse strata fall back in a fixed order while
    preserving month and session.  The fallback level is explicitly audited.
    """

    if replicates < 1:
        raise ValueError("replicates must be >= 1")
    fields = _matching_fields(bars)
    valid = (
        (fields["extreme_time"] >= pd.Timestamp(test_start))
        & (fields["extreme_time"] <= pd.Timestamp(test_end))
        & (fields["extreme_pos"] >= 240)
        & (fields["extreme_pos"] + 1 + int(maximum_horizon) <= len(bars))
    )
    universe = fields.loc[valid].copy()
    excluded = set(int(value) for value in excluded_positions)
    universe = universe.loc[~universe["extreme_pos"].isin(excluded)].reset_index(drop=True)
    if universe.empty:
        raise RuntimeError("matched-random universe is empty")

    event_fields = events[["event_id", "extreme_pos", "extreme_time"]].copy()
    event_fields = event_fields.merge(
        fields.drop(columns=["extreme_time"]),
        on="extreme_pos",
        how="left",
        validate="one_to_one",
    )
    key_levels: tuple[tuple[str, ...], ...] = (
        ("match_month", "match_session", "match_decline_bin", "match_vol_bin", "match_state_bin"),
        ("match_month", "match_session", "match_decline_bin", "match_vol_bin"),
        ("match_month", "match_session", "match_decline_bin"),
        ("match_month", "match_session"),
        ("match_month",),
    )
    pools: list[dict[tuple[object, ...], np.ndarray]] = []
    for columns in key_levels:
        mapping: dict[tuple[object, ...], np.ndarray] = {}
        grouped = universe.groupby(list(columns), dropna=False, sort=False)["extreme_pos"]
        for key_value, positions in grouped:
            values = key_value if isinstance(key_value, tuple) else (key_value,)
            normalized = tuple(None if pd.isna(value) else int(value) if isinstance(value, (np.integer, int)) else str(value) for value in values)
            mapping[normalized] = positions.to_numpy(dtype=np.int64, copy=True)
        pools.append(mapping)

    rng = np.random.default_rng(int(random_state))
    rows: list[dict[str, object]] = []
    fallback_counts = np.zeros(len(key_levels), dtype=np.int64)
    unmatched = 0
    for event in event_fields.itertuples(index=False):
        source = pd.Series(event._asdict())
        chosen_pool: np.ndarray | None = None
        chosen_level = -1
        for level, columns in enumerate(key_levels):
            candidate_pool = pools[level].get(_key(source, columns))
            if candidate_pool is not None and len(candidate_pool):
                chosen_pool = candidate_pool
                chosen_level = level
                break
        if chosen_pool is None:
            unmatched += 1
            continue
        fallback_counts[chosen_level] += 1
        picks = rng.choice(chosen_pool, size=int(replicates), replace=len(chosen_pool) < int(replicates))
        for replicate, position in enumerate(picks):
            timestamp = pd.DatetimeIndex(bars.index)[int(position)]
            rows.append(
                {
                    "event_id": f"MR_{replicate:03d}_{event.event_id}_{int(position)}",
                    "source_event_id": str(event.event_id),
                    "replicate": int(replicate),
                    "extreme_pos": int(position),
                    "extreme_time": timestamp,
                    "feature_available_time": timestamp + pd.Timedelta(minutes=1),
                    "match_fallback_level": int(chosen_level),
                }
            )
    controls = pd.DataFrame(rows)
    diagnostics = pd.DataFrame(
        [
            {"metric": "source_events", "value": int(len(event_fields))},
            {"metric": "matched_rows", "value": int(len(controls))},
            {"metric": "replicates", "value": int(replicates)},
            {"metric": "unmatched_source_events", "value": int(unmatched)},
            {"metric": "eligible_random_universe", "value": int(len(universe))},
            *[
                {"metric": f"fallback_level_{level}_source_events", "value": int(count)}
                for level, count in enumerate(fallback_counts)
            ],
        ]
    )
    if controls.empty or unmatched > max(1, int(0.01 * len(event_fields))):
        raise RuntimeError(
            f"matched-random coverage failed: rows={len(controls)} unmatched={unmatched}/{len(event_fields)}"
        )
    return controls, diagnostics
