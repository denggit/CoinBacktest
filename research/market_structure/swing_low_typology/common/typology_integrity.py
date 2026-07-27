#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Integrity audits for sparse original-typology online models.

The helpers in this module are deliberately independent from Research 18's
trading rules.  They fix label-availability timing, build deterministic placebo
labels, report episode-level ranking quality, and extract linear-model
coefficients without allowing any future label into the feature matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

EPS = 1e-12


@dataclass(frozen=True)
class PlaceboSpec:
    placebo_id: str
    kind: str
    random_state: int | None = None
    shift_days: int | None = None


FROZEN_PLACEBO_SPECS: tuple[PlaceboSpec, ...] = (
    PlaceboSpec("PERMUTE_MONTH_VOL_1", "stratified_permutation", random_state=1801),
    PlaceboSpec("PERMUTE_MONTH_VOL_2", "stratified_permutation", random_state=1802),
    PlaceboSpec("PERMUTE_MONTH_VOL_3", "stratified_permutation", random_state=1803),
    PlaceboSpec("SHIFT_PLUS_1D", "time_shift", shift_days=1),
    PlaceboSpec("SHIFT_PLUS_7D", "time_shift", shift_days=7),
)


def attach_true_type_label_availability(
    frame: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    bridge_maximum_lead_bars: int = 15,
) -> pd.DataFrame:
    """Attach the earliest timestamp at which a future original-type label exists.

    For an unmatched online candidate, the one-vs-rest negative type label is
    only resolved after the full bridge window has elapsed.  For a matched
    candidate, the historical Swing Low itself is not known until its frozen
    +1% confirmation is available.  The correct availability is therefore the
    maximum of those two timestamps.
    """

    required_frame = {
        "extreme_time",
        "reference_swing_matched",
        "reference_swing_event_id",
    }
    missing = sorted(required_frame.difference(frame.columns))
    if missing:
        raise RuntimeError(f"type-label availability frame missing columns: {missing}")
    required_hist = {"event_id", "confirmation_available_time"}
    missing = sorted(required_hist.difference(historical.columns))
    if missing:
        raise RuntimeError(f"historical typology missing columns: {missing}")
    if not historical["event_id"].astype(str).is_unique:
        raise RuntimeError("historical event_id must be unique for type-label availability")

    out = frame.copy()
    extreme = pd.to_datetime(out["extreme_time"], errors="raise")
    bridge_end = extreme + pd.Timedelta(minutes=int(bridge_maximum_lead_bars))
    lookup = historical.set_index(historical["event_id"].astype(str))[
        "confirmation_available_time"
    ]
    reference_id = out["reference_swing_event_id"].astype("string")
    confirmation = pd.to_datetime(reference_id.map(lookup), errors="coerce")
    matched = out["reference_swing_matched"].fillna(False).astype(bool)
    missing_confirmation = matched & confirmation.isna()
    if missing_confirmation.any():
        examples = reference_id[missing_confirmation].dropna().astype(str).head(10).tolist()
        raise RuntimeError(
            "matched original-type rows lack confirmation_available_time: "
            f"count={int(missing_confirmation.sum())} examples={examples}"
        )
    resolved = bridge_end.copy()
    matched_index = np.flatnonzero(matched.to_numpy())
    if matched_index.size:
        bridge_values = bridge_end.iloc[matched_index].to_numpy(dtype="datetime64[ns]")
        confirmation_values = confirmation.iloc[matched_index].to_numpy(dtype="datetime64[ns]")
        resolved.iloc[matched_index] = pd.to_datetime(
            np.maximum(bridge_values, confirmation_values)
        )
    out["type_bridge_end_time"] = bridge_end
    out["reference_confirmation_available_time"] = confirmation
    out["type_label_end_time_legacy"] = out.get("type_label_end_time", pd.NaT)
    out["type_label_end_time"] = pd.to_datetime(resolved, errors="raise")
    if (out["type_label_end_time"] < bridge_end).any():
        raise RuntimeError("type_label_end_time precedes bridge end")
    if matched.any() and (
        out.loc[matched, "type_label_end_time"]
        < out.loc[matched, "reference_confirmation_available_time"]
    ).any():
        raise RuntimeError("type_label_end_time precedes historical confirmation")
    return out


def label_availability_audit(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "extreme_time",
        "type_bridge_end_time",
        "type_label_end_time",
        "reference_swing_matched",
        "reference_confirmation_available_time",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise RuntimeError(f"label availability audit missing columns: {missing}")
    bridge = pd.to_datetime(frame["type_bridge_end_time"], errors="raise")
    resolved = pd.to_datetime(frame["type_label_end_time"], errors="raise")
    matched = frame["reference_swing_matched"].fillna(False).astype(bool)
    confirmation = pd.to_datetime(
        frame["reference_confirmation_available_time"], errors="coerce"
    )
    legacy = pd.to_datetime(frame.get("type_label_end_time_legacy"), errors="coerce")
    changed = legacy.notna() & resolved.ne(legacy)
    lag_minutes = (resolved - bridge).dt.total_seconds() / 60.0
    return pd.DataFrame(
        [
            {
                "check": "type_label_not_before_bridge_window_end",
                "passed": bool((resolved >= bridge).all()),
                "detail": f"rows={len(frame):,}",
            },
            {
                "check": "matched_type_label_not_before_swing_confirmation",
                "passed": bool(
                    not matched.any()
                    or (
                        resolved.loc[matched] >= confirmation.loc[matched]
                    ).all()
                ),
                "detail": f"matched_rows={int(matched.sum()):,}",
            },
            {
                "check": "legacy_type_label_timing_corrected",
                "passed": True,
                "detail": (
                    f"changed_rows={int(changed.sum()):,}; "
                    f"median_extra_lag_min={float(lag_minutes[matched].median()) if matched.any() else np.nan:.3f}; "
                    f"p90_extra_lag_min={float(lag_minutes[matched].quantile(0.90)) if matched.any() else np.nan:.3f}"
                ),
            },
        ]
    )


def _volatility_bucket(frame: pd.DataFrame, column: str = "realized_vol_30") -> pd.Series:
    if column not in frame.columns:
        return pd.Series("missing", index=frame.index, dtype="string")
    values = pd.to_numeric(frame[column], errors="coerce")
    finite = values[np.isfinite(values)]
    if finite.nunique() < 5:
        return pd.Series("single", index=frame.index, dtype="string")
    ranked = values.rank(method="first", pct=True)
    bucket = np.minimum(np.floor(ranked.fillna(0.5) * 5.0), 4).astype(int)
    return bucket.astype(str).astype("string")


def stratified_permutation_target(
    frame: pd.DataFrame,
    target_column: str,
    *,
    random_state: int,
    volatility_column: str = "realized_vol_30",
) -> pd.Series:
    """Permute labels within month and volatility quintile deterministically."""

    if target_column not in frame.columns:
        raise RuntimeError(f"placebo target missing: {target_column}")
    target = frame[target_column]
    if target.isna().any():
        raise RuntimeError(f"placebo target contains NA: {target_column}")
    month = pd.to_datetime(frame["extreme_time"], errors="raise").dt.to_period("M").astype(str)
    volatility = _volatility_bucket(frame, volatility_column)
    # Group on positional indices, never on the caller's index labels.  Research
    # frames frequently retain sparse/non-zero-based indices after temporal
    # masks; treating those labels as NumPy positions silently corrupts the
    # placebo or raises out-of-bounds errors.
    group_key = (month.astype(str) + "|" + volatility.astype(str)).reset_index(drop=True)
    rng = np.random.default_rng(int(random_state))
    source = target.astype(bool).to_numpy(copy=True)
    output = source.copy()
    for _, positions in group_key.groupby(group_key, sort=True).groups.items():
        loc = np.fromiter(positions, dtype=np.int64)
        if loc.size > 1:
            output[loc] = source[rng.permutation(loc)]
    return pd.Series(output, index=frame.index, dtype=bool)


def time_shift_placebo_target(
    frame: pd.DataFrame,
    target_column: str,
    *,
    shift_days: int,
) -> tuple[pd.Series, pd.Series]:
    """Assign the first candidate label at or after ``time + shift``.

    The returned validity mask excludes rows whose shifted time lies outside the
    available candidate sequence.  This preserves coarse calendar/regime
    structure while destroying the event-local association.
    """

    if int(shift_days) <= 0:
        raise ValueError("shift_days must be positive")
    ordered = frame.sort_values(["extreme_time", "event_id"], kind="mergesort")
    times = pd.to_datetime(ordered["extreme_time"], errors="raise").to_numpy(
        dtype="datetime64[ns]"
    )
    shifted = times + np.timedelta64(int(shift_days), "D")
    index = np.searchsorted(times, shifted, side="left")
    valid = index < len(ordered)
    source = ordered[target_column]
    if source.isna().any():
        raise RuntimeError(f"time-shift target contains NA: {target_column}")
    values = np.zeros(len(ordered), dtype=bool)
    values[valid] = source.astype(bool).to_numpy()[index[valid]]
    result = pd.Series(values, index=ordered.index, dtype=bool).reindex(frame.index)
    validity = pd.Series(valid, index=ordered.index, dtype=bool).reindex(frame.index)
    return result, validity


def episode_ranking_metrics(
    frame: pd.DataFrame,
    *,
    target_column: str,
    score: Sequence[float] | np.ndarray,
    episode_column: str = "reference_swing_event_id",
) -> dict[str, float | int]:
    """Evaluate one score per unique positive Swing Low and one per negative region."""

    score_array = np.asarray(score, dtype=float)
    if len(score_array) != len(frame):
        raise ValueError("episode metric score length mismatch")
    target = frame[target_column].fillna(False).astype(bool).to_numpy()
    region = frame.get("causal_region_id", frame["event_id"]).astype(str).to_numpy()
    episode = frame.get(
        episode_column, pd.Series(pd.NA, index=frame.index, dtype="string")
    ).astype("string")
    keys = np.where(
        target & episode.notna().to_numpy(),
        "POS|" + episode.fillna("").astype(str).to_numpy(),
        "NEG|" + region,
    )
    table = pd.DataFrame({"key": keys, "target": target, "score": score_array})
    table = table[np.isfinite(table["score"])].copy()
    if table.empty:
        return {
            "episode_rows": 0,
            "episode_positives": 0,
            "episode_roc_auc": np.nan,
            "episode_average_precision": np.nan,
        }
    grouped = table.groupby("key", sort=False).agg(
        target=("target", "max"), score=("score", "max")
    )
    y = grouped["target"].astype(int).to_numpy()
    s = grouped["score"].to_numpy(dtype=float)
    auc = float(roc_auc_score(y, s)) if np.unique(y).size == 2 else np.nan
    ap = float(average_precision_score(y, s)) if y.sum() else np.nan
    return {
        "episode_rows": int(len(grouped)),
        "episode_positives": int(y.sum()),
        "episode_roc_auc": auc,
        "episode_average_precision": ap,
    }


def extract_linear_coefficients(
    fitted_model: object,
    *,
    fold: str,
    expert_id: str,
    head: str,
) -> pd.DataFrame:
    """Extract deterministic standardized coefficients when the model is linear."""

    model = getattr(fitted_model, "model", fitted_model)
    feature_columns = tuple(getattr(fitted_model, "feature_columns", ()))
    pipeline_steps = getattr(model, "named_steps", None)
    estimator = pipeline_steps.get("model") if pipeline_steps is not None else model
    coef = getattr(estimator, "coef_", None)
    if coef is None or not feature_columns:
        return pd.DataFrame(
            [
                {
                    "fold": fold,
                    "expert_id": expert_id,
                    "head": head,
                    "feature": "",
                    "coefficient": np.nan,
                    "coefficient_sign": 0,
                    "status": "nonlinear_or_unavailable",
                }
            ]
        )
    values = np.asarray(coef, dtype=float)
    if values.ndim == 2:
        values = values[0]
    if values.size != len(feature_columns):
        raise RuntimeError("linear coefficient count does not match selected features")
    return pd.DataFrame(
        {
            "fold": fold,
            "expert_id": expert_id,
            "head": head,
            "feature": feature_columns,
            "coefficient": values,
            "coefficient_sign": np.sign(values).astype(int),
            "status": "ok",
        }
    )


def coefficient_stability_summary(coefficients: pd.DataFrame) -> pd.DataFrame:
    valid = coefficients[coefficients["status"].eq("ok")].copy()
    if valid.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for keys, group in valid.groupby(["expert_id", "head", "feature"], sort=True):
        expert_id, head, feature = keys
        signs = pd.to_numeric(group["coefficient_sign"], errors="coerce").dropna()
        nonzero = signs[signs.ne(0)]
        dominant_share = (
            float(nonzero.value_counts(normalize=True).max()) if len(nonzero) else np.nan
        )
        rows.append(
            {
                "expert_id": expert_id,
                "head": head,
                "feature": feature,
                "folds_available": int(group["fold"].nunique()),
                "median_coefficient": float(group["coefficient"].median()),
                "minimum_coefficient": float(group["coefficient"].min()),
                "maximum_coefficient": float(group["coefficient"].max()),
                "dominant_sign_share": dominant_share,
                "sign_stable_all_folds": bool(
                    len(nonzero) >= 2 and nonzero.nunique() == 1
                ),
            }
        )
    return pd.DataFrame(rows)


def feature_group_map(feature_columns: Sequence[str]) -> Mapping[str, tuple[str, ...]]:
    groups: dict[str, list[str]] = {
        "CURRENT_VOLATILITY": [],
        "PRICE_PATH": [],
        "ORDER_FLOW": [],
        "REGION_PROCESS": [],
        "SESSION_HTF": [],
    }
    for column in feature_columns:
        name = str(column)
        if name.startswith("region_"):
            groups["REGION_PROCESS"].append(name)
        elif name.startswith(("session_", "tf15m_", "tf60m_")):
            groups["SESSION_HTF"].append(name)
        elif any(
            token in name
            for token in (
                "delta",
                "buy_ratio",
                "notional",
                "trades",
                "absorption",
            )
        ):
            groups["ORDER_FLOW"].append(name)
        elif any(token in name for token in ("vol", "range_pct", "range_position")):
            groups["CURRENT_VOLATILITY"].append(name)
        else:
            groups["PRICE_PATH"].append(name)
    return {key: tuple(value) for key, value in groups.items()}
