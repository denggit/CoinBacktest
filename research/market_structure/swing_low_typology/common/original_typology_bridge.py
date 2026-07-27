#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Bridge retrospective Swing Low typologies to causal online candidates.

Research 16R infrastructure only. Historical type assignments and future path
outcomes are labels. Candidate generation, feature thresholds, and every input
used by an online condition stop at the current closed bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

EPS = 1e-12

HIERARCHY_COLUMNS: tuple[str, ...] = (
    "stage1_type",
    "stage2_type",
    "stage3_broad_type",
    "stage3_subtype",
)

# Direction is declared from market semantics, not selected from outcomes.
# +1 => high percentile is the hypothesised condition; -1 => low percentile.
SPECIAL_CONDITION_SPECS: Mapping[str, tuple[str, float, str]] = {
    "S1_selling_decay": (
        "region_delta_improvement",
        1.0,
        "recent cumulative delta improves versus the early part of the causal region",
    ),
    "S2_large_selling_decay": (
        "region_large_delta_improvement",
        1.0,
        "recent large-trade delta improves versus the early part of the causal region",
    ),
    "S3_absorption_improvement": (
        "region_absorption_improvement",
        1.0,
        "negative-flow price impact weakens through the causal region",
    ),
    "S4_repeated_low_test": (
        "region_candidate_retest_count",
        1.0,
        "the causal region repeatedly tests a similar running low",
    ),
    "S5_range_compression": (
        "region_range_recent_vs_early",
        -1.0,
        "recent bar range compresses versus the early part of the causal region",
    ),
    "S6_price_response_failure": (
        "sell_pressure_absorption_30",
        1.0,
        "aggressive selling produces weak downward price response over 30 bars",
    ),
    "S7_current_recovery": (
        "region_rebound_from_low",
        1.0,
        "current close has recovered from the running region low",
    ),
    "S8_return_acceleration": (
        "return_acceleration_5_30",
        1.0,
        "very recent return improves relative to the preceding decline",
    ),
    "S9_positive_macro_context": (
        "tf60m_return_3",
        1.0,
        "the last three fully available 60m bars retain positive context",
    ),
    "S10_recent_decline": (
        "price_return_30",
        -1.0,
        "the candidate follows a larger causal 30-bar decline",
    ),
}


@dataclass(frozen=True)
class BridgeFold:
    fold: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def walkforward_folds(end_date: str) -> tuple[BridgeFold, ...]:
    research_end = pd.Timestamp(end_date)
    if len(str(end_date).strip()) <= 10:
        research_end += pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return (
        BridgeFold(
            "WF_2024",
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2023-12-31 23:59:59"),
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-12-31 23:59:59"),
        ),
        BridgeFold(
            "WF_2025",
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2024-12-31 23:59:59"),
            pd.Timestamp("2025-01-01"),
            pd.Timestamp("2025-12-31 23:59:59"),
        ),
        BridgeFold(
            "WF_2026H1",
            pd.Timestamp("2023-01-01"),
            pd.Timestamp("2025-12-31 23:59:59"),
            pd.Timestamp("2026-01-01"),
            research_end,
        ),
    )


def _require_unique(frame: pd.DataFrame, key: str, name: str) -> None:
    if key not in frame.columns:
        raise RuntimeError(f"{name} missing key column {key}")
    duplicate = frame[key].astype(str).duplicated(keep=False)
    if duplicate.any():
        examples = frame.loc[duplicate, key].astype(str).head(10).tolist()
        raise RuntimeError(f"{name} contains duplicate {key}: {examples}")


def build_historical_typology_table(
    stage1: pd.DataFrame,
    stage2: pd.DataFrame,
    stage3_broad: pd.DataFrame,
    stage3_trend: pd.DataFrame,
    stage3_base: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge frozen 01/02/03 assignments without changing their definitions."""

    for name, frame in (
        ("stage1", stage1),
        ("stage2", stage2),
        ("stage3_broad", stage3_broad),
        ("stage3_trend", stage3_trend),
        ("stage3_base", stage3_base),
    ):
        _require_unique(frame, "event_id", name)

    required1 = {
        "event_id",
        "extreme_time",
        "feature_available_time",
        "extreme_pos",
        "extreme_price",
        "cluster_id",
        "split",
    }
    missing1 = sorted(required1.difference(stage1.columns))
    if missing1:
        raise RuntimeError(f"stage1 assignments missing columns: {missing1}")

    base = stage1[
        [
            "event_id",
            "extreme_time",
            "feature_available_time",
            "extreme_pos",
            "extreme_price",
            "confirmation_time",
            "confirmation_available_time",
            "completion_bars",
            "realized_confirmation_move_pct",
            "split",
            "cluster_id",
        ]
    ].copy()
    base = base.rename(columns={"cluster_id": "stage1_type", "split": "stage1_split"})

    stage2_small = stage2[["event_id", "subcluster_id"]].rename(
        columns={"subcluster_id": "stage2_type"}
    )
    broad_small = stage3_broad[["event_id", "mechanism"]].rename(
        columns={"mechanism": "stage3_broad_type"}
    )
    trend_small = stage3_trend[["event_id", "trend_subtype"]].rename(
        columns={"trend_subtype": "stage3_trend_subtype"}
    )
    base_small = stage3_base[["event_id", "base_subtype"]].rename(
        columns={"base_subtype": "stage3_base_subtype"}
    )

    merged = (
        base.merge(stage2_small, on="event_id", how="left", validate="one_to_one")
        .merge(broad_small, on="event_id", how="left", validate="one_to_one")
        .merge(trend_small, on="event_id", how="left", validate="one_to_one")
        .merge(base_small, on="event_id", how="left", validate="one_to_one")
    )
    both_subtypes = merged["stage3_trend_subtype"].notna() & merged["stage3_base_subtype"].notna()
    if both_subtypes.any():
        examples = merged.loc[both_subtypes, "event_id"].head(10).tolist()
        raise RuntimeError(f"events received both trend and base stage3 subtypes: {examples}")
    merged["stage3_subtype"] = merged["stage3_trend_subtype"].combine_first(
        merged["stage3_base_subtype"]
    )
    merged = merged.drop(columns=["stage3_trend_subtype", "stage3_base_subtype"])
    merged["extreme_time"] = pd.to_datetime(merged["extreme_time"], errors="raise")
    merged["feature_available_time"] = pd.to_datetime(
        merged["feature_available_time"], errors="raise"
    )
    merged["extreme_pos"] = pd.to_numeric(merged["extreme_pos"], errors="raise").astype(np.int64)
    merged = merged.sort_values(["extreme_pos", "event_id"], kind="mergesort").reset_index(drop=True)

    stage1_ids = set(stage1["event_id"].astype(str))
    frozen_c3_ids = set(
        merged.loc[merged["stage1_type"].astype(str).eq("C3"), "event_id"].astype(str)
    )
    stage2_ids = set(stage2["event_id"].astype(str))
    broad_ids = set(stage3_broad["event_id"].astype(str))
    expected_trend_ids = set(
        stage2.loc[stage2["subcluster_id"].astype(str).eq("C3-C"), "event_id"].astype(str)
    )
    expected_base_ids = set(
        stage2.loc[stage2["subcluster_id"].astype(str).eq("C3-E"), "event_id"].astype(str)
    )
    trend_ids = set(stage3_trend["event_id"].astype(str))
    base_ids = set(stage3_base["event_id"].astype(str))

    hierarchy_mismatches = {
        "stage2_exactly_frozen_c3": (stage2_ids, frozen_c3_ids),
        "stage3_broad_exactly_stage2": (broad_ids, stage2_ids),
        "stage3_trend_exactly_c3c": (trend_ids, expected_trend_ids),
        "stage3_base_exactly_c3e": (base_ids, expected_base_ids),
    }
    for check, (actual_ids, expected_ids) in hierarchy_mismatches.items():
        if actual_ids != expected_ids:
            missing = sorted(expected_ids.difference(actual_ids))[:10]
            extra = sorted(actual_ids.difference(expected_ids))[:10]
            raise RuntimeError(f"{check} failed: missing={missing}, extra={extra}")
    if set(merged["event_id"].astype(str)) != stage1_ids:
        raise RuntimeError("historical merge changed the frozen stage1 event universe")
    if not merged["extreme_pos"].is_unique:
        examples = merged.loc[merged["extreme_pos"].duplicated(keep=False), "extreme_pos"].head(10).tolist()
        raise RuntimeError(f"frozen historical Swing Lows contain duplicate extreme_pos: {examples}")
    if not merged["extreme_time"].is_unique:
        examples = merged.loc[merged["extreme_time"].duplicated(keep=False), "extreme_time"].head(10).tolist()
        raise RuntimeError(f"frozen historical Swing Lows contain duplicate extreme_time: {examples}")

    feature_lag = merged["feature_available_time"].sub(merged["extreme_time"])
    expected_feature_lag = pd.Timedelta(minutes=1)
    feature_lag_ok = feature_lag.eq(expected_feature_lag).all()
    if not feature_lag_ok:
        examples = feature_lag[~feature_lag.eq(expected_feature_lag)].head(10).astype(str).tolist()
        raise RuntimeError(
            "frozen historical feature_available_time must equal extreme_time + 1m; "
            f"examples={examples}"
        )

    audit = pd.DataFrame(
        [
            {
                "check": "stage1_event_id_and_extreme_unique",
                "passed": True,
                "detail": f"rows={len(stage1):,}",
            },
            {
                "check": "stage2_exactly_frozen_c3",
                "passed": True,
                "detail": f"stage2_rows={len(stage2):,}",
            },
            {
                "check": "stage3_broad_exactly_stage2",
                "passed": True,
                "detail": f"broad_rows={len(stage3_broad):,}",
            },
            {
                "check": "stage3_trend_and_base_exact_subsets",
                "passed": True,
                "detail": f"trend_rows={len(stage3_trend):,} base_rows={len(stage3_base):,}",
            },
            {
                "check": "stage3_subtype_mutually_exclusive",
                "passed": bool(not both_subtypes.any()),
                "detail": "trend and base subtype files never assign the same event",
            },
            {
                "check": "historical_feature_available_time_exact_1m",
                "passed": bool(feature_lag_ok),
                "detail": f"lag={expected_feature_lag}",
            },
        ]
    )
    return merged, audit


def typology_inventory(historical: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = max(1, len(historical))
    for level in HIERARCHY_COLUMNS:
        valid = historical[level].notna()
        level_total = int(valid.sum())
        for type_id, group in historical.loc[valid].groupby(level, sort=True):
            rows.append(
                {
                    "hierarchy_level": level,
                    "type_id": str(type_id),
                    "count": int(len(group)),
                    "share_of_all_swing_lows": float(len(group) / total),
                    "share_within_level": float(len(group) / max(1, level_total)),
                    "first_time": group["extreme_time"].min(),
                    "last_time": group["extreme_time"].max(),
                }
            )
    return pd.DataFrame(rows)


def _nearest_previous_position(
    target_positions: np.ndarray,
    source_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return source index and non-negative bars from source to target."""

    insertion = np.searchsorted(source_positions, target_positions, side="right") - 1
    valid = insertion >= 0
    lead = np.full(len(target_positions), np.nan, dtype=float)
    lead[valid] = target_positions[valid] - source_positions[insertion[valid]]
    return insertion, lead


def attach_event_bridge_flags(
    historical: pd.DataFrame,
    source_events: pd.DataFrame,
    *,
    prefix: str,
    lead_windows: Sequence[int] = (0, 3, 5, 10, 15),
) -> pd.DataFrame:
    """Attach whether each historical Swing Low had a source event at/before it."""

    if historical.empty:
        return historical.copy()
    positions = np.sort(
        pd.to_numeric(source_events.get("extreme_pos", pd.Series(dtype=float)), errors="coerce")
        .dropna()
        .astype(np.int64)
        .unique()
    )
    out = historical.copy()
    target = pd.to_numeric(out["extreme_pos"], errors="raise").to_numpy(dtype=np.int64)
    if positions.size == 0:
        lead = np.full(len(out), np.nan, dtype=float)
    else:
        _, lead = _nearest_previous_position(target, positions)
    out[f"{prefix}_nearest_lead_bars"] = lead
    for window in sorted(set(int(value) for value in lead_windows)):
        out[f"{prefix}_within_{window}b"] = np.isfinite(lead) & (lead <= int(window))
    return out


def bridge_coverage_scorecard(
    event_bridge: pd.DataFrame,
    *,
    prefixes: Sequence[str],
    lead_windows: Sequence[int],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_specs: list[tuple[str, str | None]] = [("all_swing_lows", None)]
    for level in HIERARCHY_COLUMNS:
        for type_id in sorted(event_bridge[level].dropna().astype(str).unique()):
            group_specs.append((level, type_id))

    for level, type_id in group_specs:
        if type_id is None:
            group = event_bridge
        else:
            group = event_bridge[event_bridge[level].astype(str).eq(type_id)]
        if group.empty:
            continue
        for prefix in prefixes:
            lead = pd.to_numeric(group[f"{prefix}_nearest_lead_bars"], errors="coerce")
            for window in sorted(set(int(value) for value in lead_windows)):
                flag = group[f"{prefix}_within_{window}b"].fillna(False).astype(bool)
                rows.append(
                    {
                        "hierarchy_level": level,
                        "type_id": "ALL" if type_id is None else type_id,
                        "source": prefix,
                        "lead_window_bars": int(window),
                        "historical_events": int(len(group)),
                        "covered_events": int(flag.sum()),
                        "recall": float(flag.mean()),
                        "median_nearest_lead_bars": float(lead[flag].median()) if flag.any() else np.nan,
                        "p90_nearest_lead_bars": float(lead[flag].quantile(0.90)) if flag.any() else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def map_candidates_to_future_typology(
    candidates: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    maximum_lead_bars: int = 15,
    price_tolerance_bp: float = 75.0,
) -> pd.DataFrame:
    """Attach the nearest future historical Swing Low as a label only.

    The mapping is deterministic and never feeds candidate generation or a
    feature threshold.  A price-zone check avoids linking a high candidate to a
    materially lower event merely because it occurs within the lead window.
    """

    if maximum_lead_bars < 0 or price_tolerance_bp < 0:
        raise ValueError("maximum_lead_bars and price_tolerance_bp must be non-negative")
    out = candidates.copy()
    out["extreme_pos"] = pd.to_numeric(out["extreme_pos"], errors="raise").astype(np.int64)
    if "extreme_price" not in out.columns:
        raise RuntimeError("candidate frame missing extreme_price for bridge zone check")
    hist = historical.sort_values(["extreme_pos", "event_id"], kind="mergesort").reset_index(drop=True)
    hist_pos = pd.to_numeric(hist["extreme_pos"], errors="raise").to_numpy(dtype=np.int64)
    candidate_pos = out["extreme_pos"].to_numpy(dtype=np.int64)
    index = np.searchsorted(hist_pos, candidate_pos, side="left")
    valid = index < len(hist)
    lead = np.full(len(out), np.nan, dtype=float)
    if valid.any():
        lead[valid] = hist_pos[index[valid]] - candidate_pos[valid]
    valid &= np.isfinite(lead) & (lead >= 0) & (lead <= int(maximum_lead_bars))

    candidate_price = pd.to_numeric(out["extreme_price"], errors="coerce").to_numpy(dtype=float)
    reference_price = np.full(len(out), np.nan, dtype=float)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size:
        reference_price[valid_idx] = pd.to_numeric(
            hist.iloc[index[valid_idx]]["extreme_price"], errors="coerce"
        ).to_numpy(dtype=float)
    price_distance_bp = (candidate_price / np.maximum(reference_price, EPS) - 1.0) * 10_000.0
    valid &= (
        np.isfinite(candidate_price)
        & np.isfinite(reference_price)
        & (np.abs(price_distance_bp) <= float(price_tolerance_bp))
    )

    out["reference_swing_event_id"] = pd.Series(pd.NA, index=out.index, dtype="string")
    out["reference_swing_lead_bars"] = lead
    out["reference_price_distance_bp"] = np.where(valid, price_distance_bp, np.nan)
    for column in HIERARCHY_COLUMNS:
        out[f"reference_{column}"] = pd.Series(pd.NA, index=out.index, dtype="string")

    matched_rows = np.flatnonzero(valid)
    if matched_rows.size:
        ref = hist.iloc[index[matched_rows]].reset_index(drop=True)
        out.loc[out.index[matched_rows], "reference_swing_event_id"] = ref["event_id"].astype(str).to_numpy()
        for column in HIERARCHY_COLUMNS:
            values = ref[column].astype("string").to_numpy()
            out.loc[out.index[matched_rows], f"reference_{column}"] = values
    out["reference_swing_matched"] = valid
    return out




def mapped_source_coverage_scorecard(
    mapped_source: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    source_name: str,
    lead_windows: Sequence[int] = (0, 3, 5, 10, 15),
) -> pd.DataFrame:
    """Measure frozen-type recall using both causal time and price-zone links.

    A source event only covers a historical Swing Low when
    :func:`map_candidates_to_future_typology` linked it to that exact frozen
    event.  This prevents a generic low-like observation many basis points
    above or below the eventual Swing Low from inflating online recall.
    """

    required = {
        "reference_swing_matched",
        "reference_swing_event_id",
        "reference_swing_lead_bars",
    }
    missing = sorted(required.difference(mapped_source.columns))
    if missing:
        raise RuntimeError(f"mapped source missing columns: {missing}")
    historical_ids = historical["event_id"].astype(str)
    if not historical_ids.is_unique:
        raise RuntimeError("historical event_id must be unique for mapped coverage")

    matched = mapped_source.loc[
        mapped_source["reference_swing_matched"].fillna(False).astype(bool)
    ].copy()
    matched["reference_swing_event_id"] = matched["reference_swing_event_id"].astype(str)
    matched["reference_swing_lead_bars"] = pd.to_numeric(
        matched["reference_swing_lead_bars"], errors="raise"
    )
    unknown = sorted(set(matched["reference_swing_event_id"]).difference(set(historical_ids)))
    if unknown:
        raise RuntimeError(f"mapped source linked unknown historical events: {unknown[:10]}")

    rows: list[dict[str, object]] = []
    group_specs: list[tuple[str, str, pd.DataFrame]] = [
        ("all_swing_lows", "ALL", historical)
    ]
    for level in HIERARCHY_COLUMNS:
        valid = historical[historical[level].notna()]
        for type_id, group in valid.groupby(level, sort=True):
            group_specs.append((level, str(type_id), group))

    for window in sorted(set(int(value) for value in lead_windows)):
        if window < 0:
            raise ValueError("lead windows must be non-negative")
        within = matched.loc[matched["reference_swing_lead_bars"] <= window].copy()
        by_event = within.groupby("reference_swing_event_id", sort=False)[
            "reference_swing_lead_bars"
        ].agg(closest_lead_bars="min", earliest_lead_bars="max")
        for level, type_id, group in group_specs:
            group_ids = set(group["event_id"].astype(str))
            covered_ids = sorted(group_ids.intersection(set(by_event.index.astype(str))))
            lead_stats = by_event.loc[covered_ids] if covered_ids else pd.DataFrame()
            rows.append(
                {
                    "hierarchy_level": level,
                    "type_id": type_id,
                    "source": source_name,
                    "bridge_policy": "nearest_future_swing_within_time_and_symmetric_price_zone",
                    "lead_window_bars": window,
                    "historical_events": int(len(group)),
                    "covered_events": int(len(covered_ids)),
                    "recall": float(len(covered_ids) / len(group)) if len(group) else np.nan,
                    "linked_source_rows": int(
                        within["reference_swing_event_id"].isin(covered_ids).sum()
                    ),
                    "median_closest_lead_bars": (
                        float(lead_stats["closest_lead_bars"].median())
                        if not lead_stats.empty
                        else np.nan
                    ),
                    "median_earliest_lead_bars": (
                        float(lead_stats["earliest_lead_bars"].median())
                        if not lead_stats.empty
                        else np.nan
                    ),
                    "p90_earliest_lead_bars": (
                        float(lead_stats["earliest_lead_bars"].quantile(0.90))
                        if not lead_stats.empty
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)

def prepare_sweep_only_events(
    bars: pd.DataFrame,
    decisions: pd.DataFrame,
) -> pd.DataFrame:
    """Return one deterministic, true first-sweep event per closed 1m bar.

    ``build_first_sweep_event_decisions`` also emits delayed reclaim decisions.
    Those are useful for other experiments but must never be counted as First
    Sweep when measuring its share inside the frozen Swing Low typology.
    Multiple respected levels swept on the same bar collapse to one online
    decision, matching the deployable candidate union used by Research 16.
    """

    required = {
        "event_id",
        "decision_path",
        "extreme_pos",
        "extreme_time",
        "feature_available_time",
    }
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise RuntimeError(f"First Sweep decisions missing columns: {missing}")

    sweeps = decisions.loc[decisions["decision_path"].astype(str).eq("sweep")].copy()
    if sweeps.empty:
        empty = sweeps.copy()
        empty["extreme_price"] = pd.Series(dtype=float)
        empty["swept_level_count"] = pd.Series(dtype=np.int64)
        return empty

    positions = pd.to_numeric(sweeps["extreme_pos"], errors="raise").astype(np.int64)
    if ((positions < 0) | (positions >= len(bars))).any():
        bad = positions[(positions < 0) | (positions >= len(bars))].head(10).tolist()
        raise RuntimeError(f"First Sweep positions outside bar frame: {bad}")
    sweeps["extreme_pos"] = positions

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

    counts = sweeps.groupby("extreme_pos", sort=False).size().rename("swept_level_count")
    representative = sweeps.groupby("extreme_pos", sort=False).head(1).copy()
    representative = representative.merge(
        counts,
        left_on="extreme_pos",
        right_index=True,
        how="left",
        validate="one_to_one",
    )
    lows = pd.to_numeric(bars["low"], errors="raise").to_numpy(dtype=float, copy=False)
    representative["extreme_price"] = lows[
        representative["extreme_pos"].to_numpy(dtype=np.int64)
    ]
    if not representative["extreme_pos"].is_unique:
        raise RuntimeError("sweep-only source must contain one event per 1m bar")
    return representative.reset_index(drop=True)


def source_typology_overlap(
    mapped_source: pd.DataFrame,
    historical: pd.DataFrame,
    *,
    source_name: str,
) -> pd.DataFrame:
    """Summarize precise source-to-frozen-typology links.

    ``mapped_source`` must already be mapped by
    :func:`map_candidates_to_future_typology`, including the causal lead window
    and price-zone bound.  Linked historical events are counted uniquely so
    repeated respected levels or multiple source bars cannot inflate the share.
    """

    required = {"reference_swing_matched", "reference_swing_event_id"}
    missing = sorted(required.difference(mapped_source.columns))
    if missing:
        raise RuntimeError(f"mapped source missing columns: {missing}")
    if not historical["event_id"].astype(str).is_unique:
        raise RuntimeError("historical event_id must be unique for source overlap")

    source_events = int(len(mapped_source))
    matched = mapped_source.loc[
        mapped_source["reference_swing_matched"].fillna(False).astype(bool)
    ].copy()
    matched_ids = set(matched["reference_swing_event_id"].dropna().astype(str))
    matched_source_events = int(len(matched))
    linked_total = int(len(matched_ids))

    rows: list[dict[str, object]] = []

    def append_row(level: str, type_id: str, group: pd.DataFrame) -> None:
        group_ids = set(group["event_id"].astype(str))
        linked = int(len(group_ids.intersection(matched_ids)))
        rows.append(
            {
                "source": source_name,
                "hierarchy_level": level,
                "type_id": type_id,
                "source_events": source_events,
                "matched_source_events": matched_source_events,
                "source_event_match_rate": (
                    float(matched_source_events / source_events) if source_events else np.nan
                ),
                "historical_events": int(len(group)),
                "linked_historical_events": linked,
                "linked_share_within_type": float(linked / len(group)) if len(group) else np.nan,
                "share_of_all_linked_swing_lows": (
                    float(linked / linked_total) if linked_total else np.nan
                ),
                "unique_linked_swing_lows": linked_total,
            }
        )

    append_row("all_swing_lows", "ALL", historical)
    for level in HIERARCHY_COLUMNS:
        valid = historical[historical[level].notna()]
        for type_id, group in valid.groupby(level, sort=True):
            append_row(level, str(type_id), group)
    return pd.DataFrame(rows)

def first_sweep_typology_overlap(
    historical_bridge: pd.DataFrame,
    *,
    first_sweep_prefix: str = "first_sweep",
    lead_window_bars: int = 15,
) -> pd.DataFrame:
    flag_col = f"{first_sweep_prefix}_within_{int(lead_window_bars)}b"
    if flag_col not in historical_bridge.columns:
        raise RuntimeError(f"historical bridge missing {flag_col}")
    rows: list[dict[str, object]] = []
    total_flag = historical_bridge[flag_col].fillna(False).astype(bool)
    rows.append(
        {
            "hierarchy_level": "all_swing_lows",
            "type_id": "ALL",
            "historical_events": int(len(historical_bridge)),
            "events_with_first_sweep": int(total_flag.sum()),
            "first_sweep_share_within_type": float(total_flag.mean()),
            "share_of_all_first_sweep_linked_swing_lows": 1.0 if total_flag.any() else np.nan,
        }
    )
    denominator = max(1, int(total_flag.sum()))
    for level in HIERARCHY_COLUMNS:
        for type_id, group in historical_bridge[historical_bridge[level].notna()].groupby(level, sort=True):
            flag = group[flag_col].fillna(False).astype(bool)
            rows.append(
                {
                    "hierarchy_level": level,
                    "type_id": str(type_id),
                    "historical_events": int(len(group)),
                    "events_with_first_sweep": int(flag.sum()),
                    "first_sweep_share_within_type": float(flag.mean()),
                    "share_of_all_first_sweep_linked_swing_lows": float(flag.sum() / denominator),
                }
            )
    return pd.DataFrame(rows)


def path_scorecard_by_original_type(
    frame: pd.DataFrame,
    *,
    horizons: Sequence[int] = (30, 60, 180),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    matched = frame[frame["reference_swing_matched"].fillna(False).astype(bool)].copy()
    if matched.empty:
        return pd.DataFrame()
    lead = pd.to_numeric(matched["reference_swing_lead_bars"], errors="coerce")
    matched["lead_bucket"] = pd.cut(
        lead,
        bins=[-0.5, 0.5, 3.5, 10.5, np.inf],
        labels=["exact", "lead_1_3", "lead_4_10", "lead_11_plus"],
    ).astype("string")

    for level in HIERARCHY_COLUMNS:
        ref_col = f"reference_{level}"
        valid = matched[ref_col].notna()
        for type_id, group in matched.loc[valid].groupby(ref_col, sort=True):
            for lead_bucket, subgroup in group.groupby("lead_bucket", dropna=False, sort=True):
                for horizon in sorted(set(int(value) for value in horizons)):
                    tp_col = f"tp_1_h{horizon}"
                    if tp_col not in subgroup.columns:
                        continue
                    rows.append(
                        {
                            "hierarchy_level": level,
                            "type_id": str(type_id),
                            "lead_bucket": str(lead_bucket),
                            "horizon_bars": horizon,
                            "candidate_events": int(len(subgroup)),
                            "unique_swing_events": int(subgroup["reference_swing_event_id"].nunique()),
                            "tp_0p5_rate": float(subgroup[f"tp_0p5_h{horizon}"].mean()),
                            "tp_1p0_rate": float(subgroup[tp_col].mean()),
                            "tp_1p5_rate": float(subgroup[f"tp_1p5_h{horizon}"].mean()),
                            "tp_2p0_rate": float(subgroup[f"tp_2_h{horizon}"].mean()),
                            "median_mfe_pct": float(pd.to_numeric(subgroup[f"mfe_h{horizon}_pct"], errors="coerce").median()),
                            "median_mae_pct": float(pd.to_numeric(subgroup[f"mae_h{horizon}_pct"], errors="coerce").median()),
                            "mean_mae_before_tp_1pct": float(pd.to_numeric(subgroup[f"mae_before_tp_1_h{horizon}_pct"], errors="coerce").mean()),
                            "clean_0p5_rate": float(subgroup[f"clean_0p5_h{horizon}"].mean()),
                            "deep_sweep_recovery_rate": float(subgroup[f"deep_sweep_recovery_h{horizon}"].mean()),
                            "permanent_failure_rate": float(subgroup[f"permanent_failure_h{horizon}"].mean()),
                            "median_lead_bars": float(pd.to_numeric(subgroup["reference_swing_lead_bars"], errors="coerce").median()),
                        }
                    )
    return pd.DataFrame(rows)


def _directed_threshold(series: pd.Series, *, direction: float, top_pct: int) -> float:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(numeric) < 20 or numeric.nunique() <= 1:
        return np.nan
    quantile = 1.0 - float(top_pct) / 100.0 if direction > 0 else float(top_pct) / 100.0
    return float(numeric.quantile(quantile))


def _condition_mask(series: pd.Series, *, threshold: float, direction: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if direction > 0:
        return numeric >= threshold
    return numeric <= threshold


def special_condition_scorecard(
    frame: pd.DataFrame,
    *,
    end_date: str,
    top_pcts: Sequence[int] = (20, 30, 40),
    minimum_type_test_rows: int = 20,
    horizon: int = 60,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Frozen one-factor scans inside original typology labels.

    Thresholds use only the causal feature distribution in each fold's train
    period. They do not use type labels, TP outcomes, or test rows.
    """

    required_label = f"tp_1_h{int(horizon)}"
    if required_label not in frame.columns:
        raise RuntimeError(f"condition scorecard missing {required_label}")
    rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    times = pd.to_datetime(frame["extreme_time"], errors="raise")
    label_end = pd.to_datetime(frame["label_end_time"], errors="raise")

    for fold in walkforward_folds(end_date):
        train_mask = (times >= fold.train_start) & (times <= fold.train_end) & (label_end <= fold.train_end)
        test_mask = (times >= fold.test_start) & (times <= fold.test_end) & (label_end <= fold.test_end)
        train = frame.loc[train_mask]
        test = frame.loc[test_mask]
        if train.empty or test.empty:
            continue
        for condition_id, (feature, direction, description) in SPECIAL_CONDITION_SPECS.items():
            if feature not in frame.columns:
                threshold_rows.append(
                    {
                        "fold": fold.fold,
                        "condition_id": condition_id,
                        "feature": feature,
                        "direction": direction,
                        "top_pct": np.nan,
                        "threshold": np.nan,
                        "status": "missing_feature",
                    }
                )
                continue
            for top_pct in sorted(set(int(value) for value in top_pcts)):
                threshold = _directed_threshold(train[feature], direction=direction, top_pct=top_pct)
                threshold_rows.append(
                    {
                        "fold": fold.fold,
                        "condition_id": condition_id,
                        "feature": feature,
                        "description": description,
                        "direction": direction,
                        "top_pct": top_pct,
                        "threshold": threshold,
                        "train_rows": int(len(train)),
                        "status": "ok" if np.isfinite(threshold) else "degenerate",
                    }
                )
                if not np.isfinite(threshold):
                    continue
                selected_all = _condition_mask(test[feature], threshold=threshold, direction=direction).fillna(False)
                for level in HIERARCHY_COLUMNS:
                    ref_col = f"reference_{level}"
                    for type_id in sorted(test[ref_col].dropna().astype(str).unique()):
                        type_mask = test[ref_col].astype(str).eq(type_id)
                        type_count = int(type_mask.sum())
                        if type_count < int(minimum_type_test_rows):
                            continue
                        selected_type = type_mask & selected_all
                        base_tp = float(test.loc[type_mask, required_label].mean())
                        selected_tp = (
                            float(test.loc[selected_type, required_label].mean())
                            if selected_type.any()
                            else np.nan
                        )
                        base_mae = float(
                            pd.to_numeric(test.loc[type_mask, f"mae_h{horizon}_pct"], errors="coerce").mean()
                        )
                        selected_mae = (
                            float(
                                pd.to_numeric(
                                    test.loc[selected_type, f"mae_h{horizon}_pct"], errors="coerce"
                                ).mean()
                            )
                            if selected_type.any()
                            else np.nan
                        )
                        type_positive = type_mask & test[required_label].astype(bool)
                        online_base_precision = float(type_positive.mean())
                        online_selected_precision = (
                            float(type_positive[selected_all].mean()) if selected_all.any() else np.nan
                        )
                        rows.append(
                            {
                                "fold": fold.fold,
                                "hierarchy_level": level,
                                "type_id": type_id,
                                "condition_id": condition_id,
                                "feature": feature,
                                "direction": direction,
                                "top_pct": top_pct,
                                "threshold": threshold,
                                "test_candidates": int(len(test)),
                                "selected_all_candidates": int(selected_all.sum()),
                                "selected_all_share": float(selected_all.mean()),
                                "type_candidates": type_count,
                                "selected_type_candidates": int(selected_type.sum()),
                                "selected_type_share": float(selected_type.sum() / max(1, type_count)),
                                "within_type_tp_rate_base": base_tp,
                                "within_type_tp_rate_selected": selected_tp,
                                "within_type_tp_uplift_pp": (selected_tp - base_tp) * 100.0 if np.isfinite(selected_tp) else np.nan,
                                "within_type_mean_mae_base_pct": base_mae,
                                "within_type_mean_mae_selected_pct": selected_mae,
                                "within_type_mae_change_pp": selected_mae - base_mae if np.isfinite(selected_mae) else np.nan,
                                "online_type_tp_precision_base": online_base_precision,
                                "online_type_tp_precision_selected": online_selected_precision,
                                "online_type_tp_uplift_pp": (
                                    (online_selected_precision - online_base_precision) * 100.0
                                    if np.isfinite(online_selected_precision)
                                    else np.nan
                                ),
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(threshold_rows)


def summarize_condition_candidates(scorecard: pd.DataFrame) -> pd.DataFrame:
    if scorecard.empty:
        return pd.DataFrame()
    group_columns = [
        "hierarchy_level",
        "type_id",
        "condition_id",
        "feature",
        "direction",
        "top_pct",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in scorecard.groupby(group_columns, sort=True, dropna=False):
        values = dict(zip(group_columns, keys))
        valid = group.dropna(
            subset=[
                "within_type_tp_uplift_pp",
                "within_type_mae_change_pp",
                "online_type_tp_uplift_pp",
            ]
        )
        if valid.empty:
            continue
        positive_folds = int((valid["within_type_tp_uplift_pp"] > 0.0).sum())
        nonworse_mae_folds = int((valid["within_type_mae_change_pp"] <= 0.10).sum())
        rows.append(
            {
                **values,
                "folds_available": int(valid["fold"].nunique()),
                "positive_within_type_tp_folds": positive_folds,
                "nonworse_mae_folds": nonworse_mae_folds,
                "median_within_type_tp_uplift_pp": float(valid["within_type_tp_uplift_pp"].median()),
                "minimum_within_type_tp_uplift_pp": float(valid["within_type_tp_uplift_pp"].min()),
                "median_online_type_tp_uplift_pp": float(valid["online_type_tp_uplift_pp"].median()),
                "median_mae_change_pp": float(valid["within_type_mae_change_pp"].median()),
                "minimum_selected_type_candidates": int(valid["selected_type_candidates"].min()),
                "bridge_candidate_status": (
                    "candidate"
                    if (
                        valid["fold"].nunique() >= 2
                        and positive_folds >= 2
                        and nonworse_mae_folds >= 2
                        and float(valid["within_type_tp_uplift_pp"].median()) >= 2.0
                        and int(valid["selected_type_candidates"].min()) >= 20
                    )
                    else "not_supported"
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        [
            "bridge_candidate_status",
            "median_within_type_tp_uplift_pp",
            "median_online_type_tp_uplift_pp",
        ],
        ascending=[True, False, False],
        kind="mergesort",
    ).reset_index(drop=True)


def bridge_causal_audit(
    candidates: pd.DataFrame,
    feature_columns: Sequence[str],
    *,
    htf_alignment_audit: pd.DataFrame,
) -> pd.DataFrame:
    forbidden_tokens = (
        "future",
        "reference_",
        "tp_",
        "mfe",
        "mae",
        "label_",
        "completion",
        "confirmation",
    )
    forbidden = [
        column
        for column in feature_columns
        if any(token in str(column).lower() for token in forbidden_tokens)
    ]
    feature_time = pd.to_datetime(candidates["feature_available_time"], errors="raise")
    extreme_time = pd.to_datetime(candidates["extreme_time"], errors="raise")
    feature_lag = feature_time.sub(extreme_time)
    htf_passed = bool(
        htf_alignment_audit.empty
        or (
            "passed" in htf_alignment_audit.columns
            and htf_alignment_audit["passed"].fillna(False).astype(bool).all()
        )
    )
    return pd.DataFrame(
        [
            {
                "check": "candidate_feature_available_exactly_next_1m_boundary",
                "passed": bool(feature_lag.eq(pd.Timedelta(minutes=1)).all()),
                "detail": f"min_lag={feature_lag.min()} max_lag={feature_lag.max()}",
            },
            {
                "check": "future_typology_labels_excluded_from_condition_features",
                "passed": bool(not forbidden),
                "detail": "forbidden=" + "|".join(forbidden),
            },
            {
                "check": "htf_available_time_alignment",
                "passed": htf_passed,
                "detail": f"audit_rows={len(htf_alignment_audit)}",
            },
            {
                "check": "reference_mapping_is_label_only",
                "passed": True,
                "detail": "nearest future historical Swing Low is attached only after candidate features and path labels are frozen",
            },
            {
                "check": "condition_thresholds_are_train_distribution_only",
                "passed": True,
                "detail": "thresholds are fold-train feature quantiles; no target or type label selects direction or threshold",
            },
        ]
    )
