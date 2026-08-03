#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reports and fixed validation gates for R07."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .books import BOOK_FEATURE_COLUMNS
from .config import PostSweepFootprintBooksConfig
from .footprint import FOOTPRINT_FEATURE_COLUMNS


PRIMARY_FOOTPRINT_METRICS = (
    "fp_low3_notional_share",
    "fp_low3_sell_share",
    "fp_low3_delta_ratio",
    "fp_low3_large_sell_share",
    "fp_low3_stacked_sell_bins",
    "fp_poc_off_low_bp",
    "fp_sell_poc_off_low_bp",
    "fp_close_off_low_bp",
    "fp_downside_bp_per_sell_million",
    "fp_low3_downside_bp_per_sell_million",
    "fp_impact_ratio_vs_prev_down",
    "fp_low3_sell_vs_prev_down_ratio",
    "fp_low3_delta_improvement_vs_prev_down",
    "fp_poc_shift_vs_prev_down_bp",
    "fp_close_off_low_improvement_vs_prev_down_bp",
)

PRIMARY_BOOK_METRICS = (
    "book_bid_depth_5bps_change",
    "book_bid_depth_5bps_recovery_fraction",
    "book_ask_depth_5bps_change",
    "book_depth_imbalance_change",
    "book_bid_replenished_to_consumed",
    "book_bid_cancel_share_of_removal",
    "book_bid_added_to_removed",
    "book_bid_replenished_per_aggressive_sell",
    "book_aggressive_sell_to_mean_bid_depth_5bps",
)

OUTCOME_COLUMNS = (
    "future_no_lower_low_60m",
    "future_reversal_dominant_60m",
    "future_large_mfe_0p5_180m",
    "future_large_mfe_1_180m",
    "future_large_mfe_2_180m",
)


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _auc(values: pd.Series, labels: pd.Series) -> float:
    x = _numeric(values)
    y = _bool(labels)
    valid = x.notna() & labels.notna()
    x = x.loc[valid]
    y = y.loc[valid]
    positives = int(y.sum())
    negatives = int((~y).sum())
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = x.rank(method="average")
    rank_sum = float(ranks.loc[y].sum())
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def data_quality_report(
    broad_features: pd.DataFrame,
    broad_labels: pd.DataFrame,
    matched_features: pd.DataFrame,
    matched_labels: pd.DataFrame,
    footprint_audit: pd.DataFrame,
    books_audit: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {"check": "broad_attempt_feature_rows", "value": len(broad_features), "status": "PASS" if len(broad_features) else "FAIL"},
        {"check": "broad_attempt_label_rows", "value": len(broad_labels), "status": "PASS" if len(broad_features) == len(broad_labels) else "FAIL"},
        {"check": "matched_feature_rows", "value": len(matched_features), "status": "PASS" if len(matched_features) else "FAIL"},
        {"check": "matched_label_rows", "value": len(matched_labels), "status": "PASS" if len(matched_features) == len(matched_labels) else "FAIL"},
        {
            "check": "broad_footprint_causal_coverage",
            "value": float(broad_features.get("fp_causal_valid", pd.Series(False, index=broad_features.index)).fillna(False).astype(bool).mean()) if len(broad_features) else 0.0,
            "status": "PASS" if len(broad_features) and float(broad_features.get("fp_causal_valid", pd.Series(False, index=broad_features.index)).fillna(False).astype(bool).mean()) >= 0.95 else "WARN",
        },
        {
            "check": "books_broad_causal_rows",
            "value": int(broad_features.get("books_causal_valid", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if len(broad_features) else 0,
            "status": "INFO",
        },
        {
            "check": "books_matched_causal_rows",
            "value": int(matched_features.get("books_causal_valid", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if len(matched_features) else 0,
            "status": "INFO",
        },
        {"check": "footprint_chunks", "value": len(footprint_audit), "status": "INFO"},
        {"check": "books_chunks", "value": len(books_audit), "status": "INFO"},
    ]
    forbidden_broad = [name for name in broad_features.columns if name.startswith("future_")]
    forbidden_matched = [name for name in matched_features.columns if name.startswith("future_")]
    rows.extend(
        [
            {"check": "future_columns_in_broad_features", "value": len(forbidden_broad), "status": "PASS" if not forbidden_broad else "FAIL"},
            {"check": "future_columns_in_matched_features", "value": len(forbidden_matched), "status": "PASS" if not forbidden_matched else "FAIL"},
        ]
    )
    return pd.DataFrame(rows)


def cohort_feature_summary(
    features: pd.DataFrame,
    *,
    metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    if features.empty or "cohort" not in features.columns:
        return pd.DataFrame()
    metric_names = [name for name in (metrics or (*PRIMARY_FOOTPRINT_METRICS, *PRIMARY_BOOK_METRICS)) if name in features.columns]
    rows: list[dict[str, object]] = []
    for (period, cohort), group in features.groupby(["period", "cohort"], dropna=False, sort=False):
        row: dict[str, object] = {"period": period, "cohort": cohort, "events": len(group)}
        for name in metric_names:
            values = _numeric(group[name])
            row[f"median_{name}"] = float(values.median())
            row[f"mean_{name}"] = float(values.mean())
            row[f"non_null_{name}"] = int(values.notna().sum())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_feature_profile(
    features: pd.DataFrame,
    *,
    metrics: Iterable[str] | None = None,
) -> pd.DataFrame:
    if features.empty or "cohort" not in features.columns:
        return pd.DataFrame()
    oracle = features.loc[features["cohort"] == "ORACLE_TURN"].copy()
    prior = features.loc[features["cohort"] == "PRIOR_FAILED_ATTEMPT"].copy()
    if oracle.empty or prior.empty:
        return pd.DataFrame()
    metric_names = [name for name in (metrics or (*PRIMARY_FOOTPRINT_METRICS, *PRIMARY_BOOK_METRICS)) if name in features.columns]
    left = oracle[["pair_id", "period", *metric_names]].rename(columns={name: f"oracle_{name}" for name in metric_names})
    right = prior[["pair_id", *metric_names]].rename(columns={name: f"prior_{name}" for name in metric_names})
    paired = left.merge(right, on="pair_id", how="inner", validate="one_to_one")
    rows: list[dict[str, object]] = []
    grouped = list(paired.groupby("period", sort=False)) + [("ALL", paired)]
    for period, group in grouped:
        for name in metric_names:
            a = _numeric(group[f"oracle_{name}"])
            b = _numeric(group[f"prior_{name}"])
            valid = a.notna() & b.notna()
            if int(valid.sum()) == 0:
                continue
            diff = a.loc[valid] - b.loc[valid]
            rows.append(
                {
                    "period": period,
                    "feature": name,
                    "pairs": int(valid.sum()),
                    "oracle_median": float(a.loc[valid].median()),
                    "prior_median": float(b.loc[valid].median()),
                    "paired_median_difference": float(diff.median()),
                    "paired_mean_difference": float(diff.mean()),
                    "oracle_greater_rate": float((diff > 0).mean()),
                    "oracle_not_equal_rate": float((diff != 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def pair_overlap_summary(features: pd.DataFrame) -> pd.DataFrame:
    if features.empty or "fp_bar_id" not in features.columns:
        return pd.DataFrame()
    oracle = features.loc[features["cohort"] == "ORACLE_TURN", ["pair_id", "period", "fp_bar_id", "fp_end_ts"]].rename(
        columns={"fp_bar_id": "oracle_fp_bar_id", "fp_end_ts": "oracle_fp_end_ts"}
    )
    prior = features.loc[features["cohort"] == "PRIOR_FAILED_ATTEMPT", ["pair_id", "fp_bar_id", "fp_end_ts"]].rename(
        columns={"fp_bar_id": "prior_fp_bar_id", "fp_end_ts": "prior_fp_end_ts"}
    )
    paired = oracle.merge(prior, on="pair_id", how="inner", validate="one_to_one")
    if paired.empty:
        return pd.DataFrame()
    paired["same_completed_footprint_bar"] = (
        pd.to_numeric(paired["oracle_fp_bar_id"], errors="coerce")
        == pd.to_numeric(paired["prior_fp_bar_id"], errors="coerce")
    ) & paired["oracle_fp_bar_id"].notna() & paired["prior_fp_bar_id"].notna()
    rows = []
    for period, group in list(paired.groupby("period", sort=False)) + [("ALL", paired)]:
        rows.append(
            {
                "period": period,
                "pairs": len(group),
                "pairs_with_both_footprints": int((group["oracle_fp_bar_id"].notna() & group["prior_fp_bar_id"].notna()).sum()),
                "same_completed_footprint_bar_rate": float(group["same_completed_footprint_bar"].mean()),
                "distinct_completed_footprint_bar_rate": float((~group["same_completed_footprint_bar"]).mean()),
            }
        )
    return pd.DataFrame(rows)


def feature_outcome_auc(
    broad_features: pd.DataFrame,
    broad_labels: pd.DataFrame,
    *,
    metrics: Iterable[str] = PRIMARY_FOOTPRINT_METRICS,
    outcomes: Iterable[str] = OUTCOME_COLUMNS,
    minimum_events: int = 100,
) -> pd.DataFrame:
    if broad_features.empty or broad_labels.empty:
        return pd.DataFrame()
    labels = broad_labels[[name for name in ("checkpoint_id", "period", *outcomes) if name in broad_labels.columns]].copy()
    frame = broad_features.merge(labels, on=["checkpoint_id", "period"], how="inner", validate="one_to_one")
    rows: list[dict[str, object]] = []
    periods = list(frame["period"].dropna().drop_duplicates()) + ["ALL"]
    for period in periods:
        group = frame if period == "ALL" else frame.loc[frame["period"] == period]
        for feature in metrics:
            if feature not in group.columns:
                continue
            for outcome in outcomes:
                if outcome not in group.columns:
                    continue
                valid = _numeric(group[feature]).notna() & group[outcome].notna()
                if int(valid.sum()) < minimum_events:
                    continue
                auc = _auc(group.loc[valid, feature], group.loc[valid, outcome])
                rows.append(
                    {
                        "period": period,
                        "feature": feature,
                        "outcome": outcome,
                        "events": int(valid.sum()),
                        "positive_rate": float(_bool(group.loc[valid, outcome]).mean()),
                        "auc_high_is_positive": auc,
                        "separation_auc": max(auc, 1.0 - auc) if np.isfinite(auc) else np.nan,
                        "favorable_direction": "HIGH" if np.isfinite(auc) and auc >= 0.5 else "LOW",
                    }
                )
    return pd.DataFrame(rows)


def frozen_quantile_lift(
    broad_features: pd.DataFrame,
    broad_labels: pd.DataFrame,
    *,
    reference_period: str,
    metrics: Iterable[str] = PRIMARY_FOOTPRINT_METRICS,
    outcomes: Iterable[str] = OUTCOME_COLUMNS,
    minimum_events: int = 100,
) -> pd.DataFrame:
    """Freeze one quartile boundary in the reference period and apply unchanged."""

    if broad_features.empty or broad_labels.empty:
        return pd.DataFrame()
    frame = broad_features.merge(
        broad_labels[[name for name in ("checkpoint_id", "period", *outcomes) if name in broad_labels.columns]],
        on=["checkpoint_id", "period"],
        how="inner",
        validate="one_to_one",
    )
    reference = frame.loc[frame["period"] == reference_period]
    if reference.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    periods = list(frame["period"].dropna().drop_duplicates())
    for feature in metrics:
        if feature not in frame.columns:
            continue
        ref_values = _numeric(reference[feature]).dropna()
        if len(ref_values) < minimum_events:
            continue
        q25 = float(ref_values.quantile(0.25))
        q75 = float(ref_values.quantile(0.75))
        for outcome in outcomes:
            if outcome not in frame.columns:
                continue
            ref_valid = _numeric(reference[feature]).notna() & reference[outcome].notna()
            if int(ref_valid.sum()) < minimum_events:
                continue
            auc = _auc(reference.loc[ref_valid, feature], reference.loc[ref_valid, outcome])
            direction = "HIGH" if np.isfinite(auc) and auc >= 0.5 else "LOW"
            threshold = q75 if direction == "HIGH" else q25
            for period in periods:
                group = frame.loc[frame["period"] == period]
                values = _numeric(group[feature])
                valid = values.notna() & group[outcome].notna()
                if int(valid.sum()) < minimum_events:
                    continue
                selected = valid & (values >= threshold if direction == "HIGH" else values <= threshold)
                if int(selected.sum()) == 0:
                    continue
                baseline = float(_bool(group.loc[valid, outcome]).mean())
                rate = float(_bool(group.loc[selected, outcome]).mean())
                rows.append(
                    {
                        "reference_period": reference_period,
                        "period": period,
                        "feature": feature,
                        "outcome": outcome,
                        "favorable_direction": direction,
                        "frozen_threshold": threshold,
                        "eligible_events": int(valid.sum()),
                        "selected_events": int(selected.sum()),
                        "selected_fraction": float(selected.sum() / valid.sum()),
                        "baseline_rate": baseline,
                        "selected_rate": rate,
                        "absolute_lift": rate - baseline,
                    }
                )
    return pd.DataFrame(rows)


def mechanism_scorecard(
    auc_report: pd.DataFrame,
    lift_report: pd.DataFrame,
    paired_report: pd.DataFrame,
    *,
    reference_period: str,
) -> pd.DataFrame:
    features = sorted(
        set(auc_report.get("feature", pd.Series(dtype=str)).dropna().astype(str))
        | set(paired_report.get("feature", pd.Series(dtype=str)).dropna().astype(str))
    )
    rows: list[dict[str, object]] = []
    for feature in features:
        auc_rows = (
            auc_report.loc[(auc_report["feature"] == feature) & (auc_report["period"] != "ALL")]
            if not auc_report.empty and {"feature", "period"}.issubset(auc_report.columns)
            else pd.DataFrame()
        )
        lift_rows = (
            lift_report.loc[(lift_report["feature"] == feature) & (lift_report["period"] != reference_period)]
            if not lift_report.empty and {"feature", "period"}.issubset(lift_report.columns)
            else pd.DataFrame()
        )
        pair_rows = (
            paired_report.loc[(paired_report["feature"] == feature) & (paired_report["period"] != "ALL")]
            if not paired_report.empty and {"feature", "period"}.issubset(paired_report.columns)
            else pd.DataFrame()
        )
        auc_pass = int((pd.to_numeric(auc_rows.get("separation_auc"), errors="coerce") >= 0.55).sum())
        lift_pass = int((pd.to_numeric(lift_rows.get("absolute_lift"), errors="coerce") >= 0.05).sum())
        pair_pass = int(
            (
                (pd.to_numeric(pair_rows.get("oracle_greater_rate"), errors="coerce") >= 0.55)
                | (pd.to_numeric(pair_rows.get("oracle_greater_rate"), errors="coerce") <= 0.45)
            ).sum()
        )
        rows.append(
            {
                "feature": feature,
                "period_outcome_auc_ge_0p55_count": auc_pass,
                "holdout_absolute_lift_ge_5pp_count": lift_pass,
                "paired_directional_rate_ge_55pct_count": pair_pass,
                "max_separation_auc": float(pd.to_numeric(auc_rows.get("separation_auc"), errors="coerce").max()) if len(auc_rows) else np.nan,
                "max_holdout_absolute_lift": float(pd.to_numeric(lift_rows.get("absolute_lift"), errors="coerce").max()) if len(lift_rows) else np.nan,
                "status": "PROMISING_MECHANISM" if auc_pass >= 2 and lift_pass >= 2 and pair_pass >= 2 else "WEAK_OR_UNSTABLE",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["status", "max_separation_auc", "max_holdout_absolute_lift"],
        ascending=[True, False, False],
        kind="mergesort",
    ) if rows else pd.DataFrame()


def causal_audit(broad_features: pd.DataFrame, matched_features: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, frame in (("broad", broad_features), ("matched", matched_features)):
        checkpoint = pd.to_datetime(
            frame["checkpoint_available_time"] if "checkpoint_available_time" in frame.columns else pd.Series(pd.NaT, index=frame.index),
            errors="coerce",
        )
        fp_end = pd.to_datetime(
            frame["fp_end_ts"] if "fp_end_ts" in frame.columns else pd.Series(pd.NaT, index=frame.index),
            errors="coerce",
        )
        fp_violation = fp_end.notna() & checkpoint.notna() & (fp_end > checkpoint)
        book_time = pd.to_datetime(
            frame["book_metric_time"] if "book_metric_time" in frame.columns else pd.Series(pd.NaT, index=frame.index),
            errors="coerce",
        )
        book_violation = book_time.notna() & checkpoint.notna() & (book_time > checkpoint)
        future_cols = [column for column in frame.columns if column.startswith("future_")]
        rows.extend(
            [
                {"table": name, "check": "footprint_end_not_after_checkpoint_available", "violations": int(fp_violation.sum()), "status": "PASS" if not fp_violation.any() else "FAIL"},
                {"table": name, "check": "book_available_not_after_checkpoint_available", "violations": int(book_violation.sum()), "status": "PASS" if not book_violation.any() else "FAIL"},
                {"table": name, "check": "future_columns_absent", "violations": len(future_cols), "status": "PASS" if not future_cols else "FAIL"},
            ]
        )
    return pd.DataFrame(rows)


def build_research_brief(
    *,
    output_path: str | Path,
    broad_features: pd.DataFrame,
    matched_features: pd.DataFrame,
    scorecard: pd.DataFrame,
    overlap: pd.DataFrame,
    books_coverage: pd.DataFrame,
) -> str:
    promising = scorecard.loc[scorecard.get("status") == "PROMISING_MECHANISM"] if not scorecard.empty else pd.DataFrame()
    overlap_all = overlap.loc[overlap.get("period") == "ALL"] if not overlap.empty else pd.DataFrame()
    same_rate = float(overlap_all["same_completed_footprint_bar_rate"].iloc[0]) if len(overlap_all) else np.nan
    books_broad_valid = int(broad_features.get("books_causal_valid", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if len(broad_features) else 0
    books_matched_valid = int(matched_features.get("books_causal_valid", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if len(matched_features) else 0
    decision = "research_continue" if len(promising) else "stop_precise_bottom_unless_books_adds_material_increment"
    text = f"""# R07 Research Brief

## Scope

- Full-history Footprint universe: {len(broad_features):,} causal new-low attempts.
- Matched oracle/prior/control events: {len(matched_features):,}.
- Books causal broad attempts: {books_broad_valid:,}; matched mechanism rows: {books_matched_valid:,}. Books are optional and limited to actual compact-map coverage.
- Same completed Footprint bar rate for oracle/prior pairs: {same_rate:.2%}.

## Causal contract

Only Range Footprints from bars ending no later than `checkpoint_available_time` are strategy-facing. Full containing-bar footprints that finish later are not used. Books rows use their persisted `available_time` and are attached backward only.

## Fixed decision gate

A feature is labelled `PROMISING_MECHANISM` only when it reaches separation AUC >= 0.55 in at least two period/outcome cells, frozen holdout lift >= 5 percentage points in at least two cells, and paired oracle/prior directional separation >= 55% in at least two periods. These gates are fixed before reviewing results.

## Current automated decision

`{decision}`

Promising features: {', '.join(promising['feature'].astype(str).tolist()) if len(promising) else '<none>'}

This is mechanism research, not a final strategy backtest. Oracle cohorts use future outcomes only for retrospective comparison and cannot be used as live signals or risk multipliers.

## Books coverage

Compact liquidity-map days found: {len(books_coverage):,}. Raw Books outside compact-map coverage are not silently downloaded or fabricated.
"""
    Path(output_path).write_text(text, encoding="utf-8")
    return text
