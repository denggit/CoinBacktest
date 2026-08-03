#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report builders for R06 post-sweep micro turning-point research."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

CAUSAL_TRIGGERS: tuple[str, ...] = (
    "FIRST_NEW_LOW",
    "IMPACT_COLLAPSE_67",
    "IMPACT_COLLAPSE_50",
    "IMPACT_COLLAPSE_50_HIGH_BREAK",
    "MICRO_RECLAIM_5S",
    "MINUTE_CLOSE",
)


def raw_hourly_coverage_report(raw_coverage: pd.DataFrame) -> pd.DataFrame:
    """Summarize raw-window matching by project-local start hour."""

    columns = [
        "project_hour", "attempted_windows", "checked_windows_with_raw_day",
        "matched_raw_windows", "no_trade_windows", "missing_raw_windows",
        "no_trade_rate", "match_rate",
    ]
    if raw_coverage.empty or "start_time" not in raw_coverage.columns:
        return pd.DataFrame(columns=columns)
    frame = raw_coverage.copy()
    frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce")
    frame = frame.loc[frame["start_time"].notna()].copy()
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["project_hour"] = frame["start_time"].dt.hour.astype("int64")
    status = frame.get("status", pd.Series(index=frame.index, dtype="object")).astype(str)
    frame["matched"] = status.eq("complete")
    frame["no_trade"] = status.eq("no_trades_in_window")
    frame["missing"] = status.eq("missing_raw_day")
    rows: list[dict[str, object]] = []
    for hour, group in frame.groupby("project_hour", sort=True):
        matched = int(group["matched"].sum())
        no_trade = int(group["no_trade"].sum())
        missing = int(group["missing"].sum())
        checked = matched + no_trade
        attempted = int(len(group))
        rows.append(
            {
                "project_hour": int(hour),
                "attempted_windows": attempted,
                "checked_windows_with_raw_day": checked,
                "matched_raw_windows": matched,
                "no_trade_windows": no_trade,
                "missing_raw_windows": missing,
                "no_trade_rate": float(no_trade / checked) if checked else np.nan,
                "match_rate": float(matched / attempted) if attempted else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def validate_micro_data_gate(
    universe: pd.DataFrame,
    window_features: pd.DataFrame,
    triggers: pd.DataFrame,
    raw_coverage: pd.DataFrame,
    *,
    max_no_trade_rate: float = 0.02,
    min_checked_windows: int = 10,
    max_hour_no_trade_rate: float = 0.10,
    min_hour_windows: int = 20,
) -> dict[str, object]:
    """Fail fast on timestamp, timezone, or archive-partition misalignment."""

    attempted = int(len(universe))
    complete = int(len(window_features))
    trigger_rows = int(len(triggers))
    if raw_coverage.empty:
        raise RuntimeError(
            "R06 micro data gate failed: raw coverage is empty. "
            "The raw-trade loader did not audit any event window."
        )

    status = raw_coverage.get("status", pd.Series(dtype="object")).astype(str)
    missing = int(status.eq("missing_raw_day").sum())
    legacy_cross = int(status.eq("cross_utc_day_skipped").sum())
    no_trade = int(status.eq("no_trades_in_window").sum())
    matched = int(status.eq("complete").sum())
    checked = matched + no_trade
    no_trade_rate = float(no_trade / checked) if checked else np.nan
    hourly = raw_hourly_coverage_report(raw_coverage)
    eligible_hours = hourly.loc[hourly["checked_windows_with_raw_day"] >= int(min_hour_windows)].copy()
    failed_hours = eligible_hours.loc[eligible_hours["no_trade_rate"] > float(max_hour_no_trade_rate)].copy()

    partition_mismatches = 0
    if {"start_time", "raw_archive_day"}.issubset(raw_coverage.columns):
        local_day = pd.to_datetime(raw_coverage["start_time"], errors="coerce").dt.strftime("%Y-%m-%d")
        archive_day = raw_coverage["raw_archive_day"].astype(str)
        partition_mismatches = int((local_day.notna() & archive_day.notna() & local_day.ne(archive_day)).sum())

    worst_hour = None
    if not eligible_hours.empty:
        worst = eligible_hours.sort_values(["no_trade_rate", "checked_windows_with_raw_day"], ascending=[False, False]).iloc[0]
        worst_hour = {
            "project_hour": int(worst["project_hour"]),
            "checked": int(worst["checked_windows_with_raw_day"]),
            "matched": int(worst["matched_raw_windows"]),
            "no_trade_rate": float(worst["no_trade_rate"]),
        }
    diagnostic = {
        "attempted_windows": attempted,
        "checked_windows_with_raw_day": checked,
        "matched_raw_windows": matched,
        "no_trade_windows": no_trade,
        "no_trade_rate": no_trade_rate,
        "missing_raw_windows": missing,
        "legacy_cross_utc_day_windows": legacy_cross,
        "archive_partition_mismatches": partition_mismatches,
        "hourly_failed_hours": failed_hours["project_hour"].astype(int).tolist(),
        "worst_hour": worst_hour,
        "micro_complete_windows": complete,
        "trigger_rows": trigger_rows,
    }

    alignment_failed = checked >= int(min_checked_windows) and no_trade_rate > float(max_no_trade_rate)
    hourly_failed = not failed_hours.empty
    if complete == 0 or alignment_failed or hourly_failed or partition_mismatches or legacy_cross:
        sample_cols = [
            name for name in (
                "window_id", "start_time", "end_time", "status", "raw_archive_day",
                "raw_archive_days", "requested_start_ms", "requested_end_ms",
                "raw_min_ts_ms", "raw_max_ts_ms", "raw_path",
            ) if name in raw_coverage.columns
        ]
        bad_mask = status.isin(["no_trades_in_window", "cross_utc_day_skipped"])
        sample = raw_coverage.loc[bad_mask, sample_cols].head(5)
        sample_text = sample.to_string(index=False) if not sample.empty else "<none>"
        hourly_text = failed_hours.to_string(index=False) if not failed_hours.empty else "<none>"
        raise RuntimeError(
            "R06 micro data gate failed: event windows did not produce reliable all-day 1s data.\n"
            f"diagnostic={diagnostic}\n"
            "Likely causes: timestamp unit mismatch, archive filename timezone mismatch, "
            "wrong raw symbol/path, or incomplete raw files.\n"
            f"failed_hourly_coverage:\n{hourly_text}\n"
            f"sample_bad_windows:\n{sample_text}"
        )
    return diagnostic


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _median(frame: pd.DataFrame, name: str) -> float:
    return float(_num(frame[name]).median()) if name in frame.columns and len(frame) else np.nan


def _mean_bool(frame: pd.DataFrame, name: str) -> float:
    if name not in frame.columns or not len(frame):
        return np.nan
    return float(frame[name].fillna(False).astype(bool).mean())


def data_quality_report(
    universe: pd.DataFrame,
    labels: pd.DataFrame,
    pair_audit: pd.DataFrame,
    micro_audit: pd.DataFrame,
    range_audit: pd.DataFrame,
    raw_coverage: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {"check": "attempt_feature_rows", "value": len(universe), "status": "INFO"},
        {"check": "attempt_label_rows", "value": len(labels), "status": "PASS" if len(universe) == len(labels) else "FAIL"},
        {"check": "oracle_prior_pairs", "value": len(pair_audit), "status": "PASS" if len(pair_audit) > 0 else "FAIL"},
        {"check": "micro_complete_windows", "value": int((micro_audit.get("status") == "complete").sum()) if not micro_audit.empty else 0, "status": "INFO"},
        {"check": "micro_insufficient_windows", "value": int((micro_audit.get("status") != "complete").sum()) if not micro_audit.empty else len(universe), "status": "INFO"},
        {"check": "raw_missing_windows", "value": int((raw_coverage.get("status") == "missing_raw_day").sum()) if not raw_coverage.empty else 0, "status": "INFO"},
        {"check": "raw_no_trade_windows", "value": int((raw_coverage.get("status") == "no_trades_in_window").sum()) if not raw_coverage.empty else 0, "status": "PASS" if raw_coverage.empty or int((raw_coverage.get("status") == "no_trades_in_window").sum()) == 0 else "WARN"},
        {"check": "cross_archive_day_windows_reconstructed", "value": int((pd.to_numeric(raw_coverage.get("archive_day_span"), errors="coerce") > 1).sum()) if not raw_coverage.empty and "archive_day_span" in raw_coverage else 0, "status": "INFO"},
        {"check": "legacy_cross_utc_day_windows_skipped", "value": int((raw_coverage.get("status") == "cross_utc_day_skipped").sum()) if not raw_coverage.empty else 0, "status": "PASS" if raw_coverage.empty or int((raw_coverage.get("status") == "cross_utc_day_skipped").sum()) == 0 else "FAIL"},
        {"check": "range_chunks_missing", "value": int((range_audit.get("status") == "missing_range_cache").sum()) if not range_audit.empty else 0, "status": "INFO"},
    ]
    forbidden = [name for name in universe.columns if name.startswith("future_")]
    rows.append({"check": "future_columns_in_feature_universe", "value": len(forbidden), "status": "PASS" if not forbidden else "FAIL"})
    return pd.DataFrame(rows)


def cohort_low_feature_summary(window_features: pd.DataFrame) -> pd.DataFrame:
    if window_features.empty:
        return pd.DataFrame()
    metrics = [
        "micro_low_offset_seconds", "low_delta_ratio_5s", "low_sell_share_5s",
        "low_large_delta_ratio_5s", "low_price_change_5s_bp",
        "low_downside_bp_per_sell_million_5s",
        "low_downside_bp_per_abs_negative_delta_million_5s",
        "low_impact_ratio_5s_vs_prior15s", "low_close_off_running_low_bp",
        "low_delta_improvement_5s_vs_prior15s",
    ]
    rows: list[dict[str, object]] = []
    for keys, group in window_features.groupby(["period", "cohort"], dropna=False, sort=False):
        row: dict[str, object] = {"period": keys[0], "cohort": keys[1], "events": len(group)}
        for metric in metrics:
            if metric in group.columns:
                row[f"median_{metric}"] = _median(group, metric)
                row[f"mean_{metric}"] = float(_num(group[metric]).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paired_micro_profile(window_features: pd.DataFrame) -> pd.DataFrame:
    if window_features.empty:
        return pd.DataFrame()
    oracle = window_features.loc[window_features["cohort"] == "ORACLE_TURN"].copy()
    prior = window_features.loc[window_features["cohort"] == "PRIOR_FAILED_ATTEMPT"].copy()
    if oracle.empty or prior.empty:
        return pd.DataFrame()
    metrics = [
        name for name in window_features.columns
        if name.startswith("low_") and pd.api.types.is_numeric_dtype(window_features[name])
    ]
    left = oracle[["pair_id", "period", *metrics]].rename(columns={name: f"oracle_{name}" for name in metrics})
    right = prior[["pair_id", *metrics]].rename(columns={name: f"prior_{name}" for name in metrics})
    paired = left.merge(right, on="pair_id", how="inner", validate="one_to_one")
    rows: list[dict[str, object]] = []
    for period, group in list(paired.groupby("period", sort=False)) + [("ALL", paired)]:
        for metric in metrics:
            a = _num(group[f"oracle_{metric}"])
            b = _num(group[f"prior_{metric}"])
            valid = a.notna() & b.notna()
            if valid.sum() == 0:
                continue
            diff = a[valid] - b[valid]
            rows.append(
                {
                    "period": period,
                    "feature": metric,
                    "pairs": int(valid.sum()),
                    "oracle_median": float(a[valid].median()),
                    "prior_median": float(b[valid].median()),
                    "paired_median_difference": float(diff.median()),
                    "paired_mean_difference": float(diff.mean()),
                    "oracle_greater_rate": float((diff > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def trigger_occurrence_summary(universe: pd.DataFrame, triggers: pd.DataFrame) -> pd.DataFrame:
    if universe.empty:
        return pd.DataFrame()
    denominators = universe.groupby(["period", "cohort"], dropna=False).size().rename("eligible_windows").reset_index()
    if triggers.empty:
        return denominators
    counts = (
        triggers.loc[triggers["trigger_name"].isin(CAUSAL_TRIGGERS)]
        .groupby(["period", "cohort", "trigger_name"], dropna=False)
        .size().rename("triggered_windows").reset_index()
    )
    out = counts.merge(denominators, on=["period", "cohort"], how="left")
    out["occurrence_rate"] = out["triggered_windows"] / out["eligible_windows"].replace(0, np.nan)
    return out.sort_values(["trigger_name", "period", "cohort"], kind="mergesort")


def trigger_path_summary(triggers: pd.DataFrame) -> pd.DataFrame:
    if triggers.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    groups = triggers.groupby(["period", "cohort", "trigger_name"], dropna=False, sort=False)
    for keys, group in groups:
        row: dict[str, object] = {
            "period": keys[0], "cohort": keys[1], "trigger_name": keys[2], "events": len(group),
            "median_signal_delay_seconds": _median(group, "signal_delay_from_minute_start_seconds"),
        }
        for h in (30, 60, 180, 300):
            row[f"complete_{h}s"] = int(group.get(f"path_complete_{h}s", False).fillna(False).astype(bool).sum()) if f"path_complete_{h}s" in group else 0
            row[f"median_mfe_{h}s"] = _median(group, f"mfe_{h}s")
            row[f"median_mae_{h}s"] = _median(group, f"mae_{h}s")
            row[f"median_net_close_return_{h}s"] = _median(group, f"net_close_return_{h}s")
            if f"net_close_return_{h}s" in group:
                row[f"net_positive_rate_{h}s"] = float((_num(group[f"net_close_return_{h}s"]) > 0).mean())
        for barrier in (10.0, 15.0, 25.0):
            tag = str(barrier).replace(".", "p")
            row[f"target_first_rate_{tag}bp"] = _mean_bool(group, f"target_first_{tag}bp")
            row[f"stop_first_rate_{tag}bp"] = _mean_bool(group, f"stop_first_{tag}bp")
        rows.append(row)
    return pd.DataFrame(rows)


def trigger_relative_to_baselines(triggers: pd.DataFrame) -> pd.DataFrame:
    if triggers.empty:
        return pd.DataFrame()
    key_cols = ["window_id", "checkpoint_id", "cohort", "period"]
    baseline_names = ("FIRST_NEW_LOW", "MINUTE_CLOSE")
    candidates = [
        "IMPACT_COLLAPSE_67", "IMPACT_COLLAPSE_50",
        "IMPACT_COLLAPSE_50_HIGH_BREAK", "MICRO_RECLAIM_5S",
    ]
    rows: list[pd.DataFrame] = []
    for baseline_name in baseline_names:
        baseline = triggers.loc[triggers["trigger_name"] == baseline_name].copy()
        keep = key_cols + [
            "signal_delay_from_minute_start_seconds", "mfe_60s", "mae_60s",
            "mfe_180s", "mae_180s", "mfe_300s", "mae_300s",
        ]
        baseline = baseline[[name for name in keep if name in baseline.columns]].rename(
            columns={name: f"baseline_{name}" for name in keep if name not in key_cols}
        )
        for candidate_name in candidates:
            candidate = triggers.loc[triggers["trigger_name"] == candidate_name].copy()
            merged = candidate.merge(baseline, on=key_cols, how="inner", validate="one_to_one")
            if merged.empty:
                continue
            merged["candidate_trigger"] = candidate_name
            merged["baseline_trigger"] = baseline_name
            for h in (60, 180, 300):
                merged[f"mae_improvement_{h}s"] = _num(merged[f"mae_{h}s"]) - _num(merged[f"baseline_mae_{h}s"])
                merged[f"mfe_retained_vs_baseline_{h}s"] = _num(merged[f"mfe_{h}s"]) - _num(merged[f"baseline_mfe_{h}s"])
            rows.append(merged)
    if not rows:
        return pd.DataFrame()
    full = pd.concat(rows, ignore_index=True)
    summary_rows: list[dict[str, object]] = []
    for keys, group in full.groupby(["period", "cohort", "candidate_trigger", "baseline_trigger"], sort=False):
        row: dict[str, object] = {
            "period": keys[0], "cohort": keys[1], "candidate_trigger": keys[2],
            "baseline_trigger": keys[3], "events": len(group),
            "median_extra_signal_delay_seconds": float(
                (_num(group["signal_delay_from_minute_start_seconds"]) - _num(group["baseline_signal_delay_from_minute_start_seconds"])).median()
            ),
        }
        for h in (60, 180, 300):
            row[f"median_mae_improvement_{h}s"] = _median(group, f"mae_improvement_{h}s")
            row[f"median_mfe_retained_difference_{h}s"] = _median(group, f"mfe_retained_vs_baseline_{h}s")
            row[f"mae_improved_rate_{h}s"] = float((_num(group[f"mae_improvement_{h}s"]) > 0).mean())
        summary_rows.append(row)
    return pd.DataFrame(summary_rows)


def range_pair_profile(range_features: pd.DataFrame) -> pd.DataFrame:
    if range_features.empty:
        return pd.DataFrame()
    metrics = [
        "range_bars_ending_in_attempt_minute", "down_range_bars_ending_in_attempt_minute",
        "up_range_bars_ending_in_attempt_minute", "first_up_delay_seconds",
        "down_bars_before_first_up", "last_down_duration_seconds",
        "last_down_delta_ratio", "last_down_sell_share",
        "last_down_downside_bp_per_sell_million",
        "last_down_impact_ratio_vs_previous_down",
        "first_up_duration_seconds", "first_up_delta_ratio", "first_up_sell_share",
    ]
    oracle = range_features.loc[range_features["cohort"] == "ORACLE_TURN"]
    prior = range_features.loc[range_features["cohort"] == "PRIOR_FAILED_ATTEMPT"]
    left = oracle[["pair_id", "period", "range_pct", *[m for m in metrics if m in oracle.columns]]]
    right = prior[["pair_id", "range_pct", *[m for m in metrics if m in prior.columns]]]
    paired = left.merge(right, on=["pair_id", "range_pct"], suffixes=("_oracle", "_prior"), how="inner", validate="one_to_one")
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, float | str, pd.DataFrame]] = [
        (str(k[0]), float(k[1]), g) for k, g in paired.groupby(["period", "range_pct"], sort=False)
    ]
    for range_pct, g in paired.groupby("range_pct", sort=False):
        groups.append(("ALL", float(range_pct), g))
    for period, range_pct, group in groups:
        for metric in metrics:
            a_name, b_name = f"{metric}_oracle", f"{metric}_prior"
            if a_name not in group or b_name not in group:
                continue
            a, b = _num(group[a_name]), _num(group[b_name])
            valid = a.notna() & b.notna()
            if not valid.any():
                continue
            diff = a[valid] - b[valid]
            rows.append(
                {
                    "period": period, "range_pct": range_pct, "feature": metric,
                    "pairs": int(valid.sum()), "oracle_median": float(a[valid].median()),
                    "prior_median": float(b[valid].median()),
                    "paired_median_difference": float(diff.median()),
                    "oracle_greater_rate": float((diff > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def range_pair_overlap_summary(range_features: pd.DataFrame) -> pd.DataFrame:
    """Report whether paired attempts reuse the same completed Range bars.

    Final and prior attempts are often only one or two minutes apart. A first-up
    delay improvement is mechanically optimistic when both attempts point to the
    same later Range bar, so overlap must be reported explicitly.
    """
    if range_features.empty:
        return pd.DataFrame()
    keep = [
        "pair_id", "period", "range_pct", "cohort",
        "last_down_bar_id", "first_up_bar_id",
    ]
    available = [name for name in keep if name in range_features.columns]
    frame = range_features.loc[:, available].copy()
    oracle = frame.loc[frame["cohort"] == "ORACLE_TURN"].drop(columns="cohort")
    prior = frame.loc[frame["cohort"] == "PRIOR_FAILED_ATTEMPT"].drop(columns="cohort")
    paired = oracle.merge(
        prior,
        on=["pair_id", "range_pct"],
        how="inner",
        suffixes=("_oracle", "_prior"),
        validate="one_to_one",
    )
    if paired.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, float, pd.DataFrame]] = [
        (str(keys[0]), float(keys[1]), group)
        for keys, group in paired.groupby(["period_oracle", "range_pct"], sort=False)
    ]
    groups.extend(("ALL", float(rp), group) for rp, group in paired.groupby("range_pct", sort=False))
    for period, range_pct, group in groups:
        row: dict[str, object] = {"period": period, "range_pct": range_pct, "pairs": len(group)}
        for prefix in ("last_down", "first_up"):
            left = pd.to_numeric(group.get(f"{prefix}_bar_id_oracle"), errors="coerce")
            right = pd.to_numeric(group.get(f"{prefix}_bar_id_prior"), errors="coerce")
            valid = left.notna() & right.notna()
            row[f"{prefix}_both_present_pairs"] = int(valid.sum())
            row[f"same_{prefix}_bar_rate"] = float((left[valid] == right[valid]).mean()) if valid.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def candidate_scorecard(universe: pd.DataFrame, triggers: pd.DataFrame) -> pd.DataFrame:
    if universe.empty or triggers.empty:
        return pd.DataFrame()
    denominators = universe.groupby("cohort").size().to_dict()
    rows: list[dict[str, object]] = []
    for trigger_name in (
        "IMPACT_COLLAPSE_67", "IMPACT_COLLAPSE_50",
        "IMPACT_COLLAPSE_50_HIGH_BREAK", "MICRO_RECLAIM_5S",
    ):
        selected = triggers.loc[triggers["trigger_name"] == trigger_name]
        if selected.empty:
            continue
        rates = {
            cohort: len(selected.loc[selected["cohort"] == cohort]) / max(1, int(denominators.get(cohort, 0)))
            for cohort in ("ORACLE_TURN", "PRIOR_FAILED_ATTEMPT", "CONTINUATION_CONTROL")
        }
        oracle_group = selected.loc[selected["cohort"] == "ORACLE_TURN"]
        period_lifts: list[float] = []
        for period in sorted(universe["period"].dropna().unique()):
            o_den = len(universe.loc[(universe["period"] == period) & (universe["cohort"] == "ORACLE_TURN")])
            p_den = len(universe.loc[(universe["period"] == period) & (universe["cohort"] == "PRIOR_FAILED_ATTEMPT")])
            o = len(selected.loc[(selected["period"] == period) & (selected["cohort"] == "ORACLE_TURN")]) / max(1, o_den)
            p = len(selected.loc[(selected["period"] == period) & (selected["cohort"] == "PRIOR_FAILED_ATTEMPT")]) / max(1, p_den)
            period_lifts.append(o - p)
        rows.append(
            {
                "trigger_name": trigger_name,
                "oracle_occurrence_rate": rates["ORACLE_TURN"],
                "prior_failed_occurrence_rate": rates["PRIOR_FAILED_ATTEMPT"],
                "continuation_control_occurrence_rate": rates["CONTINUATION_CONTROL"],
                "oracle_minus_prior_occurrence_lift": rates["ORACLE_TURN"] - rates["PRIOR_FAILED_ATTEMPT"],
                "minimum_period_oracle_minus_prior_lift": min(period_lifts) if period_lifts else np.nan,
                "oracle_events": len(oracle_group),
                "oracle_median_signal_delay_seconds": _median(oracle_group, "signal_delay_from_minute_start_seconds"),
                "oracle_median_mfe_180s": _median(oracle_group, "mfe_180s"),
                "oracle_median_mae_180s": _median(oracle_group, "mae_180s"),
                "oracle_target_first_rate_15bp": _mean_bool(oracle_group, "target_first_15p0bp"),
                "oracle_stop_first_rate_15bp": _mean_bool(oracle_group, "stop_first_15p0bp"),
                "causal_candidate": True,
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["minimum_period_oracle_minus_prior_lift", "oracle_minus_prior_occurrence_lift"],
        ascending=False,
        kind="mergesort",
    )


def causal_audit(universe: pd.DataFrame, triggers: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    future_cols = [name for name in universe.columns if name.startswith("future_")]
    rows.append({"check": "future_columns_in_features", "violations": len(future_cols), "status": "PASS" if not future_cols else "FAIL"})
    if triggers.empty:
        rows.append({"check": "trigger_rows_present", "violations": 0, "status": "WARN"})
        return pd.DataFrame(rows)
    causal = triggers.loc[~triggers["signal_uses_future"].fillna(False).astype(bool)]
    entry_bad = (~causal["entry_is_next_bar_open"].fillna(False).astype(bool)).sum()
    time_bad = (pd.to_datetime(causal["entry_time"]) < pd.to_datetime(causal["signal_time"])).sum()
    future_bad = causal["trigger_name"].eq("ORACLE_LOW_PLUS_1S").sum()
    rows.extend(
        [
            {"check": "entry_is_next_1s_open", "violations": int(entry_bad), "status": "PASS" if entry_bad == 0 else "FAIL"},
            {"check": "entry_time_not_before_signal", "violations": int(time_bad), "status": "PASS" if time_bad == 0 else "FAIL"},
            {"check": "oracle_trigger_excluded_from_causal_set", "violations": int(future_bad), "status": "PASS" if future_bad == 0 else "FAIL"},
        ]
    )
    return pd.DataFrame(rows)


def build_research_brief(
    universe: pd.DataFrame,
    micro_audit: pd.DataFrame,
    scorecard: pd.DataFrame,
) -> str:
    complete = int((micro_audit.get("status") == "complete").sum()) if not micro_audit.empty else 0
    best = scorecard.iloc[0].to_dict() if not scorecard.empty else {}
    best_text = (
        f"Top descriptive causal candidate: `{best.get('trigger_name')}`; "
        f"oracle-prior occurrence lift={best.get('oracle_minus_prior_occurrence_lift', np.nan):.2%}, "
        f"minimum period lift={best.get('minimum_period_oracle_minus_prior_lift', np.nan):.2%}."
        if best else "No causal micro candidate had enough completed windows to score."
    )
    return f"""# R06 Research Brief

## Scope
R06 is a future-labelled matched mechanism study, not a final strategy backtest.
It compares the final durable turning new-low attempt with the previous failed
new-low attempt in the same Swing Liquidity Zone, plus continuation controls.

## Data
- Attempt windows requested: {len(universe):,}
- Completed 1s windows: {complete:,}
- Candidate signals use closed 1s bars and enter at the next 1s bar open.
- `ORACLE_LOW_PLUS_1S` is an explicitly future-labelled upper bound and is not a
  deployable trigger.

## Current read
{best_text}

## Decision boundary
Do not promote a trigger from this report alone.  A candidate must show stable
oracle-vs-failed separation in every period, materially lower post-entry MAE,
retain enough MFE after the 0.11% round-trip cost, and then survive a separate
walk-forward backtest with natural parameter neighborhoods.
"""


__all__ = [
    "build_research_brief", "candidate_scorecard", "causal_audit",
    "cohort_low_feature_summary", "data_quality_report", "paired_micro_profile",
    "range_pair_profile", "trigger_occurrence_summary", "trigger_path_summary",
    "trigger_relative_to_baselines",
]
