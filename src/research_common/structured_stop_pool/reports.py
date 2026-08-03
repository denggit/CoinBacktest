#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report builders and hard gates for R09 structured stop-pool hypotheses."""
from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import StructuredStopPoolConfig, first_touch_specs
from .structure import FAMILY_COLUMNS, hypothesis_definitions

EPS = 1e-12
FAMILY_IDS = {column: f"H{i}" for i, column in enumerate(FAMILY_COLUMNS, start=1)}


def _mean_bool(series: pd.Series) -> float:
    value = series.dropna()
    return float(value.astype(bool).mean()) if len(value) else np.nan


def _median(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    value = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(value.median()) if len(value) else np.nan


def _mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    value = pd.to_numeric(frame[column], errors="coerce").dropna()
    return float(value.mean()) if len(value) else np.nan


def _rate(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        return np.nan
    return _mean_bool(frame[column])


def _cohorts(frame: pd.DataFrame, *, include_control: bool = False) -> Iterable[tuple[str, str, pd.DataFrame]]:
    swing = frame
    if "event_kind" in frame.columns:
        swing = frame.loc[frame["event_kind"].astype(str).eq("swing_zone_sweep")]
    yield "ALL", "ALL_SWING_ZONES", swing
    if "zone_has_any_structured_family" in swing.columns:
        yield "STRUCTURED", "ANY_STRUCTURED_FAMILY", swing.loc[swing["zone_has_any_structured_family"].fillna(False).astype(bool)]
    for column in FAMILY_COLUMNS:
        if column in swing.columns:
            yield FAMILY_IDS[column], column, swing.loc[swing[column].fillna(False).astype(bool)]
    if include_control and "event_kind" in frame.columns:
        yield "CONTROL", "MATCHED_NON_ZONE_DOWNSIDE", frame.loc[frame["event_kind"].astype(str).eq("non_zone_downside_control")]


def data_quality(
    bars: pd.DataFrame,
    levels: pd.DataFrame,
    lifecycle: pd.DataFrame,
    zones: pd.DataFrame,
    controls: pd.DataFrame,
    outcomes: pd.DataFrame,
    structure_thresholds: pd.DataFrame,
    release_calibration: pd.DataFrame,
) -> pd.DataFrame:
    gaps = bars.index.to_series().diff().dropna()
    metrics = [
        ("primary_rows", len(bars), "PASS" if len(bars) else "FAIL"),
        ("primary_start", bars.index.min() if len(bars) else pd.NaT, "PASS" if len(bars) else "FAIL"),
        ("primary_end", bars.index.max() if len(bars) else pd.NaT, "PASS" if len(bars) else "FAIL"),
        ("primary_duplicate_timestamps", int(bars.index.duplicated().sum()), "PASS" if not bars.index.duplicated().any() else "FAIL"),
        ("primary_max_gap_seconds", float(gaps.dt.total_seconds().max()) if len(gaps) else 0.0, "PASS"),
        ("swing_levels_loaded", len(levels), "PASS" if len(levels) else "FAIL"),
        ("first_swept_levels", int(pd.to_numeric(lifecycle.get("sweep_pos"), errors="coerce").ge(0).sum()), "PASS"),
        ("online_first_zone_events", len(zones), "PASS" if len(zones) else "FAIL"),
        ("matched_controls", len(controls), "PASS" if len(controls) >= max(1, int(len(zones) * 0.5)) else "WARN"),
        ("outcome_rows", len(outcomes), "PASS" if len(outcomes) >= len(zones) else "FAIL"),
        ("structure_threshold_timeframes", len(structure_thresholds), "PASS" if len(structure_thresholds) >= 4 else "WARN"),
        ("release_calibration_rows", int(pd.to_numeric(release_calibration.get("calibration_rows"), errors="coerce").max()) if len(release_calibration) else 0, "PASS" if len(release_calibration) else "FAIL"),
        ("release_baseline_coverage", float(outcomes.get("release_baseline_available", pd.Series(False, index=outcomes.index)).fillna(False).mean()) if len(outcomes) else 0.0, "PASS" if len(outcomes) and float(outcomes.get("release_baseline_available", pd.Series(False, index=outcomes.index)).fillna(False).mean()) >= 0.95 else "FAIL"),
    ]
    return pd.DataFrame(metrics, columns=["check", "value", "status"])


def hypothesis_universe_summary(level_features: pd.DataFrame, lifecycle: pd.DataFrame, zones: pd.DataFrame) -> pd.DataFrame:
    swept_ids = set(pd.to_numeric(lifecycle.loc[pd.to_numeric(lifecycle.get("sweep_pos"), errors="coerce").ge(0), "level_id"], errors="coerce").dropna().astype(int))
    rows: list[dict[str, Any]] = []
    definitions = hypothesis_definitions().set_index("feature_column")
    for column in FAMILY_COLUMNS:
        level_mask = level_features[column].fillna(False).astype(bool) if column in level_features else pd.Series(False, index=level_features.index)
        swept_mask = level_features["level_id"].astype(int).isin(swept_ids) & level_mask
        zone_mask = zones[column].fillna(False).astype(bool) if column in zones else pd.Series(False, index=zones.index)
        rows.append(
            {
                "hypothesis_id": FAMILY_IDS[column],
                "feature_column": column,
                "name": definitions.loc[column, "name"],
                "levels": int(level_mask.sum()),
                "level_share": float(level_mask.mean()) if len(level_mask) else np.nan,
                "swept_levels": int(swept_mask.sum()),
                "swept_level_rate": float(swept_mask.sum() / max(int(level_mask.sum()), 1)),
                "unique_online_zone_events": int(zone_mask.sum()),
                "zone_event_share": float(zone_mask.mean()) if len(zone_mask) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def family_release_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    periods = ["ALL", "EARLY_2023_2024", "MID_2025Q1_Q3", "BOOKS_2025Q4_2026H1"]
    for family_id, family, cohort in _cohorts(frame, include_control=True):
        for period in periods:
            part = cohort if period == "ALL" else cohort.loc[cohort["period"].astype(str).eq(period)]
            rows.append(
                {
                    "family_id": family_id,
                    "family": family,
                    "period": period,
                    "events": len(part),
                    "high_stop_release_rate": _rate(part, "high_stop_release_label"),
                    "stop_release_score_median": _median(part, "stop_release_score"),
                    "sell_notional_5m_vs_prior60_median": _median(part, "release_sell_notional_5m_vs_prior60"),
                    "trades_5m_vs_prior60_median": _median(part, "release_trades_count_5m_vs_prior60"),
                    "large_sell_5m_vs_prior60_median": _median(part, "release_large_sell_notional_5m_vs_prior60"),
                    "max_trade_1m_vs_prior60_median": _median(part, "release_max_trade_notional_1m_vs_prior60"),
                    "negative_delta_ratio_5m_median": _median(part, "release_negative_delta_ratio_5m"),
                    "price_downside_5m_bp_median": _median(part, "release_price_downside_5m_bp"),
                    "sell_impact_5m_bp_per_million_median": _median(part, "release_sell_impact_bp_per_million_5m"),
                }
            )
    return pd.DataFrame(rows)


def matched_release_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    if "matched_zone_event_id" not in frame.columns:
        return pd.DataFrame()
    zones = frame.loc[frame["event_kind"].astype(str).eq("swing_zone_sweep")].copy()
    controls = frame.loc[frame["event_kind"].astype(str).eq("non_zone_downside_control")].copy()
    if zones.empty or controls.empty:
        return pd.DataFrame()
    control_lookup = controls.set_index("matched_zone_event_id")
    rows: list[dict[str, Any]] = []
    for family_id, family, cohort in _cohorts(zones):
        pairs = cohort.loc[cohort["zone_event_id"].astype(str).isin(control_lookup.index.astype(str))].copy()
        if pairs.empty:
            continue
        ctrl = control_lookup.loc[pairs["zone_event_id"].astype(str)].reset_index(drop=True)
        score_zone = pd.to_numeric(pairs["stop_release_score"], errors="coerce").reset_index(drop=True)
        score_ctrl = pd.to_numeric(ctrl["stop_release_score"], errors="coerce").reset_index(drop=True)
        valid = score_zone.notna() & score_ctrl.notna()
        rows.append(
            {
                "family_id": family_id,
                "family": family,
                "matched_pairs": int(valid.sum()),
                "zone_high_release_rate": _rate(pairs, "high_stop_release_label"),
                "control_high_release_rate": _rate(ctrl, "high_stop_release_label"),
                "high_release_lift_pp": (_rate(pairs, "high_stop_release_label") - _rate(ctrl, "high_stop_release_label")) * 100.0,
                "zone_score_median": float(score_zone[valid].median()) if valid.any() else np.nan,
                "control_score_median": float(score_ctrl[valid].median()) if valid.any() else np.nan,
                "zone_score_pairwise_win_rate": float((score_zone[valid].to_numpy() > score_ctrl[valid].to_numpy()).mean()) if valid.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def family_path_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    periods = ["ALL", "EARLY_2023_2024", "MID_2025Q1_Q3", "BOOKS_2025Q4_2026H1"]
    for family_id, family, cohort in _cohorts(frame, include_control=True):
        for period in periods:
            part = cohort if period == "ALL" else cohort.loc[cohort["period"].astype(str).eq(period)]
            mae = pd.to_numeric(part.get("mae_low_180m"), errors="coerce")
            mfe = pd.to_numeric(part.get("mfe_high_180m"), errors="coerce")
            efficiency = mfe / mae.abs().replace(0.0, np.nan)
            rows.append(
                {
                    "family_id": family_id,
                    "family": family,
                    "period": period,
                    "events": len(part),
                    "structural_survival_60m": _rate(part, "structural_low_survival_60m"),
                    "mfe_60m_median_bp": _median(part, "mfe_high_60m") * 10_000.0,
                    "mae_60m_median_bp": _median(part, "mae_low_60m") * 10_000.0,
                    "mfe_180m_median_bp": _median(part, "mfe_high_180m") * 10_000.0,
                    "mae_180m_median_bp": _median(part, "mae_low_180m") * 10_000.0,
                    "mfe_to_abs_mae_median": float(efficiency.replace([np.inf, -np.inf], np.nan).median()) if len(efficiency.dropna()) else np.nan,
                    "tp15_before_lower_low_180m": _rate(part, "tp_0p15_before_lower_low_180m"),
                    "tp25_before_lower_low_180m": _rate(part, "tp_0p25_before_lower_low_180m"),
                    "tp50_before_lower_low_180m": _rate(part, "tp_0p5_before_lower_low_180m"),
                    "tp100_before_lower_low_180m": _rate(part, "tp_1_before_lower_low_180m"),
                }
            )
    return pd.DataFrame(rows)


def family_strategy_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    periods = ["ALL", "EARLY_2023_2024", "MID_2025Q1_Q3", "BOOKS_2025Q4_2026H1"]
    for family_id, family, cohort in _cohorts(frame, include_control=True):
        for spec in first_touch_specs():
            token = spec.name.lower()
            for period in periods:
                part = cohort if period == "ALL" else cohort.loc[cohort["period"].astype(str).eq(period)]
                net1 = pd.to_numeric(part.get(f"{token}_net_return_1x_cost"), errors="coerce").dropna()
                net2 = pd.to_numeric(part.get(f"{token}_net_return_2x_cost"), errors="coerce").dropna()
                outcome = part.get(f"{token}_outcome", pd.Series(dtype="object")).astype(str)
                wins = net1.loc[net1 > 0]
                losses = net1.loc[net1 < 0]
                pf = float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > EPS else np.nan
                rows.append(
                    {
                        "family_id": family_id,
                        "family": family,
                        "period": period,
                        "spec": spec.name,
                        "events": int(len(net1)),
                        "tp_rate": float(outcome.eq("TP").mean()) if len(outcome) else np.nan,
                        "sl_rate": float(outcome.str.startswith("SL").mean()) if len(outcome) else np.nan,
                        "time_rate": float(outcome.eq("TIME").mean()) if len(outcome) else np.nan,
                        "positive_net_rate_1x": float((net1 > 0).mean()) if len(net1) else np.nan,
                        "mean_net_1x": float(net1.mean()) if len(net1) else np.nan,
                        "median_net_1x": float(net1.median()) if len(net1) else np.nan,
                        "profit_factor_1x": pf,
                        "mean_net_2x": float(net2.mean()) if len(net2) else np.nan,
                        "same_bar_ambiguity_rate": _rate(part, f"{token}_same_bar_both_flag"),
                    }
                )
    return pd.DataFrame(rows)


def period_stability(release: pd.DataFrame, path: pd.DataFrame, strategy: pd.DataFrame) -> pd.DataFrame:
    release_v = release.loc[release["period"].ne("ALL") & release["family_id"].str.startswith("H")].copy()
    path_v = path.loc[path["period"].ne("ALL") & path["family_id"].str.startswith("H")].copy()
    strategy_v = strategy.loc[
        strategy["period"].ne("ALL")
        & strategy["family_id"].str.startswith("H")
        & strategy["spec"].eq("TP25_SL15")
    ].copy()
    merged = release_v.merge(
        path_v[["family_id", "period", "tp25_before_lower_low_180m", "mfe_to_abs_mae_median"]],
        on=["family_id", "period"],
        how="outer",
    ).merge(
        strategy_v[["family_id", "period", "mean_net_1x", "mean_net_2x", "positive_net_rate_1x"]],
        on=["family_id", "period"],
        how="outer",
    )
    return merged.sort_values(["family_id", "period"], kind="mergesort").reset_index(drop=True)


def family_overlap(frame: pd.DataFrame) -> pd.DataFrame:
    swing = frame.loc[frame["event_kind"].astype(str).eq("swing_zone_sweep")].copy() if "event_kind" in frame.columns else frame.copy()
    rows: list[dict[str, Any]] = []
    for a, b in combinations(FAMILY_COLUMNS, 2):
        ma = swing[a].fillna(False).astype(bool)
        mb = swing[b].fillna(False).astype(bool)
        inter = int((ma & mb).sum())
        union = int((ma | mb).sum())
        rows.append(
            {
                "family_a": FAMILY_IDS[a],
                "family_b": FAMILY_IDS[b],
                "events_a": int(ma.sum()),
                "events_b": int(mb.sum()),
                "intersection": inter,
                "jaccard": float(inter / union) if union else np.nan,
            }
        )
    return pd.DataFrame(rows)


def family_scorecard(
    frame: pd.DataFrame,
    release_summary: pd.DataFrame,
    strategy_summary: pd.DataFrame,
    config: StructuredStopPoolConfig,
) -> pd.DataFrame:
    cfg = config.validate()
    base_release = release_summary.loc[release_summary["family_id"].eq("ALL")].set_index("period")
    control_release = release_summary.loc[release_summary["family_id"].eq("CONTROL")].set_index("period")
    base_strategy = strategy_summary.loc[
        strategy_summary["family_id"].eq("ALL") & strategy_summary["spec"].eq("TP25_SL15")
    ].set_index("period")
    rows: list[dict[str, Any]] = []
    periods = ["EARLY_2023_2024", "MID_2025Q1_Q3", "BOOKS_2025Q4_2026H1"]
    for column in FAMILY_COLUMNS:
        family_id = FAMILY_IDS[column]
        family_release = release_summary.loc[release_summary["family_id"].eq(family_id)].set_index("period")
        family_strategy = strategy_summary.loc[
            strategy_summary["family_id"].eq(family_id) & strategy_summary["spec"].eq("TP25_SL15")
        ].set_index("period")
        event_count = int(frame.loc[frame["event_kind"].astype(str).eq("swing_zone_sweep") & frame[column].fillna(False).astype(bool)].shape[0])
        release_lifts: list[float] = []
        reversal_lifts: list[float] = []
        mean_nets: list[float] = []
        period_samples: list[int] = []
        for period in periods:
            if period not in family_release.index or period not in family_strategy.index:
                continue
            fam_rel = float(family_release.loc[period, "high_stop_release_rate"])
            comparator = float(control_release.loc[period, "high_stop_release_rate"]) if period in control_release.index else float(base_release.loc[period, "high_stop_release_rate"])
            release_lifts.append((fam_rel - comparator) * 100.0)
            fam_tp = float(family_strategy.loc[period, "tp_rate"])
            base_tp = float(base_strategy.loc[period, "tp_rate"]) if period in base_strategy.index else np.nan
            reversal_lifts.append((fam_tp - base_tp) * 100.0)
            mean_nets.append(float(family_strategy.loc[period, "mean_net_1x"]))
            period_samples.append(int(family_strategy.loc[period, "events"]))
        validation_release = release_lifts[1:] if len(release_lifts) >= 3 else release_lifts
        validation_reversal = reversal_lifts[1:] if len(reversal_lifts) >= 3 else reversal_lifts
        sample_pass = event_count >= int(cfg.minimum_family_events) and (min(period_samples) if period_samples else 0) >= int(cfg.minimum_period_events)
        release_pass = len(validation_release) >= 2 and all(np.isfinite(v) and v >= float(cfg.release_lift_gate_pp) for v in validation_release)
        reversal_pass = len(validation_reversal) >= 2 and all(np.isfinite(v) and v >= float(cfg.reversal_lift_gate_pp) for v in validation_reversal)
        cost_pass = len(mean_nets) >= 3 and all(np.isfinite(v) and v > 0 for v in mean_nets)
        if sample_pass and release_pass and reversal_pass and cost_pass:
            decision = "promote_to_backtest"
        elif sample_pass and (release_pass or reversal_pass or any(np.isfinite(v) and v > 0 for v in mean_nets)):
            decision = "research_continue"
        else:
            decision = "rejected"
        rows.append(
            {
                "family_id": family_id,
                "family": column,
                "events": event_count,
                "sample_gate_pass": sample_pass,
                "release_lift_pp_early_mid_books": "|".join(f"{v:.3f}" for v in release_lifts),
                "reversal_tp_lift_pp_early_mid_books": "|".join(f"{v:.3f}" for v in reversal_lifts),
                "mean_net_1x_early_mid_books": "|".join(f"{v:.6f}" for v in mean_nets),
                "release_gate_pass": release_pass,
                "reversal_gate_pass": reversal_pass,
                "cost_gate_pass": cost_pass,
                "decision": decision,
            }
        )
    return pd.DataFrame(rows)


def causal_audit(level_features: pd.DataFrame, zones: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    checks: list[tuple[str, int, str]] = []
    structure_late = int(
        (
            pd.to_datetime(level_features["structure_available_time"], errors="coerce")
            > pd.to_datetime(level_features["initial_available_time"], errors="coerce")
        ).sum()
    ) if len(level_features) else 0
    checks.append(("level_structure_available_after_level_available", structure_late, "PASS" if structure_late == 0 else "FAIL"))
    zone_late = int(
        (
            pd.to_datetime(zones.get("zone_member_structure_available_time_max"), errors="coerce")
            > pd.to_datetime(zones.get("event_available_time"), errors="coerce")
        ).sum()
    ) if len(zones) else 0
    checks.append(("zone_structure_available_after_sweep", zone_late, "PASS" if zone_late == 0 else "FAIL"))
    entry_not_next = 0
    if len(outcomes) and "r09_entry_time" in outcomes:
        expected = pd.to_datetime(outcomes["event_available_time"], errors="coerce")
        actual = pd.to_datetime(outcomes["r09_entry_time"], errors="coerce")
        entry_not_next = int((actual.notna() & expected.notna() & actual.ne(expected)).sum())
    checks.append(("entry_not_next_bar_open_time", entry_not_next, "PASS" if entry_not_next == 0 else "FAIL"))
    control_family = 0
    if len(outcomes) and "event_kind" in outcomes:
        controls = outcomes.loc[outcomes["event_kind"].astype(str).eq("non_zone_downside_control")]
        if len(controls):
            control_family = int(controls.loc[:, [c for c in FAMILY_COLUMNS if c in controls]].fillna(False).astype(bool).sum().sum())
    checks.append(("control_rows_with_structure_family", control_family, "PASS" if control_family == 0 else "FAIL"))
    label_in_family = int(any(name.startswith(("release_", "mfe_", "mae_", "tp_")) for name in FAMILY_COLUMNS))
    checks.append(("future_label_named_as_family_feature", label_in_family, "PASS" if label_in_family == 0 else "FAIL"))
    same_bar = 0
    for spec in first_touch_specs():
        column = f"{spec.name.lower()}_same_bar_both_flag"
        if column in outcomes:
            same_bar += int(outcomes[column].fillna(False).sum())
    checks.append(("same_bar_tp_sl_ambiguities_conservatively_sl", same_bar, "PASS"))
    return pd.DataFrame(checks, columns=["check", "violations", "status"])


def research_brief(scorecard: pd.DataFrame, universe: pd.DataFrame, matched: pd.DataFrame) -> str:
    promoted = scorecard.loc[scorecard["decision"].eq("promote_to_backtest"), "family_id"].astype(str).tolist()
    continued = scorecard.loc[scorecard["decision"].eq("research_continue"), "family_id"].astype(str).tolist()
    rejected = scorecard.loc[scorecard["decision"].eq("rejected"), "family_id"].astype(str).tolist()
    pair_text = "No matched-control table available."
    if not matched.empty:
        best = matched.sort_values("high_release_lift_pp", ascending=False).head(3)
        pair_text = "; ".join(f"{r.family_id}: {r.high_release_lift_pp:.2f}pp" for r in best.itertuples(index=False))
    return f"""# Structured Swing-Low Stop-Pool Hypothesis Study R09

## Research question

Which causally identifiable Swing Low structures actually release more stop-like selling when first swept, and which of those structures subsequently reverse with better TP-before-SL and realistic-cost expectancy?

## Hard boundary

R09 does not optimize entry timing and does not combine H1-H8 into a mined super-filter.  Structure membership is determined before the sweep.  Stop release and post-sweep reversal are separate labels.

## Decisions

- Promote to backtest: {promoted or 'none'}
- Research continue: {continued or 'none'}
- Rejected: {rejected or 'none'}

## Matched ordinary-downside comparison

Top high-stop-release lifts versus matched controls: {pair_text}

## Interpretation rule

A family is not a usable liquidity hypothesis merely because it later rebounds. It must first show abnormal sell/trade release versus matched ordinary downside, and then show stable reversal efficiency after next-open execution and costs.

## Event preservation

The report presents every family separately and reports overlaps. No event is removed just because it belongs to multiple hypotheses, and no combination threshold is selected from holdout results.
"""


def family_timeframe_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Show each hypothesis by the zone's highest source timeframe."""
    swing = frame.loc[frame["event_kind"].astype(str).eq("swing_zone_sweep")].copy() if "event_kind" in frame.columns else frame.copy()
    rows: list[dict[str, Any]] = []
    for column in FAMILY_COLUMNS:
        cohort = swing.loc[swing[column].fillna(False).astype(bool)]
        for (timeframe, period), part in cohort.groupby(["zone_primary_timeframe", "period"], dropna=False, sort=True):
            rows.append(
                {
                    "family_id": FAMILY_IDS[column],
                    "family": column,
                    "zone_primary_timeframe": timeframe,
                    "period": period,
                    "events": len(part),
                    "high_stop_release_rate": _rate(part, "high_stop_release_label"),
                    "stop_release_score_median": _median(part, "stop_release_score"),
                    "structural_survival_60m": _rate(part, "structural_low_survival_60m"),
                    "tp25_sl15_tp_rate": _rate(part, "tp25_sl15_tp_before_sl"),
                    "tp25_sl15_mean_net_1x": _mean(part, "tp25_sl15_net_return_1x_cost"),
                    "mfe_180m_median_bp": _median(part, "mfe_high_180m") * 10_000.0,
                    "mae_180m_median_bp": _median(part, "mae_low_180m") * 10_000.0,
                }
            )
    return pd.DataFrame(rows)
