#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fixed-spec evaluation helpers for ICT MSS R01."""

from __future__ import annotations

from dataclasses import asdict
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from .models import DisplacementSpec, MSSResearchSpec

EPS = 1e-12


def profit_factor(values: pd.Series | np.ndarray) -> float:
    x = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(dtype=float)
    if not len(x):
        return np.nan
    gross_profit = float(x[x > 0].sum())
    gross_loss = float(-x[x < 0].sum())
    if gross_loss <= EPS:
        return np.inf if gross_profit > 0 else np.nan
    return gross_profit / gross_loss


def _metrics(values: pd.Series, *, top_remove_n: int = 10) -> dict[str, object]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return {
            "trades": 0,
            "mean_net_r": np.nan,
            "median_net_r": np.nan,
            "win_rate": np.nan,
            "profit_factor": np.nan,
            "p05_net_r": np.nan,
            "p95_net_r": np.nan,
            "top10_removed_profit_factor": np.nan,
            "top10_removed_mean_net_r": np.nan,
        }
    ordered = x.sort_values(ascending=False)
    trimmed = ordered.iloc[min(int(top_remove_n), len(ordered)) :]
    return {
        "trades": int(len(x)),
        "mean_net_r": float(x.mean()),
        "median_net_r": float(x.median()),
        "win_rate": float((x > 0).mean()),
        "profit_factor": profit_factor(x),
        "p05_net_r": float(x.quantile(0.05)),
        "p95_net_r": float(x.quantile(0.95)),
        "top10_removed_profit_factor": profit_factor(trimmed),
        "top10_removed_mean_net_r": float(trimmed.mean()) if len(trimmed) else np.nan,
    }


def filter_spec(
    setups: pd.DataFrame,
    spec: MSSResearchSpec,
    displacement: DisplacementSpec,
    *,
    research_start: pd.Timestamp,
    research_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply only predeclared, signal-time-observable filters for one spec."""

    s = spec.validate()
    frame = setups.copy()
    counts: dict[str, int] = {"raw_pairs": int(len(frame))}
    if frame.empty:
        return frame, counts
    mask = (
        (frame["side"].eq(int(s.side)) if int(s.side) else pd.Series(True, index=frame.index))
        & frame["micro_order"].eq(int(s.micro_order))
        & frame["structure_mode"].eq(str(s.structure_mode))
        & frame["max_confirmed_order"].ge(int(s.min_htf_confirmed_order))
        & frame["max_timeframe_min"].ge(int(s.min_max_timeframe_min))
        & frame["swept_timeframe_count"].ge(int(s.min_swept_timeframe_count))
        & frame["sweep_to_displacement_bars"].le(int(s.max_sweep_to_displacement_bars))
        & frame["displacement_body_vs_past_median"].ge(float(displacement.min_body_vs_past_median))
        & frame["displacement_range_vs_past_median"].ge(float(displacement.min_range_vs_past_median))
        & frame["displacement_body_fraction"].ge(float(displacement.min_body_fraction))
        & frame["displacement_close_from_extreme_fraction"].le(float(displacement.max_close_from_extreme_fraction))
        & frame["fvg_size_bp"].ge(float(displacement.min_fvg_size_bp))
        & frame["entry_structure_valid"].astype(bool)
    )
    frame = frame.loc[mask].copy()
    counts["after_mechanism_filters"] = int(len(frame))
    if frame.empty:
        return frame, counts

    # One MSS/FVG can be associated with more than one recently swept HTF level
    # episode.  Deduplicate using pre-outcome structural significance only.
    frame = frame.sort_values(
        [
            "fvg_completion_pos",
            "side",
            "max_timeframe_min",
            "max_confirmed_order",
            "swept_timeframe_count",
            "sweep_pos",
        ],
        ascending=[True, True, False, False, False, False],
        kind="mergesort",
    )
    frame = frame.drop_duplicates(["side", "fvg_completion_pos", "micro_order", "structure_mode"], keep="first")
    counts["after_signal_dedup"] = int(len(frame))
    signal_time = pd.to_datetime(frame["fvg_available_time"], errors="coerce")
    frame = frame.loc[(signal_time >= research_start) & (signal_time <= research_end)].copy()
    counts["inside_research_period"] = int(len(frame))
    if frame.empty:
        return frame, counts
    fill_ok = frame["first_fill_pos"].ge(0) & frame["fill_wait_bars"].le(int(s.max_fill_wait_bars))
    counts["filled_within_expiry"] = int(fill_ok.sum())
    return frame.loc[fill_ok].copy(), counts


def spec_definition_table(
    specs: Iterable[MSSResearchSpec],
    displacement_specs: Mapping[str, DisplacementSpec],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs:
        d = displacement_specs[spec.displacement_name]
        row = asdict(spec)
        row.update({f"disp_{k}": v for k, v in asdict(d).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _period_label(timestamp: pd.Series) -> pd.Series:
    ts = pd.to_datetime(timestamp, errors="coerce")
    out = pd.Series("OUTSIDE", index=timestamp.index, dtype="object")
    out.loc[(ts >= pd.Timestamp("2023-01-01")) & (ts < pd.Timestamp("2024-01-01"))] = "2023"
    out.loc[(ts >= pd.Timestamp("2024-01-01")) & (ts < pd.Timestamp("2025-01-01"))] = "2024"
    out.loc[(ts >= pd.Timestamp("2025-01-01")) & (ts < pd.Timestamp("2026-01-01"))] = "2025_VALIDATION"
    out.loc[(ts >= pd.Timestamp("2026-01-01")) & (ts < pd.Timestamp("2026-07-01"))] = "2026H1_SEALED"
    return out


def evaluate_specs(
    setups: pd.DataFrame,
    specs: Iterable[MSSResearchSpec],
    displacement_specs: Mapping[str, DisplacementSpec],
    *,
    target_rs: Iterable[float],
    round_trip_cost_pct: float,
    research_start: pd.Timestamp,
    research_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[tuple[str, float], pd.DataFrame]]:
    """Return overall metrics, period metrics, funnel, and frozen trade slices."""

    overall_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []
    funnel_rows: list[dict[str, object]] = []
    slices: dict[tuple[str, float], pd.DataFrame] = {}
    for spec in specs:
        disp = displacement_specs[spec.displacement_name]
        trades, counts = filter_spec(
            setups,
            spec,
            disp,
            research_start=research_start,
            research_end=research_end,
        )
        funnel_rows.append({"spec_id": spec.spec_id, **counts})
        if not trades.empty:
            trades["period"] = _period_label(trades["first_fill_time"])
            trades["year"] = pd.to_datetime(trades["first_fill_time"]).dt.year
            trades["month"] = pd.to_datetime(trades["first_fill_time"]).dt.to_period("M").astype(str)
        for target_r in target_rs:
            token = str(float(target_r)).replace(".", "p")
            gross_r_col = f"gross_r_r{token}"
            result_col = f"result_r{token}"
            if trades.empty or gross_r_col not in trades.columns:
                metrics = _metrics(pd.Series(dtype=float))
                overall_rows.append({"spec_id": spec.spec_id, "target_r": float(target_r), **metrics})
                continue
            x = trades.copy()
            # Recompute net R from gross R so cost stress can be derived without
            # rerunning path simulation.  Constant risk sizing makes R-space the
            # correct primary comparison across different structural stop widths.
            cost_r = float(round_trip_cost_pct) / pd.to_numeric(x["risk_pct"], errors="coerce")
            x["net_r_eval"] = pd.to_numeric(x[gross_r_col], errors="coerce") - cost_r
            x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=["net_r_eval", "risk_pct"])
            slices[(spec.spec_id, float(target_r))] = x
            metrics = _metrics(x["net_r_eval"])
            pre_holdout = x.loc[pd.to_datetime(x["first_fill_time"], errors="coerce") < pd.Timestamp("2026-01-01")]
            pre_holdout_metrics = _metrics(pre_holdout["net_r_eval"])
            target_rate = float(x[result_col].eq("TARGET").mean()) if len(x) else np.nan
            stop_rate = float(x[result_col].isin(["STOP", "GAP_STOP"]).mean()) if len(x) else np.nan
            timeout_rate = float(x[result_col].eq("TIMEOUT").mean()) if len(x) else np.nan
            monthly = x.groupby("month", observed=False)["net_r_eval"].sum() if len(x) else pd.Series(dtype=float)
            overall_rows.append(
                {
                    "spec_id": spec.spec_id,
                    "target_r": float(target_r),
                    **metrics,
                    "pre_holdout_trades": int(pre_holdout_metrics["trades"]),
                    "pre_holdout_mean_net_r": pre_holdout_metrics["mean_net_r"],
                    "pre_holdout_profit_factor": pre_holdout_metrics["profit_factor"],
                    "pre_holdout_top10_removed_profit_factor": pre_holdout_metrics["top10_removed_profit_factor"],
                    "pre_holdout_top10_removed_mean_net_r": pre_holdout_metrics["top10_removed_mean_net_r"],
                    "target_rate": target_rate,
                    "stop_rate": stop_rate,
                    "timeout_rate": timeout_rate,
                    "median_risk_pct": float(x["risk_pct"].median()) if len(x) else np.nan,
                    "median_fill_wait_bars": float(x["fill_wait_bars"].median()) if len(x) else np.nan,
                    "median_sweep_to_mss_bars": float(x["sweep_to_displacement_bars"].median()) if len(x) else np.nan,
                    "positive_month_share": float((monthly > 0).mean()) if len(monthly) else np.nan,
                    "months": int(len(monthly)),
                }
            )
            for period, part in x.groupby("period", sort=False, observed=False):
                pm = _metrics(part["net_r_eval"])
                period_rows.append(
                    {
                        "spec_id": spec.spec_id,
                        "target_r": float(target_r),
                        "period": period,
                        **pm,
                        "target_rate": float(part[result_col].eq("TARGET").mean()),
                        "median_risk_pct": float(part["risk_pct"].median()),
                    }
                )
    return (
        pd.DataFrame(overall_rows),
        pd.DataFrame(period_rows),
        pd.DataFrame(funnel_rows),
        slices,
    )


def cost_stress_table(
    slices: Mapping[tuple[str, float], pd.DataFrame],
    *,
    round_trip_cost_pct: float,
    multipliers: Iterable[float] = (1.0, 2.0, 3.0),
) -> pd.DataFrame:
    """Cost stress in both pre-holdout and full-sample scopes.

    The pre-holdout scope (2023-2025) is the only scope allowed to participate
    in model promotion.  Full-sample rows are descriptive after the sealed
    2026H1 confirmation has been opened.
    """

    rows: list[dict[str, object]] = []
    for (spec_id, target_r), frame in slices.items():
        gross_col = f"gross_r_r{str(float(target_r)).replace('.', 'p')}"
        scopes = {
            "PRE_HOLDOUT_2023_2025": frame.loc[
                pd.to_datetime(frame["first_fill_time"], errors="coerce") < pd.Timestamp("2026-01-01")
            ],
            "FULL_2023_2026H1": frame,
        }
        for scope, scoped in scopes.items():
            gross_r = pd.to_numeric(scoped[gross_col], errors="coerce")
            risk = pd.to_numeric(scoped["risk_pct"], errors="coerce")
            for multiplier in multipliers:
                net = gross_r - (float(round_trip_cost_pct) * float(multiplier) / risk)
                net = net.replace([np.inf, -np.inf], np.nan).dropna()
                rows.append(
                    {
                        "spec_id": spec_id,
                        "target_r": float(target_r),
                        "scope": scope,
                        "cost_multiplier": float(multiplier),
                        **_metrics(net),
                    }
                )
    return pd.DataFrame(rows)


def build_edge_gate(
    overall: pd.DataFrame,
    periods: pd.DataFrame,
    cost_stress: pd.DataFrame,
    specs: Iterable[MSSResearchSpec],
) -> pd.DataFrame:
    """Apply a predeclared promotion gate without selecting a best holdout spec.

    2023-2024 are development evidence, 2025 is validation, and 2026H1 is a
    sealed confirmation.  Every spec/target pair is judged independently; the
    holdout is never used to rank or tune parameters.
    """

    spec_map = {s.spec_id: s for s in specs}
    rows: list[dict[str, object]] = []
    for row in overall.itertuples(index=False):
        spec_id = str(row.spec_id)
        target_r = float(row.target_r)
        if periods.empty or not {"spec_id", "target_r"}.issubset(periods.columns):
            p = pd.DataFrame()
        else:
            p = periods.loc[(periods["spec_id"] == spec_id) & (periods["target_r"] == target_r)]
        pmap = {str(x.period): x for x in p.itertuples(index=False)}
        y23 = pmap.get("2023")
        y24 = pmap.get("2024")
        y25 = pmap.get("2025_VALIDATION")
        y26 = pmap.get("2026H1_SEALED")
        if cost_stress.empty or not {"spec_id", "target_r", "scope", "cost_multiplier"}.issubset(cost_stress.columns):
            c2 = pd.DataFrame()
        else:
            c2 = cost_stress.loc[
                (cost_stress["spec_id"] == spec_id)
                & (cost_stress["target_r"] == target_r)
                & cost_stress["scope"].eq("PRE_HOLDOUT_2023_2025")
                & np.isclose(cost_stress["cost_multiplier"], 2.0)
            ]
        c2row = c2.iloc[0] if not c2.empty else None

        dev_trades = int((getattr(y23, "trades", 0) if y23 else 0) + (getattr(y24, "trades", 0) if y24 else 0))
        dev_mean_num = 0.0
        if dev_trades:
            dev_mean_num += (float(getattr(y23, "mean_net_r", 0.0)) if y23 and np.isfinite(getattr(y23, "mean_net_r", np.nan)) else 0.0) * int(getattr(y23, "trades", 0) if y23 else 0)
            dev_mean_num += (float(getattr(y24, "mean_net_r", 0.0)) if y24 and np.isfinite(getattr(y24, "mean_net_r", np.nan)) else 0.0) * int(getattr(y24, "trades", 0) if y24 else 0)
        dev_mean = dev_mean_num / dev_trades if dev_trades else np.nan
        dev_years_nonnegative = bool(
            y23 is not None
            and y24 is not None
            and float(getattr(y23, "mean_net_r", np.nan)) > -0.05
            and float(getattr(y24, "mean_net_r", np.nan)) > -0.05
        )
        dev_pass = bool(dev_trades >= 50 and np.isfinite(dev_mean) and dev_mean > 0.05 and dev_years_nonnegative)
        validation_pass = bool(
            y25 is not None
            and int(getattr(y25, "trades", 0)) >= 20
            and float(getattr(y25, "mean_net_r", np.nan)) > 0.0
            and float(getattr(y25, "profit_factor", np.nan)) >= 1.10
        )
        cost2_pass = bool(
            c2row is not None
            and int(c2row["trades"]) >= 100
            and float(c2row["mean_net_r"]) > 0.0
            and float(c2row["profit_factor"]) >= 1.05
        )
        top10_pass = bool(
            int(getattr(row, "pre_holdout_trades", 0)) >= 100
            and np.isfinite(float(getattr(row, "pre_holdout_top10_removed_mean_net_r", np.nan)))
            and float(getattr(row, "pre_holdout_top10_removed_mean_net_r", np.nan)) > 0.0
            and float(getattr(row, "pre_holdout_top10_removed_profit_factor", np.nan)) >= 1.0
        )
        frozen_before_holdout = bool(dev_pass and validation_pass and cost2_pass and top10_pass)
        holdout_pass = bool(
            y26 is not None
            and int(getattr(y26, "trades", 0)) >= 10
            and float(getattr(y26, "mean_net_r", np.nan)) >= 0.0
            and float(getattr(y26, "profit_factor", np.nan)) >= 1.0
        )
        strong_full_sample = bool(
            int(getattr(row, "trades", 0)) >= 120
            and float(getattr(row, "mean_net_r", np.nan)) >= 0.08
            and float(getattr(row, "profit_factor", np.nan)) >= 1.30
            and float(getattr(row, "positive_month_share", np.nan)) >= 0.60
        )
        rows.append(
            {
                "spec_id": spec_id,
                "target_r": target_r,
                "neighborhood_group": spec_map[spec_id].neighborhood_group,
                "development_pass": dev_pass,
                "validation_2025_pass": validation_pass,
                "cost_2x_pass": cost2_pass,
                "top10_winner_removal_pass": top10_pass,
                "frozen_before_2026_holdout": frozen_before_holdout,
                "sealed_2026h1_pass": holdout_pass,
                "strong_full_sample_pass": strong_full_sample,
                "candidate_edge_pass_pre_neighborhood": bool(frozen_before_holdout and holdout_pass and strong_full_sample),
            }
        )
    gate = pd.DataFrame(rows)
    if gate.empty:
        return gate
    group_counts = (
        gate.loc[gate["candidate_edge_pass_pre_neighborhood"]]
        .groupby(["neighborhood_group", "target_r"], observed=False)["spec_id"]
        .nunique()
        .to_dict()
    )
    gate["neighbor_pass_count"] = [
        int(group_counts.get((row.neighborhood_group, row.target_r), 0))
        for row in gate.itertuples(index=False)
    ]
    # Standalone structural variants do not have a numeric neighbor axis; the
    # core displacement trio does.  A core candidate needs at least two nearby
    # displacement definitions to pass, otherwise it is treated as brittle.
    gate["parameter_neighborhood_pass"] = np.where(
        gate["neighborhood_group"].eq("core_displacement"),
        gate["neighbor_pass_count"].ge(2),
        gate["candidate_edge_pass_pre_neighborhood"],
    )
    gate["edge_found"] = gate["candidate_edge_pass_pre_neighborhood"] & gate["parameter_neighborhood_pass"]
    return gate
