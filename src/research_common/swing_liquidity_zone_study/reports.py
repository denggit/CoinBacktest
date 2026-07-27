#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Descriptive and validation reports for R03."""

from __future__ import annotations

import json
from collections.abc import Sequence

import numpy as np
import pandas as pd

from .config import ZoneStudyConfig


def _safe_median(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.median()) if len(values) else np.nan


def _safe_quantile(series: pd.Series, q: float) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(values.quantile(float(q))) if len(values) else np.nan


def fixed_period(timestamp: pd.Series) -> pd.Series:
    ts = pd.to_datetime(timestamp, errors="coerce")
    return pd.Series(
        np.select(
            [ts < pd.Timestamp("2023-01-01"), ts < pd.Timestamp("2025-01-01"), ts < pd.Timestamp("2025-10-01")],
            ["WARMUP_PRE_2023", "EARLY_2023_2024", "MID_2025Q1_Q3"],
            default="BOOKS_2025Q4_2026H1",
        ),
        index=timestamp.index,
        dtype="object",
    )


def zone_construction_summary(raw_sweeps: int, sensitivity: dict[float, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for tolerance, frame in sensitivity.items():
        first = frame.loc[frame["is_impulse_first_event"].astype(bool)] if not frame.empty else frame
        rows.append(
            {
                "zone_merge_tolerance_bp": float(tolerance),
                "raw_level_sweeps": int(raw_sweeps),
                "same_bar_zone_events": int(len(frame)),
                "online_impulse_first_events": int(len(first)),
                "raw_to_zone_reduction": 1.0 - len(frame) / max(raw_sweeps, 1),
                "raw_to_impulse_reduction": 1.0 - len(first) / max(raw_sweeps, 1),
                "median_members_per_zone": float(pd.to_numeric(frame.get("zone_member_count"), errors="coerce").median()) if len(frame) else np.nan,
                "p99_members_per_zone": float(pd.to_numeric(frame.get("zone_member_count"), errors="coerce").quantile(0.99)) if len(frame) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def path_horizon_summary(events: pd.DataFrame, config: ZoneStudyConfig) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    cfg = config.validate()
    frame = events.copy()
    frame["period"] = fixed_period(pd.to_datetime(frame["event_available_time"]))
    rows: list[dict[str, object]] = []
    groups = ["event_kind", "zone_primary_timeframe", "period"]
    for keys, group in frame.groupby(groups, dropna=False):
        key = dict(zip(groups, keys if isinstance(keys, tuple) else (keys,)))
        for horizon in cfg.path_horizons:
            h = int(horizon)
            ret = pd.to_numeric(group[f"close_return_{h}m"], errors="coerce")
            mfe = pd.to_numeric(group[f"mfe_high_{h}m"], errors="coerce")
            mae = pd.to_numeric(group[f"mae_low_{h}m"], errors="coerce")
            rows.append(
                {
                    **key,
                    "horizon_minutes": h,
                    "events": int(ret.notna().sum()),
                    "mean_close_return": float(ret.mean()),
                    "median_close_return": float(ret.median()),
                    "positive_close_rate": float((ret > 0).mean()),
                    "median_mfe": float(mfe.median()),
                    "p75_mfe": float(mfe.quantile(0.75)),
                    "median_mae": float(mae.median()),
                    "p25_mae": float(mae.quantile(0.25)),
                    "structural_survival_rate": float(pd.to_numeric(group[f"structural_low_survival_{h}m"], errors="coerce").mean()),
                    "zone_floor_reclaim_rate": float(pd.to_numeric(group[f"zone_floor_reclaim_by_{h}m"], errors="coerce").mean()),
                    "zone_ceiling_reclaim_rate": float(pd.to_numeric(group[f"zone_ceiling_reclaim_by_{h}m"], errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows)


def structural_exit_summary(events: pd.DataFrame, config: ZoneStudyConfig) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    cfg = config.validate()
    max_h = int(max(cfg.path_horizons))
    frame = events.copy()
    frame["period"] = fixed_period(pd.to_datetime(frame["event_available_time"]))
    rows = []
    for keys, group in frame.groupby(["event_kind", "zone_primary_timeframe", "period"], dropna=False):
        kind, timeframe, period = keys
        row: dict[str, object] = {
            "event_kind": kind,
            "zone_primary_timeframe": timeframe,
            "period": period,
            "events": int(len(group)),
            "lower_low_within_max_horizon_rate": float(pd.to_numeric(group["first_lower_low_pos"], errors="coerce").ge(0).mean()),
            "median_bars_to_lower_low": _safe_median(group["bars_to_lower_low"]),
            "median_mfe_before_lower_low": _safe_median(group[f"mfe_before_lower_low_{max_h}m"]),
            "p75_mfe_before_lower_low": _safe_quantile(group[f"mfe_before_lower_low_{max_h}m"], 0.75),
            "median_return_before_lower_low_or_end": _safe_median(group[f"close_return_before_lower_low_or_{max_h}m"]),
        }
        for target in cfg.tp_returns:
            token = f"{float(target) * 100:.2f}".rstrip("0").rstrip(".").replace(".", "p")
            row[f"tp_{token}_before_lower_low_rate"] = float(pd.to_numeric(group[f"tp_{token}_before_lower_low_{max_h}m"], errors="coerce").mean())
            row[f"median_mae_before_tp_{token}"] = _safe_median(group[f"mae_before_tp_{token}"])
            row[f"median_bars_to_tp_{token}"] = _safe_median(group[f"bars_to_tp_{token}"])
        rows.append(row)
    return pd.DataFrame(rows)


def control_comparison(events: pd.DataFrame, config: ZoneStudyConfig) -> pd.DataFrame:
    if events.empty or events["event_kind"].nunique() < 2:
        return pd.DataFrame()
    cfg = config.validate()
    frame = events.copy()
    frame["period"] = fixed_period(pd.to_datetime(frame["event_available_time"]))
    rows = []
    for period, group in frame.groupby("period", dropna=False):
        for horizon in (60, 180, 720, 1440, 4320):
            if horizon not in cfg.path_horizons:
                continue
            by_kind = {}
            for kind, part in group.groupby("event_kind"):
                by_kind[kind] = {
                    "events": int(len(part)),
                    "median_return": float(pd.to_numeric(part[f"close_return_{horizon}m"], errors="coerce").median()),
                    "positive_rate": float((pd.to_numeric(part[f"close_return_{horizon}m"], errors="coerce") > 0).mean()),
                    "median_mfe": float(pd.to_numeric(part[f"mfe_high_{horizon}m"], errors="coerce").median()),
                    "median_mae": float(pd.to_numeric(part[f"mae_low_{horizon}m"], errors="coerce").median()),
                    "survival": float(pd.to_numeric(part[f"structural_low_survival_{horizon}m"], errors="coerce").mean()),
                }
            zone = by_kind.get("swing_zone_sweep")
            control = by_kind.get("non_zone_downside_control")
            if not zone or not control:
                continue
            rows.append(
                {
                    "period": period,
                    "horizon_minutes": horizon,
                    "zone_events": zone["events"],
                    "control_events": control["events"],
                    "zone_median_return": zone["median_return"],
                    "control_median_return": control["median_return"],
                    "delta_median_return": zone["median_return"] - control["median_return"],
                    "zone_positive_rate": zone["positive_rate"],
                    "control_positive_rate": control["positive_rate"],
                    "delta_positive_rate": zone["positive_rate"] - control["positive_rate"],
                    "zone_median_mfe": zone["median_mfe"],
                    "control_median_mfe": control["median_mfe"],
                    "delta_median_mfe": zone["median_mfe"] - control["median_mfe"],
                    "zone_median_mae": zone["median_mae"],
                    "control_median_mae": control["median_mae"],
                    "zone_survival_rate": zone["survival"],
                    "control_survival_rate": control["survival"],
                    "delta_survival_rate": zone["survival"] - control["survival"],
                }
            )
    return pd.DataFrame(rows)


def control_match_balance(events: pd.DataFrame) -> pd.DataFrame:
    """Check whether matched controls are similar on pre-event causal variables."""

    if events.empty or "matched_zone_event_id" not in events.columns:
        return pd.DataFrame()
    zones = events.loc[events["event_kind"].eq("swing_zone_sweep")].copy()
    controls = events.loc[events["event_kind"].eq("non_zone_downside_control")].copy()
    if zones.empty or controls.empty:
        return pd.DataFrame()
    pair = controls.merge(
        zones, left_on="matched_zone_event_id", right_on="zone_event_id",
        suffixes=("_control", "_zone"), how="inner", validate="one_to_one",
    )
    variables = [
        "pre_atr_60m_vs_past7d", "pre_atr_60m_bp", "pre_return_60m",
        "pre_down_efficiency_60m", "bar_downside_to_pre_atr_60m",
        "current_bar_range_bp", "current_bar_close_location",
    ]
    rows: list[dict[str, object]] = []
    for variable in variables:
        c = pd.to_numeric(pair.get(f"{variable}_control"), errors="coerce")
        z = pd.to_numeric(pair.get(f"{variable}_zone"), errors="coerce")
        valid = c.notna() & z.notna()
        if not valid.any():
            continue
        diff = z[valid] - c[valid]
        pooled = np.sqrt((z[valid].var(ddof=1) + c[valid].var(ddof=1)) / 2.0)
        rows.append({
            "variable": variable,
            "matched_pairs": int(valid.sum()),
            "zone_mean": float(z[valid].mean()),
            "control_mean": float(c[valid].mean()),
            "mean_difference": float(diff.mean()),
            "median_abs_difference": float(diff.abs().median()),
            "standardized_mean_difference": float(diff.mean() / pooled) if np.isfinite(pooled) and pooled > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def _reference_edges(series: pd.Series, quantiles: Sequence[float] = (0.2, 0.4, 0.6, 0.8)) -> list[float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.nunique() < 5:
        return []
    edges = sorted(set(float(values.quantile(q)) for q in quantiles))
    return edges


def feature_bin_summary(events: pd.DataFrame, config: ZoneStudyConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return pd.DataFrame(), pd.DataFrame()
    cfg = config.validate()
    zone = events.loc[events["event_kind"].eq("swing_zone_sweep")].copy()
    zone["period"] = fixed_period(pd.to_datetime(zone["event_available_time"]))
    zone["age_bucket"] = pd.cut(
        pd.to_numeric(zone["zone_age_median_minutes"], errors="coerce"),
        [-np.inf, 360, 1440, 4320, 10080, 43200, np.inf],
        labels=["<6h", "6-24h", "1-3d", "3-7d", "7-30d", ">=30d"],
        right=False,
    )
    zone["member_bucket"] = pd.cut(pd.to_numeric(zone["zone_member_count"], errors="coerce"), [-np.inf, 1.5, 2.5, 4.5, 9.5, np.inf], labels=["1", "2", "3-4", "5-9", "10+"])
    zone["fresh_share_bucket"] = pd.cut(pd.to_numeric(zone["zone_fresh_member_share"], errors="coerce"), [-np.inf, 0.25, 0.5, 0.75, 0.999999, np.inf], labels=["0-25%", "25-50%", "50-75%", "75-<100%", "100%"])
    categorical = ["zone_primary_timeframe", "age_bucket", "member_bucket", "fresh_share_bucket", "zone_confirmed_order_max", "zone_has_4H", "zone_has_1D"]
    continuous = [
        "zone_width_bp", "zone_left_high_range_20_bp_max", "zone_confirmation_reaction_close_bp_max",
        "sweep_depth_below_floor_bp", "sweep_depth_to_pre_atr_60m", "sweep_depth_to_pre_atr_240m",
        "pre_atr_60m_vs_past7d", "pre_return_60m", "pre_return_240m", "pre_down_efficiency_60m",
        "sweep_bar_close_location", "sweep_bar_lower_wick_fraction",
    ]
    reference = zone.loc[zone["period"].eq("EARLY_2023_2024")]
    if reference.empty:
        reference = zone
    rows: list[pd.DataFrame] = []
    edge_rows: list[dict[str, object]] = []
    max_h = int(max(cfg.path_horizons))

    def summarize(dimension: str, values: pd.Series) -> None:
        temp = zone.copy()
        temp["_bin"] = values
        grouped = (
            temp.groupby(["period", "_bin"], observed=False, dropna=False)
            .agg(
                events=("zone_event_id", "size"),
                median_return_60m=("close_return_60m", "median"),
                median_return_180m=("close_return_180m", "median"),
                median_return_720m=("close_return_720m", "median"),
                median_return_1440m=("close_return_1440m", "median"),
                positive_rate_60m=("close_return_60m", lambda x: float((pd.to_numeric(x, errors="coerce") > 0).mean())),
                structural_survival_1440m=("structural_low_survival_1440m", "mean"),
                median_mfe_180m=("mfe_high_180m", "median"),
                median_mae_180m=("mae_low_180m", "median"),
                median_mfe_before_lower_low=(f"mfe_before_lower_low_{max_h}m", "median"),
            )
            .reset_index()
            .rename(columns={"_bin": "bin_value"})
        )
        grouped.insert(0, "dimension", dimension)
        rows.append(grouped)

    for dimension in categorical:
        if dimension in zone.columns:
            summarize(dimension, zone[dimension])
    for dimension in continuous:
        if dimension not in zone.columns:
            continue
        edges = _reference_edges(reference[dimension])
        edge_rows.append({"dimension": dimension, "reference_period": "EARLY_2023_2024", "edges_json": json.dumps(edges)})
        if not edges:
            continue
        bins = [-np.inf, *edges, np.inf]
        labels = [f"Q{i+1}" for i in range(len(bins) - 1)]
        summarize(dimension, pd.cut(pd.to_numeric(zone[dimension], errors="coerce"), bins=bins, labels=labels, include_lowest=True))
    return (pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame(), pd.DataFrame(edge_rows))


def causal_audit(features: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if features.empty:
        return pd.DataFrame([{"check": "feature_rows_nonempty", "violations": 1}])
    rows.append({"check": "zone_levels_available_by_event", "violations": int((pd.to_datetime(features["zone_latest_level_available_time"], errors="coerce") > pd.to_datetime(features["event_available_time"], errors="coerce")).fillna(False).sum())})
    rows.append({"check": "entry_after_closed_event", "violations": int((pd.to_datetime(labels["entry_reference_time"], errors="coerce") < pd.to_datetime(features["event_available_time"], errors="coerce")).fillna(False).sum())})
    forbidden = [name for name in features.columns if str(name).startswith(("future_", "label_")) or "mfe" in str(name).lower() or "mae" in str(name).lower() or "close_return_" in str(name)]
    rows.append({"check": "no_outcome_columns_in_feature_table", "violations": int(len(forbidden)), "detail": "|".join(forbidden)})
    missing_ids = set(features["zone_event_id"].astype(str)) ^ set(labels["zone_event_id"].astype(str))
    rows.append({"check": "feature_label_id_tieout", "violations": int(len(missing_ids))})
    rows.append({"check": "feature_ids_unique", "violations": int(features["zone_event_id"].duplicated().sum())})
    rows.append({"check": "label_ids_unique", "violations": int(labels["zone_event_id"].duplicated().sum())})
    return pd.DataFrame(rows)
