#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Report tables for the broad swing-liquidity atlas."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import AtlasConfig


def fixed_period(timestamp: pd.Series) -> pd.Series:
    ts = pd.to_datetime(timestamp, errors="coerce")
    conditions = [
        ts < pd.Timestamp("2023-01-01"),
        (ts >= pd.Timestamp("2023-01-01")) & (ts < pd.Timestamp("2025-01-01")),
        (ts >= pd.Timestamp("2025-01-01")) & (ts < pd.Timestamp("2025-10-01")),
        ts >= pd.Timestamp("2025-10-01"),
    ]
    values = ["WARMUP_PRE_2023", "EARLY_2023_2024", "MID_2025Q1_Q3", "BOOKS_2025Q4_2026H1"]
    return pd.Series(np.select(conditions, values, default="UNKNOWN"), index=timestamp.index, dtype="object")


def level_summary(levels: pd.DataFrame) -> pd.DataFrame:
    if levels.empty:
        return pd.DataFrame()
    frame = levels.copy()
    frame["pivot_year"] = pd.to_datetime(frame["pivot_time"]).dt.year
    return (
        frame.groupby(["source_timeframe", "source_timeframe_min", "pivot_year", "future_max_eventual_order_label"], dropna=False)
        .agg(
            levels=("level_id", "size"),
            median_level=("level_price", "median"),
            median_range_bp=("pivot_range_bp", "median"),
            median_left_high_20_bp=("left_high_range_20_bp", "median"),
            median_confirmation_reaction_bp=("confirmation_reaction_close_bp", "median"),
        )
        .reset_index()
    )


def lifecycle_summary(lifecycle: pd.DataFrame) -> pd.DataFrame:
    if lifecycle.empty:
        return pd.DataFrame()
    frame = lifecycle.copy()
    frame["period"] = fixed_period(pd.to_datetime(frame["initial_available_time"]))
    return (
        frame.groupby(["source_timeframe", "period", "stop_liquidity_state", "support_resolution_180m"], dropna=False)
        .agg(
            levels=("level_id", "size"),
            median_age_to_sweep_min=("age_minutes_at_sweep", "median"),
            median_touch_episodes=("touch_episode_count_before_sweep", "median"),
            median_sweep_depth_bp=("sweep_depth_bp", "median"),
            clean_reclaim_rate_30m=("clean_reclaim_by_30m", "mean"),
            clean_reclaim_rate_180m=("clean_reclaim_by_180m", "mean"),
            close_revisit_rate_180m=("close_revisited_level_by_180m", "mean"),
        )
        .reset_index()
    )


def event_stage_summary(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    frame = events.copy()
    frame["period"] = fixed_period(pd.to_datetime(frame["event_available_time"]))
    return (
        frame.groupby(["event_stage", "source_timeframe", "period"], dropna=False)
        .agg(
            events=("event_id", "size"),
            unique_levels=("level_id", "nunique"),
            median_level_age_minutes=("level_age_minutes", "median"),
            median_confirmed_order=("confirmed_order_at_event", "median"),
            median_distance_close_bp=("distance_close_to_level_bp", "median"),
        )
        .reset_index()
    )


def forward_path_summary(events: pd.DataFrame, config: AtlasConfig) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    frame = events.copy()
    frame["period"] = fixed_period(pd.to_datetime(frame["event_available_time"]))
    rows: list[dict[str, object]] = []
    group_columns = ["event_stage", "source_timeframe", "period"]
    for keys, group in frame.groupby(group_columns, dropna=False):
        key_values = dict(zip(group_columns, keys if isinstance(keys, tuple) else (keys,)))
        for horizon in config.forward_horizons:
            h = int(horizon)
            ret = pd.to_numeric(group[f"close_return_{h}m"], errors="coerce").dropna()
            mfe = pd.to_numeric(group[f"mfe_close_{h}m"], errors="coerce").dropna()
            mae = pd.to_numeric(group[f"mae_close_{h}m"], errors="coerce").dropna()
            rows.append(
                {
                    **key_values,
                    "horizon_minutes": h,
                    "events": int(len(ret)),
                    "mean_close_return": float(ret.mean()) if len(ret) else np.nan,
                    "median_close_return": float(ret.median()) if len(ret) else np.nan,
                    "positive_close_rate": float((ret > 0).mean()) if len(ret) else np.nan,
                    "median_mfe_close": float(mfe.median()) if len(mfe) else np.nan,
                    "median_mae_close": float(mae.median()) if len(mae) else np.nan,
                    "hit_up_0p25_rate": float(pd.to_numeric(group[f"hit_up_0p25_{h}m"], errors="coerce").mean()),
                    "hit_up_0p50_rate": float(pd.to_numeric(group[f"hit_up_0p50_{h}m"], errors="coerce").mean()),
                    "hit_up_1p00_rate": float(pd.to_numeric(group[f"hit_up_1p00_{h}m"], errors="coerce").mean()),
                }
            )
    return pd.DataFrame(rows)


def _age_bucket(minutes: pd.Series) -> pd.Categorical:
    bins = [-np.inf, 360, 1440, 4320, 10080, 43200, np.inf]
    labels = ["<6h", "6-24h", "1-3d", "3-7d", "7-30d", ">=30d"]
    return pd.cut(pd.to_numeric(minutes, errors="coerce"), bins=bins, labels=labels, right=False)


def attribute_bin_summary(events: pd.DataFrame, config: AtlasConfig) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    sweep = events.loc[events["event_stage"].eq("sweep")].copy()
    if sweep.empty:
        return pd.DataFrame()
    sweep["age_bucket"] = _age_bucket(sweep["level_age_minutes"])
    sweep["touch_bucket"] = pd.cut(
        pd.to_numeric(sweep["touch_episode_count_before_sweep"], errors="coerce"),
        bins=[-np.inf, 0.5, 1.5, 2.5, 4.5, np.inf],
        labels=["0", "1", "2", "3-4", "5+"],
    )
    token = str(float(25.0)).replace(".", "p")
    confluence_column = f"active_timeframe_count_{token}bp"
    dimensions = ["age_bucket", "confirmed_order_at_event", "touch_bucket", confluence_column]
    rows: list[pd.DataFrame] = []
    for dimension in dimensions:
        if dimension not in sweep.columns:
            continue
        grouped = (
            sweep.groupby(["source_timeframe", dimension], observed=False, dropna=False)
            .agg(
                events=("event_id", "size"),
                median_sweep_depth_bp=("sweep_depth_bp", "median"),
                clean_reclaim_rate_30m=("clean_reclaim_by_30m", "mean"),
                clean_reclaim_rate_180m=("clean_reclaim_by_180m", "mean"),
                close_revisit_rate_180m=("close_revisited_level_by_180m", "mean"),
                median_return_30m=("close_return_30m", "median"),
                median_return_60m=("close_return_60m", "median"),
                median_mfe_60m=("mfe_close_60m", "median"),
                median_mae_60m=("mae_close_60m", "median"),
            )
            .reset_index()
            .rename(columns={dimension: "bin_value"})
        )
        grouped.insert(1, "dimension", dimension)
        rows.append(grouped)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def causal_audit(levels: pd.DataFrame, lifecycle: pd.DataFrame, events: pd.DataFrame, config: AtlasConfig) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rows.append(
        {
            "check": "initial_available_after_pivot_bar_close",
            "violations": int(
                (pd.to_datetime(levels["initial_available_time"]) <= pd.to_datetime(levels["pivot_bar_end_time"])).sum()
            ) if not levels.empty else 0,
        }
    )
    for order in config.confirmation_orders:
        column = f"order_{int(order)}_available_time"
        if levels.empty or column not in levels.columns:
            continue
        valid = levels[column].notna()
        expected_min = pd.to_datetime(levels.loc[valid, "pivot_bar_end_time"])
        rows.append(
            {
                "check": f"order_{int(order)}_not_backfilled",
                "violations": int((pd.to_datetime(levels.loc[valid, column]) < expected_min).sum()),
            }
        )
    if lifecycle.empty:
        lifecycle_violation = 0
    else:
        swept = lifecycle["sweep_pos"].ge(0)
        lifecycle_violation = int(
            (
                pd.to_datetime(lifecycle.loc[swept, "sweep_available_time"])
                < pd.to_datetime(lifecycle.loc[swept, "initial_available_time"])
            ).sum()
        )
    rows.append({"check": "sweep_after_level_available", "violations": lifecycle_violation})
    if events.empty:
        event_violation = 0
        entry_violation = 0
    else:
        event_violation = int(
            (pd.to_datetime(events["event_available_time"]) < pd.to_datetime(events["initial_available_time"])).sum()
        )
        entry_violation = int(
            (
                pd.to_datetime(events["entry_reference_time"], errors="coerce")
                < pd.to_datetime(events["event_available_time"], errors="coerce")
            ).fillna(False).sum()
        )
    rows.append({"check": "event_after_level_available", "violations": event_violation})
    rows.append({"check": "entry_reference_not_before_event_available", "violations": entry_violation})
    future_columns = [name for name in events.columns if str(name).startswith("future_")] if not events.empty else []
    rows.append({"check": "no_future_label_columns_in_event_table", "violations": len(future_columns)})
    return pd.DataFrame(rows)
