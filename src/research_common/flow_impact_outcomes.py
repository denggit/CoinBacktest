#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reusable causal forward-path labels for Flow-Impact event studies."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.research_common.flow_impact import pressure_strength_labels, response_state_labels
from src.research_common.progress import ProgressReporter

def _prefix_sum(values: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum(np.nan_to_num(values, nan=0.0), dtype=np.float64)]


def _range_sum(prefix: np.ndarray, starts: np.ndarray, ends_exclusive: np.ndarray) -> np.ndarray:
    return prefix[ends_exclusive] - prefix[starts]


def future_path_outcomes(
    bars: pd.DataFrame,
    features: pd.DataFrame,
    events: pd.DataFrame,
    *,
    horizons_bars: tuple[int, ...],
    touch_levels_bps: tuple[float, ...],
    normal_cost: float,
    fee_only_cost: float,
    release_pressure_z: float,
    bar_delta: pd.Timedelta,
    progress_enabled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if events.empty:
        return events.copy(), pd.DataFrame()

    out = events.copy().reset_index(drop=True)
    signal_pos = out["signal_bar_pos"].to_numpy(dtype=np.int64)
    entry_pos = signal_pos + 1
    max_horizon = max(horizons_bars)
    n_bars = len(bars)

    observed = bars["source_bar_observed_flag"].astype(bool).to_numpy()
    observed_prefix = np.r_[0, np.cumsum(observed.astype(np.int64))]
    valid_range = (entry_pos >= 0) & (entry_pos + max_horizon <= n_bars)
    observed_count = np.zeros(len(out), dtype=np.int64)
    valid_positions = np.flatnonzero(valid_range)
    if len(valid_positions):
        starts = entry_pos[valid_positions]
        ends = starts + max_horizon
        observed_count[valid_positions] = observed_prefix[ends] - observed_prefix[starts]
    full_path_observed = valid_range & (observed_count == max_horizon)

    index = pd.DatetimeIndex(bars.index)
    opens = pd.to_numeric(bars["open"], errors="coerce").to_numpy(dtype=float)
    highs = pd.to_numeric(bars["high"], errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(bars["low"], errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(bars["close"], errors="coerce").to_numpy(dtype=float)
    delta = pd.to_numeric(bars["delta_notional"], errors="coerce").to_numpy(dtype=float)
    notional = pd.to_numeric(bars["notional"], errors="coerce").to_numpy(dtype=float)
    flow_sign = np.sign(delta)
    delta_prefix = _prefix_sum(delta)
    notional_prefix = _prefix_sum(notional)
    positive_sign_prefix = _prefix_sum((flow_sign > 0).astype(float))
    negative_sign_prefix = _prefix_sum((flow_sign < 0).astype(float))

    out["signal_bar_start"] = pd.to_datetime(out["signal_time"])
    out["signal_available_time"] = out["signal_bar_start"] + bar_delta
    out["entry_time"] = pd.NaT
    out["entry_price"] = np.nan
    safe_entry = (entry_pos >= 0) & (entry_pos < n_bars)
    out.loc[safe_entry, "entry_time"] = index[entry_pos[safe_entry]].to_numpy()
    out.loc[safe_entry, "entry_price"] = opens[entry_pos[safe_entry]]
    out["expected_entry_time"] = out["signal_available_time"]
    out["entry_not_next_open_flag"] = pd.to_datetime(out["entry_time"]) != pd.to_datetime(out["expected_entry_time"])
    out["entry_before_signal_available_flag"] = pd.to_datetime(out["entry_time"]) < pd.to_datetime(out["signal_available_time"])
    out["entry_source_bar_observed_flag"] = safe_entry & observed[np.clip(entry_pos, 0, n_bars - 1)]
    out["full_forward_observed_flag"] = full_path_observed
    out["synthetic_bar_dependency_flag"] = ~full_path_observed
    out["response_state"] = response_state_labels(out["price_response_norm"])
    out["pressure_strength"] = pressure_strength_labels(out["pressure_z"])
    out["year"] = out["signal_bar_start"].dt.year
    out["month"] = out["signal_bar_start"].dt.to_period("M").astype(str)
    out["date"] = out["signal_bar_start"].dt.date.astype(str)

    side = out["side"].to_numpy(dtype=float)
    entry_price = pd.to_numeric(out["entry_price"], errors="coerce").to_numpy(dtype=float)
    reporter = ProgressReporter(
        "[paths] horizons",
        len(horizons_bars),
        every=1,
        enabled=progress_enabled,
    )
    for done, horizon in enumerate(horizons_bars, start=1):
        exit_pos = signal_pos + int(horizon)
        valid = full_path_observed & (exit_pos >= 0) & (exit_pos < n_bars) & np.isfinite(entry_price) & (entry_price > 0.0)
        gross = np.full(len(out), np.nan, dtype=float)
        gross[valid] = side[valid] * (closes[exit_pos[valid]] / entry_price[valid] - 1.0)
        out[f"continuation_gross_h{horizon}"] = gross
        out[f"continuation_fee_net_h{horizon}"] = gross - float(fee_only_cost)
        out[f"continuation_net_h{horizon}"] = gross - float(normal_cost)
        out[f"reversal_gross_h{horizon}"] = -gross
        out[f"reversal_fee_net_h{horizon}"] = -gross - float(fee_only_cost)
        out[f"reversal_net_h{horizon}"] = -gross - float(normal_cost)

        post_delta = np.full(len(out), np.nan, dtype=float)
        post_notional = np.full(len(out), np.nan, dtype=float)
        same_sign_bars = np.full(len(out), np.nan, dtype=float)
        idx = np.flatnonzero(valid)
        if len(idx):
            starts = entry_pos[idx]
            ends = starts + int(horizon)
            post_delta[idx] = _range_sum(delta_prefix, starts, ends)
            post_notional[idx] = _range_sum(notional_prefix, starts, ends)
            pos_counts = _range_sum(positive_sign_prefix, starts, ends)
            neg_counts = _range_sum(negative_sign_prefix, starts, ends)
            same_counts = np.where(side[idx] > 0, pos_counts, neg_counts)
            same_sign_bars[idx] = same_counts / float(horizon)
        out[f"post_flow_ratio_h{horizon}"] = side * np.divide(
            post_delta,
            post_notional,
            out=np.full(len(out), np.nan, dtype=float),
            where=post_notional > 0.0,
        )
        out[f"post_same_flow_bar_ratio_h{horizon}"] = same_sign_bars
        reporter.update(done)
    reporter.close()

    running_fav = np.full(len(out), -np.inf, dtype=float)
    running_adv = np.full(len(out), -np.inf, dtype=float)
    first_fav = {level: np.full(len(out), -1, dtype=np.int32) for level in touch_levels_bps}
    first_adv = {level: np.full(len(out), -1, dtype=np.int32) for level in touch_levels_bps}
    mfe_by_horizon: dict[int, np.ndarray] = {}
    mae_by_horizon: dict[int, np.ndarray] = {}
    horizon_set = set(horizons_bars)
    reporter = ProgressReporter(
        "[paths] MFE/MAE + first touch",
        max_horizon,
        every=max(1, max_horizon // 10),
        enabled=progress_enabled,
    )
    for step in range(1, max_horizon + 1):
        pos = entry_pos + step - 1
        valid = full_path_observed & (pos >= 0) & (pos < n_bars) & np.isfinite(entry_price) & (entry_price > 0.0)
        favorable = np.full(len(out), np.nan, dtype=float)
        adverse = np.full(len(out), np.nan, dtype=float)
        long_mask = valid & (side > 0)
        short_mask = valid & (side < 0)
        favorable[long_mask] = highs[pos[long_mask]] / entry_price[long_mask] - 1.0
        adverse[long_mask] = 1.0 - lows[pos[long_mask]] / entry_price[long_mask]
        favorable[short_mask] = 1.0 - lows[pos[short_mask]] / entry_price[short_mask]
        adverse[short_mask] = highs[pos[short_mask]] / entry_price[short_mask] - 1.0
        running_fav = np.fmax(running_fav, favorable)
        running_adv = np.fmax(running_adv, adverse)
        for level in touch_levels_bps:
            threshold = float(level) / 10_000.0
            newly_fav = (first_fav[level] < 0) & valid & (favorable >= threshold)
            newly_adv = (first_adv[level] < 0) & valid & (adverse >= threshold)
            first_fav[level][newly_fav] = step
            first_adv[level][newly_adv] = step
        if step in horizon_set:
            mfe_by_horizon[step] = np.where(full_path_observed, running_fav, np.nan).copy()
            mae_by_horizon[step] = np.where(full_path_observed, -running_adv, np.nan).copy()
        reporter.update(step)
    reporter.close()

    for horizon in horizons_bars:
        out[f"continuation_mfe_h{horizon}"] = mfe_by_horizon[horizon]
        out[f"continuation_mae_h{horizon}"] = mae_by_horizon[horizon]
        out[f"reversal_mfe_h{horizon}"] = -mae_by_horizon[horizon]
        out[f"reversal_mae_h{horizon}"] = -mfe_by_horizon[horizon]

    first_touch_rows: list[dict[str, Any]] = []
    for level in touch_levels_bps:
        fav = first_fav[level]
        adv = first_adv[level]
        label = np.full(len(out), "none", dtype=object)
        label[(fav >= 0) & ((adv < 0) | (fav < adv))] = "favorable_first"
        label[(adv >= 0) & ((fav < 0) | (adv < fav))] = "adverse_first"
        label[(fav >= 0) & (adv >= 0) & (fav == adv)] = "both_same_bar"
        label[~full_path_observed] = "invalid_path"
        out[f"first_touch_{level:g}bps"] = label
        out[f"first_favorable_step_{level:g}bps"] = np.where(fav >= 0, fav, np.nan)
        out[f"first_adverse_step_{level:g}bps"] = np.where(adv >= 0, adv, np.nan)

        valid_labels = out.loc[out[f"first_touch_{level:g}bps"] != "invalid_path", f"first_touch_{level:g}bps"]
        counts = valid_labels.value_counts()
        total = int(counts.sum())
        first_touch_rows.append(
            {
                "touch_bps": float(level),
                "events": total,
                "favorable_first_rate": float(counts.get("favorable_first", 0) / total) if total else np.nan,
                "adverse_first_rate": float(counts.get("adverse_first", 0) / total) if total else np.nan,
                "both_same_bar_rate": float(counts.get("both_same_bar", 0) / total) if total else np.nan,
                "none_rate": float(counts.get("none", 0) / total) if total else np.nan,
                "directional_first_touch_gap": float((counts.get("favorable_first", 0) - counts.get("adverse_first", 0)) / total) if total else np.nan,
            }
        )

    duration = np.full(len(out), np.nan, dtype=float)
    for window in sorted(out["pressure_window_bars"].unique()):
        suffix = f"w{int(window)}"
        pressure_z = pd.to_numeric(features[f"pressure_z_{suffix}"], errors="coerce").to_numpy(dtype=float)
        pressure_direction = pd.to_numeric(features[f"pressure_direction_{suffix}"], errors="coerce").to_numpy(dtype=float)
        event_idx = np.flatnonzero(out["pressure_window_bars"].to_numpy(dtype=int) == int(window))
        unresolved = np.ones(len(event_idx), dtype=bool)
        for step in range(1, max_horizon + 1):
            positions = signal_pos[event_idx] + step
            valid = unresolved & (positions < n_bars)
            if not valid.any():
                break
            local = np.flatnonzero(valid)
            global_event_idx = event_idx[local]
            pos = positions[local]
            alive = (
                np.isfinite(pressure_z[pos])
                & (pressure_z[pos] >= float(release_pressure_z))
                & (pressure_direction[pos] == side[global_event_idx])
            )
            ended_local = local[~alive]
            if len(ended_local):
                duration[event_idx[ended_local]] = step - 1
                unresolved[ended_local] = False
        duration[event_idx[unresolved]] = max_horizon
    out["pressure_state_duration_bars"] = duration
    out["pressure_state_duration_minutes"] = duration * (bar_delta.total_seconds() / 60.0)

    audit_columns = [
        "event_id",
        "side_name",
        "pressure_window_bars",
        "signal_bar_start",
        "signal_available_time",
        "entry_time",
        "expected_entry_time",
        "entry_not_next_open_flag",
        "entry_before_signal_available_flag",
        "entry_source_bar_observed_flag",
        "full_forward_observed_flag",
        "synthetic_bar_dependency_flag",
    ]
    audit = out[audit_columns].copy()
    audit["causal_or_data_fail_flag"] = (
        audit["entry_not_next_open_flag"].astype(bool)
        | audit["entry_before_signal_available_flag"].astype(bool)
        | (~audit["entry_source_bar_observed_flag"].astype(bool))
        | audit["synthetic_bar_dependency_flag"].astype(bool)
    )
    return out, pd.DataFrame(first_touch_rows), audit

