#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Independent alert episodes and remaining-opportunity audit for R03.3.1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .alert_audit_config import ProcessAlertAuditConfig


_EPS = 1e-12


@dataclass(frozen=True)
class AlertAuditResult:
    episode_metrics: dict[str, object]
    event_metrics: dict[str, object]
    episodes: pd.DataFrame
    event_coverage: pd.DataFrame


def build_alert_episodes(
    timestamps_ns: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    merge_gap_hours: float,
) -> pd.DataFrame:
    times = pd.DatetimeIndex(pd.to_datetime(np.asarray(timestamps_ns, dtype=np.int64)))
    values = np.asarray(scores, dtype=float)
    positions = np.flatnonzero(np.isfinite(values) & (values >= threshold))
    columns = [
        "episode_id",
        "first_pos",
        "last_pos",
        "first_alert_time",
        "last_alert_time",
        "signal_points",
        "duration_hours",
        "first_score",
        "peak_score",
    ]
    if len(positions) == 0:
        return pd.DataFrame(columns=columns)
    gap_ns = int(pd.Timedelta(hours=merge_gap_hours).value)
    position_times = times[positions].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    breaks = np.r_[True, np.diff(position_times) > gap_ns]
    group_ids = np.cumsum(breaks) - 1
    rows: list[dict[str, object]] = []
    for episode_id in np.unique(group_ids):
        member = positions[group_ids == episode_id]
        first = int(member[0])
        last = int(member[-1])
        rows.append(
            {
                "episode_id": int(episode_id),
                "first_pos": first,
                "last_pos": last,
                "first_alert_time": times[first],
                "last_alert_time": times[last],
                "signal_points": int(len(member)),
                "duration_hours": float((times[last] - times[first]) / pd.Timedelta(hours=1)),
                "first_score": float(values[first]),
                "peak_score": float(np.nanmax(values[member])),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _event_progress(process: str, event: pd.Series, alert_time: pd.Timestamp, price: float) -> float:
    if alert_time < pd.Timestamp(event["start_time"]):
        return 0.0
    if process == "volatile_range":
        duration = max((pd.Timestamp(event["end_time"]) - pd.Timestamp(event["start_time"])) / pd.Timedelta(hours=1), _EPS)
        return float(np.clip(((alert_time - pd.Timestamp(event["start_time"])) / pd.Timedelta(hours=1)) / duration, 0.0, 1.0))
    start_price = float(event["start_price"])
    target = max(float(event["target_move"]), _EPS)
    move = price / start_price - 1.0 if process == "up_expansion" else 1.0 - price / start_price
    return float(np.clip(move / target, 0.0, 1.0))


def _remaining_opportunity(process: str, event: pd.Series, price: float) -> tuple[float, float]:
    start_price = float(event["start_price"])
    if not np.isfinite(price) or price <= 0 or not np.isfinite(start_price) or start_price <= 0:
        return np.nan, np.nan
    if process == "up_expansion":
        target_price = start_price * (1.0 + float(event["target_move"]))
        extended = float(event.get("mfe_72h", np.nan))
        extension_price = start_price * (1.0 + extended) if np.isfinite(extended) else np.nan
        remaining = max(target_price / price - 1.0, 0.0)
        remaining_extension = max(extension_price / price - 1.0, 0.0) if np.isfinite(extension_price) else np.nan
        return float(remaining), float(remaining_extension)
    if process == "down_expansion":
        target_price = start_price * (1.0 - float(event["target_move"]))
        extended = float(event.get("mfe_72h", np.nan))
        extension_price = start_price * (1.0 - extended) if np.isfinite(extended) else np.nan
        remaining = max(1.0 - target_price / price, 0.0)
        remaining_extension = max(1.0 - extension_price / price, 0.0) if np.isfinite(extension_price) else np.nan
        return float(remaining), float(remaining_extension)
    up = float(event.get("up_excursion", np.nan))
    down = float(event.get("down_excursion", np.nan))
    high_target = start_price * (1.0 + up) if np.isfinite(up) else np.nan
    low_target = start_price * (1.0 - down) if np.isfinite(down) else np.nan
    remaining_up = max(high_target / price - 1.0, 0.0) if np.isfinite(high_target) else 0.0
    remaining_down = max(1.0 - low_target / price, 0.0) if np.isfinite(low_target) else 0.0
    total = remaining_up + remaining_down
    return float(total), float(max(remaining_up, remaining_down))


def _match_event(
    alert_time: pd.Timestamp,
    process_events: pd.DataFrame,
    *,
    horizon_hours: int,
) -> tuple[pd.Series | None, str, float]:
    if process_events.empty:
        return None, "false_alert", np.nan
    starts = pd.to_datetime(process_events["start_time"])
    ends = pd.to_datetime(process_events["end_time"])
    ongoing_positions = np.flatnonzero((starts <= alert_time).to_numpy() & (ends >= alert_time).to_numpy())
    if len(ongoing_positions):
        position = int(ongoing_positions[-1])
        event = process_events.iloc[position]
        signed_lead = float((pd.Timestamp(event["start_time"]) - alert_time) / pd.Timedelta(hours=1))
        return event, "ongoing", signed_lead
    future_positions = np.flatnonzero((starts > alert_time).to_numpy())
    if len(future_positions):
        position = int(future_positions[0])
        event = process_events.iloc[position]
        lead = float((pd.Timestamp(event["start_time"]) - alert_time) / pd.Timedelta(hours=1))
        if 0 < lead <= horizon_hours:
            return event, "pre_start", lead
    return None, "false_alert", np.nan


def audit_alert_episodes(
    episodes: pd.DataFrame,
    *,
    process: str,
    horizon_hours: int,
    process_events: pd.DataFrame,
    decision_prices: np.ndarray,
    fold_start: pd.Timestamp,
    fold_end: pd.Timestamp,
    config: ProcessAlertAuditConfig,
) -> AlertAuditResult:
    events = process_events.copy()
    if not events.empty:
        events["start_time"] = pd.to_datetime(events["start_time"])
        events["end_time"] = pd.to_datetime(events["end_time"])
        events = events.sort_values("start_time", kind="stable").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    minimum_remaining = (
        config.min_remaining_range_move if process == "volatile_range" else config.min_remaining_directional_move
    )
    for row in episodes.itertuples(index=False):
        alert_time = pd.Timestamp(row.first_alert_time)
        price = float(decision_prices[int(row.first_pos)])
        event, phase, signed_lead = _match_event(alert_time, events, horizon_hours=horizon_hours)
        progress = np.nan
        remaining = np.nan
        remaining_extension = np.nan
        event_uid: str | None = None
        hours_since_start = np.nan
        if event is not None:
            event_uid = str(event["event_uid"])
            start = pd.Timestamp(event["start_time"])
            hours_since_start = float((alert_time - start) / pd.Timedelta(hours=1))
            progress = _event_progress(process, event, alert_time, price)
            remaining, remaining_extension = _remaining_opportunity(process, event, price)
        early_ongoing = bool(
            phase == "ongoing"
            and np.isfinite(hours_since_start)
            and hours_since_start <= config.early_start_grace_hours
            and np.isfinite(progress)
            and progress <= config.max_actionable_progress
        )
        timing_actionable = phase == "pre_start" or early_ongoing
        remaining_actionable = bool(np.isfinite(remaining) and remaining >= minimum_remaining)
        actionable = bool(timing_actionable and remaining_actionable)
        if actionable and phase == "pre_start":
            classification = "actionable_pre_start"
        elif actionable:
            classification = "actionable_early_start"
        elif phase == "ongoing":
            classification = "late_or_spent_ongoing"
        elif phase == "pre_start":
            classification = "pre_start_low_remaining"
        else:
            classification = "false_alert"
        rows.append(
            {
                **row._asdict(),
                "process": process,
                "horizon_hours": horizon_hours,
                "alert_price": price,
                "matched_event_uid": event_uid,
                "phase": phase,
                "classification": classification,
                "signed_lead_hours": signed_lead,
                "hours_since_start": hours_since_start,
                "event_progress": progress,
                "remaining_opportunity": remaining,
                "remaining_extension_72h": remaining_extension,
                "timing_actionable": timing_actionable,
                "remaining_actionable": remaining_actionable,
                "actionable": actionable,
            }
        )
    audited = pd.DataFrame(rows)
    months = max((fold_end - fold_start) / pd.Timedelta(days=30.4375), 1.0)
    alert_count = int(len(audited))
    actionable_count = int(audited["actionable"].sum()) if alert_count else 0
    pre_start_count = int((audited["classification"] == "actionable_pre_start").sum()) if alert_count else 0
    early_count = int((audited["classification"] == "actionable_early_start").sum()) if alert_count else 0
    late_count = int((audited["classification"] == "late_or_spent_ongoing").sum()) if alert_count else 0
    false_count = int((audited["classification"] == "false_alert").sum()) if alert_count else 0
    actionable_rows = audited.loc[audited["actionable"]].copy() if alert_count else pd.DataFrame()
    episode_metrics: dict[str, object] = {
        "alert_episodes": alert_count,
        "alerts_per_month": float(alert_count / months),
        "actionable_alerts": actionable_count,
        "actionable_alert_precision": actionable_count / alert_count if alert_count else np.nan,
        "actionable_pre_start_rate": pre_start_count / alert_count if alert_count else np.nan,
        "actionable_early_start_rate": early_count / alert_count if alert_count else np.nan,
        "late_ongoing_rate": late_count / alert_count if alert_count else np.nan,
        "false_alert_rate": false_count / alert_count if alert_count else np.nan,
        "median_signed_lead_hours": float(actionable_rows["signed_lead_hours"].median()) if not actionable_rows.empty else np.nan,
        "median_event_progress": float(actionable_rows["event_progress"].median()) if not actionable_rows.empty else np.nan,
        "median_remaining_opportunity": float(actionable_rows["remaining_opportunity"].median()) if not actionable_rows.empty else np.nan,
        "p25_remaining_opportunity": float(actionable_rows["remaining_opportunity"].quantile(0.25)) if not actionable_rows.empty else np.nan,
        "median_remaining_extension_72h": float(actionable_rows["remaining_extension_72h"].median()) if not actionable_rows.empty else np.nan,
    }

    fold_events = events.loc[(events["start_time"] >= fold_start) & (events["start_time"] <= fold_end)].copy()
    coverage_rows: list[dict[str, object]] = []
    for event in fold_events.itertuples(index=False):
        uid = str(event.event_uid)
        matched = audited.loc[(audited["matched_event_uid"] == uid) & audited["actionable"]].sort_values(
            "first_alert_time", kind="stable"
        ) if alert_count else pd.DataFrame()
        first = matched.iloc[0] if not matched.empty else None
        coverage_rows.append(
            {
                "event_uid": uid,
                "process": process,
                "event_start": pd.Timestamp(event.start_time),
                "event_end": pd.Timestamp(event.end_time),
                "covered": bool(first is not None),
                "actionable_alert_episodes": int(len(matched)),
                "first_alert_time": first["first_alert_time"] if first is not None else pd.NaT,
                "first_alert_signed_lead_hours": float(first["signed_lead_hours"]) if first is not None else np.nan,
                "first_alert_progress": float(first["event_progress"]) if first is not None else np.nan,
                "first_alert_remaining_opportunity": float(first["remaining_opportunity"]) if first is not None else np.nan,
            }
        )
    coverage = pd.DataFrame(coverage_rows)
    covered = coverage.loc[coverage["covered"]].copy() if not coverage.empty else pd.DataFrame()
    event_count = int(len(coverage))
    covered_count = int(len(covered))
    event_metrics: dict[str, object] = {
        "events": event_count,
        "covered_events": covered_count,
        "event_coverage": covered_count / event_count if event_count else np.nan,
        "median_first_alert_signed_lead_hours": float(covered["first_alert_signed_lead_hours"].median()) if covered_count else np.nan,
        "median_first_alert_progress": float(covered["first_alert_progress"].median()) if covered_count else np.nan,
        "median_first_alert_remaining_opportunity": float(covered["first_alert_remaining_opportunity"].median()) if covered_count else np.nan,
        "mean_actionable_alerts_per_covered_event": float(covered["actionable_alert_episodes"].mean()) if covered_count else np.nan,
    }
    return AlertAuditResult(episode_metrics, event_metrics, audited, coverage)
