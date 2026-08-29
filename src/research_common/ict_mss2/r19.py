#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R19 causal positioning-rebuild continuation-resumption atlas."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.research_common.ict_mss2.core import normalize_1m_bars
from src.research_common.ict_mss2.r18 import (
    EPS,
    R18Config,
    _datetime_ns,
    _num,
    _split_at,
    build_positioning_unwind_paths,
    prepare_positioning_alignment,
    summarize_r18_paths,
    summarize_r18_years,
)


@dataclass(frozen=True)
class R19Config(R18Config):
    rebuild_window_minutes: int = 60

    def validate(self) -> "R19Config":
        super().validate()
        if self.rebuild_window_minutes <= 0:
            raise ValueError("rebuild window must be positive")
        return self


def build_positioning_rebuild_events(
    bars_1m: pd.DataFrame,
    oi_features: pd.DataFrame,
    *,
    config: R19Config | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the frozen build -> release -> first rebuild continuation events."""

    cfg = (config or R19Config()).validate()
    bars = normalize_1m_bars(bars_1m)
    aligned = prepare_positioning_alignment(bars, oi_features, config=cfg)
    oi5 = _num(aligned, "oi_base_change_5m")
    prior_oi5 = _num(aligned, "build_oi_base_change_5m")
    build_oi1h = _num(aligned, "build_oi_base_change_1h")
    build_price1h = _num(aligned, "build_price_return_1h")
    release_common = (
        aligned["current_oi_valid"].astype(bool)
        & aligned["build_oi_valid"].astype(bool)
        & aligned["metric_gap_valid"].astype(bool)
        & aligned["price_step_valid"].astype(bool)
        & oi5.lt(0)
        & prior_oi5.ge(0)
        & build_oi1h.gt(0)
        & np.isfinite(build_price1h)
        & build_price1h.ne(0)
    )
    release_positions = np.flatnonzero(release_common.to_numpy(bool))
    successful: list[tuple[int, int, int]] = []
    expired_gap = 0
    expired_time = 0
    right_edge_censored = 0
    rebuild_no_break = 0

    available = pd.to_datetime(aligned["available_time"], errors="coerce")
    for release_pos in release_positions:
        direction = 1 if float(build_price1h.iloc[release_pos]) > 0 else -1
        release_time = pd.Timestamp(available.iloc[release_pos])
        termination: str | None = None
        for current in range(int(release_pos) + 1, len(aligned)):
            elapsed = pd.Timestamp(available.iloc[current]) - release_time
            if elapsed > pd.Timedelta(minutes=cfg.rebuild_window_minutes):
                expired_time += 1
                termination = "time"
                break
            if not (
                bool(aligned.iloc[current]["current_oi_valid"])
                and bool(aligned.iloc[current]["metric_gap_valid"])
                and bool(aligned.iloc[current]["price_step_valid"])
            ):
                expired_gap += 1
                termination = "gap_or_invalid"
                break
            current_oi5 = float(oi5.iloc[current])
            if current_oi5 < 0:
                continue
            termination = "rebuild"
            release_high = float(aligned.iloc[release_pos]["price_high"])
            release_low = float(aligned.iloc[release_pos]["price_low"])
            rebuild_close = float(aligned.iloc[current]["price_close"])
            price_break = rebuild_close > release_high + EPS if direction > 0 else rebuild_close < release_low - EPS
            if price_break:
                successful.append((int(release_pos), int(current), int(direction)))
            else:
                rebuild_no_break += 1
            break
        if termination is None:
            # The scan can exhaust only at the dataset boundary. Keep that
            # censoring separate from market-time expiry so one episode can
            # never alter a prior episode's engineering count.
            right_edge_censored += 1

    candidate_rows: list[dict[str, object]] = []
    for release_pos, rebuild_pos, direction in successful:
        release = aligned.iloc[release_pos]
        rebuild = aligned.iloc[rebuild_pos]
        signal = max(pd.Timestamp(rebuild["available_time"]), pd.Timestamp(rebuild["price_available_time"]))
        episode = aligned.iloc[release_pos : rebuild_pos + 1]
        candidate_rows.append(
            {
                "release_pos": release_pos,
                "rebuild_pos": rebuild_pos,
                "direction": "Long" if direction > 0 else "Short",
                "trade_direction": direction,
                "research_split": _split_at(signal, cfg),
                "build_oi_metric_time": pd.Timestamp(release["build_timestamp"]),
                "build_oi_available_time": pd.Timestamp(release["build_available_time"]),
                "release_oi_metric_time": pd.Timestamp(release["timestamp"]),
                "release_oi_available_time": pd.Timestamp(release["available_time"]),
                "rebuild_oi_metric_time": pd.Timestamp(rebuild["timestamp"]),
                "rebuild_oi_available_time": pd.Timestamp(rebuild["available_time"]),
                "release_price_bar_time": pd.Timestamp(release["price_bar_time"]),
                "release_price_available_time": pd.Timestamp(release["price_available_time"]),
                "rebuild_price_bar_time": pd.Timestamp(rebuild["price_bar_time"]),
                "rebuild_price_available_time": pd.Timestamp(rebuild["price_available_time"]),
                "signal_available_time": signal,
                "release_duration_minutes": float((pd.Timestamp(rebuild["available_time"]) - pd.Timestamp(release["available_time"])) / pd.Timedelta(minutes=1)),
                "episode_observations": int(rebuild_pos - release_pos + 1),
                "build_price_return_1h": float(release["build_price_return_1h"]),
                "build_oi_base_change_1h": float(release["build_oi_base_change_1h"]),
                "release_oi_base_change_5m": float(release["oi_base_change_5m"]),
                "rebuild_oi_base_change_5m": float(rebuild["oi_base_change_5m"]),
                "rebuild_oi_base": float(rebuild["sum_open_interest"]),
                "release_bar_high": float(release["price_high"]),
                "release_bar_low": float(release["price_low"]),
                "rebuild_close": float(rebuild["price_close"]),
                "rebuild_bar_high": float(rebuild["price_high"]),
                "rebuild_bar_low": float(rebuild["price_low"]),
                "episode_low": float(_num(episode, "price_low").min()),
                "episode_high": float(_num(episode, "price_high").max()),
                "atr_5m_1h_at_rebuild": float(rebuild["atr_5m_1h"]),
                "volatility_range_1h_at_rebuild": float(rebuild["build_range_high_1h"] - rebuild["build_range_low_1h"]),
            }
        )
    candidates = pd.DataFrame(candidate_rows)
    if candidates.empty:
        seal = pd.DataFrame(
            [
                {"check": "holdout_start", "value": str(cfg.holdout_start)},
                {"check": "sealed_holdout_candidate_count", "value": 0},
                {"check": "holdout_outcome_rows_computed", "value": 0},
                {"check": "holdout_unsealed", "value": 0},
            ]
        )
        engineering = pd.DataFrame(
            [
                {"check": "release_episodes", "value": int(len(release_positions))},
                {"check": "expired_on_gap_or_invalid", "value": int(expired_gap)},
                {"check": "expired_after_60m", "value": int(expired_time)},
                {"check": "right_edge_censored", "value": int(right_edge_censored)},
                {"check": "first_rebuild_without_break", "value": int(rebuild_no_break)},
                {"check": "successful_rebuild_breaks", "value": 0},
                {"check": "visible_successful_rebuild_breaks", "value": 0},
                {"check": "visible_executable_setups", "value": 0},
            ]
        )
        return pd.DataFrame(), seal, engineering

    holdout = candidates["research_split"].eq("holdout")
    seal = pd.DataFrame(
        [
            {"check": "holdout_start", "value": str(cfg.holdout_start)},
            {"check": "sealed_holdout_candidate_count", "value": int(holdout.sum())},
            {"check": "holdout_outcome_rows_computed", "value": 0},
            {"check": "holdout_unsealed", "value": 0},
        ]
    )
    visible = candidates.loc[candidates["research_split"].isin(["discovery", "validation"])].copy()
    index_ns = _datetime_ns(bars.index)
    rows: list[dict[str, object]] = []
    for ordinal, event in enumerate(visible.itertuples(index=False), start=1):
        signal = pd.Timestamp(event.signal_available_time)
        direction = int(event.trade_direction)
        entry_pos = int(np.searchsorted(index_ns, np.datetime64(signal, "ns").astype(np.int64), side="left"))
        base = event._asdict()
        base.update(
            {
                "setup_id": f"R19_{str(event.direction).upper()}_{signal.strftime('%Y%m%dT%H%M%S%f')}_{ordinal:06d}",
                "setup_status": "pending_geometry",
            }
        )
        if entry_pos >= len(bars):
            base["setup_status"] = "next_1m_entry_unavailable"
            rows.append(base)
            continue
        entry_time = pd.Timestamp(bars.index[entry_pos])
        entry = float(bars.iloc[entry_pos]["open"])
        atr = float(event.atr_5m_1h_at_rebuild)
        stop = (
            float(event.episode_low) - cfg.stop_buffer_atr * atr
            if direction > 0
            else float(event.episode_high) + cfg.stop_buffer_atr * atr
        )
        vol_range = float(event.volatility_range_1h_at_rebuild)
        target = entry + direction * vol_range
        risk = direction * (entry - stop) / entry if entry > EPS else np.nan
        runway = vol_range / entry if entry > EPS else np.nan
        base.update(
            {
                "entry_time": entry_time,
                "entry_price": entry,
                "stop_price": stop,
                "risk_distance_pct": risk,
                "structural_target_price": target,
                "structural_target_available_time": pd.Timestamp(event.rebuild_price_available_time),
                "structural_runway_pct": runway,
                "structural_reward_risk": runway / risk if np.isfinite(risk) and risk > EPS else np.nan,
            }
        )
        if not np.isfinite(atr) or atr <= EPS or not np.isfinite(vol_range) or vol_range <= EPS:
            base["setup_status"] = "volatility_unavailable"
        elif not np.isfinite(risk) or risk <= EPS:
            base["setup_status"] = "invalid_stop_geometry"
        elif risk > cfg.max_stop_distance_pct + EPS:
            base["setup_status"] = "stop_too_wide"
        else:
            base["setup_status"] = "executable"
        rows.append(base)

    events = pd.DataFrame(rows)
    if not events.empty:
        for name in [c for c in events.columns if c.endswith("_time")]:
            events[name] = pd.to_datetime(events[name], errors="coerce")
        events = events.sort_values(["signal_available_time", "direction", "setup_id"], kind="stable").reset_index(drop=True)
    engineering = pd.DataFrame(
        [
            {"check": "release_episodes", "value": int(len(release_positions))},
            {"check": "expired_on_gap_or_invalid", "value": int(expired_gap)},
            {"check": "expired_after_60m", "value": int(expired_time)},
            {"check": "right_edge_censored", "value": int(right_edge_censored)},
            {"check": "first_rebuild_without_break", "value": int(rebuild_no_break)},
            {"check": "successful_rebuild_breaks", "value": int(len(candidates))},
            {"check": "visible_successful_rebuild_breaks", "value": int(len(visible))},
            {"check": "visible_executable_setups", "value": int(events.get("setup_status", pd.Series(dtype=str)).eq("executable").sum())},
        ]
    )
    return events, seal, engineering


def build_positioning_rebuild_paths(
    bars_1m: pd.DataFrame,
    events: pd.DataFrame,
    *,
    config: R19Config | None = None,
) -> pd.DataFrame:
    cfg = (config or R19Config()).validate()
    paths = build_positioning_unwind_paths(bars_1m, events, config=cfg)
    if not paths.empty:
        paths["target_model"] = paths["target_model"].replace(
            {"H0_1H_BUILD_RANGE": "H0_1H_VOLATILITY_RANGE"}
        )
    return paths


def summarize_r19_funnel(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (split, direction), part in events.groupby(["research_split", "direction"], sort=True):
        executable = part.loc[part["setup_status"].eq("executable")]
        rows.append(
            {
                "research_split": split,
                "direction": direction,
                "rebuild_break_candidates": int(len(part)),
                "executable_rows": int(len(executable)),
                "stop_too_wide_rows": int(part["setup_status"].eq("stop_too_wide").sum()),
                "median_release_duration_minutes": _num(part, "release_duration_minutes").median(),
                "median_risk_distance_pct": _num(executable, "risk_distance_pct").median(),
                "median_volatility_reward_risk": _num(executable, "structural_reward_risk").median(),
            }
        )
    return pd.DataFrame(rows)


def summarize_r19_paths(paths: pd.DataFrame, *, config: R19Config | None = None) -> pd.DataFrame:
    return summarize_r18_paths(paths, config=(config or R19Config()).validate())


def summarize_r19_years(paths: pd.DataFrame) -> pd.DataFrame:
    return summarize_r18_years(paths)


def r19_causal_audit(
    events: pd.DataFrame,
    paths: pd.DataFrame,
    *,
    config: R19Config | None = None,
) -> pd.DataFrame:
    cfg = (config or R19Config()).validate()
    checks: list[dict[str, object]] = []

    def add(name: str, violations: int) -> None:
        checks.append({"check": name, "violations": int(violations), "status": "PASS" if int(violations) == 0 else "FAIL"})

    if events.empty:
        add("nonempty_visible_event_table", 1)
        return pd.DataFrame(checks)
    signal = pd.to_datetime(events["signal_available_time"], errors="coerce")
    entry = pd.to_datetime(events["entry_time"], errors="coerce")
    direction = _num(events, "trade_direction")
    executable = events.loc[events["setup_status"].eq("executable")]
    add("unique_setup_id", int(events["setup_id"].duplicated().sum()))
    add("feature_schema_excludes_future_or_oracle", len([c for c in events.columns if c.startswith("future_") or "oracle" in c.lower()]))
    add("release_after_build", int((pd.to_datetime(events["release_oi_available_time"]) <= pd.to_datetime(events["build_oi_available_time"])).sum()))
    add("rebuild_after_release", int((pd.to_datetime(events["rebuild_oi_available_time"]) <= pd.to_datetime(events["release_oi_available_time"])).sum()))
    add("rebuild_information_available_by_signal", int((pd.to_datetime(events["rebuild_oi_available_time"]) > signal).sum()))
    add("price_information_available_by_signal", int((pd.to_datetime(events["rebuild_price_available_time"]) > signal).sum()))
    add("release_window_respected", int((~_num(events, "release_duration_minutes").between(0.0, float(cfg.rebuild_window_minutes), inclusive="right")).sum()))
    add("rising_oi_build", int((_num(events, "build_oi_base_change_1h") <= 0).sum()))
    add("release_then_rebuild_signs", int(((_num(events, "release_oi_base_change_5m") >= 0) | (_num(events, "rebuild_oi_base_change_5m") < 0)).sum()))
    add("directional_build_sign", int((((direction > 0) & (_num(events, "build_price_return_1h") <= 0)) | ((direction < 0) & (_num(events, "build_price_return_1h") >= 0))).sum()))
    add("continuation_break_sign", int((((direction > 0) & (_num(events, "rebuild_close") <= _num(events, "release_bar_high"))) | ((direction < 0) & (_num(events, "rebuild_close") >= _num(events, "release_bar_low")))).sum()))
    add("next_eligible_1m_open", int((entry != signal.dt.ceil("min")).sum()))
    add("holdout_absent_from_visible_events", int((signal >= cfg.holdout_start).sum()))
    if executable.empty:
        add("nonempty_executable_events", 1)
    else:
        add("maximum_stop_distance_respected", int((_num(executable, "risk_distance_pct") > cfg.max_stop_distance_pct + EPS).sum()))
        add("volatility_target_available_by_signal", int((pd.to_datetime(executable["structural_target_available_time"]) > pd.to_datetime(executable["signal_available_time"])).sum()))
    if paths.empty:
        add("nonempty_first_passage_paths", 1)
    else:
        add("paths_reference_executable_events", int((~paths["setup_id"].isin(executable["setup_id"])).sum()))
        add("path_entry_not_before_signal", int((pd.to_datetime(paths["entry_time"]) < pd.to_datetime(paths["signal_available_time"])).sum()))
        add("holdout_or_embargo_absent_from_paths", int((pd.to_datetime(paths["entry_time"]) >= cfg.embargo_start).sum()))
        add("unique_setup_target_path", int(paths.duplicated(["setup_id", "target_model"]).sum()))
    return pd.DataFrame(checks)
