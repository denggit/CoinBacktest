#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Evaluation helpers for Market State Process & Probability Map V3.

The evaluator judges a process as a state-map component, not as a standalone
strategy.  It measures stage progression, role-consistent directional/path
information, holdout stability and causal probability calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.market_state.process_map import PROCESS_DIRECTIONS, ProcessMapConfig


@dataclass(frozen=True)
class ProcessEvaluationConfig:
    horizons_bars: tuple[int, ...] = (5, 15, 30, 60, 180)
    holdout_start: str | None = "2025-07-01"
    minimum_stage_samples: int = 300
    minimum_holdout_samples: int = 80
    minimum_years: int = 3
    minimum_positive_year_ratio: float = 0.60
    minimum_direction_uplift: float = 0.00005
    minimum_win_rate_uplift: float = 0.005
    minimum_profiles: int = 2

    def validate(self) -> None:
        horizons = tuple(sorted(set(int(v) for v in self.horizons_bars)))
        if not horizons or any(v < 1 for v in horizons):
            raise ValueError("horizons_bars must contain positive integers")
        if self.minimum_stage_samples < 1 or self.minimum_holdout_samples < 1:
            raise ValueError("sample minimums must be >= 1")
        if self.minimum_years < 1 or self.minimum_profiles < 1:
            raise ValueError("minimum_years/minimum_profiles must be >= 1")
        if not 0.0 <= self.minimum_positive_year_ratio <= 1.0:
            raise ValueError("minimum_positive_year_ratio must be in [0, 1]")


def _period_label(times: pd.Series, holdout_start: str | None) -> pd.Series:
    if not holdout_start:
        return pd.Series("all", index=times.index, dtype="object")
    split = pd.Timestamp(holdout_start)
    return pd.Series(np.where(pd.to_datetime(times) >= split, "holdout", "pre_holdout"), index=times.index)


def _baseline_table(path_frame: pd.DataFrame, target_col: str) -> pd.DataFrame:
    eligible = path_frame[target_col].notna()
    if "audit_eligible" in path_frame:
        eligible &= path_frame["audit_eligible"].fillna(False).astype(bool)
    base = path_frame.loc[eligible, ["signal_year", "volatility_state", target_col]].copy()
    if base.empty:
        return pd.DataFrame()
    return (
        base.groupby(["signal_year", "volatility_state"], dropna=False)[target_col]
        .agg(baseline_mean="mean", baseline_std="std", baseline_rows="size")
        .reset_index()
    )


def _baseline_win_table(path_frame: pd.DataFrame, target_col: str) -> pd.DataFrame:
    eligible = path_frame[target_col].notna()
    if "audit_eligible" in path_frame:
        eligible &= path_frame["audit_eligible"].fillna(False).astype(bool)
    base = path_frame.loc[eligible, ["signal_year", "volatility_state", target_col]].copy()
    if base.empty:
        return pd.DataFrame()
    base["success"] = pd.to_numeric(base[target_col], errors="coerce").gt(0.0)
    return (
        base.groupby(["signal_year", "volatility_state"], dropna=False)["success"]
        .agg(baseline_win_rate="mean")
        .reset_index()
    )


def build_stage_information(
    process_frame: pd.DataFrame,
    stage_events: pd.DataFrame,
    path_frame: pd.DataFrame,
    *,
    profile: str,
    config: ProcessEvaluationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return stage summary, yearly, period and raw event evidence."""

    config.validate()
    if stage_events.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    event_rows: list[dict[str, object]] = []
    for event in stage_events.itertuples(index=False):
        pos = int(event.position)
        if pos < 0 or pos >= len(path_frame):
            continue
        row = path_frame.iloc[pos]
        family = str(event.family)
        direction = int(event.direction)
        side = "long" if direction > 0 else "short"
        for horizon in sorted(set(int(v) for v in config.horizons_bars)):
            ret_col = f"{side}_return_h{horizon}"
            mfe_col = f"{side}_mfe_h{horizon}"
            mae_col = f"{side}_mae_h{horizon}"
            if ret_col not in path_frame or pd.isna(row[ret_col]):
                continue
            event_rows.append(
                {
                    "profile": profile,
                    "family": family,
                    "direction": direction,
                    "episode_id": int(event.episode_id),
                    "stage": int(event.stage),
                    "stage_label": str(event.stage_label),
                    "event_position": pos,
                    "event_available_time": pd.Timestamp(event.available_time),
                    "signal_year": int(row["signal_year"]),
                    "volatility_state": str(row["volatility_state"]),
                    "horizon_bars": horizon,
                    "directional_return": float(row[ret_col]),
                    "success": bool(float(row[ret_col]) > 0.0),
                    "mfe": float(row[mfe_col]),
                    "mae": float(row[mae_col]),
                    "stage_evidence": float(event.stage_evidence),
                    "cumulative_confidence": float(event.cumulative_confidence),
                }
            )
    events = pd.DataFrame(event_rows)
    if events.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    events["period"] = _period_label(events["event_available_time"], config.holdout_start)

    # Match every event to the all-market baseline from the same year and
    # volatility state.  This avoids rewarding a process merely for occurring
    # in a naturally directional/high-volatility regime.
    baseline_parts: list[pd.DataFrame] = []
    for (direction, horizon), subset in events.groupby(["direction", "horizon_bars"]):
        side = "long" if int(direction) > 0 else "short"
        target_col = f"{side}_return_h{int(horizon)}"
        mean_table = _baseline_table(path_frame, target_col)
        win_table = _baseline_win_table(path_frame, target_col)
        table = mean_table.merge(win_table, on=["signal_year", "volatility_state"], how="outer")
        table["direction"] = int(direction)
        table["horizon_bars"] = int(horizon)
        baseline_parts.append(table)
    baselines = pd.concat(baseline_parts, ignore_index=True) if baseline_parts else pd.DataFrame()
    events = events.merge(
        baselines,
        on=["direction", "horizon_bars", "signal_year", "volatility_state"],
        how="left",
    )
    events["return_uplift"] = events["directional_return"] - events["baseline_mean"]
    events["win_uplift"] = events["success"].astype(float) - events["baseline_win_rate"]

    group_cols = ["profile", "family", "direction", "stage", "stage_label", "horizon_bars"]
    summary = (
        events.groupby(group_cols, dropna=False)
        .agg(
            samples=("directional_return", "size"),
            mean_return=("directional_return", "mean"),
            win_rate=("success", "mean"),
            mean_mfe=("mfe", "mean"),
            mean_mae=("mae", "mean"),
            mean_return_uplift=("return_uplift", "mean"),
            mean_win_rate_uplift=("win_uplift", "mean"),
            mean_stage_evidence=("stage_evidence", "mean"),
            mean_confidence=("cumulative_confidence", "mean"),
        )
        .reset_index()
    )
    yearly = (
        events.groupby([*group_cols, "signal_year"], dropna=False)
        .agg(
            samples=("directional_return", "size"),
            mean_return=("directional_return", "mean"),
            win_rate=("success", "mean"),
            mean_return_uplift=("return_uplift", "mean"),
            mean_win_rate_uplift=("win_uplift", "mean"),
        )
        .reset_index()
    )
    periods = (
        events.groupby([*group_cols, "period"], dropna=False)
        .agg(
            samples=("directional_return", "size"),
            mean_return=("directional_return", "mean"),
            win_rate=("success", "mean"),
            mean_return_uplift=("return_uplift", "mean"),
            mean_win_rate_uplift=("win_uplift", "mean"),
        )
        .reset_index()
    )
    return summary, yearly, periods, events


def build_stage_progression(
    episodes: pd.DataFrame,
    *,
    profile: str,
    process_config: ProcessMapConfig,
) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for family, family_rows in episodes.groupby("family"):
        max_stage = process_config.max_stage(str(family))
        for stage in range(1, max_stage + 1):
            reached = pd.to_numeric(family_rows[f"stage_{stage}_pos"], errors="coerce").notna()
            reached_count = int(reached.sum())
            if stage < max_stage:
                next_reached = pd.to_numeric(family_rows[f"stage_{stage + 1}_pos"], errors="coerce").notna()
                next_count = int((reached & next_reached).sum())
                progression_rate = next_count / reached_count if reached_count else np.nan
                delay = pd.to_numeric(
                    family_rows.loc[reached & next_reached, f"stage_{stage + 1}_delay_bars"],
                    errors="coerce",
                )
            else:
                next_count = int(family_rows.loc[reached, "completed"].fillna(False).sum())
                progression_rate = next_count / reached_count if reached_count else np.nan
                delay = pd.Series(dtype=float)
            rows.append(
                {
                    "profile": profile,
                    "family": family,
                    "direction": PROCESS_DIRECTIONS[str(family)],
                    "stage": stage,
                    "stage_label": process_config.stage_label(str(family), stage),
                    "reached_episodes": reached_count,
                    "next_stage_or_completed": next_count,
                    "progression_rate": progression_rate,
                    "median_delay_bars": float(delay.median()) if len(delay) else np.nan,
                    "p90_delay_bars": float(delay.quantile(0.90)) if len(delay) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_episode_outcomes(
    episodes: pd.DataFrame,
    process_frame: pd.DataFrame,
    path_frame: pd.DataFrame,
    *,
    profile: str,
    config: ProcessEvaluationConfig,
    process_config: ProcessMapConfig,
) -> pd.DataFrame:
    if episodes.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for episode in episodes.itertuples(index=False):
        family = str(episode.family)
        max_stage = process_config.max_stage(family)
        completion_pos_value = getattr(episode, f"stage_{max_stage}_pos")
        if not bool(episode.completed) or pd.isna(completion_pos_value):
            continue
        pos = int(completion_pos_value)
        if pos >= len(path_frame):
            continue
        direction = PROCESS_DIRECTIONS[family]
        side = "long" if direction > 0 else "short"
        state_row = process_frame.iloc[pos]
        path_row = path_frame.iloc[pos]
        for horizon in sorted(set(int(v) for v in config.horizons_bars)):
            ret_col = f"{side}_return_h{horizon}"
            if ret_col not in path_frame or pd.isna(path_row[ret_col]):
                continue
            probability_col = f"{family}_direction_probability_h{horizon}"
            sample_col = f"{family}_direction_samples_h{horizon}"
            probability = state_row.get(probability_col, np.nan)
            rows.append(
                {
                    "profile": profile,
                    "family": family,
                    "direction": direction,
                    "episode_id": int(episode.episode_id),
                    "completion_position": pos,
                    "completion_available_time": pd.Timestamp(path_row["available_time"]),
                    "signal_year": int(path_row["signal_year"]),
                    "period": "holdout" if config.holdout_start and pd.Timestamp(path_row["available_time"]) >= pd.Timestamp(config.holdout_start) else "pre_holdout",
                    "horizon_bars": horizon,
                    "probability": float(probability) if pd.notna(probability) else np.nan,
                    "probability_samples": int(state_row.get(sample_col, 0)),
                    "actual_success": float(path_row[ret_col] > 0.0),
                    "directional_return": float(path_row[ret_col]),
                }
            )
    return pd.DataFrame(rows)


def build_probability_calibration(outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if outcomes.empty:
        return pd.DataFrame(), pd.DataFrame()
    eligible = outcomes.loc[outcomes["probability"].notna()].copy()
    if eligible.empty:
        return pd.DataFrame(), pd.DataFrame()
    eligible["brier"] = (eligible["probability"] - eligible["actual_success"]) ** 2
    summary = (
        eligible.groupby(["profile", "family", "horizon_bars", "period"], dropna=False)
        .agg(
            samples=("actual_success", "size"),
            mean_probability=("probability", "mean"),
            actual_success_rate=("actual_success", "mean"),
            calibration_bias=("probability", lambda s: float(s.mean())),
            brier_score=("brier", "mean"),
            mean_directional_return=("directional_return", "mean"),
            median_probability_support=("probability_samples", "median"),
        )
        .reset_index()
    )
    summary["calibration_bias"] = summary["mean_probability"] - summary["actual_success_rate"]
    bins = pd.cut(eligible["probability"], bins=[0.0, 0.45, 0.50, 0.55, 0.60, 1.0], include_lowest=True)
    eligible["probability_bin"] = bins.astype(str)
    calibration_bins = (
        eligible.groupby(["profile", "family", "horizon_bars", "period", "probability_bin"], dropna=False)
        .agg(
            samples=("actual_success", "size"),
            mean_probability=("probability", "mean"),
            actual_success_rate=("actual_success", "mean"),
            mean_directional_return=("directional_return", "mean"),
        )
        .reset_index()
    )
    return summary, calibration_bins


def build_process_registry(
    stage_summary: pd.DataFrame,
    yearly: pd.DataFrame,
    periods: pd.DataFrame,
    progression: pd.DataFrame,
    *,
    config: ProcessEvaluationConfig,
    process_config: ProcessMapConfig,
) -> pd.DataFrame:
    if stage_summary.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for family in sorted(stage_summary["family"].unique()):
        max_stage = process_config.max_stage(family)
        default_horizon = process_config.default_horizon(family)
        final_rows = stage_summary.loc[
            stage_summary["family"].eq(family)
            & stage_summary["stage"].eq(max_stage)
            & stage_summary["horizon_bars"].eq(default_horizon)
        ]
        period_rows = periods.loc[
            periods["family"].eq(family)
            & periods["stage"].eq(max_stage)
            & periods["horizon_bars"].eq(default_horizon)
        ]
        yearly_rows = yearly.loc[
            yearly["family"].eq(family)
            & yearly["stage"].eq(max_stage)
            & yearly["horizon_bars"].eq(default_horizon)
        ]
        supported_profiles = int(
            (
                final_rows["samples"].ge(config.minimum_stage_samples)
                & final_rows["mean_return_uplift"].ge(config.minimum_direction_uplift)
                & final_rows["mean_win_rate_uplift"].ge(config.minimum_win_rate_uplift)
            ).sum()
        )
        holdout = period_rows.loc[period_rows["period"].eq("holdout")]
        holdout_supported = int(
            (
                holdout["samples"].ge(config.minimum_holdout_samples)
                & holdout["mean_return_uplift"].gt(0.0)
                & holdout["mean_win_rate_uplift"].gt(0.0)
            ).sum()
        )
        year_signs = yearly_rows.groupby("signal_year")["mean_return_uplift"].mean()
        positive_year_ratio = float((year_signs > 0.0).mean()) if len(year_signs) else np.nan
        intermediate = stage_summary.loc[
            stage_summary["family"].eq(family)
            & stage_summary["stage"].lt(max_stage)
            & stage_summary["samples"].ge(config.minimum_stage_samples)
        ]
        intermediate_positive = bool(
            (
                intermediate["mean_return_uplift"].gt(0.0)
                & intermediate["mean_win_rate_uplift"].gt(0.0)
            ).any()
        )
        if (
            supported_profiles >= config.minimum_profiles
            and holdout_supported >= 1
            and positive_year_ratio >= config.minimum_positive_year_ratio
        ):
            status = "KEEP_PROCESS_CANDIDATE"
            reason = "completed process has positive cross-profile, yearly and holdout information"
        elif intermediate_positive:
            status = "KEEP_STAGE_ONLY"
            reason = "one or more stages add information, but completed process is not yet robust"
        elif len(final_rows) and float(final_rows["mean_return_uplift"].mean()) < 0.0:
            status = "REVISE_PROCESS"
            reason = "completed process has opposite or unstable directional semantics"
        else:
            status = "DROP_PROCESS"
            reason = "no stable role-consistent information at any supported stage"
        family_progress = progression.loc[progression["family"].eq(family)]
        rows.append(
            {
                "family": family,
                "direction": PROCESS_DIRECTIONS[family],
                "max_stage": max_stage,
                "default_horizon_bars": default_horizon,
                "status": status,
                "reason": reason,
                "supported_profiles": supported_profiles,
                "holdout_supported_profiles": holdout_supported,
                "positive_year_ratio": positive_year_ratio,
                "mean_final_return_uplift": float(final_rows["mean_return_uplift"].mean()) if len(final_rows) else np.nan,
                "mean_final_win_rate_uplift": float(final_rows["mean_win_rate_uplift"].mean()) if len(final_rows) else np.nan,
                "mean_final_samples": float(final_rows["samples"].mean()) if len(final_rows) else 0.0,
                "mean_stage1_to_stage2_rate": float(
                    family_progress.loc[family_progress["stage"].eq(1), "progression_rate"].mean()
                ) if len(family_progress) else np.nan,
                "mean_completion_rate_from_penultimate": float(
                    family_progress.loc[family_progress["stage"].eq(max_stage - 1), "progression_rate"].mean()
                ) if max_stage > 1 and len(family_progress) else np.nan,
            }
        )
    return pd.DataFrame(rows)
