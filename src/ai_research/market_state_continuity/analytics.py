#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""State atlases and trading-value linkage for R03.3.3."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.future_process_forecast.intensity_config import DEFAULT_FUTURE_INTENSITY_CONFIG
from src.ai_research.future_process_forecast.intensity_targets import load_intensity_year_shard

from .config import MarketStateContinuityConfig
from .state_cache import StateYearShard, load_state_year_shard, ns_to_datetime


STATE_NAMES = {-1: "down_or_low", 0: "neutral", 1: "up_or_high"}


def _year(path: Path) -> int:
    shard = load_state_year_shard(path)
    return int(shard.year)


def _run_lengths(codes: np.ndarray) -> list[tuple[int, int]]:
    values = np.asarray(codes, dtype=np.int8)
    if not len(values):
        return []
    runs: list[tuple[int, int]] = []
    current = int(values[0])
    length = 1
    for value in values[1:]:
        value = int(value)
        if value == current:
            length += 1
        else:
            runs.append((current, length))
            current = value
            length = 1
    runs.append((current, length))
    return runs


def build_state_duration_atlas(
    state_paths: list[Path],
    config: MarketStateContinuityConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    bars_per_hour = 60 / config.decision_interval_minutes
    for path in state_paths:
        shard = load_state_year_shard(path)
        year = int(shard.year)
        for layer in ("strategic", "tactical", "entry", "activity"):
            for kind in ("raw", "stable"):
                column = f"{layer}_{'raw_state' if kind == 'raw' else 'state'}"
                values = np.asarray(shard.states[:, shard.state_index[column]], dtype=np.int8)
                all_runs = _run_lengths(values)
                flips = int(np.sum(values[1:] != values[:-1])) if len(values) > 1 else 0
                days = max(len(values) * config.decision_interval_minutes / 1440.0, 1e-9)
                for code in (-1, 0, 1):
                    durations = np.asarray(
                        [length / bars_per_hour for state, length in all_runs if state == code],
                        dtype=float,
                    )
                    rows.append(
                        {
                            "year": year,
                            "layer": layer,
                            "classification": kind,
                            "state_code": code,
                            "state_name": STATE_NAMES[code],
                            "runs": int(len(durations)),
                            "share_of_bars": float(np.mean(values == code)),
                            "median_duration_hours": float(np.median(durations)) if len(durations) else np.nan,
                            "q75_duration_hours": float(np.quantile(durations, 0.75)) if len(durations) else np.nan,
                            "q90_duration_hours": float(np.quantile(durations, 0.90)) if len(durations) else np.nan,
                            "maximum_duration_hours": float(np.max(durations)) if len(durations) else np.nan,
                            "all_state_flips": flips,
                            "flips_per_day": float(flips / days),
                        }
                    )
    return pd.DataFrame(rows)


def build_state_target_distribution(
    state_paths: list[Path],
    config: MarketStateContinuityConfig,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in state_paths:
        shard = load_state_year_shard(path)
        year = int(shard.year)
        for spec in config.targets:
            if spec.target_id not in shard.target_index:
                continue
            values = np.asarray(shard.targets[:, shard.target_index[spec.target_id]], dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            rows.append(
                {
                    "year": year,
                    "target": spec.target_id,
                    "layer": spec.layer,
                    "horizon_hours": spec.horizon_hours,
                    "rows": int(len(values)),
                    "persistence_rate": float(np.mean(values)),
                    "transition_rate": float(np.mean(1.0 - values)),
                }
            )
    return pd.DataFrame(rows)


def _intensity_paths() -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in DEFAULT_FUTURE_INTENSITY_CONFIG.target_cache_path.glob("targets_????"):
        if not (path / "manifest.json").exists():
            continue
        shard = load_intensity_year_shard(path)
        year = int(pd.to_datetime(np.asarray(shard.decision_times_ns[:1], dtype=np.int64), unit="ns")[0].year)
        paths[year] = path
    return paths


def build_state_opportunity_link(
    state_paths: list[Path],
) -> pd.DataFrame:
    intensity_map = _intensity_paths()
    rows: list[pd.DataFrame] = []
    for path in state_paths:
        state = load_state_year_shard(path)
        year = int(state.year)
        if year not in intensity_map:
            continue
        target = load_intensity_year_shard(intensity_map[year])
        common, state_positions, target_positions = np.intersect1d(
            np.asarray(state.decision_times_ns, dtype=np.int64),
            np.asarray(target.decision_times_ns, dtype=np.int64),
            assume_unique=True,
            return_indices=True,
        )
        if not len(common):
            continue
        required = (
            "future_range_pct_h6",
            "future_max_directional_pct_h6",
            "future_two_sided_pct_h6",
        )
        if any(name not in target.target_index for name in required):
            continue
        frame = pd.DataFrame({"decision_time": ns_to_datetime(common)})
        for layer in ("strategic", "tactical", "entry", "activity"):
            frame[f"{layer}_state"] = np.asarray(
                state.states[state_positions, state.state_index[f"{layer}_state"]],
                dtype=np.int8,
            )
        for name in required:
            frame[name] = np.asarray(target.targets[target_positions, target.target_index[name]], dtype=float)
        frame["year"] = year
        threshold = float(np.nanquantile(frame["future_range_pct_h6"], 0.90))
        frame["future_range_top_decile"] = frame["future_range_pct_h6"] >= threshold
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    group_columns = ["year", "strategic_state", "tactical_state", "entry_state", "activity_state"]
    grouped = (
        combined.groupby(group_columns, observed=True)
        .agg(
            rows=("future_range_pct_h6", "size"),
            mean_future_range_h6=("future_range_pct_h6", "mean"),
            median_future_range_h6=("future_range_pct_h6", "median"),
            q75_future_range_h6=("future_range_pct_h6", lambda values: values.quantile(0.75)),
            mean_future_directional_h6=("future_max_directional_pct_h6", "mean"),
            mean_future_two_sided_h6=("future_two_sided_pct_h6", "mean"),
            high_opportunity_rate=("future_range_top_decile", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(
        ["year", "mean_future_range_h6", "rows"],
        ascending=[True, False, False],
        kind="stable",
    )


def build_state_samples(state_paths: list[Path], every: int = 96) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for path in state_paths:
        shard = load_state_year_shard(path)
        positions = np.arange(0, len(shard.decision_times_ns), max(1, every))
        frame = pd.DataFrame({"decision_time": ns_to_datetime(np.asarray(shard.decision_times_ns)[positions])})
        for column in shard.state_columns:
            frame[column] = np.asarray(shard.states[positions, shard.state_index[column]], dtype=float)
        parts.append(frame)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def build_strategic_threshold_audit(state_paths: list[Path]) -> pd.DataFrame:
    """Summarise causal strategic thresholds and resulting state occupancy by year."""
    rows: list[dict[str, object]] = []
    threshold_columns = (
        "strategic_long_enter_threshold",
        "strategic_short_enter_threshold",
        "strategic_long_exit_threshold",
        "strategic_short_exit_threshold",
    )
    for path in state_paths:
        shard = load_state_year_shard(path)
        missing = [column for column in threshold_columns if column not in shard.feature_columns]
        if missing:
            continue
        state_values = np.asarray(
            shard.states[:, shard.state_index["strategic_state"]],
            dtype=np.int8,
        )
        row: dict[str, object] = {
            "year": int(shard.year),
            "rows": int(len(state_values)),
            "strategic_short_share": float(np.mean(state_values == -1)),
            "strategic_neutral_share": float(np.mean(state_values == 0)),
            "strategic_long_share": float(np.mean(state_values == 1)),
            "strategic_flips": int(np.sum(state_values[1:] != state_values[:-1]))
            if len(state_values) > 1
            else 0,
        }
        for column in threshold_columns:
            values = np.asarray(
                shard.features[:, shard.feature_columns.index(column)],
                dtype=float,
            )
            finite = values[np.isfinite(values)]
            row[f"{column}_median"] = float(np.median(finite)) if len(finite) else np.nan
            row[f"{column}_q10"] = float(np.quantile(finite, 0.10)) if len(finite) else np.nan
            row[f"{column}_q90"] = float(np.quantile(finite, 0.90)) if len(finite) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
