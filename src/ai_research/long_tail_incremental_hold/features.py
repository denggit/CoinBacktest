#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal checkpoint features and incremental holding-value labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.ai_research.long_tail_multistage_decision.features import (
    ExtendedEventPath,
    FeatureSet,
    build_checkpoint_row,
)
from src.ai_research.long_tail_exit_audit.data import MinutePathData
from src.ai_research.long_tail_exit_audit.simulator import ScoreTimeline

from .config import IncrementalHoldConfig


@dataclass(frozen=True)
class HoldTarget:
    name: str
    endpoint_minutes: int


def score_tier(percentile: float) -> str:
    value = float(percentile)
    if value >= 0.90:
        return "q90_plus"
    if value >= 0.80:
        return "q80_to_q90"
    return "q70_to_q80"


def eligible_endpoints(checkpoint_minutes: int, config: IncrementalHoldConfig) -> tuple[int, ...]:
    return tuple(value for value in config.future_endpoints_minutes if value > checkpoint_minutes)


def primary_endpoint(checkpoint_minutes: int, config: IncrementalHoldConfig) -> int:
    endpoints = eligible_endpoints(checkpoint_minutes, config)
    if not endpoints:
        raise ValueError(f"no future endpoint after checkpoint={checkpoint_minutes}")
    return endpoints[0]


def _incremental_target(
    points: pd.DataFrame,
    *,
    checkpoint_minutes: int,
    endpoint_minutes: int,
    risk_penalty: float,
) -> dict[str, float]:
    if endpoint_minutes <= checkpoint_minutes:
        raise ValueError("endpoint must be after checkpoint")
    if endpoint_minutes > len(points):
        raise ValueError("endpoint exceeds extracted path")

    current_return = float(points["close_return"].iloc[checkpoint_minutes - 1])
    future = points.iloc[checkpoint_minutes:endpoint_minutes]
    if future.empty:
        raise ValueError("future target window is empty")
    endpoint_return = float(points["close_return"].iloc[endpoint_minutes - 1])
    future_low = float(future["low_return"].min())
    future_high = float(future["high_return"].max())
    incremental_close = endpoint_return - current_return
    additional_drawdown = max(0.0, current_return - future_low)
    additional_mfe = max(0.0, future_high - current_return)
    incremental_utility = incremental_close - float(risk_penalty) * additional_drawdown
    return {
        "incremental_close_return": incremental_close,
        "additional_drawdown": additional_drawdown,
        "additional_mfe": additional_mfe,
        "incremental_utility": incremental_utility,
    }


def build_incremental_hold_row(
    extraction: ExtendedEventPath,
    *,
    checkpoint_minutes: int,
    path: MinutePathData,
    timeline: ScoreTimeline,
    config: IncrementalHoldConfig,
) -> dict[str, object]:
    """Build one causal row plus ex-post holding-value labels.

    All ``x_*`` columns come from data available through the checkpoint. Future
    path values are written only to ``label_*`` / ``actual_*`` columns.
    """

    row = build_checkpoint_row(
        extraction,
        checkpoint_minutes=checkpoint_minutes,
        path=path,
        timeline=timeline,
        config=config,  # duck-typed; the feature builder only uses shared fields.
    )
    endpoints = eligible_endpoints(checkpoint_minutes, config)
    utilities: list[tuple[int, float]] = []
    for endpoint in endpoints:
        target = _incremental_target(
            extraction.points,
            checkpoint_minutes=checkpoint_minutes,
            endpoint_minutes=endpoint,
            risk_penalty=config.risk_penalty,
        )
        prefix = f"actual_to_{endpoint}m"
        row[f"{prefix}_incremental_close_return"] = target["incremental_close_return"]
        row[f"{prefix}_additional_drawdown"] = target["additional_drawdown"]
        row[f"{prefix}_additional_mfe"] = target["additional_mfe"]
        row[f"{prefix}_incremental_utility"] = target["incremental_utility"]
        utilities.append((endpoint, float(target["incremental_utility"])))

    next_endpoint = primary_endpoint(checkpoint_minutes, config)
    row["primary_endpoint_minutes"] = next_endpoint
    row["label_next_incremental_utility"] = float(row[f"actual_to_{next_endpoint}m_incremental_utility"])
    best_endpoint, best_utility = max(utilities, key=lambda item: item[1])
    row["label_best_incremental_utility"] = float(best_utility)
    row["label_best_endpoint_minutes"] = int(best_endpoint)
    row["label_continue_positive_next"] = int(
        row["label_next_incremental_utility"] > config.positive_utility_buffer
    )
    row["label_continue_positive_any"] = int(best_utility > config.positive_utility_buffer)
    row["score_tier"] = score_tier(float(row["event_score_percentile"]))
    row["current_mark_return"] = float(row["x_path__current_return"])
    return row


def regression_targets(frame: pd.DataFrame) -> tuple[HoldTarget, ...]:
    if frame.empty:
        return ()
    endpoint = int(frame["primary_endpoint_minutes"].iloc[0])
    return (
        HoldTarget("next_incremental_utility", endpoint),
        HoldTarget("best_incremental_utility", -1),
    )


def target_values(frame: pd.DataFrame, target_name: str) -> np.ndarray:
    column = {
        "next_incremental_utility": "label_next_incremental_utility",
        "best_incremental_utility": "label_best_incremental_utility",
    }.get(target_name)
    if column is None:
        raise ValueError(f"unsupported target {target_name}")
    return frame[column].to_numpy(dtype=float)


def feature_sets(frame: pd.DataFrame) -> tuple[FeatureSet, ...]:
    path = tuple(sorted(column for column in frame.columns if column.startswith("x_path__")))
    score = tuple(sorted(column for column in frame.columns if column.startswith("x_score__")))
    mechanical_names = (
        "x_path__current_return",
        "x_path__current_mfe",
        "x_path__current_mae",
        "x_path__peak_giveback",
        "x_path__recovery_from_trough",
        "x_path__capture_of_mfe",
        "x_path__underwater_fraction",
        "x_path__last60_return",
        "x_path__path_acceleration",
        "x_path__last60_log_close_slope",
        "x_path__last60_log_close_r2",
        "x_path__down_up_vol_ratio",
        "x_path__distance_to_prior_low_60",
        "x_path__distance_to_prior_low_180",
        "x_path__broke_prior_low_60",
        "x_path__broke_prior_low_180",
        "x_path__bar15_higher_low_share",
        "x_path__bar15_lower_low_share",
        "x_path__bar15_positive_close_share",
    )
    mechanical = tuple(column for column in mechanical_names if column in frame.columns)
    return (
        FeatureSet("mechanical_ridge", mechanical),
        FeatureSet("path_structure_ridge", path),
        FeatureSet("path_structure_lightgbm", path),
        FeatureSet("score_only_lightgbm", score),
        FeatureSet("path_plus_score_lightgbm", (*path, *score)),
    )


def prediction_columns(frame: pd.DataFrame) -> list[str]:
    preferred = [
        "event_id",
        "fold_id",
        "phase",
        "scope",
        "decision_time",
        "entry_time",
        "checkpoint_minutes",
        "primary_endpoint_minutes",
        "event_score_percentile",
        "score_tier",
        "current_mark_return",
        "label_next_incremental_utility",
        "label_best_incremental_utility",
        "label_best_endpoint_minutes",
    ]
    return [column for column in preferred if column in frame.columns]


def future_label_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in frame.columns
        if column.startswith("label_") or column.startswith("actual_to_")
    )


def assert_no_future_features(columns: Iterable[str]) -> None:
    bad = [column for column in columns if column.startswith("label_") or column.startswith("actual_to_")]
    if bad:
        raise ValueError(f"future columns leaked into model features: {bad[:5]}")
