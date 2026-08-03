#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Strict rolling OOF opening scores for the holding-model training events."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ai_research.long_state_calibration.config import LongStateCalibrationConfig
from src.ai_research.long_state_calibration.modeling import fit_base_long_model, subset_period_data
from src.ai_research.long_tail_exit_audit.simulator import EventCandidate, ScoreTimeline, build_event_candidates
from src.ai_research.state_context_ablation.modeling import AblationPeriodData

from .config import LongTailMultistageConfig


@dataclass(frozen=True)
class EntryOOFResult:
    timeline: ScoreTimeline
    events: tuple[EventCandidate, ...]
    audit: pd.DataFrame
    valid_rows: int


def empirical_percentile(reference: np.ndarray, values: np.ndarray) -> np.ndarray:
    ref = np.sort(np.asarray(reference, dtype=float)[np.isfinite(reference)])
    output = np.full(len(values), np.nan, dtype=float)
    valid = np.isfinite(values)
    if len(ref):
        output[valid] = np.searchsorted(ref, np.asarray(values, dtype=float)[valid], side="right") / len(ref)
    return output


def _base_config(config: LongTailMultistageConfig) -> LongStateCalibrationConfig:
    return LongStateCalibrationConfig(
        primary_horizon_hours=config.primary_horizon_hours,
        risk_penalty=config.risk_penalty,
        base_round_trip_cost=config.base_round_trip_cost,
        signal_quantiles=(0.90, 0.95),
        independent_event_cooldown_hours=config.independent_event_cooldown_hours,
        episode_merge_gap_minutes=config.episode_merge_gap_minutes,
        oof_min_train_days=config.entry_oof_min_train_days,
        oof_blocks=max(3, config.entry_oof_blocks),
        oof_embargo_hours=config.entry_oof_embargo_hours,
        train_sample_cap=config.base_train_sample_cap,
        base_n_estimators=config.base_n_estimators,
        base_learning_rate=config.base_learning_rate,
        base_num_leaves=config.base_num_leaves,
        base_min_child_samples=config.base_min_child_samples,
        random_state=config.random_state,
    )


def _positions(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray:
    return np.flatnonzero((index >= start) & (index <= end))


def build_rolling_oof_entry_timeline(
    data: AblationPeriodData,
    *,
    event_builder_config,
    config: LongTailMultistageConfig,
) -> EntryOOFResult:
    """Generate normalized OOF scores with separate train/calibration/validation blocks.

    For every prediction block, the opening model is trained before a trailing
    calibration window. The validation scores are converted to percentiles using
    that separate calibration window. This prevents in-sample score thresholds
    from creating the holding-model training events.
    """

    index = data.index
    if len(index) < 10_000 or not index.is_monotonic_increasing:
        raise RuntimeError("insufficient or unsorted base decision data for rolling OOF entry scores")
    warmup = pd.Timedelta(days=config.entry_oof_min_train_days + config.entry_oof_calibration_days)
    first_prediction = index[0] + warmup + pd.Timedelta(hours=config.entry_oof_embargo_hours)
    prediction_positions = np.flatnonzero(index >= first_prediction)
    if len(prediction_positions) < config.entry_oof_blocks * 500:
        raise RuntimeError("insufficient prediction rows for rolling OOF entry timeline")
    blocks = [chunk for chunk in np.array_split(prediction_positions, config.entry_oof_blocks) if len(chunk)]
    percentiles = np.full(len(index), np.nan, dtype=float)
    raw_scores = np.full(len(index), np.nan, dtype=float)
    audit_rows: list[dict[str, object]] = []
    embargo = pd.Timedelta(hours=config.entry_oof_embargo_hours)
    base_config = _base_config(config)

    for block_id, block in enumerate(blocks, start=1):
        validation_start = index[int(block[0])]
        validation_end = index[int(block[-1])]
        calibration_end = validation_start - embargo
        calibration_start = calibration_end - pd.Timedelta(days=config.entry_oof_calibration_days)
        train_end = calibration_start - embargo
        train_pos = _positions(index, index[0], train_end)
        calibration_pos = _positions(index, calibration_start, calibration_end)
        validation_pos = np.asarray(block, dtype=np.int64)
        if len(train_pos) < 5_000 or len(calibration_pos) < 500 or len(validation_pos) < 100:
            continue
        model = fit_base_long_model(subset_period_data(data, train_pos), base_config)
        calibration_score = model.predict(np.asarray(data.base_x[calibration_pos], dtype=np.float32))
        validation_score = model.predict(np.asarray(data.base_x[validation_pos], dtype=np.float32))
        validation_percentile = empirical_percentile(calibration_score, validation_score)
        raw_scores[validation_pos] = validation_score
        percentiles[validation_pos] = validation_percentile
        audit_rows.append(
            {
                "block_id": block_id,
                "train_start": str(index[int(train_pos[0])]),
                "train_end": str(index[int(train_pos[-1])]),
                "calibration_start": str(index[int(calibration_pos[0])]),
                "calibration_end": str(index[int(calibration_pos[-1])]),
                "validation_start": str(validation_start),
                "validation_end": str(validation_end),
                "train_rows": int(len(train_pos)),
                "calibration_rows": int(len(calibration_pos)),
                "validation_rows": int(len(validation_pos)),
                "score_q50": float(np.nanquantile(calibration_score, 0.50)),
                "score_q70": float(np.nanquantile(calibration_score, 0.70)),
                "score_q90": float(np.nanquantile(calibration_score, 0.90)),
                "maximum_train_time_ns": int(np.asarray(data.timestamps_ns)[train_pos].max()),
                "minimum_calibration_time_ns": int(np.asarray(data.timestamps_ns)[calibration_pos].min()),
                "minimum_validation_time_ns": int(np.asarray(data.timestamps_ns)[validation_pos].min()),
                "embargo_hours": config.entry_oof_embargo_hours,
            }
        )

    valid = np.isfinite(percentiles)
    if valid.sum() < 5_000 or len(audit_rows) < 3:
        raise RuntimeError(f"rolling OOF entry timeline too small rows={int(valid.sum())} blocks={len(audit_rows)}")
    timeline = ScoreTimeline(
        decision_times_ns=np.asarray(data.timestamps_ns, dtype=np.int64),
        scores=percentiles,
        calibration_thresholds={0.50: 0.50, 0.60: 0.60, 0.70: 0.70, 0.90: 0.90, 0.95: 0.95},
    )
    events = build_event_candidates(
        timeline,
        signal_quantile=config.train_event_quantile,
        config=event_builder_config,
    )
    return EntryOOFResult(
        timeline=timeline,
        events=events,
        audit=pd.DataFrame(audit_rows),
        valid_rows=int(valid.sum()),
    )


def build_oos_percentile_timeline(
    timestamps_ns: np.ndarray,
    test_scores: np.ndarray,
    calibration_scores: np.ndarray,
) -> ScoreTimeline:
    percentiles = empirical_percentile(calibration_scores, test_scores)
    return ScoreTimeline(
        decision_times_ns=np.asarray(timestamps_ns, dtype=np.int64),
        scores=percentiles,
        calibration_thresholds={0.50: 0.50, 0.60: 0.60, 0.70: 0.70, 0.90: 0.90, 0.95: 0.95},
    )
