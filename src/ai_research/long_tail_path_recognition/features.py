#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal checkpoint features and ex-post labels for R03.4.2.2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import MinutePathData
from src.ai_research.long_tail_path_atlas.atlas import EventPathExtraction

from .config import LongTailPathRecognitionConfig


@dataclass(frozen=True)
class FeatureSet:
    name: str
    columns: tuple[str, ...]


def _safe_div(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


def _linear_slope_r2(values: np.ndarray) -> tuple[float, float]:
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < 3:
        return 0.0, 0.0
    y = y[valid]
    x = np.arange(len(y), dtype=float)
    x_mean = float(x.mean())
    y_mean = float(y.mean())
    denominator = float(np.square(x - x_mean).sum())
    if denominator <= 0:
        return 0.0, 0.0
    slope = float(((x - x_mean) * (y - y_mean)).sum() / denominator)
    fitted = y_mean + slope * (x - x_mean)
    total = float(np.square(y - y_mean).sum())
    residual = float(np.square(y - fitted).sum())
    r2 = 1.0 - residual / total if total > 1e-15 else 0.0
    return slope, float(np.clip(r2, -1.0, 1.0))


def _realized_path_features(close_ret: np.ndarray) -> dict[str, float]:
    prices = np.log1p(np.asarray(close_ret, dtype=float))
    increments = np.diff(prices)
    if not len(increments):
        return {
            "realized_vol": 0.0,
            "upside_semivol": 0.0,
            "downside_semivol": 0.0,
            "down_up_vol_ratio": 0.0,
        }
    realized = float(np.sqrt(np.square(increments).sum()))
    upside = float(np.sqrt(np.square(increments[increments > 0]).sum())) if np.any(increments > 0) else 0.0
    downside = float(np.sqrt(np.square(increments[increments < 0]).sum())) if np.any(increments < 0) else 0.0
    return {
        "realized_vol": realized,
        "upside_semivol": upside,
        "downside_semivol": downside,
        "down_up_vol_ratio": _safe_div(downside, upside + 1e-12),
    }


def _fifteen_minute_structure(points: pd.DataFrame) -> dict[str, float]:
    if points.empty:
        return {
            "bars_15m": 0.0,
            "higher_high_share_15m": 0.0,
            "higher_low_share_15m": 0.0,
            "lower_high_share_15m": 0.0,
            "lower_low_share_15m": 0.0,
            "positive_close_share_15m": 0.0,
            "last3_close_slope_15m": 0.0,
        }
    groups = np.arange(len(points)) // 15
    bars = points.groupby(groups, sort=True).agg(
        high=("high_return", "max"),
        low=("low_return", "min"),
        close=("close_return", "last"),
    )
    if len(bars) < 2:
        return {
            "bars_15m": float(len(bars)),
            "higher_high_share_15m": 0.0,
            "higher_low_share_15m": 0.0,
            "lower_high_share_15m": 0.0,
            "lower_low_share_15m": 0.0,
            "positive_close_share_15m": float((bars["close"] > 0).mean()) if len(bars) else 0.0,
            "last3_close_slope_15m": 0.0,
        }
    high_diff = bars["high"].diff().dropna().to_numpy(dtype=float)
    low_diff = bars["low"].diff().dropna().to_numpy(dtype=float)
    last = bars["close"].tail(3).to_numpy(dtype=float)
    slope, _ = _linear_slope_r2(last)
    return {
        "bars_15m": float(len(bars)),
        "higher_high_share_15m": float(np.mean(high_diff > 0)),
        "higher_low_share_15m": float(np.mean(low_diff > 0)),
        "lower_high_share_15m": float(np.mean(high_diff < 0)),
        "lower_low_share_15m": float(np.mean(low_diff < 0)),
        "positive_close_share_15m": float((bars["close"] > 0).mean()),
        "last3_close_slope_15m": float(slope),
    }


def _prefix(values: Mapping[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{name}": float(value) for name, value in values.items()}


def build_checkpoint_row(
    extraction: EventPathExtraction,
    *,
    checkpoint_minutes: int,
    path: MinutePathData,
    config: LongTailPathRecognitionConfig,
) -> dict[str, object]:
    """Build features visible at the end of a fixed checkpoint.

    Future rows are used only for labels and outcome diagnostics. All columns
    beginning with ``x_`` are computed from rows ``[0, checkpoint)`` plus
    pre-entry structural values.
    """

    if checkpoint_minutes not in config.checkpoints_minutes:
        raise ValueError(f"unsupported checkpoint: {checkpoint_minutes}")
    summary = extraction.summary
    points = extraction.points.iloc[:checkpoint_minutes].copy()
    if len(points) != checkpoint_minutes:
        raise RuntimeError("checkpoint path is incomplete")
    close_ret = points["close_return"].to_numpy(dtype=float)
    high_ret = points["high_return"].to_numpy(dtype=float)
    low_ret = points["low_return"].to_numpy(dtype=float)
    current_ret = float(close_ret[-1])
    current_mfe = float(np.max(high_ret))
    current_mae = float(-np.min(low_ret))
    current_peak_giveback = float(current_mfe - current_ret)
    recovery_from_trough = float(current_ret + current_mae)
    close_location = _safe_div(current_ret + current_mae, current_mfe + current_mae)
    capture_of_mfe = _safe_div(current_ret, current_mfe)
    range_width = float(current_mfe + current_mae)
    minutes_since_mfe = float(checkpoint_minutes - 1 - int(np.argmax(high_ret)))
    minutes_since_mae = float(checkpoint_minutes - 1 - int(np.argmin(low_ret)))
    first_half = float(close_ret[max(0, checkpoint_minutes // 2 - 1)])
    second_half_return = float(current_ret - first_half)
    first_half_return = first_half
    acceleration = float(second_half_return - first_half_return)
    last30 = float(current_ret - close_ret[max(0, checkpoint_minutes - 31)]) if checkpoint_minutes > 30 else current_ret
    last60 = float(current_ret - close_ret[max(0, checkpoint_minutes - 61)]) if checkpoint_minutes > 60 else current_ret
    slope, r2 = _linear_slope_r2(np.log1p(close_ret))
    last60_slope, last60_r2 = _linear_slope_r2(np.log1p(close_ret[-min(60, len(close_ret)) :]))

    entry_time = pd.Timestamp(summary["entry_time"])
    entry_position = path.locate_exact(entry_time)
    if entry_position is None:
        raise RuntimeError("entry timestamp is missing from minute path")
    prior_low_60 = float(path.prior_low_60[entry_position])
    prior_low_180 = float(path.prior_low_180[entry_position])
    entry_price = float(summary["entry_price"])
    checkpoint_close_price = entry_price * (1.0 + current_ret)
    checkpoint_low_price = entry_price * (1.0 + float(np.min(low_ret)))

    structural = {
        "current_return": current_ret,
        "current_mfe": current_mfe,
        "current_mae": current_mae,
        "peak_giveback": current_peak_giveback,
        "recovery_from_trough": recovery_from_trough,
        "close_location_in_path_range": close_location,
        "capture_of_mfe": capture_of_mfe,
        "path_range_width": range_width,
        "minutes_since_mfe": minutes_since_mfe,
        "minutes_since_mae": minutes_since_mae,
        "underwater_fraction": float(np.mean(close_ret < 0)),
        "closes_above_entry_fraction": float(np.mean(close_ret > 0)),
        "last30_return": last30,
        "last60_return": last60,
        "first_half_return": first_half_return,
        "second_half_return": second_half_return,
        "path_acceleration": acceleration,
        "log_close_slope": slope,
        "log_close_r2": r2,
        "last60_log_close_slope": last60_slope,
        "last60_log_close_r2": last60_r2,
        "distance_to_prior_low_60": _safe_div(checkpoint_close_price, prior_low_60) - 1.0 if np.isfinite(prior_low_60) and prior_low_60 > 0 else np.nan,
        "distance_to_prior_low_180": _safe_div(checkpoint_close_price, prior_low_180) - 1.0 if np.isfinite(prior_low_180) and prior_low_180 > 0 else np.nan,
        "broke_prior_low_60": float(np.isfinite(prior_low_60) and checkpoint_low_price < prior_low_60),
        "broke_prior_low_180": float(np.isfinite(prior_low_180) and checkpoint_low_price < prior_low_180),
        "reclaimed_entry_after_drawdown": float(np.min(close_ret) < 0 and current_ret > 0),
        "current_below_entry": float(current_ret < 0),
    }
    structural.update(_realized_path_features(close_ret))
    structural.update(_fifteen_minute_structure(points))

    score_columns = {
        "entry_score_percentile": float(summary["event_score_percentile"]),
        "score_percentile_end": float(summary.get(f"score_percentile_end_{checkpoint_minutes}m", np.nan)),
        "score_percentile_min": float(summary.get(f"score_percentile_min_{checkpoint_minutes}m", np.nan)),
        "score_percentile_max": float(summary.get(f"score_percentile_max_{checkpoint_minutes}m", np.nan)),
        "score_percentile_change": float(summary.get(f"score_percentile_change_{checkpoint_minutes}m", np.nan)),
        "q90_reconfirmations": float(summary.get(f"q90_reconfirmations_{checkpoint_minutes}m", 0.0)),
        "q95_reconfirmations": float(summary.get(f"q95_reconfirmations_{checkpoint_minutes}m", 0.0)),
        "first_below_q70": float(summary.get(f"first_below_q70_{checkpoint_minutes}m", np.nan)),
        "first_below_q50": float(summary.get(f"first_below_q50_{checkpoint_minutes}m", np.nan)),
        "longest_below_q50_decisions": float(summary.get(f"longest_below_q50_decisions_{checkpoint_minutes}m", 0.0)),
    }

    full_points = extraction.points
    future_to_6h = full_points.iloc[checkpoint_minutes:360]
    if future_to_6h.empty:
        future_giveback = 0.0
    else:
        future_low_close = float(future_to_6h["close_return"].min())
        future_giveback = float(current_mfe - future_low_close)

    ret24 = float(summary["ret_1440m"])
    ret48 = float(summary["ret_2880m"])
    mfe24 = float(summary["mfe_1440m"])
    persistent_failure = bool(summary["flag_persistent_failure"])
    recovery_eligible = bool(current_ret <= 0 or current_mae >= 0.005 or float(np.mean(close_ret < 0)) >= 0.50)
    recoverable = bool(
        recovery_eligible
        and ret24 - config.base_round_trip_cost > config.recovery_positive_24h_net
        and mfe24 >= config.recovery_min_24h_mfe
    )
    clearly_not_recoverable = bool(
        recovery_eligible
        and ret24 <= 0
        and mfe24 < config.persistent_failure_max_24h_mfe
    )
    recovery_label = 1.0 if recoverable else (0.0 if clearly_not_recoverable else np.nan)
    giveback_eligible = bool(checkpoint_minutes < 360 and current_mfe >= config.giveback_activation_mfe)
    giveback_label = (
        float(future_giveback >= config.giveback_future_loss_from_checkpoint)
        if giveback_eligible
        else np.nan
    )

    row: dict[str, object] = {
        "event_id": summary["event_id"],
        "fold_id": summary["fold_id"],
        "phase": summary["phase"],
        "decision_time": summary["decision_time"],
        "entry_time": summary["entry_time"],
        "signal_quantile": float(summary["signal_quantile"]),
        "event_score_percentile": float(summary["event_score_percentile"]),
        "is_q90": bool(float(summary["event_score_percentile"]) >= config.primary_signal_quantile),
        "is_q95": bool(float(summary["event_score_percentile"]) >= config.quality_control_quantile),
        "checkpoint_minutes": int(checkpoint_minutes),
        "checkpoint_net_exit_1x": current_ret - config.base_round_trip_cost,
        "fixed6h_net_1x": float(summary["fixed6h_net_1x"]),
        "net_24h_1x": ret24 - config.base_round_trip_cost,
        "net_48h_1x": ret48 - config.base_round_trip_cost,
        "post6_mfe_increment_24h": float(summary["post6_mfe_increment_1440m"]),
        "label_persistent_failure": float(persistent_failure),
        "eligible_recovery": float(recovery_eligible),
        "label_recovery": recovery_label,
        "eligible_giveback": float(giveback_eligible),
        "label_giveback_risk": giveback_label,
        "label_post6_continuation": float(bool(summary["flag_post6_continuation"])) if checkpoint_minutes == 360 else np.nan,
        "semantic_path_type": summary["semantic_path_type"],
    }
    row.update(_prefix(structural, "x_path__"))
    row.update(_prefix(score_columns, "x_score__"))
    return row


def feature_sets(frame: pd.DataFrame) -> tuple[FeatureSet, ...]:
    path_columns = tuple(column for column in frame.columns if column.startswith("x_path__"))
    score_columns = tuple(column for column in frame.columns if column.startswith("x_score__"))
    if not path_columns or not score_columns:
        raise RuntimeError("R03.4.2.2 causal feature columns are missing")
    mechanical_names = (
        "x_path__current_return",
        "x_path__current_mfe",
        "x_path__current_mae",
        "x_path__peak_giveback",
        "x_path__underwater_fraction",
        "x_path__capture_of_mfe",
    )
    mechanical = tuple(name for name in mechanical_names if name in frame.columns)
    return (
        FeatureSet("mechanical_path_logistic", mechanical),
        FeatureSet("score_only_logistic", score_columns),
        FeatureSet("path_structure_logistic", path_columns),
        FeatureSet("path_plus_score_logistic", (*path_columns, *score_columns)),
        FeatureSet("path_plus_score_lightgbm", (*path_columns, *score_columns)),
    )


def task_target(frame: pd.DataFrame, task: str) -> tuple[pd.DataFrame, np.ndarray]:
    if task == "persistent_failure":
        work = frame.loc[frame["label_persistent_failure"].notna()].copy()
        target = work["label_persistent_failure"].to_numpy(dtype=np.int8)
    elif task == "recovery_from_underwater":
        work = frame.loc[frame["label_recovery"].notna()].copy()
        target = work["label_recovery"].to_numpy(dtype=np.int8)
    elif task == "giveback_risk":
        work = frame.loc[frame["label_giveback_risk"].notna()].copy()
        target = work["label_giveback_risk"].to_numpy(dtype=np.int8)
    elif task == "post6_continuation":
        work = frame.loc[frame["label_post6_continuation"].notna()].copy()
        target = work["label_post6_continuation"].to_numpy(dtype=np.int8)
    else:
        raise ValueError(f"unsupported path-recognition task: {task}")
    return work, target
