#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Five-day path extraction, causal checkpoint features and future labels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src.ai_research.long_tail_exit_audit.data import MinutePathData
from src.ai_research.long_tail_exit_audit.simulator import EventCandidate, ScoreTimeline

from .config import LongTailMultistageConfig

_ONE_MINUTE_NS = int(pd.Timedelta(minutes=1).value)


@dataclass(frozen=True)
class ExtendedEventPath:
    summary: dict[str, object]
    points: pd.DataFrame


@dataclass(frozen=True)
class FeatureSet:
    name: str
    columns: tuple[str, ...]


def _safe_div(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0
    return float(numerator / denominator)


def _longest_true_run(values: np.ndarray) -> int:
    mask = np.asarray(values, dtype=bool)
    if not mask.any():
        return 0
    padded = np.concatenate([[False], mask, [False]]).astype(np.int8)
    edges = np.diff(padded)
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return int(np.max(ends - starts))


def _slope_r2(values: np.ndarray) -> tuple[float, float]:
    y = np.asarray(values, dtype=float)
    valid = np.isfinite(y)
    if valid.sum() < 3:
        return 0.0, 0.0
    y = y[valid]
    x = np.arange(len(y), dtype=float)
    x -= x.mean()
    y_centered = y - y.mean()
    denominator = float(np.square(x).sum())
    if denominator <= 0:
        return 0.0, 0.0
    slope = float(np.sum(x * y_centered) / denominator)
    fitted = y.mean() + slope * x
    total = float(np.square(y_centered).sum())
    residual = float(np.square(y - fitted).sum())
    return slope, float(1.0 - residual / total) if total > 1e-15 else 0.0


def _bar_structure(points: pd.DataFrame, minutes_per_bar: int, prefix: str) -> dict[str, float]:
    groups = np.arange(len(points)) // minutes_per_bar
    bars = points.groupby(groups, sort=True).agg(
        high=("high_return", "max"),
        low=("low_return", "min"),
        close=("close_return", "last"),
    )
    if bars.empty:
        return {
            f"{prefix}_bars": 0.0,
            f"{prefix}_higher_high_share": 0.0,
            f"{prefix}_higher_low_share": 0.0,
            f"{prefix}_lower_high_share": 0.0,
            f"{prefix}_lower_low_share": 0.0,
            f"{prefix}_positive_close_share": 0.0,
            f"{prefix}_last3_close_slope": 0.0,
        }
    high_diff = bars["high"].diff().dropna().to_numpy(dtype=float)
    low_diff = bars["low"].diff().dropna().to_numpy(dtype=float)
    slope, _ = _slope_r2(bars["close"].tail(3).to_numpy(dtype=float))
    return {
        f"{prefix}_bars": float(len(bars)),
        f"{prefix}_higher_high_share": float(np.mean(high_diff > 0)) if len(high_diff) else 0.0,
        f"{prefix}_higher_low_share": float(np.mean(low_diff > 0)) if len(low_diff) else 0.0,
        f"{prefix}_lower_high_share": float(np.mean(high_diff < 0)) if len(high_diff) else 0.0,
        f"{prefix}_lower_low_share": float(np.mean(low_diff < 0)) if len(low_diff) else 0.0,
        f"{prefix}_positive_close_share": float(np.mean(bars["close"] > 0)),
        f"{prefix}_last3_close_slope": float(slope),
    }


def _score_percentile_at_minutes(
    event: EventCandidate,
    timeline: ScoreTimeline,
    minutes: int,
) -> dict[str, float]:
    start = event.decision_time_ns
    end = start + minutes * _ONE_MINUTE_NS
    left = int(np.searchsorted(timeline.decision_times_ns, start, side="left"))
    right = int(np.searchsorted(timeline.decision_times_ns, end, side="right"))
    values = np.asarray(timeline.scores[left:right], dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "score_end": np.nan,
            "score_min": np.nan,
            "score_max": np.nan,
            "score_change": np.nan,
            "q70_reconfirmations": 0.0,
            "q90_reconfirmations": 0.0,
            "longest_below_q50": 0.0,
        }
    return {
        "score_end": float(values[-1]),
        "score_min": float(np.min(values)),
        "score_max": float(np.max(values)),
        "score_change": float(values[-1] - values[0]),
        "q70_reconfirmations": float(max(0, int(np.sum(values >= 0.70)) - int(values[0] >= 0.70))),
        "q90_reconfirmations": float(max(0, int(np.sum(values >= 0.90)) - int(values[0] >= 0.90))),
        "longest_below_q50": float(_longest_true_run(values < 0.50)),
    }


def extract_extended_event_path(
    *,
    event: EventCandidate,
    fold_id: str,
    phase: str,
    scope: str,
    path: MinutePathData,
    timeline: ScoreTimeline,
    config: LongTailMultistageConfig,
) -> ExtendedEventPath | None:
    entry_ns = event.decision_time_ns + config.entry_delay_minutes * _ONE_MINUTE_NS
    start = int(np.searchsorted(path.timestamps_ns, entry_ns, side="left"))
    rows = config.path_horizon_hours * 60
    stop = start + rows
    if start >= len(path.timestamps_ns) or stop > len(path.timestamps_ns):
        return None
    expected = entry_ns + np.arange(rows, dtype=np.int64) * _ONE_MINUTE_NS
    actual = np.asarray(path.timestamps_ns[start:stop], dtype=np.int64)
    if len(actual) != rows or not np.array_equal(expected, actual):
        return None

    open_price = np.asarray(path.open[start:stop], dtype=float)
    high = np.asarray(path.high[start:stop], dtype=float)
    low = np.asarray(path.low[start:stop], dtype=float)
    close = np.asarray(path.close[start:stop], dtype=float)
    entry_price = float(open_price[0])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None
    high_ret = high / entry_price - 1.0
    low_ret = low / entry_price - 1.0
    close_ret = close / entry_price - 1.0
    running_mfe = np.maximum.accumulate(high_ret)
    running_mae = np.maximum.accumulate(-low_ret)
    points = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(actual, unit="ns"),
            "open_price": open_price,
            "high_price": high,
            "low_price": low,
            "close_price": close,
            "open_return": open_price / entry_price - 1.0,
            "high_return": high_ret,
            "low_return": low_ret,
            "close_return": close_ret,
            "running_mfe": running_mfe,
            "running_mae": running_mae,
        }
    )
    event_id = f"{fold_id}_{phase}_{scope}_{event.event_id}"
    summary: dict[str, object] = {
        "event_id": event_id,
        "fold_id": fold_id,
        "phase": phase,
        "scope": scope,
        "decision_time": pd.Timestamp(event.decision_time_ns, unit="ns"),
        "entry_time": pd.Timestamp(entry_ns, unit="ns"),
        "entry_price": entry_price,
        "event_score_percentile": float(event.score),
        "signal_quantile": float(event.signal_quantile),
    }
    for delay in (1, 3, 5):
        position = delay - config.entry_delay_minutes
        summary[f"entry_price_delay_{delay}m"] = float(open_price[position]) if 0 <= position < len(open_price) else np.nan
    for minute in (60, 180, 360, 1440, 2880, 7200):
        position = minute - 1
        summary[f"close_price_{minute}m"] = float(close[position])
        summary[f"ret_{minute}m"] = float(close_ret[position])
        summary[f"mfe_{minute}m"] = float(np.max(high_ret[:minute]))
        summary[f"mae_{minute}m"] = float(np.max(-low_ret[:minute]))
    for minute in (180, 360, 1440):
        summary[f"open_after_{minute}m"] = float(open_price[minute]) if minute < len(open_price) else np.nan
    summary["mfe_6h_to_24h_increment"] = float(np.max(high_ret[360:1440]) - np.max(high_ret[:360]))
    summary["mfe_24h_to_48h_increment"] = float(np.max(high_ret[1440:2880]) - np.max(high_ret[:1440]))
    summary["mfe_24h_to_120h_increment"] = float(np.max(high_ret[1440:7200]) - np.max(high_ret[:1440]))
    summary["max_mfe_48h"] = float(np.max(high_ret[:2880]))
    summary["max_mfe_120h"] = float(np.max(high_ret))
    return ExtendedEventPath(summary=summary, points=points)


def build_checkpoint_row(
    extraction: ExtendedEventPath,
    *,
    checkpoint_minutes: int,
    path: MinutePathData,
    timeline: ScoreTimeline,
    config: LongTailMultistageConfig,
) -> dict[str, object]:
    if checkpoint_minutes not in config.checkpoints_minutes:
        raise ValueError(f"unsupported checkpoint {checkpoint_minutes}")
    points = extraction.points.iloc[:checkpoint_minutes].copy()
    if len(points) != checkpoint_minutes:
        raise RuntimeError("checkpoint path is incomplete")
    summary = extraction.summary
    close_ret = points["close_return"].to_numpy(dtype=float)
    high_ret = points["high_return"].to_numpy(dtype=float)
    low_ret = points["low_return"].to_numpy(dtype=float)
    current_return = float(close_ret[-1])
    current_mfe = float(np.max(high_ret))
    current_mae = float(np.max(-low_ret))
    log_close = np.log1p(np.clip(close_ret, -0.999, None))
    slope, r2 = _slope_r2(log_close)
    last60_slope, last60_r2 = _slope_r2(log_close[-min(60, len(log_close)):])
    increments = np.diff(log_close)
    upside = float(np.sqrt(np.square(increments[increments > 0]).sum())) if np.any(increments > 0) else 0.0
    downside = float(np.sqrt(np.square(increments[increments < 0]).sum())) if np.any(increments < 0) else 0.0

    entry_time = pd.Timestamp(summary["entry_time"])
    entry_position = path.locate_exact(entry_time)
    if entry_position is None:
        raise RuntimeError("entry timestamp missing from minute path")
    entry_price = float(summary["entry_price"])
    current_price = entry_price * (1.0 + current_return)
    path_low_price = entry_price * (1.0 + float(np.min(low_ret)))
    prior_low_60 = float(path.prior_low_60[entry_position])
    prior_low_180 = float(path.prior_low_180[entry_position])

    def relative_distance(price: float, reference: float) -> float:
        return price / reference - 1.0 if np.isfinite(reference) and reference > 0 else np.nan

    first_half_position = max(0, checkpoint_minutes // 2 - 1)
    last30_position = max(0, checkpoint_minutes - 31)
    last60_position = max(0, checkpoint_minutes - 61)
    last180_position = max(0, checkpoint_minutes - 181)
    structural = {
        "current_return": current_return,
        "current_mfe": current_mfe,
        "current_mae": current_mae,
        "peak_giveback": float(current_mfe - current_return),
        "recovery_from_trough": float(current_return - np.min(close_ret)),
        "capture_of_mfe": _safe_div(current_return, current_mfe),
        "close_location_in_range": _safe_div(current_return + current_mae, current_mfe + current_mae),
        "underwater_fraction": float(np.mean(close_ret < 0)),
        "longest_underwater_minutes": float(_longest_true_run(close_ret < 0)),
        "last30_return": float(current_return - close_ret[last30_position]) if checkpoint_minutes > 30 else current_return,
        "last60_return": float(current_return - close_ret[last60_position]) if checkpoint_minutes > 60 else current_return,
        "last180_return": float(current_return - close_ret[last180_position]) if checkpoint_minutes > 180 else current_return,
        "first_half_return": float(close_ret[first_half_position]),
        "second_half_return": float(current_return - close_ret[first_half_position]),
        "path_acceleration": float(current_return - 2.0 * close_ret[first_half_position]),
        "log_close_slope": float(slope),
        "log_close_r2": float(r2),
        "last60_log_close_slope": float(last60_slope),
        "last60_log_close_r2": float(last60_r2),
        "realized_vol": float(np.sqrt(np.square(increments).sum())) if len(increments) else 0.0,
        "upside_semivol": upside,
        "downside_semivol": downside,
        "down_up_vol_ratio": _safe_div(downside, upside + 1e-12),
        "distance_to_prior_low_60": relative_distance(current_price, prior_low_60),
        "distance_to_prior_low_180": relative_distance(current_price, prior_low_180),
        "broke_prior_low_60": float(np.isfinite(prior_low_60) and path_low_price < prior_low_60),
        "broke_prior_low_180": float(np.isfinite(prior_low_180) and path_low_price < prior_low_180),
        "reclaimed_entry_after_drawdown": float(np.min(close_ret) < 0 and current_return > 0),
        "current_below_entry": float(current_return < 0),
        "minutes_since_mfe": float(checkpoint_minutes - 1 - int(np.argmax(high_ret))),
        "minutes_since_mae": float(checkpoint_minutes - 1 - int(np.argmin(low_ret))),
    }
    structural.update(_bar_structure(points, 15, "bar15"))
    structural.update(_bar_structure(points, 60, "bar60"))
    score = _score_percentile_at_minutes(
        EventCandidate(
            event_id=str(summary["event_id"]),
            decision_time_ns=int(pd.Timestamp(summary["decision_time"]).value),
            score=float(summary["event_score_percentile"]),
            signal_quantile=float(summary["signal_quantile"]),
        ),
        timeline,
        checkpoint_minutes,
    )

    ret24 = float(summary["ret_1440m"])
    ret48 = float(summary["ret_2880m"])
    ret120 = float(summary["ret_7200m"])
    max_mfe48 = float(summary["max_mfe_48h"])
    weak_now = bool(current_return <= 0 or structural["underwater_fraction"] >= 0.50)
    persistent_failure = bool(max_mfe48 < config.persistent_failure_max_mfe_48h and ret48 <= 0)
    recoverable = bool(
        weak_now
        and max_mfe48 >= config.recoverable_min_mfe_48h
        and (ret24 > 0 or ret48 > 0)
    )
    healthy = bool(not persistent_failure and (current_return > 0 or ret24 > 0 or ret48 > 0))
    continuation = bool(float(summary["mfe_6h_to_24h_increment"]) >= config.continuation_increment_6h_to_24h)
    longhold = bool(
        float(summary["mfe_24h_to_120h_increment"]) >= config.longhold_increment_24h_to_120h
        and ret120 > ret24
    )
    path_class = "persistent_failure" if persistent_failure else "recoverable_drawdown" if recoverable else "healthy_hold"
    row: dict[str, object] = {
        **summary,
        "checkpoint_minutes": int(checkpoint_minutes),
        "checkpoint_time": entry_time + pd.Timedelta(minutes=checkpoint_minutes - 1),
        "checkpoint_close_price": float(points["close_price"].iloc[-1]),
        "weak_now": weak_now,
        "path_class": path_class,
        "label_persistent_failure": int(persistent_failure),
        "label_recoverable_drawdown": int(recoverable),
        "label_healthy_hold": int(healthy),
        "label_post6_continuation": int(continuation),
        "label_post24_longhold": int(longhold),
        "net_6h_1x": float(summary["ret_360m"]) - config.base_round_trip_cost,
        "net_24h_1x": ret24 - config.base_round_trip_cost,
        "net_48h_1x": ret48 - config.base_round_trip_cost,
        "net_120h_1x": ret120 - config.base_round_trip_cost,
    }
    row.update({f"x_path__{key}": float(value) for key, value in structural.items()})
    row.update({f"x_score__{key}": float(value) for key, value in score.items()})
    row["x_score__entry_percentile"] = float(summary["event_score_percentile"])
    return row


def feature_sets(frame: pd.DataFrame) -> tuple[FeatureSet, ...]:
    path = tuple(sorted(column for column in frame.columns if column.startswith("x_path__")))
    score = tuple(sorted(column for column in frame.columns if column.startswith("x_score__")))
    mechanical_names = (
        "x_path__current_return",
        "x_path__current_mfe",
        "x_path__current_mae",
        "x_path__peak_giveback",
        "x_path__underwater_fraction",
        "x_path__recovery_from_trough",
        "x_path__distance_to_prior_low_60",
        "x_path__distance_to_prior_low_180",
        "x_path__last60_return",
        "x_path__bar15_lower_low_share",
        "x_path__bar15_higher_low_share",
    )
    mechanical = tuple(column for column in mechanical_names if column in frame.columns)
    return (
        FeatureSet("mechanical_logistic", mechanical),
        FeatureSet("path_structure_logistic", path),
        FeatureSet("path_structure_lightgbm", path),
        FeatureSet("path_plus_score_logistic", (*path, *score)),
        FeatureSet("path_plus_score_lightgbm", (*path, *score)),
    )


def task_frame(frame: pd.DataFrame, task: str) -> tuple[pd.DataFrame, np.ndarray]:
    work = frame.copy()
    if task == "persistent_failure":
        target = work["label_persistent_failure"].to_numpy(dtype=np.int8)
    elif task == "recoverable_drawdown":
        work = work.loc[work["weak_now"].astype(bool)].copy()
        target = work["label_recoverable_drawdown"].to_numpy(dtype=np.int8)
    elif task == "post6_continuation":
        work = work.loc[work["checkpoint_minutes"] == 360].copy()
        target = work["label_post6_continuation"].to_numpy(dtype=np.int8)
    elif task == "post24_longhold":
        work = work.loc[work["checkpoint_minutes"] == 1440].copy()
        target = work["label_post24_longhold"].to_numpy(dtype=np.int8)
    else:
        raise ValueError(f"unsupported task {task}")
    return work.reset_index(drop=True), target
