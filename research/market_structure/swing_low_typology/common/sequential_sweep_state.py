#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal sequential state construction after a respected-macro first sweep.

The module expands one already-confirmed first-sweep decision into a small,
predeclared checkpoint sequence.  Every checkpoint is a closed 1m bar and its
entry reference is the next 1m open.  State features use only the sweep bar and
bars closed by the checkpoint.  Future lifecycle outcomes are deliberately
ignored even when they are present on the source lifecycle row.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from src.research_common.progress import ProgressReporter
except Exception:  # pragma: no cover
    ProgressReporter = None  # type: ignore[assignment]

EPS = 1e-12
SEQUENTIAL_FEATURE_GROUP = "SEQ_causal_path_state"
DEFAULT_CHECKPOINT_OFFSETS: tuple[int, ...] = (0, 1, 3, 5, 10, 15)
DEFAULT_AMPLITUDE_TARGETS_PCT: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)
DEFAULT_AMPLITUDE_HORIZONS: tuple[int, ...] = (60, 180)


@dataclass(frozen=True)
class SequentialStateBuildResult:
    frame: pd.DataFrame
    dictionary: pd.DataFrame
    group_membership: pd.DataFrame
    diagnostics: pd.DataFrame


def _numeric(frame: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in frame.columns:
        return np.full(len(frame), float(default), dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float, copy=False)


def _bar_delta(index: pd.DatetimeIndex) -> pd.Timedelta:
    if len(index) < 2:
        return pd.Timedelta(minutes=1)
    diffs = pd.Series(index[1:] - index[:-1])
    mode = diffs.mode()
    return pd.Timedelta(mode.iloc[0] if not mode.empty else diffs.median())


def _safe_pct(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= EPS:
        return np.nan
    return (float(numerator) / float(denominator) - 1.0) * 100.0


def _safe_bp(numerator: float, denominator: float) -> float:
    value = _safe_pct(numerator, denominator)
    return value * 100.0 if np.isfinite(value) else np.nan


def _feature_row(name: str, description: str) -> dict[str, object]:
    return {
        "feature": name,
        "feature_group": SEQUENTIAL_FEATURE_GROUP,
        "description": description,
        "source": "closed 1m path from first-sweep bar through current checkpoint",
        "available_rule": "checkpoint closed bar or older; no future lifecycle state",
    }


def build_sequential_checkpoint_decisions(
    bars: pd.DataFrame,
    sweep_decisions: pd.DataFrame,
    *,
    checkpoint_offsets: Sequence[int] = DEFAULT_CHECKPOINT_OFFSETS,
    accept_below_bars: int = 3,
    accept_depth_bp: float = 75.0,
    prior_target_move_pct: float = 1.0,
    show_progress: bool = False,
) -> SequentialStateBuildResult:
    """Expand causal first sweeps into closed-bar sequential decision states."""

    if bars.empty or sweep_decisions.empty:
        return SequentialStateBuildResult(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    offsets = tuple(sorted(set(int(value) for value in checkpoint_offsets)))
    if not offsets or offsets[0] != 0 or any(value < 0 for value in offsets):
        raise ValueError("checkpoint_offsets must be unique non-negative integers including 0")
    if int(accept_below_bars) < 1 or float(accept_depth_bp) <= 0.0:
        raise ValueError("accept-below thresholds must be positive")

    data = bars.sort_index()
    index = pd.DatetimeIndex(data.index)
    if index.has_duplicates or not index.is_monotonic_increasing:
        raise RuntimeError("bars index must be unique and increasing")
    delta_time = _bar_delta(index)
    open_ = _numeric(data, "open")
    high = _numeric(data, "high")
    low = _numeric(data, "low")
    close = _numeric(data, "close")

    source = sweep_decisions.copy()
    if "decision_path" in source.columns:
        source = source[source["decision_path"].astype(str).eq("sweep")]
    source = source.sort_values(["extreme_pos", "event_id"], kind="mergesort").reset_index(drop=True)
    required = {"event_id", "extreme_pos", "level_price", "level_id", "lifecycle_id"}
    missing = sorted(required.difference(source.columns))
    if missing:
        raise RuntimeError(f"sequential state builder missing sweep fields: {missing}")

    reporter = (
        ProgressReporter("[states] sequential sweep checkpoints", total=max(len(source), 1), every=max(1, len(source) // 100))
        if ProgressReporter is not None and show_progress
        else None
    )
    rows: list[dict[str, object]] = []
    skipped_outside = 0
    target = float(prior_target_move_pct) / 100.0

    passthrough = (
        "level_id", "level_price", "level_available_time", "level_expiry_time",
        "level_strength", "level_respect_count", "level_timeframe_count",
        "level_max_timeframe_min", "level_median_reaction_bp",
        "level_respect_span_bars", "level_source_timeframes",
        "fse_level_strength", "fse_level_respect_count", "fse_level_timeframe_count",
        "fse_level_max_timeframe_min", "fse_level_median_reaction_bp",
        "fse_level_respect_span_bars", "fse_level_age_bars_at_sweep",
    )

    for ordinal, event in enumerate(source.itertuples(index=False), start=1):
        event_map = event._asdict()
        sweep_pos = int(event_map["extreme_pos"])
        level_price = float(event_map["level_price"])
        if sweep_pos < 0 or sweep_pos >= len(index):
            skipped_outside += 1
            if reporter is not None:
                reporter.update(ordinal)
            continue
        sweep_close = float(close[sweep_pos])
        sweep_low = float(low[sweep_pos])
        sweep_range = max(float(high[sweep_pos] - low[sweep_pos]), EPS)
        initial_entry_pos = sweep_pos + 1

        for offset in offsets:
            checkpoint_pos = sweep_pos + int(offset)
            if checkpoint_pos >= len(index):
                skipped_outside += 1
                continue
            path_slice = slice(sweep_pos, checkpoint_pos + 1)
            path_close = close[path_slice]
            path_low = low[path_slice]
            current_close = float(close[checkpoint_pos])
            current_low = float(low[checkpoint_pos])
            current_range = max(float(high[checkpoint_pos] - low[checkpoint_pos]), 0.0)
            below = np.asarray(path_close < level_price, dtype=bool)
            above = ~below
            consecutive_below = 0
            for flag in below[::-1]:
                if not bool(flag):
                    break
                consecutive_below += 1
            reclaimed_positions = np.flatnonzero(above)
            first_reclaim_local = int(reclaimed_positions[0]) if reclaimed_positions.size else -1
            reclaimed_ever = first_reclaim_local >= 0
            reclaimed_now = bool(current_close >= level_price)
            max_penetration_bp = max((level_price - float(np.nanmin(path_low))) / max(level_price, EPS) * 10_000.0, 0.0)
            hard_invalidated = bool(
                consecutive_below >= int(accept_below_bars)
                or max_penetration_bp >= float(accept_depth_bp)
            )

            initial_entry_known = initial_entry_pos <= checkpoint_pos and initial_entry_pos < len(index)
            initial_entry_price = float(open_[initial_entry_pos]) if initial_entry_known else np.nan
            prior_tp_reached = False
            current_from_initial_entry = np.nan
            prior_runup_initial = np.nan
            prior_drawdown_initial = np.nan
            if int(offset) >= 1 and np.isfinite(initial_entry_price) and initial_entry_price > EPS:
                entry_to_checkpoint = close[initial_entry_pos : checkpoint_pos + 1]
                if entry_to_checkpoint.size:
                    prior_tp_reached = bool(np.nanmax(entry_to_checkpoint) >= initial_entry_price * (1.0 + target))
                    current_from_initial_entry = _safe_pct(current_close, initial_entry_price)
                    prior_runup_initial = max(_safe_pct(float(np.nanmax(entry_to_checkpoint)), initial_entry_price), 0.0)
                    prior_drawdown_initial = max(-_safe_pct(float(np.nanmin(entry_to_checkpoint)), initial_entry_price), 0.0)

            if hard_invalidated:
                state_status = "accepted_below"
            elif reclaimed_now:
                state_status = "reclaimed_above"
            elif reclaimed_ever:
                state_status = "reclaim_lost"
            else:
                state_status = "pending_below"

            row: dict[str, object] = {
                "event_id": f"{event_map['event_id']}_T{int(offset):03d}",
                "origin_event_id": str(event_map["event_id"]),
                "lifecycle_id": str(event_map["lifecycle_id"]),
                "causal_region_id": str(event_map.get("causal_region_id", event_map["lifecycle_id"])),
                "decision_path": "sequential_state",
                "origin_sweep_pos": sweep_pos,
                "origin_sweep_time": index[sweep_pos],
                "checkpoint_offset": int(offset),
                "extreme_pos": checkpoint_pos,
                "extreme_time": index[checkpoint_pos],
                "feature_available_time": index[checkpoint_pos] + delta_time,
                "state_status": state_status,
                "prior_tp_reached": bool(prior_tp_reached),
                "hard_invalidated": bool(hard_invalidated),
                "add_on_eligible": bool(int(offset) > 0 and not prior_tp_reached and not hard_invalidated),
                "initial_decision": bool(int(offset) == 0),
                "initial_entry_time": index[initial_entry_pos] if initial_entry_known else pd.NaT,
                "initial_entry_price": initial_entry_price,
                "seq_checkpoint_offset": float(offset),
                "seq_checkpoint_offset_log": float(np.log1p(offset)),
                "seq_close_vs_level_bp": _safe_bp(current_close, level_price),
                "seq_current_low_vs_level_bp": _safe_bp(current_low, level_price),
                "seq_max_penetration_bp": float(max_penetration_bp),
                "seq_close_recovery_from_sweep_close_bp": _safe_bp(current_close, sweep_close),
                "seq_close_recovery_from_min_close_bp": _safe_bp(current_close, float(np.nanmin(path_close))),
                "seq_close_recovery_from_sweep_low_bp": _safe_bp(current_close, sweep_low),
                "seq_return_from_sweep_close_pct": _safe_pct(current_close, sweep_close),
                "seq_max_runup_close_from_sweep_pct": max(_safe_pct(float(np.nanmax(path_close)), sweep_close), 0.0),
                "seq_max_drawdown_close_from_sweep_pct": max(-_safe_pct(float(np.nanmin(path_close)), sweep_close), 0.0),
                "seq_close_above_level_share": float(np.mean(above)),
                "seq_close_below_level_share": float(np.mean(below)),
                "seq_consecutive_below_closes": float(consecutive_below),
                "seq_reclaimed_now": float(reclaimed_now),
                "seq_reclaimed_ever": float(reclaimed_ever),
                "seq_bars_since_first_reclaim": float(offset - first_reclaim_local) if reclaimed_ever else -1.0,
                "seq_current_range_vs_sweep": float(current_range / sweep_range),
                "seq_current_return_from_initial_entry_pct": current_from_initial_entry,
                "seq_prior_max_runup_from_initial_entry_pct": prior_runup_initial,
                "seq_prior_max_drawdown_from_initial_entry_pct": prior_drawdown_initial,
            }
            for column in passthrough:
                if column in event_map:
                    row[column] = event_map[column]
            rows.append(row)
        if reporter is not None:
            reporter.update(ordinal)
    if reporter is not None:
        reporter.close()

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.sort_values(["origin_sweep_pos", "checkpoint_offset", "event_id"], kind="mergesort").reset_index(drop=True)
        if frame["event_id"].duplicated().any():
            raise RuntimeError("duplicate sequential checkpoint event_id")
        expected = frame["extreme_time"] + delta_time
        if not pd.to_datetime(frame["feature_available_time"]).equals(pd.to_datetime(expected)):
            raise RuntimeError("sequential checkpoint available_time mismatch")

    feature_descriptions = {
        "seq_checkpoint_offset": "minutes/bars elapsed since the first-sweep bar",
        "seq_checkpoint_offset_log": "log1p elapsed checkpoint offset",
        "seq_close_vs_level_bp": "current close relative to respected liquidity level",
        "seq_current_low_vs_level_bp": "current closed-bar low relative to liquidity level",
        "seq_max_penetration_bp": "maximum observed penetration through level so far",
        "seq_close_recovery_from_sweep_close_bp": "current close recovery versus sweep close",
        "seq_close_recovery_from_min_close_bp": "current close recovery versus lowest observed close",
        "seq_close_recovery_from_sweep_low_bp": "current close recovery versus sweep low",
        "seq_return_from_sweep_close_pct": "current close return from sweep close",
        "seq_max_runup_close_from_sweep_pct": "maximum close run-up from sweep close so far",
        "seq_max_drawdown_close_from_sweep_pct": "maximum close drawdown from sweep close so far",
        "seq_close_above_level_share": "share of closed path bars at or above the level",
        "seq_close_below_level_share": "share of closed path bars below the level",
        "seq_consecutive_below_closes": "consecutive closes below level at checkpoint",
        "seq_reclaimed_now": "current close is at or above level",
        "seq_reclaimed_ever": "a reclaim close has occurred by checkpoint",
        "seq_bars_since_first_reclaim": "elapsed bars since first observed reclaim, -1 if none",
        "seq_current_range_vs_sweep": "current closed-bar range relative to sweep-bar range",
        "seq_current_return_from_initial_entry_pct": "current close return from original next-open entry; unavailable at t0",
        "seq_prior_max_runup_from_initial_entry_pct": "past close run-up from initial entry; unavailable at t0",
        "seq_prior_max_drawdown_from_initial_entry_pct": "past close drawdown from initial entry; unavailable at t0",
    }
    dictionary = pd.DataFrame([_feature_row(name, description) for name, description in feature_descriptions.items()])
    membership = pd.DataFrame([
        {"feature_group": SEQUENTIAL_FEATURE_GROUP, "feature": name, "feature_count": len(feature_descriptions)}
        for name in feature_descriptions
    ])
    diagnostics = pd.DataFrame([
        {"metric": "source_sweep_events", "value": int(len(source))},
        {"metric": "checkpoint_rows", "value": int(len(frame))},
        {"metric": "checkpoint_offsets", "value": "|".join(str(value) for value in offsets)},
        {"metric": "skipped_outside_bar_range", "value": int(skipped_outside)},
        {"metric": "add_on_eligible_rows", "value": int(frame.get("add_on_eligible", pd.Series(dtype=bool)).astype(bool).sum()) if not frame.empty else 0},
        {"metric": "hard_invalidated_rows", "value": int(frame.get("hard_invalidated", pd.Series(dtype=bool)).astype(bool).sum()) if not frame.empty else 0},
        {"metric": "prior_tp_reached_rows", "value": int(frame.get("prior_tp_reached", pd.Series(dtype=bool)).astype(bool).sum()) if not frame.empty else 0},
    ])
    return SequentialStateBuildResult(frame=frame, dictionary=dictionary, group_membership=membership, diagnostics=diagnostics)


def _target_token(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def build_amplitude_ladder_close_labels(
    bars: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    targets_pct: Sequence[float] = DEFAULT_AMPLITUDE_TARGETS_PCT,
    horizons: Sequence[int] = DEFAULT_AMPLITUDE_HORIZONS,
    vectorized_chunk_size: int = 50_000,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Descriptive next-open/future-close target ladder; no model selection."""

    if decisions.empty:
        return pd.DataFrame()
    targets = tuple(sorted(set(float(value) for value in targets_pct)))
    horizon_values = tuple(sorted(set(int(value) for value in horizons)))
    if not targets or any(value <= 0.0 for value in targets):
        raise ValueError("amplitude targets must be positive")
    if not horizon_values or any(value < 1 for value in horizon_values):
        raise ValueError("amplitude horizons must be positive")
    max_horizon = max(horizon_values)
    if int(vectorized_chunk_size) < 1:
        raise ValueError("vectorized_chunk_size must be positive")

    index = pd.DatetimeIndex(bars.index)
    open_values = _numeric(bars, "open")
    close_values = _numeric(bars, "close")
    windows = np.lib.stride_tricks.sliding_window_view(close_values, max_horizon)
    reporter = (
        ProgressReporter("[labels] amplitude ladder", total=len(decisions), every=max(1, len(decisions) // 100))
        if ProgressReporter is not None and show_progress
        else None
    )
    parts: list[pd.DataFrame] = []
    processed = 0
    for start in range(0, len(decisions), int(vectorized_chunk_size)):
        source = decisions.iloc[start : start + int(vectorized_chunk_size)]
        position = pd.to_numeric(source["extreme_pos"], errors="coerce").to_numpy(dtype=np.int64)
        entry_pos = position + 1
        valid = (entry_pos >= 0) & (entry_pos < len(windows))
        chosen = np.flatnonzero(valid)
        if chosen.size:
            chunk = source.iloc[chosen][["event_id"]].reset_index(drop=True)
            ep = entry_pos[chosen]
            entry = open_values[ep]
            finite = np.isfinite(entry) & (entry > EPS)
            chunk = chunk.iloc[np.flatnonzero(finite)].reset_index(drop=True)
            ep = ep[finite]
            entry = entry[finite]
            if len(chunk):
                path = windows[ep]
                output = chunk.copy()
                output["amp_entry_time"] = index[ep]
                output["amp_entry_price"] = entry
                output["amp_label_end_time"] = index[ep + max_horizon - 1]
                for horizon in horizon_values:
                    max_close = np.nanmax(path[:, :horizon], axis=1)
                    output[f"amp_mfe_h{horizon}_pct"] = np.maximum(max_close / entry - 1.0, 0.0) * 100.0
                    for target in targets:
                        token = _target_token(target)
                        output[f"amp_tp_{token}_h{horizon}"] = max_close >= entry * (1.0 + target / 100.0)
                parts.append(output)
        processed += len(source)
        if reporter is not None:
            reporter.update(processed)
    if reporter is not None:
        reporter.close()
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
