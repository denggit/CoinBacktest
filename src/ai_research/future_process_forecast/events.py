#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Future-only event atlas and start labels for R03.3.

The event detector is used only to build supervised labels.  It may inspect the
future path, but none of its outputs are inserted into the feature matrix.
Current model rows are positive only when a new event starts strictly after the
decision timestamp and inside the requested forecast horizon.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.swing_baseline.dataset import SwingYearShard, load_year_shard
from src.research_common.progress import ProgressReporter

from .config import FutureProcessForecastConfig, PROCESS_TYPES


CACHE_SCHEMA_VERSION = 1
_EPS = 1e-12


def event_label_columns(config: FutureProcessForecastConfig) -> tuple[str, ...]:
    columns: list[str] = []
    for process in PROCESS_TYPES:
        for horizon in config.forecast_horizons_hours:
            columns.append(f"{process}_start_h{horizon}")
    for process in PROCESS_TYPES[:-1]:
        columns.extend(
            [
                f"{process}_next_lead_hours",
                f"{process}_ongoing",
                f"{process}_progress",
                f"{process}_next_event_id",
            ]
        )
    return tuple(columns)


def _minute_path_frame(shard: SwingYearShard) -> pd.DataFrame:
    index = pd.to_datetime(np.asarray(shard.minute_times_ns, dtype=np.int64))
    frame = pd.DataFrame(
        np.asarray(shard.minute_ohlc, dtype=np.float64),
        index=pd.DatetimeIndex(index),
        columns=["open", "high", "low", "close"],
    )
    frame = frame.loc[~frame.index.duplicated(keep="last")].sort_index(kind="stable")
    return frame


def build_available_15m_path(shard: SwingYearShard) -> pd.DataFrame:
    minute = _minute_path_frame(shard)
    bars = minute.resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    bars = bars.dropna(subset=["open", "high", "low", "close"])
    bars.index = bars.index + pd.Timedelta(minutes=15)
    bars.index.name = "available_time"
    return bars


def _true_range(path: pd.DataFrame) -> pd.Series:
    previous = path["close"].shift(1)
    return pd.concat(
        [
            path["high"] - path["low"],
            (path["high"] - previous).abs(),
            (path["low"] - previous).abs(),
        ],
        axis=1,
    ).max(axis=1)


def _atr_pct(path: pd.DataFrame, config: FutureProcessForecastConfig) -> np.ndarray:
    bars = max(2, int(config.event_atr_hours * 60 / config.event_scan_timeframe_minutes))
    slow = max(bars, int(config.event_atr_slow_days * 24 * 60 / config.event_scan_timeframe_minutes))
    raw = _true_range(path).rolling(bars, min_periods=bars).mean() / path["close"]
    slow_median = raw.rolling(slow, min_periods=max(bars * 4, slow // 4)).median()
    # Do not allow an isolated volatility spike to inflate the target without bound.
    robust = pd.concat([raw, slow_median * 2.5], axis=1).min(axis=1)
    return robust.to_numpy(dtype=float)


def _future_views(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) <= window:
        return np.empty((0, window), dtype=float)
    return np.lib.stride_tricks.sliding_window_view(values[1:], window_shape=window)


def _prior_directional_progress(close: np.ndarray, prior_bars: int, *, direction: int) -> np.ndarray:
    out = np.full(len(close), np.nan, dtype=float)
    if prior_bars <= 0 or len(close) <= prior_bars:
        return out
    previous = close[:-prior_bars]
    current = close[prior_bars:]
    if direction > 0:
        out[prior_bars:] = current / previous - 1.0
    else:
        out[prior_bars:] = 1.0 - current / previous
    return out


def _select_non_overlapping(
    candidate: np.ndarray,
    hit_bars: np.ndarray,
    *,
    refractory_bars: int,
) -> np.ndarray:
    selected: list[int] = []
    index = 0
    n = len(candidate)
    while index < n:
        if not bool(candidate[index]):
            index += 1
            continue
        selected.append(index)
        event_bars = int(hit_bars[index]) if np.isfinite(hit_bars[index]) else 1
        index += max(refractory_bars, event_bars, 1)
    return np.asarray(selected, dtype=np.int64)


def _directional_events(
    path: pd.DataFrame,
    atr_pct: np.ndarray,
    config: FutureProcessForecastConfig,
    *,
    direction: int,
) -> pd.DataFrame:
    spec = config.directional
    bars_per_hour = int(60 / config.event_scan_timeframe_minutes)
    horizon = int(spec.horizon_hours * bars_per_hour)
    initial = int(spec.initial_window_hours * bars_per_hour)
    prior = int(spec.prior_window_hours * bars_per_hour)
    refractory = int(spec.refractory_hours * bars_per_hour)
    close = path["close"].to_numpy(dtype=float)
    high = path["high"].to_numpy(dtype=float)
    low = path["low"].to_numpy(dtype=float)
    high_view = _future_views(high, horizon)
    low_view = _future_views(low, horizon)
    close_view = _future_views(close, horizon)
    rows = len(high_view)
    if rows == 0:
        return pd.DataFrame()
    candidate = np.zeros(rows, dtype=bool)
    hit_bars = np.full(rows, np.nan, dtype=float)
    target_move = np.full(rows, np.nan, dtype=float)
    adverse_move = np.full(rows, np.nan, dtype=float)
    initial_move = np.full(rows, np.nan, dtype=float)
    close_at_hit_move = np.full(rows, np.nan, dtype=float)
    prior_progress = _prior_directional_progress(close, prior, direction=direction)[:rows]

    chunk = 2048
    for left in range(0, rows, chunk):
        right = min(rows, left + chunk)
        entry = close[left:right]
        atr = atr_pct[left:right]
        target = np.maximum(spec.target_floor, atr * spec.target_atr_multiple)
        max_adverse = np.maximum(spec.max_adverse_floor, target * spec.max_adverse_target_fraction)
        hw = high_view[left:right]
        lw = low_view[left:right]
        cw = close_view[left:right]
        if direction > 0:
            hit = hw >= entry[:, None] * (1.0 + target[:, None])
            init = np.max(hw[:, :initial], axis=1) / entry - 1.0
        else:
            hit = lw <= entry[:, None] * (1.0 - target[:, None])
            init = 1.0 - np.min(lw[:, :initial], axis=1) / entry
        has_hit = hit.any(axis=1)
        first = np.argmax(hit, axis=1)
        prefix_low = np.minimum.accumulate(lw, axis=1)
        prefix_high = np.maximum.accumulate(hw, axis=1)
        gathered_low = prefix_low[np.arange(right - left), first]
        gathered_high = prefix_high[np.arange(right - left), first]
        gathered_close = cw[np.arange(right - left), first]
        if direction > 0:
            adverse = 1.0 - gathered_low / entry
            close_move = gathered_close / entry - 1.0
        else:
            adverse = gathered_high / entry - 1.0
            close_move = 1.0 - gathered_close / entry
        prior_chunk = prior_progress[left:right]
        valid = (
            has_hit
            & np.isfinite(atr)
            & np.isfinite(prior_chunk)
            & (adverse <= max_adverse)
            & (init >= np.maximum(spec.initial_move_floor, target * spec.initial_target_fraction))
            & (prior_chunk <= target * spec.prior_progress_cap)
            & (close_move >= target * spec.terminal_capture_fraction)
        )
        candidate[left:right] = valid
        hit_bars[left:right] = np.where(has_hit, first + 1, np.nan)
        target_move[left:right] = target
        adverse_move[left:right] = adverse
        initial_move[left:right] = init
        close_at_hit_move[left:right] = close_move

    selected = _select_non_overlapping(candidate, hit_bars, refractory_bars=refractory)
    event_rows: list[dict[str, object]] = []
    process = "up_expansion" if direction > 0 else "down_expansion"
    for event_id, pos in enumerate(selected, start=1):
        end_pos = min(len(path) - 1, int(pos + hit_bars[pos]))
        start_price = float(close[pos])
        future_72 = min(len(path), pos + 72 * bars_per_hour + 1)
        future_120 = min(len(path), pos + 120 * bars_per_hour + 1)
        if direction > 0:
            mfe72 = float(np.max(high[pos + 1 : future_72]) / start_price - 1.0) if future_72 > pos + 1 else np.nan
            mfe120 = float(np.max(high[pos + 1 : future_120]) / start_price - 1.0) if future_120 > pos + 1 else np.nan
        else:
            mfe72 = float(1.0 - np.min(low[pos + 1 : future_72]) / start_price) if future_72 > pos + 1 else np.nan
            mfe120 = float(1.0 - np.min(low[pos + 1 : future_120]) / start_price) if future_120 > pos + 1 else np.nan
        event_rows.append(
            {
                "process": process,
                "event_id": event_id,
                "start_pos": int(pos),
                "end_pos": int(end_pos),
                "start_time": path.index[pos],
                "end_time": path.index[end_pos],
                "start_price": start_price,
                "target_move": float(target_move[pos]),
                "adverse_before_target": float(adverse_move[pos]),
                "initial_move_3h": float(initial_move[pos]),
                "close_move_at_target": float(close_at_hit_move[pos]),
                "hours_to_target": float(hit_bars[pos] / bars_per_hour),
                "mfe_72h": mfe72,
                "mfe_120h": mfe120,
                "hit_7pct_72h": bool(np.isfinite(mfe72) and mfe72 >= 0.07),
                "hit_10pct_120h": bool(np.isfinite(mfe120) and mfe120 >= 0.10),
            }
        )
    return pd.DataFrame(event_rows)


def _range_reversal_counts(close_windows: np.ndarray) -> np.ndarray:
    returns = np.diff(close_windows, axis=1)
    signs = np.sign(returns)
    if signs.shape[1] < 2:
        return np.zeros(len(close_windows), dtype=int)
    # Zero changes do not create artificial reversals.
    left = signs[:, :-1]
    right = signs[:, 1:]
    return np.sum((left * right) < 0, axis=1)


def _range_events(
    path: pd.DataFrame,
    atr_pct: np.ndarray,
    config: FutureProcessForecastConfig,
) -> pd.DataFrame:
    spec = config.volatile_range
    bars_per_hour = int(60 / config.event_scan_timeframe_minutes)
    horizon = int(spec.horizon_hours * bars_per_hour)
    initial = int(spec.initial_window_hours * bars_per_hour)
    prior = int(spec.prior_window_hours * bars_per_hour)
    refractory = int(spec.refractory_hours * bars_per_hour)
    close = path["close"].to_numpy(dtype=float)
    high = path["high"].to_numpy(dtype=float)
    low = path["low"].to_numpy(dtype=float)
    high_view = _future_views(high, horizon)
    low_view = _future_views(low, horizon)
    close_view = _future_views(close, horizon)
    rows = len(high_view)
    if rows == 0:
        return pd.DataFrame()
    prior_high = pd.Series(high).shift(1).rolling(prior, min_periods=prior).max().to_numpy()[:rows]
    prior_low = pd.Series(low).shift(1).rolling(prior, min_periods=prior).min().to_numpy()[:rows]
    entry = close[:rows]
    atr = atr_pct[:rows]
    up = np.max(high_view, axis=1) / entry - 1.0
    down = 1.0 - np.min(low_view, axis=1) / entry
    total = up + down
    initial_total = np.max(high_view[:, :initial], axis=1) / entry - np.min(low_view[:, :initial], axis=1) / entry
    terminal = np.abs(close_view[:, -1] / entry - 1.0)
    prior_range = (prior_high - prior_low) / entry
    reversals = _range_reversal_counts(close_view)
    side_threshold = np.maximum(spec.side_move_floor, atr * spec.side_atr_multiple)
    total_threshold = np.maximum(spec.total_range_floor, atr * spec.total_atr_multiple)
    candidate = (
        np.isfinite(atr)
        & np.isfinite(prior_range)
        & (up >= side_threshold)
        & (down >= side_threshold)
        & (total >= total_threshold)
        & (terminal <= total * spec.terminal_share_cap)
        & (reversals >= spec.min_reversal_count)
        & (initial_total >= total * spec.initial_range_fraction)
        & (prior_range <= total * spec.prior_range_fraction_cap)
    )
    duration = np.full(rows, horizon, dtype=float)
    selected = _select_non_overlapping(candidate, duration, refractory_bars=refractory)
    event_rows: list[dict[str, object]] = []
    for event_id, pos in enumerate(selected, start=1):
        end_pos = min(len(path) - 1, pos + horizon)
        event_rows.append(
            {
                "process": "volatile_range",
                "event_id": event_id,
                "start_pos": int(pos),
                "end_pos": int(end_pos),
                "start_time": path.index[pos],
                "end_time": path.index[end_pos],
                "start_price": float(close[pos]),
                "target_move": float(total[pos]),
                "up_excursion": float(up[pos]),
                "down_excursion": float(down[pos]),
                "terminal_move": float(terminal[pos]),
                "reversal_count": int(reversals[pos]),
                "hours_to_target": float(spec.horizon_hours),
                "mfe_72h": np.nan,
                "mfe_120h": np.nan,
                "hit_7pct_72h": False,
                "hit_10pct_120h": False,
            }
        )
    return pd.DataFrame(event_rows)


def discover_events(path: pd.DataFrame, config: FutureProcessForecastConfig) -> pd.DataFrame:
    atr = _atr_pct(path, config)
    parts = [
        _directional_events(path, atr, config, direction=1),
        _directional_events(path, atr, config, direction=-1),
        _range_events(path, atr, config),
    ]
    frame = pd.concat([part for part in parts if not part.empty], ignore_index=True) if any(not p.empty for p in parts) else pd.DataFrame()
    if frame.empty:
        return frame
    frame = frame.sort_values(["start_time", "process"], kind="stable").reset_index(drop=True)
    frame["event_uid"] = [f"{row.process}:{pd.Timestamp(row.start_time).isoformat()}" for row in frame.itertuples(index=False)]
    frame["start_year"] = pd.to_datetime(frame["start_time"]).dt.year
    return frame


def _nearest_event_fields(
    decision_times: pd.DatetimeIndex,
    process_events: pd.DataFrame,
    path: pd.DataFrame,
    *,
    process: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(decision_times)
    lead = np.full(n, np.nan, dtype=np.float32)
    ongoing = np.zeros(n, dtype=np.float32)
    progress = np.zeros(n, dtype=np.float32)
    next_id = np.full(n, -1.0, dtype=np.float32)
    if process_events.empty:
        return lead, ongoing, progress, next_id
    events = process_events.sort_values("start_time", kind="stable").reset_index(drop=True)
    starts = pd.to_datetime(events["start_time"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    ends = pd.to_datetime(events["end_time"]).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    times = decision_times.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    future_idx = np.searchsorted(starts, times, side="right")
    valid_future = future_idx < len(starts)
    lead[valid_future] = ((starts[future_idx[valid_future]] - times[valid_future]) / 3.6e12).astype(np.float32)
    next_id[valid_future] = future_idx[valid_future].astype(np.float32)
    current_idx = np.searchsorted(starts, times, side="right") - 1
    valid_current = (current_idx >= 0)
    valid_current &= np.where(valid_current, times <= ends[np.clip(current_idx, 0, len(ends) - 1)], False)
    ongoing[valid_current] = 1.0
    if np.any(valid_current):
        positions = np.flatnonzero(valid_current)
        event_positions = current_idx[positions]
        if process == "volatile_range":
            duration = np.maximum(ends[event_positions] - starts[event_positions], 1)
            progress[positions] = np.clip((times[positions] - starts[event_positions]) / duration, 0.0, 1.0)
        else:
            current_close = path["close"].reindex(decision_times, method="ffill").to_numpy(dtype=float)[positions]
            start_price = events["start_price"].to_numpy(dtype=float)[event_positions]
            target = events["target_move"].to_numpy(dtype=float)[event_positions]
            if process == "up_expansion":
                move = current_close / start_price - 1.0
            else:
                move = 1.0 - current_close / start_price
            progress[positions] = np.clip(move / np.maximum(target, _EPS), 0.0, 1.0).astype(np.float32)
    return lead, ongoing, progress, next_id


def _future_range(path: pd.DataFrame, decision_times: pd.DatetimeIndex, horizon_hours: int) -> np.ndarray:
    bars = int(horizon_hours * 4)
    high = path["high"].to_numpy(dtype=float)
    low = path["low"].to_numpy(dtype=float)
    close = path["close"].to_numpy(dtype=float)
    high_view = _future_views(high, bars)
    low_view = _future_views(low, bars)
    values = np.full(len(path), np.nan, dtype=float)
    rows = len(high_view)
    if rows:
        values[:rows] = np.max(high_view, axis=1) / close[:rows] - np.min(low_view, axis=1) / close[:rows]
    series = pd.Series(values, index=path.index)
    return series.reindex(decision_times).to_numpy(dtype=float)


def build_event_label_frame(
    shard: SwingYearShard,
    events: pd.DataFrame,
    config: FutureProcessForecastConfig,
) -> pd.DataFrame:
    decision_times = pd.DatetimeIndex(pd.to_datetime(np.asarray(shard.decision_times_ns, dtype=np.int64)))
    path = build_available_15m_path(shard)
    frame = pd.DataFrame(index=decision_times)
    nearest: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for process in PROCESS_TYPES[:-1]:
        subset = events.loc[events["process"] == process].copy() if not events.empty else pd.DataFrame()
        nearest[process] = _nearest_event_fields(decision_times, subset, path, process=process)
        lead, ongoing, progress, next_id = nearest[process]
        frame[f"{process}_next_lead_hours"] = lead
        frame[f"{process}_ongoing"] = ongoing
        frame[f"{process}_progress"] = progress
        frame[f"{process}_next_event_id"] = next_id
        for horizon in config.forecast_horizons_hours:
            frame[f"{process}_start_h{horizon}"] = ((lead > 0) & (lead <= horizon)).astype(np.float32)

    atr_series = pd.Series(_atr_pct(path, config), index=path.index).reindex(decision_times).to_numpy(dtype=float)
    any_lead = np.column_stack([nearest[p][0] for p in PROCESS_TYPES[:-1]])
    for horizon in config.forecast_horizons_hours:
        scale = np.sqrt(horizon / 24.0)
        threshold = np.maximum(
            config.low_opportunity_range_floor * scale,
            atr_series * config.low_opportunity_atr_multiple * scale,
        )
        future_range = _future_range(path, decision_times, horizon)
        event_soon = np.any((any_lead > 0) & (any_lead <= horizon), axis=1)
        frame[f"low_opportunity_start_h{horizon}"] = (
            np.isfinite(future_range) & np.isfinite(threshold) & (~event_soon) & (future_range <= threshold)
        ).astype(np.float32)
    return frame[list(event_label_columns(config))]


def event_cache_path(config: FutureProcessForecastConfig, year: int) -> Path:
    return config.event_cache_path / f"labels_{year}"


def build_event_year_cache(
    base_path: Path,
    config: FutureProcessForecastConfig,
    *,
    force_rebuild: bool = False,
) -> Path:
    shard = load_year_shard(base_path)
    year = pd.to_datetime(np.asarray(shard.decision_times_ns[:1], dtype=np.int64))[0].year
    target = event_cache_path(config, year)
    manifest_path = target / "manifest.json"
    if manifest_path.exists() and not force_rebuild:
        return target
    path = build_available_15m_path(shard)
    events = discover_events(path, config)
    labels = build_event_label_frame(shard, events, config)
    temp = target.with_name(target.name + ".part")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=True)
    np.save(temp / "decision_times_ns.npy", np.asarray(shard.decision_times_ns, dtype=np.int64), allow_pickle=False)
    np.save(temp / "labels.npy", labels.to_numpy(dtype=np.float32, copy=True), allow_pickle=False)
    events.to_csv(temp / "events.csv", index=False, encoding="utf-8-sig")
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "year": int(year),
        "rows": int(len(labels)),
        "label_columns": list(labels.columns),
        "event_counts": events["process"].value_counts().to_dict() if not events.empty else {},
        "base_shard": str(base_path),
        "config": config.to_dict(),
    }
    (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    if target.exists():
        shutil.rmtree(target)
    temp.replace(target)
    return target


def build_event_caches(
    base_paths: list[Path],
    config: FutureProcessForecastConfig,
    *,
    force_rebuild: bool = False,
    progress: bool = True,
) -> list[Path]:
    config.event_cache_path.mkdir(parents=True, exist_ok=True)
    eligible: list[Path] = []
    for path in base_paths:
        shard = load_year_shard(path)
        year = pd.to_datetime(np.asarray(shard.decision_times_ns[:1], dtype=np.int64))[0].year
        if year <= pd.Timestamp(config.research_end).year:
            eligible.append(path)
    reporter = ProgressReporter("[R03.3 events] years", len(eligible), every=1, enabled=progress)
    outputs: list[Path] = []
    for index, path in enumerate(eligible, start=1):
        outputs.append(build_event_year_cache(path, config, force_rebuild=force_rebuild))
        reporter.update(index)
    reporter.close()
    return outputs


@dataclass(frozen=True)
class EventYearShard:
    path: Path
    decision_times_ns: np.ndarray
    labels: np.ndarray
    label_columns: tuple[str, ...]
    events: pd.DataFrame

    @property
    def label_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.label_columns)}


def load_event_year_shard(path: str | Path) -> EventYearShard:
    target = Path(path)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported R03.3 event cache: {target}")
    events_path = target / "events.csv"
    events = pd.read_csv(events_path, parse_dates=["start_time", "end_time"]) if events_path.exists() else pd.DataFrame()
    return EventYearShard(
        path=target,
        decision_times_ns=np.load(target / "decision_times_ns.npy", mmap_mode="r"),
        labels=np.load(target / "labels.npy", mmap_mode="r"),
        label_columns=tuple(manifest["label_columns"]),
        events=events,
    )
