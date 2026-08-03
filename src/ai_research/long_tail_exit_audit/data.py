#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Data alignment and causal one-minute path preparation for R03.4.2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.state_context_ablation.config import StateContextAblationConfig
from src.ai_research.state_context_ablation.modeling import AblationPeriodData
from src.ai_research.state_context_ablation.outcomes import load_outcome_year_shard
from src.ai_research.swing_baseline.dataset import load_year_shard
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.research_common.progress import ProgressReporter

from .config import LongTailExitAuditConfig


def _year_from_base_path(path: Path) -> int:
    shard = load_year_shard(path)
    first = int(np.asarray(shard.decision_times_ns[:1], dtype=np.int64)[0])
    return int(pd.Timestamp(first, unit="ns").year)


def collect_base_period_data(
    base_paths: list[Path],
    outcome_paths: list[Path],
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    outcome_config: StateContextAblationConfig,
) -> AblationPeriodData:
    """Collect only frozen base features and outcomes; no state cache is loaded."""

    base_map = {_year_from_base_path(path): path for path in base_paths}
    outcome_map = {load_outcome_year_shard(path).year: path for path in outcome_paths}
    time_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    outcome_parts: dict[str, list[np.ndarray]] = {name: [] for name in outcome_config.outcome_columns()}
    expected_columns: tuple[str, ...] | None = None

    for year in sorted(base_map):
        if year not in outcome_map:
            continue
        base = load_year_shard(base_map[year])
        outcome = load_outcome_year_shard(outcome_map[year])
        common, base_pos, outcome_pos = np.intersect1d(
            np.asarray(base.decision_times_ns, dtype=np.int64),
            np.asarray(outcome.decision_times_ns, dtype=np.int64),
            assume_unique=True,
            return_indices=True,
        )
        if not len(common):
            continue
        left = int(np.searchsorted(common, int(pd.Timestamp(start).value), side="left"))
        right = int(np.searchsorted(common, int(pd.Timestamp(end).value), side="right"))
        if right <= left:
            continue
        if expected_columns is None:
            expected_columns = tuple(base.full_feature_columns)
        elif tuple(base.full_feature_columns) != expected_columns:
            raise RuntimeError(f"R03.4.2 base feature schema drift in year {year}")
        time_parts.append(common[left:right])
        feature_parts.append(np.asarray(base.features[base_pos[left:right]], dtype=np.float32))
        for name in outcome_config.outcome_columns():
            outcome_parts[name].append(
                np.asarray(
                    outcome.outcomes[outcome_pos[left:right], outcome.outcome_index[name]],
                    dtype=float,
                )
            )

    if not time_parts or expected_columns is None:
        raise RuntimeError(f"R03.4.2 no base/outcome data for {start} -> {end}")
    rows = sum(len(part) for part in time_parts)
    return AblationPeriodData(
        timestamps_ns=np.concatenate(time_parts),
        base_x=np.concatenate(feature_parts, axis=0),
        state_x=np.empty((rows, 0), dtype=np.float32),
        outcomes={name: np.concatenate(parts) for name, parts in outcome_parts.items()},
        base_columns=expected_columns,
        state_columns=(),
    )


@dataclass(frozen=True)
class MinutePathData:
    index: pd.DatetimeIndex
    timestamps_ns: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    prior_low_60: np.ndarray
    prior_low_180: np.ndarray
    prior_atr_60: np.ndarray
    coverage_ratio: float

    def locate_exact(self, timestamp: pd.Timestamp) -> int | None:
        value = int(pd.Timestamp(timestamp).value)
        position = int(np.searchsorted(self.timestamps_ns, value, side="left"))
        if position >= len(self.timestamps_ns) or int(self.timestamps_ns[position]) != value:
            return None
        return position


def prepare_minute_path_frame(frame: pd.DataFrame) -> MinutePathData:
    if frame.empty:
        raise RuntimeError("R03.4.2 one-minute path data is empty")
    required = {"open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"R03.4.2 path columns missing: {missing}")
    work = frame.loc[:, ["open", "high", "low", "close"]].copy()
    index = pd.DatetimeIndex(pd.to_datetime(work.index)).astype("datetime64[ns]")
    work.index = index
    work = work[~work.index.duplicated(keep="last")].sort_index()
    if not work.index.is_monotonic_increasing:
        raise RuntimeError("R03.4.2 minute path index must be monotonic")
    values = work.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise RuntimeError("R03.4.2 minute OHLC contains non-finite values")
    if np.any(values[:, 1] < values[:, 2]):
        raise RuntimeError("R03.4.2 high below low")

    previous_close = work["close"].shift(1)
    true_range = pd.concat(
        [
            work["high"] - work["low"],
            (work["high"] - previous_close).abs(),
            (work["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    prior_low_60 = work["low"].rolling(60, min_periods=60).min().shift(1)
    prior_low_180 = work["low"].rolling(180, min_periods=180).min().shift(1)
    prior_atr_60 = true_range.rolling(60, min_periods=60).mean().shift(1)

    elapsed = int((work.index[-1] - work.index[0]) / pd.Timedelta(minutes=1)) + 1
    coverage = float(len(work) / max(elapsed, 1))
    return MinutePathData(
        index=work.index,
        timestamps_ns=work.index.to_numpy(dtype="datetime64[ns]").astype(np.int64),
        open=work["open"].to_numpy(dtype=float),
        high=work["high"].to_numpy(dtype=float),
        low=work["low"].to_numpy(dtype=float),
        close=work["close"].to_numpy(dtype=float),
        prior_low_60=prior_low_60.to_numpy(dtype=float),
        prior_low_180=prior_low_180.to_numpy(dtype=float),
        prior_atr_60=prior_atr_60.to_numpy(dtype=float),
        coverage_ratio=coverage,
    )


def load_minute_path_data(
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    data_dir: str | Path | None,
    config: LongTailExitAuditConfig,
    progress: bool = True,
) -> MinutePathData:
    """Load a long 1m path in bounded chunks and retain OHLC only.

    The public loader returns the full Trade Bar schema. Reading a whole year
    at once would temporarily hold every order-flow column even though the exit
    simulator only needs OHLC. Monthly chunks keep peak memory bounded while
    preserving the canonical ``src.data_feed`` interface.
    """

    loader = OKXTradeBarLoader(
        symbol=config.symbol,
        timeframe="1m",
        data_dir=data_dir,
    )
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start_ts
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.Timedelta(days=31) - pd.Timedelta(microseconds=1), end_ts)
        windows.append((cursor, chunk_end))
        cursor = chunk_end + pd.Timedelta(microseconds=1)
    reporter = ProgressReporter("[R03.4.2 1m path] chunks", len(windows), every=1, enabled=progress)
    parts: list[pd.DataFrame] = []
    for number, (chunk_start, chunk_end) in enumerate(windows, start=1):
        frame = loader.fetch_data_by_date_range(
            chunk_start,
            chunk_end,
            build_missing=False,
            cvd_mode="range",
        )
        if not frame.empty:
            parts.append(frame.loc[:, ["open", "high", "low", "close"]].copy())
        reporter.update(number)
    reporter.close()
    if not parts:
        raise RuntimeError("R03.4.2 one-minute path data is empty")
    return prepare_minute_path_frame(pd.concat(parts, axis=0))
