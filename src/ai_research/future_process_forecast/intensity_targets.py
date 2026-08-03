#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Continuous future opportunity targets for R03.3.2."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.swing_baseline.dataset import SwingYearShard, load_year_shard
from src.research_common.progress import ProgressReporter

from .events import build_available_15m_path
from .intensity_config import FutureIntensityConfig


CACHE_SCHEMA_VERSION = 2
_EPS = 1e-8


def _future_extremes_from_starts(
    values: np.ndarray,
    starts: np.ndarray,
    window: int,
    reducer: str,
) -> np.ndarray:
    """Reduce full future windows beginning at per-decision start positions.

    ``starts`` is the first 15-minute bar whose availability time is strictly
    later than the decision time.  This supports year-boundary decisions such
    as ``00:00`` even though the first completed 15-minute bar is available at
    ``00:15``.
    """
    array = np.asarray(values, dtype=float)
    positions = np.asarray(starts, dtype=np.int64)
    output = np.full(len(positions), np.nan, dtype=float)
    if window < 1 or len(array) < window or not len(positions):
        return output

    valid = (positions >= 0) & (positions + window <= len(array))
    if not np.any(valid):
        return output

    views = np.lib.stride_tricks.sliding_window_view(array, window_shape=window)
    selected = views[positions[valid]]
    if reducer == "max":
        output[valid] = np.nanmax(selected, axis=1)
    elif reducer == "min":
        output[valid] = np.nanmin(selected, axis=1)
    else:
        raise ValueError(f"unsupported future reducer: {reducer}")
    return output


def build_intensity_target_frame(
    path: pd.DataFrame,
    decision_index: pd.DatetimeIndex,
    entry_prices: np.ndarray,
    atr_pct_4h: np.ndarray,
    config: FutureIntensityConfig,
) -> pd.DataFrame:
    """Build future-only continuous targets on the causal decision axis."""
    if path.empty:
        raise ValueError("R03.3.2 path is empty")
    frame = path.loc[~path.index.duplicated(keep="last")].sort_index(kind="stable")
    decisions = pd.DatetimeIndex(decision_index)
    # A 15-minute bar stamped 00:15 contains the path from 00:00 through
    # 00:14 and only becomes available at 00:15.  The future window must start
    # at the first bar available strictly after each decision.  Requiring an
    # exact timestamp match incorrectly rejects January 1st 00:00 decisions.
    future_starts = frame.index.searchsorted(decisions, side="right").astype(np.int64, copy=False)
    current_positions = future_starts - 1

    prices = np.asarray(entry_prices, dtype=float)
    atr = np.asarray(atr_pct_4h, dtype=float)
    if len(prices) != len(future_starts) or len(atr) != len(future_starts):
        raise ValueError("R03.3.2 decision arrays must share one axis")

    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    fallback_close = np.full(len(future_starts), np.nan, dtype=float)
    valid_current = (current_positions >= 0) & (current_positions < len(close))
    fallback_close[valid_current] = close[current_positions[valid_current]]
    output = pd.DataFrame(index=pd.DatetimeIndex(decision_index))
    output.index.name = "decision_time"

    for horizon in config.horizons_hours:
        bars = int(horizon * 60 / config.base.event_scan_timeframe_minutes)
        future_high = _future_extremes_from_starts(high, future_starts, bars, "max")
        future_low = _future_extremes_from_starts(low, future_starts, bars, "min")
        valid_price = np.where(np.isfinite(prices) & (prices > 0), prices, fallback_close)
        up = future_high / valid_price - 1.0
        down = 1.0 - future_low / valid_price
        total_range = future_high / future_low - 1.0
        max_directional = np.maximum(up, down)
        two_sided = np.minimum(up, down)
        expected_scale = np.maximum(atr * np.sqrt(max(horizon / 4.0, 1.0)), _EPS)
        range_multiple = total_range / expected_scale

        invalid = ~(np.isfinite(future_high) & np.isfinite(future_low) & np.isfinite(valid_price))
        for values in (total_range, max_directional, two_sided, range_multiple):
            values[invalid] = np.nan
            values[values < 0] = np.nan

        output[f"future_range_pct_h{horizon}"] = total_range
        output[f"future_max_directional_pct_h{horizon}"] = max_directional
        output[f"future_two_sided_pct_h{horizon}"] = two_sided
        output[f"future_range_atr_multiple_h{horizon}"] = range_multiple

    return output


def intensity_cache_path(config: FutureIntensityConfig, year: int) -> Path:
    return config.target_cache_path / f"targets_{int(year)}"


def _context_atr_pct_4h(shard: SwingYearShard) -> np.ndarray:
    index = shard.context_index
    if "ctx_atr_pct_4h" not in index:
        raise RuntimeError("R03.3.2 requires causal ctx_atr_pct_4h from R03.2 cache")
    return np.asarray(shard.context[:, index["ctx_atr_pct_4h"]], dtype=float)


def build_intensity_year_cache(
    base_path: Path,
    config: FutureIntensityConfig,
    *,
    force_rebuild: bool = False,
) -> Path:
    shard = load_year_shard(base_path)
    year = int(pd.to_datetime(np.asarray(shard.decision_times_ns[:1], dtype=np.int64))[0].year)
    target = intensity_cache_path(config, year)
    manifest_path = target / "manifest.json"
    if manifest_path.exists() and not force_rebuild:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                int(manifest.get("schema_version", -1)) == CACHE_SCHEMA_VERSION
                and tuple(manifest.get("target_columns", ())) == config.target_names()
            ):
                return target
        except (OSError, json.JSONDecodeError):
            pass

    path = build_available_15m_path(shard)
    decision_index = pd.to_datetime(np.asarray(shard.decision_times_ns, dtype=np.int64))
    targets = build_intensity_target_frame(
        path,
        pd.DatetimeIndex(decision_index),
        np.asarray(shard.entry_prices, dtype=float),
        _context_atr_pct_4h(shard),
        config,
    )
    targets = targets.loc[:, list(config.target_names())]

    temp = target.with_name(target.name + ".part")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=True)
    np.save(temp / "decision_times_ns.npy", np.asarray(shard.decision_times_ns, dtype=np.int64), allow_pickle=False)
    np.save(temp / "targets.npy", targets.to_numpy(dtype=np.float32, copy=True), allow_pickle=False)
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "year": year,
        "rows": int(len(targets)),
        "target_columns": list(targets.columns),
        "valid_rows": {column: int(targets[column].notna().sum()) for column in targets.columns},
        "base_shard": str(base_path),
        "config": config.to_dict(),
    }
    (temp / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    if target.exists():
        shutil.rmtree(target)
    temp.replace(target)
    return target


def build_intensity_caches(
    base_paths: list[Path],
    config: FutureIntensityConfig,
    *,
    force_rebuild: bool = False,
    progress: bool = True,
) -> list[Path]:
    config.target_cache_path.mkdir(parents=True, exist_ok=True)
    eligible: list[Path] = []
    for path in base_paths:
        shard = load_year_shard(path)
        year = int(pd.to_datetime(np.asarray(shard.decision_times_ns[:1], dtype=np.int64))[0].year)
        if year <= pd.Timestamp(config.base.research_end).year:
            eligible.append(path)
    reporter = ProgressReporter("[R03.3.2 targets] years", len(eligible), every=1, enabled=progress)
    outputs: list[Path] = []
    for index, path in enumerate(eligible, start=1):
        outputs.append(build_intensity_year_cache(path, config, force_rebuild=force_rebuild))
        reporter.update(index)
    reporter.close()
    return outputs


@dataclass(frozen=True)
class IntensityYearShard:
    path: Path
    decision_times_ns: np.ndarray
    targets: np.ndarray
    target_columns: tuple[str, ...]

    @property
    def target_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.target_columns)}


def load_intensity_year_shard(path: str | Path) -> IntensityYearShard:
    target = Path(path)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported R03.3.2 intensity cache: {target}")
    return IntensityYearShard(
        path=target,
        decision_times_ns=np.load(target / "decision_times_ns.npy", mmap_mode="r"),
        targets=np.load(target / "targets.npy", mmap_mode="r"),
        target_columns=tuple(manifest["target_columns"]),
    )
