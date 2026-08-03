#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Future-only opening-value outcomes for R03.4."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.swing_baseline.dataset import SwingYearShard, load_year_shard
from src.research_common.progress import ProgressReporter

from .config import StateContextAblationConfig

CACHE_SCHEMA_VERSION = 1


def _future_extreme(values: np.ndarray, starts: np.ndarray, window: int, reducer: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    starts = np.asarray(starts, dtype=np.int64)
    output = np.full(len(starts), np.nan, dtype=float)
    if window < 1 or len(array) < window:
        return output
    valid = (starts >= 0) & (starts + window <= len(array))
    if not np.any(valid):
        return output
    reversed_series = pd.Series(array[::-1])
    rolling = reversed_series.rolling(window=window, min_periods=window)
    if reducer == "max":
        reduced = rolling.max().to_numpy(dtype=float)[::-1]
    elif reducer == "min":
        reduced = rolling.min().to_numpy(dtype=float)[::-1]
    else:
        raise ValueError(reducer)
    output[valid] = reduced[starts[valid]]
    return output


def build_opening_outcome_frame(
    shard: SwingYearShard,
    config: StateContextAblationConfig,
) -> pd.DataFrame:
    """Build future MFE/MAE/close-return outcomes from next-minute entry prices."""
    minute_times = np.asarray(shard.minute_times_ns, dtype=np.int64)
    minute_ohlc = np.asarray(shard.minute_ohlc, dtype=float)
    entry_times = np.asarray(shard.entry_times_ns, dtype=np.int64)
    entry_prices = np.asarray(shard.entry_prices, dtype=float)
    starts = np.searchsorted(minute_times, entry_times, side="left").astype(np.int64, copy=False)
    high = minute_ohlc[:, 1]
    low = minute_ohlc[:, 2]
    close = minute_ohlc[:, 3]
    frame = pd.DataFrame(index=pd.to_datetime(np.asarray(shard.decision_times_ns, dtype=np.int64), unit="ns"))
    frame.index.name = "decision_time"

    for horizon in config.horizons_hours:
        window = int(horizon * 60)
        future_high = _future_extreme(high, starts, window, "max")
        future_low = _future_extreme(low, starts, window, "min")
        end_positions = starts + window - 1
        valid_end = (starts >= 0) & (end_positions < len(close))
        future_close = np.full(len(starts), np.nan, dtype=float)
        future_close[valid_end] = close[end_positions[valid_end]]
        horizon_end_ns = entry_times + int(pd.Timedelta(hours=horizon).value)
        before_sealed_holdout = horizon_end_ns <= int(pd.Timestamp(config.sealed_holdout_start).value)
        valid = (
            np.isfinite(entry_prices)
            & (entry_prices > 0)
            & np.isfinite(future_high)
            & np.isfinite(future_low)
            & np.isfinite(future_close)
            & before_sealed_holdout
        )
        long_mfe = np.full(len(starts), np.nan, dtype=float)
        long_mae = np.full(len(starts), np.nan, dtype=float)
        close_return = np.full(len(starts), np.nan, dtype=float)
        long_mfe[valid] = future_high[valid] / entry_prices[valid] - 1.0
        long_mae[valid] = 1.0 - future_low[valid] / entry_prices[valid]
        close_return[valid] = future_close[valid] / entry_prices[valid] - 1.0
        short_mfe = long_mae.copy()
        short_mae = long_mfe.copy()
        frame[f"long_mfe_h{horizon}"] = long_mfe
        frame[f"long_mae_h{horizon}"] = long_mae
        frame[f"short_mfe_h{horizon}"] = short_mfe
        frame[f"short_mae_h{horizon}"] = short_mae
        frame[f"future_close_return_h{horizon}"] = close_return
        frame[f"long_utility_h{horizon}"] = long_mfe - config.risk_penalty * long_mae
        frame[f"short_utility_h{horizon}"] = short_mfe - config.risk_penalty * short_mae
    return frame.loc[:, list(config.outcome_columns())]


def outcome_cache_path(config: StateContextAblationConfig, year: int) -> Path:
    return config.outcome_cache_path / f"outcomes_{year}"


def build_outcome_year_cache(
    base_path: Path,
    config: StateContextAblationConfig,
    *,
    force_rebuild: bool = False,
) -> Path:
    shard = load_year_shard(base_path)
    times = np.asarray(shard.decision_times_ns, dtype=np.int64)
    year = int(pd.to_datetime(times[:1], unit="ns")[0].year)
    target = outcome_cache_path(config, year)
    manifest_path = target / "manifest.json"
    if manifest_path.exists() and not force_rebuild:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                int(manifest.get("schema_version", -1)) == CACHE_SCHEMA_VERSION
                and tuple(manifest.get("outcome_columns", ())) == config.outcome_columns()
            ):
                return target
        except (OSError, json.JSONDecodeError):
            pass
    outcomes = build_opening_outcome_frame(shard, config)
    temp = target.with_name(target.name + ".part")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=True)
    np.save(temp / "decision_times_ns.npy", times, allow_pickle=False)
    np.save(temp / "outcomes.npy", outcomes.to_numpy(dtype=np.float32, copy=True), allow_pickle=False)
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "year": year,
        "timestamp_unit": "ns",
        "rows": int(len(outcomes)),
        "outcome_columns": list(outcomes.columns),
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


def build_outcome_caches(
    base_paths: list[Path],
    config: StateContextAblationConfig,
    *,
    force_rebuild: bool = False,
    progress: bool = True,
) -> list[Path]:
    config.outcome_cache_path.mkdir(parents=True, exist_ok=True)
    reporter = ProgressReporter("[R03.4 outcomes] years", len(base_paths), every=1, enabled=progress)
    outputs: list[Path] = []
    for index, path in enumerate(base_paths, start=1):
        outputs.append(build_outcome_year_cache(path, config, force_rebuild=force_rebuild))
        reporter.update(index)
    reporter.close()
    return outputs


@dataclass(frozen=True)
class OpeningOutcomeShard:
    path: Path
    year: int
    decision_times_ns: np.ndarray
    outcomes: np.ndarray
    outcome_columns: tuple[str, ...]

    @property
    def outcome_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.outcome_columns)}


def load_outcome_year_shard(path: str | Path) -> OpeningOutcomeShard:
    target = Path(path)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported R03.4 outcome cache: {target}")
    return OpeningOutcomeShard(
        path=target,
        year=int(manifest["year"]),
        decision_times_ns=np.load(target / "decision_times_ns.npy", mmap_mode="r"),
        outcomes=np.load(target / "outcomes.npy", mmap_mode="r"),
        outcome_columns=tuple(manifest["outcome_columns"]),
    )
