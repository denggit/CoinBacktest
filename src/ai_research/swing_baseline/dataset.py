#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Public-loader data access, causal labels, and resumable R03 cache shards."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.research_common.progress import ProgressReporter

from .config import SwingBaselineConfig, SwingTargetSpec
from .features import (
    BASE_FEATURE_PROFILE,
    FLOW_MAX_COLUMNS,
    FLOW_SUM_COLUMNS,
    REQUIRED_PRICE_COLUMNS,
    build_causal_minute_grid,
    build_multitimeframe_feature_bundle,
)


CACHE_SCHEMA_VERSION = 1
REQUIRED_LOADER_COLUMNS = (
    *REQUIRED_PRICE_COLUMNS,
    "volume",
    "notional",
    "buy_notional",
    "sell_notional",
    "delta_notional",
    "large_buy_notional",
    "large_sell_notional",
    "large_delta_notional",
)
ENTRY_COLUMNS = ("entry_price",)


def _datetime_index_to_ns(index: pd.Index | pd.DatetimeIndex) -> np.ndarray:
    dt_index = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    if dt_index.tz is not None:
        dt_index = dt_index.tz_convert("UTC").tz_localize(None)
    return dt_index.to_numpy(dtype="datetime64[ns]", copy=False).astype(np.int64, copy=False)


def create_loader(config: SwingBaselineConfig, *, data_dir: str | Path | None = None) -> OKXTradeBarLoader:
    return OKXTradeBarLoader(
        symbol=config.symbol,
        timeframe=config.source_timeframe,
        data_dir=data_dir,
        align_with_okx_loader_timezone=True,
    )


def load_public_minute_bars(
    loader: OKXTradeBarLoader,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    bars = loader.fetch_data_by_date_range(
        start,
        end,
        build_missing=False,
        force_rebuild=False,
        cvd_mode="range",
    )
    if bars.empty:
        return bars
    out = bars.copy()
    index = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce"))
    if index.tz is not None:
        index = index.tz_convert("UTC").tz_localize(None)
    out.index = index
    out = out.loc[~out.index.isna()]
    return out.sort_index(kind="stable")


@dataclass(frozen=True)
class SwingPreflightResult:
    status: str
    sample_windows: tuple[dict[str, object], ...]
    loader_class: str = "src.data_feed.okx_trade_bar_loader.OKXTradeBarLoader"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "loader_class": self.loader_class,
            "required_columns": list(REQUIRED_LOADER_COLUMNS),
            "sample_windows": list(self.sample_windows),
        }


def run_public_loader_preflight(
    loader: OKXTradeBarLoader,
    config: SwingBaselineConfig,
    *,
    sample_dates: Iterable[str | pd.Timestamp] | None = None,
) -> SwingPreflightResult:
    if sample_dates is None:
        sample_dates = ("2023-01-15", "2025-06-15", "2026-06-15")
    rows: list[dict[str, object]] = []
    status = "PASS"
    for raw in sample_dates:
        start = pd.Timestamp(raw)
        end = start + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
        bars = load_public_minute_bars(loader, start, end)
        missing = sorted(set(REQUIRED_LOADER_COLUMNS) - set(bars.columns))
        monotonic = bool(bars.index.is_monotonic_increasing) if not bars.empty else False
        unique = bool(bars.index.is_unique) if not bars.empty else False
        finite = False
        if not bars.empty and not missing:
            numeric = bars[list(REQUIRED_LOADER_COLUMNS)].apply(pd.to_numeric, errors="coerce")
            finite = bool(np.isfinite(numeric.to_numpy(dtype=float, copy=False)).all())
        if bars.empty or missing or not monotonic or not unique or not finite:
            status = "BLOCKED"
        rows.append(
            {
                "start": str(start),
                "end": str(end),
                "rows": int(len(bars)),
                "missing_columns": missing,
                "monotonic": monotonic,
                "unique": unique,
                "finite": finite,
            }
        )
    return SwingPreflightResult(status=status, sample_windows=tuple(rows))


def label_columns(config: SwingBaselineConfig) -> list[str]:
    columns: list[str] = []
    for spec in config.target_specs:
        columns.extend(
            [
                f"{spec.target_id}_long_quality",
                f"{spec.target_id}_short_quality",
                f"{spec.target_id}_long_mfe",
                f"{spec.target_id}_long_mae",
                f"{spec.target_id}_short_mfe",
                f"{spec.target_id}_short_mae",
                f"{spec.target_id}_future_close_return",
            ]
        )
    return columns


def cache_signature(
    config: SwingBaselineConfig,
    *,
    feature_profile: str = BASE_FEATURE_PROFILE,
) -> str:
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "symbol": config.symbol,
        "source_timeframe": config.source_timeframe,
        "warmup_start": config.warmup_start,
        "research_start": config.research_start,
        "research_end": config.research_end,
        "decision_interval_minutes": config.decision_interval_minutes,
        "execution_delay_minutes": config.execution_delay_minutes,
        "feature_lookback_days": config.feature_lookback_days,
        "structural_swing_bars_4h": config.structural_swing_bars_4h,
        "target_specs": [spec.to_dict() for spec in config.target_specs],
        "label_columns": label_columns(config),
        "timeframe_feature_schema": feature_profile,
        "causal_availability": "bar_start_plus_timeframe",
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _forward_extreme(series: pd.Series, window: int, *, kind: str) -> pd.Series:
    reversed_series = series.iloc[::-1]
    rolling = reversed_series.rolling(window=window, min_periods=window)
    if kind == "max":
        out = rolling.max()
    elif kind == "min":
        out = rolling.min()
    else:
        raise ValueError(f"unsupported forward extreme kind: {kind}")
    return out.iloc[::-1]


def _build_labels(
    minute_grid: pd.DataFrame,
    decision_index: pd.DatetimeIndex,
    config: SwingBaselineConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    entry_times = decision_index + pd.Timedelta(minutes=config.execution_delay_minutes)
    entry_price = minute_grid["open"].reindex(entry_times)
    labels = pd.DataFrame(index=decision_index)
    entries = pd.DataFrame(index=decision_index)
    entries["entry_price"] = entry_price.to_numpy(dtype=float)

    high = minute_grid["high"].astype(float)
    low = minute_grid["low"].astype(float)
    close = minute_grid["close"].astype(float)
    for spec in config.target_specs:
        window = int(spec.horizon_hours * 60)
        future_high = _forward_extreme(high, window, kind="max")
        future_low = _forward_extreme(low, window, kind="min")
        future_close = close.shift(-(window - 1))
        sampled_high = future_high.reindex(entry_times).to_numpy(dtype=float)
        sampled_low = future_low.reindex(entry_times).to_numpy(dtype=float)
        sampled_close = future_close.reindex(entry_times).to_numpy(dtype=float)
        entry = entry_price.to_numpy(dtype=float)
        valid = np.isfinite(entry) & (entry > 0) & np.isfinite(sampled_high) & np.isfinite(sampled_low)

        long_mfe = np.full(len(entry), np.nan, dtype=float)
        long_mae = np.full(len(entry), np.nan, dtype=float)
        short_mfe = np.full(len(entry), np.nan, dtype=float)
        short_mae = np.full(len(entry), np.nan, dtype=float)
        future_return = np.full(len(entry), np.nan, dtype=float)
        long_mfe[valid] = sampled_high[valid] / entry[valid] - 1.0
        long_mae[valid] = 1.0 - sampled_low[valid] / entry[valid]
        short_mfe[valid] = 1.0 - sampled_low[valid] / entry[valid]
        short_mae[valid] = sampled_high[valid] / entry[valid] - 1.0
        close_valid = valid & np.isfinite(sampled_close)
        future_return[close_valid] = sampled_close[close_valid] / entry[close_valid] - 1.0

        labels[f"{spec.target_id}_long_mfe"] = long_mfe
        labels[f"{spec.target_id}_long_mae"] = long_mae
        labels[f"{spec.target_id}_short_mfe"] = short_mfe
        labels[f"{spec.target_id}_short_mae"] = short_mae
        labels[f"{spec.target_id}_future_close_return"] = future_return
        labels[f"{spec.target_id}_long_quality"] = np.where(
            valid,
            (long_mfe >= spec.target_move) & (long_mae <= spec.max_adverse_move),
            np.nan,
        )
        labels[f"{spec.target_id}_short_quality"] = np.where(
            valid,
            (short_mfe >= spec.target_move) & (short_mae <= spec.max_adverse_move),
            np.nan,
        )
    return labels, entries


def _year_ranges(config: SwingBaselineConfig) -> Iterator[tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp(config.research_start)
    end = pd.Timestamp(config.research_end)
    for year in range(start.year, end.year + 1):
        year_start = max(start, pd.Timestamp(year=year, month=1, day=1))
        year_end = min(end, pd.Timestamp(year=year, month=12, day=31, hour=23, minute=59, second=59))
        if year_start <= year_end:
            yield year_start, year_end


def year_cache_path(config: SwingBaselineConfig, year_start: pd.Timestamp) -> Path:
    return config.cache_path / f"samples_{year_start.year}"


def _write_year_cache(
    path: Path,
    *,
    decision_index: pd.DatetimeIndex,
    features: pd.DataFrame,
    high_feature_columns: tuple[str, ...],
    full_feature_columns: tuple[str, ...],
    context_columns: tuple[str, ...],
    labels: pd.DataFrame,
    entries: pd.DataFrame,
    minute_path: pd.DataFrame,
    config: SwingBaselineConfig,
    grid_stats: dict[str, float],
    feature_profile: str = BASE_FEATURE_PROFILE,
) -> None:
    temp = path.with_name(path.name + ".part")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=True)
    np.save(temp / "decision_times_ns.npy", _datetime_index_to_ns(decision_index), allow_pickle=False)
    np.save(
        temp / "features.npy",
        features[list(full_feature_columns)].to_numpy(dtype=np.float32, copy=True),
        allow_pickle=False,
    )
    np.save(
        temp / "context.npy",
        features[list(context_columns)].to_numpy(dtype=np.float64, copy=True),
        allow_pickle=False,
    )
    np.save(
        temp / "labels.npy",
        labels[label_columns(config)].to_numpy(dtype=np.float32, copy=True),
        allow_pickle=False,
    )
    np.save(temp / "entry_times_ns.npy", _datetime_index_to_ns(decision_index + pd.Timedelta(minutes=config.execution_delay_minutes)), allow_pickle=False)
    np.save(
        temp / "entry_prices.npy",
        entries["entry_price"].to_numpy(dtype=np.float64, copy=True),
        allow_pickle=False,
    )
    np.save(temp / "minute_times_ns.npy", _datetime_index_to_ns(minute_path.index), allow_pickle=False)
    np.save(
        temp / "minute_ohlc.npy",
        minute_path[["open", "high", "low", "close"]].to_numpy(dtype=np.float64, copy=True),
        allow_pickle=False,
    )
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_signature": cache_signature(config, feature_profile=feature_profile),
        "feature_profile": feature_profile,
        "rows": int(len(decision_index)),
        "minute_rows": int(len(minute_path)),
        "start": str(decision_index.min()),
        "end": str(decision_index.max()),
        "minute_start": str(minute_path.index.min()),
        "minute_end": str(minute_path.index.max()),
        "high_feature_columns": list(high_feature_columns),
        "full_feature_columns": list(full_feature_columns),
        "context_columns": list(context_columns),
        "label_columns": label_columns(config),
        "entry_columns": list(ENTRY_COLUMNS),
        "grid_stats": grid_stats,
    }
    (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if path.exists():
        shutil.rmtree(path)
    temp.replace(path)


def build_year_cache(
    loader: OKXTradeBarLoader,
    config: SwingBaselineConfig,
    year_start: pd.Timestamp,
    year_end: pd.Timestamp,
    *,
    force_rebuild: bool = False,
    feature_profile: str = BASE_FEATURE_PROFILE,
) -> Path:
    path = year_cache_path(config, year_start)
    manifest_path = path / "manifest.json"
    if manifest_path.exists() and not force_rebuild:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("cache_signature") == cache_signature(config, feature_profile=feature_profile):
                return path
        except (OSError, json.JSONDecodeError):
            pass

    load_start = max(pd.Timestamp(config.warmup_start), year_start - pd.Timedelta(days=config.feature_lookback_days))
    requested_future_end = year_end + pd.Timedelta(hours=config.max_horizon_hours + 1)
    load_end = min(pd.Timestamp(config.research_end), requested_future_end)
    bars = load_public_minute_bars(loader, load_start, load_end)
    if bars.empty:
        raise RuntimeError(f"public 1m loader returned no R03 data for {load_start} -> {load_end}")
    missing = sorted(set(REQUIRED_LOADER_COLUMNS) - set(bars.columns))
    if missing:
        raise RuntimeError(f"public 1m loader missing R03 columns: {missing}")

    minute_grid, grid_stats = build_causal_minute_grid(bars, load_start, load_end)
    if minute_grid.empty:
        raise RuntimeError(f"R03 minute grid is empty for year {year_start.year}")
    if grid_stats["gap_ratio"] > 0.01:
        raise RuntimeError(
            f"R03 public 1m gap ratio is unexpectedly high for {year_start.year}: {grid_stats['gap_ratio']:.4%}"
        )

    decision_start = year_start.ceil(f"{config.decision_interval_minutes}min")
    decision_end = year_end.floor(f"{config.decision_interval_minutes}min")
    decision_index = pd.date_range(
        decision_start,
        decision_end,
        freq=f"{config.decision_interval_minutes}min",
    )
    bundle = build_multitimeframe_feature_bundle(
        minute_grid,
        decision_index,
        structural_swing_bars_4h=config.structural_swing_bars_4h,
        feature_profile=feature_profile,
    )
    labels, entries = _build_labels(minute_grid, decision_index, config)
    combined = bundle.frame.join(labels, how="left").join(entries, how="left")
    required = [*bundle.full_feature_columns, *bundle.context_columns, *label_columns(config), *ENTRY_COLUMNS]
    valid = combined[required].notna().all(axis=1)
    combined = combined.loc[valid]
    if combined.empty:
        raise RuntimeError(
            f"R03 produced zero valid samples for {year_start.year}; "
            "check causal availability, warmup, and future-label coverage"
        )
    features = combined[[*bundle.full_feature_columns, *bundle.context_columns]]
    labels = combined[label_columns(config)]
    entries = combined[list(ENTRY_COLUMNS)]
    decision_index = pd.DatetimeIndex(combined.index)

    minute_path_start = year_start
    minute_path_end = min(load_end, year_end + pd.Timedelta(hours=config.max_hold_hours + 1))
    minute_path = minute_grid.loc[(minute_grid.index >= minute_path_start) & (minute_grid.index <= minute_path_end)]
    _write_year_cache(
        path,
        decision_index=decision_index,
        features=features,
        high_feature_columns=bundle.high_feature_columns,
        full_feature_columns=bundle.full_feature_columns,
        context_columns=bundle.context_columns,
        labels=labels,
        entries=entries,
        minute_path=minute_path,
        config=config,
        grid_stats=grid_stats,
        feature_profile=feature_profile,
    )
    return path


def build_yearly_cache(
    loader: OKXTradeBarLoader,
    config: SwingBaselineConfig,
    *,
    force_rebuild: bool = False,
    progress: bool = True,
    feature_profile: str = BASE_FEATURE_PROFILE,
) -> list[Path]:
    config.cache_path.mkdir(parents=True, exist_ok=True)
    ranges = list(_year_ranges(config))
    reporter = ProgressReporter("[R03 cache] years", len(ranges), every=1, enabled=progress)
    outputs: list[Path] = []
    for index, (year_start, year_end) in enumerate(ranges, start=1):
        outputs.append(
            build_year_cache(
                loader,
                config,
                year_start,
                year_end,
                force_rebuild=force_rebuild,
                feature_profile=feature_profile,
            )
        )
        reporter.update(index)
    reporter.close()
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "storage": "yearly_npy_memmap",
        "cache_signature": cache_signature(config, feature_profile=feature_profile),
        "feature_profile": feature_profile,
        "config": config.to_dict(),
        "files": [path.name for path in outputs],
    }
    (config.cache_path / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return outputs


@dataclass(frozen=True)
class SwingYearShard:
    path: Path
    decision_times_ns: np.ndarray
    features: np.ndarray
    context: np.ndarray
    labels: np.ndarray
    entry_times_ns: np.ndarray
    entry_prices: np.ndarray
    minute_times_ns: np.ndarray
    minute_ohlc: np.ndarray
    high_feature_columns: tuple[str, ...]
    full_feature_columns: tuple[str, ...]
    context_columns: tuple[str, ...]
    label_columns: tuple[str, ...]

    @property
    def label_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.label_columns)}

    @property
    def context_index(self) -> dict[str, int]:
        return {name: index for index, name in enumerate(self.context_columns)}

    def decision_positions(self, start: pd.Timestamp, end: pd.Timestamp) -> slice:
        left = int(np.searchsorted(self.decision_times_ns, int(pd.Timestamp(start).value), side="left"))
        right = int(np.searchsorted(self.decision_times_ns, int(pd.Timestamp(end).value), side="right"))
        return slice(left, right)


def load_year_shard(path: str | Path) -> SwingYearShard:
    target = Path(path)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != CACHE_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported R03 cache schema: {target}")
    return SwingYearShard(
        path=target,
        decision_times_ns=np.load(target / "decision_times_ns.npy", mmap_mode="r"),
        features=np.load(target / "features.npy", mmap_mode="r"),
        context=np.load(target / "context.npy", mmap_mode="r"),
        labels=np.load(target / "labels.npy", mmap_mode="r"),
        entry_times_ns=np.load(target / "entry_times_ns.npy", mmap_mode="r"),
        entry_prices=np.load(target / "entry_prices.npy", mmap_mode="r"),
        minute_times_ns=np.load(target / "minute_times_ns.npy", mmap_mode="r"),
        minute_ohlc=np.load(target / "minute_ohlc.npy", mmap_mode="r"),
        high_feature_columns=tuple(manifest["high_feature_columns"]),
        full_feature_columns=tuple(manifest["full_feature_columns"]),
        context_columns=tuple(manifest["context_columns"]),
        label_columns=tuple(manifest["label_columns"]),
    )


def list_cached_years(config: SwingBaselineConfig) -> list[Path]:
    return sorted(path for path in config.cache_path.glob("samples_????") if (path / "manifest.json").exists())
