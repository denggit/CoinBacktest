#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Memory-bounded 5s Trade Flow features aligned to the R03.2 decision axis."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.swing_baseline.dataset import SwingYearShard, load_year_shard
from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.research_common.progress import ProgressReporter

from .config import FutureProcessForecastConfig


CACHE_SCHEMA_VERSION = 1
REQUIRED_MICRO_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "notional",
    "delta_notional",
    "large_delta_notional",
    "trades_count",
    "large_trades_count",
    "max_trade_notional",
)


def create_micro_loader(
    config: FutureProcessForecastConfig,
    *,
    data_dir: str | Path | None = None,
) -> OKXTradeBarLoader:
    return OKXTradeBarLoader(
        symbol=config.symbol,
        timeframe=config.micro_timeframe,
        data_dir=data_dir,
        align_with_okx_loader_timezone=True,
    )


def load_micro_bars(loader: OKXTradeBarLoader, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
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
    return out.loc[~out.index.isna()].sort_index(kind="stable")


@dataclass(frozen=True)
class MicroPreflightResult:
    status: str
    rows: tuple[dict[str, object], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "timeframe": "5s",
            "required_columns": list(REQUIRED_MICRO_COLUMNS),
            "sample_windows": list(self.rows),
        }


def run_micro_preflight(
    loader: OKXTradeBarLoader,
    config: FutureProcessForecastConfig,
) -> MicroPreflightResult:
    rows: list[dict[str, object]] = []
    status = "PASS"
    for raw in ("2023-01-15", "2024-07-15", "2025-06-15"):
        start = pd.Timestamp(raw)
        end = start + pd.Timedelta(hours=2)
        bars = load_micro_bars(loader, start, end)
        missing = sorted(set(REQUIRED_MICRO_COLUMNS) - set(bars.columns))
        expected = int((end - start).total_seconds() / pd.Timedelta(config.micro_timeframe).total_seconds())
        coverage = min(1.0, len(bars) / max(expected, 1))
        valid = bool(not bars.empty and not missing and coverage >= config.minimum_micro_coverage)
        if not valid:
            status = "BLOCKED"
        rows.append(
            {
                "start": str(start),
                "end": str(end),
                "rows": int(len(bars)),
                "expected_rows": expected,
                "coverage": coverage,
                "missing_columns": missing,
                "valid": valid,
            }
        )
    return MicroPreflightResult(status=status, rows=tuple(rows))


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    safe = denominator.where(denominator.abs() > 1e-12)
    return numerator / safe


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    minimum = max(3, window // 3)
    mean = series.rolling(window, min_periods=minimum).mean()
    std = series.rolling(window, min_periods=minimum).std(ddof=0).replace(0.0, np.nan)
    return (series - mean) / std


def build_micro_decision_features(
    bars: pd.DataFrame,
    decision_index: pd.DatetimeIndex,
    config: FutureProcessForecastConfig,
) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_MICRO_COLUMNS) - set(bars.columns))
    if missing:
        raise RuntimeError(f"public {config.micro_timeframe} loader missing R03.3 fields: {missing}")
    work = bars[list(REQUIRED_MICRO_COLUMNS)].apply(pd.to_numeric, errors="coerce").copy()
    work = work.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"])
    if work.empty:
        return pd.DataFrame(index=decision_index)
    ret = work["close"].pct_change(fill_method=None).fillna(0.0)
    flow = work["delta_notional"].fillna(0.0)
    notional = work["notional"].abs().fillna(0.0)
    flow_sign = np.sign(flow)
    ret_sign = np.sign(ret)
    work["micro_ret"] = ret
    work["micro_abs_ret"] = ret.abs()
    work["flow_sign"] = flow_sign
    work["flow_alignment"] = flow_sign * ret_sign
    work["flow_flip"] = (flow_sign * flow_sign.shift(1) < 0).astype(float)
    work["buy_absorption_notional"] = flow.clip(lower=0.0).where(ret <= 0.0, 0.0)
    work["sell_absorption_notional"] = (-flow.clip(upper=0.0)).where(ret >= 0.0, 0.0)

    minute = work.resample("1min", label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "notional": "sum",
            "delta_notional": "sum",
            "large_delta_notional": "sum",
            "trades_count": "sum",
            "large_trades_count": "sum",
            "max_trade_notional": "max",
            "micro_ret": "sum",
            "micro_abs_ret": "sum",
            "flow_sign": "mean",
            "flow_alignment": "mean",
            "flow_flip": "mean",
            "buy_absorption_notional": "sum",
            "sell_absorption_notional": "sum",
        }
    )
    minute = minute.dropna(subset=["open", "high", "low", "close"])
    minute.index = minute.index + pd.Timedelta(minutes=1)
    output = pd.DataFrame(index=minute.index)
    for window in config.micro_windows_minutes:
        minimum = max(2, window // 3)
        total_notional = minute["notional"].rolling(window, min_periods=minimum).sum()
        delta = minute["delta_notional"].rolling(window, min_periods=minimum).sum()
        large_delta = minute["large_delta_notional"].rolling(window, min_periods=minimum).sum()
        total_abs_path = minute["micro_abs_ret"].rolling(window, min_periods=minimum).sum()
        net_ret = minute["micro_ret"].rolling(window, min_periods=minimum).sum()
        output[f"micro5s_flow_imb_{window}m"] = _safe_divide(delta, total_notional)
        output[f"micro5s_large_flow_imb_{window}m"] = _safe_divide(large_delta, total_notional)
        output[f"micro5s_flow_persistence_{window}m"] = minute["flow_sign"].rolling(
            window, min_periods=minimum
        ).mean()
        output[f"micro5s_flow_flip_rate_{window}m"] = minute["flow_flip"].rolling(
            window, min_periods=minimum
        ).mean()
        output[f"micro5s_return_efficiency_{window}m"] = _safe_divide(net_ret.abs(), total_abs_path)
        output[f"micro5s_impact_alignment_{window}m"] = minute["flow_alignment"].rolling(
            window, min_periods=minimum
        ).mean()
        output[f"micro5s_buy_absorption_{window}m"] = _safe_divide(
            minute["buy_absorption_notional"].rolling(window, min_periods=minimum).sum(),
            total_notional,
        )
        output[f"micro5s_sell_absorption_{window}m"] = _safe_divide(
            minute["sell_absorption_notional"].rolling(window, min_periods=minimum).sum(),
            total_notional,
        )
        output[f"micro5s_burst_share_{window}m"] = _safe_divide(
            minute["max_trade_notional"].rolling(window, min_periods=minimum).max(),
            total_notional,
        )
        output[f"micro5s_large_trade_rate_{window}m"] = _safe_divide(
            minute["large_trades_count"].rolling(window, min_periods=minimum).sum(),
            minute["trades_count"].rolling(window, min_periods=minimum).sum(),
        )
        minute_range = (minute["high"] - minute["low"]) / minute["close"]
        output[f"micro5s_realized_range_{window}m"] = minute_range.rolling(
            window, min_periods=minimum
        ).mean()
    output["micro5s_notional_z_60m"] = _rolling_zscore(np.log1p(minute["notional"]), 60)
    output["micro5s_trades_z_60m"] = _rolling_zscore(np.log1p(minute["trades_count"]), 60)
    output = output.replace([np.inf, -np.inf], np.nan)
    return output.reindex(decision_index, method="ffill")


def micro_cache_path(config: FutureProcessForecastConfig, year: int) -> Path:
    return config.micro_cache_path / f"features_{year}"


def _month_ranges(start: pd.Timestamp, end: pd.Timestamp, days: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, cursor + pd.Timedelta(days=days) - pd.Timedelta(microseconds=1))
        ranges.append((cursor, chunk_end))
        cursor = chunk_end + pd.Timedelta(microseconds=1)
    return ranges


def build_micro_year_cache(
    base_path: Path,
    loader: OKXTradeBarLoader,
    config: FutureProcessForecastConfig,
    *,
    force_rebuild: bool = False,
    progress: bool = True,
) -> Path:
    shard = load_year_shard(base_path)
    decision_times = pd.DatetimeIndex(pd.to_datetime(np.asarray(shard.decision_times_ns, dtype=np.int64)))
    year = int(decision_times[0].year)
    target = micro_cache_path(config, year)
    if (target / "manifest.json").exists() and not force_rebuild:
        return target
    warmup = pd.Timedelta(minutes=max(config.micro_windows_minutes) + 5)
    parts: list[pd.DataFrame] = []
    ranges = _month_ranges(decision_times.min(), decision_times.max(), config.micro_load_chunk_days)
    reporter = ProgressReporter(f"[R03.3 micro {year}] chunks", len(ranges), every=1, enabled=progress)
    loaded_rows = 0
    for index, (chunk_start, chunk_end) in enumerate(ranges, start=1):
        chunk_decisions = decision_times[(decision_times >= chunk_start) & (decision_times <= chunk_end)]
        bars = load_micro_bars(loader, chunk_start - warmup, chunk_end)
        loaded_rows += len(bars)
        if bars.empty:
            features = pd.DataFrame(index=chunk_decisions)
        else:
            features = build_micro_decision_features(bars, chunk_decisions, config)
        parts.append(features)
        reporter.update(index)
    reporter.close()
    frame = pd.concat(parts).reindex(decision_times) if parts else pd.DataFrame(index=decision_times)
    feature_columns = tuple(frame.columns)
    if not feature_columns:
        raise RuntimeError(f"R03.3 {config.micro_timeframe} produced zero micro features for {year}")
    coverage = float(frame.notna().all(axis=1).mean())
    if coverage < config.minimum_micro_coverage:
        raise RuntimeError(
            f"R03.3 {config.micro_timeframe} feature coverage too low for {year}: {coverage:.2%}"
        )
    frame = frame.ffill().fillna(0.0)
    temp = target.with_name(target.name + ".part")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=True)
    np.save(temp / "decision_times_ns.npy", np.asarray(shard.decision_times_ns, dtype=np.int64), allow_pickle=False)
    np.save(temp / "features.npy", frame.to_numpy(dtype=np.float32, copy=True), allow_pickle=False)
    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "year": year,
        "rows": int(len(frame)),
        "source_timeframe": config.micro_timeframe,
        "feature_columns": list(feature_columns),
        "coverage": coverage,
        "loaded_rows": int(loaded_rows),
        "base_shard": str(base_path),
    }
    (temp / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if target.exists():
        shutil.rmtree(target)
    temp.replace(target)
    return target


def build_micro_caches(
    base_paths: list[Path],
    loader: OKXTradeBarLoader,
    config: FutureProcessForecastConfig,
    *,
    force_rebuild: bool = False,
    progress: bool = True,
) -> list[Path]:
    config.micro_cache_path.mkdir(parents=True, exist_ok=True)
    eligible: list[Path] = []
    for path in base_paths:
        shard = load_year_shard(path)
        year = pd.to_datetime(np.asarray(shard.decision_times_ns[:1], dtype=np.int64))[0].year
        if year <= pd.Timestamp(config.research_end).year:
            eligible.append(path)
    outputs: list[Path] = []
    for path in eligible:
        outputs.append(
            build_micro_year_cache(
                path,
                loader,
                config,
                force_rebuild=force_rebuild,
                progress=progress,
            )
        )
    return outputs


@dataclass(frozen=True)
class MicroYearShard:
    path: Path
    decision_times_ns: np.ndarray
    features: np.ndarray
    feature_columns: tuple[str, ...]
    coverage: float


def load_micro_year_shard(path: str | Path) -> MicroYearShard:
    target = Path(path)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != CACHE_SCHEMA_VERSION:
        raise RuntimeError(f"unsupported R03.3 micro cache: {target}")
    return MicroYearShard(
        path=target,
        decision_times_ns=np.load(target / "decision_times_ns.npy", mmap_mode="r"),
        features=np.load(target / "features.npy", mmap_mode="r"),
        feature_columns=tuple(manifest["feature_columns"]),
        coverage=float(manifest["coverage"]),
    )
