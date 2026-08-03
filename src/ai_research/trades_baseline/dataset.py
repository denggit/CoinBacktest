#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal sample construction using only the public ``src.data_feed`` API.

The builder intentionally never opens raw trade files or SQLite directly. The
existing ``OKXTradeBarLoader`` owns all data-location, timezone, schema, and
cache semantics. R01 only performs a light smoke check and consumes its public
1-second interface.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd

from src.data_feed.okx_trade_bar_loader import OKXTradeBarLoader
from src.research_common.progress import ProgressReporter

from .config import TradesBaselineConfig


FLOW_COLUMNS = (
    "volume",
    "trades_count",
    "buy_volume",
    "sell_volume",
    "notional",
    "buy_notional",
    "sell_notional",
    "buy_trades_count",
    "sell_trades_count",
    "delta_volume",
    "delta_notional",
    "large_buy_notional",
    "large_sell_notional",
    "large_delta_notional",
    "large_trades_count",
    "large_buy_trades_count",
    "large_sell_trades_count",
    "max_trade_notional",
    "max_trade_size",
)
REQUIRED_COLUMNS = ("open", "high", "low", "close", *FLOW_COLUMNS)


@dataclass(frozen=True)
class PreflightResult:
    status: str
    sample_windows: tuple[dict[str, object], ...]
    required_columns: tuple[str, ...]
    loader_class: str = "src.data_feed.okx_trade_bar_loader.OKXTradeBarLoader"

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "loader_class": self.loader_class,
            "required_columns": list(self.required_columns),
            "sample_windows": list(self.sample_windows),
        }


def create_loader(config: TradesBaselineConfig, *, data_dir: str | Path | None = None) -> OKXTradeBarLoader:
    """Create the canonical public 1s loader used by all R01 work."""
    return OKXTradeBarLoader(
        symbol=config.symbol,
        timeframe=config.timeframe,
        data_dir=data_dir,
        align_with_okx_loader_timezone=True,
    )


def _ensure_naive_index(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index, errors="coerce"))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    out.index = idx
    out = out.loc[~out.index.isna()]
    return out.sort_index(kind="stable")


def _datetime_index_to_ns(index: pd.Index | pd.DatetimeIndex) -> np.ndarray:
    """Return epoch nanoseconds regardless of pandas datetime storage unit.

    Pandas may preserve SQLite/Arrow timestamps as ``datetime64[us]``. Calling
    ``.view("int64")`` on such an index returns microseconds, not nanoseconds,
    which silently breaks comparisons against nanosecond timestamps created by
    ``Timestamp.value`` or a nanosecond ``date_range``. Normalizing explicitly
    here keeps labels, cache timestamps, and live replay contracts unit-safe.
    """
    dt_index = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    if dt_index.tz is not None:
        dt_index = dt_index.tz_convert("UTC").tz_localize(None)
    return dt_index.to_numpy(dtype="datetime64[ns]", copy=False).astype(np.int64, copy=False)


def load_public_bars(
    loader: OKXTradeBarLoader,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Load existing cached 1s bars without downloading/building missing data."""
    bars = loader.fetch_data_by_date_range(
        start,
        end,
        build_missing=False,
        force_rebuild=False,
        cvd_mode="range",
    )
    return _ensure_naive_index(bars)


def run_public_loader_preflight(
    loader: OKXTradeBarLoader,
    config: TradesBaselineConfig,
    *,
    sample_dates: Iterable[str | pd.Timestamp] | None = None,
) -> PreflightResult:
    """Perform a deliberately small smoke check before research starts.

    This is not a platform-level re-audit. It verifies only that the public
    loader returns the required fields and causal timestamp ordering on a few
    representative windows. Any deeper investigation is triggered only when a
    later research result is abnormal.
    """
    if sample_dates is None:
        sample_dates = (
            config.research_start,
            "2025-06-15 12:00:00",
            "2026-06-29 12:00:00",
        )
    rows: list[dict[str, object]] = []
    status = "PASS"
    for raw_date in sample_dates:
        start = pd.Timestamp(raw_date)
        end = start + pd.Timedelta(hours=1) - pd.Timedelta(microseconds=1)
        bars = load_public_bars(loader, start, end)
        missing = sorted(set(REQUIRED_COLUMNS) - set(bars.columns))
        finite_ok = True
        ohlc_ok = True
        nonnegative_ok = True
        if bars.empty:
            status = "BLOCKED"
            finite_ok = ohlc_ok = nonnegative_ok = False
        else:
            numeric = bars[[col for col in REQUIRED_COLUMNS if col in bars.columns]].apply(
                pd.to_numeric, errors="coerce"
            )
            finite_ok = bool(np.isfinite(numeric.to_numpy(dtype=float, copy=False)).all())
            if all(col in bars.columns for col in ("open", "high", "low", "close")):
                ohlc_ok = bool(
                    (bars["high"] >= bars[["open", "close"]].max(axis=1)).all()
                    and (bars["low"] <= bars[["open", "close"]].min(axis=1)).all()
                    and (bars["high"] >= bars["low"]).all()
                )
            nonnegative_cols = [
                col
                for col in FLOW_COLUMNS
                if col in bars.columns and not col.startswith("delta") and col != "large_delta_notional"
            ]
            nonnegative_ok = bool((bars[nonnegative_cols] >= 0).all().all()) if nonnegative_cols else False
            if missing or not finite_ok or not ohlc_ok or not nonnegative_ok:
                status = "BLOCKED"
        rows.append(
            {
                "start": str(start),
                "end": str(end),
                "rows": int(len(bars)),
                "missing_columns": missing,
                "monotonic": bool(bars.index.is_monotonic_increasing),
                "unique": bool(bars.index.is_unique),
                "finite": finite_ok,
                "ohlc_valid": ohlc_ok,
                "nonnegative_flows": nonnegative_ok,
            }
        )
        if bars.empty or not bars.index.is_monotonic_increasing or not bars.index.is_unique:
            status = "BLOCKED"
    return PreflightResult(status=status, sample_windows=tuple(rows), required_columns=REQUIRED_COLUMNS)


def feature_columns(config: TradesBaselineConfig) -> list[str]:
    names: list[str] = []
    for window in config.return_windows_seconds:
        names.append(f"ret_{window}s")
    for window in config.feature_windows_seconds:
        names.extend(
            [
                f"rv_{window}s",
                f"notional_sum_{window}s",
                f"trades_sum_{window}s",
                f"imbalance_{window}s",
                f"buy_ratio_{window}s",
                f"large_imbalance_{window}s",
                f"max_trade_notional_{window}s",
                f"impact_efficiency_{window}s",
                f"absorption_{window}s",
            ]
        )
    names.extend(("avg_trade_size_1s", "taker_buy_ratio_1s", "range_pct_1s"))
    return names


def label_columns(config: TradesBaselineConfig) -> list[str]:
    names: list[str] = []
    for latency in config.latency_scenarios_seconds:
        latency_ms = int(round(latency * 1000))
        for horizon in config.horizons_seconds:
            names.append(f"gross_ret_h{horizon}_lat{latency_ms}")
    for horizon in config.horizons_seconds:
        names.extend((f"mfe_h{horizon}", f"mae_h{horizon}"))
    return names


def cache_signature(config: TradesBaselineConfig) -> str:
    payload = {
        "schema_version": 3,
        "symbol": config.symbol,
        "timeframe": config.timeframe,
        "decision_interval_seconds": config.decision_interval_seconds,
        "feature_windows_seconds": list(config.feature_windows_seconds),
        "return_windows_seconds": list(config.return_windows_seconds),
        "horizons_seconds": list(config.horizons_seconds),
        "base_latency_seconds": config.base_latency_seconds,
        "latency_scenarios_seconds": list(config.latency_scenarios_seconds),
        "feature_columns": feature_columns(config),
        "label_columns": label_columns(config),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = denominator.astype(float)
    out = numerator.astype(float) / den.where(den.abs() > 1e-12)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_one_second_grid(bars: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    bars = _ensure_naive_index(bars)
    if bars.empty:
        return pd.DataFrame()
    grid_index = pd.date_range(start.floor("s"), end.ceil("s"), freq="1s", inclusive="left")
    source = bars.loc[(bars.index >= grid_index[0]) & (bars.index <= grid_index[-1])]
    grid = source.reindex(grid_index)
    last_price = pd.to_numeric(grid["close"], errors="coerce").ffill()
    for col in ("open", "high", "low", "close"):
        grid[col] = pd.to_numeric(grid[col], errors="coerce").fillna(last_price)
    for col in FLOW_COLUMNS:
        if col not in grid.columns:
            grid[col] = 0.0
        grid[col] = pd.to_numeric(grid[col], errors="coerce").fillna(0.0)
    grid.index.name = "timestamp"
    return grid


def _causal_features(grid: pd.DataFrame, config: TradesBaselineConfig) -> pd.DataFrame:
    close = grid["close"].astype(float)
    log_return = np.log(close.where(close > 0)).diff().fillna(0.0)
    features: dict[str, pd.Series] = {}
    for window in config.return_windows_seconds:
        features[f"ret_{window}s"] = close.pct_change(window, fill_method=None)
    for window in config.feature_windows_seconds:
        min_periods = max(2, window // 2)
        notional_sum = grid["notional"].rolling(window, min_periods=min_periods).sum()
        trades_sum = grid["trades_count"].rolling(window, min_periods=min_periods).sum()
        delta_sum = grid["delta_notional"].rolling(window, min_periods=min_periods).sum()
        buy_sum = grid["buy_notional"].rolling(window, min_periods=min_periods).sum()
        large_delta = grid["large_delta_notional"].rolling(window, min_periods=min_periods).sum()
        large_total = (
            grid["large_buy_notional"].rolling(window, min_periods=min_periods).sum()
            + grid["large_sell_notional"].rolling(window, min_periods=min_periods).sum()
        )
        ret = close.pct_change(window, fill_method=None)
        imbalance = _safe_divide(delta_sum, notional_sum)
        features[f"rv_{window}s"] = log_return.rolling(window, min_periods=min_periods).std(ddof=0)
        features[f"notional_sum_{window}s"] = np.log1p(notional_sum.clip(lower=0))
        features[f"trades_sum_{window}s"] = np.log1p(trades_sum.clip(lower=0))
        features[f"imbalance_{window}s"] = imbalance
        features[f"buy_ratio_{window}s"] = _safe_divide(buy_sum, notional_sum)
        features[f"large_imbalance_{window}s"] = _safe_divide(large_delta, large_total)
        features[f"max_trade_notional_{window}s"] = np.log1p(
            grid["max_trade_notional"].rolling(window, min_periods=min_periods).max().clip(lower=0)
        )
        raw_impact = _safe_divide(ret, notional_sum / 1_000_000.0)
        features[f"impact_efficiency_{window}s"] = np.sign(raw_impact) * np.log1p(raw_impact.abs())
        features[f"absorption_{window}s"] = -(np.sign(ret.fillna(0.0)) * imbalance)
    features["avg_trade_size_1s"] = _safe_divide(grid["volume"], grid["trades_count"])
    features["taker_buy_ratio_1s"] = _safe_divide(grid["buy_volume"], grid["volume"])
    features["range_pct_1s"] = _safe_divide(grid["high"] - grid["low"], close)
    out = pd.DataFrame(features, index=grid.index)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _future_extreme(series: pd.Series, horizon_seconds: int, op: str) -> pd.Series:
    reversed_series = series.iloc[::-1]
    roller = reversed_series.rolling(horizon_seconds + 1, min_periods=1)
    if op == "max":
        result = roller.max()
    elif op == "min":
        result = roller.min()
    else:  # pragma: no cover - internal contract
        raise ValueError(op)
    return result.iloc[::-1]


def _ceil_latency_seconds(latency: float) -> int:
    # Conservative 1s-bar execution: any positive sub-second latency enters at
    # the next complete second, never at the just-observed second's open.
    return max(1, int(math.ceil(float(latency))))


def _labels_from_trade_path(
    bars: pd.DataFrame,
    grid: pd.DataFrame,
    decision_times: pd.DatetimeIndex,
    config: TradesBaselineConfig,
) -> pd.DataFrame:
    actual = bars.loc[bars.index.notna()].sort_index(kind="stable")
    actual = actual.loc[~actual.index.duplicated(keep="last")]
    trade_times = _datetime_index_to_ns(actual.index)
    trade_open = actual["open"].to_numpy(dtype=float)
    decision_ns = _datetime_index_to_ns(decision_times)
    labels: dict[str, np.ndarray] = {}

    base_entry_positions: np.ndarray | None = None
    base_entry_prices: np.ndarray | None = None
    for latency in config.latency_scenarios_seconds:
        latency_seconds = _ceil_latency_seconds(latency)
        entry_targets = decision_ns + latency_seconds * 1_000_000_000
        entry_positions = np.searchsorted(trade_times, entry_targets, side="left")
        entry_valid = entry_positions < len(trade_times)
        safe_entry_pos = np.minimum(entry_positions, max(0, len(trade_times) - 1))
        entry_prices = np.full(len(decision_times), np.nan, dtype=float)
        if len(trade_times):
            entry_prices[entry_valid] = trade_open[safe_entry_pos[entry_valid]]
        entry_actual_ns = np.full(len(decision_times), np.iinfo(np.int64).max, dtype=np.int64)
        if len(trade_times):
            entry_actual_ns[entry_valid] = trade_times[safe_entry_pos[entry_valid]]
        if latency == config.base_latency_seconds:
            base_entry_positions = safe_entry_pos.copy()
            base_entry_positions[~entry_valid] = -1
            base_entry_prices = entry_prices.copy()

        latency_ms = int(round(latency * 1000))
        for horizon in config.horizons_seconds:
            exit_targets = np.full(len(decision_times), np.iinfo(np.int64).max, dtype=np.int64)
            exit_targets[entry_valid] = entry_actual_ns[entry_valid] + int(horizon) * 1_000_000_000
            exit_positions = np.searchsorted(trade_times, exit_targets, side="left")
            valid = entry_valid & (exit_positions < len(trade_times)) & np.isfinite(entry_prices) & (entry_prices > 0)
            safe_exit_pos = np.minimum(exit_positions, max(0, len(trade_times) - 1))
            gross = np.full(len(decision_times), np.nan, dtype=np.float32)
            if len(trade_times):
                exit_prices = trade_open[safe_exit_pos]
                gross[valid] = (exit_prices[valid] / entry_prices[valid] - 1.0).astype(np.float32)
            labels[f"gross_ret_h{horizon}_lat{latency_ms}"] = gross

    # MFE/MAE are diagnostics for the frozen base-latency entry only. They are
    # not model inputs and therefore cannot leak into features.
    if base_entry_positions is None or base_entry_prices is None:
        raise RuntimeError("base latency labels were not constructed")
    grid_start_ns = int(grid.index[0].value)
    grid_positions = ((_datetime_index_to_ns(actual.index) - grid_start_ns) // 1_000_000_000).astype(np.int64)
    for horizon in config.horizons_seconds:
        future_high = _future_extreme(grid["high"].astype(float), horizon, "max").to_numpy(dtype=float)
        future_low = _future_extreme(grid["low"].astype(float), horizon, "min").to_numpy(dtype=float)
        mfe = np.full(len(decision_times), np.nan, dtype=np.float32)
        mae = np.full(len(decision_times), np.nan, dtype=np.float32)
        valid_entry = base_entry_positions >= 0
        if len(grid_positions):
            path_pos = np.zeros(len(decision_times), dtype=np.int64)
            path_pos[valid_entry] = grid_positions[base_entry_positions[valid_entry]]
            path_valid = valid_entry & (path_pos >= 0) & (path_pos < len(grid)) & np.isfinite(base_entry_prices)
            mfe[path_valid] = (future_high[path_pos[path_valid]] / base_entry_prices[path_valid] - 1.0).astype(np.float32)
            mae[path_valid] = (future_low[path_pos[path_valid]] / base_entry_prices[path_valid] - 1.0).astype(np.float32)
        labels[f"mfe_h{horizon}"] = mfe
        labels[f"mae_h{horizon}"] = mae

    return pd.DataFrame(labels, index=decision_times)


def build_day_samples(
    loader: OKXTradeBarLoader,
    day: pd.Timestamp,
    config: TradesBaselineConfig,
) -> pd.DataFrame:
    """Build one project-local day of causal 5-second decision samples."""
    day_start = pd.Timestamp(day).normalize()
    day_end = day_start + pd.Timedelta(days=1)
    load_start = day_start - pd.Timedelta(seconds=config.max_history_seconds + 5)
    load_end = day_end + pd.Timedelta(seconds=config.max_future_seconds)
    bars = load_public_bars(loader, load_start, load_end)
    if bars.empty:
        return pd.DataFrame()
    missing = sorted(set(REQUIRED_COLUMNS) - set(bars.columns))
    if missing:
        raise RuntimeError(f"public 1s loader missing required columns: {missing}")

    grid = _build_one_second_grid(bars, load_start, load_end + pd.Timedelta(seconds=1))
    if grid.empty:
        return pd.DataFrame()
    feature_frame = _causal_features(grid, config)
    decision_times = pd.date_range(
        day_start,
        day_end,
        freq=f"{config.decision_interval_seconds}s",
        inclusive="left",
    )
    feature_times = decision_times - pd.Timedelta(seconds=1)
    sampled_features = feature_frame.reindex(feature_times)
    sampled_features.index = decision_times
    sampled_features.index.name = "decision_time"
    labels = _labels_from_trade_path(bars, grid, decision_times, config)
    out = sampled_features.join(labels, how="left")
    out.insert(0, "decision_time_ns", _datetime_index_to_ns(out.index))
    # Drop only samples that lack causal history or the longest base label.
    required_feature_cols = feature_columns(config)
    base_label = f"gross_ret_h{max(config.horizons_seconds)}_lat{int(config.base_latency_seconds * 1000)}"
    before_drop = len(out)
    feature_valid_rows = int(out[required_feature_cols].notna().all(axis=1).sum())
    label_valid_rows = int(out[base_label].notna().sum())
    out = out.dropna(subset=[*required_feature_cols, base_label])
    if before_drop > 0 and out.empty:
        raise RuntimeError(
            "R01 sample construction produced zero valid rows for "
            f"{day_start:%Y-%m-%d}: bars={len(bars)} decisions={before_drop} "
            f"feature_valid={feature_valid_rows} label_valid={label_valid_rows} "
            f"bar_index_dtype={bars.index.dtype} decision_index_dtype={decision_times.dtype}. "
            "This usually indicates timestamp-unit or cache-boundary misalignment."
        )
    for col in required_feature_cols + label_columns(config):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out


def iter_month_starts(start: pd.Timestamp, end: pd.Timestamp) -> Iterator[pd.Timestamp]:
    current = pd.Timestamp(start).normalize().replace(day=1)
    final = pd.Timestamp(end).normalize().replace(day=1)
    while current <= final:
        yield current
        current = current + pd.offsets.MonthBegin(1)


def month_cache_path(config: TradesBaselineConfig, month_start: pd.Timestamp) -> Path:
    return config.cache_path / f"samples_{month_start:%Y_%m}"


def _write_month_arrays(path: Path, frame: pd.DataFrame, config: TradesBaselineConfig) -> None:
    features = feature_columns(config)
    labels = label_columns(config)
    temp = path.with_name(path.name + ".part")
    if temp.exists():
        shutil.rmtree(temp)
    temp.mkdir(parents=True, exist_ok=True)
    np.save(temp / "timestamps_ns.npy", _datetime_index_to_ns(frame.index), allow_pickle=False)
    np.save(temp / "features.npy", frame[features].to_numpy(dtype=np.float32, copy=True), allow_pickle=False)
    np.save(temp / "labels.npy", frame[labels].to_numpy(dtype=np.float32, copy=True), allow_pickle=False)
    manifest = {
        "schema_version": 3,
        "cache_signature": cache_signature(config),
        "rows": int(len(frame)),
        "feature_columns": features,
        "label_columns": labels,
        "start": str(frame.index.min()),
        "end": str(frame.index.max()),
    }
    (temp / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if path.exists():
        shutil.rmtree(path)
    temp.replace(path)


def build_monthly_sample_cache(
    loader: OKXTradeBarLoader,
    config: TradesBaselineConfig,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    force_rebuild: bool = False,
    progress: bool = True,
) -> list[Path]:
    """Build resumable monthly memory-mapped feature/label shards.

    The cache is deliberately uncompressed NumPy rather than gzip/pickle. The
    research stage reads the same months many times across folds, horizons, and
    models; memory mapping is materially faster and avoids repeated full-file
    decompression.
    """
    config.cache_path.mkdir(parents=True, exist_ok=True)
    months = list(iter_month_starts(start, end))
    reporter = ProgressReporter("[R01 cache] months", len(months), every=1, enabled=progress)
    outputs: list[Path] = []
    for month_index, month_start in enumerate(months, start=1):
        path = month_cache_path(config, month_start)
        month_end = min(month_start + pd.offsets.MonthBegin(1), end + pd.Timedelta(seconds=1))
        manifest_path = path / "manifest.json"
        compatible = False
        if manifest_path.exists() and not force_rebuild:
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
                compatible = existing.get("cache_signature") == cache_signature(config)
            except (OSError, json.JSONDecodeError):
                compatible = False
        if compatible:
            outputs.append(path)
            reporter.update(month_index)
            continue
        day_start = max(month_start, start.normalize())
        last_day = (month_end - pd.Timedelta(seconds=1)).normalize()
        days = pd.date_range(day_start, last_day, freq="D")
        daily: list[pd.DataFrame] = []
        day_reporter = ProgressReporter(
            f"[R01 cache] {month_start:%Y-%m} days",
            len(days),
            every=1,
            enabled=progress,
        )
        for day_index, day in enumerate(days, start=1):
            frame = build_day_samples(loader, day, config)
            if not frame.empty:
                frame = frame.loc[(frame.index >= start) & (frame.index <= end)]
                if not frame.empty:
                    daily.append(frame)
            day_reporter.update(day_index)
        day_reporter.close()
        if not daily:
            raise RuntimeError(f"no R01 samples produced for {month_start:%Y-%m}")
        month_frame = pd.concat(daily).sort_index(kind="stable")
        _write_month_arrays(path, month_frame, config)
        outputs.append(path)
        reporter.update(month_index)
    reporter.close()

    manifest = {
        "schema_version": 3,
        "storage": "monthly_npy_memmap",
        "cache_signature": cache_signature(config),
        "config": config.to_dict(),
        "feature_columns": feature_columns(config),
        "label_columns": label_columns(config),
        "start": str(start),
        "end": str(end),
        "files": [path.name for path in outputs],
    }
    (config.cache_path / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return outputs


def load_month_frame(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    features = list(manifest["feature_columns"])
    labels = list(manifest["label_columns"])
    timestamps = np.load(target / "timestamps_ns.npy", mmap_mode="r")
    x = np.load(target / "features.npy", mmap_mode="r")
    y = np.load(target / "labels.npy", mmap_mode="r")
    index = pd.to_datetime(np.asarray(timestamps, dtype=np.int64))
    frame = pd.DataFrame(x, index=index, columns=features, copy=False)
    for idx, name in enumerate(labels):
        frame[name] = y[:, idx]
    frame.insert(0, "decision_time_ns", np.asarray(timestamps, dtype=np.int64))
    frame.index.name = "decision_time"
    return frame


def list_cached_months(config: TradesBaselineConfig) -> list[Path]:
    return sorted(path for path in config.cache_path.glob("samples_????_??") if (path / "manifest.json").exists())


@dataclass(frozen=True)
class MonthShard:
    path: Path
    timestamps_ns: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]

    @property
    def feature_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.feature_names)}

    @property
    def label_index(self) -> dict[str, int]:
        return {name: idx for idx, name in enumerate(self.label_names)}

    def positions(self, start: pd.Timestamp, end: pd.Timestamp) -> slice:
        start_ns = int(pd.Timestamp(start).value)
        end_ns = int(pd.Timestamp(end).value)
        left = int(np.searchsorted(self.timestamps_ns, start_ns, side="left"))
        right = int(np.searchsorted(self.timestamps_ns, end_ns, side="right"))
        return slice(left, right)


def load_month_shard(path: str | Path) -> MonthShard:
    target = Path(path)
    manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 3:
        raise RuntimeError(f"unsupported R01 cache schema in {target}")
    return MonthShard(
        path=target,
        timestamps_ns=np.load(target / "timestamps_ns.npy", mmap_mode="r"),
        features=np.load(target / "features.npy", mmap_mode="r"),
        labels=np.load(target / "labels.npy", mmap_mode="r"),
        feature_names=tuple(manifest["feature_columns"]),
        label_names=tuple(manifest["label_columns"]),
    )
