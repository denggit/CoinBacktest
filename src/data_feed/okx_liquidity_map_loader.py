#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read offline OKX Books + Raw Trades liquidity-map artifacts.

This is the public data-feed boundary for research, backtests and analyze_tool.
Raw Books/Trades are prebuilt once by ``tools/prebuild_okx_offline_liquidity_map.py``;
consumers read compact day-partitioned artifacts through this loader instead of
re-scanning the source archives.

Timestamp policy
----------------
Artifacts are stored in exchange UTC epoch milliseconds.  By default this
loader exposes project-local naive timestamps using ``config.loader.TIMEZONE``
(currently UTC+8), matching the existing OHLCV loaders.  Strategy-facing
features are indexed by ``available_time`` so same-bucket information is never
silently made available early.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

try:
    from config.loader import TIMEZONE
except ImportError:  # pragma: no cover
    TIMEZONE = "+8"

from src.liquidity_map.aggregation import aggregate_heatmap_cells, timeframe_to_seconds
from src.liquidity_map.store import LiquidityFeatureStore
from src.data_feed.okx_liquidity_period_end_cache import OKXLiquidityPeriodEndCache


def _timezone_offset() -> pd.Timedelta:
    text = str(TIMEZONE).strip()
    if text.startswith("+"):
        return pd.Timedelta(hours=float(text[1:] or 0))
    if text.startswith("-"):
        return -pd.Timedelta(hours=float(text[1:] or 0))
    return pd.Timedelta(0)


def _project_naive_to_utc(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        return ts.tz_convert("UTC")
    return (ts - _timezone_offset()).tz_localize("UTC")


def _ms_to_utc(values: pd.Series) -> pd.Series:
    return pd.to_datetime(pd.to_numeric(values, errors="coerce"), unit="ms", utc=True)


def _utc_to_project_naive(values: pd.Series) -> pd.Series:
    return _ms_to_utc(values).dt.tz_convert(None) + _timezone_offset()


@dataclass(frozen=True)
class LiquidityMapCoverage:
    day: str
    features: int
    heatmap_cells: int
    metadata: str


class OKXLiquidityMapLoader:
    """Load compact causal liquidity features and display heatmap cells."""

    def __init__(
        self,
        *,
        symbol: str = "ETH-USDT-SWAP",
        books_depth: int = 400,
        data_dir: str | None = None,
    ) -> None:
        self.symbol = symbol
        self.books_depth = int(books_depth)
        self.store = LiquidityFeatureStore(symbol=symbol, books_depth=books_depth, data_dir=data_dir)
        self.period_end_cache = OKXLiquidityPeriodEndCache(self.store)

    def coverage(self) -> list[LiquidityMapCoverage]:
        return [LiquidityMapCoverage(**item) for item in self.store.coverage()]

    def has_day(self, day: str | date) -> bool:
        return self.store.has_day(day)

    def metadata(self, day: str | date) -> dict[str, Any]:
        return self.store.load_metadata(day)

    def load_features(
        self,
        start: Any,
        end: Any,
        *,
        project_time: bool = True,
        index_mode: str = "available_time",
        valid_only: bool = False,
    ) -> pd.DataFrame:
        """Load strategy-facing features.

        ``index_mode='available_time'`` is the safe default.  ``bucket_end`` is
        useful only for diagnostics/visualization and should not be used by a
        strategy unless it separately enforces ``available_time <= decision``.
        """

        if index_mode not in {"available_time", "bucket_end", "bucket_start", "none"}:
            raise ValueError("index_mode must be available_time/bucket_end/bucket_start/none")
        start_utc = _project_naive_to_utc(start) if project_time else pd.Timestamp(start)
        end_utc = _project_naive_to_utc(end) if project_time else pd.Timestamp(end)
        if start_utc.tzinfo is None:
            start_utc = start_utc.tz_localize("UTC")
        else:
            start_utc = start_utc.tz_convert("UTC")
        if end_utc.tzinfo is None:
            end_utc = end_utc.tz_localize("UTC")
        else:
            end_utc = end_utc.tz_convert("UTC")
        # Query a small lead-in because an available-time row may summarize a
        # bucket ending just before the requested decision window.
        frame = self.store.load_features(start_utc - pd.Timedelta(minutes=5), end_utc)
        if frame.empty:
            return frame
        for source, label in (
            ("bucket_start_ms", "bucket_start"),
            ("bucket_end_ms", "bucket_end"),
            ("available_time_ms", "available_time"),
        ):
            frame[f"{label}_utc"] = _ms_to_utc(frame[source])
            frame[label] = _utc_to_project_naive(frame[source]) if project_time else frame[f"{label}_utc"]
        start_cmp = pd.Timestamp(start)
        end_cmp = pd.Timestamp(end)
        if not project_time:
            if start_cmp.tzinfo is None:
                start_cmp = start_cmp.tz_localize("UTC")
            if end_cmp.tzinfo is None:
                end_cmp = end_cmp.tz_localize("UTC")
        mask = (frame["available_time"] >= start_cmp) & (frame["available_time"] <= end_cmp)
        frame = frame.loc[mask].copy()
        if valid_only:
            frame = frame.loc[pd.to_numeric(frame["book_valid"], errors="coerce").fillna(0).astype(bool)]
        if index_mode != "none":
            frame = frame.set_index(index_mode, drop=False)
            frame.index.name = "timestamp"
        return frame.sort_index() if index_mode != "none" else frame.sort_values("available_time_ms")

    def load_heatmap(
        self,
        start: Any,
        end: Any,
        *,
        project_time: bool = True,
    ) -> pd.DataFrame:
        """Load price-time heatmap cells for a display or event study."""

        frames = list(self.iter_heatmap_days(start, end, project_time=project_time))
        if not frames:
            return pd.DataFrame(columns=list(self.store.HEATMAP_DTYPES))
        price_steps = {float(frame.attrs.get("price_step", 1.0)) for frame in frames}
        heatmap_seconds = {int(frame.attrs.get("heatmap_seconds", 60)) for frame in frames}
        if len(price_steps) > 1:
            raise ValueError(f"liquidity-map range mixes incompatible price_step values: {sorted(price_steps)}")
        if len(heatmap_seconds) > 1:
            raise ValueError(
                f"liquidity-map range mixes incompatible heatmap_seconds values: {sorted(heatmap_seconds)}"
            )
        out = pd.concat(frames, ignore_index=True).sort_values(
            ["bucket_start_ms", "side_code", "price_index"]
        ).reset_index(drop=True)
        out.attrs["price_step"] = next(iter(price_steps))
        out.attrs["heatmap_seconds"] = next(iter(heatmap_seconds))
        return out

    def load_heatmap_aggregated(
        self,
        start: Any,
        end: Any,
        *,
        timeframe: str | int = "1m",
        price_step: float | None = None,
        project_time: bool = True,
    ) -> pd.DataFrame:
        """Load one canonical heatmap and aggregate it at query time.

        No per-timeframe artifacts are created. Depth is time-weighted averaged;
        book/trade flow fields are summed. This is the shared path for
        analyze_tool and future event studies/backtests.
        """

        target_seconds = timeframe_to_seconds(timeframe)
        target_ms = target_seconds * 1000
        start_utc = _project_naive_to_utc(start) if project_time else pd.Timestamp(start)
        end_utc = _project_naive_to_utc(end) if project_time else pd.Timestamp(end)
        if start_utc.tzinfo is None:
            start_utc = start_utc.tz_localize("UTC")
        else:
            start_utc = start_utc.tz_convert("UTC")
        if end_utc.tzinfo is None:
            end_utc = end_utc.tz_localize("UTC")
        else:
            end_utc = end_utc.tz_convert("UTC")
        start_ms = int(start_utc.timestamp() * 1000)
        end_ms = int(end_utc.timestamp() * 1000)
        aligned_start_ms = (start_ms // target_ms) * target_ms
        aligned_end_ms = ((end_ms + target_ms - 1) // target_ms) * target_ms
        if aligned_end_ms <= aligned_start_ms:
            aligned_end_ms = aligned_start_ms + target_ms
        aligned_start_utc = pd.to_datetime(aligned_start_ms, unit="ms", utc=True)
        aligned_end_utc = pd.to_datetime(aligned_end_ms, unit="ms", utc=True)
        if project_time:
            aligned_start = aligned_start_utc.tz_convert(None) + _timezone_offset()
            aligned_end = aligned_end_utc.tz_convert(None) + _timezone_offset()
        else:
            aligned_start, aligned_end = aligned_start_utc, aligned_end_utc
        frame = self.load_heatmap(aligned_start, aligned_end, project_time=project_time)
        if frame.empty:
            frame.attrs["heatmap_seconds"] = target_seconds
            return frame
        source_step = float(frame.attrs.get("price_step", 1.0))
        out = aggregate_heatmap_cells(
            frame,
            target_seconds=target_seconds,
            source_price_step=source_step,
            target_price_step=price_step,
        )
        out = out.loc[
            (out["bucket_start_ms"] < end_ms) & (out["bucket_end_ms"] > start_ms)
        ].copy()
        out.attrs.update(frame.attrs)
        out.attrs["source_heatmap_seconds"] = int(frame.attrs.get("heatmap_seconds", 60))
        out.attrs["heatmap_seconds"] = target_seconds
        out.attrs["price_step"] = float(
            max(source_step, source_step if price_step is None else float(price_step))
        )
        out["start_timestamp_utc"] = _ms_to_utc(out["bucket_start_ms"])
        out["end_timestamp_utc"] = _ms_to_utc(out["bucket_end_ms"])
        if project_time:
            out["start_timestamp"] = _utc_to_project_naive(out["bucket_start_ms"])
            out["end_timestamp"] = _utc_to_project_naive(out["bucket_end_ms"])
        else:
            out["start_timestamp"] = out["start_timestamp_utc"]
            out["end_timestamp"] = out["end_timestamp_utc"]
        return out

    def iter_heatmap_days_raw(
        self,
        start: Any,
        end: Any,
    ):
        """Yield canonical UTC heatmap frames without display decoration.

        Heavy prebuild jobs need only the persisted numeric columns.  The
        regular :meth:`iter_heatmap_days` path also materializes side labels,
        prices and four timestamp columns for UI consumers; avoiding those
        allocations materially reduces per-day CPU and memory while preserving
        the exact canonical rows and ordering.
        """

        start_utc = pd.Timestamp(start)
        end_utc = pd.Timestamp(end)
        if start_utc.tzinfo is None:
            start_utc = start_utc.tz_localize("UTC")
        else:
            start_utc = start_utc.tz_convert("UTC")
        if end_utc.tzinfo is None:
            end_utc = end_utc.tz_localize("UTC")
        else:
            end_utc = end_utc.tz_convert("UTC")
        price_step = self._price_step_for_range(start_utc, end_utc)
        heatmap_seconds = self._heatmap_seconds_for_range(start_utc, end_utc)
        for day in pd.date_range(start_utc.normalize(), end_utc.normalize(), freq="D"):
            day_start = max(start_utc, day)
            day_end = min(end_utc, day + pd.Timedelta(days=1))
            frame = self.store.load_heatmap(day_start, day_end)
            if frame.empty:
                continue
            frame.attrs["price_step"] = price_step
            frame.attrs["heatmap_seconds"] = heatmap_seconds
            frame.attrs["utc_day"] = day.date().isoformat()
            frame.attrs["canonical_numeric_only"] = True
            yield frame

    def iter_heatmap_days(
        self,
        start: Any,
        end: Any,
        *,
        project_time: bool = True,
    ):
        """Yield decorated heatmap frames one UTC day at a time.

        This is the memory-safe interface used by analyze_tool for multi-day
        requests.  No month-wide raw heatmap DataFrame is materialized.
        """

        start_utc = _project_naive_to_utc(start) if project_time else pd.Timestamp(start)
        end_utc = _project_naive_to_utc(end) if project_time else pd.Timestamp(end)
        if start_utc.tzinfo is None:
            start_utc = start_utc.tz_localize("UTC")
        else:
            start_utc = start_utc.tz_convert("UTC")
        if end_utc.tzinfo is None:
            end_utc = end_utc.tz_localize("UTC")
        else:
            end_utc = end_utc.tz_convert("UTC")
        price_step = self._price_step_for_range(start_utc, end_utc)
        heatmap_seconds = self._heatmap_seconds_for_range(start_utc, end_utc)
        for day in pd.date_range(start_utc.normalize(), end_utc.normalize(), freq="D"):
            day_start = max(start_utc, day)
            day_end = min(end_utc, day + pd.Timedelta(days=1))
            frame = self.store.load_heatmap(day_start, day_end)
            if frame.empty:
                continue
            frame["side"] = frame["side_code"].map({1: "bid", -1: "ask"}).fillna("unknown")
            frame["price_low"] = pd.to_numeric(frame["price_index"], errors="coerce") * price_step
            frame["price_high"] = frame["price_low"] + price_step
            frame["start_timestamp_utc"] = _ms_to_utc(frame["bucket_start_ms"])
            frame["end_timestamp_utc"] = _ms_to_utc(frame["bucket_end_ms"])
            if project_time:
                frame["start_timestamp"] = _utc_to_project_naive(frame["bucket_start_ms"])
                frame["end_timestamp"] = _utc_to_project_naive(frame["bucket_end_ms"])
                start_cmp, end_cmp = pd.Timestamp(start), pd.Timestamp(end)
            else:
                frame["start_timestamp"] = frame["start_timestamp_utc"]
                frame["end_timestamp"] = frame["end_timestamp_utc"]
                start_cmp, end_cmp = start_utc, end_utc
            frame = frame.loc[
                (frame["start_timestamp"] < end_cmp) & (frame["end_timestamp"] > start_cmp)
            ].copy()
            if frame.empty:
                continue
            frame = frame.sort_values(["bucket_start_ms", "side_code", "price_index"]).reset_index(drop=True)
            frame.attrs["price_step"] = price_step
            frame.attrs["heatmap_seconds"] = heatmap_seconds
            frame.attrs["utc_day"] = day.date().isoformat()
            yield frame

    def align_features_to_times(
        self,
        times: Any,
        *,
        project_time: bool = True,
        tolerance: str | pd.Timedelta = "5m",
        columns: list[str] | tuple[str, ...] | None = None,
    ) -> pd.DataFrame:
        """Select only the latest causal feature row needed by each timestamp.

        This avoids materializing every 1-second feature row and every feature
        column for a long Analyze Tool request.  The method first reads only
        ``available_time_ms`` to locate rows, then reopens each touched daily NPZ
        and extracts the requested columns at those exact indices.
        """

        requested = pd.DatetimeIndex(pd.to_datetime(times))
        if requested.empty:
            return pd.DataFrame(index=requested)
        tolerance_ms = int(pd.Timedelta(tolerance).total_seconds() * 1000)
        if tolerance_ms < 0:
            raise ValueError("tolerance must be >= 0")
        query_ms = []
        for value in requested:
            ts = pd.Timestamp(value)
            if project_time:
                utc = _project_naive_to_utc(ts)
            else:
                utc = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
            query_ms.append(int(utc.timestamp() * 1000))
        query_ms_array = pd.Series(query_ms, dtype="int64").to_numpy()
        start_utc = pd.to_datetime(int(query_ms_array.min()) - tolerance_ms, unit="ms", utc=True)
        end_utc = pd.to_datetime(int(query_ms_array.max()), unit="ms", utc=True)

        available_parts = []
        day_parts = []
        row_parts = []
        day_paths = []
        for day_number, day in enumerate(pd.date_range(start_utc.normalize(), end_utc.normalize(), freq="D")):
            path = self.store.paths_for_day(day.date()).features
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as data:
                if "available_time_ms" not in data.files:
                    continue
                available = np.asarray(data["available_time_ms"], dtype=np.int64)
            mask = (available >= int(start_utc.timestamp() * 1000)) & (available <= int(end_utc.timestamp() * 1000))
            positions = np.flatnonzero(mask)
            if len(positions) == 0:
                continue
            compact_day = len(day_paths)
            day_paths.append(path)
            available_parts.append(available[positions])
            day_parts.append(np.full(len(positions), compact_day, dtype=np.int16))
            row_parts.append(positions.astype(np.int64))
        selected_columns = list(columns or self.store.FEATURE_DTYPES.keys())
        for required in ("available_time_ms",):
            if required not in selected_columns:
                selected_columns.append(required)
        result = pd.DataFrame(index=requested)
        available_dtype = "datetime64[ns]" if project_time else "datetime64[ns, UTC]"
        if not available_parts:
            for name in selected_columns:
                result[name] = np.nan
            result["available_time"] = pd.Series(pd.NaT, index=result.index, dtype=available_dtype)
            return result

        available_all = np.concatenate(available_parts)
        day_all = np.concatenate(day_parts)
        row_all = np.concatenate(row_parts)
        order = np.argsort(available_all, kind="stable")
        available_all = available_all[order]
        day_all = day_all[order]
        row_all = row_all[order]
        positions = np.searchsorted(available_all, query_ms_array, side="right") - 1
        valid = positions >= 0
        clipped = np.clip(positions, 0, len(available_all) - 1)
        valid &= (query_ms_array - available_all[clipped]) <= tolerance_ms

        for name in selected_columns:
            result[name] = np.nan
        result["available_time"] = pd.Series(pd.NaT, index=result.index, dtype=available_dtype)
        if not bool(valid.any()):
            return result
        query_rows = np.flatnonzero(valid)
        selected_positions = clipped[valid]
        selected_days = day_all[selected_positions]
        selected_source_rows = row_all[selected_positions]
        selected_available = available_all[selected_positions]
        for day_number in np.unique(selected_days):
            target_mask = selected_days == day_number
            target_query_rows = query_rows[target_mask]
            source_rows = selected_source_rows[target_mask]
            path = day_paths[int(day_number)]
            with np.load(path, allow_pickle=False) as data:
                for name in selected_columns:
                    if name not in data.files:
                        continue
                    values = np.asarray(data[name])[source_rows]
                    result.iloc[target_query_rows, result.columns.get_loc(name)] = values
        available_utc = pd.to_datetime(selected_available, unit="ms", utc=True)
        available_values = available_utc.tz_convert(None) + _timezone_offset() if project_time else available_utc
        result.iloc[query_rows, result.columns.get_loc("available_time")] = list(available_values)
        result.index = requested
        return result

    def iter_period_end_snapshot_days(
        self,
        start: Any,
        end: Any,
        *,
        timeframe: str | int = "15m",
        price_step: float = 1.0,
        project_time: bool = True,
    ):
        """Yield compact exact period-end snapshots from the persistent cache.

        The first request for a day builds the cache directly from the canonical
        5-second heatmap NPZ. Later requests read only the compact file. Historical
        bars use the final completed source snapshot in the bar; a live caller can
        overwrite the current incomplete bar with its newest snapshot separately.
        """

        target_seconds = timeframe_to_seconds(timeframe)
        start_utc = _project_naive_to_utc(start) if project_time else pd.Timestamp(start)
        end_utc = _project_naive_to_utc(end) if project_time else pd.Timestamp(end)
        if start_utc.tzinfo is None:
            start_utc = start_utc.tz_localize("UTC")
        else:
            start_utc = start_utc.tz_convert("UTC")
        if end_utc.tzinfo is None:
            end_utc = end_utc.tz_localize("UTC")
        else:
            end_utc = end_utc.tz_convert("UTC")
        for frame in self.period_end_cache.iter_days(
            start_utc,
            end_utc,
            target_seconds=target_seconds,
            target_price_step=float(price_step),
        ):
            if frame.empty:
                continue
            out = frame.copy()
            out["side"] = out["side_code"].map({1: "bid", -1: "ask"}).fillna("unknown")
            effective_step = float(frame.attrs.get("price_step", price_step))
            out["price_low"] = pd.to_numeric(out["price_index"], errors="coerce") * effective_step
            out["price_high"] = out["price_low"] + effective_step
            out["depth_base"] = pd.to_numeric(out["end_depth_base"], errors="coerce").fillna(0.0)
            out["depth_usd"] = pd.to_numeric(out["end_depth_usd"], errors="coerce").fillna(0.0)
            out["order_count"] = pd.to_numeric(out["end_order_count"], errors="coerce").fillna(0).round().astype("int64")
            side_max = out.groupby(["bucket_start_ms", "side_code"], observed=True)["end_depth_base"].transform("max")
            out["local_depth_ratio"] = pd.Series(
                0.0, index=out.index, dtype=float
            ).where(side_max <= 0, out["end_depth_base"] / side_max)
            out["start_timestamp_utc"] = _ms_to_utc(out["bucket_start_ms"])
            out["end_timestamp_utc"] = _ms_to_utc(out["bucket_end_ms"])
            if project_time:
                out["start_timestamp"] = _utc_to_project_naive(out["bucket_start_ms"])
                out["end_timestamp"] = _utc_to_project_naive(out["bucket_end_ms"])
            else:
                out["start_timestamp"] = out["start_timestamp_utc"]
                out["end_timestamp"] = out["end_timestamp_utc"]
            out = out.sort_values(["bucket_start_ms", "side_code", "price_index"]).reset_index(drop=True)
            out.attrs.update(frame.attrs)
            yield out

    def load_period_end_snapshots(
        self,
        start: Any,
        end: Any,
        *,
        timeframe: str | int = "15m",
        price_step: float = 1.0,
        project_time: bool = True,
    ) -> pd.DataFrame:
        frames = list(self.iter_period_end_snapshot_days(
            start, end, timeframe=timeframe, price_step=price_step, project_time=project_time
        ))
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True).sort_values(
            ["bucket_start_ms", "side_code", "price_index"]
        ).reset_index(drop=True)
        out.attrs["price_step"] = float(frames[0].attrs.get("price_step", price_step))
        out.attrs["heatmap_seconds"] = int(frames[0].attrs.get("heatmap_seconds", timeframe_to_seconds(timeframe)))
        out.attrs["cache_hits"] = int(sum(bool(frame.attrs.get("cache_hit")) for frame in frames))
        out.attrs["cache_misses"] = int(sum(not bool(frame.attrs.get("cache_hit")) for frame in frames))
        return out

    def _price_step_for_range(self, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> float:
        steps: set[float] = set()
        for day in pd.date_range(start_utc.normalize(), end_utc.normalize(), freq="D"):
            meta = self.store.load_metadata(day.date())
            value = (meta.get("config") or {}).get("price_step")
            if value is not None:
                steps.add(float(value))
        if not steps:
            return 1.0
        if len(steps) > 1:
            raise ValueError(f"liquidity-map range mixes incompatible price_step values: {sorted(steps)}")
        return next(iter(steps))

    def _heatmap_seconds_for_range(self, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> int:
        values: set[int] = set()
        for day in pd.date_range(start_utc.normalize(), end_utc.normalize(), freq="D"):
            meta = self.store.load_metadata(day.date())
            value = (meta.get("config") or {}).get("heatmap_seconds")
            if value is not None:
                values.add(int(value))
        if not values:
            return 60
        if len(values) > 1:
            raise ValueError(
                f"liquidity-map range mixes incompatible heatmap_seconds values: {sorted(values)}"
            )
        return next(iter(values))
