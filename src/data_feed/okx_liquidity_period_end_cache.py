#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fast exact period-end snapshot cache for offline OKX liquidity maps.

The canonical liquidity-map artifact is intentionally stored at a fine 5-second
cadence.  Analyze Tool normally needs only one final snapshot per chart bar.
Re-reading tens of millions of fine-grained heatmap cells for every plugin run
is wasteful, so this module builds a compact day-partitioned cache on first use.

The cache is a pure derived view:

* one row set per chart bucket, using the latest completed source snapshot;
* exact reconstructed end-depth fields, never time-weighted averages;
* atomic writes and source-file signature validation;
* no look-ahead: a target bucket contains only a source snapshot whose end is
  at or before that bucket's end.

All file interaction stays inside ``src.data_feed`` as required by the project
architecture.  The original heatmap NPZ remains the source of truth.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src.liquidity_map.store import LiquidityFeatureStore

_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_CACHE_SCHEMA_VERSION = 2
_CACHE_COLUMNS: dict[str, Any] = {
    "bucket_start_ms": np.int64,
    "bucket_end_ms": np.int64,
    "source_bucket_start_ms": np.int64,
    "source_bucket_end_ms": np.int64,
    "price_index": np.int32,
    "side_code": np.int8,
    "flow_valid": np.uint8,
    "end_depth_base": np.float32,
    "end_depth_usd": np.float32,
    "end_order_count": np.int32,
    "added_base": np.float32,
    "removed_base": np.float32,
    "executed_base": np.float32,
    "cancelled_base": np.float32,
    "consumed_base": np.float32,
    "replenished_base": np.float32,
}
_FLOW_COLUMNS = (
    "added_base",
    "removed_base",
    "executed_base",
    "cancelled_base",
    "consumed_base",
    "replenished_base",
)


def _safe_token(value: str) -> str:
    return _SAFE_RE.sub("_", str(value)).strip("_") or "default"


def _step_token(value: float) -> str:
    text = f"{float(value):.10g}"
    return _safe_token(text.replace(".", "p"))


def _as_utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _day(value: str | date | pd.Timestamp) -> date:
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        return value
    return pd.Timestamp(value).date()


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.unlink(missing_ok=True)
    try:
        with temp.open("wb") as handle:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".part")
    temp.unlink(missing_ok=True)
    try:
        with temp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _source_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({name: pd.Series(dtype=np.dtype(dtype)) for name, dtype in _CACHE_COLUMNS.items()})


def _selected_snapshot_row_indices(bucket_starts: np.ndarray, target_ms: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return rows from the latest source snapshot inside every target bucket."""

    if len(bucket_starts) == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, empty
    unique_starts, first_indices, counts = np.unique(
        bucket_starts.astype(np.int64, copy=False),
        return_index=True,
        return_counts=True,
    )
    target_starts = (unique_starts // int(target_ms)) * int(target_ms)
    last_snapshot = np.r_[target_starts[1:] != target_starts[:-1], True]
    chosen = np.flatnonzero(last_snapshot)
    row_parts = [
        np.arange(int(first_indices[index]), int(first_indices[index] + counts[index]), dtype=np.int64)
        for index in chosen
    ]
    rows = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int64)
    return rows, unique_starts[chosen].astype(np.int64), target_starts[chosen].astype(np.int64)


def _reduce_grouped(
    *,
    target_starts: np.ndarray,
    source_starts: np.ndarray,
    source_ends: np.ndarray,
    price_indices: np.ndarray,
    side_codes: np.ndarray,
    flow_valid: np.ndarray,
    numeric: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Group selected rows by target time, output price index and side."""

    if len(target_starts) == 0:
        return {name: np.empty(0, dtype=dtype) for name, dtype in _CACHE_COLUMNS.items()}
    order = np.lexsort((price_indices, side_codes, target_starts))
    target_sorted = target_starts[order]
    source_start_sorted = source_starts[order]
    source_end_sorted = source_ends[order]
    price_sorted = price_indices[order]
    side_sorted = side_codes[order]
    flow_sorted = flow_valid[order]
    start_mask = np.r_[
        True,
        (target_sorted[1:] != target_sorted[:-1])
        | (side_sorted[1:] != side_sorted[:-1])
        | (price_sorted[1:] != price_sorted[:-1]),
    ]
    starts = np.flatnonzero(start_mask)
    output: dict[str, np.ndarray] = {
        "bucket_start_ms": target_sorted[starts].astype(np.int64, copy=False),
        "source_bucket_start_ms": np.maximum.reduceat(source_start_sorted, starts).astype(np.int64, copy=False),
        "source_bucket_end_ms": np.maximum.reduceat(source_end_sorted, starts).astype(np.int64, copy=False),
        "price_index": price_sorted[starts].astype(np.int32, copy=False),
        "side_code": side_sorted[starts].astype(np.int8, copy=False),
        "flow_valid": np.minimum.reduceat(flow_sorted, starts).astype(np.uint8, copy=False),
    }
    for name, values in numeric.items():
        reduced = np.add.reduceat(values[order], starts)
        dtype = np.int32 if name == "end_order_count" else np.float32
        if name == "end_order_count":
            reduced = np.rint(np.maximum(reduced, 0.0))
        output[name] = reduced.astype(dtype, copy=False)
    return output


@dataclass(frozen=True)
class PeriodEndCachePaths:
    data: Path
    metadata: Path


class OKXLiquidityPeriodEndCache:
    """Build and read compact exact period-end snapshots one UTC day at a time."""

    _memory_lock = threading.RLock()
    _memory_frames: "OrderedDict[tuple[str, int, int], pd.DataFrame]" = OrderedDict()
    _memory_limit = 64

    def __init__(self, store: LiquidityFeatureStore) -> None:
        self.store = store

    def paths_for_day(self, day: str | date, *, target_seconds: int, target_price_step: float) -> PeriodEndCachePaths:
        d = _day(day)
        folder = (
            self.store.root
            / "_query_cache"
            / f"period_end_v{_CACHE_SCHEMA_VERSION}"
            / f"{int(target_seconds)}s_step_{_step_token(target_price_step)}"
            / f"{d.year:04d}"
            / f"{d.month:02d}"
        )
        stem = d.isoformat()
        return PeriodEndCachePaths(
            data=folder / f"{stem}.npz",
            metadata=folder / f"{stem}.json",
        )

    def _valid_metadata(
        self,
        payload: dict[str, Any],
        *,
        source_path: Path,
        target_seconds: int,
        target_price_step: float,
    ) -> bool:
        try:
            return (
                int(payload.get("schema_version")) == _CACHE_SCHEMA_VERSION
                and int(payload.get("target_seconds")) == int(target_seconds)
                and math.isclose(float(payload.get("target_price_step")), float(target_price_step), rel_tol=0.0, abs_tol=1e-12)
                and payload.get("source_heatmap") == str(source_path)
                and payload.get("source_signature") == _source_signature(source_path)
            )
        except (OSError, TypeError, ValueError):
            return False

    def _load_cached_frame(self, paths: PeriodEndCachePaths) -> pd.DataFrame:
        stat = paths.data.stat()
        key = (str(paths.data), int(stat.st_size), int(stat.st_mtime_ns))
        with self._memory_lock:
            cached = self._memory_frames.get(key)
            if cached is not None:
                self._memory_frames.move_to_end(key)
                out = cached.copy(deep=False)
                out.attrs.update(cached.attrs)
                out.attrs["memory_cache_hit"] = True
                return out
        with np.load(paths.data, allow_pickle=False) as data:
            frame = pd.DataFrame({name: data[name] for name in _CACHE_COLUMNS})
        with self._memory_lock:
            self._memory_frames[key] = frame
            self._memory_frames.move_to_end(key)
            while len(self._memory_frames) > self._memory_limit:
                self._memory_frames.popitem(last=False)
        out = frame.copy(deep=False)
        out.attrs["memory_cache_hit"] = False
        return out

    def load_or_build_day(
        self,
        day: str | date,
        *,
        target_seconds: int,
        target_price_step: float,
    ) -> pd.DataFrame:
        d = _day(day)
        source_paths = self.store.paths_for_day(d)
        if not source_paths.heatmap.exists() or not source_paths.metadata.exists():
            return _empty_frame()
        source_metadata = self.store.load_metadata(d)
        config = source_metadata.get("config") or {}
        source_price_step = float(config.get("price_step", 1.0))
        source_seconds = int(config.get("heatmap_seconds", 60))
        target_seconds = max(int(target_seconds), source_seconds)
        if target_seconds % source_seconds:
            target_seconds = int(math.ceil(target_seconds / source_seconds)) * source_seconds
        target_price_step = max(float(target_price_step), source_price_step)
        ratio = target_price_step / source_price_step
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                f"period-end cache price step ${target_price_step:g} must be an integer multiple "
                f"of source step ${source_price_step:g}"
            )
        paths = self.paths_for_day(
            d,
            target_seconds=target_seconds,
            target_price_step=target_price_step,
        )
        payload: dict[str, Any] = {}
        if paths.data.exists() and paths.metadata.exists():
            try:
                payload = json.loads(paths.metadata.read_text(encoding="utf-8"))
            except Exception:
                payload = {}
        if payload and self._valid_metadata(
            payload,
            source_path=source_paths.heatmap,
            target_seconds=target_seconds,
            target_price_step=target_price_step,
        ):
            frame = self._load_cached_frame(paths)
            frame.attrs.update(
                {
                    "cache_hit": True,
                    "cache_path": str(paths.data),
                    "utc_day": d.isoformat(),
                    "source_price_step": source_price_step,
                    "source_heatmap_seconds": source_seconds,
                    "price_step": target_price_step,
                    "heatmap_seconds": target_seconds,
                    "source_row_count": int(payload.get("source_row_count", 0)),
                }
            )
            return frame

        arrays = self._build_arrays(
            source_paths.heatmap,
            source_price_step=source_price_step,
            source_seconds=source_seconds,
            target_seconds=target_seconds,
            target_price_step=target_price_step,
        )
        paths.metadata.unlink(missing_ok=True)
        _atomic_npz(paths.data, arrays)
        source_row_count = int((source_metadata.get("stats") or {}).get("heatmap_cells", 0))
        if source_row_count <= 0:
            source_row_count = int(self._source_row_count(source_paths.heatmap))
        metadata = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "day": d.isoformat(),
            "source_heatmap": str(source_paths.heatmap),
            "source_signature": _source_signature(source_paths.heatmap),
            "source_price_step": source_price_step,
            "source_heatmap_seconds": source_seconds,
            "target_seconds": target_seconds,
            "target_price_step": target_price_step,
            "source_row_count": source_row_count,
            "cache_row_count": int(len(arrays["bucket_start_ms"])),
            "semantics": "latest completed source snapshot inside each target bucket; retain every positive end-depth price bin",
            "storage": "uncompressed_npz_for_fast_repeated_reads",
        }
        _atomic_json(paths.metadata, metadata)
        frame = self._load_cached_frame(paths)
        frame.attrs.update(
            {
                "cache_hit": False,
                "cache_path": str(paths.data),
                "utc_day": d.isoformat(),
                "source_price_step": source_price_step,
                "source_heatmap_seconds": source_seconds,
                "price_step": target_price_step,
                "heatmap_seconds": target_seconds,
                "source_row_count": int(metadata["source_row_count"]),
            }
        )
        return frame

    @staticmethod
    def _source_row_count(path: Path) -> int:
        with np.load(path, allow_pickle=False) as data:
            return int(len(data["bucket_start_ms"]))

    @staticmethod
    def _build_arrays(
        source_path: Path,
        *,
        source_price_step: float,
        source_seconds: int,
        target_seconds: int,
        target_price_step: float,
    ) -> dict[str, np.ndarray]:
        required = {
            "bucket_start_ms",
            "bucket_end_ms",
            "price_index",
            "side_code",
            "end_depth_base",
            "end_depth_usd",
            "end_order_count",
        }
        with np.load(source_path, allow_pickle=False) as data:
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(
                    f"legacy liquidity heatmap is missing exact end-state fields {missing}; "
                    "force-rebuild the affected day"
                )
            bucket_start = np.asarray(data["bucket_start_ms"], dtype=np.int64)
            rows, chosen_source_starts, chosen_target_starts = _selected_snapshot_row_indices(
                bucket_start,
                int(target_seconds) * 1000,
            )
            if len(rows) == 0:
                return {name: np.empty(0, dtype=dtype) for name, dtype in _CACHE_COLUMNS.items()}
            bucket_end = np.asarray(data["bucket_end_ms"], dtype=np.int64)[rows]
            source_start_rows = bucket_start[rows]
            target_lookup = dict(zip(chosen_source_starts.tolist(), chosen_target_starts.tolist()))
            target_start_rows = np.fromiter(
                (target_lookup[int(value)] for value in source_start_rows),
                dtype=np.int64,
                count=len(source_start_rows),
            )
            source_price_index = np.asarray(data["price_index"], dtype=np.int64)[rows]
            price_low = source_price_index.astype(np.float64) * float(source_price_step)
            price_out = np.floor(price_low / float(target_price_step) + 1e-12).astype(np.int64)
            side_code = np.asarray(data["side_code"], dtype=np.int8)[rows]
            end_depth_base = np.asarray(data["end_depth_base"], dtype=np.float64)[rows]
            positive = np.isfinite(end_depth_base) & (end_depth_base > 1e-12)
            if not bool(positive.any()):
                return {name: np.empty(0, dtype=dtype) for name, dtype in _CACHE_COLUMNS.items()}
            target_start_rows = target_start_rows[positive]
            source_start_rows = source_start_rows[positive]
            bucket_end = bucket_end[positive]
            price_out = price_out[positive]
            side_code = side_code[positive]
            end_depth_base = end_depth_base[positive]
            flow_valid = (
                np.asarray(data["flow_valid"], dtype=np.uint8)[rows][positive]
                if "flow_valid" in data.files
                else np.ones(int(positive.sum()), dtype=np.uint8)
            )
            numeric: dict[str, np.ndarray] = {
                "end_depth_base": end_depth_base,
                "end_depth_usd": np.asarray(data["end_depth_usd"], dtype=np.float64)[rows][positive],
                "end_order_count": np.asarray(data["end_order_count"], dtype=np.float64)[rows][positive],
            }
            for name in _FLOW_COLUMNS:
                numeric[name] = (
                    np.asarray(data[name], dtype=np.float64)[rows][positive]
                    if name in data.files
                    else np.zeros(int(positive.sum()), dtype=np.float64)
                )
        grouped = _reduce_grouped(
            target_starts=target_start_rows,
            source_starts=source_start_rows,
            source_ends=bucket_end,
            price_indices=price_out,
            side_codes=side_code,
            flow_valid=flow_valid,
            numeric=numeric,
        )
        grouped["bucket_end_ms"] = grouped["bucket_start_ms"] + int(target_seconds) * 1000
        return {name: np.asarray(grouped[name], dtype=dtype) for name, dtype in _CACHE_COLUMNS.items()}

    def iter_days(
        self,
        start: Any,
        end: Any,
        *,
        target_seconds: int,
        target_price_step: float,
    ) -> Iterator[pd.DataFrame]:
        start_utc = _as_utc_timestamp(start)
        end_utc = _as_utc_timestamp(end)
        start_ms = int(start_utc.timestamp() * 1000)
        end_ms = int(end_utc.timestamp() * 1000)
        for current in pd.date_range(start_utc.normalize(), end_utc.normalize(), freq="D"):
            frame = self.load_or_build_day(
                current.date(),
                target_seconds=target_seconds,
                target_price_step=target_price_step,
            )
            if frame.empty:
                continue
            mask = (
                pd.to_numeric(frame["bucket_start_ms"], errors="coerce").to_numpy(dtype=np.int64) < end_ms
            ) & (
                pd.to_numeric(frame["bucket_end_ms"], errors="coerce").to_numpy(dtype=np.int64) > start_ms
            )
            if not bool(mask.any()):
                continue
            out = frame.loc[mask].copy()
            out.attrs.update(frame.attrs)
            yield out


__all__ = ["OKXLiquidityPeriodEndCache", "PeriodEndCachePaths"]
