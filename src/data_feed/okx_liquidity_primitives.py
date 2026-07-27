#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Low-semantic, causal order-book primitives for liquidity research.

The cache deliberately stores *research ingredients*, not wall labels.  Each
UTC day is converted once from the canonical liquidity-map heatmap into sorted
NumPy arrays plus snapshot offsets and robust relative-depth summaries.  Later
wall definitions may freely change widths, thresholds, persistence and model
features without re-reading or re-normalizing every Pandas snapshot.

Causal contract
---------------
* Cell depth is the completed source bucket's ``end_depth_base`` when present.
* Snapshot summaries become available only at ``bucket_end_ms``.
* Rolling references contain only the current and earlier completed snapshots.
* No future price outcome, touch label or wall decision is stored here.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import re
import zipfile
from typing import Any, Iterator

import numpy as np
import pandas as pd

_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")
_SCHEMA_VERSION = 1
_FLOW_COLUMNS = (
    "added_base",
    "removed_base",
    "executed_base",
    "cancelled_base",
    "consumed_base",
    "replenished_base",
)


def _safe(value: str) -> str:
    return _SAFE_RE.sub("_", str(value)).strip("_") or "default"


@dataclass(frozen=True)
class LiquidityPrimitiveConfig:
    reference_window_hours: float = 24.0
    denominator_floor_absolute: float = 1e-9
    denominator_floor_fraction_of_side_mean: float = 0.02

    def validate(self) -> None:
        if self.reference_window_hours <= 0:
            raise ValueError("reference_window_hours must be > 0")
        if self.denominator_floor_absolute <= 0:
            raise ValueError("denominator_floor_absolute must be > 0")
        if self.denominator_floor_fraction_of_side_mean < 0:
            raise ValueError("denominator_floor_fraction_of_side_mean must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference_window_hours": float(self.reference_window_hours),
            "denominator_floor_absolute": float(self.denominator_floor_absolute),
            "denominator_floor_fraction_of_side_mean": float(
                self.denominator_floor_fraction_of_side_mean
            ),
        }


@dataclass(frozen=True)
class LiquidityPrimitivePaths:
    primitives: Path
    metadata: Path


@dataclass(frozen=True)
class LiquidityPrimitiveSnapshot:
    bucket_start_ms: int
    bucket_end_ms: int
    midpoint: float
    best_bid_bin: int
    best_ask_bin: int
    price_step: float
    price_index: np.ndarray
    side_code: np.ndarray
    depth_base: np.ndarray
    added_base: np.ndarray
    removed_base: np.ndarray
    executed_base: np.ndarray
    cancelled_base: np.ndarray
    consumed_base: np.ndarray
    replenished_base: np.ndarray
    flow_valid: np.ndarray
    bid_q25: float
    bid_q50: float
    bid_q75: float
    ask_q25: float
    ask_q50: float
    ask_q75: float
    bid_total: float
    ask_total: float
    bid_max: float
    ask_max: float
    causal_q95: float
    causal_q99: float
    reference_ready_fraction: float

    def baseline(self, side: str, quantile: float) -> float:
        prefix = "bid" if side == "bid" else "ask"
        cached = {
            0.25: getattr(self, f"{prefix}_q25"),
            0.50: getattr(self, f"{prefix}_q50"),
            0.75: getattr(self, f"{prefix}_q75"),
        }
        for key, value in cached.items():
            if abs(float(quantile) - key) <= 1e-12:
                return max(float(value), 1e-12)
        code = 1 if side == "bid" else -1
        values = self.depth_base[(self.side_code == code) & (self.depth_base > 0)]
        return max(float(np.quantile(values, quantile)) if len(values) else 0.0, 1e-12)


@dataclass
class LiquidityPrimitiveDay:
    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]

    @property
    def snapshot_count(self) -> int:
        return int(len(self.arrays.get("bucket_start_ms", ())))

    @property
    def cell_count(self) -> int:
        return int(len(self.arrays.get("price_index", ())))

    @property
    def price_step(self) -> float:
        return float(self.metadata.get("price_step", 1.0))

    @property
    def source_seconds(self) -> int:
        return int(self.metadata.get("source_seconds", 5))

    def snapshot(self, index: int) -> LiquidityPrimitiveSnapshot:
        i = int(index)
        if i < 0 or i >= self.snapshot_count:
            raise IndexError(i)
        start = int(self.arrays["row_start"][i])
        end = int(self.arrays["row_end"][i])
        sl = slice(start, end)
        kwargs = {
            name: self.arrays[name][sl]
            for name in (
                "price_index",
                "side_code",
                "depth_base",
                *_FLOW_COLUMNS,
                "flow_valid",
            )
        }
        return LiquidityPrimitiveSnapshot(
            bucket_start_ms=int(self.arrays["bucket_start_ms"][i]),
            bucket_end_ms=int(self.arrays["bucket_end_ms"][i]),
            midpoint=float(self.arrays["midpoint"][i]),
            best_bid_bin=int(self.arrays["best_bid_bin"][i]),
            best_ask_bin=int(self.arrays["best_ask_bin"][i]),
            price_step=self.price_step,
            bid_q25=float(self.arrays["bid_q25"][i]),
            bid_q50=float(self.arrays["bid_q50"][i]),
            bid_q75=float(self.arrays["bid_q75"][i]),
            ask_q25=float(self.arrays["ask_q25"][i]),
            ask_q50=float(self.arrays["ask_q50"][i]),
            ask_q75=float(self.arrays["ask_q75"][i]),
            bid_total=float(self.arrays["bid_total"][i]),
            ask_total=float(self.arrays["ask_total"][i]),
            bid_max=float(self.arrays["bid_max"][i]),
            ask_max=float(self.arrays["ask_max"][i]),
            causal_q95=float(self.arrays["causal_q95"][i]),
            causal_q99=float(self.arrays["causal_q99"][i]),
            reference_ready_fraction=float(self.arrays["reference_ready_fraction"][i]),
            **kwargs,
        )

    def iter_snapshots(self) -> Iterator[LiquidityPrimitiveSnapshot]:
        for index in range(self.snapshot_count):
            yield self.snapshot(index)


class CausalPrimitiveReference:
    """Rolling maxima of per-snapshot robust depth quantiles."""

    def __init__(self, window_hours: float = 24.0, minimum: float = 1e-12) -> None:
        self.window_ms = max(1, int(float(window_hours) * 3_600_000))
        self.minimum = float(minimum)
        self._q95: deque[tuple[int, float]] = deque()
        self._q99: deque[tuple[int, float]] = deque()
        self._first_timestamp_ms: int | None = None

    @staticmethod
    def _push(queue: deque[tuple[int, float]], timestamp_ms: int, value: float, cutoff: int) -> None:
        while queue and queue[0][0] < cutoff:
            queue.popleft()
        while queue and queue[-1][1] <= value:
            queue.pop()
        queue.append((timestamp_ms, value))

    def update(self, timestamp_ms: int, q95: float, q99: float) -> tuple[float, float, float]:
        timestamp_ms = int(timestamp_ms)
        self._first_timestamp_ms = (
            timestamp_ms if self._first_timestamp_ms is None else self._first_timestamp_ms
        )
        cutoff = timestamp_ms - self.window_ms
        q95 = max(float(q95), self.minimum)
        q99 = max(float(q99), self.minimum)
        self._push(self._q95, timestamp_ms, q95, cutoff)
        self._push(self._q99, timestamp_ms, q99, cutoff)
        elapsed = max(0, timestamp_ms - int(self._first_timestamp_ms))
        ready = min(1.0, elapsed / max(self.window_ms, 1))
        return float(self._q95[0][1]), float(self._q99[0][1]), float(ready)

    def replay_arrays(
        self,
        bucket_end_ms: np.ndarray,
        snapshot_q95: np.ndarray,
        snapshot_q99: np.ndarray,
    ) -> None:
        """Restore causal state from the three compact reference arrays.

        Cell-level primitive arrays can contain millions of values per day but
        are irrelevant to the rolling 24-hour maxima.  Selective NPZ loading
        plus this method makes resume/skip paths cheap without changing the
        reference state by a single observation.
        """

        for timestamp_ms, q95, q99 in zip(
            bucket_end_ms, snapshot_q95, snapshot_q99, strict=False
        ):
            self.update(int(timestamp_ms), float(q95), float(q99))

    def replay_day(self, day: LiquidityPrimitiveDay) -> None:
        self.replay_arrays(
            day.arrays["bucket_end_ms"],
            day.arrays["snapshot_q95"],
            day.arrays["snapshot_q99"],
        )


class OKXLiquidityPrimitiveStore:
    """Day-partitioned atomic NPZ store for neutral liquidity primitives."""

    def __init__(
        self,
        *,
        symbol: str = "ETH-USDT-SWAP",
        books_depth: int = 5000,
        data_dir: str | os.PathLike[str] | None = None,
        cache_version: str = "v1",
    ) -> None:
        project_root = Path(__file__).resolve().parents[2]
        base = Path(data_dir) if data_dir else project_root / "data"
        self.symbol = str(symbol)
        self.books_depth = int(books_depth)
        self.cache_version = str(cache_version)
        self.root = (
            base
            / "okx"
            / "derived"
            / "liquidity_primitives"
            / _safe(symbol)
            / f"books_{self.books_depth}"
            / _safe(cache_version)
        )

    def paths_for_day(self, day: str | date) -> LiquidityPrimitivePaths:
        d = self._parse_day(day)
        folder = self.root / f"{d.year:04d}" / f"{d.month:02d}"
        stem = d.isoformat()
        return LiquidityPrimitivePaths(
            primitives=folder / f"{stem}.primitives.npz",
            metadata=folder / f"{stem}.metadata.json",
        )

    def has_day(self, day: str | date) -> bool:
        paths = self.paths_for_day(day)
        return paths.primitives.exists() and paths.metadata.exists()

    def save_day(
        self,
        day: str | date,
        *,
        arrays: dict[str, np.ndarray],
        metadata: dict[str, Any],
        compression_level: int = 6,
    ) -> LiquidityPrimitivePaths:
        paths = self.paths_for_day(day)
        paths.metadata.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(metadata)
        payload.update(
            {
                "schema_version": _SCHEMA_VERSION,
                "day": self._parse_day(day).isoformat(),
                "symbol": self.symbol,
                "books_depth": self.books_depth,
                "cache_version": self.cache_version,
                "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "primitives_path": str(paths.primitives),
                "compression_level": int(compression_level),
            }
        )
        paths.metadata.unlink(missing_ok=True)
        tmp_npz = paths.primitives.with_name(paths.primitives.name + ".part")
        tmp_meta = paths.metadata.with_name(paths.metadata.name + ".part")
        tmp_npz.unlink(missing_ok=True)
        tmp_meta.unlink(missing_ok=True)
        try:
            self._write_npz(tmp_npz, arrays, compression_level=int(compression_level))
            with tmp_meta.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_npz, paths.primitives)
            os.replace(tmp_meta, paths.metadata)
        finally:
            tmp_npz.unlink(missing_ok=True)
            tmp_meta.unlink(missing_ok=True)
        return paths

    def load_metadata(self, day: str | date) -> dict[str, Any]:
        paths = self.paths_for_day(day)
        if not paths.metadata.exists():
            return {}
        payload = json.loads(paths.metadata.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != _SCHEMA_VERSION:
            raise ValueError(
                f"unsupported primitive schema {payload.get('schema_version')} at {paths.metadata}"
            )
        return payload

    def load_reference_arrays(
        self, day: str | date
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load only arrays needed to replay the causal rolling reference.

        ``np.load`` extracts NPZ members lazily, so the multi-million-cell
        price/depth/flow arrays are never decompressed on cached-day resume.
        """

        paths = self.paths_for_day(day)
        if not self.has_day(day):
            raise FileNotFoundError(f"liquidity primitive cache missing: {paths.metadata}")
        self.load_metadata(day)
        required = ("bucket_end_ms", "snapshot_q95", "snapshot_q99")
        with np.load(paths.primitives, allow_pickle=False) as data:
            missing = [name for name in required if name not in data.files]
            if missing:
                raise ValueError(f"primitive reference arrays missing {missing} at {paths.primitives}")
            return tuple(np.asarray(data[name]).copy() for name in required)  # type: ignore[return-value]

    @staticmethod
    def _write_npz(
        path: Path,
        arrays: dict[str, np.ndarray],
        *,
        compression_level: int,
    ) -> None:
        level = int(compression_level)
        if not 0 <= level <= 9:
            raise ValueError("compression_level must be between 0 and 9")
        compression = zipfile.ZIP_STORED if level == 0 else zipfile.ZIP_DEFLATED
        kwargs: dict[str, Any] = {
            "mode": "w",
            "compression": compression,
            "allowZip64": True,
        }
        if compression == zipfile.ZIP_DEFLATED:
            kwargs["compresslevel"] = level
        with path.open("wb") as handle:
            with zipfile.ZipFile(handle, **kwargs) as archive:
                for name, value in arrays.items():
                    with archive.open(f"{name}.npy", "w", force_zip64=True) as member:
                        np.lib.format.write_array(
                            member, np.asanyarray(value), allow_pickle=False
                        )
            handle.flush()
            os.fsync(handle.fileno())

    def load_day(self, day: str | date) -> LiquidityPrimitiveDay:
        paths = self.paths_for_day(day)
        if not self.has_day(day):
            raise FileNotFoundError(f"liquidity primitive cache missing: {paths.metadata}")
        metadata = self.load_metadata(day)
        with np.load(paths.primitives, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        return LiquidityPrimitiveDay(arrays=arrays, metadata=metadata)

    def coverage(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if not self.root.exists():
            return rows
        for path in sorted(self.root.rglob("*.metadata.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            rows.append(
                {
                    "day": payload.get("day"),
                    "snapshots": int(payload.get("snapshot_count", 0)),
                    "cells": int(payload.get("cell_count", 0)),
                    "metadata": str(path),
                }
            )
        return rows

    @staticmethod
    def _parse_day(value: str | date) -> date:
        if isinstance(value, date):
            return value
        return datetime.strptime(str(value), "%Y-%m-%d").date()


class OKXLiquidityPrimitiveLoader:
    """Public research loader for cached neutral primitives."""

    def __init__(
        self,
        *,
        symbol: str = "ETH-USDT-SWAP",
        books_depth: int = 5000,
        data_dir: str | os.PathLike[str] | None = None,
        cache_version: str = "v1",
    ) -> None:
        self.store = OKXLiquidityPrimitiveStore(
            symbol=symbol,
            books_depth=books_depth,
            data_dir=data_dir,
            cache_version=cache_version,
        )

    def coverage(self) -> list[dict[str, Any]]:
        return self.store.coverage()

    def has_day(self, day: str | date) -> bool:
        return self.store.has_day(day)

    def load_day(self, day: str | date) -> LiquidityPrimitiveDay:
        return self.store.load_day(day)

    def iter_days(self, start: Any, end: Any) -> Iterator[LiquidityPrimitiveDay]:
        start_day = pd.Timestamp(start).date()
        end_day = pd.Timestamp(end).date()
        for current in pd.date_range(start_day, end_day, freq="D"):
            if self.store.has_day(current.date()):
                yield self.store.load_day(current.date())

def build_liquidity_primitive_day(
    heatmap_day: pd.DataFrame,
    *,
    reference: CausalPrimitiveReference,
    config: LiquidityPrimitiveConfig | None = None,
) -> LiquidityPrimitiveDay:
    """Convert one canonical heatmap day into sorted low-semantic arrays."""

    cfg = config or LiquidityPrimitiveConfig()
    cfg.validate()
    if heatmap_day is None or heatmap_day.empty:
        raise ValueError("heatmap_day must not be empty")
    required = {"bucket_start_ms", "bucket_end_ms", "price_index", "side_code"}
    missing = sorted(required.difference(heatmap_day.columns))
    if missing:
        raise ValueError(f"heatmap day missing fields: {missing}")
    depth_column = "end_depth_base" if "end_depth_base" in heatmap_day.columns else "depth_base"
    if depth_column not in heatmap_day.columns:
        raise ValueError("heatmap day needs end_depth_base or depth_base")

    columns = ["bucket_start_ms", "bucket_end_ms", "price_index", "side_code", depth_column]
    columns.extend(name for name in _FLOW_COLUMNS if name in heatmap_day.columns)
    if "flow_valid" in heatmap_day.columns:
        columns.append("flow_valid")
    work = heatmap_day.loc[:, columns].copy()
    for name in ("bucket_start_ms", "bucket_end_ms", "price_index", "side_code"):
        work[name] = pd.to_numeric(work[name], errors="coerce")
    work[depth_column] = pd.to_numeric(work[depth_column], errors="coerce").fillna(0.0)
    for name in _FLOW_COLUMNS:
        if name not in work.columns:
            work[name] = 0.0
        else:
            work[name] = pd.to_numeric(work[name], errors="coerce").fillna(0.0)
    if "flow_valid" not in work.columns:
        work["flow_valid"] = 0
    work["flow_valid"] = pd.to_numeric(work["flow_valid"], errors="coerce").fillna(0)
    work = work.dropna(subset=["bucket_start_ms", "bucket_end_ms", "price_index", "side_code"])
    has_flow = np.zeros(len(work), dtype=bool)
    for name in _FLOW_COLUMNS:
        has_flow |= work[name].to_numpy(dtype=float) > 0
    work = work.loc[
        work["side_code"].isin([1, -1])
        & ((work[depth_column] > 0).to_numpy(dtype=bool) | has_flow)
    ].copy()
    work = work.sort_values(["bucket_start_ms", "side_code", "price_index"]).drop_duplicates(
        ["bucket_start_ms", "side_code", "price_index"], keep="last"
    )
    if work.empty:
        raise ValueError("heatmap day has no positive bid/ask cells")

    bucket_start = work["bucket_start_ms"].to_numpy(dtype=np.int64)
    unique_start, row_start, counts = np.unique(bucket_start, return_index=True, return_counts=True)
    row_end = row_start + counts
    bucket_end_all = work["bucket_end_ms"].to_numpy(dtype=np.int64)
    price_index = work["price_index"].to_numpy(dtype=np.int32)
    side_code = work["side_code"].to_numpy(dtype=np.int8)
    depth = work[depth_column].to_numpy(dtype=np.float32)
    flow_arrays = {name: work[name].to_numpy(dtype=np.float32) for name in _FLOW_COLUMNS}
    flow_valid = work["flow_valid"].to_numpy(dtype=np.int8)

    n = len(unique_start)
    snapshot_arrays: dict[str, np.ndarray] = {
        "bucket_start_ms": unique_start.astype(np.int64),
        "bucket_end_ms": np.empty(n, dtype=np.int64),
        "row_start": row_start.astype(np.int64),
        "row_end": row_end.astype(np.int64),
        "best_bid_bin": np.full(n, -1, dtype=np.int32),
        "best_ask_bin": np.full(n, -1, dtype=np.int32),
        "midpoint": np.full(n, np.nan, dtype=np.float64),
    }
    for name in (
        "bid_q25", "bid_q50", "bid_q75", "ask_q25", "ask_q50", "ask_q75",
        "bid_total", "ask_total", "bid_max", "ask_max", "snapshot_q95", "snapshot_q99",
        "causal_q95", "causal_q99", "reference_ready_fraction",
    ):
        snapshot_arrays[name] = np.zeros(n, dtype=np.float32)

    price_step = float(heatmap_day.attrs.get("price_step", 1.0))
    source_seconds = int(heatmap_day.attrs.get("heatmap_seconds", 5))
    for i, (start_pos, end_pos) in enumerate(zip(row_start, row_end, strict=False)):
        sl = slice(int(start_pos), int(end_pos))
        codes = side_code[sl]
        prices = price_index[sl]
        values = depth[sl].astype(np.float64, copy=False)
        snapshot_arrays["bucket_end_ms"][i] = int(np.max(bucket_end_all[sl]))
        bid_mask = (codes == 1) & (values > 0)
        ask_mask = (codes == -1) & (values > 0)
        bid = values[bid_mask]
        ask = values[ask_mask]
        bid_prices = prices[bid_mask]
        ask_prices = prices[ask_mask]
        if len(bid_prices):
            snapshot_arrays["best_bid_bin"][i] = int(np.max(bid_prices))
        if len(ask_prices):
            snapshot_arrays["best_ask_bin"][i] = int(np.min(ask_prices))
        if len(bid_prices) and len(ask_prices):
            snapshot_arrays["midpoint"][i] = (
                float(np.max(bid_prices) + 1) + float(np.min(ask_prices))
            ) * price_step / 2.0
        for prefix, side_values in (("bid", bid), ("ask", ask)):
            if len(side_values):
                q25, q50, q75 = np.quantile(side_values, (0.25, 0.50, 0.75))
                mean = float(np.mean(side_values))
                floor = max(
                    cfg.denominator_floor_absolute,
                    mean * cfg.denominator_floor_fraction_of_side_mean,
                )
                snapshot_arrays[f"{prefix}_q25"][i] = max(float(q25), floor)
                snapshot_arrays[f"{prefix}_q50"][i] = max(float(q50), floor)
                snapshot_arrays[f"{prefix}_q75"][i] = max(float(q75), floor)
                snapshot_arrays[f"{prefix}_total"][i] = float(np.sum(side_values))
                snapshot_arrays[f"{prefix}_max"][i] = float(np.max(side_values))
        positive_values = values[values > 0]
        if len(positive_values) == 0:
            continue
        q95, q99 = np.quantile(positive_values, (0.95, 0.99))
        causal95, causal99, ready = reference.update(
            int(snapshot_arrays["bucket_end_ms"][i]), float(q95), float(q99)
        )
        snapshot_arrays["snapshot_q95"][i] = float(q95)
        snapshot_arrays["snapshot_q99"][i] = float(q99)
        snapshot_arrays["causal_q95"][i] = float(causal95)
        snapshot_arrays["causal_q99"][i] = float(causal99)
        snapshot_arrays["reference_ready_fraction"][i] = float(ready)

    arrays: dict[str, np.ndarray] = {
        **snapshot_arrays,
        "price_index": price_index,
        "side_code": side_code,
        "depth_base": depth,
        **flow_arrays,
        "flow_valid": flow_valid,
    }
    metadata = {
        "price_step": price_step,
        "source_seconds": source_seconds,
        "snapshot_count": int(n),
        "cell_count": int(len(depth)),
        "utc_day": str(heatmap_day.attrs.get("utc_day", "")),
        "config": cfg.to_dict(),
        "semantics": {
            "cache": "low-semantic causal liquidity research primitives; not wall labels",
            "depth_base": depth_column,
            "relative_depth": "q25/q50/q75 side baselines plus causal rolling q95/q99",
            "future_outcomes": "not included",
        },
    }
    return LiquidityPrimitiveDay(arrays=arrays, metadata=metadata)
