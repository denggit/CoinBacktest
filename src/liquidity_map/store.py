#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compact on-disk store for derived offline liquidity-map artifacts."""

from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .models import LiquidityBuildStats, LiquidityMapConfig
from .schemas import FEATURE_DTYPES, HEATMAP_DTYPES

_SAFE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe(value: str) -> str:
    return _SAFE_RE.sub("_", str(value)).strip("_") or "default"


@dataclass(frozen=True)
class LiquidityArtifactPaths:
    heatmap: Path
    features: Path
    metadata: Path


class LiquidityFeatureStore:
    """Save/load day-partitioned NPZ artifacts without extra dependencies."""

    FEATURE_DTYPES = FEATURE_DTYPES
    HEATMAP_DTYPES = HEATMAP_DTYPES

    def __init__(
        self,
        *,
        symbol: str = "ETH-USDT-SWAP",
        books_depth: int = 400,
        data_dir: str | os.PathLike[str] | None = None,
    ):
        project_root = Path(__file__).resolve().parents[2]
        base = Path(data_dir) if data_dir else project_root / "data"
        self.symbol = symbol
        self.books_depth = int(books_depth)
        self.root = base / "okx" / "derived" / "liquidity_map" / _safe(symbol) / f"books_{self.books_depth}"

    def paths_for_day(self, day: str | date) -> LiquidityArtifactPaths:
        d = self._parse_day(day)
        folder = self.root / f"{d.year:04d}" / f"{d.month:02d}"
        stem = d.isoformat()
        return LiquidityArtifactPaths(
            heatmap=folder / f"{stem}.heatmap.npz",
            features=folder / f"{stem}.features.npz",
            metadata=folder / f"{stem}.metadata.json",
        )

    def save_day(
        self,
        day: str | date,
        *,
        config: LiquidityMapConfig,
        feature_rows: Any,
        heatmap_rows: Any,
        stats: LiquidityBuildStats,
        source_files: Iterable[str] = (),
        compression_level: int = 6,
    ) -> LiquidityArtifactPaths:
        paths = self.paths_for_day(day)
        paths.metadata.parent.mkdir(parents=True, exist_ok=True)
        feature_arrays = self._rows_to_arrays(feature_rows, self.FEATURE_DTYPES)
        heatmap_arrays = self._rows_to_arrays(heatmap_rows, self.HEATMAP_DTYPES)
        payload = {
            "schema_version": 2,
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "day": self._parse_day(day).isoformat(),
            "symbol": self.symbol,
            "books_depth": self.books_depth,
            "config": config.to_dict(),
            "stats": stats.to_dict(),
            "heatmap_depth_semantics": {
                "depth_base": "time_weighted_mean_within_bucket",
                "end_depth_base": "exact_reconstructed_book_state_at_bucket_end_exclusive",
            },
            "source_files": sorted({str(item) for item in source_files if item}),
            "artifacts": {
                "features": str(paths.features),
                "heatmap": str(paths.heatmap),
            },
        }
        # The metadata file is the day-completion checkpoint.  Remove it
        # before rebuilding so an interrupted force-rebuild can never leave a
        # mixed old/new day that ``has_day`` would incorrectly skip.
        paths.metadata.unlink(missing_ok=True)
        temp_features = paths.features.with_name(paths.features.name + ".part")
        temp_heatmap = paths.heatmap.with_name(paths.heatmap.name + ".part")
        temp_metadata = paths.metadata.with_name(paths.metadata.name + ".part")
        temp_paths = (temp_features, temp_heatmap, temp_metadata)
        for temp_path in temp_paths:
            temp_path.unlink(missing_ok=True)

        try:
            if int(compression_level) == 6:
                # Keep the long-standing two-argument seam intact for callers
                # and fault-injection tests that monkeypatch this method.
                self._write_npz_file(temp_features, feature_arrays)
                self._write_npz_file(temp_heatmap, heatmap_arrays)
            else:
                self._write_npz_file_level(
                    temp_features,
                    feature_arrays,
                    compression_level=int(compression_level),
                )
                self._write_npz_file_level(
                    temp_heatmap,
                    heatmap_arrays,
                    compression_level=int(compression_level),
                )
            self._write_text_file(
                temp_metadata,
                json.dumps(payload, ensure_ascii=False, indent=2),
            )

            # Publish the two data files first and the metadata checkpoint last.
            # ``os.replace`` is atomic on the same filesystem on Windows and
            # Unix.  A killed process therefore leaves either a complete day or
            # a day with no metadata, which is rebuilt on the next run.
            os.replace(temp_features, paths.features)
            os.replace(temp_heatmap, paths.heatmap)
            os.replace(temp_metadata, paths.metadata)
        finally:
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)
        return paths


    @staticmethod
    def _write_npz_file(path: Path, arrays: dict[str, np.ndarray]) -> None:
        LiquidityFeatureStore._write_npz_file_level(
            path,
            arrays,
            compression_level=6,
        )

    @staticmethod
    def _write_npz_file_level(
        path: Path,
        arrays: dict[str, np.ndarray],
        *,
        compression_level: int,
    ) -> None:
        """Write a NumPy-compatible NPZ with a selectable DEFLATE level.

        NumPy's ``savez_compressed`` hard-codes the zlib default level.  Level
        1 is materially faster for large day-partitioned heatmaps while still
        retaining compression.  Arrays are streamed directly into the ZIP so
        the writer does not create a second full-size in-memory copy.
        """

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
                for name, array_value in arrays.items():
                    array = np.asanyarray(array_value)
                    # ZipExtFile is sequential and accepted by NumPy's NPY
                    # writer.  This keeps peak memory bounded by the source
                    # arrays already owned by the day builder.
                    with archive.open(f"{name}.npy", "w", force_zip64=True) as member:
                        np.lib.format.write_array(member, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())

    @staticmethod
    def _write_text_file(path: Path, text: str) -> None:
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())

    def has_day(self, day: str | date) -> bool:
        paths = self.paths_for_day(day)
        return paths.metadata.exists() and paths.features.exists() and paths.heatmap.exists()

    def load_metadata(self, day: str | date) -> dict[str, Any]:
        path = self.paths_for_day(day).metadata
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def load_features(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
        start_ts = self._as_utc_timestamp(start)
        end_ts = self._as_utc_timestamp(end)
        frames: list[pd.DataFrame] = []
        for day in pd.date_range(start_ts.normalize(), end_ts.normalize(), freq="D"):
            path = self.paths_for_day(day.date()).features
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as data:
                frame = pd.DataFrame({name: data[name] for name in data.files})
            if frame.empty:
                continue
            mask = (frame["bucket_start_ms"] <= int(end_ts.timestamp() * 1000)) & (
                frame["bucket_end_ms"] >= int(start_ts.timestamp() * 1000)
            )
            frames.append(frame.loc[mask].copy())
        if not frames:
            return pd.DataFrame(columns=list(self.FEATURE_DTYPES))
        return pd.concat(frames, ignore_index=True).sort_values("bucket_start_ms").reset_index(drop=True)

    def load_heatmap(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DataFrame:
        start_ts = self._as_utc_timestamp(start)
        end_ts = self._as_utc_timestamp(end)
        frames: list[pd.DataFrame] = []
        for day in pd.date_range(start_ts.normalize(), end_ts.normalize(), freq="D"):
            path = self.paths_for_day(day.date()).heatmap
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as data:
                frame = pd.DataFrame({name: data[name] for name in data.files})
            if frame.empty:
                continue
            mask = (frame["bucket_start_ms"] < int(end_ts.timestamp() * 1000)) & (
                frame["bucket_end_ms"] > int(start_ts.timestamp() * 1000)
            )
            frames.append(frame.loc[mask].copy())
        if not frames:
            return pd.DataFrame(columns=list(self.HEATMAP_DTYPES))
        return pd.concat(frames, ignore_index=True).sort_values(["bucket_start_ms", "side_code", "price_index"]).reset_index(drop=True)

    def coverage(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not self.root.exists():
            return out
        for path in sorted(self.root.rglob("*.metadata.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            out.append(
                {
                    "day": payload.get("day"),
                    "features": payload.get("stats", {}).get("book_feature_rows", 0),
                    "heatmap_cells": payload.get("stats", {}).get("heatmap_cells", 0),
                    "metadata": str(path),
                }
            )
        return out

    @staticmethod
    def _rows_to_arrays(rows: Any, dtypes: dict[str, Any]) -> dict[str, np.ndarray]:
        if isinstance(rows, dict):
            arrays: dict[str, np.ndarray] = {}
            lengths: set[int] = set()
            for name, dtype in dtypes.items():
                value = rows.get(name)
                if value is None:
                    raise ValueError(f"columnar artifact missing required field: {name}")
                array = np.asarray(value, dtype=dtype)
                arrays[name] = array
                lengths.add(len(array))
            if len(lengths) > 1:
                raise ValueError(f"columnar artifact has inconsistent lengths: {sorted(lengths)}")
            return arrays
        arrays = {}
        for name, dtype in dtypes.items():
            default: Any = np.nan if np.issubdtype(np.dtype(dtype), np.floating) else 0
            arrays[name] = np.asarray([row.get(name, default) for row in rows], dtype=dtype)
        return arrays


    @staticmethod
    def _as_utc_timestamp(value: str | pd.Timestamp) -> pd.Timestamp:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            return ts.tz_localize("UTC")
        return ts.tz_convert("UTC")

    @staticmethod
    def _parse_day(value: str | date) -> date:
        if isinstance(value, date):
            return value
        return datetime.strptime(str(value), "%Y-%m-%d").date()
