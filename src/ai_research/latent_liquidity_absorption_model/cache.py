#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Checkpoint caches for R01.3 source scans and per-day causal snapshots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.ai_research.latent_liquidity_execution_audit.source import SourcePaths
from .source import AbsorptionSourceScanResult

from .config import AbsorptionModelConfig


def _token(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"path": str(path), "missing": True}
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def source_key(config: AbsorptionModelConfig, paths: SourcePaths) -> str:
    payload = {
        "files": [_token(path) for path in (paths.feature, paths.label, paths.assignment, paths.manifest, paths.causal_audit)],
        "targets": list(config.target_clusters),
        "replay_cap": config.replay_sample_per_stratum,
                "chunk": config.csv_read_chunk_rows,
        "version": 1,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def replay_key(config: AbsorptionModelConfig, src_key: str, db_path: Path) -> str:
    payload = {
        "source": src_key,
        "db": _token(db_path),
        "pre": config.pre_replay_seconds,
        "post": config.post_replay_seconds,
        "fill": config.replay_max_fill_gap_seconds,
        "offsets": list(config.decision_offsets_seconds),
        "windows": list(config.recent_windows_seconds),
        "horizon": config.label_horizon_seconds,
        "absorption": [config.absorption_lookahead_seconds, config.absorption_extension_tolerance_bp],
        "stop": config.structural_stop_buffer_bp,
        "cost": config.roundtrip_cost_bp,
        "room": config.minimum_net_room_bp,
        "version": 1,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    pd.to_pickle(value, temporary, compression={"method": "gzip", "compresslevel": 1, "mtime": 1})
    temporary.replace(path)


def _read(path: Path) -> Any:
    return pd.read_pickle(path, compression="gzip")


def source_scan_path(config: AbsorptionModelConfig, key: str) -> Path:
    return config.cache_path / f"source_scan_{key}.pkl.gz"


def save_source_scan(path: Path, scan: AbsorptionSourceScanResult) -> None:
    _write(path, scan)


def load_source_scan(path: Path) -> AbsorptionSourceScanResult:
    value = _read(path)
    if not isinstance(value, AbsorptionSourceScanResult):
        raise ValueError(f"invalid R01.3 source cache: {path}")
    return value


def snapshot_root(config: AbsorptionModelConfig, key: str) -> Path:
    return config.cache_path / f"snapshot_days_{key}"


def snapshot_day_path(root: Path, day: pd.Timestamp) -> Path:
    return root / f"day={pd.Timestamp(day).strftime('%Y-%m-%d')}.pkl.gz"


def save_snapshot_day(path: Path, frame: pd.DataFrame) -> None:
    _write(path, frame)


def load_snapshot_day(path: Path) -> pd.DataFrame:
    value = _read(path)
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"invalid R01.3 snapshot cache: {path}")
    return value
