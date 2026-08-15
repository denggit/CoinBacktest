#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Checkpoint caches for expensive R01.2 source scans and 1-second replays."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import StablePathExecutionAuditConfig
from .source import SourcePaths, SourceScanResult
from .replay import ReplayResult


def _file_token(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"name": path.name, "missing": True}
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def source_cache_key(config: StablePathExecutionAuditConfig, paths: SourcePaths) -> str:
    payload = {
        "files": [_file_token(path) for path in (paths.feature, paths.label, paths.assignment, paths.manifest, paths.causal_audit)],
        "targets": list(config.target_clusters),
        "profile_cap": config.profile_sample_per_stratum,
        "replay_cap": config.replay_sample_per_stratum,
        "read_chunk": config.csv_read_chunk_rows,
        "version": 2,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def replay_cache_key(
    config: StablePathExecutionAuditConfig,
    source_key: str,
    db_path: Path,
) -> str:
    stat = db_path.stat() if db_path.exists() else None
    payload = {
        "source_key": source_key,
        "db": None if stat is None else {"path": str(db_path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "pre": config.pre_replay_seconds,
        "post": config.post_replay_seconds,
        "confirm": config.max_confirmation_seconds,
        "rules": {
            "stabilization": config.stabilization_seconds,
            "reclaim": list(config.reclaim_thresholds_bp),
            "second_push": [
                config.second_push_rebound_bp,
                config.second_push_retest_tolerance_bp,
                config.second_push_new_extreme_tolerance_bp,
            ],
        },
        "stop_buffer": config.structural_stop_buffer_bp,
        "cost": config.roundtrip_cost_bp,
        "cost_multipliers": list(config.cost_multipliers),
        "delay": list(config.entry_delay_seconds),
        "horizons": list(config.terminal_horizons_seconds),
        "replay_max_fill_gap_seconds": config.replay_max_fill_gap_seconds,
        "version": 3,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:24]


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    pd.to_pickle(payload, temp, compression={"method": "gzip", "compresslevel": 1, "mtime": 1})
    temp.replace(path)


def _read(path: Path) -> Any:
    return pd.read_pickle(path, compression="gzip")


def scan_cache_path(config: StablePathExecutionAuditConfig, key: str) -> Path:
    return config.cache_path / f"source_scan_{key}.pkl.gz"


def replay_cache_path(config: StablePathExecutionAuditConfig, key: str) -> Path:
    return config.cache_path / f"micro_replay_{key}.pkl.gz"


def save_scan_cache(path: Path, result: SourceScanResult) -> None:
    _write(path, result)


def load_scan_cache(path: Path) -> SourceScanResult:
    result = _read(path)
    if not isinstance(result, SourceScanResult):
        raise ValueError(f"invalid R01.2 source scan cache: {path}")
    return result


def save_replay_cache(path: Path, result: ReplayResult) -> None:
    _write(path, result)


def load_replay_cache(path: Path) -> ReplayResult:
    result = _read(path)
    if not isinstance(result, ReplayResult):
        raise ValueError(f"invalid R01.2 replay cache: {path}")
    return result
