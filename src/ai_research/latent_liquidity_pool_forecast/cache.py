#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.ai_research.config import PROJECT_ROOT
from src.ai_research.latent_liquidity_path_atlas.config import DEFAULT_CONFIG as ATLAS_CONFIG

from .config import LatentLiquidityPoolForecastConfig


def cache_key(config: LatentLiquidityPoolForecastConfig) -> str:
    payload = config.to_dict().copy()
    for key in ("report_dir", "model_train_cap_rows", "model_eval_cap_rows_per_period"):
        payload.pop(key, None)
    manifest = config.source_report_path / config.source_manifest_file
    if manifest.exists():
        try:
            payload["source_manifest_sha256"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        except OSError:
            payload["source_manifest_sha256"] = "UNREADABLE"
    else:
        payload["source_manifest_sha256"] = "MISSING"
    for key, filename in (
        ("source_feature_stat", config.source_feature_file),
        ("source_label_stat", config.source_label_file),
        ("source_assignment_stat", config.source_assignment_file),
    ):
        source = config.source_report_path / filename
        try:
            stat = source.stat()
            payload[key] = [int(stat.st_size), int(stat.st_mtime_ns)]
        except OSError:
            payload[key] = "MISSING"
    symbol = config.symbol.replace("-", "_")
    end = pd.Timestamp(config.research_end).strftime("%Y%m%d")
    swing = PROJECT_ROOT / ATLAS_CONFIG.swing_cache_dir / f"{symbol}_unswept_swing_lifecycle_to_{end}.csv.gz"
    try:
        stat = swing.stat()
        payload["swing_lifecycle_stat"] = [int(stat.st_size), int(stat.st_mtime_ns)]
    except OSError:
        payload["swing_lifecycle_stat"] = "MISSING"
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def dataset_cache_path(config: LatentLiquidityPoolForecastConfig) -> Path:
    return config.cache_path / cache_key(config) / "spatial_dataset.pkl.gz"


def episode_cache_path(config: LatentLiquidityPoolForecastConfig) -> Path:
    return config.cache_path / cache_key(config) / "episode_labels.pkl.gz"


def save_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_pickle(temp, compression={"method": "gzip", "compresslevel": 1, "mtime": 1})
    temp.replace(path)


def load_frame(path: Path) -> pd.DataFrame:
    value = pd.read_pickle(path, compression="gzip")
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"invalid R02 cache: {path}")
    return value


def chunk_cache_path(config: LatentLiquidityPoolForecastConfig, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    return config.cache_path / cache_key(config) / "chunks" / f"{start:%Y%m%d_%H%M%S}_{end:%Y%m%d_%H%M%S}.pkl.gz"
