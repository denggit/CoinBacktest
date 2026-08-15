#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Small deterministic caches for R02.2."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import FirstTouchLiquidityRankingConfig

CACHE_VERSION = "r02_2_first_touch_ranking_v1"


def dataset_cache_path(config: FirstTouchLiquidityRankingConfig) -> Path:
    return config.cache_path / f"first_touch_dataset_{CACHE_VERSION}.pkl.gz"


def chunk_cache_path(config: FirstTouchLiquidityRankingConfig, start: pd.Timestamp, end: pd.Timestamp) -> Path:
    return config.cache_path / "touch_chunks" / f"{start:%Y%m%d}_{end:%Y%m%d}_{CACHE_VERSION}.pkl.gz"


def save_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(path, compression="gzip")


def load_frame(path: Path) -> pd.DataFrame:
    return pd.read_pickle(path, compression="gzip")
