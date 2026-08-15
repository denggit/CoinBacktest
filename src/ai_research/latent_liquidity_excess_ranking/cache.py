#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic R02.3 processed-label cache."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import ExcessLiquidityRankingConfig

CACHE_VERSION = "r02_3_excess_liquidity_v1"


def dataset_cache_path(config: ExcessLiquidityRankingConfig) -> Path:
    return config.cache_path / f"excess_ranking_dataset_{CACHE_VERSION}.pkl.gz"


def save_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_pickle(temp, compression={"method": "gzip", "compresslevel": 1, "mtime": 1})
    temp.replace(path)


def load_frame(path: Path) -> pd.DataFrame:
    value = pd.read_pickle(path, compression="gzip")
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"invalid R02.3 cache: {path}")
    return value
