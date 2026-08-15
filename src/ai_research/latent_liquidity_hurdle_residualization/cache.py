#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Deterministic cache for R02.3.1 nuisance predictions and residual targets."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

from .config import HurdleResidualizationConfig

CACHE_VERSION = "r02_3_1_hurdle_residual_v1"


def dataset_cache_path(config: HurdleResidualizationConfig) -> Path:
    return config.cache_path / f"hurdle_residual_dataset_{CACHE_VERSION}.pkl.gz"


def save_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_pickle(temp, compression={"method": "gzip", "compresslevel": 1, "mtime": 1})
    temp.replace(path)


def load_frame(path: Path) -> pd.DataFrame:
    value = pd.read_pickle(path, compression="gzip")
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"invalid R02.3.1 cache: {path}")
    return value
