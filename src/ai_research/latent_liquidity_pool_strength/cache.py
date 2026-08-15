from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.ai_research.latent_liquidity_pool_forecast.cache import dataset_cache_path as r02_dataset_cache_path, episode_cache_path as r02_episode_cache_path
from src.ai_research.latent_liquidity_pool_forecast.config import DEFAULT_CONFIG as R02_CONFIG
from .config import LatentLiquidityPoolStrengthConfig


def cache_key(config: LatentLiquidityPoolStrengthConfig) -> str:
    payload = config.to_dict().copy()
    for key, path in (
        ("r02_spatial", r02_dataset_cache_path(R02_CONFIG)),
        ("r02_episodes", r02_episode_cache_path(R02_CONFIG)),
    ):
        try:
            stat = path.stat()
            payload[key] = [str(path), int(stat.st_size), int(stat.st_mtime_ns)]
        except OSError:
            payload[key] = [str(path), "MISSING"]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def dataset_cache_path(config: LatentLiquidityPoolStrengthConfig) -> Path:
    return config.cache_path / cache_key(config) / "strength_dataset.pkl.gz"


def save_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_pickle(tmp, compression={"method": "gzip", "compresslevel": 1, "mtime": 1})
    tmp.replace(path)


def load_frame(path: Path) -> pd.DataFrame:
    value = pd.read_pickle(path, compression="gzip")
    if not isinstance(value, pd.DataFrame):
        raise ValueError(f"invalid R02.1 cache: {path}")
    return value
