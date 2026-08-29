#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Read-only R00 shard catalogue with sealed-holdout protection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


@dataclass(frozen=True)
class DatasetShard:
    shard_id: str
    features: np.ndarray
    labels: np.ndarray
    timestamps_ns: np.ndarray
    flags: np.ndarray
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]
    flag_names: tuple[str, ...]
    sealed_holdout: bool


class DatasetCatalog:
    """Memory-mapped dataset reader.

    ``allow_sealed`` defaults to False deliberately. Future R01/R02 training
    code must opt in explicitly before it can read a shard wholly inside the
    sealed holdout period.
    """

    def __init__(self, cache_dir: str | Path, *, project_root: str | Path, allow_sealed: bool = False) -> None:
        self.cache_dir = Path(cache_dir)
        self.project_root = Path(project_root)
        self.allow_sealed = bool(allow_sealed)

    def shard_ids(self) -> list[str]:
        if not self.cache_dir.exists():
            return []
        return sorted(p.name for p in self.cache_dir.iterdir() if p.is_dir() and (p / "meta.json").exists())

    def _resolve(self, value: str) -> Path:
        p = Path(value)
        return p if p.is_absolute() else self.project_root / p

    def load(self, shard_id: str) -> DatasetShard:
        meta_path = self.cache_dir / shard_id / "meta.json"
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
        record = payload["record"]
        sealed = bool(record.get("sealed_holdout", False))
        if sealed and not self.allow_sealed:
            raise PermissionError(
                f"shard {shard_id} is sealed holdout; construct DatasetCatalog(..., allow_sealed=True) only for the final audit"
            )
        return DatasetShard(
            shard_id=shard_id,
            features=np.load(self._resolve(record["features_path"]), mmap_mode="r", allow_pickle=False),
            labels=np.load(self._resolve(record["labels_path"]), mmap_mode="r", allow_pickle=False),
            timestamps_ns=np.load(self._resolve(record["timestamps_path"]), mmap_mode="r", allow_pickle=False),
            flags=np.load(self._resolve(record["flags_path"]), mmap_mode="r", allow_pickle=False),
            feature_names=tuple(payload["feature_names"]),
            label_names=tuple(payload["label_names"]),
            flag_names=tuple(payload["flag_names"]),
            sealed_holdout=sealed,
        )

    def iter_unsealed_storage_shards(self) -> Iterator[DatasetShard]:
        """Iterate unsealed monthly storage shards without claiming label safety.

        Monthly shard boundaries are not sufficient for supervised training:
        forward labels near a right boundary can extend into the next fold or
        into the sealed holdout. Model code must use ``splits.load_purged_window``.
        """
        for shard_id in self.shard_ids():
            meta = json.loads((self.cache_dir / shard_id / "meta.json").read_text(encoding="utf-8"))
            if bool(meta["record"].get("sealed_holdout", False)):
                continue
            yield self.load(shard_id)

    def iter_training_shards(self) -> Iterator[DatasetShard]:
        """Blocked legacy API: whole monthly shards are not horizon-safe."""
        raise RuntimeError(
            "iter_training_shards() is unsafe for forward-label training; "
            "use splits.load_purged_window(...) with an explicit horizon and right boundary"
        )
