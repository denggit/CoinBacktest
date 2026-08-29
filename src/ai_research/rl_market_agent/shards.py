#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Portable NumPy shard storage with resumable metadata."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import ShardRecord, relative_or_absolute


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_save_npy(path: Path, values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, values, allow_pickle=False)
    os.replace(tmp, path)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


class ShardStore:
    def __init__(self, root: str | Path, *, project_root: str | Path) -> None:
        self.root = Path(root)
        self.project_root = Path(project_root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _dir(self, shard_id: str) -> Path:
        return self.root / shard_id

    def metadata_path(self, shard_id: str) -> Path:
        return self._dir(shard_id) / "meta.json"

    def exists(self, shard_id: str) -> bool:
        meta = self.metadata_path(shard_id)
        if not meta.exists():
            return False
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
            for name in ("features_path", "labels_path", "timestamps_path", "flags_path"):
                p = self.project_root / payload["record"][name]
                if not p.exists():
                    return False
            return True
        except (OSError, KeyError, json.JSONDecodeError):
            return False

    def load_metadata(self, shard_id: str) -> dict[str, Any]:
        return json.loads(self.metadata_path(shard_id).read_text(encoding="utf-8"))

    def write(
        self,
        *,
        shard_id: str,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        flags: pd.DataFrame,
        sealed_holdout: bool,
        extra_metadata: dict[str, Any],
    ) -> ShardRecord:
        if not features.index.equals(labels.index) or not features.index.equals(flags.index):
            raise ValueError("features, labels and flags must share exactly the same index")
        shard_dir = self._dir(shard_id)
        shard_dir.mkdir(parents=True, exist_ok=True)
        features_path = shard_dir / "features.npy"
        labels_path = shard_dir / "labels.npy"
        timestamps_path = shard_dir / "timestamps_ns.npy"
        flags_path = shard_dir / "flags.npy"
        _atomic_save_npy(features_path, features.to_numpy(dtype=np.float32, copy=True))
        _atomic_save_npy(labels_path, labels.to_numpy(dtype=np.float32, copy=True))
        _atomic_save_npy(timestamps_path, features.index.to_numpy(dtype="datetime64[ns]").astype(np.int64))
        _atomic_save_npy(flags_path, flags.to_numpy(dtype=np.uint8, copy=True))
        record = ShardRecord(
            shard_id=shard_id,
            start_time=str(features.index.min()) if len(features) else "",
            end_time=str(features.index.max()) if len(features) else "",
            rows=int(len(features)),
            feature_count=int(features.shape[1]),
            label_count=int(labels.shape[1]),
            features_path=relative_or_absolute(features_path, self.project_root),
            labels_path=relative_or_absolute(labels_path, self.project_root),
            timestamps_path=relative_or_absolute(timestamps_path, self.project_root),
            flags_path=relative_or_absolute(flags_path, self.project_root),
            sealed_holdout=bool(sealed_holdout),
            sha256_features=_sha256(features_path),
            sha256_labels=_sha256(labels_path),
        )
        payload = {"record": record.to_dict(), **extra_metadata}
        _atomic_json(self.metadata_path(shard_id), payload)
        return record

    @staticmethod
    def mmap_array(path: str | Path) -> np.ndarray:
        return np.load(Path(path), mmap_mode="r", allow_pickle=False)
