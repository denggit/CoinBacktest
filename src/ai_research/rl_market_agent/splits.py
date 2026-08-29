#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Purged walk-forward slicing for R01+.

The R00 monthly shard boundary is a storage boundary, not an information
boundary.  A row near the end of a training/calibration/OOS interval is only
safe when the *entire* forward label horizon remains inside that interval.
This module centralizes that rule so 2025 labels can never silently read into
the sealed 2026 holdout.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .dataset import DatasetCatalog


@dataclass(frozen=True)
class PurgedWindow:
    name: str
    start: pd.Timestamp
    end_exclusive: pd.Timestamp
    horizon_minutes: int

    @property
    def last_safe_decision(self) -> pd.Timestamp:
        # R00 labels include the entry minute as minute 1, therefore an h-minute
        # label consumes bars [t, t + h - 1min].
        return self.end_exclusive - pd.Timedelta(minutes=self.horizon_minutes)

    def mask(self, timestamps_ns: np.ndarray) -> np.ndarray:
        ts = pd.to_datetime(np.asarray(timestamps_ns, dtype=np.int64), unit="ns")
        label_end = ts + pd.Timedelta(minutes=self.horizon_minutes - 1)
        return np.asarray(
            (ts >= self.start) & (ts < self.end_exclusive) & (label_end < self.end_exclusive),
            dtype=bool,
        )


@dataclass(frozen=True)
class LoadedWindow:
    name: str
    features: np.ndarray
    labels: np.ndarray
    timestamps_ns: np.ndarray
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]
    rows_before_purge: int
    rows_after_purge: int


def make_purged_window(
    name: str,
    start: str | pd.Timestamp,
    end_exclusive: str | pd.Timestamp,
    horizon_minutes: int,
) -> PurgedWindow:
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end_exclusive)
    h = int(horizon_minutes)
    if start_ts.tzinfo is not None:
        start_ts = start_ts.tz_localize(None)
    if end_ts.tzinfo is not None:
        end_ts = end_ts.tz_localize(None)
    if h <= 0:
        raise ValueError("horizon_minutes must be positive")
    if end_ts <= start_ts:
        raise ValueError("end_exclusive must be after start")
    return PurgedWindow(name=name, start=start_ts, end_exclusive=end_ts, horizon_minutes=h)


def load_purged_window(
    catalog: DatasetCatalog,
    window: PurgedWindow,
    *,
    feature_names: Iterable[str] | None = None,
    label_names: Iterable[str] | None = None,
    sealed_holdout_start: str | pd.Timestamp | None = None,
) -> LoadedWindow:
    """Load an exact, horizon-safe window from monthly R00 shards.

    ``DatasetCatalog`` remains sealed by default.  This function additionally
    refuses any requested model window that reaches the sealed boundary.  The
    caller must use a separately constructed ``allow_sealed=True`` catalogue
    only for a future explicit final holdout audit.
    """

    if sealed_holdout_start is not None and not catalog.allow_sealed:
        seal = pd.Timestamp(sealed_holdout_start)
        if seal.tzinfo is not None:
            seal = seal.tz_localize(None)
        if window.end_exclusive > seal:
            raise PermissionError(
                f"window {window.name} ends at {window.end_exclusive}, beyond sealed holdout start {seal}"
            )

    wanted_features = None if feature_names is None else tuple(feature_names)
    wanted_labels = None if label_names is None else tuple(label_names)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    tss: list[np.ndarray] = []
    rows_before = 0
    base_feature_names: tuple[str, ...] | None = None
    base_label_names: tuple[str, ...] | None = None

    for shard_id in catalog.shard_ids():
        # Monthly ids sort lexicographically. Avoid opening obviously unrelated
        # shards and, critically, never call load() on a sealed month.
        month = pd.Period(shard_id, freq="M")
        shard_start = month.start_time
        shard_end_exclusive = (month + 1).start_time
        if shard_end_exclusive <= window.start or shard_start >= window.end_exclusive:
            continue
        shard = catalog.load(shard_id)
        base_feature_names = base_feature_names or shard.feature_names
        base_label_names = base_label_names or shard.label_names
        if shard.feature_names != base_feature_names or shard.label_names != base_label_names:
            raise ValueError(f"schema drift detected in shard {shard_id}")
        mask = window.mask(shard.timestamps_ns)
        rows_before += int(
            ((pd.to_datetime(shard.timestamps_ns, unit="ns") >= window.start)
             & (pd.to_datetime(shard.timestamps_ns, unit="ns") < window.end_exclusive)).sum()
        )
        if not mask.any():
            continue
        x = np.asarray(shard.features[mask], dtype=np.float32)
        y = np.asarray(shard.labels[mask], dtype=np.float32)
        ts = np.asarray(shard.timestamps_ns[mask], dtype=np.int64)
        xs.append(x)
        ys.append(y)
        tss.append(ts)

    if base_feature_names is None or base_label_names is None:
        raise FileNotFoundError(f"no R00 shards overlap window {window.name}")

    feature_idx = np.arange(len(base_feature_names))
    label_idx = np.arange(len(base_label_names))
    out_feature_names = base_feature_names
    out_label_names = base_label_names
    if wanted_features is not None:
        lookup = {name: i for i, name in enumerate(base_feature_names)}
        missing = [name for name in wanted_features if name not in lookup]
        if missing:
            raise KeyError(f"missing features: {missing[:10]}")
        feature_idx = np.asarray([lookup[name] for name in wanted_features], dtype=np.int64)
        out_feature_names = wanted_features
    if wanted_labels is not None:
        lookup = {name: i for i, name in enumerate(base_label_names)}
        missing = [name for name in wanted_labels if name not in lookup]
        if missing:
            raise KeyError(f"missing labels: {missing[:10]}")
        label_idx = np.asarray([lookup[name] for name in wanted_labels], dtype=np.int64)
        out_label_names = wanted_labels

    if not xs:
        return LoadedWindow(
            name=window.name,
            features=np.empty((0, len(feature_idx)), dtype=np.float32),
            labels=np.empty((0, len(label_idx)), dtype=np.float32),
            timestamps_ns=np.empty(0, dtype=np.int64),
            feature_names=out_feature_names,
            label_names=out_label_names,
            rows_before_purge=rows_before,
            rows_after_purge=0,
        )

    x_all = np.concatenate(xs, axis=0)[:, feature_idx]
    y_all = np.concatenate(ys, axis=0)[:, label_idx]
    ts_all = np.concatenate(tss, axis=0)
    order = np.argsort(ts_all, kind="stable")
    return LoadedWindow(
        name=window.name,
        features=x_all[order],
        labels=y_all[order],
        timestamps_ns=ts_all[order],
        feature_names=out_feature_names,
        label_names=out_label_names,
        rows_before_purge=rows_before,
        rows_after_purge=int(len(ts_all)),
    )
