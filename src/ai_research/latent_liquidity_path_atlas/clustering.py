#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Discovery-only clustering of pre-event paths with bounded memory."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import RobustScaler

from .config import LatentLiquidityPathAtlasConfig
from .features import model_feature_columns
from src.research_common.progress import ProgressReporter


@dataclass
class PathClusterModel:
    columns: tuple[str, ...]
    medians: np.ndarray
    scaler: RobustScaler
    model: MiniBatchKMeans
    train_rows: int
    eligible_train_rows: int

    def assign_batch(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        matrix = frame.loc[:, self.columns].to_numpy(dtype=np.float32, copy=True)
        invalid = ~np.isfinite(matrix)
        if invalid.any():
            matrix[invalid] = np.take(self.medians, np.where(invalid)[1])
        scaled = self.scaler.transform(matrix)
        labels = self.model.predict(scaled)
        distance = self.model.transform(scaled).min(axis=1)
        return labels.astype(np.int16), distance.astype(np.float32)


def _stratified_training_positions(
    features: pd.DataFrame,
    eligible_positions: np.ndarray,
    cap: int,
) -> np.ndarray:
    if len(eligible_positions) <= cap:
        return eligible_positions
    meta = features.iloc[eligible_positions][["event_time", "event_side"]].copy()
    meta["_year"] = pd.to_datetime(meta["event_time"]).dt.year.astype(np.int16)
    meta["_position"] = eligible_positions
    groups = list(meta.groupby(["_year", "event_side"], sort=True, dropna=False))
    per_group = max(1, cap // max(1, len(groups)))
    selected: list[int] = []
    for _, group in groups:
        take = min(len(group), per_group)
        offsets = np.linspace(0, len(group) - 1, take, dtype=np.int64)
        selected.extend(group.iloc[offsets]["_position"].astype(int).tolist())
    if len(selected) < cap:
        remaining = np.setdiff1d(eligible_positions, np.asarray(selected, dtype=np.int64), assume_unique=False)
        if len(remaining):
            offsets = np.linspace(0, len(remaining) - 1, min(cap - len(selected), len(remaining)), dtype=np.int64)
            selected.extend(remaining[offsets].astype(int).tolist())
    return np.asarray(sorted(set(selected[:cap])), dtype=np.int64)


def fit_path_clusters(
    features: pd.DataFrame,
    config: LatentLiquidityPathAtlasConfig,
) -> PathClusterModel | None:
    if features.empty:
        return None
    columns = model_feature_columns(features)
    if not columns:
        return None
    cutoff = np.datetime64(pd.Timestamp(config.cluster_train_end), "ns")
    event_ns = pd.to_datetime(features["event_time"]).to_numpy(dtype="datetime64[ns]")
    eligible_positions = np.flatnonzero(event_ns <= cutoff)
    eligible_rows = len(eligible_positions)
    if eligible_rows < config.minimum_cluster_rows:
        return None
    sample_positions = _stratified_training_positions(
        features,
        eligible_positions,
        int(config.cluster_train_sample_cap),
    )
    matrix = features.iloc[sample_positions].loc[:, columns].to_numpy(dtype=np.float32, copy=True)
    medians = np.nanmedian(matrix, axis=0).astype(np.float32, copy=False)
    medians[~np.isfinite(medians)] = 0.0
    invalid = ~np.isfinite(matrix)
    if invalid.any():
        matrix[invalid] = np.take(medians, np.where(invalid)[1])
    scaler = RobustScaler(quantile_range=(10.0, 90.0), copy=False)
    scaled = scaler.fit_transform(matrix)
    model = MiniBatchKMeans(
        n_clusters=config.cluster_count,
        random_state=config.random_state,
        batch_size=min(4096, max(256, len(sample_positions) // 8)),
        n_init=10,
        max_iter=300,
        reassignment_ratio=0.01,
    )
    model.fit(scaled)
    return PathClusterModel(
        columns=columns,
        medians=medians,
        scaler=scaler,
        model=model,
        train_rows=len(sample_positions),
        eligible_train_rows=eligible_rows,
    )


def assign_path_clusters(
    features: pd.DataFrame,
    model: PathClusterModel | None,
    *,
    batch_rows: int = 50_000,
    progress: bool = False,
) -> pd.DataFrame:
    identity = ("event_id", "event_time", "event_side", "period")
    out = features.reindex(columns=identity).copy()
    if model is None or features.empty:
        out["path_cluster"] = np.full(len(out), -1, dtype=np.int16)
        out["cluster_distance"] = np.full(len(out), np.nan, dtype=np.float32)
        return out
    labels = np.empty(len(features), dtype=np.int16)
    distances = np.empty(len(features), dtype=np.float32)
    step = max(1, int(batch_rows))
    total_batches = (len(features) + step - 1) // step
    reporter = ProgressReporter(
        label="[latent-liquidity-atlas] cluster assignment",
        total=total_batches,
        every=1,
        enabled=progress,
    )
    for batch_number, start in enumerate(range(0, len(features), step), start=1):
        stop = min(len(features), start + step)
        batch_labels, batch_distances = model.assign_batch(features.iloc[start:stop])
        labels[start:stop] = batch_labels
        distances[start:stop] = batch_distances
        reporter.update(batch_number)
    reporter.close()
    out["path_cluster"] = labels
    out["cluster_distance"] = distances
    return out
