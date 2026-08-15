#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Narrow, deterministic R01.1 Episode sampling for R01.3."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from typing import Iterator

import numpy as np
import pandas as pd

from src.ai_research.latent_liquidity_execution_audit.config import StablePathExecutionAuditConfig
from src.ai_research.latent_liquidity_execution_audit.source import (
    SourcePaths,
    resolve_source_paths,
    source_gate,
)
from src.research_common.progress import ProgressReporter

from .config import AbsorptionModelConfig

_FEATURE_COLUMNS = (
    "event_id",
    "event_time",
    "event_side",
    "period",
    "release_episode_id",
    "release_episode_ordinal",
)
_LABEL_COLUMNS = ("event_id", "event_reference_price", "outcome_type")
_ASSIGNMENT_COLUMNS = ("event_id", "path_cluster", "cluster_distance")


@dataclass
class AbsorptionSourceScanResult:
    source_gate: pd.DataFrame
    replay_samples: pd.DataFrame
    scanned_rows: int


def as_scan_config(config: AbsorptionModelConfig) -> StablePathExecutionAuditConfig:
    base = StablePathExecutionAuditConfig()
    return replace(
        base,
        source_report_dir=config.source_report_dir,
        cache_dir=config.cache_dir,
        source_feature_file=config.source_feature_file,
        source_label_file=config.source_label_file,
        source_assignment_file=config.source_assignment_file,
        source_manifest_file=config.source_manifest_file,
        source_causal_audit_file=config.source_causal_audit_file,
        csv_read_chunk_rows=config.csv_read_chunk_rows,
        target_clusters=config.target_clusters,
        target_cluster_roles=config.target_cluster_roles,
        profile_sample_per_stratum=100,
        replay_sample_per_stratum=config.replay_sample_per_stratum,
        random_state=config.random_state,
        pre_replay_seconds=config.pre_replay_seconds,
        post_replay_seconds=config.post_replay_seconds,
        replay_max_fill_gap_seconds=config.replay_max_fill_gap_seconds,
        roundtrip_cost_bp=config.roundtrip_cost_bp,
        cost_multipliers=config.cost_multipliers,
        entry_delay_seconds=config.entry_delay_seconds,
        periods=config.periods,
    )


def source_paths(config: AbsorptionModelConfig) -> SourcePaths:
    return resolve_source_paths(as_scan_config(config))


class _PrioritySamples:
    def __init__(self, cap: int):
        self.cap = int(cap)
        self.frames: dict[tuple[object, ...], pd.DataFrame] = {}

    @staticmethod
    def _priority(ids: pd.Series) -> np.ndarray:
        return pd.util.hash_pandas_object(ids.astype(str), index=False).to_numpy(dtype=np.uint64)

    def add(self, key: tuple[object, ...], frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        candidate = frame.copy()
        candidate["_priority"] = self._priority(candidate["event_id"])
        existing = self.frames.get(key)
        if existing is not None and not existing.empty:
            candidate = pd.concat([existing, candidate], ignore_index=True, copy=False)
        if len(candidate) > self.cap:
            candidate = candidate.nsmallest(self.cap, "_priority", keep="first")
        self.frames[key] = candidate.reset_index(drop=True)

    def result(self) -> pd.DataFrame:
        if not self.frames:
            return pd.DataFrame()
        frames = [frame.drop(columns="_priority", errors="ignore") for frame in self.frames.values()]
        out = pd.concat(frames, ignore_index=True, copy=False)
        return out.drop_duplicates("event_id").sort_values("event_time", kind="mergesort").reset_index(drop=True)


def _readers(paths: SourcePaths, chunk_rows: int) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    feature_reader = pd.read_csv(
        paths.feature,
        usecols=lambda name: name in _FEATURE_COLUMNS,
        chunksize=chunk_rows,
        low_memory=False,
    )
    label_reader = pd.read_csv(
        paths.label,
        usecols=lambda name: name in _LABEL_COLUMNS,
        chunksize=chunk_rows,
        low_memory=False,
    )
    assignment_reader = pd.read_csv(
        paths.assignment,
        usecols=lambda name: name in _ASSIGNMENT_COLUMNS,
        chunksize=chunk_rows,
        low_memory=False,
    )
    while True:
        try:
            feature = next(feature_reader)
        except StopIteration:
            try:
                next(label_reader)
                raise RuntimeError("R01.1 label table has more rows than feature table")
            except StopIteration:
                pass
            try:
                next(assignment_reader)
                raise RuntimeError("R01.1 assignment table has more rows than feature table")
            except StopIteration:
                pass
            break
        try:
            label = next(label_reader)
            assignment = next(assignment_reader)
        except StopIteration as exc:
            raise RuntimeError("R01.1 narrow source tables have unequal row counts") from exc
        if len(feature) != len(label) or len(feature) != len(assignment):
            raise RuntimeError("R01.1 narrow chunk row-count mismatch")
        ids = feature["event_id"].astype(str).to_numpy()
        if not np.array_equal(ids, label["event_id"].astype(str).to_numpy()):
            raise RuntimeError("R01.1 feature/label event alignment mismatch")
        if not np.array_equal(ids, assignment["event_id"].astype(str).to_numpy()):
            raise RuntimeError("R01.1 feature/assignment event alignment mismatch")
        yield feature, label, assignment


def scan_sources(config: AbsorptionModelConfig, *, progress: bool = True) -> AbsorptionSourceScanResult:
    paths = source_paths(config)
    gate = source_gate(as_scan_config(config), paths)
    failures = gate.loc[gate["status"].eq("FAIL"), "check"].tolist()
    if failures:
        raise RuntimeError(f"R01.3 source gate failed: {failures}")
    store = _PrioritySamples(config.replay_sample_per_stratum)
    scanned = 0
    estimated = 1
    if paths.manifest.exists():
        try:
            manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
            total_rows = int(manifest.get("feature_rows", manifest.get("joined_rows", 0)))
            if total_rows > 0:
                estimated = max(1, int(math.ceil(total_rows / config.csv_read_chunk_rows)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            estimated = 1
    reporter = ProgressReporter(
        label="[latent-liquidity-r01.3] stream narrow R01.1 Episode metadata",
        total=estimated,
        every=1,
        enabled=progress,
    )
    for chunk_number, (feature, label, assignment) in enumerate(_readers(paths, config.csv_read_chunk_rows), start=1):
        scanned += len(feature)
        joined = pd.DataFrame(
            {
                "event_id": feature["event_id"].astype(str),
                "event_time": pd.to_datetime(feature["event_time"], errors="coerce"),
                "event_side": feature["event_side"].astype(str),
                "period": feature["period"].astype(str),
                "release_episode_id": feature["release_episode_id"].astype(str),
                "release_episode_ordinal": pd.to_numeric(feature["release_episode_ordinal"], errors="coerce"),
                "path_cluster": pd.to_numeric(assignment["path_cluster"], errors="coerce").fillna(-1).astype(np.int16),
                "cluster_distance": pd.to_numeric(assignment["cluster_distance"], errors="coerce"),
                "event_reference_price": pd.to_numeric(label["event_reference_price"], errors="coerce"),
                "outcome_type": label["outcome_type"].astype(str),
            }
        )
        candidates = joined.loc[
            joined["path_cluster"].isin(config.target_clusters)
            & joined["release_episode_ordinal"].eq(1)
            & joined["event_time"].notna()
            & joined["event_reference_price"].gt(0)
        ]
        for keys, group in candidates.groupby(["path_cluster", "event_side", "period"], sort=False):
            store.add(tuple(keys), group)
        reporter.update(min(chunk_number, estimated))
    reporter.close()
    replay = store.result()
    return AbsorptionSourceScanResult(source_gate=gate, replay_samples=replay, scanned_rows=scanned)
