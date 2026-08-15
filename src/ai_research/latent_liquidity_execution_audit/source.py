#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stream and validate the large R01.1 source tables without loading them whole."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from .config import StablePathExecutionAuditConfig
from src.research_common.progress import ProgressReporter

_FEATURE_PREFIXES = (
    "path_",
    "distance_prior_",
    "position_in_prior_",
    "macro_",
    "unswept_",
)
_FEATURE_BLOCKED_SUFFIXES = ("_time", "available_time")
_META_FEATURE_COLUMNS = (
    "event_id",
    "event_time",
    "event_side",
    "period",
    "release_episode_id",
    "release_episode_ordinal",
    "release_episode_size",
    "release_episode_weight",
)
_LABEL_COLUMNS = (
    "event_id",
    "event_time",
    "event_side",
    "event_reference_price",
    "future_extension_bp",
    "future_time_to_extreme_seconds",
    "future_immediate_reversal_bp",
    "future_reversal_after_extreme_bp",
    "future_acceptance_fraction_60s",
    "future_stable_after_extreme",
    "outcome_type",
    "favorable_reversal",
)
_ASSIGNMENT_COLUMNS = (
    "event_id",
    "event_time",
    "event_side",
    "period",
    "path_cluster",
    "cluster_distance",
)


@dataclass(frozen=True)
class SourcePaths:
    root: Path
    feature: Path
    label: Path
    assignment: Path
    manifest: Path
    causal_audit: Path


@dataclass
class SourceScanResult:
    source_gate: pd.DataFrame
    episode_rows: pd.DataFrame
    profile_samples: dict[tuple[object, ...], pd.DataFrame]
    replay_samples: pd.DataFrame
    feature_columns: tuple[str, ...]
    scanned_rows: int


def resolve_source_paths(config: StablePathExecutionAuditConfig) -> SourcePaths:
    root = config.source_report_path
    return SourcePaths(
        root=root,
        feature=root / config.source_feature_file,
        label=root / config.source_label_file,
        assignment=root / config.source_assignment_file,
        manifest=root / config.source_manifest_file,
        causal_audit=root / config.source_causal_audit_file,
    )


def _sampled_sha256(path: Path, sample_bytes: int = 1024 * 1024) -> str:
    """Fast source fingerprint: file size plus first/last sample, not a full multi-GB reread."""
    digest = hashlib.sha256()
    size = path.stat().st_size
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as handle:
        digest.update(handle.read(sample_bytes))
        if size > sample_bytes:
            handle.seek(max(0, size - sample_bytes))
            digest.update(handle.read(sample_bytes))
    return digest.hexdigest()


def source_gate(config: StablePathExecutionAuditConfig, paths: SourcePaths) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label, path in (
        ("feature_table", paths.feature),
        ("label_table", paths.label),
        ("assignment_table", paths.assignment),
        ("manifest", paths.manifest),
        ("causal_audit", paths.causal_audit),
    ):
        rows.append(
            {
                "check": f"{label}_exists",
                "value": str(path),
                "status": "PASS" if path.exists() and path.stat().st_size > 0 else "FAIL",
            }
        )
    if paths.manifest.exists():
        try:
            manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}
        stage = str(manifest.get("stage_id", manifest.get("stage", "")))
        rows.append(
            {
                "check": "source_stage_is_r01_1",
                "value": stage,
                "status": "PASS" if "R01.1" in stage or "R01_1" in stage else "WARN",
            }
        )
    if paths.causal_audit.exists():
        audit = pd.read_csv(paths.causal_audit)
        failures = int(audit.get("status", pd.Series(dtype=str)).astype(str).eq("FAIL").sum())
        rows.append(
            {
                "check": "source_causal_audit_passed",
                "value": failures,
                "status": "PASS" if failures == 0 else "FAIL",
            }
        )
    for label, path in (("feature", paths.feature), ("label", paths.label), ("assignment", paths.assignment)):
        if path.exists() and path.stat().st_size > 0:
            rows.append(
                {
                    "check": f"{label}_sampled_sha256",
                    "value": _sampled_sha256(path),
                    "status": "INFO",
                }
            )
    return pd.DataFrame(rows)


def discover_feature_columns(feature_path: Path) -> tuple[str, ...]:
    header = pd.read_csv(feature_path, nrows=0).columns.tolist()
    columns = []
    for name in header:
        if not name.startswith(_FEATURE_PREFIXES):
            continue
        if name in {"macro_bar_start_time", "macro_available_time", "macro_pre_event_close", "unswept_max_level_available_time"}:
            continue
        if any(name.endswith(suffix) for suffix in _FEATURE_BLOCKED_SUFFIXES):
            continue
        columns.append(name)
    if not columns:
        raise RuntimeError("R01.2 found no model/path feature columns in the R01.1 feature table")
    return tuple(columns)


class _PrioritySamples:
    """Deterministic smallest-hash samples, avoiding chronological truncation."""

    def __init__(self, cap: int):
        self.cap = int(cap)
        self.frames: dict[tuple[object, ...], pd.DataFrame] = {}

    @staticmethod
    def _priorities(event_ids: pd.Series) -> np.ndarray:
        return pd.util.hash_pandas_object(event_ids.astype(str), index=False).to_numpy(dtype=np.uint64)

    def add(self, key: tuple[object, ...], frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        candidate = frame.copy()
        candidate["_sample_priority"] = self._priorities(candidate["event_id"])
        existing = self.frames.get(key)
        if existing is not None and not existing.empty:
            candidate = pd.concat([existing, candidate], ignore_index=True, copy=False)
        if len(candidate) > self.cap:
            candidate = candidate.nsmallest(self.cap, "_sample_priority", keep="first")
        self.frames[key] = candidate.reset_index(drop=True)

    def clean(self) -> dict[tuple[object, ...], pd.DataFrame]:
        return {
            key: frame.drop(columns="_sample_priority", errors="ignore").reset_index(drop=True)
            for key, frame in self.frames.items()
        }


def _aligned_readers(paths: SourcePaths, usecols_feature: list[str], chunk_rows: int) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    feature_reader = pd.read_csv(paths.feature, usecols=usecols_feature, chunksize=chunk_rows, low_memory=False)
    label_reader = pd.read_csv(paths.label, usecols=lambda name: name in _LABEL_COLUMNS, chunksize=chunk_rows, low_memory=False)
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
            raise RuntimeError("R01.1 full tables have unequal row counts") from exc
        if len(feature) != len(label) or len(feature) != len(assignment):
            raise RuntimeError("R01.1 chunk alignment row-count mismatch")
        ids = feature["event_id"].astype(str).to_numpy()
        if not np.array_equal(ids, label["event_id"].astype(str).to_numpy()):
            raise RuntimeError("R01.1 feature/label event_id alignment mismatch")
        if not np.array_equal(ids, assignment["event_id"].astype(str).to_numpy()):
            raise RuntimeError("R01.1 feature/assignment event_id alignment mismatch")
        yield feature, label, assignment


def scan_source_tables(config: StablePathExecutionAuditConfig, progress: bool = True) -> SourceScanResult:
    paths = resolve_source_paths(config)
    gate = source_gate(config, paths)
    failures = gate.loc[gate["status"].eq("FAIL"), "check"].tolist()
    if failures:
        raise RuntimeError(f"R01.2 source gate failed: {failures}")
    feature_columns = discover_feature_columns(paths.feature)
    feature_usecols = list(dict.fromkeys([*_META_FEATURE_COLUMNS, *feature_columns]))
    profile_store = _PrioritySamples(config.profile_sample_per_stratum)
    replay_store = _PrioritySamples(config.replay_sample_per_stratum)
    episode_parts: list[pd.DataFrame] = []
    scanned = 0
    estimated_chunks = max(1, int(paths.feature.stat().st_size // (32 * 1024 * 1024)) + 1)
    reporter = ProgressReporter(
        label="[latent-liquidity-r01.2] stream R01.1 tables",
        total=estimated_chunks,
        every=1,
        enabled=progress,
    )
    for chunk_number, (feature, label, assignment) in enumerate(
        _aligned_readers(paths, feature_usecols, config.csv_read_chunk_rows), start=1
    ):
        scanned += len(feature)
        joined_meta = pd.DataFrame(
            {
                "event_id": feature["event_id"].astype(str),
                "event_time": pd.to_datetime(feature["event_time"], errors="coerce"),
                "event_side": feature["event_side"].astype(str),
                "period": feature["period"].astype(str),
                "release_episode_id": feature["release_episode_id"].astype(str),
                "release_episode_ordinal": pd.to_numeric(feature["release_episode_ordinal"], errors="coerce"),
                "release_episode_size": pd.to_numeric(feature["release_episode_size"], errors="coerce"),
                "path_cluster": pd.to_numeric(assignment["path_cluster"], errors="coerce").fillna(-1).astype(np.int16),
                "cluster_distance": pd.to_numeric(assignment.get("cluster_distance"), errors="coerce"),
                "event_reference_price": pd.to_numeric(label.get("event_reference_price"), errors="coerce"),
                "future_extension_bp": pd.to_numeric(label.get("future_extension_bp"), errors="coerce"),
                "future_time_to_extreme_seconds": pd.to_numeric(label.get("future_time_to_extreme_seconds"), errors="coerce"),
                "future_reversal_after_extreme_bp": pd.to_numeric(label.get("future_reversal_after_extreme_bp"), errors="coerce"),
                "future_acceptance_fraction_60s": pd.to_numeric(label.get("future_acceptance_fraction_60s"), errors="coerce"),
                "outcome_type": label["outcome_type"].astype(str),
                "favorable_reversal": label["favorable_reversal"].astype(str).str.lower().isin({"true", "1"}),
            }
        )
        target_mask = joined_meta["path_cluster"].isin(config.target_clusters)
        first_mask = joined_meta["release_episode_ordinal"].eq(1)
        episode = joined_meta.loc[target_mask & first_mask].copy()
        if not episode.empty:
            episode_parts.append(episode)
            replay_store.add(
                ("ALL_TARGETS",),
                episode[
                    [
                        "event_id",
                        "event_time",
                        "event_side",
                        "period",
                        "release_episode_id",
                        "path_cluster",
                        "cluster_distance",
                        "event_reference_price",
                        "outcome_type",
                    ]
                ],
            )
            for keys, group in episode.groupby(["path_cluster", "event_side", "period"], sort=False):
                replay_store.add(tuple(keys), group)

        profile_frame = feature.loc[:, ["event_id", *feature_columns]].copy()
        profile_frame.insert(1, "event_side", joined_meta["event_side"].to_numpy())
        profile_frame.insert(2, "period", joined_meta["period"].to_numpy())
        profile_frame.insert(3, "path_cluster", joined_meta["path_cluster"].to_numpy())
        for keys, positions in joined_meta.groupby(["event_side", "period"], sort=False).groups.items():
            profile_store.add(("BASE", *keys), profile_frame.loc[positions])
        target_positions = joined_meta.index[target_mask]
        if len(target_positions):
            target_profile = profile_frame.loc[target_positions]
            for keys, positions in joined_meta.loc[target_positions].groupby(
                ["path_cluster", "event_side", "period"], sort=False
            ).groups.items():
                profile_store.add(tuple(keys), target_profile.loc[positions])
        reporter.update(min(chunk_number, estimated_chunks))
    reporter.close()
    episodes = pd.concat(episode_parts, ignore_index=True, copy=False) if episode_parts else pd.DataFrame()
    if not episodes.empty:
        episodes = episodes.sort_values(["event_time", "event_side"], kind="mergesort").reset_index(drop=True)
        if episodes["release_episode_id"].duplicated().any():
            raise RuntimeError("R01.2 episode representative rows are not unique")
    replay_frames = replay_store.clean()
    replay = pd.concat(
        [frame.assign(_sample_key="|".join(map(str, key))) for key, frame in replay_frames.items() if key != ("ALL_TARGETS",)],
        ignore_index=True,
        copy=False,
    ) if replay_frames else pd.DataFrame()
    if not replay.empty:
        replay = replay.drop_duplicates("event_id").sort_values("event_time", kind="mergesort").reset_index(drop=True)
    return SourceScanResult(
        source_gate=gate,
        episode_rows=episodes,
        profile_samples=profile_store.clean(),
        replay_samples=replay,
        feature_columns=feature_columns,
        scanned_rows=scanned,
    )
