#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stream R01.1 labels into one representative row per release episode."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

from src.ai_research.latent_liquidity_execution_audit.source import SourcePaths, source_gate
from src.research_common.progress import ProgressReporter

from .config import EconomicCeilingConfig


@dataclass(frozen=True)
class EconomicCeilingSource:
    source_gate: pd.DataFrame
    episodes: pd.DataFrame
    scanned_rows: int


def _paths(config: EconomicCeilingConfig) -> SourcePaths:
    root = config.source_report_path
    return SourcePaths(
        root=root,
        feature=root / config.source_feature_file,
        label=root / config.source_label_file,
        assignment=root / config.source_assignment_file,
        manifest=root / config.source_manifest_file,
        causal_audit=root / config.source_causal_audit_file,
    )


def _required_label_columns(config: EconomicCeilingConfig) -> tuple[str, ...]:
    cols = [
        "event_id", "event_time", "event_side", "event_reference_price",
        "future_extension_bp", "future_time_to_extreme_seconds",
        "future_reversal_after_extreme_bp", "future_acceptance_fraction_60s",
        "outcome_type", "favorable_reversal",
    ]
    for horizon in config.horizons_seconds:
        cols.extend(
            [
                f"future_same_direction_extension_{horizon}s_bp",
                f"future_opposite_excursion_{horizon}s_bp",
                f"future_close_return_{horizon}s_bp",
            ]
        )
    return tuple(cols)


def _aligned_readers(paths: SourcePaths, config: EconomicCeilingConfig) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    feature_cols = (
        "event_id", "event_time", "event_side", "period", "release_episode_id",
        "release_episode_ordinal", "release_episode_size",
    )
    label_cols = _required_label_columns(config)
    assignment_cols = ("event_id", "path_cluster", "cluster_distance")

    feature_header = set(pd.read_csv(paths.feature, nrows=0).columns)
    label_header = set(pd.read_csv(paths.label, nrows=0).columns)
    assignment_header = set(pd.read_csv(paths.assignment, nrows=0).columns)
    missing = sorted((set(feature_cols) - feature_header) | (set(label_cols) - label_header) | (set(assignment_cols) - assignment_header))
    if missing:
        raise RuntimeError(f"R02.4 source is missing required columns: {missing}")

    fr = pd.read_csv(paths.feature, usecols=list(feature_cols), chunksize=config.csv_read_chunk_rows, low_memory=False)
    lr = pd.read_csv(paths.label, usecols=list(label_cols), chunksize=config.csv_read_chunk_rows, low_memory=False)
    ar = pd.read_csv(paths.assignment, usecols=list(assignment_cols), chunksize=config.csv_read_chunk_rows, low_memory=False)
    while True:
        try:
            f = next(fr)
        except StopIteration:
            try:
                next(lr)
                raise RuntimeError("label table has more rows than feature table")
            except StopIteration:
                pass
            try:
                next(ar)
                raise RuntimeError("assignment table has more rows than feature table")
            except StopIteration:
                pass
            break
        try:
            l = next(lr)
            a = next(ar)
        except StopIteration as exc:
            raise RuntimeError("R01.1 source tables have unequal row counts") from exc
        if not (len(f) == len(l) == len(a)):
            raise RuntimeError("R01.1 source chunk row-count mismatch")
        ids = f["event_id"].astype(str).to_numpy()
        if not (ids == l["event_id"].astype(str).to_numpy()).all() or not (ids == a["event_id"].astype(str).to_numpy()).all():
            raise RuntimeError("R01.1 source event_id alignment mismatch")
        yield f, l, a


def load_release_episode_source(config: EconomicCeilingConfig, *, progress: bool = True) -> EconomicCeilingSource:
    config.validate()
    paths = _paths(config)
    # Reuse the strong R01.2 source gate against the same frozen R01.1 artifacts.
    gate = source_gate(config, paths)  # type: ignore[arg-type]
    failures = gate.loc[gate["status"].astype(str).eq("FAIL"), "check"].tolist()
    if failures:
        raise RuntimeError(f"R02.4 upstream source gate failed: {failures}")

    parts: list[pd.DataFrame] = []
    scanned = 0
    estimated = max(1, int(paths.feature.stat().st_size // (32 * 1024 * 1024)) + 1)
    reporter = ProgressReporter("[latent-liquidity-r02.4] stream R01.1 episode labels", estimated, every=1, enabled=progress)
    for n, (f, l, a) in enumerate(_aligned_readers(paths, config), start=1):
        scanned += len(f)
        ordinal = pd.to_numeric(f["release_episode_ordinal"], errors="coerce")
        keep = ordinal.eq(1)
        if keep.any():
            fi = f.loc[keep].reset_index(drop=True)
            li = l.loc[keep].reset_index(drop=True)
            ai = a.loc[keep].reset_index(drop=True)
            out = pd.DataFrame(
                {
                    "event_id": fi["event_id"].astype(str),
                    "event_time": pd.to_datetime(fi["event_time"], errors="coerce"),
                    "event_side": fi["event_side"].astype(str),
                    "period": fi["period"].astype(str),
                    "release_episode_id": fi["release_episode_id"].astype(str),
                    "release_episode_size": pd.to_numeric(fi["release_episode_size"], errors="coerce"),
                    "event_reference_price": pd.to_numeric(li["event_reference_price"], errors="coerce"),
                    "future_extension_bp": pd.to_numeric(li["future_extension_bp"], errors="coerce"),
                    "future_time_to_extreme_seconds": pd.to_numeric(li["future_time_to_extreme_seconds"], errors="coerce"),
                    "future_reversal_after_extreme_bp": pd.to_numeric(li["future_reversal_after_extreme_bp"], errors="coerce"),
                    "future_acceptance_fraction_60s": pd.to_numeric(li["future_acceptance_fraction_60s"], errors="coerce"),
                    "outcome_type": li["outcome_type"].astype(str),
                    "favorable_reversal": li["favorable_reversal"].astype(str).str.lower().isin({"true", "1"}),
                    "path_cluster": pd.to_numeric(ai["path_cluster"], errors="coerce").fillna(-1).astype("int16"),
                    "cluster_distance": pd.to_numeric(ai["cluster_distance"], errors="coerce"),
                }
            )
            for horizon in config.horizons_seconds:
                for prefix in ("future_same_direction_extension", "future_opposite_excursion", "future_close_return"):
                    col = f"{prefix}_{horizon}s_bp"
                    out[col] = pd.to_numeric(li[col], errors="coerce")
            parts.append(out)
        reporter.update(min(n, estimated))
    reporter.close()
    episodes = pd.concat(parts, ignore_index=True, copy=False) if parts else pd.DataFrame()
    if episodes.empty:
        raise RuntimeError("R02.4 found no release-episode representatives")
    episodes = episodes.loc[episodes["event_time"].notna() & episodes["event_reference_price"].gt(0)].copy()
    episodes = episodes.sort_values(["event_time", "event_side"], kind="mergesort").reset_index(drop=True)
    if episodes["release_episode_id"].duplicated().any():
        raise RuntimeError("R02.4 release_episode_id representatives are not unique")
    return EconomicCeilingSource(source_gate=gate, episodes=episodes, scanned_rows=scanned)
