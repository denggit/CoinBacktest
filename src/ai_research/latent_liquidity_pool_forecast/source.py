#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R01.1 episode-label source for R02 spatial forecasting."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.ai_research.latent_liquidity_execution_audit.config import StablePathExecutionAuditConfig
from src.ai_research.latent_liquidity_execution_audit.source import resolve_source_paths, source_gate
from src.research_common.progress import ProgressReporter

from .config import LatentLiquidityPoolForecastConfig

_FEATURE_COLUMNS = (
    "event_id", "event_time", "event_side", "period", "release_episode_id",
    "release_episode_ordinal", "release_episode_size", "down_event_score", "up_event_score",
)
_LABEL_COLUMNS = (
    "event_id", "event_reference_price", "future_extension_bp",
    "future_reversal_after_extreme_bp", "future_time_to_extreme_seconds",
    "outcome_type", "favorable_reversal",
)
_ASSIGNMENT_COLUMNS = ("event_id", "path_cluster", "cluster_distance")


def _scan_config(config: LatentLiquidityPoolForecastConfig) -> StablePathExecutionAuditConfig:
    base = StablePathExecutionAuditConfig()
    return replace(
        base,
        source_report_dir=config.source_report_dir,
        source_feature_file=config.source_feature_file,
        source_label_file=config.source_label_file,
        source_assignment_file=config.source_assignment_file,
        source_manifest_file=config.source_manifest_file,
        source_causal_audit_file=config.source_causal_audit_file,
        csv_read_chunk_rows=config.csv_read_chunk_rows,
    )


def _read_aligned(paths, chunk_rows: int):
    fr = pd.read_csv(paths.feature, usecols=lambda n: n in _FEATURE_COLUMNS, chunksize=chunk_rows, low_memory=False)
    lr = pd.read_csv(paths.label, usecols=lambda n: n in _LABEL_COLUMNS, chunksize=chunk_rows, low_memory=False)
    ar = pd.read_csv(paths.assignment, usecols=lambda n: n in _ASSIGNMENT_COLUMNS, chunksize=chunk_rows, low_memory=False)
    while True:
        try:
            f = next(fr)
        except StopIteration:
            break
        try:
            l = next(lr); a = next(ar)
        except StopIteration as exc:
            raise RuntimeError("R02 source tables have unequal row counts") from exc
        if len(f) != len(l) or len(f) != len(a):
            raise RuntimeError("R02 source chunk row-count mismatch")
        ids = f["event_id"].astype(str).to_numpy()
        if not np.array_equal(ids, l["event_id"].astype(str).to_numpy()) or not np.array_equal(ids, a["event_id"].astype(str).to_numpy()):
            raise RuntimeError("R02 source event alignment mismatch")
        yield f, l, a


def source_gate_only(config: LatentLiquidityPoolForecastConfig) -> tuple[pd.DataFrame, int]:
    scan_cfg = _scan_config(config)
    paths = resolve_source_paths(scan_cfg)
    gate = source_gate(scan_cfg, paths)
    rows = 0
    if paths.manifest.exists():
        try:
            payload = json.loads(paths.manifest.read_text(encoding="utf-8"))
            rows = int(payload.get("feature_rows", payload.get("joined_rows", 0)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            rows = 0
    return gate, rows


def load_episode_table(config: LatentLiquidityPoolForecastConfig, *, progress: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    scan_cfg = _scan_config(config)
    paths = resolve_source_paths(scan_cfg)
    gate = source_gate(scan_cfg, paths)
    failures = gate.loc[gate["status"].astype(str).eq("FAIL"), "check"].tolist()
    if failures:
        raise RuntimeError(f"R02 source gate failed: {failures}")
    estimated = 1
    if paths.manifest.exists():
        try:
            payload = json.loads(paths.manifest.read_text(encoding="utf-8"))
            total = int(payload.get("feature_rows", payload.get("joined_rows", 0)))
            estimated = max(1, int(np.ceil(total / config.csv_read_chunk_rows))) if total else 1
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            estimated = 1
    reporter = ProgressReporter("[latent-liquidity-r02] stream R01.1 Episode labels", estimated, every=1, enabled=progress)
    parts: list[pd.DataFrame] = []
    scanned = 0
    for i, (f, l, a) in enumerate(_read_aligned(paths, config.csv_read_chunk_rows), 1):
        scanned += len(f)
        ordinal = pd.to_numeric(f["release_episode_ordinal"], errors="coerce")
        mask = ordinal.eq(1)
        if mask.any():
            ff, ll, aa = f.loc[mask].reset_index(drop=True), l.loc[mask].reset_index(drop=True), a.loc[mask].reset_index(drop=True)
            side = ff["event_side"].astype(str)
            release_score = np.where(side.eq("DOWN"), pd.to_numeric(ff.get("down_event_score"), errors="coerce"), pd.to_numeric(ff.get("up_event_score"), errors="coerce"))
            part = pd.DataFrame({
                "event_id": ff["event_id"].astype(str),
                "event_time": pd.to_datetime(ff["event_time"], errors="coerce"),
                "event_side": side,
                "period": ff["period"].astype(str),
                "release_episode_id": ff["release_episode_id"].astype(str),
                "release_episode_size": pd.to_numeric(ff["release_episode_size"], errors="coerce").fillna(1).astype(np.int32),
                "release_score": pd.to_numeric(release_score, errors="coerce"),
                "event_reference_price": pd.to_numeric(ll["event_reference_price"], errors="coerce"),
                "future_extension_bp": pd.to_numeric(ll["future_extension_bp"], errors="coerce"),
                "future_reversal_after_extreme_bp": pd.to_numeric(ll["future_reversal_after_extreme_bp"], errors="coerce"),
                "future_time_to_extreme_seconds": pd.to_numeric(ll["future_time_to_extreme_seconds"], errors="coerce"),
                "outcome_type": ll["outcome_type"].astype(str),
                "favorable_reversal": ll["favorable_reversal"].astype(str).str.lower().isin({"true", "1"}),
                "path_cluster": pd.to_numeric(aa["path_cluster"], errors="coerce").fillna(-1).astype(np.int16),
                "cluster_distance": pd.to_numeric(aa["cluster_distance"], errors="coerce"),
            })
            parts.append(part.loc[part["event_time"].notna() & part["event_reference_price"].gt(0)])
        reporter.update(min(i, estimated))
    reporter.close()
    if not parts:
        return pd.DataFrame(), gate, scanned
    episodes = pd.concat(parts, ignore_index=True, copy=False).sort_values("event_time", kind="mergesort").reset_index(drop=True)
    episodes["release_density_proxy"] = np.log1p(episodes["release_episode_size"].astype(float)) * np.log1p(episodes["release_score"].clip(lower=0).fillna(0.0))
    return episodes, gate, scanned
