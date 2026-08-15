#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Source adapters for R02.2. Reuse completed R02/R01.1 artifacts only."""
from __future__ import annotations

import pandas as pd

from src.ai_research.latent_liquidity_pool_forecast.cache import (
    dataset_cache_path as r02_dataset_cache_path,
    episode_cache_path as r02_episode_cache_path,
    load_frame as load_r02_frame,
)
from src.ai_research.latent_liquidity_pool_forecast.config import DEFAULT_CONFIG as R02_CONFIG
from src.ai_research.latent_liquidity_pool_forecast.source import load_episode_table, source_gate_only


def load_r02_audit_lattice_and_episodes() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = r02_dataset_cache_path(R02_CONFIG)
    if not path.exists():
        raise RuntimeError(
            "R02.2 requires the completed R02 spatial cache. Run first: "
            "python research\\eth_ai_trading\\eth_latent_liquidity_path_v1\\02_latent_pool_location_depth_model.py"
        )
    frame = load_r02_frame(path)
    gate, _ = source_gate_only(R02_CONFIG)
    failures = gate.loc[gate["status"].astype(str).eq("FAIL"), "check"].tolist()
    if failures:
        raise RuntimeError(f"R02.2 source gate failed: {failures}")
    if "full_lattice_audit_group" not in frame:
        raise RuntimeError("R02 spatial cache predates complete-lattice audit groups")
    audit = frame.loc[frame["full_lattice_audit_group"].astype(bool)].copy()
    audit["decision_time"] = pd.to_datetime(audit["decision_time"], errors="coerce")
    audit = audit.loc[audit["decision_time"].notna()].sort_values(
        ["decision_time", "zone_side", "zone_distance_bp"], kind="mergesort"
    ).reset_index(drop=True)
    if audit.empty:
        raise RuntimeError("R02.2 complete-lattice audit sample is empty")
    ep_path = r02_episode_cache_path(R02_CONFIG)
    if ep_path.exists():
        episodes = load_r02_frame(ep_path)
    else:
        episodes, ep_gate, _ = load_episode_table(R02_CONFIG, progress=True)
        failures = ep_gate.loc[ep_gate["status"].astype(str).eq("FAIL"), "check"].tolist()
        if failures:
            raise RuntimeError(f"R02.2 Episode source gate failed: {failures}")
    episodes["event_time"] = pd.to_datetime(episodes["event_time"], errors="coerce")
    episodes = episodes.loc[episodes["event_time"].notna()].sort_values("event_time", kind="mergesort").reset_index(drop=True)
    return audit, episodes, gate
