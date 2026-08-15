#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R02.3 source adapter. Reuses the completed R02.2 exact-first-touch cache."""
from __future__ import annotations

import pandas as pd

from src.ai_research.latent_liquidity_first_touch_ranking.cache import (
    dataset_cache_path as r02_2_dataset_cache_path,
    load_frame as load_r02_2_frame,
)
from src.ai_research.latent_liquidity_first_touch_ranking.config import DEFAULT_CONFIG as R02_2_CONFIG
from src.ai_research.latent_liquidity_pool_forecast.source import source_gate_only
from src.ai_research.latent_liquidity_pool_forecast.config import DEFAULT_CONFIG as R02_CONFIG


def load_r02_2_exact_first_touch_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    path = r02_2_dataset_cache_path(R02_2_CONFIG)
    if not path.exists():
        raise RuntimeError(
            "R02.3 requires the completed R02.2 exact-first-touch cache. Run first: "
            "python research\\eth_ai_trading\\eth_latent_liquidity_path_v1\\02_2_first_touch_relative_liquidity_ranking.py"
        )
    frame = load_r02_2_frame(path)
    gate, _ = source_gate_only(R02_CONFIG)
    failures = gate.loc[gate["status"].astype(str).eq("FAIL"), "check"].tolist()
    if failures:
        raise RuntimeError(f"R02.3 upstream R01.1/R02 source gate failed: {failures}")
    frame = frame.copy()
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], errors="coerce")
    frame["feature_available_time"] = pd.to_datetime(frame["feature_available_time"], errors="coerce")
    frame["first_touch_time"] = pd.to_datetime(frame["first_touch_time"], errors="coerce")
    frame["first_touch_available_time"] = frame["first_touch_time"] + pd.Timedelta(seconds=1)
    frame["r02_touch_consistent"] = frame["touch_720m"].astype(bool)
    # R02.3 uses exact 1s replay as the canonical source.  Any row whose old R02
    # 1m touch flag disagrees is quarantined rather than silently accepted.
    frame["r02_3_source_eligible"] = (
        frame["first_touch_label_complete"].astype(bool)
        & frame["first_touch_time"].notna()
        & frame["decision_time"].notna()
        & frame["first_touch_available_time"].gt(frame["decision_time"])
        & frame["first_touch_time"].lt(frame["decision_time"] + pd.Timedelta(hours=12))
        & frame["r02_touch_consistent"].astype(bool)
    )
    return frame, gate
