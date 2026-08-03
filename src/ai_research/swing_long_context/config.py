#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen configuration for R03.2 long-context swing opportunity research."""

from __future__ import annotations

from dataclasses import replace

from src.ai_research.swing_baseline.config import DEFAULT_SWING_BASELINE_CONFIG
from src.ai_research.swing_baseline.features import LONG_CONTEXT_PROFILE
from src.ai_research.swing_entry_mvp.config import SwingEntryMvpConfig

STAGE_ID = "R03.2"
STAGE_NAME = "Long-context 3%-5% swing opportunity model"
FEATURE_PROFILE = LONG_CONTEXT_PROFILE

LONG_CONTEXT_BASE_CONFIG = replace(
    DEFAULT_SWING_BASELINE_CONFIG,
    feature_lookback_days=420,
    cache_dir="data/cache/eth_ai_trading/r03_2_long_context",
    report_dir="data/reports/research/eth_ai_trading/03_2_swing_long_context",
)

DEFAULT_SWING_LONG_CONTEXT_CONFIG = SwingEntryMvpConfig(
    base=LONG_CONTEXT_BASE_CONFIG,
    exact_label_cache_dir="data/cache/eth_ai_trading/r03_2_exact_outcomes",
    report_dir="data/reports/research/eth_ai_trading/03_2_swing_long_context",
    architectures=(
        "high_logistic",
        "high_lightgbm",
        "hierarchical_lightgbm",
    ),
    direction_modes=("long", "short"),
)


def validate_long_context_contract(config: SwingEntryMvpConfig = DEFAULT_SWING_LONG_CONTEXT_CONFIG) -> None:
    config.validate()
    if config.base.feature_lookback_days < 400:
        raise ValueError("R03.2 must retain at least 400 calendar days of causal feature warmup")
    if "r03_2_long_context" not in config.base.cache_dir:
        raise ValueError("R03.2 must use an isolated long-context cache")
    if "r03_2_exact_outcomes" not in config.exact_label_cache_dir:
        raise ValueError("R03.2 exact outcomes must not overwrite R03.1 caches")
