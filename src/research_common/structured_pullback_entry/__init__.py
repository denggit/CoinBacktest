#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R10 structured pullback-entry research helpers."""
from .config import PullbackTargetSpec, StructuredPullbackConfig, target_specs
from .execution import attach_limit_fills, attach_trade_outcomes
from .reports import (
    candidate_funnel_summary,
    causal_audit,
    data_quality,
    family_geometry_summary,
    family_outcome_summary,
    family_overlap,
    family_scorecard,
    family_timeframe_summary,
    fill_age_summary,
    period_stability,
    research_brief,
)
from .universe import (
    ALL_FAMILY_IDS,
    BASELINE_ID,
    FAMILY_IDS,
    FAMILY_NAMES,
    build_pullback_candidate_universe,
    hypothesis_definitions,
    load_or_build_r09_level_features,
)

__all__ = [
    "PullbackTargetSpec",
    "StructuredPullbackConfig",
    "target_specs",
    "attach_limit_fills",
    "attach_trade_outcomes",
    "candidate_funnel_summary",
    "causal_audit",
    "data_quality",
    "family_geometry_summary",
    "family_outcome_summary",
    "family_overlap",
    "family_scorecard",
    "family_timeframe_summary",
    "fill_age_summary",
    "period_stability",
    "research_brief",
    "ALL_FAMILY_IDS",
    "BASELINE_ID",
    "FAMILY_IDS",
    "FAMILY_NAMES",
    "build_pullback_candidate_universe",
    "hypothesis_definitions",
    "load_or_build_r09_level_features",
]
