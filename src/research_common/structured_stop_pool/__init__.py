#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R09 structured stop-pool hypothesis research helpers."""
from .config import FirstTouchSpec, StructuredStopPoolConfig, first_touch_specs
from .structure import (
    FAMILY_COLUMNS,
    attach_zone_hypotheses,
    build_level_structure_features,
    hypothesis_definitions,
)
from .release import attach_stop_release_labels, calibrate_release_score
from .universe import audit_r02_bar_alignment, build_r09_universe, load_or_build_r02
from .outcomes import attach_first_touch_outcomes
from .reports import (
    causal_audit,
    data_quality,
    family_overlap,
    family_path_summary,
    family_release_summary,
    family_scorecard,
    family_strategy_summary,
    family_timeframe_summary,
    hypothesis_universe_summary,
    matched_release_comparison,
    period_stability,
    research_brief,
)

__all__ = [
    "FirstTouchSpec",
    "StructuredStopPoolConfig",
    "first_touch_specs",
    "FAMILY_COLUMNS",
    "build_level_structure_features",
    "attach_zone_hypotheses",
    "hypothesis_definitions",
    "attach_stop_release_labels",
    "calibrate_release_score",
    "load_or_build_r02",
    "audit_r02_bar_alignment",
    "build_r09_universe",
    "attach_first_touch_outcomes",
    "data_quality",
    "hypothesis_universe_summary",
    "family_release_summary",
    "matched_release_comparison",
    "family_path_summary",
    "family_strategy_summary",
    "family_timeframe_summary",
    "period_stability",
    "family_overlap",
    "family_scorecard",
    "causal_audit",
    "research_brief",
]
