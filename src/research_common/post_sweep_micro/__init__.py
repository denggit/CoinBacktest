#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reusable R06 post-sweep micro turning-point research components."""

from .config import PostSweepMicroConfig
from .micro import TRIGGER_NAMES, analyze_micro_window, regularize_window
from .range_context import analyze_event_range_context, extract_range_context
from .reports import (
    build_research_brief,
    candidate_scorecard,
    causal_audit,
    cohort_low_feature_summary,
    data_quality_report,
    paired_micro_profile,
    range_pair_overlap_summary,
    range_pair_profile,
    trigger_occurrence_summary,
    trigger_path_summary,
    trigger_relative_to_baselines,
)
from .universe import (
    attach_optional_oi_context,
    build_attempt_universe,
    load_binance_oi_context,
    load_optional_r05_oi,
    load_r04_micro_source,
)

__all__ = [
    "PostSweepMicroConfig", "attach_optional_oi_context", "TRIGGER_NAMES", "analyze_event_range_context",
    "analyze_micro_window", "build_attempt_universe", "build_research_brief",
    "candidate_scorecard", "causal_audit", "cohort_low_feature_summary",
    "data_quality_report", "extract_range_context", "load_binance_oi_context", "load_optional_r05_oi",
    "load_r04_micro_source", "paired_micro_profile", "range_pair_overlap_summary", "range_pair_profile",
    "regularize_window", "trigger_occurrence_summary", "trigger_path_summary",
    "trigger_relative_to_baselines",
]
