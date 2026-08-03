#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reusable components for the R07 footprint/books absorption study."""

from .books import attach_books_context, books_coverage_table
from .config import PostSweepFootprintBooksConfig
from .footprint import (
    FOOTPRINT_FEATURE_COLUMNS,
    aggregate_footprint_bars,
    attach_footprint_context,
    build_footprint_context,
)
from .reports import (
    build_research_brief,
    causal_audit,
    cohort_feature_summary,
    data_quality_report,
    feature_outcome_auc,
    frozen_quantile_lift,
    mechanism_scorecard,
    paired_feature_profile,
    pair_overlap_summary,
)
from .universe import load_r04_attempt_universe, load_r06_matched_universe

__all__ = [
    "PostSweepFootprintBooksConfig",
    "FOOTPRINT_FEATURE_COLUMNS",
    "aggregate_footprint_bars",
    "attach_footprint_context",
    "build_footprint_context",
    "attach_books_context",
    "books_coverage_table",
    "load_r04_attempt_universe",
    "load_r06_matched_universe",
    "data_quality_report",
    "cohort_feature_summary",
    "paired_feature_profile",
    "pair_overlap_summary",
    "feature_outcome_auc",
    "frozen_quantile_lift",
    "mechanism_scorecard",
    "causal_audit",
    "build_research_brief",
]
