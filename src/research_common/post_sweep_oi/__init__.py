#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R05 post-sweep Binance OI research helpers."""

from .config import PostSweepOIConfig
from .align import (
    add_attempt_pair_features,
    build_future_oi_labels,
    causal_align_oi,
    first_per_event,
    load_r04_tables,
    oracle_turning_points,
    pair_oracle_with_prior_attempt,
    split_features_labels,
)
from .reports import (
    attempt_mechanism_summary,
    causal_audit,
    compact_event_sample,
    coverage_by_period,
    data_quality_report,
    fixed_oi_bin_summary,
    large_mfe_oi_profile,
    new_low_attempt_oi_summary,
    oracle_pair_summary,
    position_flow_state_summary,
    rebound_oi_path_summary,
    taker_ratio_summary,
)

__all__ = [
    "PostSweepOIConfig", "add_attempt_pair_features", "build_future_oi_labels",
    "causal_align_oi", "first_per_event", "load_r04_tables", "oracle_turning_points",
    "pair_oracle_with_prior_attempt", "split_features_labels",
    "attempt_mechanism_summary", "causal_audit", "compact_event_sample",
    "coverage_by_period", "data_quality_report", "fixed_oi_bin_summary",
    "large_mfe_oi_profile", "new_low_attempt_oi_summary", "oracle_pair_summary",
    "position_flow_state_summary", "rebound_oi_path_summary", "taker_ratio_summary",
]
