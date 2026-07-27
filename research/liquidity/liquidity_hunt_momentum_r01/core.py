#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stable public facade for Liquidity Hunt Momentum R01.

The implementation is split by responsibility so research orchestration,
feature construction, event definitions, simulation and diagnostics remain
independently testable.
"""

from .events import build_events
from .features import (
    BOOK_CONTEXT_COLUMNS,
    aggregate_footprint_features,
    align_book_features_to_times,
    attach_book_context,
    attach_footprint_features,
    build_range_features,
    datetime_index_to_ns_int64,
    prepare_book_features,
)
from .metrics import (
    build_causal_audit,
    chronological_split_labels,
    profit_factor,
    summarize_returns,
)
from .models import LiquidityHuntConfig, StrategyVariant
from .simulation import attach_forward_time_outcomes, simulate_events

__all__ = [
    "BOOK_CONTEXT_COLUMNS",
    "LiquidityHuntConfig",
    "StrategyVariant",
    "aggregate_footprint_features",
    "align_book_features_to_times",
    "attach_book_context",
    "attach_footprint_features",
    "attach_forward_time_outcomes",
    "build_causal_audit",
    "build_events",
    "build_range_features",
    "chronological_split_labels",
    "datetime_index_to_ns_int64",
    "prepare_book_features",
    "profit_factor",
    "simulate_events",
    "summarize_returns",
]
