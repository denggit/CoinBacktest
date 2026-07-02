#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reusable event-study module for CoinBacktest research.

The module is intentionally independent from concrete LF/HF strategy scripts.
It provides closed-bar signal / next-open outcome labels, causal context
alignment helpers, standard summaries, and report writers. Existing research
scripts can migrate into this module gradually without changing strategy logic.
"""

from .binning import fixed_threshold_labels, qcut_labels
from .causal import add_available_time_index, audit_context_available_times, audit_next_open_entries, causal_align_context
from .models import CostConfig, EventStudyConfig, EventStudyResult
from .outcomes import first_touch_outcome, forward_mfe_mae, normalize_side, signed_close_to_close_return, signed_next_open_return
from .runner import run_event_study
from .stats import condition_contrast, payoff_ratio, profit_factor, summarize_many, summarize_returns, top_winner_dependency

__all__ = [
    "CostConfig",
    "EventStudyConfig",
    "EventStudyResult",
    "add_available_time_index",
    "audit_context_available_times",
    "audit_next_open_entries",
    "causal_align_context",
    "condition_contrast",
    "first_touch_outcome",
    "fixed_threshold_labels",
    "forward_mfe_mae",
    "normalize_side",
    "payoff_ratio",
    "profit_factor",
    "qcut_labels",
    "run_event_study",
    "signed_close_to_close_return",
    "signed_next_open_return",
    "summarize_many",
    "summarize_returns",
    "top_winner_dependency",
]
