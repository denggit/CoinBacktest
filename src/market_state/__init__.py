#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal market-state map shared by research, backtests and visualization."""

from src.market_state.causal_alignment import causal_merge_context, timeframe_to_timedelta
from src.market_state.data_bundle import MarketStateDataBundle
from src.market_state.models import (
    DataQualityReport,
    MarketStateConfig,
    MarketStateResult,
    MarketStateSegment,
    MarketStateSnapshot,
)
from src.market_state.state_engine import MarketStateEngine

__all__ = [
    "DataQualityReport",
    "MarketStateConfig",
    "MarketStateDataBundle",
    "MarketStateEngine",
    "MarketStateResult",
    "MarketStateSegment",
    "MarketStateSnapshot",
    "causal_merge_context",
    "timeframe_to_timedelta",
]

# Role-aware state-map research helpers.  Kept in the shared domain so future
# research, backtests and live parity checks can use identical definitions.
from src.market_state.conditional_map import (
    ConditionDefinition,
    ConditionalMapConfig,
    ConditionEvaluation,
)

__all__.extend([
    "ConditionDefinition",
    "ConditionalMapConfig",
    "ConditionEvaluation",
])

from src.market_state.process_map import (
    PROCESS_DIRECTIONS,
    PROCESS_FAMILIES,
    PROCESS_STAGE_LABELS,
    PROCESS_STAGE_LABELS_LEGACY,
    PROCESS_STAGE_LABELS_V3_1,
    ProcessMapConfig,
    ProcessMapEngine,
    ProcessMapResult,
    completed_process_mask,
    stage_event_mask,
)

__all__.extend([
    "PROCESS_DIRECTIONS",
    "PROCESS_FAMILIES",
    "PROCESS_STAGE_LABELS",
    "PROCESS_STAGE_LABELS_LEGACY",
    "PROCESS_STAGE_LABELS_V3_1",
    "ProcessMapConfig",
    "ProcessMapEngine",
    "ProcessMapResult",
    "completed_process_mask",
    "stage_event_mask",
])
