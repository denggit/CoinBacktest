#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""R11 liquidity-magnet and risk-frontier research helpers."""
from .config import LiquidityMagnetConfig, stop_model_definitions
from .universe import build_liquidity_magnet_universe, load_r02_and_r09_levels
from .outcomes import attach_risk_frontier_outcomes
from .reports import (
    candidate_funnel,
    causal_audit,
    data_quality,
    design_table,
    directional_magnet_summary,
    manifest_json,
    period_stability,
    research_brief,
    risk_frontier_summary,
    scorecard,
    structure_family_summary,
    timeframe_confluence_summary,
)

__all__ = [
    "LiquidityMagnetConfig",
    "stop_model_definitions",
    "build_liquidity_magnet_universe",
    "load_r02_and_r09_levels",
    "attach_risk_frontier_outcomes",
    "design_table",
    "data_quality",
    "candidate_funnel",
    "risk_frontier_summary",
    "directional_magnet_summary",
    "timeframe_confluence_summary",
    "structure_family_summary",
    "period_stability",
    "scorecard",
    "causal_audit",
    "research_brief",
    "manifest_json",
]
