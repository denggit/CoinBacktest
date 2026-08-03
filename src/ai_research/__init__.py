#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Governance and reusable research contracts for the ETH AI programme.

This package belongs to CoinBacktest. It contains research-stage contracts and
pure research computation only; it does not contain live exchange execution or
AetherEdge runtime code.
"""

from .config import AIResearchConfig, DEFAULT_AI_RESEARCH_CONFIG
from .models import ResearchPlan, StageDefinition, StageOwner
from .plan import DEFAULT_RESEARCH_PLAN, validate_research_plan
from .trades_baseline import DEFAULT_TRADES_BASELINE_CONFIG, TradesBaselineConfig

__all__ = [
    "AIResearchConfig",
    "DEFAULT_AI_RESEARCH_CONFIG",
    "DEFAULT_RESEARCH_PLAN",
    "DEFAULT_TRADES_BASELINE_CONFIG",
    "TradesBaselineConfig",
    "ResearchPlan",
    "StageDefinition",
    "StageOwner",
    "validate_research_plan",
]
