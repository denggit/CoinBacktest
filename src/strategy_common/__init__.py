#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reusable strategy-development contracts for CoinBacktest.

This package deliberately sits between research and portfolio code:
research may discover a market phenomenon, but a strategy is not eligible for
portfolio work until it has a complete executable contract and a healthy
research-to-trade funnel.
"""

from .funnel import FunnelAudit, FunnelPolicy, FunnelStage, audit_funnel
from .models import StrategyContract

__all__ = [
    "FunnelAudit",
    "FunnelPolicy",
    "FunnelStage",
    "StrategyContract",
    "audit_funnel",
]
