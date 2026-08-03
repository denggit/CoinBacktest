#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Built-in analyze_tool plugins."""

from __future__ import annotations

from analyze_tool.plugin_api import PluginRegistry
from analyze_tool.plugins.ai_market_state_timeline import AIMarketStateTimelinePlugin
from analyze_tool.plugins.long_shadow import LongShadowPlugin
from analyze_tool.plugins.liquidity_wall_discovery import LiquidityWallDiscoveryPlugin
from analyze_tool.plugins.liquidation_heatmap import LiquidationHeatmapPlugin
from analyze_tool.plugins.market_state_map import MarketStateMapPlugin
from analyze_tool.plugins.orderbook_liquidity_heatmap import OrderBookLiquidityHeatmapPlugin

try:
    from analyze_tool.plugins.panic_selloff_recovery import PanicSelloffRecoveryPlugin
except Exception:  # optional research plugin must not block Analyze Tool startup
    PanicSelloffRecoveryPlugin = None  # type: ignore[assignment,misc]

try:
    from analyze_tool.plugins.swing_extreme_move import SwingExtremeMovePlugin
except Exception:  # optional legacy plugin must not block the analyzer
    SwingExtremeMovePlugin = None  # type: ignore[assignment,misc]


def build_default_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register(LongShadowPlugin())
    if PanicSelloffRecoveryPlugin is not None:
        registry.register(PanicSelloffRecoveryPlugin())
    registry.register(LiquidationHeatmapPlugin())
    registry.register(OrderBookLiquidityHeatmapPlugin())
    registry.register(LiquidityWallDiscoveryPlugin())
    registry.register(MarketStateMapPlugin())
    registry.register(AIMarketStateTimelinePlugin())
    if SwingExtremeMovePlugin is not None:
        registry.register(SwingExtremeMovePlugin())
    return registry
