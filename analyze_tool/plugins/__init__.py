#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Built-in analyze_tool plugins."""

from __future__ import annotations

from analyze_tool.plugin_api import PluginRegistry
from analyze_tool.plugins.long_shadow import LongShadowPlugin
from analyze_tool.plugins.panic_selloff_recovery import PanicSelloffRecoveryPlugin


def build_default_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register(LongShadowPlugin())
    registry.register(PanicSelloffRecoveryPlugin())
    return registry
