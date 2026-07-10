#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Built-in analyze_tool plugins."""

from __future__ import annotations

from analyze_tool.plugin_api import PluginRegistry
from analyze_tool.plugins.long_shadow import LongShadowPlugin


def build_default_registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register(LongShadowPlugin())
    return registry
