#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plugin API for analyze_tool.

Plugins are intentionally small and data-frame based: a plugin receives the
already-loaded OHLCV/order-flow DataFrame and returns marker rows for the chart.
This keeps research annotations separate from src.data_feed and strategy code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import pandas as pd


@dataclass(frozen=True)
class PluginParam:
    name: str
    label: str
    kind: str = "number"
    default: Any = None
    description: str = ""
    min_value: float | None = None
    max_value: float | None = None
    step: float | None = None
    choices: list[dict[str, Any]] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "name": self.name,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "description": self.description,
        }
        if self.min_value is not None:
            out["min"] = self.min_value
        if self.max_value is not None:
            out["max"] = self.max_value
        if self.step is not None:
            out["step"] = self.step
        if self.choices is not None:
            out["choices"] = self.choices
        return out


@dataclass(frozen=True)
class Marker:
    timestamp: str
    label: str
    color: str = "#facc15"
    reason: str = ""
    fields: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "label": self.label,
            "color": self.color,
            "reason": self.reason,
            "fields": self.fields,
        }


@dataclass(frozen=True)
class PluginRunResult:
    markers: list[Marker]
    summary: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "markers": [m.as_dict() for m in self.markers],
            "summary": self.summary,
        }


class AnalyzePlugin(Protocol):
    plugin_id: str
    name: str
    description: str
    params: list[PluginParam]

    def run(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> PluginRunResult:
        ...


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, AnalyzePlugin] = {}

    def register(self, plugin: AnalyzePlugin) -> None:
        if not plugin.plugin_id:
            raise ValueError("plugin_id cannot be empty")
        if plugin.plugin_id in self._plugins:
            raise ValueError(f"duplicate plugin_id: {plugin.plugin_id}")
        self._plugins[plugin.plugin_id] = plugin

    def get(self, plugin_id: str) -> AnalyzePlugin:
        try:
            return self._plugins[plugin_id]
        except KeyError as exc:
            raise KeyError(f"unknown plugin: {plugin_id}") from exc

    def list_plugins(self) -> list[dict[str, Any]]:
        return [
            {
                "id": p.plugin_id,
                "name": p.name,
                "description": p.description,
                "params": [param.as_dict() for param in p.params],
            }
            for p in self._plugins.values()
        ]
