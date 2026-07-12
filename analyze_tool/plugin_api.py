#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Plugin API for analyze_tool.

Plugins receive an already-loaded DataFrame and return point markers plus
optional time regions.  Keeping this protocol small lets research detectors be
shared without pushing chart-specific code into ``src.data_feed``.
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
    role: str = "node"
    position: str = "top"
    price: float | None = None
    symbol: str = "auto"

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "label": self.label,
            "color": self.color,
            "reason": self.reason,
            "fields": self.fields,
            "role": self.role,
            "position": self.position,
            "price": self.price,
            "symbol": self.symbol,
        }


@dataclass(frozen=True)
class Region:
    start_timestamp: str
    end_timestamp: str
    label: str = ""
    color: str = "#fb923c"
    opacity: float = 0.08
    status: str = ""
    fields: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "label": self.label,
            "color": self.color,
            "opacity": self.opacity,
            "status": self.status,
            "fields": self.fields,
        }


@dataclass(frozen=True)
class PluginRunResult:
    markers: list[Marker]
    regions: list[Region] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "markers": [m.as_dict() for m in self.markers],
            "regions": [r.as_dict() for r in self.regions],
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
