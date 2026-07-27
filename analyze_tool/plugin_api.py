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
class PluginRunContext:
    """Data and request context for plugins needing more than one DataFrame.

    ``display_df`` is the chart-aligned frame. ``visible_df`` is the requested
    visible slice, while ``analysis_frames`` can hold causally prepared parent
    frames (for example a 1m analysis frame behind a 15s display).
    """

    display_df: pd.DataFrame
    visible_df: pd.DataFrame
    analysis_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    request: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def data_type(self) -> str:
        return str(self.request.get("data_type", self.meta.get("data_type", "unknown")))

    @property
    def timeframe(self) -> str:
        return str(self.request.get("timeframe", self.meta.get("timeframe", "")))

    @property
    def range_pct(self) -> float:
        value = self.request.get("range_pct", self.meta.get("range_pct", 0.0))
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0


@dataclass(frozen=True)
class IndicatorTrack:
    """A compact indicator series aligned 1:1 with plugin input rows."""

    track_id: str
    label: str
    values: list[Any]
    color: str = "#38bdf8"
    min_value: float = 0.0
    max_value: float = 1.0
    reference_value: float | None = None
    description: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.track_id,
            "label": self.label,
            "values": self.values,
            "color": self.color,
            "min": self.min_value,
            "max": self.max_value,
            "reference": self.reference_value,
            "description": self.description,
        }


@dataclass(frozen=True)
class StateBandCategory:
    code: int
    label: str
    color: str
    opacity: float = 0.08
    status: str = ""
    fields: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "label": self.label,
            "color": self.color,
            "opacity": self.opacity,
            "status": self.status,
            "fields": self.fields,
        }


@dataclass(frozen=True)
class StateBand:
    """Categorical band aligned 1:1 with plugin input rows.

    ``render_mode="background"`` shades the full price pane.
    ``render_mode="strip"`` draws a compact lane so orthogonal states do not
    visually overwrite each other.
    """

    band_id: str
    label: str
    codes: list[int | None]
    categories: list[StateBandCategory]
    description: str = ""
    render_mode: str = "background"
    position: str = "bottom"
    height_px: int = 10

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.band_id,
            "label": self.label,
            "codes": self.codes,
            "categories": [category.as_dict() for category in self.categories],
            "description": self.description,
            "render_mode": self.render_mode,
            "position": self.position,
            "height_px": self.height_px,
        }


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
class PriceRegion:
    """Price-time rectangle rendered as an optional chart overlay."""

    start_timestamp: str
    end_timestamp: str
    price_low: float
    price_high: float
    label: str = ""
    color: str = "#facc15"
    opacity: float = 0.04
    border_width: float = 1.5
    side: str = ""
    status: str = ""
    fields: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "price_low": self.price_low,
            "price_high": self.price_high,
            "label": self.label,
            "color": self.color,
            "opacity": self.opacity,
            "border_width": self.border_width,
            "side": self.side,
            "status": self.status,
            "fields": self.fields,
        }


@dataclass(frozen=True)
class PriceHeatmapCell:
    """Price-time heatmap rectangle rendered behind candles."""

    start_timestamp: str
    end_timestamp: str
    price_low: float
    price_high: float
    intensity: float
    side: str = ""
    color: str = "#facc15"
    label: str = ""
    confidence: float = 0.0
    fields: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "price_low": self.price_low,
            "price_high": self.price_high,
            "intensity": self.intensity,
            "side": self.side,
            "color": self.color,
            "label": self.label,
            "confidence": self.confidence,
            "fields": self.fields,
        }


@dataclass(frozen=True)
class PluginRunResult:
    markers: list[Marker]
    regions: list[Region] = field(default_factory=list)
    tracks: list[IndicatorTrack] = field(default_factory=list)
    bands: list[StateBand] = field(default_factory=list)
    heatmap: list[PriceHeatmapCell] = field(default_factory=list)
    price_regions: list[PriceRegion] = field(default_factory=list)
    row_fields: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    heatmap_compact: dict[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "markers": [m.as_dict() for m in self.markers],
            "regions": [r.as_dict() for r in self.regions],
            "tracks": [track.as_dict() for track in self.tracks],
            "bands": [band.as_dict() for band in self.bands],
            "heatmap": [cell.as_dict() for cell in self.heatmap],
            "price_regions": [region.as_dict() for region in self.price_regions],
            "row_fields": self.row_fields,
            "summary": self.summary,
        }
        if self.heatmap_compact is not None:
            payload["heatmap_compact"] = self.heatmap_compact
        return payload


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
