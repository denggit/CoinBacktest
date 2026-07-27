#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visual overlay for Liquidity Wall Discovery V1 report segments.

The overlay is intentionally separate from the production heatmap wall detector.
It reads the research-generated causal lifecycle segments so the user can compare
candidate walls with the raw heatmap without replacing the existing plugin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_tool.plugin_api import PluginParam, PluginRunContext, PluginRunResult, PriceRegion

try:
    from config.loader import TIMEZONE
except ImportError:  # pragma: no cover
    TIMEZONE = "+8"


DEFAULT_REPORT_DIR = Path("data/reports/research/liquidity/liquidity_wall_discovery_v1")


def _timezone_offset() -> pd.Timedelta:
    text = str(TIMEZONE).strip()
    if text.startswith("+"):
        return pd.Timedelta(hours=float(text[1:] or 0))
    if text.startswith("-"):
        return -pd.Timedelta(hours=float(text[1:] or 0))
    return pd.Timedelta(0)


def _local_to_utc_ms(value: Any) -> int:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        ts = (ts - _timezone_offset()).tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return int(ts.timestamp() * 1000)


def _utc_ms_to_local_text(value: Any) -> str:
    ts = pd.to_datetime(int(value), unit="ms", utc=True).tz_convert(None) + _timezone_offset()
    return ts.isoformat(sep=" ")


def _as_float(params: dict[str, Any], name: str, default: float) -> float:
    try:
        value = float(params.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    return value if np.isfinite(value) else float(default)


class LiquidityWallDiscoveryPlugin:
    plugin_id = "liquidity_wall_discovery_v1"
    name = "流动性墙发现 V1（研究覆盖层）"
    description = "显示墙发现研究生成的因果生命周期片段，用于和原始热力图做肉眼验收；不是生产墙标签。"
    params = [
        PluginParam(
            name="minimum_shape_score",
            label="最低结构分",
            kind="number",
            default=35,
            min_value=0,
            max_value=200,
            step=1,
            description="只影响显示，不改变研究结果。",
        ),
        PluginParam(
            name="side",
            label="方向",
            kind="select",
            default="all",
            choices=[
                {"value": "all", "label": "双向"},
                {"value": "bid", "label": "下方 Bid"},
                {"value": "ask", "label": "上方 Ask"},
            ],
        ),
        PluginParam(
            name="show_ghost",
            label="显示漂移幽灵墙",
            kind="select",
            default="no",
            choices=[
                {"value": "no", "label": "隐藏"},
                {"value": "yes", "label": "显示"},
            ],
        ),
        PluginParam(
            name="maximum_regions",
            label="最多显示区域",
            kind="number",
            default=1200,
            min_value=50,
            max_value=5000,
            step=50,
        ),
    ]

    def __init__(self, report_dir: str | Path | None = None) -> None:
        root = Path(__file__).resolve().parents[2]
        self.report_dir = Path(report_dir) if report_dir is not None else root / DEFAULT_REPORT_DIR

    @property
    def segment_path(self) -> Path:
        return self.report_dir / "13_wall_overlay_segments.csv"

    def run(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> PluginRunResult:
        if df is None or df.empty:
            return PluginRunResult(markers=[], summary={"display": "没有K线数据"})
        request = {
            "start": str(df.index.min()),
            "end": str(df.index.max()),
        }
        context = PluginRunContext(display_df=df, visible_df=df, request=request)
        return self.run_with_context(context, params)

    def run_with_context(
        self,
        context: PluginRunContext,
        params: dict[str, Any] | None = None,
    ) -> PluginRunResult:
        options = params or {}
        if not self.segment_path.exists():
            return PluginRunResult(
                markers=[],
                summary={
                    "display": "尚未生成墙发现覆盖层；先运行 Liquidity Wall Discovery V1",
                    "wall_overlay_label": "墙发现候选",
                    "segment_path": str(self.segment_path),
                },
            )
        frame = pd.read_csv(self.segment_path)
        if frame.empty:
            return PluginRunResult(markers=[], summary={"display": "墙发现覆盖层为空"})
        for column in ("start_ms", "end_ms", "price_low", "price_high", "shape_score", "retention", "is_ghost"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["start_ms", "end_ms", "price_low", "price_high"])
        start_value = context.request.get("start") or context.visible_df.index.min()
        end_value = context.request.get("end") or context.visible_df.index.max()
        start_ms = _local_to_utc_ms(start_value)
        end_ms = _local_to_utc_ms(end_value)
        frame = frame.loc[(frame["start_ms"] <= end_ms) & (frame["end_ms"] >= start_ms)].copy()
        minimum_score = _as_float(options, "minimum_shape_score", 35.0)
        frame = frame.loc[frame["shape_score"].fillna(0.0) >= minimum_score]
        requested_side = str(options.get("side", "all"))
        if requested_side in {"bid", "ask"}:
            frame = frame.loc[frame["side"].astype(str) == requested_side]
        if str(options.get("show_ghost", "no")) != "yes":
            frame = frame.loc[frame["is_ghost"].fillna(0).astype(int) == 0]
        maximum_regions = max(50, int(_as_float(options, "maximum_regions", 1200)))
        if len(frame) > maximum_regions:
            frame = frame.sort_values(["shape_score", "retention"], ascending=False).head(maximum_regions)
        frame = frame.sort_values(["start_ms", "wall_id"])

        regions: list[PriceRegion] = []
        for row in frame.to_dict("records"):
            side = str(row.get("side", ""))
            ghost = int(row.get("is_ghost", 0)) == 1
            color = "#94a3b8" if ghost else ("#22c55e" if side == "bid" else "#ef4444")
            morphology = str(row.get("morphology", ""))
            label = f"W{int(row.get('wall_id', 0))} {morphology}"
            regions.append(
                PriceRegion(
                    start_timestamp=_utc_ms_to_local_text(row["start_ms"]),
                    end_timestamp=_utc_ms_to_local_text(row["end_ms"]),
                    price_low=float(row["price_low"]),
                    price_high=float(row["price_high"]),
                    label=label,
                    color=color,
                    opacity=0.035 if not ghost else 0.02,
                    border_width=1.2 if not ghost else 0.8,
                    side=side,
                    status="GHOST" if ghost else morphology,
                    fields={
                        "wall_id": int(row.get("wall_id", 0)),
                        "shape_score": float(row.get("shape_score", np.nan)),
                        "retention": float(row.get("retention", np.nan)),
                        "age_seconds": float(row.get("age_seconds", np.nan)),
                        "research_only": True,
                    },
                )
            )
        return PluginRunResult(
            markers=[],
            price_regions=regions,
            summary={
                "display": f"墙发现研究候选 {len(regions)} 个片段（仅供肉眼验收，不是生产墙）",
                "wall_overlay_label": "墙发现候选",
                "segments": len(regions),
                "report_dir": str(self.report_dir),
                "research_only": True,
                "ui": {
                    "wall_overlay_control": True,
                    "wall_overlay_label": "墙发现候选",
                    "wall_overlay_default": True,
                },
            },
        )
