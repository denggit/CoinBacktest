#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Built-in long upper/lower shadow marker plugin."""

from __future__ import annotations

from typing import Any

import pandas as pd

from analyze_tool.plugin_api import Marker, PluginParam, PluginRunResult


class LongShadowPlugin:
    plugin_id = "long_shadow"
    name = "长影线标记"
    description = "按影线/实体比例和影线占价格比例，标记长上影线、长下影线或两者。"
    params = [
        PluginParam(
            name="direction",
            label="方向",
            kind="select",
            default="both",
            choices=[
                {"value": "both", "label": "上下影线都标记"},
                {"value": "upper", "label": "只标记长上影线"},
                {"value": "lower", "label": "只标记长下影线"},
            ],
        ),
        PluginParam(
            name="min_wick_body_ratio",
            label="影线/实体最小倍数",
            kind="number",
            default=2.5,
            min_value=0.1,
            max_value=50,
            step=0.1,
        ),
        PluginParam(
            name="min_wick_pct",
            label="影线最小价格占比",
            kind="number",
            default=0.0015,
            min_value=0,
            max_value=0.05,
            step=0.0001,
            description="0.0015 = 0.15%",
        ),
        PluginParam(
            name="color",
            label="高亮颜色",
            kind="color",
            default="#facc15",
        ),
    ]

    def run(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> PluginRunResult:
        if df is None or df.empty:
            return PluginRunResult(markers=[], summary={"matched": 0, "input_rows": 0})
        p = params or {}
        direction = str(p.get("direction", "both"))
        min_ratio = float(p.get("min_wick_body_ratio", 2.5))
        min_wick_pct = float(p.get("min_wick_pct", 0.0015))
        color = str(p.get("color", "#facc15")) or "#facc15"

        need = {"open", "high", "low", "close"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"long_shadow plugin requires columns: {sorted(need)}, missing={sorted(missing)}")

        work = df.copy()
        for col in ["open", "high", "low", "close"]:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        work = work.dropna(subset=["open", "high", "low", "close"])
        if work.empty:
            return PluginRunResult(markers=[], summary={"matched": 0, "input_rows": len(df)})

        body = (work["close"] - work["open"]).abs()
        # Avoid tiny doji exploding the ratio to infinity; use a price-scaled floor.
        floor = (work["close"].abs() * 1e-6).clip(lower=1e-9)
        body_safe = body.where(body > floor, floor)
        upper_wick = work["high"] - work[["open", "close"]].max(axis=1)
        lower_wick = work[["open", "close"]].min(axis=1) - work["low"]
        price_base = work["close"].abs().clip(lower=1e-9)

        upper_mask = (upper_wick / body_safe >= min_ratio) & (upper_wick / price_base >= min_wick_pct)
        lower_mask = (lower_wick / body_safe >= min_ratio) & (lower_wick / price_base >= min_wick_pct)
        if direction == "upper":
            mask = upper_mask
        elif direction == "lower":
            mask = lower_mask
        else:
            mask = upper_mask | lower_mask

        markers: list[Marker] = []
        for idx, row in work.loc[mask].iterrows():
            is_upper = bool(upper_mask.loc[idx])
            is_lower = bool(lower_mask.loc[idx])
            if is_upper and is_lower:
                label = "长上下影"
            elif is_upper:
                label = "长上影"
            else:
                label = "长下影"
            ts = pd.Timestamp(idx).strftime("%Y-%m-%d %H:%M:%S")
            fields = {
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "upper_wick": float(upper_wick.loc[idx]),
                "lower_wick": float(lower_wick.loc[idx]),
                "body": float(body.loc[idx]),
                "upper_wick_body_ratio": float(upper_wick.loc[idx] / body_safe.loc[idx]),
                "lower_wick_body_ratio": float(lower_wick.loc[idx] / body_safe.loc[idx]),
                "upper_wick_pct": float(upper_wick.loc[idx] / price_base.loc[idx]),
                "lower_wick_pct": float(lower_wick.loc[idx] / price_base.loc[idx]),
            }
            markers.append(Marker(timestamp=ts, label=label, color=color, reason=label, fields=fields))

        return PluginRunResult(
            markers=markers,
            summary={
                "input_rows": int(len(df)),
                "matched": int(len(markers)),
                "upper_count": int(upper_mask.sum()),
                "lower_count": int(lower_mask.sum()),
                "direction": direction,
                "min_wick_body_ratio": min_ratio,
                "min_wick_pct": min_wick_pct,
            },
        )
