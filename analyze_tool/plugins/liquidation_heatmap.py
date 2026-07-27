#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Estimated Liquidation Heatmap V1 plugin.

Default UI is intentionally minimal: price-time heatmap plus one concise card.
The heatmap is a transparent model built from public OI/funding/order-flow data;
it is not exchange account position data and must not be presented as exact.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from analyze_tool.plugin_api import PluginParam, PluginRunContext, PluginRunResult, PriceHeatmapCell
from src.data_feed.okx_derivatives_loader import OKXDerivativesLoader
from src.liquidation_map import EstimatedLiquidationMapEngine, LiquidationMapConfig


def _values(series: pd.Series) -> list[Any]:
    out: list[Any] = []
    for value in series:
        if value is None or pd.isna(value):
            out.append(None)
        elif isinstance(value, (np.integer, int)):
            out.append(int(value))
        elif isinstance(value, (np.floating, float)):
            number = float(value)
            out.append(number if math.isfinite(number) else None)
        else:
            out.append(str(value))
    return out


def _categorical(series: pd.Series) -> dict[str, Any]:
    categories: dict[str, str] = {}
    values: list[int | None] = []
    code_by_value: dict[str, int] = {}
    for raw in series:
        if raw is None or pd.isna(raw):
            values.append(None)
            continue
        text = str(raw)
        if text not in code_by_value:
            code = len(code_by_value) + 1
            code_by_value[text] = code
            categories[str(code)] = text
        values.append(code_by_value[text])
    return {"values": values, "categories": categories}


def _fmt_zone(price: Any, distance: Any, *, side: str) -> str:
    if price is None or pd.isna(price) or distance is None or pd.isna(distance):
        return "暂无可用热区"
    sign = "+" if float(distance) >= 0 else ""
    name = "空头清算" if side == "short" else "多头清算"
    return f"{name} {float(price):.2f}（{sign}{float(distance) * 100:.2f}%）"


class LiquidationHeatmapPlugin:
    plugin_id = "estimated_liquidation_heatmap_v1"
    name = "推定清算热力图 V1.3"
    description = (
        "以公开 OI、Funding、Mark、真实清算事件和 Trade Bar 订单流构建透明的推定清算热力图。"
        "颜色越亮只代表模型估计密度越高，不代表真实账户仓位，也不是价格必然被吸引。"
    )
    params = [
        PluginParam(
            name="display_mode",
            label="显示模式",
            kind="select",
            default="simple",
            choices=[
                {"value": "simple", "label": "极简热力图（推荐）"},
                {"value": "dense", "label": "更密集热力图"},
            ],
        ),
        PluginParam("price_bucket_pct", "价格格宽", default=0.0025, min_value=0.0005, max_value=0.02, step=0.0005),
        PluginParam("max_distance_pct", "显示当前价上下范围", default=0.12, min_value=0.02, max_value=0.50, step=0.01),
        PluginParam("snapshot_every_bars", "热力快照间隔（bars）", default=5, min_value=1, max_value=60, step=1),
        PluginParam("cohort_half_life_hours", "推定仓位半衰期（小时）", default=72, min_value=4, max_value=720, step=4),
        PluginParam("minimum_oi_delta_usd", "最小 OI 增量（USD）", default=100000, min_value=0, max_value=100000000, step=100000),
        PluginParam("maintenance_margin_rate", "维护保证金率近似", default=0.005, min_value=0.0, max_value=0.05, step=0.001),
        PluginParam("liquidation_fee_buffer", "清算费用/缓冲近似", default=0.002, min_value=0.0, max_value=0.03, step=0.001),
    ]

    def run(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> PluginRunResult:
        context = PluginRunContext(display_df=df, visible_df=df, request={"symbol": "ETH-USDT-SWAP"})
        return self.run_with_context(context, params)

    def run_with_context(self, context: PluginRunContext, params: dict[str, Any] | None) -> PluginRunResult:
        bars = context.visible_df
        if bars is None or bars.empty:
            return PluginRunResult(markers=[], summary={"input_rows": 0, "matched": 0, "display": "无数据"})

        p = params or {}
        dense = str(p.get("display_mode", "simple")) == "dense"
        config = LiquidationMapConfig(
            price_bucket_pct=float(p.get("price_bucket_pct", 0.0025)),
            max_distance_pct=float(p.get("max_distance_pct", 0.12)),
            snapshot_every_bars=int(float(p.get("snapshot_every_bars", 5))),
            cohort_half_life_hours=float(p.get("cohort_half_life_hours", 72)),
            minimum_oi_delta_usd=float(p.get("minimum_oi_delta_usd", 100000)),
            maintenance_margin_rate=float(p.get("maintenance_margin_rate", 0.005)),
            liquidation_fee_buffer=float(p.get("liquidation_fee_buffer", 0.002)),
            max_cells_per_snapshot=100 if dense else 45,
        )
        config.validate()

        symbol = str(context.meta.get("symbol") or context.request.get("symbol") or "ETH-USDT-SWAP")
        start = pd.Timestamp(bars.index.min())
        end = pd.Timestamp(bars.index.max())
        loader = OKXDerivativesLoader(symbol=symbol)
        oi = loader.load_open_interest(start, end)
        funding = loader.load_funding_rates(start - pd.Timedelta(days=7), end)
        timeframe = str(context.timeframe or "1m")
        mark = loader.load_mark_prices(start, end, timeframe=timeframe)
        if mark.empty and timeframe != "1m":
            mark = loader.load_mark_prices(start, end, timeframe="1m")
        liquidations = loader.load_liquidations(start, end)

        result = EstimatedLiquidationMapEngine(config).compute(
            bars,
            open_interest=oi,
            funding_rates=funding,
            mark_prices=mark,
            liquidations=liquidations,
        )
        if not result.diagnostics.get("ready"):
            coverage = {item.dataset: item.rows for item in loader.coverage()}
            return PluginRunResult(
                markers=[],
                summary={
                    "input_rows": int(len(bars)),
                    "matched": 0,
                    "display": "缺少本地 OI 数据；先运行 tools\\prebuild_okx_liquidation_inputs.py",
                    "reason": result.diagnostics.get("reason"),
                    "coverage": coverage,
                    "ui": {"compact": True, "brief_available": False, "advanced_collapsed": True},
                },
            )

        heatmap = [
            PriceHeatmapCell(
                start_timestamp=pd.Timestamp(cell.start_timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                end_timestamp=pd.Timestamp(cell.end_timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                price_low=float(cell.price_low),
                price_high=float(cell.price_high),
                intensity=float(cell.intensity),
                side=cell.side,
                color="#ef4444" if cell.side == "long" else "#22d3ee",
                label=cell.label,
                confidence=float(cell.confidence),
                fields={"estimated_notional": float(cell.raw_notional), **cell.fields},
            )
            for cell in result.cells
        ]

        aligned = result.row_frame.reindex(pd.DatetimeIndex(pd.to_datetime(bars.index)))
        balance = pd.to_numeric(aligned["liquidation_balance"], errors="coerce").fillna(0.0)
        distribution = pd.Series("上下相对均衡", index=aligned.index, dtype="object")
        distribution[balance > 0.12] = "上方空头清算密度更强"
        distribution[balance < -0.12] = "下方多头清算密度更强"
        upper = pd.Series(
            [_fmt_zone(px, dist, side="short") for px, dist in zip(aligned["nearest_short_liq_price"], aligned["nearest_short_liq_distance_pct"])],
            index=aligned.index,
            dtype="object",
        )
        lower = pd.Series(
            [_fmt_zone(px, dist, side="long") for px, dist in zip(aligned["nearest_long_liq_price"], aligned["nearest_long_liq_distance_pct"])],
            index=aligned.index,
            dtype="object",
        )
        confidence = pd.to_numeric(aligned["model_confidence"], errors="coerce").fillna(0.0)
        confidence_text = pd.Series("低", index=aligned.index, dtype="object")
        confidence_text[confidence >= 0.45] = "中"
        confidence_text[confidence >= 0.65] = "较高"
        reason_1 = pd.Series([f"模型置信度：{value}" for value in confidence_text], index=aligned.index)
        reason_2 = pd.Series("基于公开 OI 与透明杠杆桶推定，不是真实账户仓位", index=aligned.index)
        observed_rows = int(result.diagnostics.get("observed_liquidation_rows", 0))
        reason_3 = pd.Series(
            f"当前区间真实清算事件覆盖：{observed_rows} 根 Bar" if observed_rows else "当前区间无真实清算事件校准，置信度会降低",
            index=aligned.index,
        )
        advice = pd.Series("热区表示潜在连锁清算燃料；必须结合订单流、流动性路径和市场过程判断。", index=aligned.index)

        row_fields = {
            "brief_direction": _categorical(distribution),
            "brief_phase": _categorical(upper),
            "brief_process": _categorical(lower),
            "brief_reason_1": _categorical(reason_1),
            "brief_reason_2": _categorical(reason_2),
            "brief_reason_3": _categorical(reason_3),
            "brief_advice": _categorical(advice),
            "brief_context_detail": _categorical(distribution),
            "model_confidence": _values(confidence),
            "liquidation_balance": _values(balance),
            "nearest_short_liq_price": _values(aligned["nearest_short_liq_price"]),
            "nearest_short_liq_distance_pct": _values(aligned["nearest_short_liq_distance_pct"]),
            "nearest_long_liq_price": _values(aligned["nearest_long_liq_price"]),
            "nearest_long_liq_distance_pct": _values(aligned["nearest_long_liq_distance_pct"]),
            "oi_usd": _values(aligned["oi_usd"]),
            "oi_delta_usd": _values(aligned["oi_delta_usd"]),
        }

        top_zone_text = []
        for zone in result.current_zones:
            label = "多头清算" if zone.side == "long" else "空头清算"
            top_zone_text.append(f"{label}@{zone.center_price:.2f}({zone.distance_pct * 100:+.2f}%)")

        if heatmap:
            display_text = f"推定热区 {len(heatmap):,} 格；当前强区：{'，'.join(top_zone_text[:6]) or '暂无'}"
        elif result.diagnostics.get("empty_reason") == "oi_value_units_unavailable":
            display_text = "OI 有记录但缺少可用 USD/base-asset 数值；请重新运行 V1.3 数据预构建"
        elif result.diagnostics.get("empty_reason") == "no_positive_oi_updates":
            display_text = "OI 已加载，但区间内没有正向 OI 增量，因此未生成新仓群组"
        elif result.diagnostics.get("empty_reason") == "oi_delta_below_threshold":
            maximum = float(result.diagnostics.get("max_positive_oi_delta_usd", 0.0) or 0.0)
            threshold = float(result.diagnostics.get("minimum_oi_delta_usd", 0.0) or 0.0)
            display_text = f"OI 已加载，但最大正增量 ${maximum:,.0f} 未超过阈值 ${threshold:,.0f}"
        elif result.diagnostics.get("empty_reason"):
            display_text = f"未生成热区：{result.diagnostics.get('empty_reason')}"
        else:
            display_text = "当前时点暂无显著清算热区"

        summary = {
            **result.diagnostics,
            "matched": len(heatmap),
            "display": display_text,
            "top_zones": top_zone_text,
            "disclaimer": "Estimated Liquidation Heatmap：不是 CoinGlass 数据，也不代表真实账户清算价。",
            "ui": {
                "compact": True,
                "brief_available": True,
                "advanced_collapsed": True,
                "brief_labels": ["清算分布", "最近上方", "最近下方"],
                "brief_disclaimer": "热区是透明估算，不是交易信号，也不是价格必达目标。",
            },
        }
        return PluginRunResult(markers=[], heatmap=heatmap, row_fields=row_fields, summary=summary)
