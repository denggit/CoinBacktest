#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize causal multi-bar panic selloff/rejection/recovery episodes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from analyze_tool.plugin_api import Marker, PluginParam, PluginRunResult, Region
from research.liquidity.panic_selloff_rejection_recovery_long.common.panic_episode import (
    PanicEpisodeConfig,
    detect_panic_episodes,
)


NODE_STYLE = {
    "start": ("#f59e0b", "top", "start"),
    "acceleration": ("#ef4444", "low", "node"),
    "low_candidate": ("#e879f9", "low", "node"),
    "exhaustion": ("#facc15", "low", "node"),
    "signal": ("#22c55e", "low", "signal"),
}


class PanicSelloffRecoveryPlugin:
    plugin_id = "panic_selloff_recovery"
    name = "多阶段恐慌卖压恢复"
    description = (
        "不是单根K线事件：连续卖压开始观察 → 卖压加速 → 低点候选 → "
        "卖压衰减/拒绝 → 恢复确认。绿色节点是因果信号，浅色区间表示整段 episode。"
    )
    params = [
        PluginParam(
            name="baseline_window",
            label="历史基线窗口（bars）",
            kind="number",
            default=60,
            min_value=20,
            max_value=500,
            step=1,
        ),
        PluginParam(
            name="selloff_window",
            label="连续卖压窗口（bars）",
            kind="number",
            default=5,
            min_value=2,
            max_value=30,
            step=1,
        ),
        PluginParam(
            name="min_red_bars",
            label="窗口内最少下跌 bars",
            kind="number",
            default=3,
            min_value=1,
            max_value=30,
            step=1,
        ),
        PluginParam(
            name="observe_drop_pct",
            label="开始观察累计跌幅",
            kind="number",
            default=0.0045,
            min_value=0.0005,
            max_value=0.10,
            step=0.0005,
            description="0.0045 = 0.45%",
        ),
        PluginParam(
            name="observe_drop_vol_mult",
            label="跌速/历史波动最小倍数",
            kind="number",
            default=2.5,
            min_value=0.5,
            max_value=20,
            step=0.1,
        ),
        PluginParam(
            name="observe_volume_ratio",
            label="窗口成交量最小倍数",
            kind="number",
            default=1.10,
            min_value=0.1,
            max_value=10,
            step=0.05,
        ),
        PluginParam(
            name="panic_drop_pct",
            label="进入恐慌阶段总跌幅",
            kind="number",
            default=0.0075,
            min_value=0.001,
            max_value=0.20,
            step=0.0005,
            description="相对卖压启动前参考价；0.0075 = 0.75%",
        ),
        PluginParam(
            name="panic_volume_ratio",
            label="恐慌 bar 成交量倍数",
            kind="number",
            default=1.35,
            min_value=0.1,
            max_value=10,
            step=0.05,
        ),
        PluginParam(
            name="stabilization_bars",
            label="低点后稳定 bars",
            kind="number",
            default=2,
            min_value=1,
            max_value=20,
            step=1,
        ),
        PluginParam(
            name="min_rebound_from_low_pct",
            label="低点后最小反弹",
            kind="number",
            default=0.0020,
            min_value=0.0001,
            max_value=0.10,
            step=0.0001,
        ),
        PluginParam(
            name="reclaim_fraction",
            label="恢复确认需收回恐慌区间比例",
            kind="number",
            default=0.35,
            min_value=0.05,
            max_value=0.95,
            step=0.05,
        ),
        PluginParam(
            name="max_episode_bars",
            label="episode 最大 bars",
            kind="number",
            default=30,
            min_value=5,
            max_value=300,
            step=1,
        ),
    ]

    def run(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> PluginRunResult:
        if df is None or df.empty:
            return PluginRunResult(markers=[], regions=[], summary={"input_rows": 0, "matched": 0})
        p = params or {}
        cfg = PanicEpisodeConfig(
            baseline_window=int(float(p.get("baseline_window", 60))),
            selloff_window=int(float(p.get("selloff_window", 5))),
            min_red_bars=int(float(p.get("min_red_bars", 3))),
            observe_drop_pct=float(p.get("observe_drop_pct", 0.0045)),
            observe_drop_vol_mult=float(p.get("observe_drop_vol_mult", 2.5)),
            observe_volume_ratio=float(p.get("observe_volume_ratio", 1.10)),
            panic_drop_pct=float(p.get("panic_drop_pct", 0.0075)),
            panic_volume_ratio=float(p.get("panic_volume_ratio", 1.35)),
            stabilization_bars=int(float(p.get("stabilization_bars", 2))),
            min_rebound_from_low_pct=float(p.get("min_rebound_from_low_pct", 0.0020)),
            reclaim_fraction=float(p.get("reclaim_fraction", 0.35)),
            max_episode_bars=int(float(p.get("max_episode_bars", 30))),
        )
        result = detect_panic_episodes(df, cfg)

        markers: list[Marker] = []
        regions: list[Region] = []
        signal_returns_15: list[float] = []
        positive_15 = 0
        valid_15 = 0

        for episode in result.episodes:
            success = episode.signal_time is not None
            region_color = "#fb923c" if success else "#94a3b8"
            region_opacity = 0.10 if success else 0.055
            region_fields = {
                "episode_id": episode.episode_id,
                "status": episode.status,
                "reference_price": episode.reference_price,
                "episode_low": episode.episode_low,
                "signal_time": _ts_text(episode.signal_time),
                "signal_price": episode.signal_price,
                **episode.fields,
            }
            regions.append(
                Region(
                    start_timestamp=_ts_text(episode.start_time),
                    end_timestamp=_ts_text(episode.end_time),
                    label=f"Panic episode #{episode.episode_id}",
                    color=region_color,
                    opacity=region_opacity,
                    status=episode.status,
                    fields=region_fields,
                )
            )
            for node in episode.nodes:
                color, position, role = NODE_STYLE.get(node.kind, ("#facc15", "top", "node"))
                fields = {"episode_id": episode.episode_id, "node_kind": node.kind, **node.fields}
                markers.append(
                    Marker(
                        timestamp=_ts_text(node.timestamp),
                        label=node.label,
                        color=color,
                        reason=node.label,
                        fields=fields,
                        role=role,
                        position=position,
                        price=float(node.price) if node.price is not None else None,
                    )
                )
                if node.kind == "signal":
                    value = node.fields.get("outcome_return_15b")
                    if value is not None and np.isfinite(float(value)):
                        value = float(value)
                        signal_returns_15.append(value)
                        valid_15 += 1
                        positive_15 += int(value > 0)

        signal_count = result.signal_count
        episode_count = len(result.episodes)
        failed_count = sum(ep.signal_time is None for ep in result.episodes)
        median_15 = float(np.median(signal_returns_15)) if signal_returns_15 else None
        positive_rate_15 = positive_15 / valid_15 if valid_15 else None
        display = (
            f"Episode {episode_count} 个；恢复信号 {signal_count} 个；无信号/未完成 {failed_count} 个；"
            f"信号后15 bars 中位收益 {_pct(median_15)}；上涨比例 {_pct(positive_rate_15)}"
        )
        return PluginRunResult(
            markers=markers,
            regions=regions,
            summary={
                "input_rows": int(len(df)),
                "matched": signal_count,
                "episode_count": episode_count,
                "signal_count": signal_count,
                "failed_or_incomplete_count": failed_count,
                "signal_rate": signal_count / episode_count if episode_count else None,
                "outcome_15b_valid": valid_15,
                "outcome_15b_median_return": median_15,
                "outcome_15b_positive_rate": positive_rate_15,
                "display": display,
                "note": "forward outcomes are diagnostics only; they do not participate in signal generation",
            },
        )


def _ts_text(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "-"
    return f"{value * 100:.2f}%"
