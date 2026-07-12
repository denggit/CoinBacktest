#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Retrospectively highlight swing extremes followed by a target move.

The detector uses a percentage directional-change (zigzag) process. A swing low
is confirmed only after price rises by the configured percentage from the
tracked low; a swing high is confirmed only after price falls by that amount
from the tracked high. The confirmation must occur within ``max_completion_bars``
for the extreme to be displayed.

This is intentionally a historical path-labeling plugin. Marker timestamps are
placed on the extreme bar, while confirmation happens later, so the marker must
not be interpreted as a real-time signal available at the extreme timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Literal

import numpy as np
import pandas as pd

from analyze_tool.plugin_api import Marker, PluginParam, PluginRunResult


SwingDirection = Literal["low", "high"]


@dataclass(frozen=True)
class ConfirmedSwing:
    direction: SwingDirection
    extreme_index: int
    confirmation_index: int
    extreme_price: float
    confirmation_price: float
    move_pct: float

    @property
    def completion_bars(self) -> int:
        return self.confirmation_index - self.extreme_index


def _iter_confirmed_swings(high: np.ndarray, low: np.ndarray, threshold: float) -> Iterator[ConfirmedSwing]:
    """Yield alternating percentage-confirmed swing highs and lows.

    A newly updated extreme is never confirmed on the same bar. This avoids
    assuming whether the high or low happened first inside one OHLC bar.
    """

    n = len(high)
    if n < 2:
        return

    candidate_high = 0
    candidate_low = 0
    mode: SwingDirection | None = None  # ``high`` means seek/track a swing high.

    for i in range(1, n):
        if mode is None:
            if high[i] > high[candidate_high]:
                candidate_high = i
            if low[i] < low[candidate_low]:
                candidate_low = i

            low_ready = i > candidate_low and high[i] >= low[candidate_low] * (1.0 + threshold)
            high_ready = i > candidate_high and low[i] <= high[candidate_high] * (1.0 - threshold)

            if low_ready and high_ready:
                # The earlier extreme is the only chronologically defensible
                # first pivot. Same-bar ambiguity is deliberately skipped.
                if candidate_low < candidate_high:
                    high_ready = False
                elif candidate_high < candidate_low:
                    low_ready = False
                else:
                    continue

            if low_ready:
                extreme = float(low[candidate_low])
                confirmation = float(high[i])
                yield ConfirmedSwing(
                    direction="low",
                    extreme_index=candidate_low,
                    confirmation_index=i,
                    extreme_price=extreme,
                    confirmation_price=confirmation,
                    move_pct=confirmation / extreme - 1.0,
                )
                mode = "high"
                candidate_high = i
                continue

            if high_ready:
                extreme = float(high[candidate_high])
                confirmation = float(low[i])
                yield ConfirmedSwing(
                    direction="high",
                    extreme_index=candidate_high,
                    confirmation_index=i,
                    extreme_price=extreme,
                    confirmation_price=confirmation,
                    move_pct=1.0 - confirmation / extreme,
                )
                mode = "low"
                candidate_low = i
                continue

        elif mode == "high":
            updated = False
            if high[i] > high[candidate_high]:
                candidate_high = i
                updated = True
            if not updated and i > candidate_high and low[i] <= high[candidate_high] * (1.0 - threshold):
                extreme = float(high[candidate_high])
                confirmation = float(low[i])
                yield ConfirmedSwing(
                    direction="high",
                    extreme_index=candidate_high,
                    confirmation_index=i,
                    extreme_price=extreme,
                    confirmation_price=confirmation,
                    move_pct=1.0 - confirmation / extreme,
                )
                mode = "low"
                candidate_low = i

        else:  # mode == "low"
            updated = False
            if low[i] < low[candidate_low]:
                candidate_low = i
                updated = True
            if not updated and i > candidate_low and high[i] >= low[candidate_low] * (1.0 + threshold):
                extreme = float(low[candidate_low])
                confirmation = float(high[i])
                yield ConfirmedSwing(
                    direction="low",
                    extreme_index=candidate_low,
                    confirmation_index=i,
                    extreme_price=extreme,
                    confirmation_price=confirmation,
                    move_pct=confirmation / extreme - 1.0,
                )
                mode = "high"
                candidate_high = i


class SwingExtremeMovePlugin:
    plugin_id = "swing_extreme_move"
    name = "Swing Extreme 后续涨跌幅"
    description = (
        "回看标记满足条件的 Swing Low / Swing High：extreme 出现后，必须在限定 bars 内完成指定涨跌幅。"
        "标记会回画在 extreme 位置，属于事后路径分析，不是 extreme 当时可用的实时信号。"
    )
    params = [
        PluginParam(
            name="direction",
            label="方向",
            kind="select",
            default="both",
            choices=[
                {"value": "both", "label": "Swing Low + Swing High"},
                {"value": "low", "label": "只看 Swing Low（后续上涨）"},
                {"value": "high", "label": "只看 Swing High（后续下跌）"},
            ],
        ),
        PluginParam(
            name="move_pct",
            label="后续涨跌幅（%）",
            kind="number",
            default=1.0,
            min_value=0.05,
            max_value=20.0,
            step=0.05,
            description="1 = 1%",
        ),
        PluginParam(
            name="max_completion_bars",
            label="最多完成时间（bars）",
            kind="number",
            default=60,
            min_value=1,
            max_value=5000,
            step=1,
        ),
    ]

    def run(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> PluginRunResult:
        if df is None or df.empty:
            return PluginRunResult(markers=[], summary={"input_rows": 0, "matched": 0, "display": "无数据"})

        p = params or {}
        direction = str(p.get("direction", "both")).lower()
        if direction not in {"both", "low", "high"}:
            raise ValueError(f"direction must be one of both/low/high, got {direction!r}")

        move_pct_input = float(p.get("move_pct", 1.0))
        max_completion_bars = int(float(p.get("max_completion_bars", 60)))
        if not 0.0 < move_pct_input < 100.0:
            raise ValueError("后续涨跌幅必须大于 0 且小于 100")
        if max_completion_bars < 1:
            raise ValueError("最多完成时间必须至少为 1 bar")
        threshold = move_pct_input / 100.0

        missing = {"high", "low"} - set(df.columns)
        if missing:
            raise ValueError(f"swing_extreme_move requires high/low columns, missing={sorted(missing)}")

        work = df[["high", "low"]].copy()
        work["high"] = pd.to_numeric(work["high"], errors="coerce")
        work["low"] = pd.to_numeric(work["low"], errors="coerce")
        valid = work["high"].notna() & work["low"].notna() & (work["high"] > 0) & (work["low"] > 0)
        work = work.loc[valid]
        if len(work) < 2:
            return PluginRunResult(
                markers=[],
                summary={"input_rows": int(len(df)), "matched": 0, "display": "有效 high/low 数据不足"},
            )

        timestamps = pd.DatetimeIndex(work.index)
        high = work["high"].to_numpy(dtype=float, copy=True)
        low = work["low"].to_numpy(dtype=float, copy=True)

        markers: list[Marker] = []
        total_confirmed = 0
        expired_by_horizon = 0
        low_count = 0
        high_count = 0

        for swing in _iter_confirmed_swings(high, low, threshold):
            total_confirmed += 1
            if swing.completion_bars > max_completion_bars:
                expired_by_horizon += 1
                continue
            if direction != "both" and swing.direction != direction:
                continue

            extreme_ts = timestamps[swing.extreme_index]
            confirmation_ts = timestamps[swing.confirmation_index]
            is_low = swing.direction == "low"
            if is_low:
                low_count += 1
                color = "#22c55e"
                label = f"Swing Low +{swing.move_pct * 100:.2f}%"
                reason = (
                    f"低点后 {swing.completion_bars} bars 内上涨 "
                    f"{swing.move_pct * 100:.3f}%（目标 {move_pct_input:.3f}%）"
                )
                position = "low"
                symbol = "arrow_up"
            else:
                high_count += 1
                color = "#ef4444"
                label = f"Swing High -{swing.move_pct * 100:.2f}%"
                reason = (
                    f"高点后 {swing.completion_bars} bars 内下跌 "
                    f"{swing.move_pct * 100:.3f}%（目标 {move_pct_input:.3f}%）"
                )
                position = "high"
                symbol = "arrow_down"

            markers.append(
                Marker(
                    timestamp=extreme_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    label=label,
                    color=color,
                    reason=reason,
                    role="signal",
                    position=position,
                    price=swing.extreme_price,
                    symbol=symbol,
                    fields={
                        "direction": swing.direction,
                        "extreme_timestamp": extreme_ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "extreme_price": swing.extreme_price,
                        "confirmation_timestamp": confirmation_ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "confirmation_price": swing.confirmation_price,
                        "completion_bars": swing.completion_bars,
                        "target_move_pct": move_pct_input,
                        "realized_move_pct": swing.move_pct * 100.0,
                        "retrospective_label": True,
                        "uses_future_confirmation": True,
                    },
                )
            )

        selected_label = {"both": "双向", "low": "Swing Low", "high": "Swing High"}[direction]
        display = (
            f"{selected_label}：{len(markers)} 个；Low {low_count}，High {high_count}；"
            f"目标 {move_pct_input:g}%，限时 {max_completion_bars} bars"
        )
        return PluginRunResult(
            markers=markers,
            summary={
                "input_rows": int(len(df)),
                "matched": int(len(markers)),
                "low_count": int(low_count),
                "high_count": int(high_count),
                "total_directional_change_pivots": int(total_confirmed),
                "excluded_over_horizon": int(expired_by_horizon),
                "direction": direction,
                "move_pct": move_pct_input,
                "max_completion_bars": max_completion_bars,
                "retrospective_label": True,
                "display": display,
            },
        )
