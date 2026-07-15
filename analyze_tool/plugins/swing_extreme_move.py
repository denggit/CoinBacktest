#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Retrospectively highlight swing extremes followed by a target move.

Swing extreme selection remains wick-based: swing lows are tracked from ``low``
and swing highs from ``high``. Return evaluation starts from the next bar's
``open`` after the extreme. A long target is completed when a later ``high``
reaches the configured gain from that next open; a short target is completed
when a later ``low`` reaches the configured decline from that next open.

The confirmation must occur within ``max_completion_bars`` bars measured from
the extreme bar. This is intentionally a historical path-labeling plugin.
Marker timestamps are placed on the extreme bar, while confirmation happens
later, so the marker must not be interpreted as a real-time signal available at
the extreme timestamp.
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
    entry_index: int
    confirmation_index: int
    extreme_price: float
    entry_price: float
    confirmation_price: float
    move_pct: float

    @property
    def completion_bars(self) -> int:
        """Bars from the swing extreme to target completion."""
        return self.confirmation_index - self.extreme_index

    @property
    def bars_from_entry(self) -> int:
        """Bars elapsed after entering at the next bar open."""
        return self.confirmation_index - self.entry_index


def _iter_confirmed_swings(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    threshold: float,
) -> Iterator[ConfirmedSwing]:
    """Yield alternating percentage-confirmed swing highs and lows.

    Extreme-bar selection uses ``low`` for swing lows and ``high`` for swing
    highs. The return baseline is always the extreme bar's next ``open``.

    A newly updated extreme is never confirmed on the same bar. This preserves
    the original intrabar ambiguity protection: when a bar creates a newer
    extreme, that bar cannot simultaneously confirm the reversal.
    """

    n = len(high)
    if n < 2:
        return

    candidate_high = 0
    candidate_low = 0
    mode: SwingDirection | None = None  # ``high`` means seek/track a swing high.

    def low_ready(candidate_index: int, current_index: int) -> bool:
        entry_index = candidate_index + 1
        return (
            entry_index <= current_index
            and high[current_index] >= open_[entry_index] * (1.0 + threshold)
        )

    def high_ready(candidate_index: int, current_index: int) -> bool:
        entry_index = candidate_index + 1
        return (
            entry_index <= current_index
            and low[current_index] <= open_[entry_index] * (1.0 - threshold)
        )

    def build_low(candidate_index: int, confirmation_index: int) -> ConfirmedSwing:
        entry_index = candidate_index + 1
        entry_price = float(open_[entry_index])
        confirmation_price = float(high[confirmation_index])
        return ConfirmedSwing(
            direction="low",
            extreme_index=candidate_index,
            entry_index=entry_index,
            confirmation_index=confirmation_index,
            extreme_price=float(low[candidate_index]),
            entry_price=entry_price,
            confirmation_price=confirmation_price,
            move_pct=confirmation_price / entry_price - 1.0,
        )

    def build_high(candidate_index: int, confirmation_index: int) -> ConfirmedSwing:
        entry_index = candidate_index + 1
        entry_price = float(open_[entry_index])
        confirmation_price = float(low[confirmation_index])
        return ConfirmedSwing(
            direction="high",
            extreme_index=candidate_index,
            entry_index=entry_index,
            confirmation_index=confirmation_index,
            extreme_price=float(high[candidate_index]),
            entry_price=entry_price,
            confirmation_price=confirmation_price,
            move_pct=1.0 - confirmation_price / entry_price,
        )

    for i in range(1, n):
        if mode is None:
            if high[i] > high[candidate_high]:
                candidate_high = i
            if low[i] < low[candidate_low]:
                candidate_low = i

            is_low_ready = low_ready(candidate_low, i)
            is_high_ready = high_ready(candidate_high, i)

            if is_low_ready and is_high_ready:
                # The earlier extreme is the only chronologically defensible
                # first pivot. Same-bar ambiguity is deliberately skipped.
                if candidate_low < candidate_high:
                    is_high_ready = False
                elif candidate_high < candidate_low:
                    is_low_ready = False
                else:
                    continue

            if is_low_ready:
                yield build_low(candidate_low, i)
                mode = "high"
                candidate_high = i
                continue

            if is_high_ready:
                yield build_high(candidate_high, i)
                mode = "low"
                candidate_low = i
                continue

        elif mode == "high":
            updated = False
            if high[i] > high[candidate_high]:
                candidate_high = i
                updated = True
            if not updated and high_ready(candidate_high, i):
                yield build_high(candidate_high, i)
                mode = "low"
                candidate_low = i

        else:  # mode == "low"
            updated = False
            if low[i] < low[candidate_low]:
                candidate_low = i
                updated = True
            if not updated and low_ready(candidate_low, i):
                yield build_low(candidate_low, i)
                mode = "high"
                candidate_high = i


class SwingExtremeMovePlugin:
    plugin_id = "swing_extreme_move"
    name = "Swing Extreme 后续涨跌幅"
    description = (
        "回看标记满足条件的 Swing Low / Swing High：low/high 用于确定 extreme，"
        "收益从 extreme 后下一根 K 线 open 开始计算，并在限定 bars 内由后续 high/low 完成目标。"
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
            description="1 = 1%；从 Swing extreme 后下一根 K 线 open 开始计算",
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

        required = {"open", "high", "low"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                "swing_extreme_move requires open/high/low columns, "
                f"missing={sorted(missing)}"
            )

        work = df[["open", "high", "low"]].copy()
        for column in ("open", "high", "low"):
            work[column] = pd.to_numeric(work[column], errors="coerce")
        valid = (
            work["open"].notna()
            & work["high"].notna()
            & work["low"].notna()
            & (work["open"] > 0)
            & (work["high"] > 0)
            & (work["low"] > 0)
        )
        work = work.loc[valid]
        if len(work) < 2:
            return PluginRunResult(
                markers=[],
                summary={"input_rows": int(len(df)), "matched": 0, "display": "有效 open/high/low 数据不足"},
            )

        timestamps = pd.DatetimeIndex(work.index)
        open_ = work["open"].to_numpy(dtype=float, copy=True)
        high = work["high"].to_numpy(dtype=float, copy=True)
        low = work["low"].to_numpy(dtype=float, copy=True)

        markers: list[Marker] = []
        total_confirmed = 0
        expired_by_horizon = 0
        low_count = 0
        high_count = 0

        for swing in _iter_confirmed_swings(open_, high, low, threshold):
            total_confirmed += 1
            if swing.completion_bars > max_completion_bars:
                expired_by_horizon += 1
                continue
            if direction != "both" and swing.direction != direction:
                continue

            extreme_ts = timestamps[swing.extreme_index]
            entry_ts = timestamps[swing.entry_index]
            confirmation_ts = timestamps[swing.confirmation_index]
            is_low = swing.direction == "low"
            if is_low:
                low_count += 1
                color = "#22c55e"
                label = f"Swing Low +{swing.move_pct * 100:.2f}%"
                reason = (
                    f"Swing Low 后次根开盘入场，{swing.completion_bars} bars 内上涨 "
                    f"{swing.move_pct * 100:.3f}%（目标 {move_pct_input:.3f}%）"
                )
                position = "low"
                symbol = "arrow_up"
            else:
                high_count += 1
                color = "#ef4444"
                label = f"Swing High -{swing.move_pct * 100:.2f}%"
                reason = (
                    f"Swing High 后次根开盘入场，{swing.completion_bars} bars 内下跌 "
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
                        "entry_timestamp": entry_ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "entry_price": swing.entry_price,
                        "entry_price_source": "next_open",
                        "confirmation_timestamp": confirmation_ts.strftime("%Y-%m-%d %H:%M:%S"),
                        "confirmation_price": swing.confirmation_price,
                        "confirmation_price_source": "high" if is_low else "low",
                        "completion_bars": swing.completion_bars,
                        "bars_from_entry": swing.bars_from_entry,
                        "target_move_pct": move_pct_input,
                        "realized_move_pct": swing.move_pct * 100.0,
                        "extreme_price_source": "low" if is_low else "high",
                        "retrospective_label": True,
                        "uses_future_confirmation": True,
                    },
                )
            )

        selected_label = {"both": "双向", "low": "Swing Low", "high": "Swing High"}[direction]
        display = (
            f"{selected_label}：{len(markers)} 个；Low {low_count}，High {high_count}；"
            f"目标 {move_pct_input:g}%（next open 起算），限时 {max_completion_bars} bars"
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
                "move_baseline": "next_open",
                "max_completion_bars": max_completion_bars,
                "retrospective_label": True,
                "display": display,
            },
        )
