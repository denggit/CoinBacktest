#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Observable Market State Map V3.1 plugin.

The plugin is a chart adapter for ``src.market_state``.  It does not duplicate
feature logic and displays every closed-bar state at its causal available time.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from analyze_tool.plugin_api import (
    IndicatorTrack,
    Marker,
    PluginParam,
    PluginRunContext,
    PluginRunResult,
    StateBand,
    StateBandCategory,
)
from src.market_state import (
    MarketStateConfig,
    MarketStateDataBundle,
    MarketStateEngine,
    ProcessMapConfig,
    ProcessMapEngine,
)
from src.market_state.causal_alignment import timeframe_to_timedelta


TREND_LABELS = {"up": "上涨背景", "balanced": "平衡背景", "down": "下跌背景", "warmup": "预热中"}
QUALITY_LABELS = {"high_order": "相对高有序", "normal": "正常质量", "noisy": "相对高噪声", "warmup": "预热中"}
PHASE_LABELS = {
    "startup": "启动", "continuation": "延续", "mature": "成熟", "decay": "衰减",
    "balanced": "平衡", "transition": "切换确认中", "warmup": "预热中",
}
PULSE_LABELS = {"up_pulse": "短期向上脉冲", "down_pulse": "短期向下脉冲", "neutral": "短期中性", "warmup": "预热中"}
VOLATILITY_LABELS = {
    "dormant": "市场沉寂", "compression": "波动压缩", "normal": "正常波动",
    "expansion": "波动扩张", "shock": "极端冲击", "warmup": "预热中",
}
FLOW_LABELS = {
    "unavailable": "订单流不可用", "balanced": "订单流平衡",
    "buy_pressure": "主动买压", "buy_building": "买压增强", "buy_persistent": "持续买压",
    "sell_pressure": "主动卖压", "sell_building": "卖压增强", "sell_persistent": "持续卖压",
}
IMPACT_LABELS = {
    "unavailable": "冲击不可用", "neutral": "价格响应中性", "mixed_response": "流量与价格混合",
    "buy_effective": "买压有效推动", "sell_effective": "卖压有效推动",
    "sell_absorbed": "卖压被吸收", "buy_absorbed": "买压被吸收",
}
LOCATION_LABELS = {
    "warmup": "位置预热中", "lower_zone": "结构下部", "middle_zone": "结构中部", "upper_zone": "结构上部",
    "near_support": "接近滚动支撑", "near_resistance": "接近滚动阻力",
    "downside_sweep_reclaim": "下扫后收回", "upside_sweep_reject": "上扫后拒绝",
    "breakout_accept": "向上突破接受", "breakdown_accept": "向下突破接受",
}
CONTEXT_LABELS = {
    "warmup": "上下文预热中", "wait": "等待", "risk_off": "冲击期暂停",
    "long_reversal_watch": "多头反转观察", "short_reversal_watch": "空头反转观察",
    "long_continuation_watch": "多头延续观察", "short_continuation_watch": "空头延续观察",
    "conflicted": "方向冲突",
}

TREND_COLORS = {"up": "#22c55e", "balanced": "#64748b", "down": "#ef4444", "warmup": "#334155"}
VOLATILITY_COLORS = {"dormant": "#475569", "compression": "#38bdf8", "normal": "#94a3b8", "expansion": "#fb923c", "shock": "#dc2626", "warmup": "#334155"}
FLOW_COLORS = {
    "unavailable": "#475569", "balanced": "#94a3b8",
    "buy_pressure": "#4ade80", "buy_building": "#22c55e", "buy_persistent": "#16a34a",
    "sell_pressure": "#fb7185", "sell_building": "#ef4444", "sell_persistent": "#b91c1c",
}
IMPACT_COLORS = {
    "unavailable": "#475569", "neutral": "#94a3b8", "mixed_response": "#facc15",
    "buy_effective": "#22c55e", "sell_effective": "#ef4444",
    "sell_absorbed": "#22d3ee", "buy_absorbed": "#d946ef",
}
LOCATION_COLORS = {
    "warmup": "#334155", "lower_zone": "#38bdf8", "middle_zone": "#64748b", "upper_zone": "#f59e0b",
    "near_support": "#2dd4bf", "near_resistance": "#f97316",
    "downside_sweep_reclaim": "#06b6d4", "upside_sweep_reject": "#e879f9",
    "breakout_accept": "#22c55e", "breakdown_accept": "#ef4444",
}
CONTEXT_COLORS = {
    "warmup": "#334155", "wait": "#64748b", "risk_off": "#7f1d1d",
    "long_reversal_watch": "#22d3ee", "short_reversal_watch": "#e879f9",
    "long_continuation_watch": "#22c55e", "short_continuation_watch": "#ef4444",
    "conflicted": "#facc15",
}

DIRECTION_LABELS = {"up": "上涨结构已确认", "balanced": "平衡结构", "down": "下跌结构已确认", "warmup": "预热中"}
DIRECTION_COLORS = TREND_COLORS
DISPLAY_PHASE_LABELS = {
    "warmup": "预热中",
    "up_continuation": "多头延续",
    "up_pullback": "多头回撤",
    "down_continuation": "空头延续",
    "down_rebound": "空头反弹",
    "balanced": "震荡整理",
    "compression": "波动压缩",
    "weakening": "趋势衰减",
    "transition": "方向确认中",
    "shock": "极端冲击",
}
DISPLAY_PHASE_COLORS = {
    "warmup": "#334155",
    "up_continuation": "#22c55e",
    "up_pullback": "#38bdf8",
    "down_continuation": "#ef4444",
    "down_rebound": "#f59e0b",
    "balanced": "#64748b",
    "compression": "#0ea5e9",
    "weakening": "#facc15",
    "transition": "#a78bfa",
    "shock": "#991b1b",
}
OBSERVATION_LABELS = {
    "warmup": "预热中",
    "wait": "观望",
    "risk_off": "暂停",
    "long_watch": "多头观察",
    "short_watch": "空头观察",
    "conflicted": "方向冲突",
}
OBSERVATION_COLORS = {
    "warmup": "#334155",
    "wait": "#64748b",
    "risk_off": "#7f1d1d",
    "long_watch": "#22c55e",
    "short_watch": "#ef4444",
    "conflicted": "#facc15",
}

PROCESS_LABELS = {
    "none:idle": "暂无多阶段过程",
    "conflict:idle": "多空过程冲突",
    "compression_setup:compression": "波动压缩 · 等待方向选择",
    "long_reversal:sell_pressure": "多头反转 1/4 · 卖压阶段",
    "long_reversal:sell_absorption": "多头反转 2/4 · 卖压吸收",
    "long_reversal:sweep_reclaim": "多头反转 3/4 · 下扫收回",
    "long_reversal:confirmed_buy_recovery": "多头反转 4/4 · 严格买盘恢复",
    "short_reversal:buy_pressure": "空头反转 1/4 · 买压阶段",
    "short_reversal:buy_absorption": "空头反转 2/4 · 买压吸收",
    "short_reversal:sweep_reject": "空头反转 3/4 · 上扫拒绝",
    "short_reversal:confirmed_sell_recovery": "空头反转 4/4 · 严格卖盘恢复",
    "long_breakout:compression_ready": "向上突破 1/3 · 压缩已成熟",
    "long_breakout:breakout_impulse": "向上突破 2/3 · 新买方突破冲击",
    "long_breakout:retest_hold_accept": "向上突破 3/3 · 回踩/停留接受",
    "short_breakdown:compression_ready": "向下突破 1/3 · 压缩已成熟",
    "short_breakdown:breakdown_impulse": "向下突破 2/3 · 新卖方跌破冲击",
    "short_breakdown:retest_hold_accept": "向下突破 3/3 · 回抽/停留接受",
}
PROCESS_COLORS = {
    "none:idle": "#64748b", "conflict:idle": "#facc15",
    "compression_setup:compression": "#38bdf8",
    "long_reversal:sell_pressure": "#64748b", "long_reversal:sell_absorption": "#22d3ee",
    "long_reversal:sweep_reclaim": "#06b6d4", "long_reversal:confirmed_buy_recovery": "#22c55e",
    "short_reversal:buy_pressure": "#64748b", "short_reversal:buy_absorption": "#d946ef",
    "short_reversal:sweep_reject": "#e879f9", "short_reversal:confirmed_sell_recovery": "#ef4444",
    "long_breakout:compression_ready": "#38bdf8", "long_breakout:breakout_impulse": "#4ade80",
    "long_breakout:retest_hold_accept": "#16a34a",
    "short_breakdown:compression_ready": "#38bdf8", "short_breakdown:breakdown_impulse": "#fb7185",
    "short_breakdown:retest_hold_accept": "#b91c1c",
}


def _as_list(series: pd.Series, digits: int = 4) -> list[Any]:
    numeric = pd.to_numeric(series, errors="coerce").round(digits)
    return numeric.astype(object).where(numeric.notna(), None).tolist()


def _categorical_field(series: pd.Series, labels: dict[str, str] | None = None) -> dict[str, Any]:
    mapping = labels or {}
    values = series.astype(object).where(series.notna(), None)
    unique = [str(value) for value in dict.fromkeys(values) if value is not None and str(value) != "nan"]
    codes = {value: index + 1 for index, value in enumerate(unique)}
    return {
        "values": [None if value is None or str(value) == "nan" else codes[str(value)] for value in values],
        "categories": {str(code): mapping.get(value, value) for value, code in codes.items()},
    }


def _fmt_time(value: pd.Timestamp) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def _aligned_to_display(result_frame: pd.DataFrame, display_index: pd.DatetimeIndex) -> pd.DataFrame:
    source = result_frame.copy()
    source["source_bar_timestamp"] = pd.DatetimeIndex(source.index)
    source.index = pd.DatetimeIndex(pd.to_datetime(source["available_time"]))
    source = source.loc[~source.index.duplicated(keep="last")].sort_index()
    return source.reindex(display_index)


def _derive_display_phase(frame: pd.DataFrame) -> pd.Series:
    out: list[str] = []
    for ready, trend, phase, pulse, volatility in zip(
        frame["data_ready"].eq(True).to_numpy(dtype=bool),
        frame["trend_state"].astype(str),
        frame["trend_phase"].astype(str),
        frame["fast_pulse_state"].astype(str),
        frame["volatility_state"].astype(str),
    ):
        if not ready or trend == "warmup":
            out.append("warmup")
        elif volatility == "shock":
            out.append("shock")
        elif trend == "balanced":
            if phase == "transition":
                out.append("transition")
            elif volatility in {"dormant", "compression"}:
                out.append("compression")
            else:
                out.append("balanced")
        elif phase == "decay":
            out.append("weakening")
        elif trend == "up" and pulse == "down_pulse":
            out.append("up_pullback")
        elif trend == "down" and pulse == "up_pulse":
            out.append("down_rebound")
        elif trend == "up":
            out.append("up_continuation")
        else:
            out.append("down_continuation")
    return pd.Series(out, index=frame.index, dtype="object")


def _derive_observation_state(frame: pd.DataFrame) -> pd.Series:
    mapping = {
        "warmup": "warmup",
        "wait": "wait",
        "risk_off": "risk_off",
        "long_reversal_watch": "long_watch",
        "long_continuation_watch": "long_watch",
        "short_reversal_watch": "short_watch",
        "short_continuation_watch": "short_watch",
        "conflicted": "conflicted",
    }
    return frame["trade_context_state"].astype(str).map(mapping).fillna("wait")


def _build_brief_fields(frame: pd.DataFrame) -> dict[str, Any]:
    reason_1: list[Any] = []
    reason_2: list[Any] = []
    reason_3: list[Any] = []
    advice: list[Any] = []
    context_detail: list[Any] = []
    for row in frame.itertuples(index=False):
        ready = bool(getattr(row, "data_ready", False))
        trend = str(getattr(row, "trend_state", "warmup"))
        process_key = str(getattr(row, "process_display_state", "none:idle"))
        process_status = str(getattr(row, "process_status", "idle"))
        probability = getattr(row, "process_direction_probability", None)
        uplift = getattr(row, "process_direction_probability_uplift", None)
        samples_raw = getattr(row, "process_direction_samples", 0)
        ttl_raw = getattr(row, "process_ttl_remaining_bars", 0)
        samples = 0 if pd.isna(samples_raw) else int(samples_raw)
        ttl = 0 if pd.isna(ttl_raw) else int(ttl_raw)
        if not ready:
            reason_1.append("状态仍在预热")
            reason_2.append("等待足够历史数据")
            reason_3.append(None)
            advice.append("暂不使用当前状态辅助判断。")
            context_detail.append("预热中")
            continue
        reason_1.append(f"历史结构：{DIRECTION_LABELS.get(trend, trend)}")
        reason_2.append(PROCESS_LABELS.get(process_key, process_key))
        if pd.notna(probability) and samples > 0:
            reason_3.append("历史结算样本已达到概率显示门槛")
        else:
            reason_3.append("历史结算样本不足，暂不显示概率")
        if process_status == "conflict":
            advice.append("多空过程同时推进且强度接近，优先观望。")
        elif process_status == "complete":
            advice.append("多阶段顺序已完成，但仍需具体策略确认；这不是开仓信号。")
        elif process_status == "building":
            advice.append("过程正在推进，等待后续阶段在有效期内出现。")
        else:
            advice.append("暂无完整市场过程，继续等待订单流、吸收与位置按顺序形成。")
        context_detail.append(PROCESS_LABELS.get(process_key, process_key))
    return {
        "brief_reason_1": _categorical_field(pd.Series(reason_1, index=frame.index, dtype="object")),
        "brief_reason_2": _categorical_field(pd.Series(reason_2, index=frame.index, dtype="object")),
        "brief_reason_3": _categorical_field(pd.Series(reason_3, index=frame.index, dtype="object")),
        "brief_advice": _categorical_field(pd.Series(advice, index=frame.index, dtype="object")),
        "brief_context_detail": _categorical_field(pd.Series(context_detail, index=frame.index, dtype="object")),
    }


def _categorical_band(
    *,
    frame: pd.DataFrame,
    column: str,
    band_id: str,
    label: str,
    labels: dict[str, str],
    colors: dict[str, str],
    description: str,
    render_mode: str = "strip",
    height_px: int = 9,
) -> StateBand:
    present = [str(value) for value in dict.fromkeys(frame[column].astype(str)) if str(value) != "nan"]
    categories: list[StateBandCategory] = []
    codes: dict[str, int] = {}
    for state_name in present:
        code = len(categories) + 1
        codes[state_name] = code
        categories.append(
            StateBandCategory(
                code=code,
                label=labels.get(state_name, state_name),
                color=colors.get(state_name, "#64748b"),
                opacity=0.84 if render_mode == "strip" else (0.072 if state_name != "balanced" else 0.045),
                status=state_name,
                fields={column: labels.get(state_name, state_name)},
            )
        )
    return StateBand(
        band_id=band_id,
        label=label,
        codes=[codes.get(str(value)) for value in frame[column]],
        categories=categories,
        description=description,
        render_mode=render_mode,
        position="bottom",
        height_px=height_px,
    )


class MarketStateMapPlugin:
    plugin_id = "market_state_map_v0"
    name = "市场状态地图 V3.1"
    description = (
        "把订单流、吸收、Sweep/Reclaim 与突破接受组织成有顺序和有效期的多阶段过程；"
        "概率只使用当时已经结算的历史过程，不是交易信号。"
    )
    params = [
        PluginParam(
            name="view_mode",
            label="展示模式",
            kind="select",
            default="trading",
            choices=[
                {"value": "trading", "label": "交易视图（默认，极简）"},
                {"value": "research", "label": "研究视图（完整指标）"},
            ],
        ),
        PluginParam("fast_trend_window", "短期脉冲窗口（bars）", default=16, min_value=6, max_value=500, step=1),
        PluginParam("trend_window", "中期结构窗口（bars）", default=64, min_value=16, max_value=2000, step=1),
        PluginParam("slow_trend_window", "慢速背景窗口（bars）", default=240, min_value=48, max_value=5000, step=1),
        PluginParam("volatility_window", "波动观察窗口（bars）", default=30, min_value=8, max_value=1000, step=1),
        PluginParam("activity_window", "市场活动窗口（bars）", default=12, min_value=3, max_value=500, step=1),
        PluginParam("baseline_window", "因果历史基线（bars）", default=720, min_value=240, max_value=10000, step=1),
        PluginParam("flow_fast_window", "订单流快速窗口（bars）", default=3, min_value=2, max_value=60, step=1),
        PluginParam("flow_window", "订单流主窗口（bars）", default=12, min_value=3, max_value=240, step=1),
        PluginParam("flow_slow_window", "订单流慢速窗口（bars）", default=30, min_value=6, max_value=500, step=1),
        PluginParam("flow_threshold", "订单流方向阈值", default=0.06, min_value=0.01, max_value=0.50, step=0.01),
        PluginParam("absorption_threshold", "吸收观察阈值", default=0.35, min_value=0.10, max_value=0.90, step=0.05),
        PluginParam("location_window", "局部关键位置窗口（bars）", default=60, min_value=20, max_value=2000, step=1),
        PluginParam("structure_window", "结构位置窗口（bars）", default=240, min_value=60, max_value=5000, step=1),
        PluginParam("trend_enter_threshold", "趋势进入阈值", default=0.24, min_value=0.05, max_value=0.8, step=0.01),
        PluginParam("trend_exit_threshold", "趋势退出阈值", default=0.10, min_value=0.01, max_value=0.5, step=0.01),
        PluginParam("trend_confirm_bars", "趋势切换确认（bars）", default=3, min_value=1, max_value=30, step=1),
        PluginParam("min_state_bars", "方向状态最短持续（bars）", default=15, min_value=1, max_value=240, step=1),
        PluginParam(
            name="show_watch_markers",
            label="显示多阶段过程完成节点（非交易信号）",
            kind="select",
            default="no",
            choices=[{"value": "no", "label": "不显示（推荐）"}, {"value": "yes", "label": "显示"}],
        ),
    ]

    def run(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> PluginRunResult:
        context = PluginRunContext(
            display_df=df,
            visible_df=df,
            analysis_frames={},
            request={"data_type": "unknown", "timeframe": "", "range_pct": 0.0},
            meta={},
        )
        return self.run_with_context(context, params)

    def run_with_context(self, context: PluginRunContext, params: dict[str, Any] | None) -> PluginRunResult:
        df = context.visible_df
        if df is None or df.empty:
            return PluginRunResult(markers=[], summary={"input_rows": 0, "matched": 0, "display": "无数据"})

        p = params or {}
        view_mode = str(p.get("view_mode", "trading")).lower()
        if view_mode not in {"trading", "research"}:
            view_mode = "trading"
        config = MarketStateConfig(
            fast_trend_window=int(float(p.get("fast_trend_window", 16))),
            trend_window=int(float(p.get("trend_window", 64))),
            slow_trend_window=int(float(p.get("slow_trend_window", 240))),
            volatility_window=int(float(p.get("volatility_window", 30))),
            activity_window=int(float(p.get("activity_window", 12))),
            baseline_window=int(float(p.get("baseline_window", 720))),
            flow_fast_window=int(float(p.get("flow_fast_window", 3))),
            flow_window=int(float(p.get("flow_window", 12))),
            flow_slow_window=int(float(p.get("flow_slow_window", 30))),
            flow_threshold=float(p.get("flow_threshold", 0.06)),
            absorption_threshold=float(p.get("absorption_threshold", 0.35)),
            location_window=int(float(p.get("location_window", 60))),
            structure_window=int(float(p.get("structure_window", 240))),
            directional_threshold=float(p.get("trend_enter_threshold", 0.24)),
            trend_exit_threshold=float(p.get("trend_exit_threshold", 0.10)),
            trend_confirm_bars=int(float(p.get("trend_confirm_bars", 3))),
            min_state_bars=int(float(p.get("min_state_bars", 15))),
        )
        config.validate()

        is_range = context.data_type == "range_bar"
        timestamp_semantics = "bar_end" if is_range or not context.timeframe else "bar_start"
        bar_duration = None if timestamp_semantics == "bar_end" else timeframe_to_timedelta(context.timeframe)
        bundle = MarketStateDataBundle.from_frame(
            df,
            source=context.data_type,
            timestamp_semantics=timestamp_semantics,
            bar_duration=bar_duration,
            metadata={"timeframe": context.timeframe, "range_pct": context.range_pct},
        )
        result = MarketStateEngine(config).compute(bundle)
        process_result = ProcessMapEngine(ProcessMapConfig()).compute(result.frame)
        display_index = pd.DatetimeIndex(pd.to_datetime(df.index))
        frame = _aligned_to_display(process_result.frame, display_index)
        frame["display_phase_state"] = _derive_display_phase(frame)
        frame["display_observation_state"] = _derive_observation_state(frame)
        frame["process_display_state"] = (
            frame["process_family"].astype(str) + ":" + frame["process_stage_label"].astype(str)
        )
        brief_fields = _build_brief_fields(frame)

        trading_bands = [
            _categorical_band(
                frame=frame, column="trend_state", band_id="direction_permission", label="历史结构",
                labels=DIRECTION_LABELS, colors=DIRECTION_COLORS,
                description="只描述已经确认的历史价格结构，不代表未来方向许可",
                render_mode="background", height_px=10,
            ),
            _categorical_band(
                frame=frame, column="display_phase_state", band_id="market_phase", label="当前阶段",
                labels=DISPLAY_PHASE_LABELS, colors=DISPLAY_PHASE_COLORS,
                description="区分延续、回撤/反弹、整理、衰减和冲击", height_px=9,
            ),
            _categorical_band(
                frame=frame, column="process_display_state", band_id="market_process", label="多阶段过程",
                labels=PROCESS_LABELS, colors=PROCESS_COLORS,
                description="按先后顺序和有效期推进；完成过程仍只是条件概率上下文，不是开仓信号", height_px=11,
            ),
        ]

        research_bands = [
            _categorical_band(
                frame=frame, column="trend_state", band_id="trend_background", label="结构方向",
                labels=TREND_LABELS, colors=TREND_COLORS,
                description="背景只表示稳定后的上涨/平衡/下跌，不被阶段或质量切碎",
                render_mode="background", height_px=10,
            ),
            _categorical_band(
                frame=frame, column="volatility_state", band_id="volatility_regime", label="波动状态",
                labels=VOLATILITY_LABELS, colors=VOLATILITY_COLORS,
                description="波动压缩、扩张与冲击", height_px=8,
            ),
            _categorical_band(
                frame=frame, column="flow_state", band_id="orderflow_regime", label="主动订单流",
                labels=FLOW_LABELS, colors=FLOW_COLORS,
                description="仅使用真实 trade-bar 主动买卖字段；普通K线不会伪造订单流", height_px=9,
            ),
            _categorical_band(
                frame=frame, column="impact_state", band_id="impact_regime", label="冲击与吸收",
                labels=IMPACT_LABELS, colors=IMPACT_COLORS,
                description="区分买卖压力有效推动价格，还是被对手方吸收", height_px=9,
            ),
            _categorical_band(
                frame=frame, column="location_state", band_id="location_regime", label="关键位置",
                labels=LOCATION_LABELS, colors=LOCATION_COLORS,
                description="相对当前bar之前的滚动高低点，包含扫过、拒绝和突破接受", height_px=9,
            ),
            _categorical_band(
                frame=frame, column="trade_context_state", band_id="trade_context", label="观察条件",
                labels=CONTEXT_LABELS, colors=CONTEXT_COLORS,
                description="规则化观察标签，尚未经过收益回测，不是开仓信号", height_px=10,
            ),
        ]
        bands = trading_bands if view_mode == "trading" else research_bands

        markers: list[Marker] = []
        if str(p.get("show_watch_markers", "no")).lower() == "yes":
            completed = frame["process_status"].astype(str).eq("complete")
            changed = completed & ~completed.shift(1, fill_value=False)
            rows = frame.loc[changed]
            for idx, row in rows.iterrows():
                family = str(row["process_family"])
                direction = int(row["process_direction"])
                display_key = f"{family}:{row['process_stage_label']}"
                markers.append(
                    Marker(
                        timestamp=_fmt_time(pd.Timestamp(idx)),
                        label=PROCESS_LABELS.get(display_key, display_key),
                        color=PROCESS_COLORS.get(display_key, "#facc15"),
                        reason="多阶段顺序已经完成；概率仅来自当时可见的历史结算样本",
                        role="node",
                        position="bottom" if direction > 0 else "top",
                        symbol="arrowUp" if direction > 0 else "arrowDown",
                        fields={
                            "source_bar_timestamp": _fmt_time(pd.Timestamp(row["source_bar_timestamp"])),
                            "available_time": _fmt_time(pd.Timestamp(row["available_time"])),
                            "process_family": family,
                            "process_confidence": row["process_confidence"],
                            "direction_probability": row["process_direction_probability"],
                            "direction_probability_uplift": row["process_direction_probability_uplift"],
                            "probability_samples": row["process_direction_samples"],
                            "probability_horizon_bars": row["process_probability_horizon_bars"],
                            "causal": True,
                            "not_trade_signal": True,
                        },
                    )
                )

        research_tracks = [
            IndicatorTrack("trend_score", "结构趋势", _as_list(frame["trend_score"]), "#38bdf8", -1.0, 1.0, 0.0, "中慢周期价格结构"),
            IndicatorTrack("fast_trend_score", "短期价格脉冲", _as_list(frame["fast_trend_score"]), "#facc15", -1.0, 1.0, 0.0, "短期价格方向"),
            IndicatorTrack("volatility_score", "波动状态", _as_list(frame["volatility_score"]), "#fb923c", 0.0, 1.0, 0.5, "相对历史波动"),
            IndicatorTrack("activity_score", "市场活动", _as_list(frame["activity_score"]), "#a78bfa", 0.0, 1.0, 0.5, "成交量及成交笔数"),
            IndicatorTrack("flow_score", "主动订单流", _as_list(frame["flow_score"]), "#2dd4bf", -1.0, 1.0, 0.0, "正值主动买入占优，负值主动卖出占优"),
            IndicatorTrack("flow_persistence", "订单流持续性", _as_list(frame["flow_persistence"]), "#14b8a6", -1.0, 1.0, 0.0, "方向连续性"),
            IndicatorTrack("flow_price_effectiveness", "订单流价格有效性", _as_list(frame["flow_price_effectiveness"]), "#60a5fa", -1.0, 1.0, 0.0, "正值流量推动价格，负值流量被逆向消化"),
            IndicatorTrack("signed_absorption_score", "净吸收", _as_list(frame["signed_absorption_score"]), "#d946ef", -1.0, 1.0, 0.0, "正值卖压被吸收，负值买压被吸收"),
            IndicatorTrack("structural_location_score", "结构位置", _as_list(frame["structural_location_score"]), "#f59e0b", -1.0, 1.0, 0.0, "-1靠近结构下部，+1靠近结构上部"),
            IndicatorTrack("trade_context_score", "观察条件倾向", _as_list(frame["trade_context_score"]), "#e2e8f0", -1.0, 1.0, 0.0, "旧版同bar观察分数，仅供对照"),
            IndicatorTrack("process_progress", "过程完成度", _as_list(frame["process_progress"]), "#facc15", 0.0, 1.0, 0.0, "多阶段过程当前完成比例"),
            IndicatorTrack("process_direction_probability_uplift", "历史概率增量", _as_list(frame["process_direction_probability_uplift"]), "#a78bfa", -0.25, 0.25, 0.0, "相对同方向全市场历史基线；低样本时为空"),
        ]
        tracks = [] if view_mode == "trading" else research_tracks

        brief_row_fields = {
            "brief_direction": _categorical_field(frame["trend_state"], DIRECTION_LABELS),
            "brief_phase": _categorical_field(frame["display_phase_state"], DISPLAY_PHASE_LABELS),
            "brief_process": _categorical_field(frame["process_display_state"], PROCESS_LABELS),
            "brief_process_probability": _as_list(frame["process_direction_probability"]),
            "brief_process_probability_uplift": _as_list(frame["process_direction_probability_uplift"]),
            "brief_process_samples": _as_list(frame["process_direction_samples"], digits=0),
            **brief_fields,
        }
        research_row_fields = {
            "trend_phase": _categorical_field(frame["trend_phase"], PHASE_LABELS),
            "trend_quality": _categorical_field(frame["trend_quality_state"], QUALITY_LABELS),
            "fast_pulse": _categorical_field(frame["fast_pulse_state"], PULSE_LABELS),
            "flow_state": _categorical_field(frame["flow_state"], FLOW_LABELS),
            "impact_state": _categorical_field(frame["impact_state"], IMPACT_LABELS),
            "location_state": _categorical_field(frame["location_state"], LOCATION_LABELS),
            "trade_context": _categorical_field(frame["trade_context_state"], CONTEXT_LABELS),
            "process_state": _categorical_field(frame["process_display_state"], PROCESS_LABELS),
            "process_status": _categorical_field(frame["process_status"]),
            "process_progress": _as_list(frame["process_progress"]),
            "process_confidence": _as_list(frame["process_confidence"]),
            "process_ttl_remaining": _as_list(frame["process_ttl_remaining_bars"], digits=0),
            "process_completion_probability": _as_list(frame["process_completion_probability"]),
            "process_completion_samples": _as_list(frame["process_completion_samples"], digits=0),
            "process_direction_probability": _as_list(frame["process_direction_probability"]),
            "process_direction_probability_uplift": _as_list(frame["process_direction_probability_uplift"]),
            "process_direction_samples": _as_list(frame["process_direction_samples"], digits=0),
            "trend_candidate": _categorical_field(frame["trend_candidate_state"], TREND_LABELS),
            "trend_candidate_progress": _as_list(frame["trend_candidate_progress"]),
            "trend_state_age": _as_list(frame["trend_state_age"], digits=0),
            "volatility_state_age": _as_list(frame["volatility_state_age"], digits=0),
            "orderflow_available": frame["orderflow_available"].astype(object).where(frame["orderflow_available"].notna(), None).tolist(),
            "local_support": _as_list(frame["local_support"], digits=4),
            "local_resistance": _as_list(frame["local_resistance"], digits=4),
            "structural_support": _as_list(frame["structural_support"], digits=4),
            "structural_resistance": _as_list(frame["structural_resistance"], digits=4),
            "sell_absorption": _as_list(frame["sell_absorption_score"]),
            "buy_absorption": _as_list(frame["buy_absorption_score"]),
            **brief_row_fields,
        }
        row_fields = brief_row_fields if view_mode == "trading" else research_row_fields


        aligned_ready_rows = int(frame["data_ready"].eq(True).sum())
        orderflow_ready_rows = int(frame["orderflow_available"].eq(True).sum())
        location_ready_rows = int(frame["location_available"].eq(True).sum())
        semantics_text = (
            "Range Bar end_ts 已是可用时间，状态画在同一根 bar"
            if timestamp_semantics == "bar_end"
            else f"{context.timeframe} 左标签 bar 的状态向后移动到 timestamp + {bar_duration} 才显示"
        )
        orderflow_text = (
            f"订单流可用 {orderflow_ready_rows} 根"
            if orderflow_ready_rows > 0
            else "订单流不可用：请使用 Trade Bar 或带主动成交字段的 Range Bar，并确认 buy/sell/delta_notional 已回填"
        )
        mode_text = "交易视图" if view_mode == "trading" else "研究视图"
        display = (
            f"V3.1 {mode_text}：基础状态 {aligned_ready_rows}/{len(frame)}；{orderflow_text}；"
            f"关键位置可用 {location_ready_rows} 根；多阶段过程与概率不是交易信号"
        )
        return PluginRunResult(
            markers=markers,
            tracks=tracks,
            bands=bands,
            row_fields=row_fields,
            summary={
                "input_rows": int(len(frame)),
                "matched": aligned_ready_rows,
                "ready_rows": aligned_ready_rows,
                "warmup_rows": int(len(frame) - aligned_ready_rows),
                "segment_count": int(len(result.segments)),
                "watch_marker_count": int(len(markers)),
                "orderflow_ready_rows": orderflow_ready_rows,
                "location_ready_rows": location_ready_rows,
                "orderflow_status": orderflow_text,
                "orderflow_coverage": result.metadata.get("orderflow_coverage", {}),
                "trend_state_counts": result.metadata.get("trend_state_counts", {}),
                "volatility_state_counts": result.metadata.get("volatility_state_counts", {}),
                "flow_state_counts": result.metadata.get("flow_state_counts", {}),
                "impact_state_counts": result.metadata.get("impact_state_counts", {}),
                "location_state_counts": result.metadata.get("location_state_counts", {}),
                "trade_context_counts": result.metadata.get("trade_context_counts", {}),
                "process_metadata": process_result.metadata,
                "process_state_counts": frame["process_display_state"].value_counts().to_dict(),
                "timestamp_semantics": timestamp_semantics,
                "state_display_lag_bars": 0 if timestamp_semantics == "bar_end" else 1,
                "causal_availability": semantics_text,
                "data_quality": result.data_quality.as_dict(),
                "not_trade_signal": True,
                "ui": {
                    "view_mode": view_mode,
                    "compact": view_mode == "trading",
                    "brief_available": True,
                    "advanced_collapsed": True,
                },
                "config": {
                    "fast_trend_window": config.fast_trend_window,
                    "trend_window": config.trend_window,
                    "slow_trend_window": config.slow_trend_window,
                    "volatility_window": config.volatility_window,
                    "activity_window": config.activity_window,
                    "baseline_window": config.baseline_window,
                    "flow_fast_window": config.flow_fast_window,
                    "flow_window": config.flow_window,
                    "flow_slow_window": config.flow_slow_window,
                    "flow_threshold": config.flow_threshold,
                    "absorption_threshold": config.absorption_threshold,
                    "location_window": config.location_window,
                    "structure_window": config.structure_window,
                },
                "display": display,
            },
        )
