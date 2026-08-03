#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Visualize the frozen R03.3.3.1 market-state context on Analyze Tool candles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analyze_tool.ai_market_state_artifacts import (
    DEFAULT_ARTIFACT_DIR,
    load_activity_prediction_frame,
    load_state_frame_for_range,
)
from analyze_tool.plugin_api import (
    IndicatorTrack,
    Marker,
    PluginParam,
    PluginRunContext,
    PluginRunResult,
    StateBand,
    StateBandCategory,
)
from src.ai_research.market_state_continuity.config import (
    DEFAULT_MARKET_STATE_CONTINUITY_CONFIG,
    MarketStateContinuityConfig,
)

DIRECTION_LABELS = {-1: "偏空", 0: "中性", 1: "偏多"}
TACTICAL_LABELS = {-1: "向下过程", 0: "整理/不明确", 1: "向上过程"}
ENTRY_LABELS = {-1: "局部向下", 0: "局部中性", 1: "局部向上"}
ACTIVITY_LABELS = {-1: "低活跃/压缩", 0: "普通活跃", 1: "高活跃/扩张"}

DIRECTION_COLORS = {-1: "#ef4444", 0: "#64748b", 1: "#22c55e"}
ACTIVITY_COLORS = {-1: "#38bdf8", 0: "#94a3b8", 1: "#fb923c"}


def _datetime_index_ns(values: Any) -> pd.DatetimeIndex:
    """Normalize all chart/state time axes to timezone-naive nanoseconds.

    Pandas 2/3 may preserve ``datetime64[us]`` from SQLite/Arrow while model
    artifacts are stored as ``datetime64[ns]``. ``merge_asof`` requires exact
    dtype equality, so normalization must happen on both sides immediately
    before every causal join/reindex.
    """
    index = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    if index.tz is not None:
        index = index.tz_convert(None)
    try:
        return index.as_unit("ns")
    except AttributeError:
        return pd.DatetimeIndex(index.to_numpy(dtype="datetime64[ns]"))


def _number_list(series: pd.Series, digits: int = 4) -> list[Any]:
    values = pd.to_numeric(series, errors="coerce").round(digits)
    return values.astype(object).where(values.notna(), None).tolist()


def _categorical_field(series: pd.Series, labels: dict[int, str]) -> dict[str, Any]:
    numeric = pd.to_numeric(series, errors="coerce")
    return {
        "values": [None if pd.isna(value) else int(value) for value in numeric],
        "categories": {str(code): label for code, label in labels.items()},
    }


def _text_field(values: list[str | None]) -> dict[str, Any]:
    unique = [value for value in dict.fromkeys(values) if value]
    categories = {index + 1: value for index, value in enumerate(unique)}
    reverse = {value: code for code, value in categories.items()}
    return {
        "values": [None if not value else reverse[value] for value in values],
        "categories": {str(code): value for code, value in categories.items()},
    }


def _state_band(
    *,
    frame: pd.DataFrame,
    column: str,
    band_id: str,
    label: str,
    labels: dict[int, str],
    colors: dict[int, str],
    background: bool = False,
    height_px: int = 9,
) -> StateBand:
    categories = [
        StateBandCategory(
            code=code,
            label=labels[code],
            color=colors[code],
            opacity=0.055 if background else 0.84,
            status=labels[code],
            fields={"state_code": code, "not_trade_signal": True},
        )
        for code in (-1, 0, 1)
    ]
    codes = [None if pd.isna(value) else int(value) for value in pd.to_numeric(frame[column], errors="coerce")]
    return StateBand(
        band_id=band_id,
        label=label,
        codes=codes,
        categories=categories,
        description="R03.3.3.1 因果状态缓存；只描述上下文，不是开仓信号",
        render_mode="background" if background else "strip",
        position="bottom",
        height_px=height_px,
    )


def _align_causally(source: pd.DataFrame, display_index: pd.DatetimeIndex) -> pd.DataFrame:
    display_index_ns = _datetime_index_ns(display_index)
    if source.empty:
        empty = pd.DataFrame(index=display_index_ns, columns=source.columns)
        empty["state_time"] = pd.NaT
        return empty

    normalized_source = source.copy()
    normalized_source.index = _datetime_index_ns(normalized_source.index)
    normalized_source = normalized_source.sort_index()
    if normalized_source.index.has_duplicates:
        normalized_source = normalized_source[~normalized_source.index.duplicated(keep="last")]

    left = pd.DataFrame({"display_time": display_index_ns}).sort_values("display_time")
    right = normalized_source.reset_index(names="state_time")
    # reset_index can retain the original datetime resolution on some pandas
    # versions. Cast the actual merge columns again, not only their source index.
    left["display_time"] = left["display_time"].astype("datetime64[ns]")
    right["state_time"] = right["state_time"].astype("datetime64[ns]")
    aligned = pd.merge_asof(
        left,
        right,
        left_on="display_time",
        right_on="state_time",
        direction="backward",
        tolerance=pd.Timedelta(minutes=30),
        allow_exact_matches=True,
    )
    aligned.index = _datetime_index_ns(aligned.pop("display_time"))
    return aligned.reindex(display_index_ns)


def _state_name(value: Any, labels: dict[int, str]) -> str:
    if value is None or pd.isna(value):
        return "无缓存"
    return labels.get(int(value), str(value))


class AIMarketStateTimelinePlugin:
    plugin_id = "ai_market_state_timeline_r03_3_3_1"
    name = "AI多周期市场状态 R03.3.3.1"
    description = (
        "把冻结的战略、战术、入场、活跃度状态和2024/2025完整OOS活跃持续概率按时序叠加到K线；"
        "用于人工验证状态语义，不是交易信号。"
    )
    params = [
        PluginParam(
            name="view_mode",
            label="展示模式",
            kind="select",
            default="overview",
            choices=[
                {"value": "overview", "label": "概览（4层状态 + 活跃概率）"},
                {"value": "research", "label": "研究（完整分数、年龄与对齐）"},
            ],
        ),
        PluginParam(
            name="show_transition_markers",
            label="显示状态切换节点",
            kind="select",
            default="activity_strategic",
            choices=[
                {"value": "none", "label": "不显示"},
                {"value": "activity_strategic", "label": "仅战略与活跃度（推荐）"},
                {"value": "all", "label": "全部四层"},
            ],
        ),
        PluginParam(
            name="show_outcome_audit",
            label="显示历史未来3小时结果审计",
            kind="select",
            default="yes",
            choices=[
                {"value": "yes", "label": "显示（推荐，用于核对预测）"},
                {"value": "no", "label": "隐藏"},
            ],
            description="该色带使用未来3小时真实结果，只用于历史审计，绝不是模型输入或实盘状态。",
        ),
    ]

    def __init__(
        self,
        *,
        config: MarketStateContinuityConfig = DEFAULT_MARKET_STATE_CONTINUITY_CONFIG,
        artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    ) -> None:
        self.config = config
        self.artifact_dir = Path(artifact_dir)

    def run(self, df: pd.DataFrame, params: dict[str, Any] | None = None) -> PluginRunResult:
        return self.run_with_context(
            PluginRunContext(
                display_df=df,
                visible_df=df,
                request={"data_type": "unknown", "timeframe": "", "symbol": self.config.symbol},
                meta={},
            ),
            params,
        )

    def run_with_context(self, context: PluginRunContext, params: dict[str, Any] | None) -> PluginRunResult:
        df = context.visible_df
        if df is None or df.empty:
            return PluginRunResult(markers=[], summary={"input_rows": 0, "matched": 0, "display": "无K线数据"})
        symbol = str(context.request.get("symbol") or context.meta.get("symbol") or self.config.symbol)
        if symbol != self.config.symbol:
            return PluginRunResult(
                markers=[],
                summary={
                    "input_rows": int(len(df)),
                    "matched": 0,
                    "display": f"该模型只支持 {self.config.symbol}，当前为 {symbol}",
                    "not_trade_signal": True,
                },
            )

        p = params or {}
        view_mode = str(p.get("view_mode", "overview")).lower()
        if view_mode not in {"overview", "research"}:
            view_mode = "overview"
        marker_mode = str(p.get("show_transition_markers", "activity_strategic")).lower()
        display_index = _datetime_index_ns(df.index).sort_values()
        read_start = pd.Timestamp(display_index.min()) - pd.Timedelta(minutes=30)
        read_end = pd.Timestamp(display_index.max())
        state_frame = load_state_frame_for_range(start=read_start, end=read_end, config=self.config)
        prediction_frame = load_activity_prediction_frame(
            start=read_start,
            end=read_end,
            artifact_dir=self.artifact_dir,
        )
        state_frame = state_frame.copy()
        prediction_frame = prediction_frame.copy()
        state_frame.index = _datetime_index_ns(state_frame.index)
        prediction_frame.index = _datetime_index_ns(prediction_frame.index)
        combined = state_frame.join(prediction_frame, how="left")
        aligned = _align_causally(combined, display_index)
        aligned = aligned.reindex(_datetime_index_ns(df.index))

        bands = [
            _state_band(
                frame=aligned,
                column="strategic_state",
                band_id="ai_strategic_state",
                label="战略状态 1D/4H",
                labels=DIRECTION_LABELS,
                colors=DIRECTION_COLORS,
                background=True,
                height_px=10,
            ),
            _state_band(
                frame=aligned,
                column="tactical_state",
                band_id="ai_tactical_state",
                label="战术状态 4H/1H",
                labels=TACTICAL_LABELS,
                colors=DIRECTION_COLORS,
                height_px=9,
            ),
            _state_band(
                frame=aligned,
                column="entry_state",
                band_id="ai_entry_state",
                label="入场状态 30m→1m",
                labels=ENTRY_LABELS,
                colors=DIRECTION_COLORS,
                height_px=9,
            ),
            _state_band(
                frame=aligned,
                column="activity_state",
                band_id="ai_activity_state",
                label="活跃度状态",
                labels=ACTIVITY_LABELS,
                colors=ACTIVITY_COLORS,
                height_px=10,
            ),
        ]
        show_outcome_audit = str(p.get("show_outcome_audit", "yes")).lower() == "yes"
        if show_outcome_audit:
            audit_labels = {0: "未来3h发生转换", 1: "未来3h保持"}
            audit_colors = {0: "#ef4444", 1: "#22c55e"}
            audit_values = pd.to_numeric(aligned["activity_actual_persist_h3"], errors="coerce")
            bands.append(
                StateBand(
                    band_id="ai_activity_outcome_audit",
                    label="历史结果审计（未来3h）",
                    codes=[None if pd.isna(value) else int(value) for value in audit_values],
                    categories=[
                        StateBandCategory(
                            code=code,
                            label=audit_labels[code],
                            color=audit_colors[code],
                            opacity=0.84,
                            status="仅历史结果审计",
                            fields={"uses_future_outcome_for_audit_only": True, "not_model_input": True},
                        )
                        for code in (0, 1)
                    ],
                    description="使用未来3小时真实状态核对模型预测；该色带包含未来结果，仅供历史审计",
                    render_mode="strip",
                    position="bottom",
                    height_px=8,
                )
            )

        overview_tracks = [
            IndicatorTrack(
                "activity_score",
                "活跃度分数",
                _number_list(aligned["activity_score"]),
                "#fb923c",
                -1.0,
                1.0,
                0.0,
                "多周期波动压缩/扩张状态分数",
            ),
            IndicatorTrack(
                "activity_persist_h3_probability",
                "活跃状态持续3h概率",
                _number_list(aligned["activity_persist_h3_probability"]),
                "#22c55e",
                0.0,
                1.0,
                0.5,
                "仅2024/2025完整OOS预测；越高表示当前活跃度状态更可能连续维持3小时",
            ),
            IndicatorTrack(
                "activity_transition_risk_h3",
                "活跃状态3h转换风险",
                _number_list(aligned["activity_transition_risk_h3"]),
                "#ef4444",
                0.0,
                1.0,
                0.5,
                "1 - 活跃状态持续概率；是风险上下文，不是平仓或反向信号",
            ),
        ]
        research_tracks = [
            IndicatorTrack("strategic_score", "战略分数", _number_list(aligned["strategic_score"]), "#22c55e", -1.0, 1.0, 0.0, "1D/4H长期方向强度"),
            IndicatorTrack("tactical_score", "战术分数", _number_list(aligned["tactical_score"]), "#38bdf8", -1.0, 1.0, 0.0, "4H/1H推动、回调与恢复"),
            IndicatorTrack("entry_score", "入场层分数", _number_list(aligned["entry_score"]), "#facc15", -1.0, 1.0, 0.0, "30m至1m局部方向"),
            *overview_tracks,
            IndicatorTrack("all_direction_alignment", "三层方向一致性", _number_list(aligned["all_direction_alignment"]), "#a78bfa", -1.0, 1.0, 0.0, "战略、战术、入场三层方向一致程度"),
        ]
        tracks = overview_tracks if view_mode == "overview" else research_tracks

        markers: list[Marker] = []
        layers = []
        if marker_mode == "activity_strategic":
            layers = [
                ("strategic_state", "战略", DIRECTION_LABELS, DIRECTION_COLORS),
                ("activity_state", "活跃", ACTIVITY_LABELS, ACTIVITY_COLORS),
            ]
        elif marker_mode == "all":
            layers = [
                ("strategic_state", "战略", DIRECTION_LABELS, DIRECTION_COLORS),
                ("tactical_state", "战术", TACTICAL_LABELS, DIRECTION_COLORS),
                ("entry_state", "入场", ENTRY_LABELS, DIRECTION_COLORS),
                ("activity_state", "活跃", ACTIVITY_LABELS, ACTIVITY_COLORS),
            ]
        for column, layer_label, labels, colors in layers:
            values = pd.to_numeric(aligned[column], errors="coerce")
            previous = values.shift(1)
            changed = values.notna() & previous.notna() & values.ne(previous)
            for timestamp, value in values.loc[changed].items():
                code = int(value)
                markers.append(
                    Marker(
                        timestamp=pd.Timestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
                        label=f"{layer_label}→{labels[code]}",
                        color=colors[code],
                        reason="R03.3.3.1 因果迟滞状态发生切换",
                        role="node",
                        position="bottom" if code > 0 else "top",
                        symbol="circle",
                        fields={"layer": layer_label, "state_code": code, "not_trade_signal": True},
                    )
                )

        strategic_names = [_state_name(value, DIRECTION_LABELS) for value in aligned["strategic_state"]]
        tactical_names = [_state_name(value, TACTICAL_LABELS) for value in aligned["tactical_state"]]
        entry_names = [_state_name(value, ENTRY_LABELS) for value in aligned["entry_state"]]
        activity_names = [_state_name(value, ACTIVITY_LABELS) for value in aligned["activity_state"]]
        combined_process = [
            None if strategic == "无缓存" else f"{entry} · {activity}"
            for strategic, entry, activity in zip(strategic_names, entry_names, activity_names)
        ]

        probability = pd.to_numeric(aligned["activity_persist_h3_probability"], errors="coerce")
        risk = pd.to_numeric(aligned["activity_transition_risk_h3"], errors="coerce")

        core_row_fields = {
            "brief_direction": _text_field(strategic_names),
            "brief_phase": _text_field(tactical_names),
            "brief_process": _text_field(combined_process),
            "brief_process_probability": _number_list(probability),
            "brief_advice": _text_field(["用于核对市场上下文与连续性；不能单独开仓、平仓或反向"] * len(aligned)),
            "strategic_state": _categorical_field(aligned["strategic_state"], DIRECTION_LABELS),
            "tactical_state": _categorical_field(aligned["tactical_state"], TACTICAL_LABELS),
            "entry_state": _categorical_field(aligned["entry_state"], ENTRY_LABELS),
            "activity_state": _categorical_field(aligned["activity_state"], ACTIVITY_LABELS),
            "strategic_age_days": _number_list(aligned["strategic_age_bars"] / 96.0, digits=2),
            "tactical_age_hours": _number_list(aligned["tactical_age_bars"] / 4.0, digits=2),
            "entry_age_hours": _number_list(aligned["entry_age_bars"] / 4.0, digits=2),
            "activity_age_hours": _number_list(aligned["activity_age_bars"] / 4.0, digits=2),
            "activity_persist_h3_probability": _number_list(probability),
            "activity_transition_risk_h3": _number_list(risk),
            "strategic_tactical_alignment": _number_list(aligned["strategic_tactical_alignment"]),
            "tactical_entry_alignment": _number_list(aligned["tactical_entry_alignment"]),
            "all_direction_alignment": _number_list(aligned["all_direction_alignment"]),
        }
        if show_outcome_audit:
            core_row_fields["audit_actual_activity_persist_h3"] = _number_list(
                aligned["activity_actual_persist_h3"], digits=0
            )
        row_fields = core_row_fields
        if view_mode == "research":
            row_fields.update(
                {
                    "strategic_score": _number_list(aligned["strategic_score"]),
                    "tactical_score": _number_list(aligned["tactical_score"]),
                    "entry_score": _number_list(aligned["entry_score"]),
                    "activity_score": _number_list(aligned["activity_score"]),
                    "strategic_boundary_margin": _number_list(aligned["strategic_boundary_margin"]),
                    "tactical_boundary_margin": _number_list(aligned["tactical_boundary_margin"]),
                    "entry_boundary_margin": _number_list(aligned["entry_boundary_margin"]),
                    "activity_boundary_margin": _number_list(aligned["activity_boundary_margin"]),
                    "long_pullback_setup": _number_list(aligned["long_pullback_setup"]),
                    "short_pullback_setup": _number_list(aligned["short_pullback_setup"]),
                    "trend_momentum_long": _number_list(aligned["trend_momentum_long"]),
                    "trend_momentum_short": _number_list(aligned["trend_momentum_short"]),
                    "prediction_fold": _text_field(
                        [None if pd.isna(value) else str(value) for value in aligned["prediction_fold"]]
                    ),
                    "state_available_time": _text_field(
                        [
                            None if pd.isna(value) else pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")
                            for value in aligned["state_time"]
                        ]
                    ),
                }
            )


        matched = int(aligned["strategic_state"].notna().sum())
        probability_rows = int(probability.notna().sum())
        display = (
            f"R03.3.3.1状态 {matched}/{len(aligned)} 根；"
            f"完整OOS活跃持续概率 {probability_rows}/{len(aligned)} 根；"
            "状态为15分钟因果缓存，向后对齐且最多延续30分钟"
        )
        return PluginRunResult(
            markers=markers,
            tracks=tracks,
            bands=bands,
            row_fields=row_fields,
            summary={
                "input_rows": int(len(aligned)),
                "matched": matched,
                "state_rows": matched,
                "prediction_rows": probability_rows,
                "marker_count": int(len(markers)),
                "state_cache": str(self.config.cache_path),
                "prediction_artifact": str(self.artifact_dir),
                "prediction_coverage": "2024-2025 full OOS only",
                "sealed_2026": True,
                "contains_future_outcome_audit": bool(show_outcome_audit),
                "future_outcome_audit_usage": "historical visual validation only; never a feature or live state",
                "causal_alignment": "last state decision_time <= displayed candle timestamp; tolerance 30 minutes",
                "not_trade_signal": True,
                "payload_mode": "compact-overview" if view_mode == "overview" else "research",
                                "ui": {
                    "view_mode": view_mode,
                    "compact": view_mode == "overview",
                    "brief_available": True,
                    "brief_labels": ["战略", "战术", "入场/活跃"],
                    "brief_disclaimer": "AI市场状态仅用于辅助后续方向、入场和持仓模型，不是独立交易信号。",
                    "advanced_collapsed": view_mode == "overview",
                },
                "display": display,
            },
        )
