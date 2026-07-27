#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Role-aware conditional information map for causal market states.

The market-state map is not a standalone trading strategy.  This module asks
whether each state axis separates the future variable it is intended to
describe:

* direction/flow/impact/location -> signed path bias and continuation/reversal;
* volatility/activity -> future path width and excursion;
* maturity/quality -> strengthening or weakening of the already-confirmed path;
* predefined state ladders -> incremental information added by each new layer.

Conditions are constructed only from columns available on the current closed
bar.  Future values are labels only and are never used to build a condition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


_EPS = 1e-12


@dataclass(frozen=True)
class ConditionalMapConfig:
    """Configuration for Market State Conditional Map V2."""

    horizons_bars: tuple[int, ...] = (5, 15, 30, 60, 180)
    sample_stride_bars: int = 5
    sequence_lookback_bars: int = 15
    minimum_samples: int = 500
    minimum_holdout_samples: int = 100
    minimum_years: int = 3
    holdout_start: str | None = "2025-07-01"

    # Practical information thresholds.  These are not profitability gates.
    min_direction_uplift: float = 0.00002
    min_direction_win_rate_uplift: float = 0.005
    min_range_relative_uplift: float = 0.03
    min_effect_size: float = 0.01
    minimum_positive_year_ratio: float = 0.60
    minimum_supported_profiles: int = 2
    minimum_supported_horizons: int = 2

    def validate(self) -> None:
        if not self.horizons_bars or any(int(v) < 1 for v in self.horizons_bars):
            raise ValueError("horizons_bars must contain positive integers")
        if self.sample_stride_bars < 1:
            raise ValueError("sample_stride_bars must be >= 1")
        if self.sequence_lookback_bars < 1:
            raise ValueError("sequence_lookback_bars must be >= 1")
        if self.minimum_samples < 1 or self.minimum_holdout_samples < 1:
            raise ValueError("minimum sample counts must be >= 1")
        if self.minimum_years < 1:
            raise ValueError("minimum_years must be >= 1")
        if self.min_direction_uplift < 0.0 or self.min_direction_win_rate_uplift < 0.0:
            raise ValueError("direction thresholds must be non-negative")
        if self.min_range_relative_uplift < 0.0 or self.min_effect_size < 0.0:
            raise ValueError("range/effect thresholds must be non-negative")
        if not 0.0 <= self.minimum_positive_year_ratio <= 1.0:
            raise ValueError("minimum_positive_year_ratio must be in [0, 1]")
        if self.minimum_supported_profiles < 1 or self.minimum_supported_horizons < 1:
            raise ValueError("support thresholds must be >= 1")


@dataclass(frozen=True)
class ConditionDefinition:
    """One causal state condition and the role-specific target it should affect."""

    condition_name: str
    axis: str
    description: str
    intended_role: str
    target_kind: str  # directional_return | future_range
    direction: int  # +1 long, -1 short, 0 non-directional
    expected_sign: int  # +1 higher than baseline, -1 lower than baseline
    baseline_columns: tuple[str, ...]
    mask: pd.Series
    parent_condition: str | None = None
    ladder_name: str | None = None
    ladder_stage: int | None = None

    def catalog_row(self) -> dict[str, object]:
        return {
            "condition_name": self.condition_name,
            "axis": self.axis,
            "description": self.description,
            "intended_role": self.intended_role,
            "target_kind": self.target_kind,
            "direction": self.direction,
            "expected_sign": self.expected_sign,
            "baseline_columns": "|".join(self.baseline_columns),
            "parent_condition": self.parent_condition or "",
            "ladder_name": self.ladder_name or "",
            "ladder_stage": self.ladder_stage if self.ladder_stage is not None else "",
            "eligible_rows": int(self.mask.fillna(False).sum()),
        }


@dataclass(frozen=True)
class ConditionEvaluation:
    summary: pd.DataFrame
    yearly: pd.DataFrame
    periods: pd.DataFrame


def _as_bool(series: pd.Series) -> pd.Series:
    return series.fillna(False).astype(bool)


def _state_start(mask: pd.Series) -> pd.Series:
    current = _as_bool(mask)
    return current & ~current.shift(1, fill_value=False)


def _recently(mask: pd.Series, lookback: int, *, include_current: bool = True) -> pd.Series:
    values = _as_bool(mask)
    if not include_current:
        values = values.shift(1, fill_value=False)
    return values.rolling(int(lookback), min_periods=1).max().fillna(0.0).astype(bool)


def _sample_mask(index: pd.Index, stride: int) -> pd.Series:
    values = np.zeros(len(index), dtype=bool)
    values[:: max(1, int(stride))] = True
    return pd.Series(values, index=index, dtype=bool)


def _activity_state(frame: pd.DataFrame) -> pd.Series:
    z = pd.to_numeric(frame.get("activity_z", np.nan), errors="coerce")
    out = pd.Series("normal", index=frame.index, dtype="object")
    out.loc[z <= -0.80] = "low"
    out.loc[z >= 0.80] = "high"
    out.loc[z >= 2.00] = "extreme"
    out.loc[z.isna()] = "warmup"
    return out


def _age_bucket(frame: pd.DataFrame, *, medium_window: int, slow_window: int) -> pd.Series:
    age = pd.to_numeric(frame.get("trend_state_age", 0), errors="coerce").fillna(0).astype(int)
    early_max = max(12, int(medium_window // 2))
    mid_max = max(early_max + 1, int(slow_window // 2))
    out = pd.Series("late", index=frame.index, dtype="object")
    out.loc[age <= mid_max] = "mid"
    out.loc[age <= early_max] = "early"
    out.loc[age <= 0] = "warmup"
    return out


def _flow_recovery_long(frame: pd.DataFrame) -> pd.Series:
    flow_state = frame["flow_state"].astype(str)
    flow_score = pd.to_numeric(frame.get("flow_score", np.nan), errors="coerce")
    return flow_state.isin({"buy_pressure", "buy_building", "buy_persistent"}) | (
        flow_score.gt(0.0) & flow_score.shift(1).le(0.0)
    )


def _flow_recovery_short(frame: pd.DataFrame) -> pd.Series:
    flow_state = frame["flow_state"].astype(str)
    flow_score = pd.to_numeric(frame.get("flow_score", np.nan), errors="coerce")
    return flow_state.isin({"sell_pressure", "sell_building", "sell_persistent"}) | (
        flow_score.lt(0.0) & flow_score.shift(1).ge(0.0)
    )


def build_condition_definitions(
    frame: pd.DataFrame,
    config: ConditionalMapConfig,
    *,
    medium_trend_window: int = 64,
    slow_trend_window: int = 240,
) -> list[ConditionDefinition]:
    """Build fixed, causal state-axis and nested ladder definitions.

    No definition references a future-return/path column.  The returned masks
    are append-invariant as long as the underlying state frame is causal.
    """

    config.validate()
    required = {
        "data_ready",
        "trend_state",
        "trend_phase",
        "trend_quality_state",
        "volatility_state",
        "flow_state",
        "impact_state",
        "location_state",
        "trade_context_state",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"market-state frame missing condition columns: {missing}")

    ready = _as_bool(frame["data_ready"])
    flow_ready = _as_bool(frame.get("orderflow_available", pd.Series(False, index=frame.index)))
    location_ready = _as_bool(frame.get("location_available", pd.Series(False, index=frame.index)))
    sampled = _sample_mask(frame.index, config.sample_stride_bars)
    base = ready & sampled

    trend = frame["trend_state"].astype(str)
    phase = frame["trend_phase"].astype(str)
    quality = frame["trend_quality_state"].astype(str)
    volatility = frame["volatility_state"].astype(str)
    flow = frame["flow_state"].astype(str)
    impact = frame["impact_state"].astype(str)
    location = frame["location_state"].astype(str)
    context = frame["trade_context_state"].astype(str)
    activity = _activity_state(frame)
    age_bucket = _age_bucket(frame, medium_window=medium_trend_window, slow_window=slow_trend_window)

    year_vol = ("signal_year", "volatility_state")
    year_vol_trend = ("signal_year", "volatility_state", "trend_state")
    year_only = ("signal_year",)

    definitions: list[ConditionDefinition] = []

    def add(
        name: str,
        axis: str,
        description: str,
        role: str,
        target: str,
        direction: int,
        expected_sign: int,
        baseline_columns: tuple[str, ...],
        mask: pd.Series,
        *,
        parent: str | None = None,
        ladder: str | None = None,
        stage: int | None = None,
    ) -> None:
        definitions.append(
            ConditionDefinition(
                condition_name=name,
                axis=axis,
                description=description,
                intended_role=role,
                target_kind=target,
                direction=int(direction),
                expected_sign=1 if int(expected_sign) >= 0 else -1,
                baseline_columns=baseline_columns,
                mask=_as_bool(mask),
                parent_condition=parent,
                ladder_name=ladder,
                ladder_stage=stage,
            )
        )

    # Confirmed direction is historical structure, not a direct permission.
    add("trend_up_structure", "trend_direction", "已确认上涨结构", "direction_context", "directional_return", 1, 1, year_vol, base & trend.eq("up"))
    add("trend_down_structure", "trend_direction", "已确认下跌结构", "direction_context", "directional_return", -1, 1, year_vol, base & trend.eq("down"))

    # Trend phase and age answer whether a confirmed path is strengthening or weakening.
    for direction_state, direction in (("up", 1), ("down", -1)):
        for phase_name in ("startup", "continuation"):
            add(
                f"{direction_state}_{phase_name}",
                "trend_phase",
                f"{direction_state} structure in {phase_name} phase",
                "continuation_strength",
                "directional_return",
                direction,
                1,
                year_vol_trend,
                base & trend.eq(direction_state) & phase.eq(phase_name),
            )
        for phase_name in ("mature", "decay"):
            add(
                f"{direction_state}_{phase_name}",
                "trend_phase",
                f"{direction_state} structure in {phase_name} phase",
                "continuation_weakening",
                "directional_return",
                direction,
                -1,
                year_vol_trend,
                base & trend.eq(direction_state) & phase.eq(phase_name),
            )
        add(
            f"{direction_state}_age_early",
            "trend_age",
            f"{direction_state} structure early age bucket",
            "continuation_strength",
            "directional_return",
            direction,
            1,
            year_vol_trend,
            base & trend.eq(direction_state) & age_bucket.eq("early"),
        )
        add(
            f"{direction_state}_age_late",
            "trend_age",
            f"{direction_state} structure late age bucket",
            "continuation_weakening",
            "directional_return",
            direction,
            -1,
            year_vol_trend,
            base & trend.eq(direction_state) & age_bucket.eq("late"),
        )
        add(
            f"{direction_state}_high_order",
            "trend_quality",
            f"{direction_state} structure with high relative orderliness",
            "path_quality",
            "directional_return",
            direction,
            1,
            year_vol_trend,
            base & trend.eq(direction_state) & quality.eq("high_order"),
        )
        add(
            f"{direction_state}_noisy",
            "trend_quality",
            f"{direction_state} structure with low relative orderliness",
            "path_quality",
            "directional_return",
            direction,
            -1,
            year_vol_trend,
            base & trend.eq(direction_state) & quality.eq("noisy"),
        )

    # Volatility and activity have a path-width role, not a direction role.
    for state_name, expected_sign, role in (
        ("quiet", -1, "future_range_low"),
        ("compression", -1, "future_range_low"),
        ("expansion", 1, "future_range_high"),
        ("shock", 1, "future_range_high"),
    ):
        add(
            f"volatility_{state_name}",
            "volatility",
            f"volatility state={state_name}",
            role,
            "future_range",
            0,
            expected_sign,
            year_only,
            base & volatility.eq(state_name),
        )
    for state_name, expected_sign, role in (
        ("low", -1, "future_range_low"),
        ("high", 1, "future_range_high"),
        ("extreme", 1, "future_range_high"),
    ):
        add(
            f"activity_{state_name}",
            "activity",
            f"activity state={state_name}",
            role,
            "future_range",
            0,
            expected_sign,
            year_only,
            base & activity.eq(state_name),
        )

    # Order flow and impact are matched inside the current trend/volatility context.
    for state_name, direction in (
        ("buy_building", 1),
        ("buy_persistent", 1),
        ("buy_pressure", 1),
        ("sell_building", -1),
        ("sell_persistent", -1),
        ("sell_pressure", -1),
    ):
        add(
            f"flow_{state_name}",
            "orderflow",
            f"active order flow={state_name}",
            "directional_pressure",
            "directional_return",
            direction,
            1,
            year_vol_trend,
            base & flow_ready & flow.eq(state_name),
        )

    for state_name, direction, role in (
        ("buy_effective", 1, "impact_continuation"),
        ("sell_effective", -1, "impact_continuation"),
        ("sell_absorbed", 1, "reversal_risk"),
        ("buy_absorbed", -1, "reversal_risk"),
    ):
        add(
            f"impact_{state_name}",
            "impact_absorption",
            f"price response={state_name}",
            role,
            "directional_return",
            direction,
            1,
            year_vol_trend,
            base & flow_ready & impact.eq(state_name),
        )

    for state_name, direction, role in (
        ("downside_sweep_reclaim", 1, "reversal_location"),
        ("upside_sweep_reject", -1, "reversal_location"),
        ("breakout_accept", 1, "continuation_location"),
        ("breakdown_accept", -1, "continuation_location"),
        ("near_support", 1, "support_context"),
        ("near_resistance", -1, "resistance_context"),
    ):
        add(
            f"location_{state_name}",
            "location",
            f"structural location={state_name}",
            role,
            "directional_return",
            direction,
            1,
            year_vol_trend,
            base & location_ready & location.eq(state_name),
        )

    for state_name, direction, role in (
        ("long_reversal_watch", 1, "multi_axis_reversal_context"),
        ("short_reversal_watch", -1, "multi_axis_reversal_context"),
        ("long_continuation_watch", 1, "multi_axis_continuation_context"),
        ("short_continuation_watch", -1, "multi_axis_continuation_context"),
    ):
        add(
            f"context_{state_name}",
            "existing_context",
            f"existing V0.2 context={state_name}",
            role,
            "directional_return",
            direction,
            1,
            year_vol,
            base & context.eq(state_name),
        )

    # Causal transition states.
    sell_effective_recent = _recently(impact.eq("sell_effective"), config.sequence_lookback_bars, include_current=False)
    buy_effective_recent = _recently(impact.eq("buy_effective"), config.sequence_lookback_bars, include_current=False)
    sell_absorbed_recent = _recently(impact.eq("sell_absorbed"), config.sequence_lookback_bars)
    buy_absorbed_recent = _recently(impact.eq("buy_absorbed"), config.sequence_lookback_bars)
    compression_recent = _recently(volatility.isin({"quiet", "compression"}), max(30, config.sequence_lookback_bars))
    long_recovery = _flow_recovery_long(frame)
    short_recovery = _flow_recovery_short(frame)

    add(
        "transition_sell_effective_to_absorbed",
        "transition",
        "recent effective selling changes to sell absorption",
        "reversal_transition",
        "directional_return",
        1,
        1,
        year_vol_trend,
        base & _state_start(impact.eq("sell_absorbed") & sell_effective_recent),
    )
    add(
        "transition_buy_effective_to_absorbed",
        "transition",
        "recent effective buying changes to buy absorption",
        "reversal_transition",
        "directional_return",
        -1,
        1,
        year_vol_trend,
        base & _state_start(impact.eq("buy_absorbed") & buy_effective_recent),
    )
    add(
        "transition_sell_absorbed_to_buy_recovery",
        "transition",
        "sell absorption followed by active buy-flow recovery",
        "reversal_confirmation",
        "directional_return",
        1,
        1,
        year_vol_trend,
        base & _state_start(sell_absorbed_recent & long_recovery),
    )
    add(
        "transition_buy_absorbed_to_sell_recovery",
        "transition",
        "buy absorption followed by active sell-flow recovery",
        "reversal_confirmation",
        "directional_return",
        -1,
        1,
        year_vol_trend,
        base & _state_start(buy_absorbed_recent & short_recovery),
    )
    add(
        "transition_compression_to_buy_effective",
        "transition",
        "recent compression followed by effective active buying",
        "volatility_direction_transition",
        "directional_return",
        1,
        1,
        year_vol,
        base & _state_start(compression_recent & impact.eq("buy_effective")),
    )
    add(
        "transition_compression_to_sell_effective",
        "transition",
        "recent compression followed by effective active selling",
        "volatility_direction_transition",
        "directional_return",
        -1,
        1,
        year_vol,
        base & _state_start(compression_recent & impact.eq("sell_effective")),
    )

    # Nested conditional ladders.  They are fixed before reading forward labels.
    low_location = location.isin({"lower_zone", "near_support", "downside_sweep_reclaim"})
    high_location = location.isin({"upper_zone", "near_resistance", "upside_sweep_reject"})
    long_flow = flow.isin({"buy_pressure", "buy_building", "buy_persistent"})
    short_flow = flow.isin({"sell_pressure", "sell_building", "sell_persistent"})

    lr1 = base & sell_absorbed_recent
    lr2 = lr1 & low_location
    lr3 = lr2 & location.eq("downside_sweep_reclaim")
    lr4 = lr3 & long_flow
    add("ladder_long_reversal_1_absorption", "conditional_ladder", "recent sell absorption", "nested_reversal_context", "directional_return", 1, 1, year_vol_trend, lr1, ladder="long_reversal", stage=1)
    add("ladder_long_reversal_2_low_location", "conditional_ladder", "sell absorption plus low structural location", "nested_reversal_context", "directional_return", 1, 1, year_vol_trend, lr2, parent="ladder_long_reversal_1_absorption", ladder="long_reversal", stage=2)
    add("ladder_long_reversal_3_sweep_reclaim", "conditional_ladder", "absorption at low location plus downside sweep reclaim", "nested_reversal_context", "directional_return", 1, 1, year_vol_trend, lr3, parent="ladder_long_reversal_2_low_location", ladder="long_reversal", stage=3)
    add("ladder_long_reversal_4_buy_recovery", "conditional_ladder", "long reversal ladder plus active buy-flow recovery", "nested_reversal_context", "directional_return", 1, 1, year_vol_trend, lr4, parent="ladder_long_reversal_3_sweep_reclaim", ladder="long_reversal", stage=4)

    sr1 = base & buy_absorbed_recent
    sr2 = sr1 & high_location
    sr3 = sr2 & location.eq("upside_sweep_reject")
    sr4 = sr3 & short_flow
    add("ladder_short_reversal_1_absorption", "conditional_ladder", "recent buy absorption", "nested_reversal_context", "directional_return", -1, 1, year_vol_trend, sr1, ladder="short_reversal", stage=1)
    add("ladder_short_reversal_2_high_location", "conditional_ladder", "buy absorption plus high structural location", "nested_reversal_context", "directional_return", -1, 1, year_vol_trend, sr2, parent="ladder_short_reversal_1_absorption", ladder="short_reversal", stage=2)
    add("ladder_short_reversal_3_sweep_reject", "conditional_ladder", "absorption at high location plus upside sweep reject", "nested_reversal_context", "directional_return", -1, 1, year_vol_trend, sr3, parent="ladder_short_reversal_2_high_location", ladder="short_reversal", stage=3)
    add("ladder_short_reversal_4_sell_recovery", "conditional_ladder", "short reversal ladder plus active sell-flow recovery", "nested_reversal_context", "directional_return", -1, 1, year_vol_trend, sr4, parent="ladder_short_reversal_3_sweep_reject", ladder="short_reversal", stage=4)

    lc1 = base & trend.eq("up")
    lc2 = lc1 & flow.isin({"buy_building", "buy_persistent"})
    lc3 = lc2 & impact.eq("buy_effective")
    lc4 = lc3 & location.eq("breakout_accept")
    add("ladder_long_continuation_1_up_structure", "conditional_ladder", "confirmed up structure", "nested_continuation_context", "directional_return", 1, 1, year_vol, lc1, ladder="long_continuation", stage=1)
    add("ladder_long_continuation_2_buy_flow", "conditional_ladder", "up structure plus persistent/building buy flow", "nested_continuation_context", "directional_return", 1, 1, year_vol, lc2, parent="ladder_long_continuation_1_up_structure", ladder="long_continuation", stage=2)
    add("ladder_long_continuation_3_buy_effective", "conditional_ladder", "long continuation ladder plus effective price response", "nested_continuation_context", "directional_return", 1, 1, year_vol, lc3, parent="ladder_long_continuation_2_buy_flow", ladder="long_continuation", stage=3)
    add("ladder_long_continuation_4_breakout_accept", "conditional_ladder", "long continuation ladder plus breakout acceptance", "nested_continuation_context", "directional_return", 1, 1, year_vol, lc4, parent="ladder_long_continuation_3_buy_effective", ladder="long_continuation", stage=4)

    sc1 = base & trend.eq("down")
    sc2 = sc1 & flow.isin({"sell_building", "sell_persistent"})
    sc3 = sc2 & impact.eq("sell_effective")
    sc4 = sc3 & location.eq("breakdown_accept")
    add("ladder_short_continuation_1_down_structure", "conditional_ladder", "confirmed down structure", "nested_continuation_context", "directional_return", -1, 1, year_vol, sc1, ladder="short_continuation", stage=1)
    add("ladder_short_continuation_2_sell_flow", "conditional_ladder", "down structure plus persistent/building sell flow", "nested_continuation_context", "directional_return", -1, 1, year_vol, sc2, parent="ladder_short_continuation_1_down_structure", ladder="short_continuation", stage=2)
    add("ladder_short_continuation_3_sell_effective", "conditional_ladder", "short continuation ladder plus effective price response", "nested_continuation_context", "directional_return", -1, 1, year_vol, sc3, parent="ladder_short_continuation_2_sell_flow", ladder="short_continuation", stage=3)
    add("ladder_short_continuation_4_breakdown_accept", "conditional_ladder", "short continuation ladder plus breakdown acceptance", "nested_continuation_context", "directional_return", -1, 1, year_vol, sc4, parent="ladder_short_continuation_3_sell_effective", ladder="short_continuation", stage=4)

    cb_long1 = base & compression_recent
    cb_long2 = cb_long1 & flow.isin({"buy_building", "buy_persistent"})
    cb_long3 = cb_long2 & impact.eq("buy_effective")
    cb_long4 = cb_long3 & location.eq("breakout_accept")
    add("ladder_compression_long_1_recent_compression", "conditional_ladder", "recent quiet/compression", "nested_breakout_context", "directional_return", 1, 1, year_vol, cb_long1, ladder="compression_long", stage=1)
    add("ladder_compression_long_2_buy_flow", "conditional_ladder", "recent compression plus buy-flow build", "nested_breakout_context", "directional_return", 1, 1, year_vol, cb_long2, parent="ladder_compression_long_1_recent_compression", ladder="compression_long", stage=2)
    add("ladder_compression_long_3_buy_effective", "conditional_ladder", "compression breakout ladder plus effective buying", "nested_breakout_context", "directional_return", 1, 1, year_vol, cb_long3, parent="ladder_compression_long_2_buy_flow", ladder="compression_long", stage=3)
    add("ladder_compression_long_4_breakout_accept", "conditional_ladder", "compression breakout ladder plus breakout acceptance", "nested_breakout_context", "directional_return", 1, 1, year_vol, cb_long4, parent="ladder_compression_long_3_buy_effective", ladder="compression_long", stage=4)

    cb_short1 = base & compression_recent
    cb_short2 = cb_short1 & flow.isin({"sell_building", "sell_persistent"})
    cb_short3 = cb_short2 & impact.eq("sell_effective")
    cb_short4 = cb_short3 & location.eq("breakdown_accept")
    add("ladder_compression_short_1_recent_compression", "conditional_ladder", "recent quiet/compression", "nested_breakout_context", "directional_return", -1, 1, year_vol, cb_short1, ladder="compression_short", stage=1)
    add("ladder_compression_short_2_sell_flow", "conditional_ladder", "recent compression plus sell-flow build", "nested_breakout_context", "directional_return", -1, 1, year_vol, cb_short2, parent="ladder_compression_short_1_recent_compression", ladder="compression_short", stage=2)
    add("ladder_compression_short_3_sell_effective", "conditional_ladder", "compression breakdown ladder plus effective selling", "nested_breakout_context", "directional_return", -1, 1, year_vol, cb_short3, parent="ladder_compression_short_2_sell_flow", ladder="compression_short", stage=3)
    add("ladder_compression_short_4_breakdown_accept", "conditional_ladder", "compression breakdown ladder plus breakdown acceptance", "nested_breakout_context", "directional_return", -1, 1, year_vol, cb_short4, parent="ladder_compression_short_3_sell_effective", ladder="compression_short", stage=4)

    return definitions


def attach_conditional_targets(path_frame: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    """Attach role-specific labels derived from the existing forward path."""

    out = path_frame.copy()
    for horizon in sorted(set(int(v) for v in horizons)):
        required = {
            f"long_return_h{horizon}",
            f"long_mfe_h{horizon}",
            f"long_mae_h{horizon}",
            f"short_return_h{horizon}",
            f"short_mfe_h{horizon}",
            f"short_mae_h{horizon}",
        }
        missing = sorted(required.difference(out.columns))
        if missing:
            raise ValueError(f"path frame missing horizon {horizon} columns: {missing}")
        long_return = pd.to_numeric(out[f"long_return_h{horizon}"], errors="coerce")
        long_mfe = pd.to_numeric(out[f"long_mfe_h{horizon}"], errors="coerce")
        long_mae = pd.to_numeric(out[f"long_mae_h{horizon}"], errors="coerce")
        out[f"future_range_h{horizon}"] = long_mfe - long_mae
        out[f"future_abs_return_h{horizon}"] = long_return.abs()
        out[f"future_max_excursion_h{horizon}"] = pd.concat([long_mfe, -long_mae], axis=1).max(axis=1)
    return out


def _target_columns(definition: ConditionDefinition, horizon: int) -> dict[str, str]:
    if definition.target_kind == "future_range":
        return {
            "target": f"future_range_h{horizon}",
            "return": f"long_return_h{horizon}",
            "mfe": f"long_mfe_h{horizon}",
            "mae": f"long_mae_h{horizon}",
            "trap": "",
        }
    if definition.target_kind != "directional_return" or definition.direction not in {-1, 1}:
        raise ValueError(f"invalid condition target: {definition}")
    side = "long" if definition.direction > 0 else "short"
    return {
        "target": f"{side}_return_h{horizon}",
        "return": f"{side}_return_h{horizon}",
        "mfe": f"{side}_mfe_h{horizon}",
        "mae": f"{side}_mae_h{horizon}",
        "trap": f"{side}_trap_h{horizon}",
    }


def _baseline_table(
    frame: pd.DataFrame,
    *,
    target_col: str,
    return_col: str,
    baseline_columns: Sequence[str],
) -> pd.DataFrame:
    columns = list(dict.fromkeys([*baseline_columns, target_col, return_col]))
    base = frame.loc[frame[target_col].notna(), columns].copy()
    if base.empty:
        return pd.DataFrame()
    grouped = base.groupby(list(baseline_columns), dropna=False)
    table = grouped.agg(
        baseline_mean=(target_col, "mean"),
        baseline_std=(target_col, "std"),
        baseline_win_rate=(return_col, lambda s: float((pd.to_numeric(s, errors="coerce") > 0.0).mean())),
        baseline_rows=(target_col, "size"),
    ).reset_index()
    return table


def _period_label(available_time: pd.Series, holdout_start: str | None) -> pd.Series:
    if not holdout_start:
        return pd.Series("all", index=available_time.index, dtype="object")
    cutoff = pd.Timestamp(holdout_start)
    return pd.Series(
        np.where(pd.to_datetime(available_time) >= cutoff, "holdout", "pre_holdout"),
        index=available_time.index,
        dtype="object",
    )


def _evaluate_one_condition(
    frame: pd.DataFrame,
    definition: ConditionDefinition,
    config: ConditionalMapConfig,
    *,
    profile: str,
    baseline_cache: dict[tuple[str, str, tuple[str, ...]], pd.DataFrame],
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    positions = np.flatnonzero(definition.mask.to_numpy(dtype=bool))
    if positions.size == 0:
        return [], [], []

    summaries: list[dict[str, object]] = []
    yearly_rows: list[dict[str, object]] = []
    period_rows: list[dict[str, object]] = []

    for horizon in config.horizons_bars:
        cols = _target_columns(definition, int(horizon))
        target_col = cols["target"]
        if target_col not in frame:
            continue
        selected_columns = list(
            dict.fromkeys(
                [
                    *definition.baseline_columns,
                    "available_time",
                    "signal_year",
                    target_col,
                    cols["return"],
                    cols["mfe"],
                    cols["mae"],
                    *([cols["trap"]] if cols["trap"] and cols["trap"] in frame else []),
                ]
            )
        )
        selected = frame.iloc[positions][selected_columns].copy()
        selected = selected.loc[selected[target_col].notna()].copy()
        if selected.empty:
            continue
        cache_key = (target_col, cols["return"], tuple(definition.baseline_columns))
        baseline = baseline_cache.get(cache_key)
        if baseline is None:
            baseline = _baseline_table(
                frame,
                target_col=target_col,
                return_col=cols["return"],
                baseline_columns=definition.baseline_columns,
            )
            baseline_cache[cache_key] = baseline
        if baseline.empty:
            continue
        selected = selected.reset_index(names="signal_time").merge(
            baseline,
            on=list(definition.baseline_columns),
            how="left",
        )
        selected["period"] = _period_label(selected["available_time"], config.holdout_start).to_numpy()
        selected["target_value"] = pd.to_numeric(selected[target_col], errors="coerce")
        selected["return_value"] = pd.to_numeric(selected[cols["return"]], errors="coerce")
        selected["mfe_value"] = pd.to_numeric(selected[cols["mfe"]], errors="coerce")
        selected["mae_value"] = pd.to_numeric(selected[cols["mae"]], errors="coerce")
        selected["raw_uplift"] = selected["target_value"] - selected["baseline_mean"]
        selected["primary_uplift"] = float(definition.expected_sign) * selected["raw_uplift"]
        selected["primary_effect"] = selected["primary_uplift"] / selected["baseline_std"].replace(0.0, np.nan)
        selected["win_value"] = selected["return_value"] > 0.0
        selected["primary_win_uplift"] = float(definition.expected_sign) * (
            selected["win_value"].astype(float) - selected["baseline_win_rate"]
        )
        selected["path_advantage"] = selected["mfe_value"] + selected["mae_value"]
        if cols["trap"] and cols["trap"] in selected:
            selected["trap_value"] = selected[cols["trap"]].fillna(False).astype(bool)
        else:
            selected["trap_value"] = False

        mean_target = float(selected["target_value"].mean())
        mean_baseline = float(selected["baseline_mean"].mean())
        relative = (mean_target / mean_baseline - 1.0) if abs(mean_baseline) > _EPS else float("nan")
        primary_relative = float(definition.expected_sign) * relative if np.isfinite(relative) else float("nan")
        summaries.append(
            {
                "profile": profile,
                "condition_name": definition.condition_name,
                "axis": definition.axis,
                "description": definition.description,
                "intended_role": definition.intended_role,
                "target_kind": definition.target_kind,
                "direction": definition.direction,
                "expected_sign": definition.expected_sign,
                "horizon_bars": int(horizon),
                "samples": int(len(selected)),
                "unique_days": int(pd.to_datetime(selected["available_time"]).dt.normalize().nunique()),
                "condition_mean": mean_target,
                "baseline_mean": mean_baseline,
                "raw_uplift": float(selected["raw_uplift"].mean()),
                "primary_uplift": float(selected["primary_uplift"].mean()),
                "primary_relative_uplift": primary_relative,
                "effect_size": float(selected["primary_effect"].replace([np.inf, -np.inf], np.nan).mean()),
                "condition_win_rate": float(selected["win_value"].mean()),
                "baseline_win_rate": float(selected["baseline_win_rate"].mean()),
                "primary_win_rate_uplift": float(selected["primary_win_uplift"].mean()),
                "mean_mfe": float(selected["mfe_value"].mean()),
                "mean_mae": float(selected["mae_value"].mean()),
                "mean_path_advantage": float(selected["path_advantage"].mean()),
                "trap_rate": float(selected["trap_value"].mean()),
                "parent_condition": definition.parent_condition or "",
                "ladder_name": definition.ladder_name or "",
                "ladder_stage": definition.ladder_stage if definition.ladder_stage is not None else np.nan,
            }
        )

        for year, group in selected.groupby("signal_year", dropna=False):
            year_target = float(group["target_value"].mean())
            year_base = float(group["baseline_mean"].mean())
            year_rel = (year_target / year_base - 1.0) if abs(year_base) > _EPS else float("nan")
            yearly_rows.append(
                {
                    "profile": profile,
                    "condition_name": definition.condition_name,
                    "axis": definition.axis,
                    "horizon_bars": int(horizon),
                    "signal_year": int(year),
                    "samples": int(len(group)),
                    "condition_mean": year_target,
                    "baseline_mean": year_base,
                    "primary_uplift": float(group["primary_uplift"].mean()),
                    "primary_relative_uplift": float(definition.expected_sign) * year_rel if np.isfinite(year_rel) else float("nan"),
                    "effect_size": float(group["primary_effect"].replace([np.inf, -np.inf], np.nan).mean()),
                    "primary_win_rate_uplift": float(group["primary_win_uplift"].mean()),
                }
            )
        for period_name, group in selected.groupby("period", dropna=False):
            period_target = float(group["target_value"].mean())
            period_base = float(group["baseline_mean"].mean())
            period_rel = (period_target / period_base - 1.0) if abs(period_base) > _EPS else float("nan")
            period_rows.append(
                {
                    "profile": profile,
                    "condition_name": definition.condition_name,
                    "axis": definition.axis,
                    "horizon_bars": int(horizon),
                    "period": str(period_name),
                    "samples": int(len(group)),
                    "condition_mean": period_target,
                    "baseline_mean": period_base,
                    "primary_uplift": float(group["primary_uplift"].mean()),
                    "primary_relative_uplift": float(definition.expected_sign) * period_rel if np.isfinite(period_rel) else float("nan"),
                    "effect_size": float(group["primary_effect"].replace([np.inf, -np.inf], np.nan).mean()),
                    "primary_win_rate_uplift": float(group["primary_win_uplift"].mean()),
                }
            )
    return summaries, yearly_rows, period_rows


def evaluate_conditions(
    path_frame: pd.DataFrame,
    definitions: Iterable[ConditionDefinition],
    config: ConditionalMapConfig,
    *,
    profile: str,
    progress_callback=None,
) -> ConditionEvaluation:
    """Evaluate all definitions against role-aware matched baselines."""

    config.validate()
    definitions_list = list(definitions)
    summaries: list[dict[str, object]] = []
    yearly: list[dict[str, object]] = []
    periods: list[dict[str, object]] = []
    baseline_cache: dict[tuple[str, str, tuple[str, ...]], pd.DataFrame] = {}
    for i, definition in enumerate(definitions_list, start=1):
        s, y, p = _evaluate_one_condition(
            path_frame,
            definition,
            config,
            profile=profile,
            baseline_cache=baseline_cache,
        )
        summaries.extend(s)
        yearly.extend(y)
        periods.extend(p)
        if progress_callback is not None:
            progress_callback(i, len(definitions_list))
    return ConditionEvaluation(
        summary=pd.DataFrame(summaries),
        yearly=pd.DataFrame(yearly),
        periods=pd.DataFrame(periods),
    )


def build_ladder_incremental_summary(
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    periods: pd.DataFrame,
) -> pd.DataFrame:
    """Compare every nested ladder stage with its direct parent stage."""

    if summary.empty:
        return pd.DataFrame()
    ladder = summary.loc[summary["ladder_name"].astype(str).ne("")].copy()
    if ladder.empty:
        return pd.DataFrame()
    parent_lookup = summary.set_index(["profile", "condition_name", "horizon_bars"])
    rows: list[dict[str, object]] = []
    for row in ladder.itertuples(index=False):
        parent_name = str(row.parent_condition or "")
        if not parent_name:
            continue
        key = (row.profile, parent_name, int(row.horizon_bars))
        if key not in parent_lookup.index:
            continue
        parent = parent_lookup.loc[key]
        if isinstance(parent, pd.DataFrame):
            parent = parent.iloc[0]
        child_years = yearly.loc[
            yearly["profile"].eq(row.profile)
            & yearly["condition_name"].eq(row.condition_name)
            & yearly["horizon_bars"].eq(row.horizon_bars)
        ]
        parent_years = yearly.loc[
            yearly["profile"].eq(row.profile)
            & yearly["condition_name"].eq(parent_name)
            & yearly["horizon_bars"].eq(row.horizon_bars)
        ][["signal_year", "primary_uplift"]].rename(columns={"primary_uplift": "parent_primary_uplift"})
        year_merge = child_years.merge(parent_years, on="signal_year", how="inner")
        year_increment = year_merge["primary_uplift"] - year_merge["parent_primary_uplift"] if not year_merge.empty else pd.Series(dtype=float)

        child_holdout = periods.loc[
            periods["profile"].eq(row.profile)
            & periods["condition_name"].eq(row.condition_name)
            & periods["horizon_bars"].eq(row.horizon_bars)
            & periods["period"].eq("holdout")
        ]
        parent_holdout = periods.loc[
            periods["profile"].eq(row.profile)
            & periods["condition_name"].eq(parent_name)
            & periods["horizon_bars"].eq(row.horizon_bars)
            & periods["period"].eq("holdout")
        ]
        holdout_increment = float("nan")
        if not child_holdout.empty and not parent_holdout.empty:
            holdout_increment = float(child_holdout.iloc[0]["primary_uplift"] - parent_holdout.iloc[0]["primary_uplift"])
        rows.append(
            {
                "profile": row.profile,
                "ladder_name": row.ladder_name,
                "ladder_stage": row.ladder_stage,
                "condition_name": row.condition_name,
                "parent_condition": parent_name,
                "horizon_bars": int(row.horizon_bars),
                "parent_samples": int(parent["samples"]),
                "child_samples": int(row.samples),
                "retention_ratio": float(row.samples / parent["samples"]) if int(parent["samples"]) > 0 else float("nan"),
                "parent_primary_uplift": float(parent["primary_uplift"]),
                "child_primary_uplift": float(row.primary_uplift),
                "incremental_primary_uplift": float(row.primary_uplift - parent["primary_uplift"]),
                "incremental_win_rate_uplift": float(row.primary_win_rate_uplift - parent["primary_win_rate_uplift"]),
                "positive_increment_year_ratio": float((year_increment > 0.0).mean()) if len(year_increment) else float("nan"),
                "holdout_incremental_uplift": holdout_increment,
            }
        )
    return pd.DataFrame(rows)


def build_information_registry(
    summary: pd.DataFrame,
    yearly: pd.DataFrame,
    periods: pd.DataFrame,
    config: ConditionalMapConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Grade condition/profile/horizon evidence and aggregate condition status."""

    if summary.empty:
        return pd.DataFrame(), pd.DataFrame()
    config.validate()
    rows = summary.copy()
    year_stats = (
        yearly.groupby(["profile", "condition_name", "horizon_bars"], dropna=False)
        .agg(
            years=("signal_year", "nunique"),
            positive_years=("primary_uplift", lambda s: int((pd.to_numeric(s, errors="coerce") > 0.0).sum())),
            mean_year_uplift=("primary_uplift", "mean"),
            worst_year_uplift=("primary_uplift", "min"),
        )
        .reset_index()
    )
    rows = rows.merge(year_stats, on=["profile", "condition_name", "horizon_bars"], how="left")
    holdout = periods.loc[periods["period"].eq("holdout")].copy()
    holdout = holdout.rename(
        columns={
            "samples": "holdout_samples",
            "primary_uplift": "holdout_primary_uplift",
            "primary_relative_uplift": "holdout_primary_relative_uplift",
            "effect_size": "holdout_effect_size",
            "primary_win_rate_uplift": "holdout_primary_win_rate_uplift",
        }
    )
    keep = [
        "profile",
        "condition_name",
        "horizon_bars",
        "holdout_samples",
        "holdout_primary_uplift",
        "holdout_primary_relative_uplift",
        "holdout_effect_size",
        "holdout_primary_win_rate_uplift",
    ]
    rows = rows.merge(holdout[keep], on=["profile", "condition_name", "horizon_bars"], how="left")
    rows["positive_year_ratio"] = rows["positive_years"] / rows["years"].replace(0, np.nan)
    rows["sample_ok"] = rows["samples"].ge(config.minimum_samples)
    rows["holdout_sample_ok"] = rows["holdout_samples"].fillna(0).ge(config.minimum_holdout_samples)
    rows["year_ok"] = rows["years"].fillna(0).ge(config.minimum_years)
    rows["year_direction_ok"] = rows["positive_year_ratio"].fillna(0.0).ge(config.minimum_positive_year_ratio)
    rows["holdout_direction_ok"] = rows["holdout_primary_uplift"].fillna(-np.inf).gt(0.0)
    rows["effect_ok"] = rows["effect_size"].fillna(-np.inf).ge(config.min_effect_size)

    directional = rows["target_kind"].eq("directional_return")
    range_target = rows["target_kind"].eq("future_range")
    rows["practical_magnitude_ok"] = False
    rows.loc[directional, "practical_magnitude_ok"] = (
        rows.loc[directional, "primary_uplift"].ge(config.min_direction_uplift)
        | rows.loc[directional, "primary_win_rate_uplift"].ge(config.min_direction_win_rate_uplift)
    )
    rows.loc[range_target, "practical_magnitude_ok"] = rows.loc[range_target, "primary_relative_uplift"].ge(
        config.min_range_relative_uplift
    )
    rows["information_flag"] = (
        rows["sample_ok"]
        & rows["holdout_sample_ok"]
        & rows["year_ok"]
        & rows["year_direction_ok"]
        & rows["holdout_direction_ok"]
        & rows["effect_ok"]
        & rows["practical_magnitude_ok"]
    )

    rows["weak_information_flag"] = (
        rows["sample_ok"]
        & rows["holdout_sample_ok"]
        & rows["year_ok"]
        & rows["primary_uplift"].gt(0.0)
        & rows["holdout_primary_uplift"].fillna(-np.inf).gt(0.0)
        & rows["positive_year_ratio"].fillna(0.0).ge(0.50)
        & (
            rows["effect_size"].fillna(-np.inf).gt(0.0)
            | rows["primary_win_rate_uplift"].fillna(-np.inf).gt(0.0)
            | rows["primary_relative_uplift"].fillna(-np.inf).gt(0.0)
        )
    )
    rows["opposite_flag"] = (
        rows["sample_ok"]
        & rows["holdout_sample_ok"]
        & rows["year_ok"]
        & rows["primary_uplift"].lt(0.0)
        & rows["holdout_primary_uplift"].fillna(np.inf).lt(0.0)
        & rows["positive_year_ratio"].fillna(1.0).le(1.0 - config.minimum_positive_year_ratio)
        & rows["effect_size"].abs().ge(config.min_effect_size)
    )

    aggregate_rows: list[dict[str, object]] = []
    for condition_name, group in rows.groupby("condition_name", dropna=False):
        supported = group.loc[group["information_flag"]]
        opposite = group.loc[group["opposite_flag"]]
        supported_profiles = int(supported["profile"].nunique())
        supported_horizons = int(supported["horizon_bars"].nunique())
        opposite_profiles = int(opposite["profile"].nunique())
        opposite_horizons = int(opposite["horizon_bars"].nunique())
        weak = group.loc[group["weak_information_flag"]]
        weak_profiles = int(weak["profile"].nunique())
        weak_horizons = int(weak["horizon_bars"].nunique())
        if supported_profiles >= config.minimum_supported_profiles and supported_horizons >= config.minimum_supported_horizons:
            status = "KEEP"
        elif opposite_profiles >= config.minimum_supported_profiles and opposite_horizons >= config.minimum_supported_horizons:
            status = "REVISE_SEMANTICS"
        elif weak_profiles >= 1 and weak_horizons >= 1:
            status = "KEEP_CONTEXT_ONLY"
        else:
            status = "DROP"
        best = group.sort_values(
            ["information_flag", "primary_uplift", "effect_size"],
            ascending=[False, False, False],
        ).iloc[0]
        aggregate_rows.append(
            {
                "condition_name": condition_name,
                "axis": best["axis"],
                "description": best["description"],
                "intended_role": best["intended_role"],
                "target_kind": best["target_kind"],
                "evidence_status": status,
                "supported_profiles": supported_profiles,
                "supported_horizons": supported_horizons,
                "opposite_profiles": opposite_profiles,
                "opposite_horizons": opposite_horizons,
                "best_profile": best["profile"],
                "best_horizon_bars": int(best["horizon_bars"]),
                "best_primary_uplift": float(best["primary_uplift"]),
                "best_effect_size": float(best["effect_size"]) if pd.notna(best["effect_size"]) else float("nan"),
                "best_holdout_uplift": float(best["holdout_primary_uplift"]) if pd.notna(best["holdout_primary_uplift"]) else float("nan"),
                "best_positive_year_ratio": float(best["positive_year_ratio"]) if pd.notna(best["positive_year_ratio"]) else float("nan"),
            }
        )
    registry = pd.DataFrame(aggregate_rows)
    if not registry.empty:
        order = pd.Categorical(
            registry["evidence_status"],
            categories=["KEEP", "KEEP_CONTEXT_ONLY", "REVISE_SEMANTICS", "DROP"],
            ordered=True,
        )
        registry = registry.assign(_order=order).sort_values(["_order", "axis", "condition_name"]).drop(columns="_order")
    return rows, registry.reset_index(drop=True)


def build_transition_matrix(
    frame: pd.DataFrame,
    *,
    profile: str,
    horizons_bars: Sequence[int] = (1, 5, 15, 30),
    axes: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Return empirical state-to-state transition probabilities."""

    axes = axes or {
        "trend": "trend_state",
        "phase": "trend_phase",
        "volatility": "volatility_state",
        "flow": "flow_state",
        "impact": "impact_state",
        "location": "location_state",
        "context": "trade_context_state",
    }
    ready = _as_bool(frame.get("data_ready", pd.Series(True, index=frame.index)))
    rows: list[pd.DataFrame] = []
    for axis_name, column in axes.items():
        if column not in frame:
            continue
        current = frame[column].astype(str)
        for horizon in horizons_bars:
            future = current.shift(-int(horizon))
            valid = ready & future.notna() & ~current.isin({"warmup", "unavailable"}) & ~future.isin({"warmup", "unavailable"})
            if not valid.any():
                continue
            table = (
                pd.DataFrame({"current_state": current[valid], "future_state": future[valid]})
                .groupby(["current_state", "future_state"], dropna=False)
                .size()
                .rename("transitions")
                .reset_index()
            )
            table["current_total"] = table.groupby("current_state")["transitions"].transform("sum")
            table["transition_probability"] = table["transitions"] / table["current_total"]
            table["profile"] = profile
            table["axis"] = axis_name
            table["horizon_bars"] = int(horizon)
            rows.append(table[["profile", "axis", "horizon_bars", "current_state", "future_state", "transitions", "current_total", "transition_probability"]])
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_state_duration_summary(
    frame: pd.DataFrame,
    *,
    profile: str,
    axes: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Summarize causal run lengths for each state axis."""

    axes = axes or {
        "trend": "trend_state",
        "phase": "trend_phase",
        "volatility": "volatility_state",
        "flow": "flow_state",
        "impact": "impact_state",
        "location": "location_state",
        "context": "trade_context_state",
    }
    rows: list[dict[str, object]] = []
    total_bars = max(1, len(frame))
    for axis_name, column in axes.items():
        if column not in frame:
            continue
        values = frame[column].astype(str)
        group_id = values.ne(values.shift(1)).cumsum()
        segments = pd.DataFrame({"state": values, "group": group_id}).groupby("group", sort=False).agg(state=("state", "first"), bars=("state", "size"))
        segments = segments.loc[~segments["state"].isin({"warmup", "unavailable"})]
        for state_name, group in segments.groupby("state", dropna=False):
            bar_count = int(group["bars"].sum())
            rows.append(
                {
                    "profile": profile,
                    "axis": axis_name,
                    "state": str(state_name),
                    "segments": int(len(group)),
                    "bars": bar_count,
                    "bar_coverage_ratio": float(bar_count / total_bars),
                    "mean_duration_bars": float(group["bars"].mean()),
                    "median_duration_bars": float(group["bars"].median()),
                    "p90_duration_bars": float(group["bars"].quantile(0.90)),
                    "max_duration_bars": int(group["bars"].max()),
                }
            )
    return pd.DataFrame(rows)
