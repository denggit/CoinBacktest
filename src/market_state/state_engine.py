#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reusable causal engine for Market State Map V0.2."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

from src.market_state.data_bundle import MarketStateDataBundle
from src.market_state.feature_engine import compute_market_state_features
from src.market_state.location_features import compute_location_features
from src.market_state.models import MarketStateConfig, MarketStateResult, MarketStateSnapshot
from src.market_state.orderflow_features import compute_orderflow_features
from src.market_state.transitions import (
    build_state_segments,
    stabilize_direction_scores,
    stabilize_volatility_scores,
)


class MarketStateEngine:
    def __init__(self, config: MarketStateConfig | None = None) -> None:
        self.config = config or MarketStateConfig()
        self.config.validate()

    def compute(self, bundle: MarketStateDataBundle) -> MarketStateResult:
        price_features = compute_market_state_features(bundle.primary, self.config)
        orderflow_features, orderflow_coverage = compute_orderflow_features(
            bundle.primary,
            price_features,
            self.config,
        )
        location_features = compute_location_features(bundle.primary, price_features, self.config)
        frame = bundle.primary.copy().join(price_features).join(orderflow_features).join(location_features)
        frame["available_time"] = bundle.available_times

        raw_trend, trend_state, trend_age, trend_candidate, trend_candidate_progress = (
            stabilize_direction_scores(
                frame["trend_score"].to_numpy(dtype=float),
                frame["data_ready"].fillna(False).to_numpy(dtype=bool),
                enter_threshold=self.config.directional_threshold,
                exit_threshold=self.config.trend_exit_threshold,
                confirm_bars=self.config.trend_confirm_bars,
                min_duration_bars=self.config.min_state_bars,
            )
        )
        raw_volatility, volatility_state, volatility_age, volatility_candidate, volatility_candidate_progress = (
            stabilize_volatility_scores(
                frame["volatility_z"].to_numpy(dtype=float),
                frame["activity_z"].to_numpy(dtype=float),
                frame["bar_return_z"].to_numpy(dtype=float),
                frame["data_ready"].fillna(False).to_numpy(dtype=bool),
                quiet_enter=self.config.volatility_quiet_z,
                quiet_exit=self.config.volatility_quiet_exit_z,
                expand_enter=self.config.volatility_expand_z,
                expand_exit=self.config.volatility_expand_exit_z,
                shock_enter=self.config.volatility_shock_z,
                shock_exit=self.config.volatility_shock_exit_z,
                confirm_bars=self.config.volatility_confirm_bars,
                min_duration_bars=self.config.volatility_min_state_bars,
            )
        )

        frame["raw_trend_state"] = raw_trend
        frame["trend_state"] = trend_state
        frame["trend_state_age"] = trend_age
        frame["trend_candidate_state"] = trend_candidate
        frame["trend_candidate_progress"] = trend_candidate_progress
        frame["raw_volatility_state"] = raw_volatility
        frame["volatility_state"] = volatility_state
        frame["volatility_state_age"] = volatility_age
        frame["volatility_candidate_state"] = volatility_candidate
        frame["volatility_candidate_progress"] = volatility_candidate_progress
        frame["trend_quality_state"] = self._classify_trend_quality(frame)
        frame["fast_pulse_state"] = self._classify_fast_pulse(frame)
        frame["trend_phase"] = self._classify_trend_phase(frame)
        frame["flow_state"] = self._classify_flow_state(frame)
        frame["impact_state"] = self._classify_impact_state(frame)
        frame["location_state"] = self._classify_location_state(frame)
        frame["trade_context_score"] = self._compute_trade_context_score(frame)
        frame["trade_context_state"] = self._classify_trade_context(frame)
        frame["primary_state"] = frame["trend_state"].where(frame["data_ready"], "warmup")
        frame["market_context_state"] = [
            "warmup" if trend == "warmup" or volatility == "warmup" else f"{trend}|{volatility}"
            for trend, volatility in zip(frame["trend_state"], frame["volatility_state"])
        ]

        segments = build_state_segments(frame)
        metadata: dict[str, Any] = {
            "version": "0.2",
            "source": bundle.source,
            "timestamp_semantics": bundle.timestamp_semantics,
            "bar_duration_seconds": (
                None if bundle.bar_duration is None else bundle.bar_duration.total_seconds()
            ),
            "rows": int(len(frame)),
            "ready_rows": int(frame["data_ready"].fillna(False).sum()),
            "orderflow_ready_rows": int(frame["orderflow_available"].fillna(False).sum()),
            "location_ready_rows": int(frame["location_available"].fillna(False).sum()),
            "orderflow_coverage": orderflow_coverage,
            "trend_state_counts": dict(Counter(frame.loc[frame["data_ready"], "trend_state"].astype(str))),
            "trend_quality_counts": dict(Counter(frame.loc[frame["data_ready"], "trend_quality_state"].astype(str))),
            "volatility_state_counts": dict(Counter(frame.loc[frame["data_ready"], "volatility_state"].astype(str))),
            "flow_state_counts": dict(Counter(frame["flow_state"].astype(str))),
            "impact_state_counts": dict(Counter(frame["impact_state"].astype(str))),
            "location_state_counts": dict(Counter(frame["location_state"].astype(str))),
            "trade_context_counts": dict(Counter(frame["trade_context_state"].astype(str))),
            **bundle.metadata,
        }
        return MarketStateResult(
            frame=frame,
            segments=segments,
            data_quality=bundle.data_quality,
            metadata=metadata,
        )

    def snapshot_at(self, result: MarketStateResult, timestamp: pd.Timestamp | str) -> MarketStateSnapshot:
        row = result.frame.loc[pd.Timestamp(timestamp)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[-1]

        def optional_float(value: Any) -> float | None:
            return None if pd.isna(value) else float(value)

        return MarketStateSnapshot(
            timestamp=pd.Timestamp(row.name),
            available_time=pd.Timestamp(row["available_time"]),
            trend_score=optional_float(row["trend_score"]),
            fast_trend_score=optional_float(row["fast_trend_score"]),
            medium_trend_score=optional_float(row["medium_trend_score"]),
            slow_trend_score=optional_float(row["slow_trend_score"]),
            trend_alignment_score=optional_float(row["trend_alignment_score"]),
            orderliness_score=optional_float(row["orderliness_score"]),
            orderliness_percentile=optional_float(row["orderliness_percentile"]),
            volatility_score=optional_float(row["volatility_score"]),
            volatility_z=optional_float(row["volatility_z"]),
            activity_score=optional_float(row["activity_score"]),
            activity_z=optional_float(row["activity_z"]),
            trend_state=str(row["trend_state"]),
            trend_quality_state=str(row["trend_quality_state"]),
            fast_pulse_state=str(row["fast_pulse_state"]),
            trend_phase=str(row["trend_phase"]),
            trend_candidate_state=str(row["trend_candidate_state"]),
            trend_candidate_progress=float(row["trend_candidate_progress"]),
            volatility_state=str(row["volatility_state"]),
            primary_state=str(row["primary_state"]),
            trend_state_age=int(row["trend_state_age"]),
            volatility_state_age=int(row["volatility_state_age"]),
            data_ready=bool(row["data_ready"]),
            orderflow_available=bool(row["orderflow_available"]),
            flow_score=optional_float(row["flow_score"]),
            flow_persistence=optional_float(row["flow_persistence"]),
            flow_acceleration=optional_float(row["flow_acceleration"]),
            flow_state=str(row["flow_state"]),
            flow_price_effectiveness=optional_float(row["flow_price_effectiveness"]),
            sell_absorption_score=optional_float(row["sell_absorption_score"]),
            buy_absorption_score=optional_float(row["buy_absorption_score"]),
            impact_state=str(row["impact_state"]),
            structural_location_score=optional_float(row["structural_location_score"]),
            location_state=str(row["location_state"]),
            trade_context_state=str(row["trade_context_state"]),
            trade_context_score=optional_float(row["trade_context_score"]),
        )

    def _classify_trend_quality(self, frame: pd.DataFrame) -> list[str]:
        low = self.config.orderliness_low_quantile
        high = self.config.orderliness_high_quantile
        out: list[str] = []
        for ready, percentile in zip(
            frame["data_ready"].fillna(False).to_numpy(dtype=bool),
            frame["orderliness_percentile"].to_numpy(dtype=float),
        ):
            if not ready or not np.isfinite(percentile):
                out.append("warmup")
            elif percentile >= high:
                out.append("high_order")
            elif percentile <= low:
                out.append("noisy")
            else:
                out.append("normal")
        return out

    def _classify_fast_pulse(self, frame: pd.DataFrame) -> list[str]:
        threshold = self.config.fast_pulse_threshold
        out: list[str] = []
        for ready, score in zip(
            frame["data_ready"].fillna(False).to_numpy(dtype=bool),
            frame["fast_trend_score"].to_numpy(dtype=float),
        ):
            if not ready or not np.isfinite(score):
                out.append("warmup")
            elif score >= threshold:
                out.append("up_pulse")
            elif score <= -threshold:
                out.append("down_pulse")
            else:
                out.append("neutral")
        return out

    def _classify_trend_phase(self, frame: pd.DataFrame) -> list[str]:
        cfg = self.config
        out: list[str] = []
        for ready, state, age, fast, medium, structural, candidate, progress in zip(
            frame["data_ready"].fillna(False).to_numpy(dtype=bool),
            frame["trend_state"].astype(str),
            frame["trend_state_age"].to_numpy(dtype=int),
            frame["fast_trend_score"].to_numpy(dtype=float),
            frame["medium_trend_score"].to_numpy(dtype=float),
            frame["trend_score"].to_numpy(dtype=float),
            frame["trend_candidate_state"].astype(str),
            frame["trend_candidate_progress"].to_numpy(dtype=float),
        ):
            if not ready or state == "warmup":
                out.append("warmup")
                continue
            if state == "balanced":
                out.append("transition" if progress > 0.0 and candidate != state else "balanced")
                continue
            sign = 1.0 if state == "up" else -1.0
            aligned_fast = sign * fast
            aligned_medium = sign * medium
            aligned_structural = sign * structural
            if progress > 0.0 and candidate != state:
                out.append("decay")
            elif age <= max(6, cfg.trend_confirm_bars * 3) and aligned_fast >= cfg.fast_pulse_threshold:
                out.append("startup")
            elif aligned_fast <= -0.5 * cfg.fast_pulse_threshold or aligned_structural <= 1.5 * cfg.trend_exit_threshold:
                out.append("decay")
            elif age >= max(30, cfg.slow_trend_window // 2) and aligned_fast < aligned_medium:
                out.append("mature")
            else:
                out.append("continuation")
        return out

    def _classify_flow_state(self, frame: pd.DataFrame) -> list[str]:
        cfg = self.config
        out: list[str] = []
        for available, flow, persistence, acceleration in zip(
            frame["orderflow_available"].fillna(False).to_numpy(dtype=bool),
            frame["flow_score"].to_numpy(dtype=float),
            frame["flow_persistence"].to_numpy(dtype=float),
            frame["flow_acceleration"].to_numpy(dtype=float),
        ):
            if not available or not np.isfinite(flow):
                out.append("unavailable")
            elif flow >= cfg.flow_threshold:
                if acceleration >= cfg.flow_acceleration_threshold:
                    out.append("buy_building")
                elif persistence >= 0.25:
                    out.append("buy_persistent")
                else:
                    out.append("buy_pressure")
            elif flow <= -cfg.flow_threshold:
                if acceleration <= -cfg.flow_acceleration_threshold:
                    out.append("sell_building")
                elif persistence <= -0.25:
                    out.append("sell_persistent")
                else:
                    out.append("sell_pressure")
            else:
                out.append("balanced")
        return out

    def _classify_impact_state(self, frame: pd.DataFrame) -> list[str]:
        cfg = self.config
        out: list[str] = []
        for available, flow, effectiveness, sell_absorption, buy_absorption in zip(
            frame["orderflow_available"].fillna(False).to_numpy(dtype=bool),
            frame["flow_score"].to_numpy(dtype=float),
            frame["flow_price_effectiveness"].to_numpy(dtype=float),
            frame["sell_absorption_score"].to_numpy(dtype=float),
            frame["buy_absorption_score"].to_numpy(dtype=float),
        ):
            if not available or not np.isfinite(flow):
                out.append("unavailable")
            elif np.isfinite(sell_absorption) and sell_absorption >= cfg.absorption_threshold:
                out.append("sell_absorbed")
            elif np.isfinite(buy_absorption) and buy_absorption >= cfg.absorption_threshold:
                out.append("buy_absorbed")
            elif flow >= cfg.flow_threshold and effectiveness >= cfg.impact_effective_threshold:
                out.append("buy_effective")
            elif flow <= -cfg.flow_threshold and effectiveness >= cfg.impact_effective_threshold:
                out.append("sell_effective")
            elif abs(flow) >= cfg.flow_threshold:
                out.append("mixed_response")
            else:
                out.append("neutral")
        return out

    def _classify_location_state(self, frame: pd.DataFrame) -> list[str]:
        out: list[str] = []
        for ready, downside_sweep, upside_sweep, breakout, breakdown, near_support, near_resistance, position in zip(
            frame["location_available"].fillna(False).to_numpy(dtype=bool),
            frame["downside_sweep_reclaim"].fillna(False).to_numpy(dtype=bool),
            frame["upside_sweep_reject"].fillna(False).to_numpy(dtype=bool),
            frame["breakout_accept"].fillna(False).to_numpy(dtype=bool),
            frame["breakdown_accept"].fillna(False).to_numpy(dtype=bool),
            frame["near_support"].fillna(False).to_numpy(dtype=bool),
            frame["near_resistance"].fillna(False).to_numpy(dtype=bool),
            frame["structural_position"].to_numpy(dtype=float),
        ):
            if not ready or not np.isfinite(position):
                out.append("warmup")
            elif downside_sweep:
                out.append("downside_sweep_reclaim")
            elif upside_sweep:
                out.append("upside_sweep_reject")
            elif breakout:
                out.append("breakout_accept")
            elif breakdown:
                out.append("breakdown_accept")
            elif near_support:
                out.append("near_support")
            elif near_resistance:
                out.append("near_resistance")
            elif position <= 0.25:
                out.append("lower_zone")
            elif position >= 0.75:
                out.append("upper_zone")
            else:
                out.append("middle_zone")
        return out

    def _compute_trade_context_score(self, frame: pd.DataFrame) -> pd.Series:
        trend = pd.to_numeric(frame["trend_score"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
        flow = pd.to_numeric(frame["flow_score"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
        signed_absorption = pd.to_numeric(frame["signed_absorption_score"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
        location = pd.to_numeric(frame["structural_location_score"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
        # Low structural location is supportive for long reversal context and
        # high location supportive for short reversal context.  This is only an
        # observable confluence score, not a fitted return predictor.
        score = 0.35 * trend + 0.30 * flow + 0.20 * signed_absorption - 0.15 * location
        shock_penalty = frame["volatility_state"].astype(str).eq("shock")
        return score.where(~shock_penalty, score * 0.35).clip(-1.0, 1.0)

    def _classify_trade_context(self, frame: pd.DataFrame) -> list[str]:
        out: list[str] = []
        long_locations = {"downside_sweep_reclaim", "near_support", "lower_zone"}
        short_locations = {"upside_sweep_reject", "near_resistance", "upper_zone"}
        for ready, trend, flow_state, impact, location, volatility in zip(
            frame["data_ready"].fillna(False).to_numpy(dtype=bool),
            frame["trend_state"].astype(str),
            frame["flow_state"].astype(str),
            frame["impact_state"].astype(str),
            frame["location_state"].astype(str),
            frame["volatility_state"].astype(str),
        ):
            if not ready:
                out.append("warmup")
            elif volatility == "shock":
                out.append("risk_off")
            elif impact == "sell_absorbed" and location in long_locations:
                out.append("long_reversal_watch")
            elif impact == "buy_absorbed" and location in short_locations:
                out.append("short_reversal_watch")
            elif trend == "up" and flow_state in {"buy_building", "buy_persistent"} and impact == "buy_effective" and location not in short_locations:
                out.append("long_continuation_watch")
            elif trend == "down" and flow_state in {"sell_building", "sell_persistent"} and impact == "sell_effective" and location not in long_locations:
                out.append("short_continuation_watch")
            elif trend == "up" and impact == "sell_effective":
                out.append("conflicted")
            elif trend == "down" and impact == "buy_effective":
                out.append("conflicted")
            else:
                out.append("wait")
        return out
