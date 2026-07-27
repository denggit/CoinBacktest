#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Causal multi-stage market-process map and online probability calibration.

V3.1 keeps the causal ordered-process framework and corrects two semantics
that V3 made too broad: reversal recovery now requires genuinely new reverse
flow plus price response after Sweep/Reclaim, while breakout completion requires
mature compression, a fresh level-breaking impulse, then later retest/hold
acceptance.  Every family has explicit stage order and stage time-to-live.  Probabilities are estimated only from
historical episodes whose completion/failure and forward outcome are already
observable at the current bar.

The module is reusable by research, analyze_tool and future live execution.  It
contains no exchange/data-loader code and never uses future values to construct
current process states.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
import pandas as pd

_EPS = 1e-12


PROCESS_FAMILIES: tuple[str, ...] = (
    "long_reversal",
    "short_reversal",
    "long_breakout",
    "short_breakdown",
)

PROCESS_DIRECTIONS: dict[str, int] = {
    "long_reversal": 1,
    "short_reversal": -1,
    "long_breakout": 1,
    "short_breakdown": -1,
}

PROCESS_STAGE_LABELS_LEGACY: dict[str, tuple[str, ...]] = {
    "long_reversal": ("idle", "sell_pressure", "sell_absorption", "sweep_reclaim", "buy_recovery"),
    "short_reversal": ("idle", "buy_pressure", "buy_absorption", "sweep_reject", "sell_recovery"),
    "long_breakout": ("idle", "compression", "buy_impulse", "breakout_accept"),
    "short_breakdown": ("idle", "compression", "sell_impulse", "breakdown_accept"),
}

PROCESS_STAGE_LABELS_V3_1: dict[str, tuple[str, ...]] = {
    "long_reversal": (
        "idle", "sell_pressure", "sell_absorption", "sweep_reclaim", "confirmed_buy_recovery"
    ),
    "short_reversal": (
        "idle", "buy_pressure", "buy_absorption", "sweep_reject", "confirmed_sell_recovery"
    ),
    "long_breakout": (
        "idle", "compression_ready", "breakout_impulse", "retest_hold_accept"
    ),
    "short_breakdown": (
        "idle", "compression_ready", "breakdown_impulse", "retest_hold_accept"
    ),
}

# Public default represents the current strict semantics.  Legacy V3 remains
# reproducible through ``ProcessMapConfig(semantic_version="v3")``.
PROCESS_STAGE_LABELS = PROCESS_STAGE_LABELS_V3_1


@dataclass(frozen=True)
class ProcessMapConfig:
    """V3/V3.1 process timing, semantic gates and probability configuration."""

    semantic_version: str = "v3_1"

    reversal_pressure_to_absorption_bars: int = 45
    reversal_absorption_to_sweep_bars: int = 30
    reversal_sweep_to_recovery_bars: int = 20
    reversal_completed_ttl_bars: int = 15
    reversal_recovery_min_delay_bars: int = 2
    reversal_recovery_flow_threshold: float = 0.08
    reversal_recovery_fast_flow_threshold: float = 0.10
    reversal_recovery_effectiveness_threshold: float = 0.08
    reversal_recovery_reclaim_atr: float = 0.10

    breakout_compression_min_bars: int = 8
    breakout_compression_to_impulse_bars: int = 30
    breakout_impulse_to_accept_bars: int = 20
    breakout_completed_ttl_bars: int = 20
    breakout_exit_grace_bars: int = 3
    breakout_impulse_flow_threshold: float = 0.08
    breakout_impulse_effectiveness_threshold: float = 0.20
    breakout_impulse_intensity_z: float = 0.50
    breakout_impulse_break_atr: float = 0.05
    breakout_accept_min_delay_bars: int = 2
    breakout_accept_hold_bars: int = 3
    breakout_accept_retest_atr: float = 0.25
    breakout_accept_hold_atr: float = 0.03
    breakout_accept_failure_atr: float = 0.20

    probability_horizons_bars: tuple[int, ...] = (15, 60, 180)
    default_reversal_horizon_bars: int = 15
    default_breakout_horizon_bars: int = 60
    probability_prior_successes: float = 1.0
    probability_prior_failures: float = 1.0
    minimum_probability_samples: int = 30
    baseline_sample_stride_bars: int = 5

    def validate(self) -> None:
        positive_ints = (
            "reversal_pressure_to_absorption_bars",
            "reversal_absorption_to_sweep_bars",
            "reversal_sweep_to_recovery_bars",
            "reversal_completed_ttl_bars",
            "reversal_recovery_min_delay_bars",
            "breakout_compression_min_bars",
            "breakout_compression_to_impulse_bars",
            "breakout_impulse_to_accept_bars",
            "breakout_completed_ttl_bars",
            "breakout_exit_grace_bars",
            "breakout_accept_min_delay_bars",
            "breakout_accept_hold_bars",
            "minimum_probability_samples",
            "baseline_sample_stride_bars",
        )
        for name in positive_ints:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.semantic_version not in {"v3", "v3_1"}:
            raise ValueError("semantic_version must be 'v3' or 'v3_1'")
        bounded_floats = (
            "reversal_recovery_flow_threshold",
            "reversal_recovery_fast_flow_threshold",
            "reversal_recovery_effectiveness_threshold",
            "reversal_recovery_reclaim_atr",
            "breakout_impulse_flow_threshold",
            "breakout_impulse_effectiveness_threshold",
            "breakout_impulse_break_atr",
            "breakout_accept_retest_atr",
            "breakout_accept_hold_atr",
            "breakout_accept_failure_atr",
        )
        for name in bounded_floats:
            value = float(getattr(self, name))
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if float(self.breakout_impulse_intensity_z) < 0.0:
            raise ValueError("breakout_impulse_intensity_z must be >= 0")
        horizons = tuple(sorted(set(int(v) for v in self.probability_horizons_bars)))
        if not horizons or any(v < 1 for v in horizons):
            raise ValueError("probability_horizons_bars must contain positive integers")
        if self.default_reversal_horizon_bars not in horizons:
            raise ValueError("default_reversal_horizon_bars must be in probability_horizons_bars")
        if self.default_breakout_horizon_bars not in horizons:
            raise ValueError("default_breakout_horizon_bars must be in probability_horizons_bars")
        if self.probability_prior_successes <= 0.0 or self.probability_prior_failures <= 0.0:
            raise ValueError("probability priors must be > 0")

    def stage_ttl(self, family: str, stage: int) -> int:
        if family in {"long_reversal", "short_reversal"}:
            return {
                1: self.reversal_pressure_to_absorption_bars,
                2: self.reversal_absorption_to_sweep_bars,
                3: self.reversal_sweep_to_recovery_bars,
                4: self.reversal_completed_ttl_bars,
            }[int(stage)]
        if family in {"long_breakout", "short_breakdown"}:
            return {
                1: self.breakout_compression_to_impulse_bars,
                2: self.breakout_impulse_to_accept_bars,
                3: self.breakout_completed_ttl_bars,
            }[int(stage)]
        raise KeyError(f"unknown process family: {family}")

    def stage_labels(self, family: str) -> tuple[str, ...]:
        mapping = PROCESS_STAGE_LABELS_LEGACY if self.semantic_version == "v3" else PROCESS_STAGE_LABELS_V3_1
        return mapping[family]

    def stage_label(self, family: str, stage: int) -> str:
        return self.stage_labels(family)[int(stage)]

    def max_stage(self, family: str) -> int:
        return len(self.stage_labels(family)) - 1

    def default_horizon(self, family: str) -> int:
        return (
            self.default_reversal_horizon_bars
            if family in {"long_reversal", "short_reversal"}
            else self.default_breakout_horizon_bars
        )


@dataclass
class _ActiveEpisode:
    family: str
    episode_id: int
    direction: int
    stage: int
    start_pos: int
    stage_pos: int
    deadline_pos: int
    confidence_sum: float
    confidence_count: int
    stage_positions: dict[int, int] = field(default_factory=dict)
    stage_times: dict[int, pd.Timestamp] = field(default_factory=dict)
    completion_recorded: bool = False

    @property
    def confidence(self) -> float:
        if self.confidence_count <= 0:
            return 0.0
        return float(np.clip(self.confidence_sum / self.confidence_count, 0.0, 1.0))


@dataclass(frozen=True)
class ProcessMapResult:
    frame: pd.DataFrame
    episodes: pd.DataFrame
    stage_events: pd.DataFrame
    metadata: dict[str, Any] = field(default_factory=dict)


class ProcessMapEngine:
    """Build ordered, expiring process episodes from a causal state frame."""

    def __init__(self, config: ProcessMapConfig | None = None) -> None:
        self.config = config or ProcessMapConfig()
        self.config.validate()

    def compute(self, state_frame: pd.DataFrame) -> ProcessMapResult:
        required = {
            "open",
            "high",
            "low",
            "close",
            "available_time",
            "data_ready",
            "volatility_state",
            "flow_state",
            "flow_score",
            "flow_strength",
            "impact_state",
            "sell_absorption_score",
            "buy_absorption_score",
            "location_state",
        }
        if self.config.semantic_version == "v3_1":
            required.update({
                "atr_pct",
                "flow_fast_score",
                "flow_acceleration",
                "flow_intensity_z",
                "price_move_score",
                "flow_price_effectiveness",
                "local_support",
                "local_resistance",
            })
        missing = sorted(required.difference(state_frame.columns))
        if missing:
            raise ValueError(f"market-state frame missing process columns: {missing}")
        if not state_frame.index.is_monotonic_increasing:
            raise ValueError("state_frame index must be monotonic increasing")

        frame = state_frame.copy()
        index = pd.DatetimeIndex(frame.index)
        available_time = pd.DatetimeIndex(pd.to_datetime(frame["available_time"]))
        n = len(frame)

        signals, evidence = self._build_signals(frame)
        family_arrays: dict[str, dict[str, np.ndarray]] = {}
        for family in PROCESS_FAMILIES:
            family_arrays[family] = {
                "stage": np.zeros(n, dtype=np.int16),
                "episode_id": np.zeros(n, dtype=np.int64),
                "age": np.zeros(n, dtype=np.int32),
                "stage_age": np.zeros(n, dtype=np.int32),
                "ttl_remaining": np.zeros(n, dtype=np.int32),
                "confidence": np.full(n, np.nan, dtype=float),
                "completed": np.zeros(n, dtype=bool),
                "stage_advanced": np.zeros(n, dtype=bool),
            }

        active: dict[str, _ActiveEpisode | None] = {family: None for family in PROCESS_FAMILIES}
        episode_counter: dict[str, int] = {family: 0 for family in PROCESS_FAMILIES}
        episode_rows: list[dict[str, Any]] = []
        stage_rows: list[dict[str, Any]] = []

        def start_episode(family: str, pos: int) -> _ActiveEpisode:
            episode_counter[family] += 1
            confidence = float(evidence[family][1][pos])
            episode = _ActiveEpisode(
                family=family,
                episode_id=episode_counter[family],
                direction=PROCESS_DIRECTIONS[family],
                stage=1,
                start_pos=pos,
                stage_pos=pos,
                deadline_pos=pos + self.config.stage_ttl(family, 1),
                confidence_sum=confidence,
                confidence_count=1,
                stage_positions={1: pos},
                stage_times={1: available_time[pos]},
            )
            stage_rows.append(self._stage_event_row(episode, pos, index, available_time, confidence))
            return episode

        def finalize_episode(
            episode: _ActiveEpisode,
            pos: int,
            status: str,
            reason: str,
        ) -> None:
            row: dict[str, Any] = {
                "family": episode.family,
                "direction": episode.direction,
                "episode_id": episode.episode_id,
                "status": status,
                "expiry_reason": reason,
                "max_stage_reached": episode.stage,
                "completed": status == "completed",
                "start_pos": episode.start_pos,
                "end_pos": pos,
                "start_timestamp": index[episode.start_pos],
                "start_available_time": available_time[episode.start_pos],
                "end_timestamp": index[min(pos, n - 1)],
                "end_available_time": available_time[min(pos, n - 1)],
                "duration_bars": max(0, pos - episode.start_pos),
                "confidence": episode.confidence,
            }
            max_stage = self.config.max_stage(episode.family)
            for stage in range(1, max_stage + 1):
                row[f"stage_{stage}_pos"] = episode.stage_positions.get(stage, np.nan)
                row[f"stage_{stage}_available_time"] = episode.stage_times.get(stage, pd.NaT)
                if stage > 1:
                    current = episode.stage_positions.get(stage)
                    previous = episode.stage_positions.get(stage - 1)
                    row[f"stage_{stage}_delay_bars"] = (
                        np.nan if current is None or previous is None else int(current - previous)
                    )
            episode_rows.append(row)

        for pos in range(n):
            ready = bool(frame["data_ready"].iloc[pos])
            for family in PROCESS_FAMILIES:
                episode = active[family]
                if episode is not None and pos > episode.deadline_pos:
                    if episode.stage >= self.config.max_stage(family):
                        if not episode.completion_recorded:
                            finalize_episode(episode, episode.stage_pos, "completed", "sequence_completed")
                            episode.completion_recorded = True
                    else:
                        finalize_episode(episode, episode.deadline_pos, "expired", f"stage_{episode.stage}_timeout")
                    active[family] = None
                    episode = None

                if not ready:
                    continue

                if episode is None:
                    if bool(signals[family][1][pos]):
                        episode = start_episode(family, pos)
                        active[family] = episode
                elif episode.stage < self.config.max_stage(family):
                    next_stage = episode.stage + 1
                    # A stage must be observed on a later closed bar.  This is
                    # the central V3 change from same-bar condition ladders.
                    if self._can_advance(frame, signals, family, next_stage, pos, episode):
                        confidence = float(evidence[family][next_stage][pos])
                        episode.stage = next_stage
                        episode.stage_pos = pos
                        episode.deadline_pos = pos + self.config.stage_ttl(family, next_stage)
                        episode.confidence_sum += confidence
                        episode.confidence_count += 1
                        episode.stage_positions[next_stage] = pos
                        episode.stage_times[next_stage] = available_time[pos]
                        stage_rows.append(self._stage_event_row(episode, pos, index, available_time, confidence))
                        if next_stage >= self.config.max_stage(family):
                            # The completed state remains observable for a
                            # short TTL, but the episode outcome is known now.
                            finalize_episode(episode, pos, "completed", "sequence_completed")
                            episode.completion_recorded = True

                if episode is not None:
                    arrays = family_arrays[family]
                    arrays["stage"][pos] = episode.stage
                    arrays["episode_id"][pos] = episode.episode_id
                    arrays["age"][pos] = pos - episode.start_pos
                    arrays["stage_age"][pos] = pos - episode.stage_pos
                    arrays["ttl_remaining"][pos] = max(0, episode.deadline_pos - pos)
                    arrays["confidence"][pos] = episode.confidence
                    arrays["completed"][pos] = episode.stage >= self.config.max_stage(family)
                    arrays["stage_advanced"][pos] = episode.stage_pos == pos

        # Preserve still-open episodes as open-ended audit rows.
        for family, episode in active.items():
            if episode is not None and episode.stage < self.config.max_stage(family):
                finalize_episode(episode, n - 1, "open", "data_end")

        for family, arrays in family_arrays.items():
            frame[f"{family}_stage"] = arrays["stage"]
            frame[f"{family}_stage_label"] = [
                self.config.stage_label(family, int(stage)) for stage in arrays["stage"]
            ]
            frame[f"{family}_episode_id"] = arrays["episode_id"]
            frame[f"{family}_age_bars"] = arrays["age"]
            frame[f"{family}_stage_age_bars"] = arrays["stage_age"]
            frame[f"{family}_ttl_remaining_bars"] = arrays["ttl_remaining"]
            frame[f"{family}_confidence"] = arrays["confidence"]
            frame[f"{family}_completed"] = arrays["completed"]
            frame[f"{family}_stage_advanced"] = arrays["stage_advanced"]

        self._attach_primary_process(frame)
        episodes = pd.DataFrame(episode_rows)
        if not episodes.empty:
            episodes = episodes.sort_values(["start_available_time", "family", "episode_id"]).reset_index(drop=True)
        stage_events = pd.DataFrame(stage_rows)
        if not stage_events.empty:
            stage_events = stage_events.sort_values(["available_time", "family", "episode_id", "stage"]).reset_index(drop=True)

        frame = attach_causal_process_probabilities(frame, episodes, self.config)
        metadata = {
            "version": "3.1" if self.config.semantic_version == "v3_1" else "3.0",
            "semantic_version": self.config.semantic_version,
            "families": list(PROCESS_FAMILIES),
            "episodes": int(len(episodes)),
            "completed_episodes": int(episodes.get("completed", pd.Series(dtype=bool)).fillna(False).sum()) if len(episodes) else 0,
            "family_episode_counts": episodes["family"].value_counts().to_dict() if len(episodes) else {},
            "family_completed_counts": (
                episodes.loc[episodes["completed"].eq(True), "family"].value_counts().to_dict()
                if len(episodes) else {}
            ),
            "probability_horizons_bars": list(self.config.probability_horizons_bars),
            "minimum_probability_samples": self.config.minimum_probability_samples,
        }
        return ProcessMapResult(frame=frame, episodes=episodes, stage_events=stage_events, metadata=metadata)

    def _build_signals(
        self,
        frame: pd.DataFrame,
    ) -> tuple[
        dict[str, dict[int, np.ndarray]],
        dict[str, dict[int, np.ndarray]],
    ]:
        flow_state = frame["flow_state"].astype(str)
        impact_state = frame["impact_state"].astype(str)
        location_state = frame["location_state"].astype(str)
        volatility_state = frame["volatility_state"].astype(str)
        flow_score = pd.to_numeric(frame["flow_score"], errors="coerce").fillna(0.0).clip(-1.0, 1.0)
        flow_strength = pd.to_numeric(frame["flow_strength"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        sell_absorption = pd.to_numeric(frame["sell_absorption_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        buy_absorption = pd.to_numeric(frame["buy_absorption_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        structural_location = pd.to_numeric(
            frame.get("structural_location_score", 0.0), errors="coerce"
        ).fillna(0.0).clip(-1.0, 1.0)

        sell_flow = flow_state.isin({"sell_pressure", "sell_building", "sell_persistent"})
        buy_flow = flow_state.isin({"buy_pressure", "buy_building", "buy_persistent"})
        compression = volatility_state.isin({"dormant", "compression"})

        if self.config.semantic_version == "v3":
            buy_recovery = buy_flow | (flow_score.gt(0.0) & flow_score.shift(1).le(0.0))
            sell_recovery = sell_flow | (flow_score.lt(0.0) & flow_score.shift(1).ge(0.0))
            compression_ready = compression
            long_impulse = buy_flow & impact_state.eq("buy_effective")
            short_impulse = sell_flow & impact_state.eq("sell_effective")
            long_accept = location_state.eq("breakout_accept")
            short_accept = location_state.eq("breakdown_accept")
        else:
            # One start event per contiguous compression episode.  The old V3
            # opened an episode on every compression bar and then waited up to
            # 90 bars, making an eventual impulse almost guaranteed.
            group_id = (~compression).cumsum()
            compression_streak = compression.groupby(group_id).cumsum()
            compression_ready = compression_streak.eq(self.config.breakout_compression_min_bars)

            flow_fast = pd.to_numeric(frame["flow_fast_score"], errors="coerce")
            flow_effectiveness = pd.to_numeric(frame["flow_price_effectiveness"], errors="coerce")
            flow_intensity_z = pd.to_numeric(frame["flow_intensity_z"], errors="coerce")
            price_move = pd.to_numeric(frame["price_move_score"], errors="coerce")
            atr_abs = pd.to_numeric(frame["atr_pct"], errors="coerce") * pd.to_numeric(frame["close"], errors="coerce")
            local_resistance = pd.to_numeric(frame["local_resistance"], errors="coerce")
            local_support = pd.to_numeric(frame["local_support"], errors="coerce")
            prior_compression = compression.shift(1, fill_value=False).rolling(
                self.config.breakout_exit_grace_bars,
                min_periods=1,
            ).max().astype(bool)
            fresh_volatility_exit = (~compression) & prior_compression & volatility_state.isin({"normal", "expansion", "shock"})
            long_impulse = (
                fresh_volatility_exit
                & buy_flow
                & impact_state.eq("buy_effective")
                & flow_score.ge(self.config.breakout_impulse_flow_threshold)
                & flow_fast.ge(self.config.breakout_impulse_flow_threshold)
                & flow_effectiveness.ge(self.config.breakout_impulse_effectiveness_threshold)
                & flow_intensity_z.ge(self.config.breakout_impulse_intensity_z)
                & price_move.gt(0.0)
                & pd.to_numeric(frame["close"], errors="coerce").gt(
                    local_resistance + self.config.breakout_impulse_break_atr * atr_abs
                )
            )
            short_impulse = (
                fresh_volatility_exit
                & sell_flow
                & impact_state.eq("sell_effective")
                & flow_score.le(-self.config.breakout_impulse_flow_threshold)
                & flow_fast.le(-self.config.breakout_impulse_flow_threshold)
                & flow_effectiveness.ge(self.config.breakout_impulse_effectiveness_threshold)
                & flow_intensity_z.ge(self.config.breakout_impulse_intensity_z)
                & price_move.lt(0.0)
                & pd.to_numeric(frame["close"], errors="coerce").lt(
                    local_support - self.config.breakout_impulse_break_atr * atr_abs
                )
            )
            # Final strict checks are episode-anchor dependent and therefore run
            # in ``_can_advance``.  These broad masks only keep the table shape.
            buy_recovery = buy_flow
            sell_recovery = sell_flow
            long_accept = pd.Series(True, index=frame.index)
            short_accept = pd.Series(True, index=frame.index)

        signals: dict[str, dict[int, np.ndarray]] = {
            "long_reversal": {
                1: (sell_flow | impact_state.eq("sell_effective")).to_numpy(dtype=bool),
                2: impact_state.eq("sell_absorbed").to_numpy(dtype=bool),
                3: location_state.eq("downside_sweep_reclaim").to_numpy(dtype=bool),
                4: buy_recovery.to_numpy(dtype=bool),
            },
            "short_reversal": {
                1: (buy_flow | impact_state.eq("buy_effective")).to_numpy(dtype=bool),
                2: impact_state.eq("buy_absorbed").to_numpy(dtype=bool),
                3: location_state.eq("upside_sweep_reject").to_numpy(dtype=bool),
                4: sell_recovery.to_numpy(dtype=bool),
            },
            "long_breakout": {
                1: compression_ready.to_numpy(dtype=bool),
                2: long_impulse.to_numpy(dtype=bool),
                3: long_accept.to_numpy(dtype=bool),
            },
            "short_breakdown": {
                1: compression_ready.to_numpy(dtype=bool),
                2: short_impulse.to_numpy(dtype=bool),
                3: short_accept.to_numpy(dtype=bool),
            },
        }

        evidence: dict[str, dict[int, np.ndarray]] = {
            "long_reversal": {
                1: np.maximum((-flow_score).clip(0.0, 1.0), flow_strength * impact_state.eq("sell_effective")).to_numpy(float),
                2: sell_absorption.to_numpy(float),
                3: ((1.0 - structural_location) / 2.0).clip(0.0, 1.0).to_numpy(float),
                4: flow_score.clip(0.0, 1.0).to_numpy(float),
            },
            "short_reversal": {
                1: np.maximum(flow_score.clip(0.0, 1.0), flow_strength * impact_state.eq("buy_effective")).to_numpy(float),
                2: buy_absorption.to_numpy(float),
                3: ((1.0 + structural_location) / 2.0).clip(0.0, 1.0).to_numpy(float),
                4: (-flow_score).clip(0.0, 1.0).to_numpy(float),
            },
            "long_breakout": {
                1: volatility_state.eq("compression").astype(float).add(
                    0.5 * volatility_state.eq("dormant")
                ).clip(0.0, 1.0).to_numpy(float),
                2: (0.5 * flow_strength + 0.5 * flow_score.clip(0.0, 1.0)).clip(0.0, 1.0).to_numpy(float),
                3: ((1.0 + structural_location) / 2.0).clip(0.0, 1.0).to_numpy(float),
            },
            "short_breakdown": {
                1: volatility_state.eq("compression").astype(float).add(
                    0.5 * volatility_state.eq("dormant")
                ).clip(0.0, 1.0).to_numpy(float),
                2: (0.5 * flow_strength + 0.5 * (-flow_score).clip(0.0, 1.0)).clip(0.0, 1.0).to_numpy(float),
                3: ((1.0 - structural_location) / 2.0).clip(0.0, 1.0).to_numpy(float),
            },
        }
        return signals, evidence

    def _can_advance(
        self,
        frame: pd.DataFrame,
        signals: dict[str, dict[int, np.ndarray]],
        family: str,
        next_stage: int,
        pos: int,
        episode: _ActiveEpisode,
    ) -> bool:
        if pos <= episode.stage_pos or not bool(signals[family][next_stage][pos]):
            return False
        if self.config.semantic_version == "v3":
            return True
        if family in {"long_reversal", "short_reversal"} and next_stage == 4:
            return self._strict_reversal_recovery(frame, family, pos, episode)
        if family in {"long_breakout", "short_breakdown"} and next_stage == 3:
            return self._strict_breakout_acceptance(frame, family, pos, episode)
        return True

    def _strict_reversal_recovery(
        self,
        frame: pd.DataFrame,
        family: str,
        pos: int,
        episode: _ActiveEpisode,
    ) -> bool:
        stage3_pos = int(episode.stage_positions[3])
        if pos - stage3_pos < self.config.reversal_recovery_min_delay_bars:
            return False
        row = frame.iloc[pos]
        previous = frame.iloc[pos - 1]
        sweep = frame.iloc[stage3_pos]
        close = float(row["close"])
        sweep_close = float(sweep["close"])
        atr_abs = float(row["atr_pct"]) * close
        if not np.isfinite(atr_abs) or atr_abs <= 0.0:
            return False
        flow_score = float(row["flow_score"])
        flow_fast = float(row["flow_fast_score"])
        flow_acceleration = float(row["flow_acceleration"])
        effectiveness = float(row["flow_price_effectiveness"])
        price_move = float(row["price_move_score"])
        previous_fast = float(previous["flow_fast_score"])
        previous_score = float(previous["flow_score"])
        if family == "long_reversal":
            crossed_after_sweep = (
                (np.isfinite(previous_fast) and previous_fast < self.config.reversal_recovery_fast_flow_threshold)
                or (np.isfinite(previous_score) and previous_score <= 0.0)
            )
            return bool(
                crossed_after_sweep
                and flow_score >= self.config.reversal_recovery_flow_threshold
                and flow_fast >= self.config.reversal_recovery_fast_flow_threshold
                and flow_acceleration > 0.0
                and effectiveness >= self.config.reversal_recovery_effectiveness_threshold
                and price_move > 0.0
                and close >= sweep_close + self.config.reversal_recovery_reclaim_atr * atr_abs
            )
        crossed_after_sweep = (
            (np.isfinite(previous_fast) and previous_fast > -self.config.reversal_recovery_fast_flow_threshold)
            or (np.isfinite(previous_score) and previous_score >= 0.0)
        )
        return bool(
            crossed_after_sweep
            and flow_score <= -self.config.reversal_recovery_flow_threshold
            and flow_fast <= -self.config.reversal_recovery_fast_flow_threshold
            and flow_acceleration < 0.0
            and effectiveness >= self.config.reversal_recovery_effectiveness_threshold
            and price_move < 0.0
            and close <= sweep_close - self.config.reversal_recovery_reclaim_atr * atr_abs
        )

    def _strict_breakout_acceptance(
        self,
        frame: pd.DataFrame,
        family: str,
        pos: int,
        episode: _ActiveEpisode,
    ) -> bool:
        impulse_pos = int(episode.stage_positions[2])
        delay = pos - impulse_pos
        if delay < self.config.breakout_accept_min_delay_bars:
            return False
        segment = frame.iloc[impulse_pos + 1 : pos + 1]
        if len(segment) < self.config.breakout_accept_hold_bars:
            return False
        impulse = frame.iloc[impulse_pos]
        close = pd.to_numeric(segment["close"], errors="coerce")
        low = pd.to_numeric(segment["low"], errors="coerce")
        high = pd.to_numeric(segment["high"], errors="coerce")
        current = frame.iloc[pos]
        current_close = float(current["close"])
        current_atr = float(current["atr_pct"]) * current_close
        if not np.isfinite(current_atr) or current_atr <= 0.0:
            return False
        flow_score = float(current["flow_score"])
        effectiveness = float(current["flow_price_effectiveness"])
        hold_window = close.iloc[-self.config.breakout_accept_hold_bars :]
        if family == "long_breakout":
            anchor = float(impulse["local_resistance"])
            if not np.isfinite(anchor):
                return False
            failed = bool((close < anchor - self.config.breakout_accept_failure_atr * current_atr).any())
            retested = bool((low <= anchor + self.config.breakout_accept_retest_atr * current_atr).any())
            held = bool((hold_window >= anchor - self.config.breakout_accept_failure_atr * current_atr).all())
            stayed_above = bool((hold_window >= anchor + self.config.breakout_accept_hold_atr * current_atr).all())
            accepted = current_close >= anchor + self.config.breakout_accept_hold_atr * current_atr
            return bool(
                not failed and (retested or stayed_above) and held and accepted
                and flow_score >= -0.02 and effectiveness >= -0.05
            )
        anchor = float(impulse["local_support"])
        if not np.isfinite(anchor):
            return False
        failed = bool((close > anchor + self.config.breakout_accept_failure_atr * current_atr).any())
        retested = bool((high >= anchor - self.config.breakout_accept_retest_atr * current_atr).any())
        held = bool((hold_window <= anchor + self.config.breakout_accept_failure_atr * current_atr).all())
        stayed_below = bool((hold_window <= anchor - self.config.breakout_accept_hold_atr * current_atr).all())
        accepted = current_close <= anchor - self.config.breakout_accept_hold_atr * current_atr
        return bool(
            not failed and (retested or stayed_below) and held and accepted
            and flow_score <= 0.02 and effectiveness >= -0.05
        )

    def _stage_event_row(
        self,
        episode: _ActiveEpisode,
        pos: int,
        index: pd.DatetimeIndex,
        available_time: pd.DatetimeIndex,
        stage_evidence: float,
    ) -> dict[str, Any]:
        return {
            "family": episode.family,
            "direction": episode.direction,
            "episode_id": episode.episode_id,
            "stage": episode.stage,
            "stage_label": self.config.stage_label(episode.family, episode.stage),
            "position": pos,
            "timestamp": index[pos],
            "available_time": available_time[pos],
            "stage_evidence": float(stage_evidence),
            "cumulative_confidence": episode.confidence,
        }

    def _attach_primary_process(self, frame: pd.DataFrame) -> None:
        n = len(frame)
        family_out = np.full(n, "none", dtype=object)
        direction_out = np.zeros(n, dtype=np.int8)
        stage_out = np.zeros(n, dtype=np.int16)
        stage_label_out = np.full(n, "idle", dtype=object)
        progress_out = np.zeros(n, dtype=float)
        confidence_out = np.full(n, np.nan, dtype=float)
        age_out = np.zeros(n, dtype=np.int32)
        ttl_out = np.zeros(n, dtype=np.int32)
        status_out = np.full(n, "idle", dtype=object)

        # Scalar DataFrame.iloc inside a 2M-row loop is prohibitively slow and
        # can become super-linear on a fragmented frame.  Extract compact arrays
        # once and keep the sequential selection loop purely NumPy/Python.
        stages = {family: frame[f"{family}_stage"].to_numpy(dtype=np.int16) for family in PROCESS_FAMILIES}
        confidences = {
            family: pd.to_numeric(frame[f"{family}_confidence"], errors="coerce").to_numpy(float)
            for family in PROCESS_FAMILIES
        }
        ages = {family: frame[f"{family}_age_bars"].to_numpy(dtype=np.int32) for family in PROCESS_FAMILIES}
        ttls = {family: frame[f"{family}_ttl_remaining_bars"].to_numpy(dtype=np.int32) for family in PROCESS_FAMILIES}

        for pos in range(n):
            candidates: list[tuple[float, float, int, str, int]] = []
            for family in PROCESS_FAMILIES:
                stage = int(stages[family][pos])
                if stage <= 0:
                    continue
                confidence_value = confidences[family][pos]
                if not np.isfinite(confidence_value):
                    confidence_value = 0.0
                progress = stage / self.config.max_stage(family)
                candidates.append((progress, float(confidence_value), stage, family, PROCESS_DIRECTIONS[family]))
            if not candidates:
                continue
            active_families = {candidate[3] for candidate in candidates}
            if active_families == {"long_breakout", "short_breakdown"} and all(candidate[2] == 1 for candidate in candidates):
                family_out[pos] = "compression_setup"
                direction_out[pos] = 0
                stage_out[pos] = 1
                stage_label_out[pos] = "compression"
                progress_out[pos] = 1.0 / 3.0
                confidence_out[pos] = max(candidate[1] for candidate in candidates)
                age_out[pos] = max(int(ages[candidate[3]][pos]) for candidate in candidates)
                ttl_out[pos] = max(int(ttls[candidate[3]][pos]) for candidate in candidates)
                status_out[pos] = "building"
                continue
            candidates.sort(reverse=True)
            best = candidates[0]
            opposite_same_progress = any(
                candidate[4] != best[4]
                and abs(candidate[0] - best[0]) <= 0.05
                and abs(candidate[1] - best[1]) <= 0.15
                for candidate in candidates[1:]
            )
            if opposite_same_progress:
                family_out[pos] = "conflict"
                status_out[pos] = "conflict"
                direction_out[pos] = 0
                progress_out[pos] = best[0]
                confidence_out[pos] = best[1]
                continue
            _, confidence, stage, family, direction = best
            family_out[pos] = family
            direction_out[pos] = direction
            stage_out[pos] = stage
            stage_label_out[pos] = self.config.stage_label(family, stage)
            progress_out[pos] = stage / self.config.max_stage(family)
            confidence_out[pos] = confidence
            age_out[pos] = int(ages[family][pos])
            ttl_out[pos] = int(ttls[family][pos])
            status_out[pos] = "complete" if stage >= self.config.max_stage(family) else "building"

        primary = pd.DataFrame(
            {
                "process_family": family_out,
                "process_direction": direction_out,
                "process_stage": stage_out,
                "process_stage_label": stage_label_out,
                "process_progress": progress_out,
                "process_confidence": confidence_out,
                "process_age_bars": age_out,
                "process_ttl_remaining_bars": ttl_out,
                "process_status": status_out,
            },
            index=frame.index,
        )
        for column in primary:
            frame[column] = primary[column].to_numpy()


def _beta_probability(successes: np.ndarray, counts: np.ndarray, cfg: ProcessMapConfig) -> np.ndarray:
    return (
        successes + cfg.probability_prior_successes
    ) / (
        counts + cfg.probability_prior_successes + cfg.probability_prior_failures
    )


def _event_position_map(frame: pd.DataFrame) -> dict[pd.Timestamp, int]:
    available = pd.DatetimeIndex(pd.to_datetime(frame["available_time"]))
    return {pd.Timestamp(value): pos for pos, value in enumerate(available)}


def attach_causal_process_probabilities(
    frame: pd.DataFrame,
    episodes: pd.DataFrame,
    config: ProcessMapConfig,
) -> pd.DataFrame:
    """Attach online probabilities using only resolved historical outcomes.

    The function mutates the supplied process frame in place to avoid another
    multi-million-row copy.  Every contribution is placed at the bar where the
    episode result or forward outcome becomes observable, then cumulatively
    summed.  The current episode can never train its own displayed probability.
    """

    out = frame
    n = len(out)
    if n == 0:
        return out
    close = pd.to_numeric(out["close"], errors="coerce").to_numpy(float)
    open_ = pd.to_numeric(out["open"], errors="coerce").to_numpy(float)
    probability_columns: dict[str, np.ndarray] = {}

    # Vectorized all-market causal baselines.
    baseline_curves: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for horizon in sorted(set(int(v) for v in config.probability_horizons_bars)):
        signal_pos = np.arange(0, max(0, n - horizon), config.baseline_sample_stride_bars, dtype=np.int64)
        entry_pos = signal_pos + 1
        outcome_pos = entry_pos + horizon - 1
        valid = (
            (outcome_pos < n)
            & np.isfinite(open_[entry_pos])
            & np.isfinite(close[outcome_pos])
            & (open_[entry_pos] != 0.0)
        )
        entry_pos = entry_pos[valid]
        outcome_pos = outcome_pos[valid]
        ratio_return = close[outcome_pos] / open_[entry_pos] - 1.0
        for direction, name in ((1, "long"), (-1, "short")):
            known_count = np.zeros(n, dtype=float)
            known_success = np.zeros(n, dtype=float)
            np.add.at(known_count, outcome_pos, 1.0)
            np.add.at(known_success, outcome_pos, (direction * ratio_return > 0.0).astype(float))
            cumulative_count = np.cumsum(known_count)
            cumulative_success = np.cumsum(known_success)
            probability = _beta_probability(cumulative_success, cumulative_count, config)
            probability[cumulative_count < config.minimum_probability_samples] = np.nan
            baseline_curves[(direction, horizon)] = (probability, cumulative_count)
            probability_columns[f"baseline_{name}_probability_h{horizon}"] = probability
            probability_columns[f"baseline_{name}_samples_h{horizon}"] = cumulative_count.astype(np.int64)

    completion_probability = np.full(n, np.nan, dtype=float)
    completion_samples = np.zeros(n, dtype=np.int64)
    direction_probability = np.full(n, np.nan, dtype=float)
    direction_uplift = np.full(n, np.nan, dtype=float)
    direction_samples = np.zeros(n, dtype=np.int64)
    probability_horizon = np.zeros(n, dtype=np.int32)

    if not episodes.empty:
        completion_curves: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        direction_curves: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        for family in PROCESS_FAMILIES:
            max_stage = config.max_stage(family)
            family_episodes = episodes.loc[episodes["family"].eq(family)].copy()
            for stage in range(1, max_stage):
                known_count = np.zeros(n, dtype=float)
                known_success = np.zeros(n, dtype=float)
                stage_column = f"stage_{stage}_pos"
                if stage_column not in family_episodes.columns:
                    completion_curves[(family, stage)] = (
                        np.full(n, np.nan, dtype=float),
                        np.zeros(n, dtype=float),
                    )
                    continue
                reached_mask = pd.to_numeric(
                    family_episodes[stage_column], errors="coerce"
                ).notna()
                reached = family_episodes.loc[reached_mask]
                resolved_pos = pd.to_numeric(reached["end_pos"], errors="coerce").dropna().astype(np.int64).to_numpy()
                success = reached.loc[pd.to_numeric(reached["end_pos"], errors="coerce").notna(), "completed"].fillna(False).astype(float).to_numpy()
                valid = (resolved_pos >= 0) & (resolved_pos < n)
                resolved_pos = resolved_pos[valid]
                success = success[valid]
                np.add.at(known_count, resolved_pos, 1.0)
                np.add.at(known_success, resolved_pos, success)
                cumulative_count = np.cumsum(known_count)
                cumulative_success = np.cumsum(known_success)
                probability = _beta_probability(cumulative_success, cumulative_count, config)
                probability[cumulative_count < config.minimum_probability_samples] = np.nan
                completion_curves[(family, stage)] = (probability, cumulative_count)

            completed = family_episodes.loc[family_episodes["completed"].eq(True)]
            final_stage_column = f"stage_{max_stage}_pos"
            if final_stage_column in completed.columns:
                completion_positions = pd.to_numeric(
                    completed[final_stage_column], errors="coerce"
                ).dropna().astype(np.int64).to_numpy()
            else:
                completion_positions = np.array([], dtype=np.int64)
            direction = PROCESS_DIRECTIONS[family]
            for horizon in sorted(set(int(v) for v in config.probability_horizons_bars)):
                known_count = np.zeros(n, dtype=float)
                known_success = np.zeros(n, dtype=float)
                entry_pos = completion_positions + 1
                outcome_pos = entry_pos + horizon - 1
                valid = (
                    (entry_pos >= 0)
                    & (outcome_pos < n)
                    & np.isfinite(open_[np.clip(entry_pos, 0, n - 1)])
                    & np.isfinite(close[np.clip(outcome_pos, 0, n - 1)])
                )
                entry_valid = entry_pos[valid]
                outcome_valid = outcome_pos[valid]
                returns = direction * (close[outcome_valid] / open_[entry_valid] - 1.0)
                np.add.at(known_count, outcome_valid, 1.0)
                np.add.at(known_success, outcome_valid, (returns > 0.0).astype(float))
                cumulative_count = np.cumsum(known_count)
                cumulative_success = np.cumsum(known_success)
                probability = _beta_probability(cumulative_success, cumulative_count, config)
                probability[cumulative_count < config.minimum_probability_samples] = np.nan
                direction_curves[(family, horizon)] = (probability, cumulative_count)
                probability_columns[f"{family}_direction_probability_h{horizon}"] = probability
                probability_columns[f"{family}_direction_samples_h{horizon}"] = cumulative_count.astype(np.int64)

        family_values = out["process_family"].astype(str).to_numpy()
        stage_values = pd.to_numeric(out["process_stage"], errors="coerce").fillna(0).to_numpy(dtype=np.int16)
        direction_values = pd.to_numeric(out["process_direction"], errors="coerce").fillna(0).to_numpy(dtype=np.int8)
        # Vectorized assignment family by family/stage instead of scalar iloc.
        for family in PROCESS_FAMILIES:
            family_mask = family_values == family
            if not family_mask.any():
                continue
            max_stage = config.max_stage(family)
            for stage in range(1, max_stage):
                mask = family_mask & (stage_values == stage)
                if not mask.any() or (family, stage) not in completion_curves:
                    continue
                probability, counts = completion_curves[(family, stage)]
                completion_probability[mask] = probability[mask]
                completion_samples[mask] = counts[mask].astype(np.int64)
            completed_mask = family_mask & (stage_values >= max_stage)
            completion_probability[completed_mask] = 1.0
            completion_samples[completed_mask] = 0

            horizon = config.default_horizon(family)
            probability_horizon[family_mask] = horizon
            probability, counts = direction_curves.get(
                (family, horizon),
                (np.full(n, np.nan), np.zeros(n)),
            )
            direction_probability[family_mask] = probability[family_mask]
            direction_samples[family_mask] = counts[family_mask].astype(np.int64)
            baseline_probability, _ = baseline_curves[(PROCESS_DIRECTIONS[family], horizon)]
            valid = family_mask & np.isfinite(probability) & np.isfinite(baseline_probability)
            direction_uplift[valid] = probability[valid] - baseline_probability[valid]

    probability_columns.update(
        {
            "process_completion_probability": completion_probability,
            "process_completion_samples": completion_samples,
            "process_direction_probability": direction_probability,
            "process_direction_probability_uplift": direction_uplift,
            "process_direction_samples": direction_samples,
            "process_probability_horizon_bars": probability_horizon,
        }
    )
    probability_frame = pd.DataFrame(probability_columns, index=out.index)
    for column in probability_frame:
        out[column] = probability_frame[column].to_numpy()
    return out


def stage_event_mask(
    process_frame: pd.DataFrame,
    family: str,
    stage: int,
) -> pd.Series:
    """Return a causal mask for first entry into one process stage."""

    column = f"{family}_stage"
    if column not in process_frame:
        raise KeyError(column)
    values = pd.to_numeric(process_frame[column], errors="coerce").fillna(0).astype(int)
    return values.eq(int(stage)) & values.shift(1, fill_value=0).ne(int(stage))


def completed_process_mask(process_frame: pd.DataFrame, family: str) -> pd.Series:
    return stage_event_mask(process_frame, family, len(PROCESS_STAGE_LABELS[family]) - 1)
