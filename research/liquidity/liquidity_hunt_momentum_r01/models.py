#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fixed definitions for Liquidity Hunt Momentum R01."""

from __future__ import annotations

from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class LiquidityHuntConfig:
    """Fixed, predeclared research definition.

    Thresholds are intentionally few and coarse.  The research reports nested
    stages and nearby range sizes instead of mining a large parameter grid.
    """

    support_lookback_bars: int = 20
    notional_median_bars: int = 50
    notional_min_periods: int = 20
    attack_buy_ratio: float = 0.70
    attack_sell_ratio: float = 0.70
    attack_notional_multiple: float = 1.50
    reclaim_volume_ratio_max: float = 0.85
    obi_extreme: float = 0.30
    obi_reversal_positive: float = 0.20
    obi_sustained: float = 0.30
    obi_neutral: float = 0.05
    void_depth_ratio_max: float = 0.60
    void_side_ratio_max: float = 0.75
    rebuilt_depth_base_min: float = 50.0
    rebuilt_distance_bps_max: float = 12.0
    flow_window_seconds: int = 5
    book_reference_minutes: int = 10
    cooldown_minutes: int = 15
    mode1_stop_buffer_pct: float = 0.0020
    mode2_stop_buffer_pct: float = 0.0010
    minimum_raw_rr: float = 0.75
    fallback_target_r: float = 1.50
    time_stop_minutes: int = 15
    max_holding_minutes: int = 60
    decay_flow_ratio: float = 0.60
    round_trip_cost: float = 0.0011

    def validate(self) -> None:
        if self.support_lookback_bars < 2:
            raise ValueError("support_lookback_bars must be >= 2")
        if self.notional_median_bars < 2:
            raise ValueError("notional_median_bars must be >= 2")
        if not 0.5 <= self.attack_buy_ratio <= 1.0:
            raise ValueError("attack_buy_ratio must be in [0.5, 1]")
        if not 0.5 <= self.attack_sell_ratio <= 1.0:
            raise ValueError("attack_sell_ratio must be in [0.5, 1]")
        if self.attack_notional_multiple <= 0:
            raise ValueError("attack_notional_multiple must be > 0")
        if not 0 <= self.reclaim_volume_ratio_max <= 2:
            raise ValueError("reclaim_volume_ratio_max must be in [0, 2]")
        if self.flow_window_seconds <= 0:
            raise ValueError("flow_window_seconds must be > 0")
        if self.book_reference_minutes <= 0:
            raise ValueError("book_reference_minutes must be > 0")
        if self.time_stop_minutes <= 0 or self.max_holding_minutes < self.time_stop_minutes:
            raise ValueError("invalid time-stop/max-holding settings")
        if self.round_trip_cost < 0:
            raise ValueError("round_trip_cost must be >= 0")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StrategyVariant:
    name: str
    cost_multiplier: float = 1.0
    entry_delay_bars: int = 1
    use_liquidity_target: bool = True
    use_dynamic_decay_exit: bool = True
    use_time_stop: bool = True

    def validate(self) -> None:
        if self.cost_multiplier < 0:
            raise ValueError("cost_multiplier must be >= 0")
        if self.entry_delay_bars < 1:
            raise ValueError("entry_delay_bars must be >= 1")
