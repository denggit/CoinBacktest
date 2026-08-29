#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Frozen ETH Portfolio V2 strategy-development catalog.

The catalog contains executable destinations, not generic edge topics.  Each
item must eventually produce trades that can be replayed independently and then
combined by the portfolio layer.
"""

from __future__ import annotations

from src.strategy_common import StrategyContract


PORTFOLIO_V2_ID = "ETH_PORTFOLIO_V2"


def build_core_strategy_catalog() -> tuple[StrategyContract, ...]:
    rows = (
        StrategyContract(
            strategy_id="ETH_STRATEGY_TREND_BREAKOUT_V1",
            name="ETH Trend Breakout V1",
            sleeve_id="ETH_SLEEVE_TREND_BREAKOUT",
            family="trend_breakout",
            strategy_class="core",
            stage="in_development",
            symbol="ETH-USDT-SWAP",
            signal_timeframe="15m",
            execution_timeframe="15m",
            setup="Price closes beyond a causal prior structure range after the range is fully known.",
            trigger="First close outside the prior breakout level; feature quality is scored rather than stacked as hard filters.",
            entry="Enter on the next 15m bar open with slippage stress applied.",
            initial_stop="Causal recent opposite structure extreme, bounded only by portfolio risk/execution sanity limits.",
            exit_logic="Preplaced protective stop or fixed 2R target; close-known discretionary exits are disabled in V1.",
            sizing="Risk-based sizing multiplied by continuous setup quality; no martingale or averaging down.",
            invalidation="Invalid stop geometry, non-finite execution price, portfolio risk cap, or end-of-data forced close.",
            target_min_trades=300,
        ),
        StrategyContract(
            strategy_id="ETH_STRATEGY_TREND_PULLBACK_V1",
            name="ETH Trend Pullback V1",
            sleeve_id="ETH_SLEEVE_TREND_PULLBACK",
            family="trend_pullback",
            strategy_class="core",
            stage="planned",
            symbol="ETH-USDT-SWAP",
            signal_timeframe="15m",
            execution_timeframe="15m",
            setup="Existing directional trend with a controlled pullback that has not invalidated structure.",
            trigger="Causal re-acceleration/reclaim after pullback; microstructure variables are scores, not serial filters.",
            entry="Next-bar open or predeclared causal pullback order model.",
            initial_stop="Beyond pullback invalidation structure.",
            exit_logic="Trend structure failure, R protection/target and causal profit-protection logic.",
            sizing="Risk budget scaled by trend compatibility and pullback quality.",
            invalidation="Trend structure breaks before entry or stop/risk geometry is not tradable.",
            target_min_trades=300,
        ),
        StrategyContract(
            strategy_id="ETH_STRATEGY_LIQUIDITY_REVERSAL_V1",
            name="ETH Liquidity Reversal V1",
            sleeve_id="ETH_SLEEVE_LIQUIDITY_REVERSAL",
            family="liquidity_reversal",
            strategy_class="core",
            stage="planned",
            symbol="ETH-USDT-SWAP",
            signal_timeframe="1m/2m",
            execution_timeframe="1m/2m",
            setup="A classified external liquidity pool is swept; sweep itself is not an entry.",
            trigger="Post-sweep reclaim/displacement/MSS/flow evidence reaches a causal quality score.",
            entry="Explicit post-confirmation next-open or causal pullback entry.",
            initial_stop="Beyond swept-liquidity extreme plus predeclared buffer.",
            exit_logic="Opposing external liquidity is primary target with structural protection/exit.",
            sizing="Risk scaled by liquidity quality, post-sweep response score and portfolio conflict budget.",
            invalidation="Acceptance beyond swept level, failed confirmation, stale setup or risk limit.",
            target_min_trades=300,
        ),
        StrategyContract(
            strategy_id="ETH_STRATEGY_RANGE_REVERSION_V1",
            name="ETH Range Mean Reversion V1",
            sleeve_id="ETH_SLEEVE_RANGE_REVERSION",
            family="range_mean_reversion",
            strategy_class="core",
            stage="planned",
            symbol="ETH-USDT-SWAP",
            signal_timeframe="5m/15m",
            execution_timeframe="5m/15m",
            setup="Market shows low directional efficiency and price reaches a causal range extreme.",
            trigger="Failure to expand plus reclaim/flow exhaustion near range edge.",
            entry="Next-bar open after causal range-edge rejection trigger.",
            initial_stop="Outside range invalidation/liquidity extreme.",
            exit_logic="Value/range midpoint and opposing structure targets with fast invalidation when expansion begins.",
            sizing="Risk scaled by range-state confidence and distance-to-invalidation.",
            invalidation="Directional expansion or range break invalidates the setup.",
            target_min_trades=300,
        ),
        StrategyContract(
            strategy_id="ETH_STRATEGY_VOL_EXPANSION_V1",
            name="ETH Volatility Expansion V1",
            sleeve_id="ETH_SLEEVE_VOL_EXPANSION",
            family="volatility_expansion",
            strategy_class="core",
            stage="planned",
            symbol="ETH-USDT-SWAP",
            signal_timeframe="5m/15m",
            execution_timeframe="5m/15m",
            setup="Realised range/volatility compresses without using future range information.",
            trigger="Causal expansion plus directional price/flow impulse.",
            entry="Next-bar open after expansion confirmation.",
            initial_stop="Inside failed-expansion structure or opposite compression boundary.",
            exit_logic="Expansion failure, structural stop, R target and causal momentum decay protection.",
            sizing="Risk scaled by expansion probability and directional quality.",
            invalidation="Expansion fails, direction reverses or execution/risk cap is violated.",
            target_min_trades=300,
        ),
    )
    for row in rows:
        row.validate()
    return rows


__all__ = ["PORTFOLIO_V2_ID", "build_core_strategy_catalog"]
