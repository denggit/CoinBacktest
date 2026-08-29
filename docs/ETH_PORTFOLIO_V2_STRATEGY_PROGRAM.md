# ETH Portfolio V2 — Strategy Program

## Frozen goal

Build one ETH-USDT-SWAP perpetual portfolio that can progress through:

`complete strategy -> backtest -> robustness -> portfolio -> AetherEdge shadow/live -> copy trading`.

Research-only edge evidence is never the final deliverable.

## Core sleeves

1. Trend Breakout
2. Trend Pullback
3. Liquidity Reversal
4. Range Mean Reversion
5. Volatility Expansion

Each core sleeve must define setup, trigger, entry, initial stop, exit logic,
sizing and invalidation before it is eligible for portfolio work.

## Anti-funnel rule

Quality variables should default to score/sizing inputs.  A hard filter that
retains less than 50% of the previous stage is explicitly surfaced; less than
20% fails the default core funnel gate.  Core sleeves also fail the default
frequency gate below 300 executed trades over the standard test window.

This does not prohibit rare-event sleeves.  They must be declared as rare-event
strategies instead of pretending to be a core source of portfolio frequency.

## Standard time/cost window

- Warmup: 2022-01-01
- Backtest: 2023-01-01 through 2026-06-30
- Instrument: ETH-USDT-SWAP
- Default round-trip fee: 0.11%
- Slippage is additional
- Closed-bar signals only; next-bar execution
- Higher-timeframe context must use its real available time

## Current stage

### 01 Framework + Funnel Gate

Implemented in:

- `src/strategy_common/`
- `src/portfolio_common/strategy_catalog.py`
- `backtest/portfolio/eth_portfolio_V2_framework_audit.py`

### 02 Trend Breakout V1

Implemented immediately rather than stopping at framework work:

- `src/sleeve_lib/trend_breakout_v1/`
- `backtest/mf/trend_breakout/eth_trend_breakout_v1_backtest.py`

Trend Breakout V1 deliberately keeps the structure-break universe broad.
Trend alignment, breakout depth and candle quality change risk size instead of
being stacked as hard filters.

## Local commands

Framework audit:

```text
python backtest/portfolio/eth_portfolio_V2_framework_audit.py
```

Trend Breakout V1 full run:

```text
python backtest/mf/trend_breakout/eth_trend_breakout_v1_backtest.py --symbol ETH-USDT-SWAP --warmup-start-date 2022-01-01 --start-date 2023-01-01 --end-date "2026-06-30 23:59:59"
```

The strategy reads only through `src.data_feed.OKXTradeBarLoader`.  It does not
implement or bypass the project's data interfaces.
