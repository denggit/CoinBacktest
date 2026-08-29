# R21 Precommitment — Canonical Daily Channel Trend Following

Date frozen: 2026-08-17, before R21 outcomes are calculated.

## Why this is independent

Repository audit already rejects intraday compression breakout, expansion
exhaustion, impulse continuation, post-1m CVD confirmation, and Range-Bar
activity as directional entries. R17 rejects a multi-timeframe pullback
re-acceleration sequence, and R20 rejects V10B's unlevered 4H components.

R21 changes the horizon and mechanism. It tests canonical daily time-series
trend following with no order flow, OI, ICT sweep, reclaim, micro filter,
quality multiplier, add-on, leverage, or optimized structural stop.

## Frozen models

Primary `D20_X10`:

- Long entry: completed daily close above the prior 20 complete daily highs.
- Short entry: completed daily close below the prior 20 complete daily lows.
- Entry: next calendar-day 00:00 1m open.
- Initial stop: 2.0× completed daily ATR(20) from entry, fixed for the trade.
- Long exit: completed daily close below the prior 10 complete daily lows,
  executed at the next day open.
- Short exit: completed daily close above the prior 10 complete daily highs,
  executed at the next day open.

Canonical sensitivity `D55_X20` changes only entry/exit channels to 55/20;
ATR and stop remain identical. It is diagnostic and cannot replace a failed
primary after outcomes.

Long and Short are simulated as separate sleeves. Each permits one position,
no pyramiding, and no time-profit exit. A stop touched after entry exits at the
first causal 1m touch; a gap through the stop exits at the worse 1m open.

## Time and split contract

- Warmup: 2022-01-01 onward.
- Discovery: independent simulation reset at 2023-01-01 and ends before
  2025-01-01.
- Validation: independent simulation reset at 2025-01-01 and ends before
  2025-07-01.
- An open position at a split boundary is censored, not force-profited.
- The completed June 30 daily bar is unavailable until the July boundary, so
  it cannot signal a validation entry.
- July is embargoed; holdout begins 2025-08-01. No holdout outcome is loaded.

## Costs and return unit

- Gross return is the unlevered signed entry-to-exit price return.
- 1×/2×/3× round-trip costs are 0.11%/0.22%/0.33%.
- No compounding, leverage, or risk sizing enters the primary PF.
- Same-bar ambiguity is stop-first; this strategy has no fixed target.

## Required reporting

For model, split, and direction separately:

- trades and trades/month;
- gross and 1×/2×/3× PF and expectancy;
- win rate, holding duration, risk distance;
- positive months, longest entry gap, flat-duration distribution;
- yearly results;
- top-five/top-ten winner removal;
- exact signal/entry/exit/stop/cost causal audit;
- manual recent, best, worst, and yearly samples.

## Decision rule

The primary direction is only a research candidate if discovery and validation
both have positive 2× expectancy and PF >=1.4, discovery has at least 8 closed
trades, validation at least 2, discovery top-five removal remains positive, and
the same direction in `D55_X20` has positive discovery expectancy. These are
minimum evidence gates, not live approval.

No parameter change, trend filter, volatility filter, trailing stop, add-on,
target, or holdout result may rescue a failure.

