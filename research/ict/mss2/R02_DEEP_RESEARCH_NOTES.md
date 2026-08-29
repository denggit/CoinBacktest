# R02 Deep Research Notes - Liquidity Targets, Stack Consumption, and ETH Timeframes

Date: 2026-08-15

This note records the external conceptual research used to shape R02. It does **not** replace empirical ETH testing. Any ICT concept is treated as a hypothesis until CoinBacktest data supports it.

## 1. ICT 2022: draw on liquidity is an objective, not just an entry label

### Episode 12 - market structure / draw on liquidity
Archived transcript material frames the core directional question as whether price is likely to move toward buy-side/sell-side stops or toward imbalance/rebalancing. It also distinguishes short-, intermediate-, and longer-term structure instead of treating every visible swing as equivalent.

R02 implication:
- do not define every swing as one equal liquidity unit;
- retain hierarchy/source timeframe/order confirmation;
- test whether a stronger opposite-side liquidity cluster is a better target than a fixed-R or short time exit.

### Episode 26 - tape reading practice
Archived transcript material shows a long idea after sell-side liquidity was already taken, with buy-side liquidity used as the next draw/objective. The example also discusses taking the bulk of the position around the nearer objective rather than requiring the most ambitious best-case target.

R02 implication:
- opposing liquidity is a legitimate candidate exit geometry to test;
- compare nearest level versus stronger pool / higher-timeframe targets;
- do not assume the farthest possible draw must be reached.

### Episode 39 - specific liquidity pool as objective
Archived transcript material repeatedly uses a specified liquidity pool as the objective/draw and emphasizes higher-timeframe levels subordinating lower-timeframe interpretation.

R02 implication:
- target selection should be frozen from the active book available at entry;
- target hierarchy should include 1H/4H/1D and multi-timeframe pools;
- context timeframe must be empirical rather than simply copying a 15m index template.

Source-quality note: the episode evidence above was checked against archived transcript/SRT mirrors of the ICT YouTube material. Treat transcript wording as archival evidence rather than an official specification/API.

## 2. ETH does not justify hard-coded US-index session assumptions

Hansen, Kim & Kimbrough, *Periodicity in Cryptocurrency Volatility and Liquidity*, Journal of Financial Econometrics 22(1), 2024 (published online 2022), documents systematic Ether/Bitcoin variation across day-of-week, hour-of-day and within-hour frequencies across centralized/decentralized venues.

R02 implication:
- clock effects are real enough to measure;
- their existence does **not** justify hard-coding NY cash open as an admission gate;
- compare Asia/London/New York/weekend/weekday after event construction, and require cross-year stability before using a time filter.

Jasiak & Zhong, *Intraday and daily dynamics of cryptocurrency*, International Review of Economics & Finance 96 (2024), also reports intraday/intraweek periodicity in cryptocurrency markets and links patterns to major market operating times.

R02 implication:
- traditional-market hours may influence crypto activity, but they should be tested as contextual variables, not treated as deterministic killzones.

## 3. R01 empirical bridge into R02

The strongest R01 follow-up result did not come from a classic MSS/FVG parameter. It came from changing the unit from individual swing rows to independent price pools consumed by the same impulse.

Using R01 raw outputs and 10bp price-pool merging:
- one consumed pool had approximately flat 60m long-side drift;
- two pools were stronger;
- three pools stronger again;
- >=4 pools showed a much larger long-side reversal response;
- after 180-minute event separation, the >=4-pool long sample still showed positive 60m/180m average paths, but 2026 H1 weakened.

This is the direct empirical reason R02 studies **liquidity stack exhaustion** instead of optimizing FVG width/displacement thresholds.

## 4. R02 falsifiable hypotheses

R02 should reject or weaken the idea if any of these fail:

1. Pool-count effect disappears when events are deduplicated into causal sweep episodes.
2. Effect exists only at one arbitrary 10bp clustering tolerance and not neighboring 5/20bp definitions.
3. Effect is only 2023-2025 and becomes strongly negative in 2026 rather than merely weaker.
4. The forward path exists, but every causal reclaim/MSS entry occurs too late to cover costs.
5. Opposing liquidity targets are unavailable too often or produce excessive censoring.
6. Structural stops require so much risk that target R-multiples are unattractive.
7. Long-side effect does not survive independent-event spacing.
8. A supposed session advantage is one-year/hour specific.

## 5. Why R02 does not optimize partial exits yet

ICT examples can include taking bulk profit at an intermediate liquidity objective and leaving a runner. R02 intentionally postpones that management layer.

First identify whether one of these target geometries has a stable expectancy/availability hierarchy:
- nearest opposing level;
- >=2-level opposing pool;
- >=2-level, >=2-timeframe opposing pool;
- 1H+;
- 4H+;
- 1D+.

Only after that hierarchy is frozen should a later version test partial-at-near-liquidity + runner-to-stronger-liquidity, break-even migration, or trailing structure. This avoids fitting multiple management degrees of freedom simultaneously.
