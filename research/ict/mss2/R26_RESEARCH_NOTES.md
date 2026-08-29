# R26 Research Notes — Relative Positioning Leadership Repricing

Date: 2026-08-17

## Frozen hypothesis

R26 tested a standalone positioning-leadership transition. Binance top-trader
position long share crossing above global-account long share armed Long; the
downward mirror armed Short. The spread had to retain its new sign until the
first same-direction completed OKX 5m close through the prior 5m extreme,
within one hour. Entry used the next eligible OKX 1m open.

This was not an R18/R19 filter. Base OI, OI change, taker imbalance, liquidity
sweeps, funding, session, volatility regime, and learned thresholds were absent.

## Source and causality

- The process physically loaded only 2022-01-01 through 2025-06-30.
- No July 2025 or holdout row was loaded or evaluated.
- Visible Binance ratio rows: 262,341.
- Duplicate metric timestamps: 0.
- Top-trader position-share nulls: 35; global-account-share nulls: 27.
- Non-exact 5m intervals: 416; every affected edge was excluded.
- Maximum metric gap: 10.5 hours; no interpolation was allowed.
- OKX 1m rows: 1,838,880, with zero duplicate or non-minute intervals.
- Signal entry was the first 1m open at or after metric publication and price
  confirmation were both available.
- Same-bar target/stop ambiguity was stop-first. Later gap-through stop fills
  used the worse of the stop and bar open.
- Discovery and validation were independently boundary-censored and each
  direction/target simulation forbade position overlap.

The R26/MSS2 holdout has no event or economic output. Other repository projects
historically inspected later periods, so any eventual live approval still
requires a genuinely new forward seal; those external results are not R26
selection evidence.

Eighteen internal causal checks pass. A separate validator reconstructs ratio
crosses, retained-sign episodes, first price confirmation, cross-time target,
entry, stop, 1m first passage, gap fills, costs, split boundaries, and position
overlap directly from loader outputs. It passes 17 checks across all 634 visible
events and 1,944 target paths.

## Event funnel

The full warmup-plus-visible input contains 546 Long and 545 Short raw crosses.
There are 663 confirmed episodes, of which 634 signal in discovery or
validation and 486 have executable structural geometry.

| Split | Direction | Confirmed | Executable | Non-overlap structural trades | Median risk | Median delay |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Discovery | Long | 236 | 173 | 160 | 0.317% | 10m |
| Discovery | Short | 223 | 182 | 169 | 0.340% | 15m |
| Validation | Long | 85 | 60 | 59 | 0.451% | 15m |
| Validation | Short | 90 | 71 | 69 | 0.412% | 10m |

Most non-executable setups had already consumed the cross-time one-hour target;
this was frozen geometry, not an outcome filter.

## Primary structural-target result

| Split | Direction | Trades | Gross PF | 1x PF | 2x PF | Mean 2x return | Positive months at 2x | Top-5-removed 2x PF |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Discovery | Long | 160 | 1.22 | 0.68 | 0.39 | -0.183% | 16.7% | 0.25 |
| Validation | Long | 59 | 1.20 | 0.73 | 0.45 | -0.179% | 33.3% | 0.15 |
| Discovery | Short | 169 | 0.93 | 0.51 | 0.29 | -0.233% | 8.3% | 0.18 |
| Validation | Short | 69 | 0.95 | 0.61 | 0.40 | -0.232% | 16.7% | 0.19 |

Every visible year and direction loses at 2x cost. Discovery Long loses
11.43% in 2023 and 17.82% in 2024 on summed per-trade returns; Discovery Short
loses 16.35% and 23.05%. Validation Long and Short lose 10.56% and 16.02%.

The fixed-R diagnostic paths do not reveal an exit problem. Every 1R/2R/3R
cell has PF below one after 1x cost in both visible splits. The best-looking
gross cell, validation Short 2R at PF 1.25, falls to 0.93 at 1x and 0.70 at 2x.
Top-five and top-ten removal remain negative throughout.

## Interpretation

The relative-positioning cross is observable and sometimes precedes a gross
move, but the edge is too weak for market execution. Median risk is only
0.32–0.45%, so a 0.22% stressed round trip consumes much of the opportunity;
however, the primary Short path is already below one gross and Long gross PF is
only about 1.2. This is not merely a stop/target problem.

Frequency is adequate at roughly 6.7–11.5 structural trades per month, but it
does not translate into economic expectancy. Discovery longest entry gaps are
102–120 days, reflecting long stretches without a usable ratio crossing and
also failing to solve portfolio coverage.

## Frozen conclusion

Reject R26 Long and Short. No direction passes the precommitted gate; neither is
eligible for a strategy sleeve or holdout opening.

Stop the relative-positioning ratio branch. Do not tune spread magnitude,
replace the zero cross with quantiles, vary the one-hour confirmation window,
change price confirmation, add base OI/taker/session/volatility filters, select
a fixed-R exit, alter the structural stop/target, or use ML to rescue it.

R26 adds one useful negative result: complete positioning ratios do not solve
the missing-edge problem when reduced to a causal leadership-cross mechanism.
The master goal remains open, but the next study must use a genuinely different
economic state rather than another transformation of the same Binance metrics.
