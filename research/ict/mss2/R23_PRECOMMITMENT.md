# R23 Precommitment — Frozen Panic-Wick Structural Long Falsification

Date frozen: 2026-08-17, before R23 visible-split outcomes are calculated.

## Provenance warning

The historical shadow-wick branch used the full 2023–2026 window to inspect
many wick/volatility/flow/session buckets, three entry policies, nine V1 exit
modes, delays 0/1/2, and a later seven-mode exit upgrade ladder. Its reported
`priority_union + multi_sweep_deeper_higher_low_trail + delay2` result (about
332 trades, PF 1.58, +54.65%, four positive years) is therefore a contaminated
prior, not untouched evidence.

R23 freezes that one existing rule. It will not compare or select any alternate
entry, threshold, session, delay, exit, stop, target, or hold variant.

## Frozen entry

Use 1m trade-derived bars from `src.data_feed.OKXTradeBarLoader`.

- Lower-wick share >= 0.50 and lower wick >= 0.55× rolling 60m mean true range.
- Volume >= 2.0× the prior 240m median.
- Volatility regime is mid-high or extreme using the historical fixed ATR/price
  bins.
- Prior flush: 30m return <= -0.5% or 120m return <= -1.0%.
- Trend is down: close below EMA(240) and 60m EMA slope < -0.05%.
- `strict_flow`: close location >=0.55 and delta ratio <=-0.10 or taker-buy
  ratio <=0.45.
- `strict_reclaim`: close location >=0.66.
- `priority_union`: exact-minute union, with flow event priority.
- The event minute must be source-observed and the full prior 240-minute feature
  window must have source bars.
- Historical delay two means entry at the open three minutes after the closed
  event bar: the normal next minute plus two additional bars.

## Frozen exit

Only `multi_sweep_deeper_higher_low_trail` is allowed.

- Track distinct excursions below the event low.
- After at least two excursions, exit at the next 1m open if the current low is
  more than 0.15% below the event low and its close remains below the event low.
- Once the event high is reclaimed on a close, initialize protection at the
  event low and raise it only to causally confirmed three-bar higher lows above
  the event low.
- Exit at the next 1m open after a close below the active trail.
- No fixed target, time-profit exit, add-on, leverage, or sizing.

## Data-gap and split policy

The local trade-bar source has 1,837,343 of 1,838,880 visible minutes and 72
gaps, including one 10h12m gap. R23 reindexes the calendar axis with flat,
zero-volume placeholders only to preserve clock semantics. Synthetic minutes
cannot form events or entries. Any open path reaching a missing source minute
is censored and the sleeve restarts only after source data resumes.

- Discovery resets at 2023-01-01 and ends before 2025-01-01.
- Validation resets at 2025-01-01 and ends before 2025-07-01.
- Boundary-open positions are censored.
- July and the 2025-08-01 holdout are not loaded.

Gross return is the unlevered Long entry-to-exit move. Round-trip 1×/2×/3×
costs are 0.11%/0.22%/0.33%.

## Decision rule

The frozen Long rule only remains a research candidate if:

- discovery and validation 2× PF are each >=1.4 with positive expectancy;
- discovery has at least 100 closed trades and validation at least 20;
- 2023, 2024, and 2025H1 each have positive 2× sum;
- discovery top-ten-winner removal remains positive;
- data-gap plus split censoring is <=5% in each split; and
- no causal, raw-entry, next-open-exit, cost, or holdout audit fails.

Passing would justify forward incubation, not live approval. Failure archives
the shadow-wick shortcut without session, flow, wick, or exit rescue.

