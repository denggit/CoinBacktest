# R06 — Adaptive Risk + Protected Position Lifecycle

## Why R06 exists

R05 established that the project should stop chasing a single ultra-strict setup. The broad Long `N>=3 + (4H OR LT)` family already offers a useful opportunity rate, while structural trailing can preserve a positive 2x-cost edge across years. R06 asks whether that edge can become a *portfolio-quality trading engine* with smoother capital growth.

This study is intentionally not a new feature-search round. It freezes the entry family and changes only risk / lifecycle mechanics.

## Base universe

Only the first causal `n3_4h_or_lt` stage per episode from R05 is used. 1m/2m/5m episode-reclaim entries are all retained for sensitivity.

Initial quality tiers are causal and ordinal:

- **B**: the first N>=3 key-liquidity stage contains exactly 3 independent pools;
- **A**: the same first qualifying stage itself jumps directly to >=4 pools;
- **A+**: that same fast >=4 stage already contains both 4H and LT liquidity.

A later N=4 stage can never upgrade the initial risk of an earlier N=3 entry.

## Anchor formation != stop promotion

R05 moved stops immediately when an ITL/LTL became knowable. R06 adds a delayed-protection model.

For each 5m LTL and 15m ITL/LTL:

1. Wait until the recursive swing hierarchy makes the low causally knowable.
2. Freeze a confirmation high using only HTF bars closed by that availability time.
3. Wait for a *later* HTF close above that frozen high.
4. Only then promote the low into an active protected stop anchor.
5. The stop is active from the first eligible 1m bar at/after the confirming HTF close and may only move upward.

This models the user's practical observation that an ITL may form but should not necessarily be trailed immediately; the market can be given room until that low becomes demonstrably protected.

A 15m q95 bullish displacement bar that also creates a bullish FVG is retained as a separate immediate protected anchor candidate from its close onward. No 1m trailing exists.

## Management variants

The comparison set is deliberately small:

1. `r05_immediate_ltl5` — old R05 baseline;
2. `protected_ltl5` — delayed 5m LTL promotion after HH confirmation;
3. `protected_ltl5_or_shock15fvg` — delayed 5m LTL plus 15m shock/FVG anchors;
4. `protected_ltl5_then_itl15_major` — 5m protected structure until the trade causally reaches +3%, then only promoted 15m ITL/LTL anchors;
5. `protected_ltl5_then_ltl15_major` — same, but major state uses only promoted 15m LTL.

The +3% milestone is not a TP. It only changes management state and is observable causally when price reaches it.

## Add-on semantics

Only one add-on is allowed in the research variant:

- must occur after entry;
- must be caused by a `protected_ltl_5m_hh` promotion;
- add price must be at/above the original entry (no averaging down);
- common stop is the already-promoted structural stop;
- base open risk to that stop is recomputed;
- add-on notional uses only remaining setup risk capacity;
- total open risk after add-on must remain <= the setup's risk budget;
- total notional is capped.

This is risk recycling/pyramiding after confirmation, not martingale averaging.

## Risk schedules

Three fixed diagnostic schedules are reported. They are not optimized:

- `equal_1pct`: B/A/A+ all 1.0% equity risk budget;
- `tiered_conservative`: 0.75% / 1.00% / 1.25%;
- `tiered_full`: 1.00% / 1.50% / 2.00%.

Position notional is derived from structural stop distance and capped at 3x equity. Cost stress is 1x/2x/3x the 0.11% round-trip convention.

## Portfolio semantics

ETH is treated as one net Long position for this research family. While one base setup remains open, a later independent episode is skipped and recorded in the overlap audit. The internal R06 add-on does not count as a new independent base position.

No time stop is used. An unresolved trade may remain open until the market data ends; that is right-edge censoring.

## What “smooth equity” means in R06

PF is insufficient. The portfolio scorecard includes:

- executed trades/month after overlap;
- daily mark-to-market maximum drawdown;
- longest drawdown duration;
- Ulcer index;
- log-equity linear-trend R²;
- positive month and positive quarter rates;
- median/worst month;
- rolling-90d positive-return share;
- market exposure;
- maximum days between executed entries;
- consecutive-loss streak;
- add-on usage;
- final equity after zeroing top 5 and top 10 winners;
- per-year return / MDD / positive-month rate.

The desired candidate is not necessarily the scenario with highest terminal equity. Prefer a robust upward curve across years and cost stresses with limited concentration in a few tail winners.

## Causality / leakage rules

- No future N=4 backfill into initial risk tier.
- Structural low must be causally known before protection logic starts.
- HH promotion must occur strictly after candidate availability.
- Add-on must occur strictly after entry.
- No 1m trailing structure.
- Stop only ratchets upward.
- +3/+5/+10% are future diagnostics or real-time state milestones, never entry features.

## Engineering

R06 daily MTM is O(days + trades) per scenario. It does not repeatedly filter the full trade table for each day. All empty aggregations are checked before median/quantile calculation; the end-to-end smoke passes with RuntimeWarning promoted to an exception.
