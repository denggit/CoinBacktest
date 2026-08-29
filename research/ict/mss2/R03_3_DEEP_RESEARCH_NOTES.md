# R03.3 Deep Research Notes — ICT Hierarchy, Entry/Exit, CVD

Date: 2026-08-15

## Why R03.3 exists

R02/R03.2 showed that raw `N pools` is useful but incomplete. The strongest observed long-side cohort was associated with multi-pool sell-side consumption and 4H+ involvement, while generic footprint expansion did not rescue the broader >=3 cohort. The user also correctly questioned whether all pools should be treated equally, whether ICT STH/STL -> ITH/ITL -> LTH/LTL hierarchy had been represented correctly, whether MSS is mandatory, and whether entry/exit target families and CVD should be studied separately.

R03.3 therefore changes the research question from:

`How many pools were swept?`

to:

`What kind of liquidity was swept, how many additional pools were consumed around it, how fast was that consumption, what causal confirmation was used to enter, and where was the opposing structural draw?`

## ICT source distillation used for this version

Primary source material reviewed for the mechanical hierarchy comes from ICT 2022 Mentorship Episodes 11 and 12 transcripts/notes.

### Swing-on-swing hierarchy

The source material describes nested market structure rather than a fixed pivot-order score:

- a short-term swing is the local building block;
- an intermediate-term high/low is formed by the relationship of neighboring short-term highs/lows;
- a long-term high/low is recursively formed from neighboring intermediate-term highs/lows.

R03.3 implements only the part that can be frozen mechanically and causally:

- ST = every already-confirmed base swing candidate in the existing R02 lifecycle;
- ITH = an ST high above its immediate left/right ST highs; ITL is mirrored;
- LTH = an ITH above its immediate left/right ITHs; LTL is mirrored;
- an IT label is usable only after the right ST is itself confirmed;
- an LT label is usable only after the right IT is itself confirmed;
- the sweep decision uses the **start of the 1m sweep bar** as the knowledge cutoff.

This is intentionally separate from the older `confirmed_order_at_sweep` / order-1/2/3/5 pivot taxonomy. Both can remain in reports, but they must never be treated as synonyms.

### Episode-12 FVG-rebalance alternative

Episode 12 also presents a second way of recognizing an intermediate-term swing around the rebalance of an imbalance/FVG. R03.3 documents this but does **not** silently approximate it. A future version may implement it only after the exact causal definition of the relevant imbalance, rebalance event, and resulting swing is frozen and unit-tested.

### MSS is not universally required

Episode 12 includes an aggressive-entry example where the broader market-structure premise is considered sufficient and a lower swing break is not required before entry. Therefore R03.3 does not assume MSS is mandatory. It compares the already-existing causal R02 entry families on identical hierarchy-defined stages:

- stage reclaim;
- episode reclaim;
- structural MSS market;
- structural MSS + FVG limit.

This does **not** claim that the project-defined reclaim trigger is literally an ICT named setup. It is a causal non-MSS comparator.

### Order block / FVG context

The source material also uses higher-timeframe order-block context and lower-timeframe FVG/imbalance entries. R03.3 does not introduce a new order-block detector because the exact candle/range semantics would add another large degree of freedom. The corrected post-reclaim FVG overlay is treated as execution research only. If that is useful, a later research version can freeze one exact order-block/FVG-overlap definition before testing it.

## Pool-quality decomposition

Each cumulative 10bp price pool is annotated using information available by the sweep:

- strongest ICT swing rank: ST / IT / LT;
- max source timeframe;
- number of source timeframes represented;
- multi-timeframe flag;
- 4H+ flag;
- external-50 flag;
- clean-first-sweep flag;
- pretested history remains visible at the level/stage layer.

No fitted score is used. `structural_key` is only a descriptive categorical union (`IT+ OR 4H+ OR multi-TF`) and is never treated as a learned ranking.

Two complementary analyses are produced:

1. first-crossing hierarchy cohorts (first IT, first LT, first 4H+, etc.);
2. **fixed-N decomposition**: at the same N=1/2/3/4 raw-pool crossing, compare outcomes with and without IT/LT/4H+/multi-TF/external/clean liquidity.

The second analysis is important because it prevents a misleading conclusion such as "4H is better" when 4H simply happens to occur later in a larger exhaustion episode.

## Entry and exit research

### Entries

R03.3 compares, by direction and 1m/2m/5m execution timeframe:

- `stage_reclaim`;
- `episode_reclaim`;
- `mss_structural_market`;
- `mss_structural_fvg_limit`.

The old R03 finding that `stack -> first FVG` by itself was weak remains a separate failed/weak alternative trigger, not evidence against FVG as an execution tool after a valid reclaim.

### Exits

The same hierarchy-defined cohorts compare:

- nearest opposing active liquidity;
- >=2-level opposing pool;
- opposing multi-timeframe pool;
- opposing 1H+ liquidity;
- opposing 4H+ liquidity;
- opposing 1D+ liquidity;
- 2R / 3R / 5R diagnostic controls.

No fixed 60m/180m profit exit is reintroduced. The existing R02 7-day boundary remains censoring, not a forced close.

## CVD extension

CVD is **not presented as an ICT 2022 rule**. It is a separate ETH order-flow extension.

R03.3 rebuilds episode-anchored CVD from causal 1m trade-bar `delta_notional`, instead of trusting a stored cumulative CVD whose origin can depend on cache/read boundaries. At each decision it records:

- episode CVD end/minimum/recovery;
- recovery ratio;
- 3m / 5m / 15m bullish price-vs-CVD low divergence;
- existing trade-bar absorption fields.

Only completed trade bars strictly before the decision are eligible.

## R03.2 execution-overlay bug corrected here

The user's real R03.2 report contained only `reclaim_market` rows and no FVG variants. Two issues were found:

1. R03.2 converted a datetime Series to integer and compared it with `Timestamp.value`; pandas environments can expose different datetime integer units (for example microseconds versus nanoseconds), which caused the FVG search start to fall outside the execution frame.
2. opportunities with no 4H target were skipped entirely, reducing 269 frozen core opportunities to 266.

R03.3 fixes both:

- FVG search uses `DatetimeIndex.searchsorted(Timestamp)` directly, with no integer-unit conversion;
- every frozen core opportunity is preserved even when its 4H target is unavailable;
- each core opportunity must have exactly four execution variants for each 1m/2m/5m FVG timeframe;
- each timeframe must contain non-zero FVG signal rows or the script hard-fails;
- expected overlay rows = `core opportunities x 3 FVG TFs x 4 variants`;
- R02 reclaim-market outcome/gross-return tie-out is mandatory before execution results are accepted.

Execution variants remain:

1. original reclaim market;
2. post-reclaim FVG-confirmed market;
3. post-reclaim FVG proximal limit;
4. 50% reclaim market + 50% FVG proximal limit.

All variants keep the same frozen R02 structural stop and opposing 4H target. No target is reselected after waiting for the FVG.

## Anti-overfit interpretation rules

- Do not select a new pool-quality rule because one subgroup has the highest PF.
- Require directionally consistent yearly/forward behavior and adequate sample size.
- Fixed-N quality decomposition is descriptive; it does not authorize threshold mining.
- CVD/Trade-Bar features need forward stability; a train-only uplift is rejection evidence.
- Footprint is not expanded further in R03.3 because corrected R03.2 did not show stable forward uplift.
- NY Open is not an admission gate.
- Long/short are reported separately; symmetry is never assumed.

## R03.3.1 addendum — post-sweep MSS + open-form displacement research

The user correctly identified two remaining modeling gaps.

### Post-sweep newly formed ST swing MSS

Earlier R01/R02 MSS references were deliberately restricted to pivots known before the sweep execution bar began. That is causally safe, but incomplete: after sell-side liquidity is swept, price can first create a new small execution-timeframe STH and only later break that STH. R03.3.1 adds a third reference mode, `post_sweep_st`, without changing the older modes.

For bullish research the sequence is now:

1. liquidity sweep stage occurs;
2. a new execution-TF ST high has pivot position strictly after the sweep execution bar;
3. the right-hand confirmation bar closes, making that ST high causally available;
4. only a later eligible execution bar may close above the latest currently-known post-sweep ST high;
5. market entry remains next eligible 1m open; FVG-limit variants remain later execution choices.

The bearish path is mirrored with a newly formed post-sweep ST low. `recent`, `structural`, and `post_sweep_st` are reported separately so their payoff cannot be conflated.

### Displacement is not hard-coded

No displacement threshold is used for admission. R03.3.1 records a feature family instead:

- displacement ATR distance;
- ATR-per-minute speed;
- path efficiency;
- leg range / ATR;
- maximum directional candle body / ATR;
- directional-body share;
- break distance / ATR;
- MSS bar body and body ratio;
- FVG count, density and maximum width;
- pre-sweep attack displacement, speed and efficiency;
- reversal/attack distance ratio;
- reversal/attack speed ratio.

The reversal is explicitly **not required** to be stronger or faster than the attack into the extreme. 2023-2024 quartiles are frozen and applied to 2025-2026 so the report can reveal monotonic, non-monotonic or middle-strength payoff shapes rather than assuming that the strongest displacement is best.

A dedicated relative-strength table bins reversal/attack ratios below and above 1.0 so weaker-than-attack reversals can be evaluated directly.
