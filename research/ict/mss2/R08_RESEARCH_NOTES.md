# R08 — Full-Trend ICT Structure Atlas

## Why this version exists

Manual review exposed a foundation risk: the existing fixed-order / recursive swing taxonomy can still produce too many locally valid pivots that do not match the larger ICT structure a discretionary trader would call the important historical liquidity.

R08 therefore pauses strategy optimization.  Its only objective is to verify that CoinBacktest can reconstruct ICT market structure before those levels are reused as SSL/BSL.

## Frozen source interpretation

Classical hierarchy is built recursively on each chart timeframe:

- Short-term high/low: a strict three-bar swing.
- Intermediate-term high: a short-term high more extreme than the immediately adjacent short-term highs; ITL is the mirror.
- Long-term high: recursive extreme on the ITH sequence; LTL is the mirror.

ICT 2022 Episode 12 also discusses imbalance-rebalance swings as intermediate-term structure.  R08 deliberately does **not** silently merge that discretionary extension into the classical hierarchy.  It must be added later as a separately auditable class if the classical labels first pass chart review.

## Full-trend requirement

R08 does not call a rolling-window maximum/minimum a Long-Term Swing.  It first uses classical LT anchors, then constructs an entire opposite-LT leg.  A leg is considered clean only when:

1. both IT-high and IT-low sequences contain enough structure to evaluate;
2. bullish legs have rising ITHs and rising ITLs; bearish legs are the mirror;
3. after the terminal LT anchor, price closes through the latest opposing IT level (IT-BOS), confirming that the prior directional leg ended;
4. the whole move is at least the requested research scale (3%, 5%, or 7%).

The 3/5/7% values are CoinBacktest scale sensitivities, not ICT canonical thresholds.

## Causality

The leg may only qualify historical liquidity after all of the following are known:

- origin LT label;
- terminal LT label;
- BOS-reference IT label;
- the reversal BOS close.

No label is backfilled into earlier timestamps.  An individual historical IT level whose recursive confirmation occurs later than the leg-level confirmation activates only when its own IT label is also known.

## Which swings become future liquidity

ST swings are construction-only.

For a clean completed bearish leg, future BSL candidates are:

- the trend-origin LTH;
- internal ITH retracement highs.

For a clean completed bullish leg, future SSL candidates are:

- the trend-origin LTL;
- internal ITL retracement lows.

The last IT level broken by the reversal BOS is normally already consumed and therefore cannot remain active future liquidity.  R08 explicitly checks consumption before activation and excludes consumed levels from the active manual-review list.

## Required manual review before any strategy reuse

Do **not** promote R08 levels into R09 trading logic until manual chart review agrees with the labels.

Start with:

`manual_review/01_recent_30_completed_clean_trend_legs.csv`

Check the full origin-to-terminal move on the stated source timeframe.  The file includes complete ITH/ITL sequences and the BOS that ended the leg.

Then inspect:

`manual_review/02_recent_60_active_key_liquidity_levels.csv`

These are the only trend-qualified IT/LT levels R08 would allow into the next liquidity study.

## Next step only if manual review passes

Rebuild SSL/BSL sweep research using this trend-qualified liquidity universe.  Compare 3/5/7% historical-leg scales and do not reintroduce ST liquidity merely to increase trade count.

## R08.1 projection taxonomy correction

R08 report review found that `build_trend_qualified_liquidity` selected any IT swing whose pivot time fell inside a completed trend leg, regardless of swing timeframe. This mixed three materially different objects under the trend timeframe label:

1. `native`: swing timeframe == completed trend timeframe. This is now the canonical full-trend ICT liquidity set.
2. `nested_lower_tf`: lower-timeframe causally confirmed IT/LT swing inside a completed higher-timeframe trend. Kept separately for research; never mixed into canonical counts.
3. `invalid_higher_tf_projection`: higher-timeframe swing projected into a lower-timeframe trend. Rejected from future key-liquidity use.

The canonical `05_trend_qualified_key_liquidity.csv.gz` is native-only in R08.1. Nested lower-TF liquidity is written to `05b_nested_lower_tf_liquidity.csv.gz`; rejected projections to `05c_rejected_higher_tf_projection.csv.gz`. Summaries now report physical unique levels separately from context rows.

### Preliminary effect audit from already-generated reports

Before rerunning R08.1, the old R08 sweep timestamps were matched to R01 sweep-forward labels to estimate whether the taxonomy correction is likely to improve raw directional edge. This is a bridge audit only (not the final R08.1 direct bare-K result) and is stored in `R08_1_PRELIMINARY_EFFECT_AUDIT.csv`.

Key 2x-cost sweep-only observations:
- Native SSL remained positive: 60m PF about 1.28, mean net about +0.145%; 180m PF about 1.26, mean net about +0.184%.
- Native BSL remained negative: 60m PF about 0.74; 180m PF about 0.93.
- Nested lower-TF SSL inside completed higher-TF trends was substantially stronger in this preliminary bridge: 60m PF about 2.68, mean net about +0.708%; 180m PF about 2.03, mean net about +0.695%.
- Therefore R08.1 does NOT assume native-only will maximize profitability. The correction is semantic first: native and nested must be separated. Nested lower-TF structure remains an explicit candidate taxonomy because it may carry real SSL edge.
- The final R08.1 run computes direct next-bar-open sweep impact from bare 1m K at 1h/3h/6h/12h/1d and 1x/2x/3x costs, avoiding dependence on the legacy R01 match bridge.
