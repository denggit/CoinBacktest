# R02.2 Delivery — First-Touch Relative Liquidity Ranking

## Goal

Remove R02.1 exposure-time and absolute-threshold contamination before adding any new data family.

## Primary question

At one decision time and one side of ETH price, which of the complete 25 candidate price zones will release the most liquidity during a fixed window after its **first actual touch**?

## Frozen implementation

- Exact first touch: 1m first-crossing minute -> exact 1s crossing.
- Label windows: 30/60/180/300 seconds after first touch.
- Primary target: R01.1 release-density sum over first 180 seconds.
- Ranking group: `decision_time x zone_side`.
- Primary ranker excludes Swing and Touch probability.
- Full-with-Swing ranker exists only for ablation.
- Distance is a mechanical baseline, not a learned pool model.
- No q80/q90 absolute pool threshold.
- No order placement or stop tuning.

## Command

```bat
python research\eth_ai_trading\eth_latent_liquidity_path_v1\02_2_first_touch_relative_liquidity_ranking.py
```

## Report directory

`data\reports\research\eth_ai_trading\eth_latent_liquidity_path_v1\02_2_first_touch_relative_liquidity_ranking`

## Important outputs

- `04_ranking_metrics.csv`
- `05_top_zone_summary.csv`
- `06_first_touch_horizon_profile.csv`
- `09_swing_ablation.csv`
- `10_distance_first_touch_profile.csv`
- `11_causal_audit.csv`
- `13_decision.md`
- `gpt_review_pack.zip`

## Work-log requirement

Every later cumulative patch must preserve this stage's design, actual result, failed branches and next decision in `CUMULATIVE_STAGE_RESULTS.md`.
## Validation

- R02.2 dedicated: 8 passed with `-W error`.
- R01 -> R02.2 focused regression: 71 passed.
- All AI Research: 313 passed.
- AI Research + Data Feed + Research Common with Runtime/FutureWarning promoted to errors: 336 passed.
- New import-boundary violations: 0.
- Full repository collection: 639 tests discovered; blocked only by the same 5 pre-existing Liquidity / Analyze Tool missing-module errors.

## Performance contract

The implementation must retain the algorithmic fast path: 1m first-touch minute search, 1s refinement only inside that minute, NumPy prefix-sum post-touch labels, indexed Episode aggregation, chunk cache / resume, and no row-by-row full-history Pandas loops.

