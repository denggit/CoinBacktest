# R02.2 Post-run Review — 2026-08-07

## Status

Raw first-touch release-density ranking is retired as the primary latent-pool objective.

## Why

- Holdout PATH_NO_SWING Spearman: 0.041 DOWN / 0.074 UP.
- Nearest-distance baseline: 0.174 DOWN / 0.156 UP.
- Raw first-touch density remains mechanically larger close to current price.
- Path-selected farther zones nevertheless showed materially better favorable-vs-continuation behavior than the nearest 10bp baseline.

## Quality audit correction

The 8,282 `first_touch_time <= decision_time` flags were caused by auditing 1-second **bar start timestamps** as if they were availability timestamps. Correct semantics are `first_touch_available_time = first_touch_time + 1s`.

The 22 exact-touch / old-R02-touch disagreements are not dismissed; R02.3 quarantines them from modeling and reports them explicitly.

## Swing

No stable cross-period incremental value. Remains ablation-only.

## Next modeling question

Use TRAIN-only distance-normalized Excess Liquidity, separate Reversal Quality, and separate Sweep Geometry. Do not add Range/Footprint/OI until the corrected target is tested.
