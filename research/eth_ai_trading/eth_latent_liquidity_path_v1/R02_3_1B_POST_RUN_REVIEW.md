# R02.3.1b Post-Run Review

## Decision

`BLOCKED_R02_3_1B_TARGET_CONSISTENCY_STILL_DISTANCE_CONTAMINATED`

## What this run proved

The R02.3.1 Excess target did contain a real scale mismatch, but fixing it did **not** fix the research problem.

Corrected absolute distance correlation of the primary L2 mean-aligned residual:

- DOWN Validation: **0.090**;
- DOWN Holdout: **0.140**;
- UP Validation: **0.149**;
- UP Holdout: **0.203**.

The frozen 0.12 gate therefore fails in DOWN Holdout and both UP future periods.

## Huber versus L2 conclusion

The positive-log Huber -> L2 mean change was small: mean absolute objective gaps were roughly **0.005-0.009**. Do not open another stage to tune the regression objective.

## Dominant finding — release-regime nonstationarity

Release-hurdle AUC:

- TRAIN DOWN/UP: **0.691 / 0.689**;
- Validation: **0.513 / 0.511**;
- Holdout: **0.513 / 0.496**.

Actual release rate:

- 2023 DOWN/UP: **7.1% / 9.5%**;
- 2024: **27.9% / 29.1%**;
- 2025: **43.4% / 43.5%**;
- 2026: **44.9% / 42.5%**.

The 2023-2024 background therefore no longer describes 2025-2026 well enough to support a stable frozen residual target.

## Causal status

All 16 R02.3.1b causal/source checks passed, including:

- complete lattice;
- feature availability;
- exact first-touch availability;
- period separation;
- nuisance feature restrictions;
- no Swing / no zone-specific path in nuisance;
- TRAIN expanding past-only nuisance prediction;
- Validation/Holdout full-TRAIN-frozen nuisance;
- exact same-scale hurdle expectation identity.

This is not a future-function failure.

## Research decision

Stop the R02.3.1 target-rescue loop. Do not create 1c/1d by tuning loss, thresholds or Swing.

Next stage is R02.4 Economic Ceiling Audit. It asks whether the true favorable release mechanism has enough ideal post-cost MFE/reward-risk to justify any more identification work at all.
