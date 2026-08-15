# R02.3.1 Post-Run Review — 2026-08-08

## Report reviewed

`02_3_1_hurdle_nuisance_residualization`

Formal decision:

`BLOCKED_R02_3_1_RESIDUALIZATION_NOT_REMOVED`

This document freezes the empirical result separately from the next code change.

## What passed

All 17 causal/source checks passed:

- exact first-touch availability remained strictly after decision time;
- no old R02 touch-mismatch row entered train/evaluation;
- period-boundary overlap rows were purged;
- TRAIN nuisance predictions were expanding past-only OOS;
- Validation/Holdout nuisance predictions used full-2023-2024-TRAIN frozen models;
- nuisance inputs were group-level activity/calendar or raw distance only;
- primary rankers excluded raw distance, nuisance activity and Swing.

Therefore the stage failure is not attributed to a discovered lookahead/timestamp error.

## Excess residualization result

Absolute distance correlation, raw density -> Excess Residual:

| Side | Period | Raw | Residual | Interpretation |
|---|---|---:|---:|---|
| DOWN | Validation | 0.1482 | 0.0886 | improved / within limit |
| DOWN | Holdout | 0.1611 | 0.1362 | improved but above 0.12 |
| UP | Validation | 0.1354 | 0.1519 | worsened |
| UP | Holdout | 0.1729 | 0.2074 | materially worsened |

The current Excess target is not distance-clean and cannot be promoted.

## Reversal residualization result

Absolute distance correlation, raw Reversal Quality -> Reversal Residual:

| Side | Period | Raw | Residual |
|---|---|---:|---:|
| DOWN | Validation | 0.2706 | 0.0619 |
| DOWN | Holdout | 0.2587 | 0.0967 |
| UP | Validation | 0.2387 | 0.0666 |
| UP | Holdout | 0.2448 | 0.0830 |

The mechanical nuisance removal worked materially better for Reversal Quality. However the post-residual PATH signal was weak in Holdout (DOWN 0.0024, UP 0.0696 Spearman), showing that a large portion of the older raw Reversal Quality edge was mechanical distance information.

## Nuisance drift

Release-hurdle ROC AUC:

- TRAIN DOWN/UP: 0.6922 / 0.6874;
- Validation: 0.5117 / 0.5110;
- Holdout: 0.5124 / 0.4964.

Actual release rate rose from ~23-24% in TRAIN to ~42-45% in Validation/Holdout, while future frozen predicted means remained ~25-29%.

Expected raw-density actual/predicted ratios were:

- Validation DOWN/UP: 1.453 / 1.444;
- Holdout DOWN/UP: 1.889 / 1.764.

The 2023-2024 nuisance background is therefore not stable enough by itself for 2025-2026.

## Retained geometry

Holdout:

- Sweep Depth DOWN/UP: 0.2456 / 0.1704 Spearman;
- Reversal Room DOWN/UP: 0.2591 / 0.2643.

These remain frozen independent tasks. They do not rescue the failed Excess target, but they remain useful if a valid spatial-pool target is later recovered.

## Swing

Holdout Swing Spearman uplift remained small:

- Excess DOWN +0.0071;
- Excess UP +0.0178;
- Reversal DOWN +0.0018;
- Reversal UP +0.0027.

Validation signs were mixed. Swing remains 15m+ supplemental ablation only.

## Code-review finding after the report

The empirical report itself does not test the following distinction, but source review identified it as the next bounded research issue.

R02.3.1 modeled positive magnitude on `log1p(density)` and then formed:

`legacy residual = log1p(actual density) - log1p(P(release) * expected raw positive density)`.

For zero-inflated `Y`, this is generally not equal to subtracting the conditional expectation of the target variable:

`E[log1p(Y)|X] = P(Y>0|X) * E[log1p(Y)|Y>0,X]`.

The old positive-log head also used Huber loss, which is a robust-location objective rather than the L2 conditional mean required by this expectation identity.

This does not invalidate the R02.3.1 blocked decision. It means the blocked decision is specifically about the **old target construction**, not yet proof that the whole latent-liquidity spatial hypothesis is dead.

## Frozen next step

Proceed only to `R02.3.1b Hurdle Target Consistency + Residual Distance Audit`.

Do not add Range Bar, Footprint, OI, Books or another PATH ranker until that audit resolves whether a same-scale mean-aligned nuisance expectation can remove the mechanical distance component.
