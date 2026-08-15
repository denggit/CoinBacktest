# R02.3 Post-Run Review — robust median/IQR excess normalization failed under zero inflation

## Formal result

The R02.3 report completed with all causal gates passing. However, the requested `Distance-normalized Excess Liquidity` target did **not** achieve the intended normalization and must not be used as proof that distance bias was removed.

## Root cause

The fixed 180-second first-touch Release Density target is strongly zero-inflated.

The TRAIN-only side x distance table contains 50 buckets (25 distances x 2 sides):

- `expected_density == 0` in **50 / 50** buckets;
- `IQR == 0` in **34 / 50** buckets, forcing the fixed `0.1` scale floor;
- therefore many `excess_liquidity_z` values reduce to a scaled version of `log1p(raw_density)` rather than a true distance-conditioned abnormal-liquidity residual.

This is a target-construction failure, not a causal failure.

## Out-of-sample evidence that distance contamination remained

Distance-vs-Excess Spearman:

- Validation DOWN: **-0.1709**;
- Holdout DOWN: **-0.1843**;
- Validation UP: **-0.1547**;
- Holdout UP: **-0.1586**.

The TRAIN values were near zero only because TRAIN itself defined the median/IQR normalizer. The distance effect returned immediately out of sample.

Mean Excess Z also drifted sharply:

- DOWN: TRAIN **2.64** -> Validation **5.44** -> Holdout **5.94**;
- UP: TRAIN **2.00** -> Validation **4.06** -> Holdout **4.20**.

Therefore R02.3 may not be followed by Range/Footprint/OI on the same failed target.

## Retained No-Swing path evidence

Excess Liquidity ranking remained weak after the failed normalization:

- DOWN Validation/Holdout PATH_NO_SWING Spearman: **0.0968 / 0.0685**;
- UP Validation/Holdout: **0.0504 / 0.0293**.

Swing did not provide stable incremental value:

- DOWN Holdout Excess: No-Swing **0.0685**, With-Swing **0.0472**;
- UP Holdout Excess: No-Swing **0.0293**, With-Swing **0.0440**, but Validation moved the opposite way (**0.0504 vs 0.0428**).

Primary research remains No-Swing.

## Reversal Quality survived better than Pool Strength

No-Swing Reversal Quality ranking was materially more stable:

- DOWN Validation/Holdout Spearman: **0.1740 / 0.1697**;
- UP Validation/Holdout: **0.1478 / 0.1884**.

However, far-distance mechanical baselines were stronger in Holdout, so this cannot yet be called path-specific reversal edge. Distance itself must be removed from Reversal Quality as a nuisance too.

## Sweep geometry retained

Holdout:

- DOWN Sweep Depth Spearman: **0.2652**;
- UP Sweep Depth: **0.1662**;
- DOWN Reversal Room: **0.2594**;
- UP Reversal Room: **0.2570**.

These remain separate geometry tasks and are retained for later placement research if spatial pool-location residual edge becomes real.

## Causal / quality status

R02.3 causal audit passed completely.

- prior 1s bar-start/available-time audit issue is fixed;
- old R02 exact-touch disagreements remain quarantined and zero quarantined rows entered model evaluation;
- raw distance, Swing and normalizer outputs were excluded from the primary ranker.

The failure is specifically the statistical normalizer under a zero-inflated target.

## Frozen decision

- retire `median/IQR(log1p density) by side x distance` as the R02.3 excess target;
- do **not** add new data families yet;
- proceed to R02.3.1 with a two-part hurdle nuisance expectation;
- residualize Reversal Quality against the same mechanical distance/activity nuisance family;
- primary residual rankers remain No-Swing and also exclude nuisance activity inputs;
- retain Sweep Depth / Reversal Room independently.
