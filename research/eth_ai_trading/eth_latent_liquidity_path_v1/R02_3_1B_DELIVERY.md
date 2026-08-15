# R02.3.1b Delivery

## Stage

Hurdle Target Consistency + Residual Distance Audit.

## Command

```bat
python research\eth_ai_trading\eth_latent_liquidity_path_v1\02_3_1b_target_consistency_audit.py
```

Force rebuild once if the stage cache is stale/missing metadata:

```bat
python research\eth_ai_trading\eth_latent_liquidity_path_v1\02_3_1b_target_consistency_audit.py --no-cache
```

## Why this stage exists

The real R02.3.1 run passed all causal checks but was formally blocked because the Excess Residual still contained too much distance dependence, especially UP.

Code review then found two target-consistency issues:

1. positive magnitude was modeled on `log1p(density)` but the final residual subtracted `log1p(raw expected density)`;
2. the positive-log head used Huber loss, while the expectation identity needs the conditional mean on the log scale.

R02.3.1b isolates these issues before any new data family or PATH model is allowed.

## Three frozen comparisons

For `Z = log1p(release density)`:

1. legacy diagnostic: `log1p(P(release) * smearing-adjusted positive raw density)`;
2. formula-only diagnostic: `P(release) * Huber[Z | release]`;
3. primary target-consistent expectation: `P(release) * L2-mean[Z | release]`.

Primary residual:

`Z - P(release) * E_L2[Z | release, X]`.

## Causal boundary

- source: completed R02.2 exact-first-touch labels;
- old R02 touch disagreements remain quarantined;
- nuisance features: raw distance + calendar/session + broad group-level activity/volatility only;
- TRAIN: expanding past-only OOS predictions with 13h purge;
- Validation/Holdout: one full-2023-2024-TRAIN frozen nuisance family;
- Swing: prohibited from nuisance estimation;
- zone-specific path structure: prohibited from nuisance estimation;
- no PATH ranker is trained in this stage.

## Reports

- `03_nuisance_feature_audit.csv`
- `04_nuisance_expanding_fold_audit.csv`
- `05_nuisance_model_metrics.csv`
- `07_target_consistency_stability.csv`
- `08_distance_cell_residual_audit.csv`
- `09_yearly_stability.csv`
- `10_transform_gap_audit.csv`
- `11_causal_audit.csv`
- `12_decision.md`
- `13_target_consistency_sample.csv`
- `gpt_review_pack.zip`

## Hard decision boundary

Corrected Validation/Holdout residual-distance correlation must remain below the frozen **0.12** tolerance and improve over raw mechanical distance dependence.

Failure means remain blocked: no Range/Footprint/OI/Books and no new PATH ranker.

Passing target consistency does not automatically reopen PATH. If release nuisance calibration/discrimination still drifts, the next bounded stage is nuisance-regime conditioning only.

## Performance design

- reuses R02.2 exact-first-touch source; no historical 1m->1s replay;
- fixed small number of expanding LightGBM fits;
- vectorized target construction and audits;
- all CPU cores for LightGBM;
- standard project progress reporter;
- no row-wise historical scan or parameter grid.

## Engineering validation at delivery

- dedicated R02.3.1b tests: **9 passed**;
- R02.3.1 + R02.3.1b focused strict regression: **17 passed**;
- all `tests/ai_research`: **337 passed**;
- `tests/data_feed tests/research_common`: **23 passed** under RuntimeWarning/FutureWarning strict;
- target expectation identity and report-writer smoke tests: PASS;
- causal chronology/source semantics tests: PASS;
- import-boundary audit: baseline historical violations **155**, patched tree **155**, R02.3.1b additions **0**;
- full repository collection remains blocked only by the same **5 pre-existing Liquidity / Analyze Tool missing-module errors**;
- no empirical R02.3.1b market result is claimed before the user runs the command.
