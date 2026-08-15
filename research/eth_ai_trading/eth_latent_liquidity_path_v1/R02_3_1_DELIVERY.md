# R02.3.1 Delivery

## Stage

Zero-inflated Hurdle Nuisance Residualization + Reversal Residual Ranking.

## Command

```bat
python research\eth_ai_trading\eth_latent_liquidity_path_v1\02_3_1_hurdle_nuisance_residualization.py
```

## Reused source

Completed R02.2 exact-first-touch cache with R02.3's exact 1s available-time and old-R02 mismatch quarantine semantics. No historical 1m->1s replay is repeated.

## Main correction

R02.3's median/IQR normalizer is retired because the zero-inflated target produced zero TRAIN medians in all 50 side x distance buckets and zero IQR in 34/50 buckets.

R02.3.1 instead estimates:

1. `P(release > 0 | distance + broad current activity)`;
2. `E[log1p(density) | release > 0, distance + broad current activity]`;
3. `E[raw reversal quality | release > 0, distance + broad current activity]`.

Then:

- `expected_density = P(release) * smearing-adjusted E[density | release]`;
- `Excess Residual = log1p(actual density) - log1p(expected density)`;
- `Reversal Residual = raw reversal quality - expected reversal quality`.

## Strong anti-leak boundary

TRAIN residual labels are generated from **expanding past-only out-of-sample nuisance predictions**. The first six TRAIN months are nuisance warm-up and cannot train/evaluate the residual ranker. Validation/Holdout use nuisance models frozen on all causally eligible 2023-2024 TRAIN after a 13-hour period-boundary purge.

The last 13 hours before the 2025-01-01 and 2025-10-01 period boundaries are purged so future first-touch labels cannot overlap the next evaluation period.

## Nuisance-only information

Nuisance models may use only:

- raw zone distance;
- calendar/session;
- broad group-level notional/trade-count/realized-vol/range activity features that are constant across all zones within a decision_time x side group.

They may not use Swing or zone-specific liquidity-path structure.

## Primary residual ranker

Primary No-Swing rankers exclude:

- raw zone distance;
- all nuisance activity features;
- nuisance predictions / residual target metadata;
- Swing;
- all future labels.

Zone-specific historical boundary/path structure remains available because this is the candidate edge family being tested after mechanical nuisance removal. Quality-control metadata and split-purge flags are explicitly excluded from both residual and geometry feature schemas.

## Quality gate before interpreting model signal

Validation/Holdout must first prove that distance correlations of Excess Residual and Reversal Residual remain below frozen tolerances. If residualization fails, the stage is blocked and Range/Footprint/OI additions are prohibited.

## Outputs

- `03_nuisance_feature_audit.csv`
- `04_nuisance_expanding_fold_audit.csv`
- `05_nuisance_model_metrics.csv`
- `07_residualization_stability.csv`
- `08_ranking_metrics.csv`
- `09_sweep_geometry_metrics.csv`
- `10_top_zone_summary.csv`
- `11_swing_ablation.csv`
- `14_causal_audit.csv`
- `16_decision.md`
- `gpt_review_pack.zip`

## Performance

The stage reuses R02.2 exact-touch labels. Nuisance fitting uses a small number of LightGBM fits (six-month forward blocks by default) with progress reporting. Target/residual construction is vectorized. There is no row-wise historical 12h scan or Episode x Zone matrix.

## Engineering validation

- R02.3.1 dedicated: **8 passed** under full `-W error`;
- R01 through R02.3.1 focused regression: **86 passed** under full `-W error`;
- all AI Research: **328 passed** under full `-W error`;
- Data Feed + Research Common: **23 passed**, and **23 passed** with RuntimeWarning/FutureWarning strict;
- CLI + compileall: PASS;
- Import Boundary: historical unexpected **155**, R02.3.1 new **0**;
- full repository: **654 tests collected**, with the same 5 pre-existing Liquidity / Analyze Tool collection blockers.

No empirical R02.3.1 market result is claimed by this delivery. The real run must be appended to `CUMULATIVE_STAGE_RESULTS.md` before any R02.4 or feature-family expansion.

A clean `CoinBacktest(8)` baseline with **only this cumulative patch** reproduced 86 focused tests, 328 AI Research tests, 23 Data Feed/Research Common tests, CLI/compile PASS, and zero new Import Boundary violations.
