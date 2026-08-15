# R02.3 Delivery

## Stage

Distance-normalized Excess Liquidity + Reversal Quality Ranking.

## Command

```bat
python research\eth_ai_trading\eth_latent_liquidity_path_v1\02_3_distance_normalized_excess_liquidity_ranking.py
```

## Reused source

Completed R02.2 exact-first-touch cache. R02.3 does not repeat the 1m->1s historical replay.

## Core outputs

- `03_distance_normalizer_train_only.csv`
- `05_ranking_metrics.csv`
- `06_sweep_geometry_metrics.csv`
- `07_top_zone_summary.csv`
- `08_distance_normalized_profile.csv`
- `09_target_stability.csv`
- `10_swing_ablation.csv`
- `13_causal_audit.csv`
- `15_decision.md`
- `gpt_review_pack.zip`

## Frozen boundaries

- No raw-density rescue.
- No raw distance in the primary model.
- No Swing in the primary model.
- No absolute q80/q90 pool threshold.
- No limit-order backtest in R02.3.
- No sealed-validation or live-approval claim.
- 22 old R02 touch-cache mismatches are quarantined, never silently corrected.

## Validation

- R01 -> R02.3 focused regression: 78 passed under `-W error`.
- all AI Research: 320 passed under `-W error`.
- Data Feed + Research Common: 23 passed; also 23 passed with RuntimeWarning/FutureWarning strict.
- Import-boundary historical violations: 155; R02.3 additions: 0.
- Full repository: 646 tests collected, still blocked by 5 pre-existing missing Liquidity/Analyze Tool modules.
