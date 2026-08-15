# R02 Delivery — Pre-event latent liquidity-pool location and sweep-depth model

## Run

Windows / Unix from repository root:

```text
python research\eth_ai_trading\eth_latent_liquidity_path_v1\02_latent_pool_location_depth_model.py
```

Unix path separators may be used on Unix.

## Data

R02 is cache-first and does not download missing bars by default.  It expects:

- R01.1 full-history report tables;
- local 1m Trade Bar coverage;
- local 1s Trade Bar coverage;
- existing R01.1 15m+ all-unswept-Swing lifecycle cache (required for the explicit Swing incremental ablation).

The decision lattice is causal every 15 minutes.  Price space is covered continuously from current price through approximately +/-500bp using fixed spatial cells; this grid is numerical coverage, not a stop-location hypothesis.  The last 12 hours of the requested research window are reserved only as future-label support, so incomplete-horizon rows can never become false negatives.

## Runtime safety

- chunk size: 14 days;
- each completed spatial chunk is checkpointed;
- R01.1 Episode labels are cached separately;
- untouched and touched/no-release controls are deterministic inverse-probability model samples;
- an independent 5% sample of decision-side groups retains the complete 25-cell price lattice for unbiased Top-zone/calibration diagnostics;
- every release-zone row is retained;
- feature matrices are bounded before LightGBM fitting;
- no all-history 1-second frame is loaded at once;
- `--no-cache` bypasses both final and per-chunk caches;
- release labels are audited to imply an actual primary-horizon touch.

## Key report files

- `03_zone_label_summary.csv`
- `04_model_metrics.csv`
- `05_pool_score_deciles.csv`
- `06_feature_importance.csv`
- `07_feature_family_importance.csv`
- `08_calibration_thresholds.csv`
- `09_top_zone_summary.csv`
- `10_causal_audit.csv`
- `12_decision.md`
- `gpt_review_pack.zip`

## Interpretation boundary

Passing R02 only allows an R02.1 limit-placement study.  It does not authorize live capital.
