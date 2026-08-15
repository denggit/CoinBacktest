# R01.3 Delivery

## Stage

`R01.3 — Absorption completion and remaining-space supervised audit`

## Run

```bat
python research\eth_ai_trading\eth_latent_liquidity_path_v1\01_3_absorption_remaining_space_model.py
```

## Report directory

```text
data\reports\research\eth_ai_trading\eth_latent_liquidity_path_v1\01_3_absorption_remaining_space_model
```

## Important outputs

- `04_model_metrics.csv`
- `05_score_decile_lift.csv`
- `06_feature_importance.csv`
- `07_calibration_thresholds.csv`
- `08_selected_trade_summary.csv`
- `09_selected_cluster_summary.csv`
- `10_selected_monthly_summary.csv`
- `12_causal_audit.csv`
- `13_decision.md`
- `gpt_review_pack.zip`

## Expected runtime behavior

- first run scans the R01.1 full tables and creates a deterministic Episode sample;
- sampled 1-second paths are cached per day;
- a later interruption resumes from completed daily snapshot caches;
- default sample cap is 400 Episodes per cluster × side × period stratum;
- no R01.1 three-hour atlas rebuild is required.

## Interpretation

- `PROMOTE_*_TO_R02_FORMAL_STRATEGY_BACKTEST` is research promotion only;
- `STOP_LATENT_LIQUIDITY_PATH_V1_EXECUTION_NOT_VIABLE` is a formal stop for this executable path family;
- no output is live approval.
