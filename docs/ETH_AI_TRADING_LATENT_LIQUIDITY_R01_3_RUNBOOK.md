# ETH Latent Liquidity Pool Path Learning V1 — R01.3 Runbook

## Purpose

R01.3 is the final bounded commercial gate for the latent-liquidity path family. It learns whether post-release absorption is complete and whether enough reversal room remains after realistic cost and structural risk.

## Prerequisites

- completed R01.1 full-history report tables;
- local `ETH-USDT-SWAP` 1-second Trade Bar data;
- R01.2 replay-quality hotfix applied;
- LightGBM installed in the active environment.

## Windows command

```bat
python research\eth_ai_trading\eth_latent_liquidity_path_v1\01_3_absorption_remaining_space_model.py
```

## Causal chronology

- train: 2023–2024;
- calibration and q90 threshold: 2025 Q1–Q3;
- evaluation-only holdout: 2025 Q4–2026 H1;
- decision snapshots: 15/30/45/60/90/120/180/240/300 seconds after release;
- entry: next 1-second open;
- stop: decision-time known extreme plus 3bp;
- costs: 11/22/33bp;
- delay: 1/3/5 seconds.

## Cache behavior

The source scan reads only narrow Episode metadata from the R01.1 tables. The 1-second snapshot replay is cached by day under:

```text
data\cache\research\eth_ai_trading\eth_latent_liquidity_path_v1\r01_3_absorption_remaining_space_model
```

A rerun reuses completed daily caches. Use `--no-cache` only for an intentional full rebuild.

## Key reports

- `04_model_metrics.csv`: full model, baseline and absorption metrics;
- `05_score_decile_lift.csv`: monotonic score quality;
- `06_feature_importance.csv`: causal dynamic feature importance;
- `07_calibration_thresholds.csv`: validation-only frozen thresholds;
- `08_selected_trade_summary.csv`: cost/delay execution gate;
- `10_selected_monthly_summary.csv`: month concentration;
- `12_causal_audit.csv`: availability, next-open and threshold audit;
- `13_decision.md`: formal promote/stop decision;
- `gpt_review_pack.zip`: compact review bundle.

## Interpretation

A promotion only authorizes a formal R02 account backtest for the passing direction. It is not live approval. A stop decision ends this executable model family and forbids further confirmation/stop/threshold patching on the same evidence.
