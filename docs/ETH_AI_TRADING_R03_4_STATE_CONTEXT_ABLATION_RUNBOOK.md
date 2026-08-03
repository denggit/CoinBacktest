# ETH AI Trading R03.4 — State Context Ablation

## Goal

Test whether the frozen R03.3.3.1 multi-timescale state context improves the same directional opening-value model on 2024 and 2025 pure OOS data.

## Fixed comparison

All variants share the same 15-minute decisions, next-minute-open entry, 6-hour future outcomes, LightGBM settings, calibration windows, signal quantiles and cost assumptions. Only the available state-context columns change.

The 6-hour close is a diagnostic comparison exit, not a proposed live exit.

## Prerequisite

R03.4 deliberately does not rebuild R03.2 long-context caches, because rebuilding the 2025 shard could read beyond the sealed 2026 boundary. Existing `samples_2023`, `samples_2024`, and `samples_2025` caches are required.

## Run

```text
python research\eth_ai_trading\03_4_state_context_ablation.py
```

Optional forced rebuild:

```text
python research\eth_ai_trading\03_4_state_context_ablation.py --force-rebuild-outcomes
```

## Key outputs

```text
data\reports\research\eth_ai_trading\03_4_state_context_ablation\04_model_metrics.csv
data\reports\research\eth_ai_trading\03_4_state_context_ablation\05_cost_aware_signal_metrics.csv
data\reports\research\eth_ai_trading\03_4_state_context_ablation\06_uplift_vs_base.csv
data\reports\research\eth_ai_trading\03_4_state_context_ablation\07_stable_uplift_candidates.csv
data\reports\research\eth_ai_trading\03_4_state_context_ablation\99_decision.md
```
