# R02.1 Delivery

## Stage

`R02.1 — Conditional pool-strength / release-density deconfounding`

## Purpose

Separate "where price is likely to arrive" from "how much latent liquidity is likely to release if that zone is reached".  The primary score does not multiply Touch probability and does not use Swing.

## Run

```bat
python research\eth_ai_trading\eth_latent_liquidity_path_v1\02_1_pool_strength_density_model.py
```

R02.1 reuses the completed R02 spatial cache and R01.1 Episode cache.  It does not rerun the multi-hour 1s spatial-path build.

## Main reports

- `04_model_metrics.csv`
- `05_strength_score_deciles.csv`
- `07_feature_family_importance.csv`
- `09_top1_zone_summary.csv`
- `10_q90_zone_summary.csv`
- `11_distance_strength_profile.csv`
- `12_causal_audit.csv`
- `14_decision.md`
- `gpt_review_pack.zip`

## Hard boundaries

- strict future horizon `(t, t+12h)`;
- train 2023–2024, calibration 2025 Q1–Q3, development holdout 2025 Q4–2026 H1;
- untouched zones do not become zero-liquidity supervision;
- no order placement;
- no Swing admission gate;
- no live approval.

## Full-history performance hotfix

If a pre-hotfix run stays at:

```text
[stage] aggregate all future release Episodes into touched-zone strength labels
```

with sustained CPU/RAM usage but no progress, stop that run and apply the latest cumulative R02.1 performance hotfix. The old implementation performs the correct computation with excessive pandas object overhead and does not persist a completed R02.1 strength cache until that stage finishes.

The hotfix keeps the exact same labels and replaces only the implementation with bounded NumPy aggregation. A new progress display begins with:

```text
[strength-aggregation] decision_times=... chunks=... chunk_size=1,024 Episodes=...
[latent-liquidity-r02.1] strength aggregation [...]
```

After this stage completes, the existing R02.1 dataset cache is written and later runs can reuse it.
