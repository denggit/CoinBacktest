# R01.2 Delivery

## Stage

`R01.2 — Stable path explanation and executable-confirmation audit`

## New command

```bat
python research\eth_ai_trading\eth_latent_liquidity_path_v1\01_2_stable_path_execution_audit.py
```

The script consumes the completed R01.1 report tables directly. It does **not** rerun the 3-hour full-history atlas.

## Required local files

```text
data/reports/research/eth_ai_trading/eth_latent_liquidity_path_v1/01_1_liquidity_first_path_atlas/
    00_manifest.json
    09_causal_audit.csv
    12_feature_table.csv.gz
    13_label_table.csv.gz
    14_cluster_assignment.csv.gz
```

The local `1s` Trade Bar DB is required for causal replay.

R01.2 writes source-scan and replay checkpoints under:

```text
data/cache/research/eth_ai_trading/eth_latent_liquidity_path_v1/r01_2_stable_path_execution_audit
```

Use `--no-cache` only when intentionally rebuilding the audit.

## Report directory

```text
data/reports/research/eth_ai_trading/eth_latent_liquidity_path_v1/01_2_stable_path_execution_audit
```

## Main report files

```text
03_episode_cluster_stability.csv
06_day_block_bootstrap_ci.csv
07_cluster_feature_profile.csv
08_feature_family_profile.csv
09_cluster_runtime_signature.csv
10_event_aligned_price_path.csv
11_event_aligned_flow_path.csv
13_confirmation_detection.csv
14_confirmation_rule_summary.csv
15_confirmation_period_stability.csv
16_causal_audit.csv
17_decision.md
gpt_review_pack.zip
```

## Scope

- Explains the stable R01.1 discovery clusters.
- Tests fixed causal confirmations with real 1-second execution timing.
- Does not optimize entry parameters.
- Does not claim direct knowledge of private stop orders.
- Does not promote a cluster ID directly into a live strategy.
- Swing remains supplementary only.
