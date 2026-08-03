# ETH AI Trading R03.4.1 Runbook

## Purpose

Test whether soft strategic/activity context improves long-opportunity selection **after** the frozen R03.4 base model.

The state layer is not allowed to trigger long entries directly. Tactical and entry discrete direction labels are excluded.

## Required existing caches

- R03.2 long-context base shards for 2023, 2024 and 2025.
- R03.3.3.1 universal state cache. The script can build it if missing.
- R03.4 outcome cache. The script reuses/builds only 2023-2025 outcomes and keeps 2026 sealed.

## Run

```text
python research\eth_ai_trading\03_4_1_long_state_meta_calibration.py
```

Force only reusable caches when necessary:

```text
python research\eth_ai_trading\03_4_1_long_state_meta_calibration.py --force-rebuild-state-cache --force-rebuild-outcomes
```

## Main outputs

```text
data\reports\research\eth_ai_trading\03_4_1_long_state_meta_calibration
```

Review in order:

1. `03_oof_stacking_audit.csv`
2. `04_model_metrics.csv`
3. `06_common_candidate_rerank_metrics.csv`
4. `07_fixed_candidate_multiplier_metrics.csv`
5. `08_uplift_vs_controls.csv`
6. `09_stable_candidates.csv`
7. `99_decision.md`
8. `gpt_review_pack.zip`

## Decision meanings

- `PASS_STATE_META_CALIBRATION_UPLIFT`: state improves long ranking and common-candidate selection in both 2024 and 2025.
- `PASS_STATE_RISK_SCALING_ONLY`: state is useful only for sizing/risk reduction on fixed base events.
- `FAIL_NO_STABLE_LONG_STATE_UPLIFT`: stop adding state context to the opening model.

A pass is still not a live strategy. Six-hour fixed close remains a diagnostic exit.
