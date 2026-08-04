# ETH AI Trading R03.4.2.17 Runbook

## Purpose

Diagnose why frozen C2 failed January-June 2026 but recovered in July. This stage is post-seal diagnosis only. It cannot approve V1 or a new gate.

## Frozen inputs

- R03.4.2.15 final account report;
- R03.4.2.16 failed sealed H1 report;
- R03.4.2.16.1 completed July forward report;
- exact pre-2026 fit, Q4-2025 q70 threshold and feature schema;
- public 1m OKX Trade Bar loader.

## Run

```bat
python research\eth_ai_trading\03_4_2_17_state_gate_diagnostic.py
```

Optional cache rebuild:

```bat
python research\eth_ai_trading\03_4_2_17_state_gate_diagnostic.py --force-rebuild-base --force-rebuild-outcomes
```

## Reports

```text
data\reports\research\eth_ai_trading\03_4_2_17_state_gate_diagnostic
```

Read first:

```text
99_decision.md
06_c2_state_summary.csv
08_fixed6h_state_summary.csv
10_score_state_summary.csv
11_monthly_market_vs_c2.csv
12_counterfactual_gate_summary.csv
13_attribution_findings.csv
14_model_recipe_audit.csv
15_causal_audit.csv
gpt_review_pack.zip
```

## State contract

- 4H bar is available only at `bar_start + 4h`.
- 1D bar is available only at `bar_start + 1d`.
- UP requires completed close above EMA20 above EMA50 and positive three-bar EMA20 slope.
- DOWN is the symmetric rule.
- Drawdown and volatility bands are fixed economic diagnostics, not optimized on returns.

## Interpretation

Possible diagnostic conclusions:

```text
DIAGNOSIS_REGIME_DEPENDENCE_AND_SCORE_DRIFT
DIAGNOSIS_REGIME_DEPENDENCE_SUPPORTED
DIAGNOSIS_SCORE_DRIFT_DOMINANT
DIAGNOSIS_MIXED_NO_SIMPLE_GATE
```

None means live approval. Any V2 hypothesis must be separately versioned and validated on future untouched data.
