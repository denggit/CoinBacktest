# ETH AI R03.4.2.7 Runbook

## Command

```text
python research\eth_ai_trading\03_4_2_7_non_time_structural_exit.py
```

## What this stage tests

- One identical structural rule set in 2024 and 2025.
- Causal confirmed swing lows/highs.
- Break and reclaim rather than immediate stop on first breach.
- Confirmed lower-high/lower-low failure.
- Structure-confirmed profit protection.
- Wide disaster protection.
- No fixed or maximum holding-time exit.

## How unresolved positions are handled

A position still open at OOS end or a data gap is marked to market and labelled censored. It is not treated as a strategy exit. A candidate cannot pass if censoring is excessive.

## Read first when reviewing results

1. `99_decision.md`
2. `11_stable_candidates.csv`
3. `10_vs_fixed6h_comparison.csv`
4. `08_censoring_audit.csv`
5. `07_exit_reason_summary.csv`
6. `06_score_tier_summary.csv`
