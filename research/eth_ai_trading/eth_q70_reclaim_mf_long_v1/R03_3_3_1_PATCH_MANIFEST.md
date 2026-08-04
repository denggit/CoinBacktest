# R03.3.3.1 Patch Manifest

## Purpose

Calibrate the strategic state causally and audit whether the multi-timescale continuity model adds information beyond mechanically predictable state persistence.

The stage is auxiliary context only. It never opens, closes, reverses or sizes positions.

## Main changes

- causal rolling strategic thresholds instead of unreachable fixed `±0.30`;
- strict uninterrupted persistence labels;
- state-boundary margin features;
- age-only, margin-only and age+margin+state baselines;
- independent transition-warning episodes;
- stricter decision labels separating learned increment from mechanical persistence;
- independent cache/report directories.

## Command

```bat
python research\eth_ai_trading\03_3_3_1_market_state_continuity_audit.py
```

## Cache

```text
data/cache/eth_ai_trading/r03_3_3_1_universal_state
```

## Report

```text
data/reports/research/eth_ai_trading/03_3_3_1_market_state_continuity_audit
```

## Validation

- Python compilation;
- 69 AI Research tests;
- 8 Data Feed tests;
- strategic threshold future-perturbation test;
- flip-away-and-return persistence test;
- independent warning-episode merge test;
- command entrypoint.
