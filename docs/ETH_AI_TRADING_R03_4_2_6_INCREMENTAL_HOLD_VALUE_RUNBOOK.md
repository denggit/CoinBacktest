# ETH AI Trading R03.4.2.6 Runbook

## Objective

Predict the incremental economic value of continuing an existing q70 long position instead of exiting at the current checkpoint. This is a signal-discovery stage for a future recurrent non-time exit state machine.

## Run

```text
python research\eth_ai_trading\03_4_2_6_incremental_hold_value.py
```

## What it does

- Builds strict rolling OOF q50 events for holding-model training.
- Evaluates frozen q70 and q90 opening pools in 2024 and 2025.
- Extracts causal price-path features at 180m, 360m, 720m, 24h and 48h.
- Computes next-checkpoint and best-future incremental utility labels.
- Compares mechanical/path/score-only/path+score models.
- Reports q70-q80, q80-q90 and q90+ separately.

## Interpretation

`PASS_INCREMENTAL_HOLD_VALUE_SIGNAL` means a non-score-only model can rank next-checkpoint incremental utility in both 2024 and 2025. It does **not** mean the final exit strategy is complete.

`RESEARCH_CONTINUE_RANKING_ONLY` means local information exists but cannot yet drive a live exit.

`FAIL_NO_INCREMENTAL_HOLD_MODELS` means the modeling path should stop or be redesigned.

## Safety

- 2026 remains sealed.
- Checkpoints are not mandatory exits.
- 120h is a research label/censoring horizon.
- No market-state output is loaded.
- The wide disaster stop remains a separate risk-control candidate.
