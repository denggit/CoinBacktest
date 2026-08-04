# R03.4.2.6 Delivery — Incremental Holding Value

## Purpose

Research whether causal price-path structure can predict the economic benefit of continuing an existing q70 long position versus exiting now.

## New code

- `research/eth_ai_trading/03_4_2_6_incremental_hold_value.py`
- `src/ai_research/long_tail_incremental_hold/config.py`
- `src/ai_research/long_tail_incremental_hold/features.py`
- `src/ai_research/long_tail_incremental_hold/modeling.py`
- `src/ai_research/long_tail_incremental_hold/pipeline.py`
- `src/ai_research/long_tail_incremental_hold/reports.py`
- `src/ai_research/long_tail_incremental_hold/__init__.py`
- `tests/ai_research/test_long_tail_incremental_hold.py`

## Run

```text
python research\eth_ai_trading\03_4_2_6_incremental_hold_value.py
```

Optional cache rebuild:

```text
python research\eth_ai_trading\03_4_2_6_incremental_hold_value.py --force-rebuild-outcomes
```

## Report directory

```text
data\reports\research\eth_ai_trading\03_4_2_6_incremental_hold_value
```

Key outputs:

- `05_model_metrics.csv`
- `06_prediction_deciles.csv`
- `08_action_diagnostics.csv`
- `09_score_tier_diagnostics.csv`
- `10_score_ablation.csv`
- `13_stable_candidates.csv`
- `99_decision.md`
- `gpt_review_pack.zip`

## Research contract

- q70 is not removed.
- q70-q80/q80-q90/q90+ remain separate.
- Checkpoints are decisions, not forced exits.
- 120h is a research label/censoring window, not a final time exit.
- Opening score is ablated and cannot automatically control holding.
- No abandoned market-state feature is loaded.
- 2026 remains sealed.

## Validation executed before delivery

- New tests: 8 passed.
- All `tests/ai_research`: 131 passed.
- All `tests/data_feed`: 8 passed.
- Python compilation: passed.
- CLI `--help`: passed.
