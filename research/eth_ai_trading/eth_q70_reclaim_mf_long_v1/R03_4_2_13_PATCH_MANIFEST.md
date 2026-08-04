# R03.4.2.13 Patch Manifest

## Purpose

Add a causal account replay that tests whether the frozen q70/q80/q90 opening tiers deserve different initial account risk on top of the passed C2 stop/exit chain.

## Added files

- `src/ai_research/long_tail_score_risk_sizing/`
- `research/eth_ai_trading/03_4_2_13_score_risk_sizing.py`
- `tests/ai_research/test_long_tail_score_risk_sizing.py`
- `docs/ETH_AI_TRADING_R03_4_2_13_SCORE_RISK_SIZING_RUNBOOK.md`
- `research/eth_ai_trading/R03_4_2_13_DELIVERY.md`

## Updated cumulative files

- `README.md`
- `RESEARCH_HANDOFF.md`
- `COMPLETED_WORK.md`
- `OPEN_ITEMS_AND_ROADMAP.md`
- `DECISION_LOG.md`
- `STAGE_DELIVERY.md`

## Frozen behavior

No q70 model, C2 stop, soft failure, `failed_reclaim`, cost grid, delay grid or 2026 boundary is modified.
