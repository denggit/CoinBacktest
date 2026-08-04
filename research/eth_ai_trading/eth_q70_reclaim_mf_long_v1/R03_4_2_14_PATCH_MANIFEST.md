# R03.4.2.14 Patch Manifest

## Purpose

Audit whether bounded causal entry timing improves the frozen equal-risk C2 strategy by reducing initial MAE, stop-outs or winner damage without filtering away trades.

## Added files

- `src/ai_research/long_tail_entry_timing_mae/`
- `research/eth_ai_trading/03_4_2_14_entry_timing_mae.py`
- `tests/ai_research/test_long_tail_entry_timing_mae.py`
- `docs/ETH_AI_TRADING_R03_4_2_14_ENTRY_TIMING_MAE_RUNBOOK.md`
- `research/eth_ai_trading/R03_4_2_14_DELIVERY.md`

## Updated cumulative files

- `README.md`
- `RESEARCH_HANDOFF.md`
- `COMPLETED_WORK.md`
- `OPEN_ITEMS_AND_ROADMAP.md`
- `DECISION_LOG.md`
- `STAGE_DELIVERY.md`

## Frozen behavior

No q70 model, equal-one-R sizing, C2 2%/1.5% stop chain, `failed_reclaim`, cost grid, delay grid or 2026 boundary is modified.
