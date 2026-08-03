# R03.4.2.8A Patch Manifest

Stage: `R03.4.2.8A`

Goal: build a causal occupied-q70 signal atlas and a hard eligibility gate before any second Tranche or pyramiding is simulated.

This patch freezes the q70 opening model, `failed_reclaim` parameters and 3% disaster floor. It does not add size, alter AetherEdge, open 2026, tune exit parameters or train another holding model.

## Added code

- `research/eth_ai_trading/03_4_2_8a_occupied_signal_atlas.py`
- `src/ai_research/long_tail_tranche_eligibility/__init__.py`
- `src/ai_research/long_tail_tranche_eligibility/config.py`
- `src/ai_research/long_tail_tranche_eligibility/simulator.py`
- `src/ai_research/long_tail_tranche_eligibility/analysis.py`
- `src/ai_research/long_tail_tranche_eligibility/pipeline.py`
- `src/ai_research/long_tail_tranche_eligibility/reports.py`
- `tests/ai_research/test_long_tail_tranche_eligibility.py`
- `docs/ETH_AI_TRADING_R03_4_2_8A_OCCUPIED_SIGNAL_ATLAS_RUNBOOK.md`

## Cumulative handoff updates

- `README.md`
- `RESEARCH_HANDOFF.md`
- `COMPLETED_WORK.md`
- `OPEN_ITEMS_AND_ROADMAP.md`
- `DECISION_LOG.md`
- `STAGE_DELIVERY.md`
- `R03_4_2_8A_DELIVERY.md`

## Empirical status

Code delivered; full local-data run pending. No R03.4.2.8A profitability claim is included in the patch.

Validation: 8 new tests passed; 17 new plus frozen R03.4.2.7 tests passed; 156 AI research/data-feed tests passed. Full repository collection and the existing import-boundary test remain red only because of unrelated pre-existing missing/cross-research modules outside this patch.
