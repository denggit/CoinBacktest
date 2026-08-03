# R03.4.2.9 Patch Manifest

## Added

- `research/eth_ai_trading/03_4_2_9_dynamic_risk_release.py`
- `research/eth_ai_trading/R03_4_2_9_DELIVERY.md`
- `research/eth_ai_trading/R03_4_2_9_PATCH_MANIFEST.md`
- `src/ai_research/long_tail_dynamic_risk_release/__init__.py`
- `src/ai_research/long_tail_dynamic_risk_release/config.py`
- `src/ai_research/long_tail_dynamic_risk_release/inputs.py`
- `src/ai_research/long_tail_dynamic_risk_release/protection.py`
- `src/ai_research/long_tail_dynamic_risk_release/simulator.py`
- `src/ai_research/long_tail_dynamic_risk_release/analysis.py`
- `src/ai_research/long_tail_dynamic_risk_release/reports.py`
- `src/ai_research/long_tail_dynamic_risk_release/pipeline.py`
- `tests/ai_research/test_long_tail_dynamic_risk_release.py`

## Updated cumulatively

- `research/eth_ai_trading/README.md`
- `research/eth_ai_trading/RESEARCH_HANDOFF.md`
- `research/eth_ai_trading/COMPLETED_WORK.md`
- `research/eth_ai_trading/OPEN_ITEMS_AND_ROADMAP.md`
- `research/eth_ai_trading/DECISION_LOG.md`
- `research/eth_ai_trading/STAGE_DELIVERY.md`

## Frozen behavior not changed

- q70 model and score tiers.
- `failed_reclaim` state machine and parameters.
- 3% disaster protection.
- next-open entry timing.
- 2026 sealed holdout.
- public `src.data_feed` Loader boundary.
