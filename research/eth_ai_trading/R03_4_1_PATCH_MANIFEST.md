# R03.4.1 Patch Manifest

## Added

- `src/ai_research/long_state_calibration/`
- `research/eth_ai_trading/03_4_1_long_state_meta_calibration.py`
- `tests/ai_research/test_long_state_calibration.py`
- `docs/ETH_AI_TRADING_R03_4_1_LONG_STATE_META_CALIBRATION_RUNBOOK.md`

## Research contract

- Long-only second-stage calibration.
- Frozen first-stage multiframe base model.
- Expanding-window OOF stacking with embargo.
- Strategic/activity soft state only.
- No tactical/entry discrete direction labels.
- Common base-candidate reranking and fixed-event position multiplier audits.
- 2024 and 2025 pure OOS; 2026 sealed.
