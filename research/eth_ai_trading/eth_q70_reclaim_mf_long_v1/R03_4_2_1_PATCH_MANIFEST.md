# R03.4.2.1 Patch Manifest

## Added

- `src/ai_research/long_tail_path_atlas/`
- `research/eth_ai_trading/03_4_2_1_long_tail_path_atlas.py`
- `docs/ETH_AI_TRADING_R03_4_2_1_LONG_TAIL_PATH_ATLAS_RUNBOOK.md`
- `tests/ai_research/test_long_tail_path_atlas.py`

## Contract

- Freeze the R03.4.1 base q90 long model and next-minute-open entry.
- Do not test any new exit rule in this stage.
- Extract every complete q90 event, including winners and losers, for an exact 48-hour one-minute path.
- Record path returns, MFE/MAE, target timing, underwater duration, peak giveback, post-six-hour continuation, and causal base-score evolution.
- Build fixed semantic path labels plus discovery-only KMeans path clusters.
- Fit the 2024 cluster model only from 2023Q4 calibration paths; fit the 2025 cluster model only from paths available through 2024Q4.
- Keep 2026 sealed.
- Keep the abandoned market-state model out of the research and trading chain.
