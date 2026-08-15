# R02.3.1b cumulative patch manifest

This patch is cumulative through the R02.3.1b code/research-state update and is intended to be applied directly on the supplied `CoinBacktest(9)` baseline.

## New stage files

- `src/ai_research/latent_liquidity_target_consistency/`
- `research/eth_ai_trading/eth_latent_liquidity_path_v1/02_3_1b_target_consistency_audit.py`
- `tests/ai_research/test_latent_liquidity_target_consistency.py`
- `R02_3_1_POST_RUN_REVIEW.md`
- `R02_3_1B_DELIVERY.md`
- `R02_3_1B_PATCH_MANIFEST.md`

## Updated cumulative research memory

- `CUMULATIVE_STAGE_RESULTS.md`: appends the full real R02.3.1 result plus the R02.3.1b design/pre-run status.
- `DECISION_LOG.md`: records why R02.3.1 is blocked and why R02.3.1b is the only allowed next step.
- `CURRENT_STATE.md`: makes R02.3.1b the active audit stage.
- `OPEN_ITEMS_AND_ROADMAP.md`: blocks new data/PATH work until target/nuisance quality passes.
- `README.md`: updates stage status.

## Frozen historical behavior

R02.3.1 source/module/report logic is not modified. The old blocked result remains reproducible and historically interpretable.

## Validation at patch freeze

- R02.3.1b dedicated: 9 passed under RuntimeWarning/FutureWarning strict;
- R02.3.1 + R02.3.1b focused strict regression: 17 passed;
- all AI Research: 337 passed;
- Data Feed / Research Common strict: 23 passed;
- import-boundary audit: baseline 155 historical violations, patched tree 155, new violations 0;
- full repository remains blocked by the same 5 pre-existing Liquidity / Analyze Tool collection errors;
- no git commit is performed.
