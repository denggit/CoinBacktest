# R02.3.1 cumulative patch manifest

This patch is cumulative through R02.3.1 and is intended to be applied directly on the supplied CoinBacktest baseline without stacking older latent-liquidity patches.

New stage files:

- `src/ai_research/latent_liquidity_hurdle_residualization/`
- `research/eth_ai_trading/eth_latent_liquidity_path_v1/02_3_1_hurdle_nuisance_residualization.py`
- `tests/ai_research/test_latent_liquidity_hurdle_residualization.py`
- `R02_3_POST_RUN_REVIEW.md`
- `R02_3_1_DELIVERY.md`

Also included:

- every cumulative source/doc/test from archived Q70 through R01/R01.1/R01.2/R01.3/R02/R02.1/R02.2/R02.3;
- R02.3 real-run evidence and the zero-inflated normalizer failure;
- updated cumulative stage results / decision log / current state / roadmap;
- the algorithm-speed rule and all prior causal/performance hotfixes.

Validation at patch freeze:

- dedicated R02.3.1: 8 passed;
- focused R01 -> R02.3.1: 86 passed;
- all AI Research: 328 passed;
- Data Feed / Research Common: 23 passed;
- new Import Boundary violations: 0.
