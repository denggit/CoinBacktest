# R02.1 Patch Manifest

## Stage

`R02.1 — Conditional pool-strength / release-density deconfounding`

## New code

- `research/eth_ai_trading/eth_latent_liquidity_path_v1/02_1_pool_strength_density_model.py`
- `src/ai_research/latent_liquidity_pool_strength/__init__.py`
- `src/ai_research/latent_liquidity_pool_strength/config.py`
- `src/ai_research/latent_liquidity_pool_strength/cache.py`
- `src/ai_research/latent_liquidity_pool_strength/labels.py`
- `src/ai_research/latent_liquidity_pool_strength/modeling.py`
- `src/ai_research/latent_liquidity_pool_strength/pipeline.py`
- `src/ai_research/latent_liquidity_pool_strength/reports.py`
- `tests/ai_research/test_latent_liquidity_pool_strength.py`

## R02 compatibility fix

- `src/ai_research/latent_liquidity_pool_forecast/labels.py`
  - future release horizon right edge changed from inclusive to exclusive;
  - exact `t + 720m` Episodes no longer violate `release_implies_primary_touch`.
- `tests/ai_research/test_latent_liquidity_pool_forecast.py`
  - exact-right-edge regression test added.

## Research rules

- Primary `pool_strength_score` is path/no-Swing and excludes Touch probability.
- Strength supervision is conditional on touched zones only.
- All future release Episodes in a zone are aggregated; first-Release binary label is not the pool-strength target.
- 15m+ unswept Swing remains only as an explicit supplemental ablation.
- q80 high-strength target threshold is frozen from complete TRAIN full-lattice audit zones, not the sampled model-control distribution.
- R02.1 does not place orders.

## Tests

- R01→R02.1 related: 62 passed.
- all `tests/ai_research`: 304 passed.
- Data Feed + Research Common: 23 passed.
- RuntimeWarning/FutureWarning strict subset: 22 passed.
- full repository collection remains blocked by 5 pre-existing Liquidity/Analyze Tool missing modules.
- Import Boundary: 155 historical unexpected violations, R02.1 new violations = 0.
