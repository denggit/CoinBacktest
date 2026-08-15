# R02 Patch Manifest

## New module

`src/ai_research/latent_liquidity_pool_forecast/`

- `config.py` — frozen spatial lattice, chronology and model settings;
- `source.py` — aligned narrow R01.1 Episode labels;
- `spatial.py` — causal 1s/1m path + arbitrary price-zone + all-active 15m+ unswept-Swing supplemental features;
- `labels.py` — future touch/release/reversal/depth labels and weighted control sampling;
- `modeling.py` — distance baseline, no-Swing liquidity path, full-with-Swing, favorable and depth models;
- `reports.py` — compact diagnostics, Swing ablation, causal gate and R02 decision;
- `cache.py` — source, full dataset and per-chunk checkpoint keys;
- `pipeline.py` — end-to-end orchestration.

## New entry point

`research/eth_ai_trading/eth_latent_liquidity_path_v1/02_latent_pool_location_depth_model.py`

## New tests

`tests/ai_research/test_latent_liquidity_pool_forecast.py`

## Updated cumulative handoff

- `CUMULATIVE_STAGE_RESULTS.md`
- `CURRENT_STATE.md`
- `OPEN_ITEMS_AND_ROADMAP.md`
- `DECISION_LOG.md`
- `R02_DELIVERY.md`

No Git commit is executed by this patch.

## Final pre-run hardening

- incomplete 12h future tails are excluded instead of becoming negative labels;
- release-implies-touch is a causal/data-quality gate;
- `--no-cache` now also bypasses per-chunk checkpoints;
- report threshold filtering is vectorized;
- top-zone sample is compact rather than a 50k-row wide feature dump;
- missing R01.1 all-unswept 15m+ Swing lifecycle is an explicit setup error.

- added a complete-lattice audit sample for unbiased spatial Top-zone diagnostics and calibration thresholds;
- AUC/AP now respect inverse-probability control weights;
- touch semantics use candidate-cell entry boundaries rather than requiring the numerical cell center.
