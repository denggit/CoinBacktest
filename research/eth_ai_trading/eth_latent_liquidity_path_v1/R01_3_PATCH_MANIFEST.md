# R01.3 Patch Manifest

## New package

```text
src/ai_research/latent_liquidity_absorption_model/
```

Modules:

- `config.py`: frozen stage chronology, model and commercial gates;
- `source.py`: deterministic R01.1 Episode sampling;
- `cache.py`: source and per-day replay checkpoint caches;
- `replay.py`: causal multi-checkpoint 1-second snapshots and stress labels;
- `modeling.py`: fixed multi-task LightGBM and mechanical baseline;
- `evaluation.py`: calibration-only q90 threshold, first-snapshot selection and stress PnL;
- `reports.py`: compact report, causal audit and final commercial decision;
- `pipeline.py`: complete R01.3 orchestration.

## New command

```text
research/eth_ai_trading/eth_latent_liquidity_path_v1/01_3_absorption_remaining_space_model.py
```

## New tests

```text
tests/ai_research/test_latent_liquidity_absorption_model.py
```

## Frozen research behavior

- R01.1 source cluster assignments are not changed;
- no sub-15m Swing is introduced;
- no direct Swing/unswept-Swing feature is used by R01.3;
- completed-second features and next-second-open execution only;
- train/calibration/holdout chronology is fixed;
- fee and delay stress are fixed before local execution.
