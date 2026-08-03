# ETH AI Trading R02 + R03 Patch Manifest

## Scope

This patch upgrades the unified three-sleeve research contract and adds the first medium-horizon Swing supervised-learning baseline.

## Added

- `src/ai_research/sleeves/`: shared contracts, registry, and framework artifacts for short-horizon, intraday-trend, and swing sleeves.
- `src/ai_research/swing_baseline/`: public-loader data pipeline, causal multi-timeframe features, path labels, chronological models, structural-exit backtest, reports, and orchestration.
- `research/eth_ai_trading/02_upgrade_three_sleeve_framework.py`
- `research/eth_ai_trading/03_swing_multiframe_supervised_baseline.py`
- R02/R03 docs and tests.

## Modified

- Research plan upgraded to version 3 with R00-R14 stages.
- AI research README and framework artifact test updated for the 15-stage plan.

## Explicitly unchanged

- Existing strategy logic and prior research outputs.
- `src.data_feed` public loader behavior.
- Raw trade files and Trade Bar databases.
- AetherEdge.
- Git history; no commit is executed.

## Validation

- AI research tests.
- Relevant data-feed and causal-alignment tests.
- Python compilation.
- Synthetic end-to-end yearly cache, two-layer LightGBM, prediction, threshold, and structural-exit replay.
- New-scope import-boundary scan.
