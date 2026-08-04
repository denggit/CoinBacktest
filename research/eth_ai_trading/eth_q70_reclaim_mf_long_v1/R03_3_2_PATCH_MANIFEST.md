# R03.3.2 Patch Manifest

## Purpose

Replace sparse future-state event classification with continuous future opportunity intensity ranking.

## Added

- `research/eth_ai_trading/03_3_2_future_opportunity_intensity.py`
- `src/ai_research/future_process_forecast/intensity_config.py`
- `src/ai_research/future_process_forecast/intensity_targets.py`
- `src/ai_research/future_process_forecast/intensity_modeling.py`
- `src/ai_research/future_process_forecast/intensity_pipeline.py`
- `src/ai_research/future_process_forecast/intensity_reports.py`
- `tests/ai_research/test_future_process_intensity.py`
- `docs/ETH_AI_TRADING_R03_3_2_FUTURE_INTENSITY_RUNBOOK.md`

## Updated

- `src/ai_research/future_process_forecast/__init__.py`
- `research/eth_ai_trading/README.md`
- `docs/ETH_AI_TRADING_RESEARCH_PLAN.md`

## Frozen research contract

- Current market state is a causal multi-dimensional feature vector, not a future-derived discrete label.
- Targets are future 6h/12h total range, maximum one-sided excursion, two-sided excursion, and range relative to current causal 4H ATR.
- 2026H1 remains sealed.
- No Raw Trades reconstruction.
- Existing R03.2 and R03.3 caches are not overwritten.
