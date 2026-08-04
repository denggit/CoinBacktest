# R03.4.2.12 Patch Manifest

## New files

```text
src/ai_research/long_tail_soft_failure_tail_compression/__init__.py
src/ai_research/long_tail_soft_failure_tail_compression/config.py
src/ai_research/long_tail_soft_failure_tail_compression/inputs.py
src/ai_research/long_tail_soft_failure_tail_compression/simulator.py
src/ai_research/long_tail_soft_failure_tail_compression/analysis.py
src/ai_research/long_tail_soft_failure_tail_compression/reports.py
src/ai_research/long_tail_soft_failure_tail_compression/pipeline.py
research/eth_ai_trading/03_4_2_12_soft_failure_tail_compression.py
tests/ai_research/test_long_tail_soft_failure_tail_compression.py
docs/ETH_AI_TRADING_R03_4_2_12_SOFT_FAILURE_TAIL_COMPRESSION_RUNBOOK.md
research/eth_ai_trading/R03_4_2_12_DELIVERY.md
research/eth_ai_trading/R03_4_2_12_PATCH_MANIFEST.md
```

## Updated cumulative documents

```text
research/eth_ai_trading/README.md
research/eth_ai_trading/RESEARCH_HANDOFF.md
research/eth_ai_trading/COMPLETED_WORK.md
research/eth_ai_trading/OPEN_ITEMS_AND_ROADMAP.md
research/eth_ai_trading/DECISION_LOG.md
research/eth_ai_trading/STAGE_DELIVERY.md
```

## Boundaries

- No q70 retraining or threshold change.
- No `failed_reclaim` parameter change.
- No fixed-time final exit.
- No Pivot hard stop, split entry, Turtle add, pyramid add, or repeated-q70 add.
- 2026 remains sealed.
- No research entrypoint imports another research entrypoint.
- Data remains behind public `src.data_feed` loaders.
