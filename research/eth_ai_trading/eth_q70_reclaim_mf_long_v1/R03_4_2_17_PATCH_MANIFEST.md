# R03.4.2.17 Patch Manifest

## Added

```text
src/ai_research/long_tail_state_gate_diagnostic/__init__.py
src/ai_research/long_tail_state_gate_diagnostic/config.py
src/ai_research/long_tail_state_gate_diagnostic/analysis.py
src/ai_research/long_tail_state_gate_diagnostic/pipeline.py
src/ai_research/long_tail_state_gate_diagnostic/reports.py
research/eth_ai_trading/03_4_2_17_state_gate_diagnostic.py
tests/ai_research/test_long_tail_state_gate_diagnostic.py
docs/ETH_AI_TRADING_R03_4_2_17_STATE_GATE_DIAGNOSTIC_RUNBOOK.md
research/eth_ai_trading/R03_4_2_17_DELIVERY.md
research/eth_ai_trading/R03_4_2_17_PATCH_MANIFEST.md
research/eth_ai_trading/R03_4_2_17_REPORT_LOGIC_HOTFIX3.md
```

## Updated cumulatively

```text
research/eth_ai_trading/README.md
research/eth_ai_trading/RESEARCH_HANDOFF.md
research/eth_ai_trading/COMPLETED_WORK.md
research/eth_ai_trading/DECISION_LOG.md
research/eth_ai_trading/OPEN_ITEMS_AND_ROADMAP.md
research/eth_ai_trading/STAGE_DELIVERY.md
```

## Boundaries

- no research entrypoint imports another research entrypoint;
- no C2 trade rule changes;
- no 2026 parameter search;
- counterfactual gates are not validation;
- V1 remains not live-approved.

## Hotfix3 reporting corrections

- dynamic regime-attribution details use actual calculated means;
- exact calendar account monthly returns are loaded from frozen source reports;
- entry-month cohort results are retained under explicit cohort labels only;
- trade-path MAE and account drawdown are kept as separate metrics when `full_mae` is unavailable;
- gate positivity is separated from genuine all-period uplift.
