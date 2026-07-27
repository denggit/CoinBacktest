# Flow-Impact R03 Patch Manifest

## Added

- `src/research_common/flow_pa_accumulation.py`
- `research/mhf/flow_impact_state/03_accumulated_pressure_pa.py`
- `tests/research_common/test_flow_pa_accumulation.py`
- `docs/FLOW_IMPACT_STATE_R03_WEB_RESEARCH.md`
- `docs/FLOW_IMPACT_STATE_R03_RUNBOOK.md`

## Updated

- `research/mhf/flow_impact_state/00_research_log.md`

## Boundaries

- No Books/Liquidity data used.
- No fixed TP/SL parameter grid.
- No ML model.
- No direct SQLite reads from the research script.
- No future pivots: swing levels are delayed by right-confirmation bars plus one extra bar.
- No git commit executed.
