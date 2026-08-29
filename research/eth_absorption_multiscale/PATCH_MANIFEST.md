# R01 Patch Manifest

## Added

- `src/research_common/multiscale_absorption.py`
  - causal multi-bar pressure/impact features;
  - impact-decay features;
  - repeated floor/ceiling defense morphology;
  - spring/upthrust reclaim morphology;
  - vectorized fixed-horizon next-open outcomes.
- `research/eth_absorption_multiscale/01_multiscale_absorption_floor_atlas.py`
  - monthly/chunked 5s-to-4H atlas runner;
  - cache-only `src.data_feed.OKXTradeBarLoader` access;
  - 1m source reused for all 1m+ scales;
  - progress reporting and GPT review pack.
- `research/eth_absorption_multiscale/00_research_log.md`
- `research/eth_absorption_multiscale/README.md`
- `tests/research_common/test_multiscale_absorption.py`

## Default run

`python research/eth_absorption_multiscale/01_multiscale_absorption_floor_atlas.py`

## Default research window

- warmup floor: 2022-01-01
- research: 2023-01-01 through 2026-06-30
- full fixed-horizon economic cost diagnostic: 0.11%

## Tests performed

- `PYTHONPATH=. pytest tests/research_common/test_multiscale_absorption.py tests/research_common/test_flow_impact.py -q` -> 12 passed
- `PYTHONPATH=. pytest tests/research_common -q` -> 202 passed
- end-to-end synthetic 1m SQLite smoke run -> report + GPT review pack produced successfully

## Important interpretation rule

R01 is an event/path study. It must not be presented as a profitable strategy unless the local full-data outputs first show stable, economically thick and cross-year morphology differences.
