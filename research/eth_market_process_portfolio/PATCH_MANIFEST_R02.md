# R02 Patch Manifest

## Added

- `research/eth_market_process_portfolio/integration/01_environment_conditioned_strategy_lab.py`
- `research/eth_market_process_portfolio/integration/00_research_log.md`
- `src/research_common/market_process/__init__.py`
- `src/research_common/market_process/environment_features.py`
- `src/research_common/market_process/strategy_replay.py`
- `tests/research/eth_market_process_portfolio/test_environment_strategy_r02.py`

## Updated

- `research/eth_market_process_portfolio/README.md`
- `research/eth_market_process_portfolio/order_flow/00_research_log.md`

## Data boundaries

- Trade bars: `OKXTradeBarLoader`, tzplus8 local cache, `build_missing=False`.
- Range bars: `OKXRangeBarLoader.load_local_data`, tzplus8 local cache.
- No direct SQLite, archive parsing, HTTP, WebSocket or exchange requests in R02.
- No Books, range footprint, OI, funding or liquidation data in this round.

## Causal safeguards

- Range bars are visible only at completed `end_ts`.
- Closed 1m signal, next 1m open entry plus explicit delay scenarios.
- Trailing changes become active on the following bar.
- Stop-first treatment for same-bar stop/target ambiguity.
- Cross-year sleeve overlap is removed.

## Performance design

- One calendar year per chunk with bounded warmup and replay tail.
- Range context aligned with vectorized `numpy.searchsorted`.
- Rolling range counts use paired search operations, not event-window scans.
- OHLC converted to contiguous NumPy arrays once per chunk and shared by all
  candidates and stress scenarios.
- Actual 2025 alignment benchmark: 134,969 range bars into 525,600 minute rows
  in approximately 0.11 seconds in the delivery environment.

## Validation

```text
PYTHONPATH=. pytest -q tests/research/eth_market_process_portfolio
16 passed

R02 focused tests
7 passed

py_compile / compileall
passed
```

The repository-wide import-boundary checker still reports pre-existing
`research -> research` violations in older swing-low studies. R02 introduces no
such violation; its reusable helpers are under `src/research_common`.
