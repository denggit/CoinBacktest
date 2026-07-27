# Patch Manifest — Foundation R00

## Added

- Portfolio research domain with six responsibility-separated packages
- Frozen research governance configuration
- Local-only SQLite data coverage auditor
- Research charter and usage README
- Unit tests for mandatory, optional, and partial data gates

## Run

```bash
python research/eth_market_process_portfolio/00_data_coverage_audit.py
```

Strict gate:

```bash
python research/eth_market_process_portfolio/00_data_coverage_audit.py --fail-on-incomplete-core
```

## Validation

- `python -m pytest tests/research/eth_market_process_portfolio/test_coverage.py tests/market_state/test_causal_alignment.py -q`
- Result: 5 passed
- `python -m compileall -q research/eth_market_process_portfolio tests/research/eth_market_process_portfolio`
- Result: passed

The repository-wide import-boundary test still reports pre-existing violations in other research directories. This patch introduced no new `research -> research` absolute import violation.
