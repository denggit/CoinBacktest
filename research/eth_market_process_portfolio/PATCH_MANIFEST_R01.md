# Patch Manifest — Order Flow R01

## Added

- `research/eth_market_process_portfolio/order_flow/01_order_flow_process_event_study.py`
- `research/eth_market_process_portfolio/order_flow/00_research_log.md`
- `tests/research/eth_market_process_portfolio/test_order_flow_r01.py`

## Research design

- Uses only `src.data_feed.OKXTradeBarLoader`.
- Explicitly uses timezone-aligned local trade bars (`tzplus8`).
- Loads one calendar year at a time with causal warmup and forward-tail overlap.
- Closed 1m bar signal and next-bar-open execution.
- Deducts 0.11% round-trip cost from all return labels.
- Screens four pre-declared mechanisms without a parameter grid.
- Reports 5/15/30/60-minute net paths, MFE/MAE, process/year breakdowns and causal-entry flags.

## Run

```bash
python research/eth_market_process_portfolio/order_flow/01_order_flow_process_event_study.py
```

## Validation

```bash
python -m pytest tests/research/eth_market_process_portfolio/test_order_flow_r01.py tests/research/eth_market_process_portfolio/test_coverage.py -q
```

Result: `9 passed`.

```bash
python -m compileall -q research/eth_market_process_portfolio/order_flow tests/research/eth_market_process_portfolio
```

Result: passed.
