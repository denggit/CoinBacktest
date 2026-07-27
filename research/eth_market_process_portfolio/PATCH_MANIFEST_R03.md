# Patch Manifest — R03 Broad Order-Flow + Single-PA Path Atlas

## Added

- `research/eth_market_process_portfolio/order_flow/02_broad_order_flow_pa_path_atlas.py`
- `src/research_common/market_process/broad_order_flow_paths.py`
- `tests/research/eth_market_process_portfolio/test_broad_order_flow_pa_r03.py`

## Updated

- `src/research_common/market_process/__init__.py`
- `research/eth_market_process_portfolio/order_flow/00_research_log.md`
- `research/eth_market_process_portfolio/README.md`
- `research/eth_market_process_portfolio/00_research_charter.md`

## Boundaries

- Data is read only through `OKXTradeBarLoader` with timezone alignment enabled.
- No exchange request, raw archive parser, SQLite table access or new aggregation
  path exists in the research script.
- Pressure and PA features use closed bars; entries are labelled from next open.
- Prior-trend and prior-extreme context explicitly exclude the signal bar.
- Windows and bins are frozen before results; no parameter grid is searched.
- Research is yearly chunked. Rolling pressure uses cumulative sums; forward path
  labels use vectorized fixed-forward rolling windows.
