# Patch Manifest — R04 Sell-Pressure Shock Path Study

## Added

- `research/eth_market_process_portfolio/order_flow/03_sell_pressure_shock_path_study.py`
- `src/research_common/market_process/sell_pressure_shock_paths.py`
- `tests/research/eth_market_process_portfolio/test_sell_pressure_shock_r04.py`

## Included prerequisite (unchanged from R03)

- `src/research_common/market_process/broad_order_flow_paths.py`

## Updated

- `research/eth_market_process_portfolio/order_flow/00_research_log.md`
- `research/eth_market_process_portfolio/README.md`

## Research boundaries

- Data is loaded only through `src.data_feed.OKXTradeBarLoader` using local
  timezone-aligned trade bars.
- No raw archive parsing, exchange request, SQLite query or research-local data
  acquisition path is introduced.
- Pressure windows compare adjacent, non-overlapping equal-length periods.
- The prior-low reference excludes the complete shock window.
- Same-window signals enter at next open; delayed reclaim and breakdown
  acceptance enter only after their confirmation bar closes.
- Both continuation and reversal outcomes are evaluated from the same base shock
  universe.
- Conditions are added individually; no trend/Range/Books/footprint/OI filter or
  TP/SL parameter search is included.

## Performance

- Pressure and activity use cumulative sums: O(N) per window.
- Shock PA uses vectorized rolling arrays.
- Delayed reclaim/acceptance uses a single-pass O(N) state machine.
- Annual chunks bound memory; all horizons share precomputed next-open outcomes.
