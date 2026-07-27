# ETH Market Process Portfolio

A mechanism-first ETH research program organised around liquidity, order flow, volatility, and positioning.

## First command

```bash
python research/eth_market_process_portfolio/00_data_coverage_audit.py
```

Use `--fail-on-incomplete-core` in automated checks when mandatory full-history inputs must be present.

## Directory responsibilities

- `liquidity/`: resting liquidity and liquidity-event mechanisms
- `order_flow/`: trades, CVD, book response, impact and exhaustion
- `volatility/`: compression/expansion and path state
- `positioning/`: OI, funding, basis and liquidation
- `integration/`: causal cross-module market processes
- `portfolio/`: promoted sleeves, conflict routing and portfolio tests
- `common/`: frozen governance configuration and research-local utilities

Read `00_research_charter.md` before adding a numbered study.

## Hard data boundary

Every study reads data through `src.data_feed`. It may not download, parse exchange
archives, or query private cache schemas directly. Missing generic acquisition or
aggregation capability must first be implemented as a reusable `src.data_feed`
interface.

## Current research study

```bash
python research/eth_market_process_portfolio/order_flow/03_sell_pressure_shock_path_study.py
```

R04 tests whether abrupt multi-window sell pressure becomes a reversal only
when price shows a causal spike/reclaim response, or remains a continuation when
price accepts below the swept level. It starts from broad shock candidates,
reports both directions, and adds only one activity/PA condition at a time.
